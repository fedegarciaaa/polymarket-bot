"""Maker-order lifecycle for crypto-lag.

Translates a model decision ("we think P(YES) = 0.55, the book is 0.50/0.51,
quote a YES bid at 0.51") into actual orders, and handles repricing /
cancellation as the model output drifts.

This module is backend-agnostic: it talks to either `PaperExecutor` (DEMO) or
a future LIVE adapter via the same shape (`place_order`, `cancel_order`,
`poll_fills`). The cycle owns the executor instance and passes it in.

v2 changes (May 2026 — F2.1/F2.2/F2.3 of profitability plan):
  - Avellaneda-Stoikov two-sided quoting: simultaneous BID + ASK around a
    reservation price that skews against current inventory. Single-sided
    quoting was leaving half the maker spread on the table.
  - Fee-aware edge: the Polymarket fee curve is `0.072·p·(1-p)·notional` for
    crypto markets and makers receive 20% of the fee as a rebate. We now
    add the expected rebate to the effective edge on each side, which lets
    profitable thin-edge quotes that were previously below the threshold
    actually fire.
  - Inventory-aware sizing: when `|inventory| / per_market_max` > 0.7 we
    suppress the side that would deepen inventory and skew the reservation
    price hard against it, encouraging mean-reverting fills that flatten
    the book.
  - The legacy single-sided `build_decision` is preserved so that callers
    that still pass through it (and the existing unit tests / cycle code)
    keep working unchanged. The cycle will be migrated to
    `build_decision_two_sided` separately.

Design choices:
  - Penny-jump: post `best_bid + tick` only if our edge > 2 cents; otherwise
    join at `best_bid` and rely on time priority. (Avoids paying 1c spread for
    marginal-edge quotes.)
  - Replace-only-on-significant-drift: cancel/replace if the model's fair_mid
    has moved by ≥ `replace_threshold_cents`, OR our level is no longer best,
    OR the order has been resting for `max_order_age_seconds`.
  - ALWAYS post-only: prices are clamped strictly inside the visible spread
    so we never cross the book and pay taker fees.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, Protocol

from .state import PolyCryptoMarket, RestingOrder
from .risk import CryptoLagRisk


# Quote modes — the engine branches on these in `build_decision_two_sided`.
#
#   maker             : legacy. Post-only inside the visible spread, clamped to
#                       `[poly_bid + tick, poly_ask - tick]`. In 1-tick spreads
#                       this collapses to top-of-book joining (back of queue) —
#                       what the May-2026 audit showed produces 0 fills.
#   penny_aggressive  : when the visible spread allows it (>= 2 ticks), post
#                       1 tick INSIDE (true penny-jump). When the spread is
#                       exactly 1 tick — the common case for crypto-lag 5/15m
#                       markets — lift the opposite side (taker, paying fee)
#                       provided the net edge exceeds `edge_threshold` and
#                       `cross_threshold_ticks` × tick. Stays passive when the
#                       book is wider than the model thinks is real.
#   ioc_taker         : never quote passive. Cross the book whenever
#                       `|fair_mid - poly_mid|` exceeds `cross_threshold_ticks`
#                       × tick. Pays the Polymarket taker fee on every fill;
#                       the A/B vs penny_aggressive measures whether the real
#                       edge survives net of fees.
QUOTE_MODE_MAKER = "maker"
QUOTE_MODE_PENNY_AGGRESSIVE = "penny_aggressive"
QUOTE_MODE_IOC_TAKER = "ioc_taker"
_VALID_QUOTE_MODES = {
    QUOTE_MODE_MAKER, QUOTE_MODE_PENNY_AGGRESSIVE, QUOTE_MODE_IOC_TAKER,
}

logger = logging.getLogger("polymarket_bot.crypto_lag.engine")


# ─── fee defaults (Polymarket crypto, April 2026) ──────────────────
# Peak fee = FEE_RATE_CRYPTO · 0.5 · 0.5 = 1.80% in p=0.5. Makers pay 0 and
# receive MAKER_REBATE_SHARE of the taker's fee back as rebate.
FEE_RATE_CRYPTO = 0.072
MAKER_REBATE_SHARE = 0.20


def parabolic_fee(p: float, fee_rate: float = FEE_RATE_CRYPTO) -> float:
    """Polymarket's effective taker fee as a fraction of notional at price `p`.

    Curve: fee = rate · p · (1-p). Peaks at p=0.5 and decays to 0 at the
    extremes — so quotes near 0.05 / 0.95 collect very little rebate even if
    they fill.
    """
    p = max(0.0, min(1.0, float(p)))
    return float(fee_rate) * p * (1.0 - p)


def expected_maker_rebate(
    p: float,
    fee_rate: float = FEE_RATE_CRYPTO,
    rebate_share: float = MAKER_REBATE_SHARE,
) -> float:
    """Per-unit-notional maker rebate at fill price `p`. Positive number
    measured in the same units as `p` (so a 0.0036 rebate at p=0.5 means
    36 basis points of notional)."""
    return parabolic_fee(p, fee_rate) * float(rebate_share)


class _ExecutorProto(Protocol):
    """Minimal interface required from the executor (paper or live)."""
    async def place_order(self, order: RestingOrder, token_id: str) -> str: ...
    async def cancel_order(self, order_id: str) -> bool: ...


@dataclass
class _DecisionCtx:
    """Internal: precomputed shared state passed into the per-mode decision
    helpers. Public-facing fields (edges, reservation, half_spread) eventually
    land on the QuoteDecision so the snapshot row keeps the same shape across
    quote_modes — diagnostics queries don't need to special-case anything.
    """
    fair_mid: float
    poly_best_bid: float
    poly_best_ask: float
    target_size_usdc: float
    tick: float
    edge_bid: float
    edge_ask: float
    edge_bid_net: float
    edge_ask_net: float
    reservation: float
    half_spread: float
    raw_bid: float
    raw_ask: float
    inv_blocks_bid: bool
    inv_blocks_ask: bool
    q_norm: float
    inv_skew: float


@dataclass
class QuoteDecision:
    """Output of the decision step. The order_engine consumes this and the
    cycle persists it to the snapshot log for forensics.

    `side` is one of:
      - "NONE" → no quote
      - "BID"  → only buy YES
      - "ASK"  → only sell YES
      - "BOTH" → quote both sides simultaneously (two-sided MM)
    """
    side: str
    fair_mid: float
    edge_bid: float                   # gross edge = fair_mid - poly_best_bid
    edge_ask: float                   # gross edge = poly_best_ask - fair_mid
    poly_best_bid: float
    poly_best_ask: float
    target_size_usdc: float           # symmetric size if BOTH; else size for the chosen side
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    bid_size_usdc: Optional[float] = None  # asymmetric sizing (None → use target_size_usdc)
    ask_size_usdc: Optional[float] = None
    # Diagnostics — these are persisted alongside the snapshot for debugging.
    edge_bid_net: float = 0.0         # edge_bid + maker_rebate
    edge_ask_net: float = 0.0         # edge_ask + maker_rebate
    reservation_price: float = 0.0    # AS reservation price including inventory skew
    inventory_skew: float = 0.0       # AS skew that was applied (signed)
    half_spread: float = 0.0          # AS half-spread (>= 0)
    # True when the BID leg crosses the book at placement (taker fee owed).
    bid_is_taker: bool = False
    # True when the ASK leg crosses the book at placement (taker fee owed).
    ask_is_taker: bool = False
    # Mode that produced this decision — propagated for logging/diagnostics.
    mode: str = QUOTE_MODE_MAKER


class MakerOrderEngine:
    def __init__(
        self,
        executor: _ExecutorProto,
        risk: CryptoLagRisk,
        edge_threshold_cents: float = 2.0,
        replace_threshold_cents: float = 1.0,
        max_order_age_seconds: float = 30.0,
        gamma: float = 0.10,
        arrival_intensity_k: float = 1.5,
        inventory_skew_threshold: float = 0.7,
        fee_rate: float = FEE_RATE_CRYPTO,
        maker_rebate_share: float = MAKER_REBATE_SHARE,
        quote_mode: str = QUOTE_MODE_MAKER,
        cross_threshold_ticks: float = 4.0,
        placement_logger: Optional[Callable[[RestingOrder, float], None]] = None,
    ):
        self.executor = executor
        self.risk = risk
        self.edge_threshold = edge_threshold_cents / 100.0
        self.replace_threshold = replace_threshold_cents / 100.0
        self.max_order_age = max_order_age_seconds

        # Avellaneda-Stoikov tunables. γ is risk aversion; k is the order
        # arrival intensity. They calibrate jointly with the spread, and the
        # ones below are conservative defaults that the backtest framework
        # will tune later.
        self.gamma = float(gamma)
        self.arrival_intensity_k = float(max(arrival_intensity_k, 1e-3))
        self.inventory_skew_threshold = float(inventory_skew_threshold)

        # Fee model
        self.fee_rate = float(fee_rate)
        self.maker_rebate_share = float(maker_rebate_share)

        # Quote-mode dispatch (see module-level docstring on QUOTE_MODE_*).
        if quote_mode not in _VALID_QUOTE_MODES:
            raise ValueError(
                f"unknown quote_mode {quote_mode!r}; expected one of {_VALID_QUOTE_MODES}"
            )
        self.quote_mode = quote_mode
        # Threshold (in TICKS, not cents) above which an aggressive mode will
        # cross the book. For ioc_taker this is the only gate; for
        # penny_aggressive it's only consulted in the 1-tick-spread branch.
        self.cross_threshold_ticks = float(max(cross_threshold_ticks, 0.0))

        # Optional callback fired for every order acknowledged by the executor.
        # Used to persist placements to crypto_lag_quotes (status='placed') so
        # we can compute fill-rate per variant without parsing the journal.
        self.placement_logger = placement_logger

        # condition_id → side("BID"|"ASK") → RestingOrder
        self._open: dict[str, dict[str, RestingOrder]] = {}

        # Lightweight per-period counters consumed by the cycle's hourly
        # CRYPTO_LAG_VARIANT_STATS event. Reset by `reset_period_counters`.
        self.placements_period: int = 0
        self.placements_period_taker: int = 0
        self.placements_lifetime: int = 0

    def reset_period_counters(self) -> None:
        """Zero per-period counters after the cycle emits its hourly stats."""
        self.placements_period = 0
        self.placements_period_taker = 0

    # ─── public API ─────────────────────────────────────────────
    async def reconcile(
        self,
        market: PolyCryptoMarket,
        decision: QuoteDecision,
        now_ts: Optional[float] = None,
    ) -> None:
        """Make our open orders match the decision: cancel stale, replace
        drifted, place missing. No-op if decision.side == 'NONE'."""
        now = now_ts or time.time()
        cid = market.condition_id
        existing = self._open.get(cid, {})

        # 1. Cancel ASK if we no longer want to quote it
        if "ASK" in existing and decision.side not in ("ASK", "BOTH"):
            await self._cancel(cid, "ASK")
        # 2. Cancel BID if not wanted
        if "BID" in existing and decision.side not in ("BID", "BOTH"):
            await self._cancel(cid, "BID")

        if decision.side == "NONE" or decision.target_size_usdc <= 0:
            return

        # 3. (Re)place BID
        if decision.side in ("BID", "BOTH") and decision.bid_price is not None:
            bid_size = decision.bid_size_usdc if decision.bid_size_usdc is not None \
                else decision.target_size_usdc
            if bid_size > 0:
                await self._upsert(
                    market=market, side="BID", price=decision.bid_price,
                    size_usdc=bid_size, now=now,
                    is_taker=bool(decision.bid_is_taker),
                )
        # 4. (Re)place ASK
        if decision.side in ("ASK", "BOTH") and decision.ask_price is not None:
            ask_size = decision.ask_size_usdc if decision.ask_size_usdc is not None \
                else decision.target_size_usdc
            if ask_size > 0:
                await self._upsert(
                    market=market, side="ASK", price=decision.ask_price,
                    size_usdc=ask_size, now=now,
                    is_taker=bool(decision.ask_is_taker),
                )

    async def cancel_all_for(self, condition_id: str) -> None:
        for side in list(self._open.get(condition_id, {}).keys()):
            await self._cancel(condition_id, side)

    async def cancel_all(self) -> None:
        for cid in list(self._open.keys()):
            await self.cancel_all_for(cid)

    def open_orders(self, condition_id: str) -> dict[str, RestingOrder]:
        return dict(self._open.get(condition_id, {}))

    # ─── decision helpers ───────────────────────────────────────
    def build_decision(
        self,
        fair_mid: float,
        poly_best_bid: float,
        poly_best_ask: float,
        target_size_usdc: float,
        tick: float = 0.01,
    ) -> QuoteDecision:
        """LEGACY single-sided quoting. Kept for backwards compatibility with
        callers that haven't migrated to `build_decision_two_sided`.

        Behaviour: only quote the side with the larger gross edge that exceeds
        the threshold. No AS reservation pricing, no fee-aware tweak.
        """
        edge_bid = fair_mid - poly_best_bid
        edge_ask = poly_best_ask - fair_mid

        bid_price: Optional[float] = None
        ask_price: Optional[float] = None
        side = "NONE"

        if edge_bid > self.edge_threshold and edge_bid >= edge_ask:
            raw_bid = poly_best_bid + tick if edge_bid >= 2 * tick else poly_best_bid
            ceiling = round(poly_best_ask - tick, 2) if poly_best_ask > 0 else 0.99
            bid_price = round(max(0.01, min(raw_bid, ceiling, 0.99)), 2)
            side = "BID"
        elif edge_ask > self.edge_threshold:
            raw_ask = poly_best_ask - tick if edge_ask >= 2 * tick else poly_best_ask
            floor_ = round(poly_best_bid + tick, 2) if poly_best_bid > 0 else 0.01
            ask_price = round(min(0.99, max(raw_ask, floor_, 0.01)), 2)
            side = "ASK"

        return QuoteDecision(
            side=side, fair_mid=fair_mid,
            edge_bid=edge_bid, edge_ask=edge_ask,
            poly_best_bid=poly_best_bid, poly_best_ask=poly_best_ask,
            target_size_usdc=target_size_usdc,
            bid_price=bid_price, ask_price=ask_price,
        )

    def build_decision_two_sided(
        self,
        fair_mid: float,
        poly_best_bid: float,
        poly_best_ask: float,
        target_size_usdc: float,
        tick: float = 0.01,
        sigma_per_sqrt_s: float = 0.0,
        t_remaining_s: float = 0.0,
        inventory_usdc: float = 0.0,
        per_market_max_inventory_usdc: float = 100.0,
    ) -> QuoteDecision:
        """Top-level decision builder. Computes the AS reservation/half-spread
        and net edges once, then dispatches to the per-mode quote builder.

        See QUOTE_MODE_* constants at the top of this module for the semantics
        of each mode. The shared fields on the returned QuoteDecision (edges,
        rebates, AS reservation) are populated identically across modes so
        diagnostics in the snapshot table work the same regardless.
        """
        # ── 1. Reservation price + AS half-spread (shared across modes) ──
        sigma2_t = max(0.0, (sigma_per_sqrt_s ** 2) * float(max(t_remaining_s, 0.0)))
        denom = max(1e-6, float(per_market_max_inventory_usdc))
        q_norm = max(-1.0, min(1.0, float(inventory_usdc) / denom))

        inv_skew = q_norm * self.gamma * sigma2_t
        reservation = fair_mid - inv_skew

        gamma_eff = max(self.gamma, 1e-3)
        half_spread = 0.5 * gamma_eff * sigma2_t + (1.0 / gamma_eff) * math.log(
            1.0 + gamma_eff / self.arrival_intensity_k
        )
        visible_half_spread = max(0.0, 0.5 * (poly_best_ask - poly_best_bid))
        half_spread = max(half_spread, float(tick))
        half_spread = min(half_spread, max(2.0 * float(tick), 1.5 * visible_half_spread))

        # ── 2. Fee-aware edges ─────────────────────────────────
        edge_bid = fair_mid - poly_best_bid
        edge_ask = poly_best_ask - fair_mid
        rebate_at_bid = expected_maker_rebate(
            poly_best_bid, fee_rate=self.fee_rate,
            rebate_share=self.maker_rebate_share,
        )
        rebate_at_ask = expected_maker_rebate(
            poly_best_ask, fee_rate=self.fee_rate,
            rebate_share=self.maker_rebate_share,
        )
        edge_bid_net = edge_bid + rebate_at_bid
        edge_ask_net = edge_ask + rebate_at_ask

        # Inventory penalty: kill the inventory-deepening side
        inv_blocks_bid = q_norm > self.inventory_skew_threshold
        inv_blocks_ask = q_norm < -self.inventory_skew_threshold

        ctx = _DecisionCtx(
            fair_mid=fair_mid,
            poly_best_bid=poly_best_bid, poly_best_ask=poly_best_ask,
            target_size_usdc=target_size_usdc, tick=float(tick),
            edge_bid=edge_bid, edge_ask=edge_ask,
            edge_bid_net=edge_bid_net, edge_ask_net=edge_ask_net,
            reservation=reservation, half_spread=half_spread,
            raw_bid=reservation - half_spread, raw_ask=reservation + half_spread,
            inv_blocks_bid=inv_blocks_bid, inv_blocks_ask=inv_blocks_ask,
            q_norm=q_norm, inv_skew=inv_skew,
        )

        if self.quote_mode == QUOTE_MODE_IOC_TAKER:
            return self._decide_ioc_taker(ctx)
        if self.quote_mode == QUOTE_MODE_PENNY_AGGRESSIVE:
            return self._decide_penny_aggressive(ctx)
        return self._decide_maker(ctx)

    # ─── per-mode decision builders ──────────────────────────────────
    def _decide_maker(self, c: "_DecisionCtx") -> QuoteDecision:
        """Original strict maker. Post-only, clamped strictly inside the visible
        spread. Joins top-of-queue when the visible spread is exactly 1 tick.
        """
        bid_price: Optional[float] = None
        ask_price: Optional[float] = None
        post_bid = (c.edge_bid_net > self.edge_threshold) and not c.inv_blocks_bid
        post_ask = (c.edge_ask_net > self.edge_threshold) and not c.inv_blocks_ask

        if post_bid:
            penny = c.poly_best_bid + c.tick if c.edge_bid_net >= 2.0 * c.tick else c.poly_best_bid
            cand = max(c.raw_bid, penny)
            ceiling = round(c.poly_best_ask - c.tick, 4) if c.poly_best_ask > 0 else 0.99
            bid_price = _round_tick(max(0.01, min(cand, ceiling, 0.99)), c.tick)

        if post_ask:
            penny = c.poly_best_ask - c.tick if c.edge_ask_net >= 2.0 * c.tick else c.poly_best_ask
            cand = min(c.raw_ask, penny)
            floor_ = round(c.poly_best_bid + c.tick, 4) if c.poly_best_bid > 0 else 0.01
            ask_price = _round_tick(min(0.99, max(cand, floor_, 0.01)), c.tick)

        return self._finalize_decision(c, bid_price, ask_price,
                                       bid_is_taker=False, ask_is_taker=False,
                                       mode=QUOTE_MODE_MAKER)

    def _decide_penny_aggressive(self, c: "_DecisionCtx") -> QuoteDecision:
        """When the visible spread allows it (>= 2 ticks), post 1 tick INSIDE
        as a true maker penny-jump. When the visible spread is exactly 1 tick
        — the dominant case for crypto-lag 5/15m markets — lift the opposite
        side as a taker provided net edge > max(edge_threshold, cross_threshold).

        Crossed legs are flagged is_taker=True so the executor charges fees.
        """
        bid_price: Optional[float] = None
        ask_price: Optional[float] = None
        bid_is_taker = False
        ask_is_taker = False

        visible_spread = c.poly_best_ask - c.poly_best_bid
        wide_book = visible_spread >= 1.5 * c.tick   # >= 2 ticks accounting for fp slop
        cross_thr = max(self.edge_threshold, self.cross_threshold_ticks * c.tick)

        post_bid = (c.edge_bid_net > self.edge_threshold) and not c.inv_blocks_bid
        post_ask = (c.edge_ask_net > self.edge_threshold) and not c.inv_blocks_ask

        if post_bid:
            if wide_book:
                # True penny-jump: post inside the visible spread.
                cand = c.poly_best_bid + c.tick
                ceiling = round(c.poly_best_ask - c.tick, 4)
                bid_price = _round_tick(max(0.01, min(cand, ceiling, 0.99)), c.tick)
            elif c.edge_bid_net > cross_thr and c.poly_best_ask > 0:
                # Tight book: cross by 1 tick (lift the offer).
                bid_price = _round_tick(min(0.99, c.poly_best_ask), c.tick)
                bid_is_taker = True

        if post_ask:
            if wide_book:
                cand = c.poly_best_ask - c.tick
                floor_ = round(c.poly_best_bid + c.tick, 4)
                ask_price = _round_tick(min(0.99, max(cand, floor_, 0.01)), c.tick)
            elif c.edge_ask_net > cross_thr and c.poly_best_bid > 0:
                ask_price = _round_tick(max(0.01, c.poly_best_bid), c.tick)
                ask_is_taker = True

        return self._finalize_decision(c, bid_price, ask_price,
                                       bid_is_taker=bid_is_taker,
                                       ask_is_taker=ask_is_taker,
                                       mode=QUOTE_MODE_PENNY_AGGRESSIVE)

    def _decide_ioc_taker(self, c: "_DecisionCtx") -> QuoteDecision:
        """Cross-only mode. Lift the opposite side whenever the GROSS edge
        (pre-rebate, since this leg pays a fee instead) exceeds
        `cross_threshold_ticks * tick`. Never quote passively.
        """
        bid_price: Optional[float] = None
        ask_price: Optional[float] = None
        bid_is_taker = False
        ask_is_taker = False

        thr = self.cross_threshold_ticks * c.tick

        # IOC-taker uses the GROSS edge: the rebate doesn't apply when crossing,
        # and the fee is collected by the executor on fill.
        if (c.edge_bid > thr and c.edge_bid > self.edge_threshold
                and not c.inv_blocks_bid and c.poly_best_ask > 0):
            bid_price = _round_tick(min(0.99, c.poly_best_ask), c.tick)
            bid_is_taker = True

        if (c.edge_ask > thr and c.edge_ask > self.edge_threshold
                and not c.inv_blocks_ask and c.poly_best_bid > 0):
            ask_price = _round_tick(max(0.01, c.poly_best_bid), c.tick)
            ask_is_taker = True

        return self._finalize_decision(c, bid_price, ask_price,
                                       bid_is_taker=bid_is_taker,
                                       ask_is_taker=ask_is_taker,
                                       mode=QUOTE_MODE_IOC_TAKER)

    def _finalize_decision(
        self, c: "_DecisionCtx",
        bid_price: Optional[float], ask_price: Optional[float],
        bid_is_taker: bool, ask_is_taker: bool, mode: str,
    ) -> QuoteDecision:
        # If both candidate prices ended up crossed (bid >= ask), drop the side
        # with the smaller net edge so we don't self-trade in a single tick.
        if bid_price is not None and ask_price is not None:
            if bid_price >= ask_price - 1e-9:
                if c.edge_bid_net >= c.edge_ask_net:
                    ask_price = None
                    ask_is_taker = False
                else:
                    bid_price = None
                    bid_is_taker = False

        if bid_price is not None and ask_price is not None:
            side = "BOTH"
        elif bid_price is not None:
            side = "BID"
        elif ask_price is not None:
            side = "ASK"
        else:
            side = "NONE"

        bid_size: Optional[float] = None
        ask_size: Optional[float] = None
        if side == "BOTH":
            tilt = c.q_norm
            bid_size = c.target_size_usdc * max(0.25, 1.0 - max(0.0, tilt))
            ask_size = c.target_size_usdc * max(0.25, 1.0 + min(0.0, tilt))

        return QuoteDecision(
            side=side, fair_mid=c.fair_mid,
            edge_bid=c.edge_bid, edge_ask=c.edge_ask,
            poly_best_bid=c.poly_best_bid, poly_best_ask=c.poly_best_ask,
            target_size_usdc=c.target_size_usdc,
            bid_price=bid_price, ask_price=ask_price,
            bid_size_usdc=bid_size, ask_size_usdc=ask_size,
            edge_bid_net=c.edge_bid_net, edge_ask_net=c.edge_ask_net,
            reservation_price=c.reservation,
            inventory_skew=c.inv_skew,
            half_spread=c.half_spread,
            bid_is_taker=bid_is_taker,
            ask_is_taker=ask_is_taker,
            mode=mode,
        )

    # ─── internals ──────────────────────────────────────────────
    async def _upsert(
        self,
        market: PolyCryptoMarket,
        side: str,                 # "BID" | "ASK"
        price: float,
        size_usdc: float,
        now: float,
        is_taker: bool = False,
    ) -> None:
        cid = market.condition_id
        existing = self._open.setdefault(cid, {}).get(side)
        # Decide if we need to do anything
        if existing is not None:
            age = now - existing.last_replace_ts
            drift = abs(existing.price - price)
            if drift < self.replace_threshold and age < self.max_order_age \
               and abs(existing.size_usdc - size_usdc) < 0.01:
                return
            # Cancel before replace
            await self._cancel(cid, side)
        # Place fresh
        # YES BID = BUY YES at `price`. YES ASK = SELL YES at `price`.
        order_side = "BUY" if side == "BID" else "SELL"
        order = RestingOrder(
            order_id=str(uuid.uuid4()),
            external_order_id=None,
            symbol=market.symbol,
            condition_id=cid,
            side=order_side,
            outcome="YES",
            price=price,
            size_usdc=size_usdc,
            placed_ts=now,
            last_replace_ts=now,
            # Snapshot market metadata so the cycle can resolve this position
            # even after the market drops off Gamma's active list.
            end_ts=float(market.end_ts),
            strike_price=float(market.strike_price),
            market_slug=str(market.market_slug),
            tick_size=float(market.tick_size),
            is_taker=bool(is_taker),
        )
        try:
            ext = await self.executor.place_order(order, market.token_yes)
            order.external_order_id = ext
            # Re-create the per-cid bucket if `_cancel` above just removed it.
            # `_cancel` calls `_open.pop(cid)` when the bucket becomes empty
            # after the cancellation, so the assignment `_open[cid][side]`
            # would otherwise KeyError on the cid hash. Using setdefault is
            # idempotent: re-uses the existing dict if present, creates one
            # otherwise.
            self._open.setdefault(cid, {})[side] = order
            self.risk.on_order_open()
            taker_tag = " TAKER" if is_taker else ""
            self.placements_period += 1
            self.placements_lifetime += 1
            if is_taker:
                self.placements_period_taker += 1
            logger.info(
                f"placed {side}{taker_tag} {market.symbol} {market.market_slug[:30]}: "
                f"{size_usdc:.2f}@{price:.4f} (ext={ext})"
            )
            if self.placement_logger is not None:
                # Persist this placement so the dashboard / analyses can compute
                # fill-rate per variant without parsing the journal. Read the
                # queue debt the simulator just initialized (real LIVE will
                # report 0 for taker fills, which is consistent).
                try:
                    qd = 0.0
                    getter = getattr(self.executor, "get_queue_debt", None)
                    if getter is not None:
                        qd = float(getter(order.order_id) or 0.0)
                    self.placement_logger(order, qd)
                except Exception as exc:
                    logger.debug(f"placement_logger error: {exc}")
        except Exception as exc:
            logger.warning(
                f"place {side} {market.market_slug}: {type(exc).__name__}: {exc}"
            )

    async def _cancel(self, condition_id: str, side: str) -> None:
        existing = self._open.get(condition_id, {}).get(side)
        if existing is None:
            return
        try:
            await self.executor.cancel_order(existing.order_id)
            self.risk.on_order_close()
        except Exception as exc:
            logger.warning(f"cancel {side} {condition_id[:12]}...: {exc}")
        finally:
            self._open[condition_id].pop(side, None)
            if not self._open[condition_id]:
                self._open.pop(condition_id, None)


def _round_tick(price: float, tick: float) -> float:
    """Round to nearest tick, clamped to [0.01, 0.99] and never NaN."""
    tick = float(tick) if tick > 0 else 0.01
    if not math.isfinite(price):
        return 0.01
    rounded = round(round(price / tick) * tick, 6)
    return max(0.01, min(0.99, rounded))
