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
from typing import Optional, Protocol

from .state import PolyCryptoMarket, RestingOrder
from .risk import CryptoLagRisk

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

        # condition_id → side("BID"|"ASK") → RestingOrder
        self._open: dict[str, dict[str, RestingOrder]] = {}

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
                )
        # 4. (Re)place ASK
        if decision.side in ("ASK", "BOTH") and decision.ask_price is not None:
            ask_size = decision.ask_size_usdc if decision.ask_size_usdc is not None \
                else decision.target_size_usdc
            if ask_size > 0:
                await self._upsert(
                    market=market, side="ASK", price=decision.ask_price,
                    size_usdc=ask_size, now=now,
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
        """Two-sided maker quoting with Avellaneda-Stoikov reservation +
        spread, plus fee-aware edge thresholds.

        Reservation price (skews against inventory):
            r = fair_mid - q_norm · γ · σ²·(T-t)
        where q_norm = inventory_usdc / per_market_max_inventory_usdc ∈ [-1, 1].

        Optimal half-spread:
            δ = γ·σ²·(T-t)/2 + (1/γ)·ln(1 + γ/k)

        Final quotes are floored/ceiled to stay inside Polymarket's visible
        spread (always post-only) and rounded to the tick. Sides whose
        fee-adjusted net edge is below `edge_threshold` are suppressed. If
        the inventory penalty triggers, the inventory-deepening side is also
        suppressed entirely.
        """
        # ── 1. Reservation price + AS half-spread ──────────────
        sigma2_t = max(0.0, (sigma_per_sqrt_s ** 2) * float(max(t_remaining_s, 0.0)))
        denom = max(1e-6, float(per_market_max_inventory_usdc))
        q_norm = max(-1.0, min(1.0, float(inventory_usdc) / denom))

        inv_skew = q_norm * self.gamma * sigma2_t
        reservation = fair_mid - inv_skew

        # γ is bounded away from 0 so this is well-defined.
        gamma_eff = max(self.gamma, 1e-3)
        half_spread = 0.5 * gamma_eff * sigma2_t + (1.0 / gamma_eff) * math.log(
            1.0 + gamma_eff / self.arrival_intensity_k
        )
        # The AS optimal spread can balloon when σ²·T is large (early in a
        # 1-hour market). Cap it at half the visible Polymarket spread so we
        # never quote outside the book.
        visible_half_spread = max(0.0, 0.5 * (poly_best_ask - poly_best_bid))
        # Allow up to 1.5× the visible half-spread so we can still quote
        # tighter than chronic single-tick books while ditching wild values.
        half_spread = max(half_spread, float(tick))
        half_spread = min(half_spread, max(2.0 * float(tick), 1.5 * visible_half_spread))

        # ── 2. Raw AS quotes around the reservation price ──────
        raw_bid = reservation - half_spread
        raw_ask = reservation + half_spread

        # ── 3. Fee-aware edges (rebate boosts the edge) ────────
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

        # ── 4. Inventory penalty: kill the inventory-deepening side
        # when |q_norm| crosses the threshold ───────────────────
        inv_blocks_bid = q_norm > self.inventory_skew_threshold   # too long → don't buy more
        inv_blocks_ask = q_norm < -self.inventory_skew_threshold  # too short → don't sell more

        # ── 5. Decide which sides we'll actually post ──────────
        post_bid = (edge_bid_net > self.edge_threshold) and not inv_blocks_bid
        post_ask = (edge_ask_net > self.edge_threshold) and not inv_blocks_ask

        bid_price: Optional[float] = None
        ask_price: Optional[float] = None

        if post_bid:
            # Penny-jump if the edge is comfortable, else join the queue.
            penny = poly_best_bid + tick if edge_bid_net >= 2.0 * tick else poly_best_bid
            cand = max(raw_bid, penny)
            ceiling = round(poly_best_ask - tick, 4) if poly_best_ask > 0 else 0.99
            bid_price = _round_tick(
                max(0.01, min(cand, ceiling, 0.99)), tick
            )

        if post_ask:
            penny = poly_best_ask - tick if edge_ask_net >= 2.0 * tick else poly_best_ask
            cand = min(raw_ask, penny)
            floor_ = round(poly_best_bid + tick, 4) if poly_best_bid > 0 else 0.01
            ask_price = _round_tick(
                min(0.99, max(cand, floor_, 0.01)), tick
            )

        # If both candidate prices crossed (bid >= ask after clamping), prefer
        # the side with the larger net edge and drop the other.
        if bid_price is not None and ask_price is not None:
            if bid_price >= ask_price - 1e-9:
                if edge_bid_net >= edge_ask_net:
                    ask_price = None
                else:
                    bid_price = None

        if bid_price is not None and ask_price is not None:
            side = "BOTH"
        elif bid_price is not None:
            side = "BID"
        elif ask_price is not None:
            side = "ASK"
        else:
            side = "NONE"

        # ── 6. Asymmetric sizing under inventory pressure ─────
        bid_size: Optional[float] = None
        ask_size: Optional[float] = None
        if side == "BOTH":
            # Size each leg according to how much room we have on that side.
            # Long inventory → smaller bid, larger ask (mean-revert toward 0).
            tilt = q_norm  # ∈ [-1, 1]
            bid_size = target_size_usdc * max(0.25, 1.0 - max(0.0, tilt))
            ask_size = target_size_usdc * max(0.25, 1.0 + min(0.0, tilt))

        return QuoteDecision(
            side=side, fair_mid=fair_mid,
            edge_bid=edge_bid, edge_ask=edge_ask,
            poly_best_bid=poly_best_bid, poly_best_ask=poly_best_ask,
            target_size_usdc=target_size_usdc,
            bid_price=bid_price, ask_price=ask_price,
            bid_size_usdc=bid_size, ask_size_usdc=ask_size,
            edge_bid_net=edge_bid_net, edge_ask_net=edge_ask_net,
            reservation_price=reservation,
            inventory_skew=inv_skew,
            half_spread=half_spread,
        )

    # ─── internals ──────────────────────────────────────────────
    async def _upsert(
        self,
        market: PolyCryptoMarket,
        side: str,                 # "BID" | "ASK"
        price: float,
        size_usdc: float,
        now: float,
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
            logger.info(
                f"placed {side} {market.symbol} {market.market_slug[:30]}: "
                f"{size_usdc:.2f}@{price:.4f} (ext={ext})"
            )
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
