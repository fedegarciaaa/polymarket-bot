"""DEMO-mode order executor for crypto-lag.

Simulates fills against the real Polymarket CLOB orderbook (read via REST).
The executor is the single point that translates "we placed an order at price P
size S" into trade events. In LIVE mode the order_engine talks to a different
backend (not implemented yet); in DEMO this is what runs.

Fill model (per polled book snapshot):
  - YES BID at price P: fills only when the real best_ask drops to P (or
    below). Filled size is bounded by the visible ask depth at-or-below our
    price AND by our queue position — see "queue position model" below.
  - YES ASK at price P (a sell of YES): symmetric — fills when real best_bid
    rises to P.
  - With probability `q_toxic` the fill is tagged as adverse — we still fill,
    but record an adverse_selection cost so DEMO P&L matches LIVE more closely.
  - Partial fills are supported: an order can be filled across multiple ticks.

Queue position model (F0.2 of the profitability plan):
  - When a maker order is first placed at a price `P` matching the visible
    best on that side, we record the size visible at that level as our
    "queue debt" — the cumulative volume that has to clear ahead of us
    before we get filled.
  - Each book poll, we decrement queue debt by however much volume CLEARED
    at our level since the last snapshot (visible size went down, or the
    book stepped past our price). When queue debt hits zero, we're at the
    front of the queue and any cross will fill us.
  - This model is conservative: we always assume we joined at the BACK of
    the queue. It's a v1 — a more accurate model would use the LOB delta
    feed to track our actual position, but Polymarket WS doesn't surface
    that; so this is the best we can do without a self-trade tag.
  - When we PENNY-JUMP (post inside the visible spread), queue debt is 0
    because we're alone on our level until someone else comes along.

Fee model (F0.4):
  - Crypto markets: fee curve is `0.072 · p · (1-p) · notional`, paid by
    takers. Makers pay 0 and receive `MAKER_REBATE_SHARE` (default 20%) of
    the taker fee back as rebate. We apply the rebate as a credit on each
    fill, immediately reflecting it in realized PnL — that mirrors LIVE
    behaviour where Polymarket disburses the rebate daily.
  - Position settlement at expiry uses the actual YES outcome value (1.0
    or 0.0). Realized PnL = payoff − cost_basis + accumulated_rebate.

Resolution model:
  - At endDate the YES outcome resolves to 1.0 if Binance close ≥ strike (tie
    counts as UP per Polymarket convention), else 0.0. The executor receives
    the resolution price from the cycle (which has the Binance feed),
    credits/debits the position, and closes it.

Threading: this object is created and used inside the crypto_lag asyncio task.
It does its own HTTP polling via a private aiohttp ClientSession.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

try:
    import aiohttp
except ImportError as e:
    raise ImportError("crypto_lag.paper_executor requires aiohttp") from e

from .order_engine import (
    FEE_RATE_CRYPTO, MAKER_REBATE_SHARE,
    expected_maker_rebate, parabolic_fee,
)
from .state import RestingOrder, Position


def _best_level(levels, want_max: bool):
    """Pick the best (price, size) from a Polymarket book ladder.

    Polymarket returns bids and asks as lists of {"price": str, "size": str},
    where bids are sorted ASCENDING (worst → best) and asks DESCENDING
    (worst → best). To be defensive against any future ordering change
    (and to handle empty / malformed entries), we scan the entire ladder
    and pick the max-price level on the bid side, min-price on the ask
    side. Returns (price, size) as floats, or None if no valid level.
    """
    if not levels:
        return None
    best_p: float = None  # type: ignore[assignment]
    best_s: float = 0.0
    for lvl in levels:
        try:
            p = float(lvl["price"])
            s = float(lvl.get("size", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        if s <= 0:
            continue
        if best_p is None:
            best_p, best_s = p, s
            continue
        if want_max and p > best_p:
            best_p, best_s = p, s
        elif (not want_max) and p < best_p:
            best_p, best_s = p, s
    if best_p is None:
        return None
    return (best_p, best_s)

logger = logging.getLogger("polymarket_bot.crypto_lag.paper")


@dataclass
class _FillEvent:
    order_id: str
    condition_id: str
    symbol: str
    side: str             # BUY | SELL
    outcome: str          # YES | NO
    fill_price: float
    fill_size_usdc: float
    is_adverse: bool
    ts: float
    rebate_usdc: float = 0.0   # maker rebate credited by this fill (≥ 0)
    fee_paid_usdc: float = 0.0 # 0 for makers, > 0 only if executor models taker
                                # crossings (currently we never cross)
    # Market metadata copied from the RestingOrder so downstream loggers
    # (database.log_crypto_lag_fill) can persist `market_slug` instead of NULL.
    market_slug: str = ""


@dataclass
class _CloseEvent:
    condition_id: str
    symbol: str
    realized_pnl_usdc: float
    final_yes_price: float        # 1.0 or 0.0 in resolution; spot mid otherwise
    ts: float
    reason: str
    accumulated_rebate_usdc: float = 0.0  # for forensics on the dashboard


class PaperExecutor:
    """In-process maker simulator. NOT thread-safe — call from one asyncio task."""

    CLOB_BOOK_URL = "https://clob.polymarket.com/book"

    def __init__(
        self,
        clob_base: str = "https://clob.polymarket.com",
        book_poll_seconds: float = 1.5,
        q_toxic: float = 0.30,
        adverse_haircut_pct: float = 0.015,
        rng_seed: Optional[int] = None,
        fee_rate: float = FEE_RATE_CRYPTO,
        maker_rebate_share: float = MAKER_REBATE_SHARE,
        queue_position_enabled: bool = True,
        # ─── LIVE-realism knobs (added 2026-05-08) ──────────────────
        # Polymarket CLOB does NOT pay maker rebates — the simulator was
        # crediting 20% of the parabolic fee back to makers, which doesn't
        # exist in LIVE. Set this true to zero out rebates and match
        # production reality. The flag exists (instead of just hardcoding
        # 0) so old runs can be replayed with the original assumption.
        live_realistic_rebates: bool = True,
        # Order race-loss probability for taker fills. In LIVE the bot's
        # IOC order takes 100-300ms to reach the matching engine; in that
        # window the book level can disappear (filled by someone else,
        # cancelled by the maker, etc.). 0.25 = 25% of crossings turn
        # into 0-fill cancels.
        taker_race_lost_pct: float = 0.25,
        # Adverse-selection scaling at extreme prices. base_q_toxic is
        # used at p=0.5; as p approaches 0 or 1, q_toxic ramps up to
        # base + (1-base) × extreme_factor where extreme_factor ∈ [0,1].
        # extreme_factor = (1 - 4·p·(1-p))^2 — quadratic in distance
        # from p=0.5. At p=0.10 that's ~0.4 → q_toxic ≈ 0.58 if base=0.30.
        q_toxic_extreme_scaling: bool = True,
        # Depth haircut for thin books. In LIVE, the visible size at the
        # best level is partially "spoof" (cancelled before our IOC lands)
        # in markets with low overall liquidity. The haircut applies a
        # discount to the visible USDC at the matching level so we fill
        # less than the simulator would otherwise allow. Empirically:
        #   visible < $200  → 50% haircut (very thin / spoofy)
        #   visible < $500  → 35% haircut
        #   visible < $1000 → 20% haircut
        #   visible ≥ $1000 → 0%
        depth_haircut_enabled: bool = True,
        # Maker race-lost: when the book crosses into our resting maker,
        # in LIVE we sometimes lose the fill to a competitor maker who
        # posted at the same price milliseconds earlier (FIFO priority)
        # or to a faster IOC that hit before us. Higher in thin books
        # where queue jostling is dominated by a few players.
        maker_race_lost_pct: float = 0.15,
        # Queue advancement haircut: in LIVE, when visible size at our
        # level drops, ~50% of those size reductions are cancellations
        # (which do NOT advance us in the FIFO queue) while ~50% are
        # actual fills (which do). The simulator pre-fix advanced us
        # for 100% of reductions — too generous. With this factor, only
        # `queue_advance_credit_pct` of the observed shrinkage counts
        # toward decreasing our queue debt.
        queue_advance_credit_pct: float = 0.50,
        # F1 — adverse-fill size attenuation. When the toxicity coin flip
        # marks a fill as adverse, the simulator pre-fix only nudged
        # fill_price by `adverse_haircut_pct`. In LIVE an informed
        # counterparty almost always reduces SIZE (or cancels) rather than
        # giving us full notional at a slightly worse mark. The fill_size
        # is multiplied by `(1 - q_tox * adverse_size_attenuation)`, with a
        # floor at `min_fill_usdc` below which the match is treated as a
        # cancel. attenuation=1.0 + q_tox=0.7 → 70% size cut, which matches
        # Glosten-Milgrom adverse-selection magnitudes for binary markets.
        adverse_size_attenuation: float = 1.0,
        min_fill_usdc: float = 0.50,
        # F2 — extreme-price race-lost scaling. The same quadratic shape
        # used by `_effective_q_toxic` ramps `maker_race_lost_pct` from its
        # base value at p=0.5 up to `maker_race_lost_max` at p∈{0,1}.
        # Captures the empirical observation that FIFO competition in tail
        # quotes is dominated by a few informed market makers.
        maker_race_lost_max: float = 0.65,
        # F3 — absolute cap on `_effective_q_toxic`. Without it, q_tox can
        # exceed 0.85 at p<0.05; combined with F1 that would zero out almost
        # every tail fill and leave the simulator under-trading. 0.70 keeps
        # the model directionally honest.
        q_toxic_extreme_cap: float = 0.70,
        # F4 — extreme-price depth multiplier. Scales the visible-size
        # haircut by an extra factor when the order rests in the tails
        # (spoof rates are empirically 50-70% there vs 20-30% near mid).
        depth_extreme_multiplier: float = 0.50,
        depth_near_extreme_multiplier: float = 0.75,
    ):
        self.clob_base = clob_base.rstrip("/")
        self.book_poll_seconds = book_poll_seconds
        self.q_toxic = q_toxic
        self.adverse_haircut_pct = adverse_haircut_pct
        self._rng = random.Random(rng_seed)
        self.fee_rate = float(fee_rate)
        # Force rebate share to 0 in live-realistic mode — Polymarket
        # CLOB doesn't pay makers; the previous +20% was simulator-only fiction.
        self.maker_rebate_share = 0.0 if live_realistic_rebates else float(maker_rebate_share)
        self.queue_position_enabled = bool(queue_position_enabled)
        self.live_realistic_rebates = bool(live_realistic_rebates)
        self.taker_race_lost_pct = float(max(0.0, min(1.0, taker_race_lost_pct)))
        self.q_toxic_extreme_scaling = bool(q_toxic_extreme_scaling)
        self.depth_haircut_enabled = bool(depth_haircut_enabled)
        self.maker_race_lost_pct = float(max(0.0, min(1.0, maker_race_lost_pct)))
        self.queue_advance_credit_pct = float(max(0.0, min(1.0, queue_advance_credit_pct)))
        # F1
        self.adverse_size_attenuation = float(max(0.0, adverse_size_attenuation))
        self.min_fill_usdc = float(max(0.0, min_fill_usdc))
        # F2
        self.maker_race_lost_max = float(max(self.maker_race_lost_pct, min(1.0, maker_race_lost_max)))
        # F3
        self.q_toxic_extreme_cap = float(max(self.q_toxic, min(1.0, q_toxic_extreme_cap)))
        # F4
        self.depth_extreme_multiplier = float(max(0.0, min(1.0, depth_extreme_multiplier)))
        self.depth_near_extreme_multiplier = float(max(0.0, min(1.0, depth_near_extreme_multiplier)))
        # F7 — counters surfaced to the dashboard / structured logs
        self.adverse_size_truncated_count: int = 0
        self.extreme_race_lost_count: int = 0
        self.extreme_q_toxic_capped_count: int = 0
        self.depth_extreme_haircut_count: int = 0

        self._session: Optional[aiohttp.ClientSession] = None
        self._book_cache: dict[str, dict] = {}        # token_id → book
        self._book_cache_ts: dict[str, float] = {}
        self._resting: dict[str, RestingOrder] = {}   # order_id → order
        self._positions: dict[str, Position] = {}     # condition_id → Position
        self._fill_log: list[_FillEvent] = []
        self._close_log: list[_CloseEvent] = []
        # Token id → outcome ("YES" | "NO"). Populated by the cycle when it
        # places an order so we know which side of the book we're matching.
        self._token_to_outcome: dict[str, str] = {}
        # Token id → condition_id and symbol, used by resolve()
        self._token_meta: dict[str, dict] = {}
        # order_id → queue debt in USDC at our price level. Decrements on each
        # poll as visible size at our level drops (i.e. fills clear the queue
        # ahead of us).
        self._queue_debt: dict[str, float] = {}
        # condition_id → accumulated rebate (in USDC) over the life of the
        # position. Settled on resolve_market() into realized PnL.
        self._rebate_acc: dict[str, float] = {}

    # ─── lifecycle ──────────────────────────────────────────────
    async def start(self) -> None:
        if self._session is None:
            self._session = aiohttp.ClientSession()

    async def stop(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    # ─── public API for order_engine ────────────────────────────
    def register_market_tokens(
        self, condition_id: str, symbol: str, token_yes: str, token_no: str
    ) -> None:
        self._token_to_outcome[token_yes] = "YES"
        self._token_to_outcome[token_no] = "NO"
        self._token_meta[token_yes] = {"condition_id": condition_id, "symbol": symbol}
        self._token_meta[token_no] = {"condition_id": condition_id, "symbol": symbol}

    async def place_order(self, order: RestingOrder, token_id: str) -> str:
        """Accept a maker order. Returns the local order_id and stores it for
        match attempts. `token_id` is the CLOB token we're pretending to send to.
        Stored on the order via `order.external_order_id` for auditing."""
        order.external_order_id = f"paper-{token_id[:8]}-{order.order_id[:8]}"
        order.status = "open"
        # Tag the order with the token id (we need it on match)
        order._paper_token_id = token_id  # type: ignore[attr-defined]
        self._resting[order.order_id] = order
        # Initialize queue position from the most recent book snapshot we have.
        self._init_queue_debt(order, token_id)
        return order.external_order_id

    async def cancel_order(self, order_id: str) -> bool:
        o = self._resting.pop(order_id, None)
        if o is None:
            return False
        o.status = "canceled"
        self._queue_debt.pop(order_id, None)
        return True

    async def poll_fills(self) -> list[_FillEvent]:
        """Refresh orderbooks for all tokens with resting orders, attempt to
        match each resting order, return any fills produced this call."""
        if not self._resting:
            return []
        # Group resting orders by token id so we only poll each book once
        by_token: dict[str, list[RestingOrder]] = {}
        for o in list(self._resting.values()):
            tk = getattr(o, "_paper_token_id", None)
            if not tk:
                continue
            by_token.setdefault(tk, []).append(o)

        fills: list[_FillEvent] = []
        for token_id, orders in by_token.items():
            book = await self._get_book(token_id)
            if book is None:
                continue
            for order in orders:
                if order.status not in ("open", "partially_filled"):
                    continue
                self._update_queue_debt(order, book)
                ev = self._try_match(order, book)
                if ev is not None:
                    fills.append(ev)
                    self._fill_log.append(ev)
                    self._apply_fill_to_position(ev, order)
                    if order.filled_size_usdc >= order.size_usdc - 1e-6:
                        order.status = "filled"
                        self._resting.pop(order.order_id, None)
                        self._queue_debt.pop(order.order_id, None)
                    else:
                        order.status = "partially_filled"
        return fills

    async def resolve_market(
        self, condition_id: str, yes_outcome_value: float, ts: Optional[float] = None
    ) -> Optional[_CloseEvent]:
        """Mark a condition resolved; settle any open position at YES=value
        (1.0 or 0.0 in real resolution). Cancels any resting orders for that
        condition. Adds accumulated maker rebates to realized PnL."""
        ts = ts or time.time()
        # Cancel resting orders on this market
        for oid, o in list(self._resting.items()):
            if o.condition_id == condition_id:
                self._resting.pop(oid, None)
                o.status = "canceled"
                self._queue_debt.pop(oid, None)

        pos = self._positions.pop(condition_id, None)
        rebate = self._rebate_acc.pop(condition_id, 0.0)
        if pos is None:
            if rebate <= 0:
                return None
            # Pure rebate flow with no holding (filled-and-flatted earlier).
            ev = _CloseEvent(
                condition_id=condition_id,
                symbol="",
                realized_pnl_usdc=float(rebate),
                final_yes_price=float(yes_outcome_value),
                ts=ts,
                reason="rebate_only",
                accumulated_rebate_usdc=float(rebate),
            )
            self._close_log.append(ev)
            return ev
        # P&L = (yes_value - avg_entry_price) * size_usdc / avg_entry_price for a YES long
        # but since size is in USDC of notional, simpler:
        #   shares = size_usdc / avg_entry_price  (if YES long)
        #   payoff = shares * yes_outcome_value
        #   pnl = payoff - size_usdc + accumulated_rebate
        if pos.outcome == "YES":
            shares = abs(pos.size_usdc) / max(0.001, pos.avg_entry_price)
            payoff = shares * float(yes_outcome_value)
        else:  # NO position
            shares = abs(pos.size_usdc) / max(0.001, pos.avg_entry_price)
            payoff = shares * (1.0 - float(yes_outcome_value))
        # If the position is short YES (size_usdc < 0), the cost basis was
        # CREDITED to us at entry (we received the proceeds). At settlement
        # we owe the YES payout. PnL = entry_proceeds - payout.
        # For long YES (size_usdc > 0): PnL = payout - cost_basis.
        cost_basis = abs(pos.size_usdc)
        if pos.size_usdc >= 0:
            pnl_directional = payoff - cost_basis
        else:
            pnl_directional = cost_basis - payoff
        pnl = pnl_directional + float(rebate)
        ev = _CloseEvent(
            condition_id=condition_id,
            symbol=pos.symbol,
            realized_pnl_usdc=float(pnl),
            final_yes_price=float(yes_outcome_value),
            ts=ts,
            reason="resolved",
            accumulated_rebate_usdc=float(rebate),
        )
        self._close_log.append(ev)
        return ev

    def get_queue_debt(self, order_id: str) -> float:
        """Diagnostics accessor — returns the current modeled queue debt
        (USDC of volume ahead of us at our level) for a resting order. Used
        by the engine's placement_logger to persist alongside the order in
        crypto_lag_quotes for fill-rate analysis."""
        return float(self._queue_debt.get(order_id, 0.0))

    def get_position(self, condition_id: str) -> Optional[Position]:
        return self._positions.get(condition_id)

    def get_all_resting(self) -> list[RestingOrder]:
        return list(self._resting.values())

    def drain_close_log(self) -> list[_CloseEvent]:
        out = self._close_log[:]
        self._close_log.clear()
        return out

    def accumulated_rebate(self, condition_id: str) -> float:
        return float(self._rebate_acc.get(condition_id, 0.0))

    # ─── internals ──────────────────────────────────────────────
    async def _get_book(self, token_id: str) -> Optional[dict]:
        now = time.time()
        if (now - self._book_cache_ts.get(token_id, 0.0)) < self.book_poll_seconds:
            return self._book_cache.get(token_id)
        if self._session is None:
            return None
        try:
            async with self._session.get(
                f"{self.clob_base}/book", params={"token_id": token_id}, timeout=5
            ) as r:
                if r.status != 200:
                    return self._book_cache.get(token_id)
                data = await r.json()
            bids = data.get("bids") or []
            asks = data.get("asks") or []
            # CRITICAL: Polymarket CLOB returns bids in ASCENDING price order
            # (bids[0] is the WORST bid, e.g. 0.01) and asks in DESCENDING
            # order (asks[0] is the WORST ask, e.g. 0.99). Best bid is the
            # MAX bid price; best ask is the MIN ask price. Using bids[0]
            # / asks[0] like the v1 code did was a critical bug that made
            # every market look like spread=0.98 → NO_BOOK.
            best_bid_lvl = _best_level(bids, want_max=True)
            best_ask_lvl = _best_level(asks, want_max=False)
            book = {
                "best_bid": best_bid_lvl[0] if best_bid_lvl is not None else None,
                "best_ask": best_ask_lvl[0] if best_ask_lvl is not None else None,
                "bid_size": best_bid_lvl[1] if best_bid_lvl is not None else 0.0,
                "ask_size": best_ask_lvl[1] if best_ask_lvl is not None else 0.0,
                # Keep the full ladder for queue-position simulation.
                "_raw_bids": bids,
                "_raw_asks": asks,
            }
            self._book_cache[token_id] = book
            self._book_cache_ts[token_id] = now
            return book
        except Exception as exc:
            logger.debug(f"book fetch error for {token_id[:12]}...: {exc}")
            return self._book_cache.get(token_id)

    def _init_queue_debt(self, order: RestingOrder, token_id: str) -> None:
        """Initialize queue debt for a freshly-placed order using the cached
        book snapshot. If our price is BETTER than the best (penny-jump), debt
        is 0 (we're alone). If we JOIN the existing best, debt = visible size
        at that level. Otherwise (we posted at a price NOT yet on the book at
        all) debt is the cumulative visible size at our price or worse — i.e.
        every order strictly better than ours.
        """
        if not self.queue_position_enabled:
            self._queue_debt[order.order_id] = 0.0
            return
        book = self._book_cache.get(token_id)
        if book is None:
            self._queue_debt[order.order_id] = 0.0
            return
        debt = self._compute_queue_debt(order, book)
        self._queue_debt[order.order_id] = debt

    def _compute_queue_debt(self, order: RestingOrder, book: dict) -> float:
        """USDC of volume that must clear ahead of us before we fill.

        BID convention (we're buying YES): the "queue" we're behind is OTHER
        bids at our price or higher. ASK is symmetric on the other side.
        """
        if not self.queue_position_enabled:
            return 0.0
        side_levels_key = "_raw_bids" if order.side == "BUY" else "_raw_asks"
        levels = book.get(side_levels_key) or []
        # Sort BIDS desc (highest first), ASKS asc (lowest first).
        try:
            parsed = [(float(lvl["price"]), float(lvl.get("size", 0.0))) for lvl in levels]
        except (TypeError, ValueError, KeyError):
            return 0.0
        if order.side == "BUY":
            parsed.sort(key=lambda x: -x[0])  # high-to-low
            # Queue ahead = sum of size at price > ours, plus all of size at
            # our price (we joined at the back).
            debt_shares = 0.0
            for price, size in parsed:
                if price > order.price + 1e-9:
                    debt_shares += size
                elif abs(price - order.price) <= 1e-9:
                    debt_shares += size
                    break
                else:
                    break
            # Convert shares × price → USDC notional.
            return float(debt_shares * order.price)
        else:
            parsed.sort(key=lambda x: x[0])  # low-to-high
            debt_shares = 0.0
            for price, size in parsed:
                if price < order.price - 1e-9:
                    debt_shares += size
                elif abs(price - order.price) <= 1e-9:
                    debt_shares += size
                    break
                else:
                    break
            return float(debt_shares * order.price)

    def _update_queue_debt(self, order: RestingOrder, book: dict) -> None:
        """Each poll, decrement our queue debt by however much the visible
        depth at our level shrank since the last snapshot. This is a
        conservative proxy for "the queue moved" without access to a true
        delta feed.

        LIVE-realism: visible-depth shrinkage is NOT 100% fills. A large
        fraction is cancellations, which do NOT advance us in the FIFO queue
        (the cancelled order never occupied our slot). We credit only
        `queue_advance_credit_pct` of observed shrinkage. Empirically in
        Polymarket crypto books, ~50% of shrinkage events are cancels in
        thin / volatile markets.
        """
        if not self.queue_position_enabled:
            self._queue_debt[order.order_id] = 0.0
            return
        prev = self._queue_debt.get(order.order_id, 0.0)
        if prev <= 0.0:
            return
        new_debt_raw = self._compute_queue_debt(order, book)
        if new_debt_raw < prev:
            # Only credit a fraction of the apparent advancement.
            shrinkage = prev - new_debt_raw
            credited_shrinkage = shrinkage * self.queue_advance_credit_pct
            new_debt = prev - credited_shrinkage
        else:
            new_debt = new_debt_raw
        # Debt only ever monotonically decreases until we cancel/replace; if
        # the book grew (someone else joined behind us, or above our level)
        # we conservatively reset to the new value (it reflects current
        # standing) — this means joining a thicker book penalizes us.
        self._queue_debt[order.order_id] = min(prev, new_debt)

    def _depth_multiplier(
        self, visible_size_usdc: float, price: Optional[float] = None
    ) -> float:
        """Return a multiplier ∈ [0.25, 1.0] applied to visible book depth to
        emulate spoof / pulled liquidity in LIVE. Thinner books → bigger
        haircut. Tuned to be conservative: real spoof rate in low-liquidity
        Polymarket crypto books has been observed at 30-50% in audits, and
        50-70% in the deep tails (F4).
        """
        if not self.depth_haircut_enabled:
            return 1.0
        v = float(visible_size_usdc or 0.0)
        if v < 200.0:
            base = 0.50
        elif v < 500.0:
            base = 0.65
        elif v < 1000.0:
            base = 0.80
        else:
            base = 1.0
        if price is None:
            return base
        # F4 — extra haircut in the tails. Spoof rate is empirically much
        # higher when the resting level is far from mid; visible size there
        # is mostly decorative quotes that cancel before our match lands.
        p = max(0.0, min(1.0, float(price)))
        if p < 0.15 or p > 0.85:
            self.depth_extreme_haircut_count += 1
            return base * self.depth_extreme_multiplier
        if p < 0.25 or p > 0.75:
            return base * self.depth_near_extreme_multiplier
        return base

    def _effective_q_toxic(self, price: float) -> float:
        """Scale q_toxic up at extreme prices, capped at q_toxic_extreme_cap.

        Rationale: in LIVE, a counterparty willing to sell YES at $0.05 or
        buy at $0.95 almost certainly has private information. Adverse
        selection ramps up as p moves away from 0.5. The quadratic
        `(1 - 4·p·(1-p))^2` gives 0 at p=0.5 and 1 at p∈{0,1}.

        F3 — the absolute ceiling `q_toxic_extreme_cap` (default 0.70)
        prevents the formula from saturating near 1.0 at p<0.05; without it,
        F1 (size attenuation) would zero out almost every tail fill and the
        simulator would stop trading entirely in the tails — losing the
        ability to discover whether a real edge exists there.
        """
        if not self.q_toxic_extreme_scaling:
            return self.q_toxic
        p = max(0.001, min(0.999, float(price)))
        info_signal = (1.0 - 4.0 * p * (1.0 - p)) ** 2  # ∈ [0, 1]
        # Blend: at p=0.5 → base; at extremes → base + (1 - base) × signal
        raw = self.q_toxic + (1.0 - self.q_toxic) * info_signal
        if raw > self.q_toxic_extreme_cap:
            self.extreme_q_toxic_capped_count += 1
            return float(self.q_toxic_extreme_cap)
        return float(raw)

    def _effective_maker_race_lost_pct(self, price: float) -> float:
        """F2 — ramp `maker_race_lost_pct` from base at p=0.5 up to
        `maker_race_lost_max` in the tails using the same quadratic shape
        as `_effective_q_toxic`. In LIVE, FIFO competition in tail quotes
        is dominated by a few informed market makers; race losses there
        are empirically 50-65%, not the 15% applied uniformly today.
        """
        if self.maker_race_lost_max <= self.maker_race_lost_pct:
            return self.maker_race_lost_pct
        p = max(0.001, min(0.999, float(price)))
        info_signal = (1.0 - 4.0 * p * (1.0 - p)) ** 2  # ∈ [0, 1]
        return float(
            self.maker_race_lost_pct
            + (self.maker_race_lost_max - self.maker_race_lost_pct) * info_signal
        )

    def _try_match(self, order: RestingOrder, book: dict) -> Optional[_FillEvent]:
        if order.side == "BUY":
            counter_price = book.get("best_ask")
            counter_size_shares = float(book.get("ask_size", 0.0))
            counter_size_usdc = counter_size_shares * (counter_price or 0.0)
            crossed = counter_price is not None and counter_price <= order.price + 1e-9
            adverse_direction = +1
        else:  # SELL
            counter_price = book.get("best_bid")
            counter_size_shares = float(book.get("bid_size", 0.0))
            counter_size_usdc = counter_size_shares * (counter_price or 0.0)
            crossed = counter_price is not None and counter_price >= order.price - 1e-9
            adverse_direction = -1
        if not crossed:
            return None
        # Queue debt gates the fill regardless of side.
        debt = self._queue_debt.get(order.order_id, 0.0)
        if self.queue_position_enabled and debt > 0:
            return None
        # F4 — visible-size haircut keyed off our resting price.
        counter_size_usdc *= self._depth_multiplier(counter_size_usdc, price=order.price)
        remaining = max(0.0, order.size_usdc - order.filled_size_usdc)
        fill_size = min(remaining, counter_size_usdc)
        if fill_size <= 0:
            return None
        # LIVE latency: taker race may lose. The crossing was visible via
        # REST polling (≥1.5s lag), but our IOC needs ~150ms to reach the
        # matching engine; the maker may cancel or another taker beat us.
        if order.is_taker and self._rng.random() < self.taker_race_lost_pct:
            return None
        # F2 — maker race-lost ramps with distance to mid. FIFO competition
        # in the tails is dominated by a few informed market makers, so the
        # effective race-lost rate is much higher than the flat base value.
        if not order.is_taker:
            race_lost_pct = self._effective_maker_race_lost_pct(order.price)
            if self._rng.random() < race_lost_pct:
                if race_lost_pct > self.maker_race_lost_pct + 1e-6:
                    self.extreme_race_lost_count += 1
                return None
        q_tox = self._effective_q_toxic(order.price)
        is_adverse = self._rng.random() < q_tox
        fill_price = order.price
        if is_adverse:
            # F1 — informed counterparty: reduce SIZE proportional to q_tox
            # AND nudge the effective fill price against us. Glosten-Milgrom
            # adverse-selection magnitudes for binary prediction markets.
            size_haircut = max(
                0.0,
                1.0 - min(1.0, q_tox * self.adverse_size_attenuation),
            )
            new_size = fill_size * size_haircut
            if new_size + 1e-9 < fill_size:
                self.adverse_size_truncated_count += 1
            fill_size = new_size
            if fill_size < self.min_fill_usdc:
                # Treat as a no-fill: in LIVE the informed counter would
                # have cancelled rather than show up at all.
                return None
            # Mark moves against us. adverse_direction = +1 for BUY (price
            # rises after we hit), -1 for SELL (price falls).
            fill_price = order.price * (1.0 + adverse_direction * self.adverse_haircut_pct)
            fill_price = max(0.01, min(0.99, fill_price))

        order.filled_size_usdc += fill_size
        # Fee accounting splits on whether this order crossed the book at
        # placement (is_taker) or rested:
        #   * Maker fill → 0 fee paid, expected rebate credited.
        #   * Taker fill → fee paid (parabolic fee on filled notional), no
        #                  rebate. The rebate share that the COUNTERPARTY (a
        #                  resting maker) receives is THEIR side, not ours.
        if order.is_taker:
            fee_paid = (
                parabolic_fee(fill_price, fee_rate=self.fee_rate) * fill_size
            )
            rebate = 0.0
            prev_fee = self._rebate_acc.get(order.condition_id, 0.0)
            # Reuse _rebate_acc as a signed accumulator (rebates += positive,
            # fees += negative). resolve_market sums it directly into PnL.
            self._rebate_acc[order.condition_id] = prev_fee - float(fee_paid)
        else:
            rebate = (
                expected_maker_rebate(
                    fill_price,
                    fee_rate=self.fee_rate,
                    rebate_share=self.maker_rebate_share,
                )
                * fill_size
            )
            fee_paid = 0.0
            prev_rebate = self._rebate_acc.get(order.condition_id, 0.0)
            self._rebate_acc[order.condition_id] = prev_rebate + float(rebate)
        return _FillEvent(
            order_id=order.order_id,
            condition_id=order.condition_id,
            symbol=order.symbol,
            side=order.side,
            outcome=order.outcome,
            fill_price=fill_price,
            fill_size_usdc=fill_size,
            is_adverse=is_adverse,
            ts=time.time(),
            rebate_usdc=float(rebate),
            fee_paid_usdc=float(fee_paid),
            market_slug=order.market_slug,
        )

    def _apply_fill_to_position(
        self, ev: _FillEvent, order: Optional[RestingOrder] = None
    ) -> None:
        pos = self._positions.get(ev.condition_id)
        # Map (side, outcome) → signed direction for our YES-centric position
        # We track position.outcome as the side the bot is long on.
        if pos is None:
            self._positions[ev.condition_id] = Position(
                condition_id=ev.condition_id,
                symbol=ev.symbol,
                outcome=ev.outcome,
                size_usdc=ev.fill_size_usdc if ev.side == "BUY" else -ev.fill_size_usdc,
                avg_entry_price=ev.fill_price,
                opened_ts=ev.ts,
                last_fill_ts=ev.ts,
                # Snapshot market metadata from the RestingOrder so resolution
                # works even after Polymarket drops the market off Gamma.
                end_ts=float(order.end_ts) if order is not None else 0.0,
                strike_price=float(order.strike_price) if order is not None else 0.0,
                market_slug=str(order.market_slug) if order is not None else "",
            )
            return
        # Two-sided MM: BUYs add long YES, SELLs add short YES. We track the
        # net position with a signed size; cost_basis VWAP is updated with the
        # absolute notional so we can compute a sensible avg_entry_price.
        if ev.side == "BUY":
            new_size = pos.size_usdc + ev.fill_size_usdc
        else:
            new_size = pos.size_usdc - ev.fill_size_usdc
        # VWAP only if we're not flipping sign and not flat
        if (pos.size_usdc >= 0 and new_size >= 0 and ev.side == "BUY") or \
           (pos.size_usdc <= 0 and new_size <= 0 and ev.side == "SELL"):
            tot_notional_prev = abs(pos.size_usdc) * pos.avg_entry_price
            new_notional_add = ev.fill_size_usdc * ev.fill_price
            tot_abs = abs(new_size)
            if tot_abs > 1e-9:
                pos.avg_entry_price = (tot_notional_prev + new_notional_add) / tot_abs
        pos.size_usdc = new_size
        pos.last_fill_ts = ev.ts
