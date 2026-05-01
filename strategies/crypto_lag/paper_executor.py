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

from .order_engine import FEE_RATE_CRYPTO, MAKER_REBATE_SHARE, expected_maker_rebate
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
    ):
        self.clob_base = clob_base.rstrip("/")
        self.book_poll_seconds = book_poll_seconds
        self.q_toxic = q_toxic
        self.adverse_haircut_pct = adverse_haircut_pct
        self._rng = random.Random(rng_seed)
        self.fee_rate = float(fee_rate)
        self.maker_rebate_share = float(maker_rebate_share)
        self.queue_position_enabled = bool(queue_position_enabled)

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
                    self._apply_fill_to_position(ev)
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
        delta feed."""
        if not self.queue_position_enabled:
            self._queue_debt[order.order_id] = 0.0
            return
        prev = self._queue_debt.get(order.order_id, 0.0)
        if prev <= 0.0:
            return
        new_debt = self._compute_queue_debt(order, book)
        # Debt only ever monotonically decreases until we cancel/replace; if
        # the book grew (someone else joined behind us, or above our level)
        # we conservatively reset to the new value (it reflects current
        # standing) — this means joining a thicker book penalizes us.
        self._queue_debt[order.order_id] = min(prev, new_debt)

    def _try_match(self, order: RestingOrder, book: dict) -> Optional[_FillEvent]:
        if order.side == "BUY":
            ask = book.get("best_ask")
            ask_size_shares = float(book.get("ask_size", 0.0))
            ask_size_usdc = ask_size_shares * (ask or 0.0)
            if ask is None or ask > order.price + 1e-9:
                return None
            # If we still owe queue debt, we can't fill yet.
            debt = self._queue_debt.get(order.order_id, 0.0)
            if self.queue_position_enabled and debt > 0:
                return None
            remaining = max(0.0, order.size_usdc - order.filled_size_usdc)
            fill_size = min(remaining, ask_size_usdc)
            if fill_size <= 0:
                return None
            is_adverse = self._rng.random() < self.q_toxic
            fill_price = order.price
            if is_adverse:
                # Toxic taker — we get filled but the mark immediately moves
                # adverse_haircut_pct against us. We model this by recording a
                # WORSE effective fill price.
                fill_price = order.price * (1.0 + self.adverse_haircut_pct)
                fill_price = min(fill_price, 0.99)
        else:  # SELL
            bid = book.get("best_bid")
            bid_size_shares = float(book.get("bid_size", 0.0))
            bid_size_usdc = bid_size_shares * (bid or 0.0)
            if bid is None or bid < order.price - 1e-9:
                return None
            debt = self._queue_debt.get(order.order_id, 0.0)
            if self.queue_position_enabled and debt > 0:
                return None
            remaining = max(0.0, order.size_usdc - order.filled_size_usdc)
            fill_size = min(remaining, bid_size_usdc)
            if fill_size <= 0:
                return None
            is_adverse = self._rng.random() < self.q_toxic
            fill_price = order.price
            if is_adverse:
                fill_price = order.price * (1.0 - self.adverse_haircut_pct)
                fill_price = max(fill_price, 0.01)

        order.filled_size_usdc += fill_size
        # Maker rebate: paid on fill_price, on the notional that was filled.
        # In real Polymarket the rebate accrues daily — we book it immediately
        # so DEMO PnL matches LIVE economics from the first fill.
        rebate = (
            expected_maker_rebate(
                fill_price,
                fee_rate=self.fee_rate,
                rebate_share=self.maker_rebate_share,
            )
            * fill_size
        )
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
            fee_paid_usdc=0.0,
        )

    def _apply_fill_to_position(self, ev: _FillEvent) -> None:
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
