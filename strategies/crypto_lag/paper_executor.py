"""DEMO-mode order executor for crypto-lag.

Simulates fills against the real Polymarket CLOB orderbook (read via REST).
The executor is the single point that translates "we placed an order at price P
size S" into trade events. In LIVE mode the order_engine talks to a different
backend (not implemented yet); in DEMO this is what runs.

Fill model (per polled book snapshot):
  - YES BID at price P: fills when real best_ask ≤ P. Filled size is
    min(remaining_order_size_usdc, ask_size_at_or_below_P_in_usdc).
  - YES ASK at price P (a sell of YES): fills when real best_bid ≥ P.
  - With probability `q_toxic` the fill is tagged as adverse — we still fill,
    but record an adverse_selection cost so DEMO P&L matches LIVE more closely.
  - Partial fills are supported: an order can be filled across multiple ticks.

Resolution model:
  - At endDate the YES outcome resolves to 1.0 if Binance close > strike, else
    0.0. The executor receives the resolution price from the cycle (which has
    the Binance feed), credits/debits the position, and closes it.

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

from .state import RestingOrder, Position

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


@dataclass
class _CloseEvent:
    condition_id: str
    symbol: str
    realized_pnl_usdc: float
    final_yes_price: float        # 1.0 or 0.0 in resolution; spot mid otherwise
    ts: float
    reason: str


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
    ):
        self.clob_base = clob_base.rstrip("/")
        self.book_poll_seconds = book_poll_seconds
        self.q_toxic = q_toxic
        self.adverse_haircut_pct = adverse_haircut_pct
        self._rng = random.Random(rng_seed)

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
        return order.external_order_id

    async def cancel_order(self, order_id: str) -> bool:
        o = self._resting.pop(order_id, None)
        if o is None:
            return False
        o.status = "canceled"
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
                ev = self._try_match(order, book)
                if ev is not None:
                    fills.append(ev)
                    self._fill_log.append(ev)
                    self._apply_fill_to_position(ev)
                    if order.filled_size_usdc >= order.size_usdc - 1e-6:
                        order.status = "filled"
                        self._resting.pop(order.order_id, None)
                    else:
                        order.status = "partially_filled"
        return fills

    async def resolve_market(
        self, condition_id: str, yes_outcome_value: float, ts: Optional[float] = None
    ) -> Optional[_CloseEvent]:
        """Mark a condition resolved; settle any open position at YES=value
        (1.0 or 0.0 in real resolution). Cancels any resting orders for that
        condition."""
        ts = ts or time.time()
        # Cancel resting orders on this market
        for oid, o in list(self._resting.items()):
            if o.condition_id == condition_id:
                self._resting.pop(oid, None)
                o.status = "canceled"

        pos = self._positions.pop(condition_id, None)
        if pos is None:
            return None
        # P&L = (yes_value - avg_entry_price) * size_usdc / avg_entry_price for a YES long
        # but since size is in USDC of notional, simpler:
        #   shares = size_usdc / avg_entry_price  (if YES long)
        #   payoff = shares * yes_outcome_value
        #   pnl = payoff - size_usdc
        if pos.outcome == "YES":
            shares = pos.size_usdc / max(0.001, pos.avg_entry_price)
            payoff = shares * yes_outcome_value
        else:  # NO position
            shares = pos.size_usdc / max(0.001, pos.avg_entry_price)
            payoff = shares * (1.0 - yes_outcome_value)
        pnl = payoff - pos.size_usdc
        ev = _CloseEvent(
            condition_id=condition_id,
            symbol=pos.symbol,
            realized_pnl_usdc=pnl,
            final_yes_price=yes_outcome_value,
            ts=ts,
            reason="resolved",
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
            book = {
                "best_bid": float(bids[0]["price"]) if bids else None,
                "best_ask": float(asks[0]["price"]) if asks else None,
                "bid_size": float(bids[0].get("size", 0.0)) if bids else 0.0,
                "ask_size": float(asks[0].get("size", 0.0)) if asks else 0.0,
            }
            self._book_cache[token_id] = book
            self._book_cache_ts[token_id] = now
            return book
        except Exception as exc:
            logger.debug(f"book fetch error for {token_id[:12]}...: {exc}")
            return self._book_cache.get(token_id)

    def _try_match(self, order: RestingOrder, book: dict) -> Optional[_FillEvent]:
        if order.side == "BUY":
            ask = book.get("best_ask")
            ask_size_usdc = float(book.get("ask_size", 0.0)) * (ask or 0.0)
            if ask is None or ask > order.price + 1e-9:
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
            bid_size_usdc = float(book.get("bid_size", 0.0)) * (bid or 0.0)
            if bid is None or bid < order.price - 1e-9:
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
        )

    def _apply_fill_to_position(self, ev: _FillEvent) -> None:
        pos = self._positions.get(ev.condition_id)
        # Map (side, outcome) → signed direction for our YES-centric position
        # We track position.outcome as the side the bot is long on.
        # Simplification for v1: only YES positions are aggregated (extend later).
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
        # Average up/down with VWAP if same side, else reduce
        if ev.side == "BUY":
            new_size = pos.size_usdc + ev.fill_size_usdc
            if new_size > 0:
                pos.avg_entry_price = (
                    (pos.avg_entry_price * pos.size_usdc + ev.fill_price * ev.fill_size_usdc)
                    / new_size
                )
            pos.size_usdc = new_size
        else:
            pos.size_usdc -= ev.fill_size_usdc
        pos.last_fill_ts = ev.ts
