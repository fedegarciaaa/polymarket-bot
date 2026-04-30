"""Maker-order lifecycle for crypto-lag.

Translates a model decision ("we think P(YES) = 0.55, the book is 0.50/0.51,
quote a YES bid at 0.51") into actual orders, and handles repricing /
cancellation as the model output drifts.

This module is backend-agnostic: it talks to either `PaperExecutor` (DEMO) or
a future LIVE adapter via the same shape (`place_order`, `cancel_order`,
`poll_fills`). The cycle owns the executor instance and passes it in.

Design choices:
  - Penny-jump: post `best_bid + tick` only if our edge > 2 cents; otherwise
    join at `best_bid` and rely on time priority. (Avoids paying 1c spread for
    marginal-edge quotes.)
  - Replace-only-on-significant-drift: cancel/replace if the model's fair_mid
    has moved by ≥ `replace_threshold_cents`, OR our level is no longer best,
    OR the order has been resting for `max_order_age_seconds`.
  - One BID and one ASK at most per market — no quoting both sides yet (v1).
    Two-sided market making is a clean v2 once the basic single-sided strategy
    proves out.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, Protocol

from .state import PolyCryptoMarket, RestingOrder
from .risk import CryptoLagRisk

logger = logging.getLogger("polymarket_bot.crypto_lag.engine")


class _ExecutorProto(Protocol):
    """Minimal interface required from the executor (paper or live)."""
    async def place_order(self, order: RestingOrder, token_id: str) -> str: ...
    async def cancel_order(self, order_id: str) -> bool: ...


@dataclass
class QuoteDecision:
    """Output of the cycle's decision step. The order_engine consumes this."""
    side: str               # "BID" | "ASK" | "BOTH" | "NONE"
    fair_mid: float         # the model's best estimate of YES probability
    edge_bid: float         # how much we'd earn if filled at best_bid
    edge_ask: float
    poly_best_bid: float
    poly_best_ask: float
    target_size_usdc: float
    bid_price: Optional[float] = None  # if quoting BID, the price to post
    ask_price: Optional[float] = None


class MakerOrderEngine:
    def __init__(
        self,
        executor: _ExecutorProto,
        risk: CryptoLagRisk,
        edge_threshold_cents: float = 2.0,
        replace_threshold_cents: float = 1.0,
        max_order_age_seconds: float = 30.0,
    ):
        self.executor = executor
        self.risk = risk
        self.edge_threshold = edge_threshold_cents / 100.0
        self.replace_threshold = replace_threshold_cents / 100.0
        self.max_order_age = max_order_age_seconds

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
            await self._upsert(
                market=market, side="BID", price=decision.bid_price,
                size_usdc=decision.target_size_usdc, now=now,
            )
        # 4. (Re)place ASK
        if decision.side in ("ASK", "BOTH") and decision.ask_price is not None:
            await self._upsert(
                market=market, side="ASK", price=decision.ask_price,
                size_usdc=decision.target_size_usdc, now=now,
            )

    async def cancel_all_for(self, condition_id: str) -> None:
        for side in list(self._open.get(condition_id, {}).keys()):
            await self._cancel(condition_id, side)

    async def cancel_all(self) -> None:
        for cid in list(self._open.keys()):
            await self.cancel_all_for(cid)

    def open_orders(self, condition_id: str) -> dict[str, RestingOrder]:
        return dict(self._open.get(condition_id, {}))

    # ─── decision helper (used by cycle) ────────────────────────
    def build_decision(
        self,
        fair_mid: float,
        poly_best_bid: float,
        poly_best_ask: float,
        target_size_usdc: float,
        tick: float = 0.01,
    ) -> QuoteDecision:
        """Convert a model probability + book snapshot into a quote decision.

        We only quote the side where we have the larger edge after threshold —
        no two-sided market making yet. If both sides have edge above threshold
        the bigger one wins.
        """
        edge_bid = fair_mid - poly_best_bid          # value of buying at bid (we'd long YES cheap)
        edge_ask = poly_best_ask - fair_mid          # value of selling at ask (short YES expensive)

        bid_price: Optional[float] = None
        ask_price: Optional[float] = None
        side = "NONE"

        if edge_bid > self.edge_threshold and edge_bid >= edge_ask:
            # Penny-jump only if edge ≥ 2 ticks; otherwise join the queue.
            # Clamp strictly inside spread so we never cross the book (which
            # would either be rejected by the maker post-only check or executed
            # as a taker — paying fees, defeating the strategy).
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
            self._open[cid][side] = order
            self.risk.on_order_open()
            logger.info(
                f"placed {side} {market.symbol} {market.market_slug[:30]}: "
                f"{size_usdc:.2f}@{price:.2f} (ext={ext})"
            )
        except Exception as exc:
            logger.warning(f"place {side} {market.market_slug}: {exc}")

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
