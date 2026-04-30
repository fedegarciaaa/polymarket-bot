"""Persistent Binance WebSocket feed.

Maintains a `MarketState` per symbol via the public combined-streams endpoint:
  wss://stream.binance.com:9443/stream?streams=btcusdt@bookTicker/btcusdt@aggTrade/...

Two streams per symbol:
  - bookTicker: best bid/ask updates (every change)
  - aggTrade:   aggregated trades (signed by `m` flag → buyer is maker = sell flow)

The feed runs as a single `asyncio.Task` that auto-reconnects with exponential
backoff. The `MarketState` instances are shared by reference with consumers
(probability_model / cycle), so consumers always read the latest snapshot.

Staleness is the consumer's responsibility — call `state.is_stale(now, N)`
before quoting. We don't gate writes here; we always write what we receive.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional

try:
    import websockets
    from websockets.exceptions import ConnectionClosed
except ImportError as e:
    raise ImportError(
        "crypto_lag.binance_feed requires the `websockets` package "
        "(>=12.0). Install with: pip install websockets"
    ) from e

from .state import MarketState

logger = logging.getLogger("polymarket_bot.crypto_lag.feed")


class BinanceFeed:
    def __init__(
        self,
        symbols: list[str],
        ws_url: str = "wss://stream.binance.com/stream",
        ws_url_fallback: Optional[str] = "wss://data-stream.binance.vision/stream",
        reconnect_initial_seconds: float = 1.0,
        reconnect_max_seconds: float = 30.0,
    ):
        self.symbols = [s.upper() for s in symbols]
        # Rotate through these URLs on consecutive failures so a regional
        # block on one mirror doesn't kill the feed.
        self._ws_urls: list[str] = [u for u in (ws_url, ws_url_fallback) if u]
        self._url_idx: int = 0
        self.reconnect_initial = reconnect_initial_seconds
        self.reconnect_max = reconnect_max_seconds
        self._states: dict[str, MarketState] = {
            s: MarketState(symbol=s) for s in self.symbols
        }
        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        # On reconnect we mark all states stale — consumers will skip quoting
        # until we've had `2 * stale_seconds` of fresh data again.
        self._reconnect_freeze_until: float = 0.0

    @property
    def ws_url(self) -> str:
        return self._ws_urls[self._url_idx]

    # ─── public surface ────────────────────────────────────────
    def get_state(self, symbol: str) -> Optional[MarketState]:
        return self._states.get(symbol.upper())

    def all_states(self) -> dict[str, MarketState]:
        return dict(self._states)

    def is_in_reconnect_freeze(self, now_ts: Optional[float] = None) -> bool:
        return (now_ts or time.time()) < self._reconnect_freeze_until

    async def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run_forever())

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()

    # ─── internals ──────────────────────────────────────────────
    def _build_stream_url(self, base_url: str) -> str:
        streams = []
        for s in self.symbols:
            sl = s.lower()
            streams.append(f"{sl}@bookTicker")
            streams.append(f"{sl}@aggTrade")
        return f"{base_url}?streams={'/'.join(streams)}"

    async def _run_forever(self) -> None:
        backoff = self.reconnect_initial
        consecutive_failures = 0
        while not self._stop.is_set():
            base = self.ws_url
            url = self._build_stream_url(base)
            try:
                logger.info(f"connecting to Binance WS at {base} ({len(self.symbols)} symbols)")
                async with websockets.connect(
                    url, ping_interval=20, ping_timeout=10, open_timeout=15
                ) as ws:
                    backoff = self.reconnect_initial  # reset on successful connect
                    consecutive_failures = 0
                    async for raw in ws:
                        if self._stop.is_set():
                            break
                        try:
                            self._handle_message(raw)
                        except Exception as exc:
                            logger.warning(f"handler error: {exc}")
            except (ConnectionClosed, OSError) as exc:
                logger.warning(f"WS disconnect ({base}): {exc} — reconnecting in {backoff:.1f}s")
                consecutive_failures += 1
            except Exception as exc:
                logger.error(f"WS error ({base}): {exc} — reconnecting in {backoff:.1f}s")
                consecutive_failures += 1
            finally:
                if self._stop.is_set():
                    break
                # mark a freeze window so consumers don't quote on stale data
                # immediately after reconnect (Binance can replay or skip ticks)
                self._reconnect_freeze_until = time.time() + 2.0 * 5.0
                # Rotate to the next URL after 3 consecutive failures.
                if consecutive_failures >= 3 and len(self._ws_urls) > 1:
                    self._url_idx = (self._url_idx + 1) % len(self._ws_urls)
                    consecutive_failures = 0
                    logger.warning(f"rotating to fallback Binance URL: {self.ws_url}")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self.reconnect_max)
        logger.info("Binance feed stopped")

    def _handle_message(self, raw) -> None:
        msg = json.loads(raw)
        # Combined-stream messages wrap the payload in {"stream": "...", "data": {...}}.
        data = msg.get("data") or msg
        stream = msg.get("stream", "")
        ev_type = data.get("e") or ""
        symbol = (data.get("s") or "").upper()
        if not symbol or symbol not in self._states:
            return
        st = self._states[symbol]
        now = time.time()

        # bookTicker payload doesn't have "e"; identify by fields.
        if "b" in data and "a" in data and ev_type == "":
            try:
                st.best_bid = float(data["b"])
                st.best_ask = float(data["a"])
                st.best_bid_qty = float(data.get("B", 0.0))
                st.best_ask_qty = float(data.get("A", 0.0))
                st.last_update_ts = now
                st.price_history.append((now, st.mid))
            except (TypeError, ValueError):
                pass
            return

        if ev_type == "aggTrade":
            try:
                price = float(data["p"])
                qty = float(data["q"])
                # `m=True` means the buyer is the maker → it's a SELL aggressor.
                signed = qty if not bool(data.get("m", False)) else -qty
                st.last_trade_price = price
                st.last_trade_qty = qty
                st.last_update_ts = now
                # 5s rolling signed flow with exponential decay (~3s half-life).
                st.trade_flow_5s = 0.7 * st.trade_flow_5s + signed
            except (TypeError, ValueError, KeyError):
                pass
            return
