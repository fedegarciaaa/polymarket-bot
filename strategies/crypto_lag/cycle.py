"""Main async cycle for the crypto-lag strategy.

The cycle is the only place that ties feed + registry + model + executor +
risk together. Each tick:

  1. Read feed snapshots and orderbook from cached state.
  2. For each active market in event window:
     a. Capture strike if not yet known (snap from feed at first sight).
     b. Compute realized vol from rolling history.
     c. Run probability_model.prob_up.
     d. Compare model fair_mid to Polymarket bid/ask, build QuoteDecision.
     e. Reconcile via order_engine.
  3. Poll executor for fills, update risk inventory, log to DB.
  4. Resolve any markets whose endDate passed: settle position with the actual
     Binance close vs strike, log close event.
  5. Sleep `refresh_seconds`.

This task runs forever; cancel via the runner's stop event.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional

from .binance_feed import BinanceFeed
from .order_engine import MakerOrderEngine, QuoteDecision
from .paper_executor import PaperExecutor
from .poly_markets import CryptoMarketRegistry
from .probability_model import (
    ProbInputs,
    prob_up,
    realized_vol_per_sqrt_s,
)
from .risk import CryptoLagRisk
from .state import CryptoLagSnapshot, PolyCryptoMarket

logger = logging.getLogger("polymarket_bot.crypto_lag.cycle")


class CryptoLagCycle:
    def __init__(
        self,
        config: dict,
        feed: BinanceFeed,
        registry: CryptoMarketRegistry,
        executor: PaperExecutor,
        engine: MakerOrderEngine,
        risk: CryptoLagRisk,
        db=None,
        notifier=None,
    ):
        cfg = config.get("crypto_lag", {}) or {}
        self.refresh_seconds = float(cfg.get("refresh_seconds", 3.0))
        self.kelly_fraction = float(cfg.get("kelly_fraction", 0.10))
        self.imbalance_alpha = float(cfg.get("imbalance_alpha", 0.03))
        self.poly_blend_weight = float(cfg.get("poly_blend_weight", 0.10))
        self.flatten_before_resolution = float(
            cfg.get("flatten_before_resolution_seconds", 30.0)
        )
        self.feed = feed
        self.registry = registry
        self.executor = executor
        self.engine = engine
        self.risk = risk
        self.db = db
        self.notifier = notifier
        self._stop = asyncio.Event()
        # Per-market heartbeat rate limit (condition_id → last snapshot ts)
        self._last_heartbeat_ts: dict[str, float] = {}

    async def run_forever(self) -> None:
        logger.info("crypto_lag cycle started")
        try:
            while not self._stop.is_set():
                t0 = time.time()
                try:
                    await self._tick(t0)
                except Exception as exc:
                    logger.exception(f"cycle tick error: {exc}")
                # Sleep until next tick, accounting for tick duration
                elapsed = time.time() - t0
                wait = max(0.1, self.refresh_seconds - elapsed)
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=wait)
                except asyncio.TimeoutError:
                    pass
        finally:
            logger.info("crypto_lag cycle stopping; cancelling all orders")
            try:
                await self.engine.cancel_all()
            except Exception as exc:
                logger.warning(f"cancel_all on shutdown: {exc}")

    def stop(self) -> None:
        self._stop.set()

    # ─── tick body ──────────────────────────────────────────────
    async def _tick(self, now: float) -> None:
        # 0. Global gate
        ok, reason = self.risk.can_quote_globally(now)
        if not ok:
            # Still process fills/resolutions even if we don't open new orders
            await self._process_fills_and_resolutions(now)
            if reason != "ok":
                logger.debug(f"globally-gated: {reason}")
            return

        in_freeze = self.feed.is_in_reconnect_freeze(now)

        # 1. Iterate active markets per symbol
        for sym in self.feed.symbols:
            feed_state = self.feed.get_state(sym)
            if feed_state is None:
                continue
            markets = self.registry.active_for(sym, now)
            if not markets:
                # No market in its event window → emit a per-symbol heartbeat
                # so the dashboard can show binance_mid and "PRE_EVENT" status.
                # Polymarket crypto-updown markets only open during US trading
                # hours; the heartbeat keeps the chart populated outside that window.
                self._symbol_heartbeat(sym, feed_state, now)
                continue
            for market in markets:
                await self._handle_market(market, feed_state, now, in_freeze)

        # 2. Drain fills and resolutions
        await self._process_fills_and_resolutions(now)

    async def _handle_market(
        self, market: PolyCryptoMarket, feed_state, now: float, in_freeze: bool
    ) -> None:
        # 2a. Per-market gate
        ok, reason = self.risk.can_quote_market(
            market.condition_id, feed_state, now, in_reconnect_freeze=in_freeze,
        )
        if not ok:
            logger.debug(f"gated {market.symbol} {market.market_slug[:30]}: {reason}")
            await self.engine.cancel_all_for(market.condition_id)
            self._heartbeat_snapshot(market, feed_state, now, decision="GATED")
            return

        # 2b. Capture strike on first sight (we entered the event window)
        if market.strike_price <= 0:
            self.registry.set_strike(market.condition_id, feed_state.mid)
            logger.info(
                f"strike captured {market.symbol} {market.market_slug[:30]}: {feed_state.mid:.2f}"
            )
            self._heartbeat_snapshot(market, feed_state, now, decision="STRIKE_CAPTURED")
            return  # next tick will quote with strike known

        t_remaining = market.end_ts - now
        if t_remaining <= self.flatten_before_resolution:
            # Stop quoting in the resolution window — let any open orders die
            # (or be filled at face value) instead of replacing.
            await self.engine.cancel_all_for(market.condition_id)
            self._heartbeat_snapshot(market, feed_state, now, decision="RESOLUTION_WINDOW")
            return

        # 2c. Compute realized vol and run model
        sigma = realized_vol_per_sqrt_s(list(feed_state.price_history))
        # Keep sigma in a reasonable band (avoid divide-by-zero when history is
        # short or perfectly flat — assume 1bp/sqrt(s) as a floor)
        sigma = max(sigma, 1e-5)

        # 2d. Pull Polymarket book from the executor's cache (it polls on its own)
        book = await self.executor._get_book(market.token_yes)  # noqa: SLF001
        if not book or book.get("best_bid") is None or book.get("best_ask") is None:
            self._heartbeat_snapshot(market, feed_state, now, decision="NO_BOOK", sigma=sigma)
            return
        poly_bid = float(book["best_bid"])
        poly_ask = float(book["best_ask"])
        poly_mid = 0.5 * (poly_bid + poly_ask)

        inputs = ProbInputs(
            spot_now=feed_state.mid,
            strike=market.strike_price,
            sigma_per_sqrt_s=sigma,
            t_remaining_s=t_remaining,
            book_imbalance=feed_state.book_imbalance,
            trade_flow_5s=feed_state.trade_flow_5s,
            poly_mid=poly_mid,
        )
        out = prob_up(
            inputs,
            imbalance_alpha=self.imbalance_alpha,
            poly_blend_weight=self.poly_blend_weight,
        )
        fair_mid = out.p_blended

        # 2e. Sizing — Kelly on the larger edge
        edge = max(fair_mid - poly_bid, poly_ask - fair_mid)
        kelly_size = self._kelly_size(edge, poly_bid, poly_ask)
        target_size = self.risk.order_size_usdc(kelly_size)
        if target_size <= 0:
            await self.engine.cancel_all_for(market.condition_id)
            # Still record the snapshot so the dashboard can see what happened
            zero_decision = self.engine.build_decision(
                fair_mid=fair_mid, poly_best_bid=poly_bid, poly_best_ask=poly_ask,
                target_size_usdc=0.0, tick=market.tick_size,
            )
            self._maybe_log_snapshot(market, feed_state, sigma, out, poly_bid, poly_ask, zero_decision, now)
            return

        decision = self.engine.build_decision(
            fair_mid=fair_mid,
            poly_best_bid=poly_bid,
            poly_best_ask=poly_ask,
            target_size_usdc=target_size,
            tick=market.tick_size,
        )
        await self.engine.reconcile(market, decision, now)

        # 2f. Snapshot for forensics
        self._maybe_log_snapshot(market, feed_state, sigma, out, poly_bid, poly_ask, decision, now)

    async def _process_fills_and_resolutions(self, now: float) -> None:
        # Fills
        try:
            fills = await self.executor.poll_fills()
        except Exception as exc:
            logger.warning(f"poll_fills error: {exc}")
            fills = []
        for f in fills:
            signed = f.fill_size_usdc if f.side == "BUY" else -f.fill_size_usdc
            self.risk.on_fill(f.condition_id, signed)
            logger.info(
                f"fill {f.symbol} {f.side} {f.outcome} {f.fill_size_usdc:.2f}@{f.fill_price:.4f} "
                f"adverse={f.is_adverse}"
            )
            self._record_trade(f, now)
            if self.notifier is not None:
                try:
                    self.notifier.notify_crypto_lag_fill(
                        symbol=f.symbol, side=f.side, outcome=f.outcome,
                        fill_price=f.fill_price, fill_size_usdc=f.fill_size_usdc,
                        is_adverse=f.is_adverse,
                    )
                except Exception as exc:
                    logger.debug(f"notify_crypto_lag_fill failed: {exc}")

        # Resolutions: any market whose endDate passed → settle
        for cid, pos in list(self.executor._positions.items()):  # noqa: SLF001
            mkt = next(
                (m for m in self._all_known_markets() if m.condition_id == cid),
                None,
            )
            if mkt is None:
                # market dropped off registry (already past end). Use stored end_ts
                # via a per-symbol Binance close. Fall through.
                continue
            if mkt.end_ts > now:
                continue
            yes_value = self._resolve_yes_value(mkt, now)
            ev = await self.executor.resolve_market(cid, yes_value, ts=now)
            if ev is not None:
                self.risk.on_close(ev.realized_pnl_usdc)
                logger.info(
                    f"resolve {ev.symbol} {ev.condition_id[:12]}... pnl={ev.realized_pnl_usdc:+.3f}"
                )
                self._record_close(ev)
                if self.notifier is not None:
                    try:
                        self.notifier.notify_crypto_lag_close(
                            symbol=ev.symbol,
                            realized_pnl_usdc=ev.realized_pnl_usdc,
                            final_yes_price=ev.final_yes_price,
                            reason=ev.reason,
                            market_slug=mkt.market_slug if mkt else "",
                        )
                    except Exception as exc:
                        logger.debug(f"notify_crypto_lag_close failed: {exc}")

        # Drain any lingering close events the executor produced
        for ev in self.executor.drain_close_log():
            # already handled above for ones we triggered; this is a safety drain
            pass

    # ─── small helpers ──────────────────────────────────────────
    def _kelly_size(self, edge: float, bid: float, ask: float) -> float:
        """Fractional Kelly assuming the YES outcome behaves binary at expiry.

        We're approximating: with edge `e` and price `p`, optimal Kelly fraction
        is roughly `e / (p * (1 - p))`. Multiplied by `kelly_fraction` (config).
        Then converted to USDC notional via the risk module's bankroll cap.
        """
        if edge <= 0:
            return 0.0
        p = max(0.01, min(0.99, 0.5 * (bid + ask)))
        f = edge / max(0.001, p * (1.0 - p))
        f = max(0.0, min(0.25, f))   # cap raw kelly at 25% before fractional
        return f * self.kelly_fraction * 1000.0  # in USDC; risk caps further

    def _all_known_markets(self) -> list[PolyCryptoMarket]:
        out: list[PolyCryptoMarket] = []
        for sym in self.feed.symbols:
            out.extend(self.registry.active_for(sym, time.time() - 3600))  # past hour
        return out

    def _resolve_yes_value(self, market: PolyCryptoMarket, now: float) -> float:
        """At resolution: YES wins if Binance close > strike at endDate.

        We use the most recent Binance mid as a proxy for the close price (the
        cycle ticks every few seconds, so we'll sample within seconds of the
        actual close). For a more robust LIVE implementation, snap the close
        from Binance's REST `/klines` endpoint at endDate.
        """
        st = self.feed.get_state(market.symbol)
        if st is None or st.mid <= 0 or market.strike_price <= 0:
            return 0.5  # unknown — settle at midpoint (conservative)
        return 1.0 if st.mid > market.strike_price else 0.0

    # Rate limit heartbeat snapshots to one every 30s per condition_id so we
    # don't fill the DB with thousands of "GATED" rows when many markets are
    # being filtered out.
    _HEARTBEAT_INTERVAL_S: float = 30.0

    def _symbol_heartbeat(self, symbol: str, feed_state, now: float) -> None:
        """Emit a per-symbol heartbeat when there is no market in event window.

        Uses condition_id = `pre_event:<SYMBOL>` so the rate limiter treats
        each symbol independently. The decision is `PRE_EVENT` so the dashboard
        donut shows the bot is alive but waiting for markets to open.
        """
        if self.db is None or not hasattr(self.db, "log_crypto_lag_snapshot"):
            return
        synthetic_id = f"pre_event:{symbol}"
        last = self._last_heartbeat_ts.get(synthetic_id, 0.0)
        if (now - last) < self._HEARTBEAT_INTERVAL_S:
            return
        self._last_heartbeat_ts[synthetic_id] = now
        try:
            snap = CryptoLagSnapshot(
                ts=now, symbol=symbol, binance_mid=feed_state.mid,
                sigma_realized=0.0, book_imbalance=feed_state.book_imbalance,
                p_model=0.5, poly_bid=0.0, poly_ask=0.0, poly_mid=0.0,
                fair_mid=0.5, edge_bid=0.0, edge_ask=0.0,
                decision="PRE_EVENT",
            )
            self.db.log_crypto_lag_snapshot(snap)
        except Exception as exc:
            logger.debug(f"symbol heartbeat error: {exc}")

    def _heartbeat_snapshot(
        self, market, feed_state, now: float, decision: str, sigma: float = 0.0
    ) -> None:
        """Write a minimal snapshot for dashboard heartbeat when we early-return.

        Decision codes used here are non-quote codes:
          GATED              - risk gate blocked (stale feed, max orders, etc)
          STRIKE_CAPTURED    - first time entering event window, strike snapped
          RESOLUTION_WINDOW  - within flatten-before-resolution buffer
          NO_BOOK            - Polymarket orderbook empty / unavailable
        """
        if self.db is None:
            return
        if not hasattr(self.db, "log_crypto_lag_snapshot"):
            return
        last = self._last_heartbeat_ts.get(market.condition_id, 0.0)
        if (now - last) < self._HEARTBEAT_INTERVAL_S:
            return
        self._last_heartbeat_ts[market.condition_id] = now
        try:
            snap = CryptoLagSnapshot(
                ts=now, symbol=market.symbol, binance_mid=feed_state.mid,
                sigma_realized=sigma, book_imbalance=feed_state.book_imbalance,
                p_model=0.5, poly_bid=0.0, poly_ask=0.0, poly_mid=0.0,
                fair_mid=0.5, edge_bid=0.0, edge_ask=0.0,
                decision=decision,
            )
            self.db.log_crypto_lag_snapshot(snap)
        except Exception as exc:
            logger.debug(f"heartbeat snapshot error: {exc}")

    def _maybe_log_snapshot(
        self, market, feed_state, sigma, out, poly_bid, poly_ask, decision, now
    ) -> None:
        # Snapshot logging: keep light unless explicitly enabled.
        if self.db is None:
            return
        try:
            snap = CryptoLagSnapshot(
                ts=now, symbol=market.symbol, binance_mid=feed_state.mid,
                sigma_realized=sigma, book_imbalance=feed_state.book_imbalance,
                p_model=out.p_model, poly_bid=poly_bid, poly_ask=poly_ask,
                poly_mid=0.5 * (poly_bid + poly_ask),
                fair_mid=out.p_blended,
                edge_bid=decision.edge_bid, edge_ask=decision.edge_ask,
                decision=decision.side,
            )
            if hasattr(self.db, "log_crypto_lag_snapshot"):
                self.db.log_crypto_lag_snapshot(snap)
        except Exception:
            pass

    def _record_trade(self, fill, now) -> None:
        if self.db is None or not hasattr(self.db, "log_crypto_lag_fill"):
            return
        try:
            self.db.log_crypto_lag_fill(fill)
        except Exception:
            pass

    def _record_close(self, ev) -> None:
        if self.db is None or not hasattr(self.db, "log_crypto_lag_close"):
            return
        try:
            self.db.log_crypto_lag_close(ev)
        except Exception:
            pass
