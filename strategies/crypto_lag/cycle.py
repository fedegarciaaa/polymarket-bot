"""Main async cycle for the crypto-lag strategy.

The cycle is the only place that ties feed + registry + model + executor +
risk together. Each tick:

  1. Read feed snapshots and orderbook from cached state.
  2. For each active market in event window:
     a. Capture strike if not yet known (snap from feed at first sight).
     b. Compose realized + implied + GARCH vol into a single sigma.
     c. Run probability_model.prob_up.
     d. Compare model fair_mid to Polymarket bid/ask, build a two-sided
        QuoteDecision (Avellaneda-Stoikov + fee-aware + inventory-skew).
     e. Reconcile via order_engine.
  3. Poll executor for fills, update risk inventory, log to DB.
  4. Resolve any markets whose endDate passed: settle position with the actual
     Binance close vs strike, log close event.
  5. Sleep `refresh_seconds`.

This task runs forever; cancel via the runner's stop event.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Optional

from .binance_feed import BinanceFeed
from .garch import Garch11, returns_from_prices
from .order_engine import MakerOrderEngine, QuoteDecision
from .paper_executor import PaperExecutor
from .poly_markets import CryptoMarketRegistry
from .probability_model import (
    ProbInputs,
    blend_volatility,
    prob_up,
    realized_vol_per_sqrt_s,
)
from .risk import CryptoLagRisk
from .state import CryptoLagSnapshot, PolyCryptoMarket

logger = logging.getLogger("polymarket_bot.crypto_lag.cycle")


# Late-window phases (seconds remaining until endDate). Tuned per Substack
# benjamin.bigdev observation that 85% of the resolution direction is decided
# in the last 10 seconds of a 5-min market.
LATE_WINDOW_TIGHTEN_S = 60.0    # T ∈ (30, 60]: tighten spread, allow new quotes
LATE_WINDOW_HOLD_S = 30.0       # T ∈ (10, 30]: don't place new orders, hold existing
LATE_WINDOW_FLAT_S = 10.0       # T ≤ 10: cancel all, flat


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
        deribit_iv=None,    # DeribitIVProvider | None
        variant: str = "main",
        variant_overrides: Optional[dict] = None,
    ):
        # variant identifies this instance in DB / dashboard; e.g. "main" for
        # the strict simulator and "permissive" for the optimistic shadow.
        self.variant = str(variant)
        # `variant_overrides` lets the runner inject per-variant params (e.g.
        # edge_threshold, queue_position_enabled, ...) without mutating the
        # shared `config` dict between siblings. Falls back to the legacy
        # `crypto_lag` block when None.
        cfg_global = config.get("crypto_lag", {}) or {}
        cfg = dict(cfg_global)
        if variant_overrides:
            cfg.update(variant_overrides)
        self.refresh_seconds = float(cfg.get("refresh_seconds", 3.0))
        self.kelly_fraction = float(cfg.get("kelly_fraction", 0.10))
        self.imbalance_alpha = float(cfg.get("imbalance_alpha", 0.03))
        self.poly_blend_weight = float(cfg.get("poly_blend_weight", 0.10))
        self.poly_obi_alpha = float(cfg.get("poly_obi_alpha", 0.02))
        self.flatten_before_resolution = float(
            cfg.get("flatten_before_resolution_seconds", LATE_WINDOW_FLAT_S)
        )
        self.max_book_spread = float(cfg.get("max_book_spread", 0.80))
        self.use_two_sided = bool(cfg.get("two_sided_quoting", True))

        # Volatility blend weights (realized vs IV vs GARCH). Defaults follow
        # the F1.1 plan: 0.5 realized, 0.3 IV, 0.2 GARCH.
        vbw = cfg.get("vol_blend_weights") or {}
        self.vol_weights = {
            "realized": float(vbw.get("realized", 0.5)),
            "iv": float(vbw.get("iv", 0.3)),
            "garch": float(vbw.get("garch", 0.2)),
        }
        # Whether to use EWMA on the rolling realized vol (recommended).
        self.realized_mode = str(cfg.get("realized_vol_mode", "ewma"))
        self.ewma_lambda = float(cfg.get("ewma_lambda", 0.94))

        self.feed = feed
        self.registry = registry
        self.executor = executor
        self.engine = engine
        self.risk = risk
        self.db = db
        self.notifier = notifier
        self.deribit_iv = deribit_iv
        self._stop = asyncio.Event()
        # Per-market heartbeat rate limit (condition_id → last snapshot ts)
        self._last_heartbeat_ts: dict[str, float] = {}
        # Per-symbol GARCH(1,1) instances. Lazily initialized on first use so
        # we don't pay the warm-up cost for symbols with no active markets.
        self._garch: dict[str, Garch11] = {}
        # Last hour-bucket we refit GARCH on, so we refit at most once per hour.
        self._garch_last_refit_hour: dict[str, int] = {}

        # Hourly variant stats — see _emit_stats_if_due. We emit one
        # CRYPTO_LAG_VARIANT_STATS event per variant per hour into events.jsonl
        # so the analyses can show rolling fill-rate / pnl without reloading
        # 2.6M state-snapshot rows.
        self._stats_period_start: float = time.time()
        self._fills_period: int = 0
        self._gross_pnl_period: float = 0.0
        self._fees_period: float = 0.0
        self._rebates_period: float = 0.0
        self._closes_period: int = 0

    async def run_forever(self) -> None:
        logger.info("crypto_lag cycle started")
        try:
            while not self._stop.is_set():
                t0 = time.time()
                try:
                    await self._tick(t0)
                except Exception as exc:
                    logger.exception(f"cycle tick error: {exc}")
                # Emit hourly variant stats (cheap; checked once per tick).
                try:
                    self._emit_stats_if_due(t0)
                except Exception as exc:
                    logger.debug(f"stats emit error: {exc}")
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
            # GARCH housekeeping (refit at most once per hour per symbol)
            self._refit_garch_if_due(sym, now)
            markets = self.registry.active_for(sym, now)
            if not markets:
                # No market in its event window → emit a per-symbol heartbeat
                # so the dashboard can show binance_mid and "PRE_EVENT" status.
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
            # Cancel any previously-placed orders on this cid so the
            # `risk.open_orders_count` doesn't leak while we wait for the next
            # tick to (re)quote with the freshly-captured strike.
            await self.engine.cancel_all_for(market.condition_id)
            self._heartbeat_snapshot(market, feed_state, now, decision="STRIKE_CAPTURED")
            return  # next tick will quote with strike known

        t_remaining = market.end_ts - now

        # 2b.1 Late-window policy: cancel everything in the flat zone.
        if t_remaining <= self.flatten_before_resolution:
            await self.engine.cancel_all_for(market.condition_id)
            self._heartbeat_snapshot(
                market, feed_state, now,
                decision="RESOLUTION_WINDOW",
            )
            return

        # 2c. Compose realized + IV + GARCH into one σ-per-√s
        sigma = self._compose_sigma(market.symbol, feed_state, now)

        # 2d. Pull Polymarket book from the executor's cache (it polls on its own)
        book = await self.executor._get_book(market.token_yes)  # noqa: SLF001
        if not book or book.get("best_bid") is None or book.get("best_ask") is None:
            # Cancel any orders that might be resting on this cid so the
            # max_concurrent_orders counter doesn't leak when the book vanishes.
            await self.engine.cancel_all_for(market.condition_id)
            self._heartbeat_snapshot(market, feed_state, now, decision="NO_BOOK", sigma=sigma)
            return
        poly_bid = float(book["best_bid"])
        poly_ask = float(book["best_ask"])
        poly_mid = 0.5 * (poly_bid + poly_ask)

        # Gate: skip markets with no real liquidity (a wide spread means the
        # book is essentially empty — quoting at 0.02/0.98 will never fill).
        if poly_ask - poly_bid > self.max_book_spread:
            await self.engine.cancel_all_for(market.condition_id)
            self._heartbeat_snapshot(market, feed_state, now, decision="NO_BOOK", sigma=sigma)
            return

        # 2e. Polymarket-side OBI (own-book imbalance, separate from Binance).
        poly_obi = _book_imbalance(book.get("bid_size", 0.0), book.get("ask_size", 0.0))

        inputs = ProbInputs(
            spot_now=feed_state.mid,
            strike=market.strike_price,
            sigma_per_sqrt_s=sigma,
            t_remaining_s=t_remaining,
            book_imbalance=feed_state.book_imbalance,
            trade_flow_5s=feed_state.trade_flow_5s,
            poly_mid=poly_mid,
            poly_book_imbalance=poly_obi,
        )
        out = prob_up(
            inputs,
            imbalance_alpha=self.imbalance_alpha,
            poly_blend_weight=self.poly_blend_weight,
            poly_obi_alpha=self.poly_obi_alpha,
        )
        fair_mid = out.p_blended

        # 2f. Sizing — Kelly on the larger gross edge (rebate is added inside
        # the engine's net-edge check, so the *threshold* for sizing here is
        # generous on purpose).
        edge = max(fair_mid - poly_bid, poly_ask - fair_mid)
        kelly_size = self._kelly_size(edge, poly_bid, poly_ask)
        target_size = self.risk.order_size_usdc(kelly_size)

        # 2g. Late-window phases — narrow the action space as we approach
        # resolution. We never place NEW orders inside HOLD; existing orders
        # ride out as the model still cancels-and-replaces on drift.
        in_hold_phase = t_remaining <= LATE_WINDOW_HOLD_S
        in_tighten_phase = t_remaining <= LATE_WINDOW_TIGHTEN_S

        if target_size <= 0 or in_hold_phase:
            # Don't place new — but DO let the engine cancel anything stale.
            zero_decision = QuoteDecision(
                side="NONE", fair_mid=fair_mid,
                edge_bid=fair_mid - poly_bid, edge_ask=poly_ask - fair_mid,
                poly_best_bid=poly_bid, poly_best_ask=poly_ask,
                target_size_usdc=0.0,
            )
            await self.engine.reconcile(market, zero_decision, now)
            self._maybe_log_snapshot(
                market, feed_state, sigma, out, poly_bid, poly_ask, zero_decision, now,
                phase="HOLD" if in_hold_phase else "NO_SIZE",
            )
            return

        # Inventory: positive USDC = net long YES, negative = net short.
        inventory_usdc = self.risk.state.inventory_by_market.get(market.condition_id, 0.0)

        if self.use_two_sided:
            decision = self.engine.build_decision_two_sided(
                fair_mid=fair_mid,
                poly_best_bid=poly_bid,
                poly_best_ask=poly_ask,
                target_size_usdc=target_size,
                tick=market.tick_size,
                sigma_per_sqrt_s=sigma,
                t_remaining_s=t_remaining,
                inventory_usdc=inventory_usdc,
                per_market_max_inventory_usdc=self.risk.per_market_max_inventory_usdc,
            )
        else:
            decision = self.engine.build_decision(
                fair_mid=fair_mid,
                poly_best_bid=poly_bid,
                poly_best_ask=poly_ask,
                target_size_usdc=target_size,
                tick=market.tick_size,
            )

        # In tighten phase: shrink size to half so we don't get caught with
        # large quotes during the high-toxicity final minute.
        if in_tighten_phase and decision.side != "NONE":
            decision.target_size_usdc = decision.target_size_usdc * 0.5
            if decision.bid_size_usdc is not None:
                decision.bid_size_usdc = decision.bid_size_usdc * 0.5
            if decision.ask_size_usdc is not None:
                decision.ask_size_usdc = decision.ask_size_usdc * 0.5

        await self.engine.reconcile(market, decision, now)

        # 2h. Snapshot for forensics
        self._maybe_log_snapshot(
            market, feed_state, sigma, out, poly_bid, poly_ask, decision, now,
            phase="TIGHTEN" if in_tighten_phase else "NORMAL",
        )

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
            self._fills_period += 1
            self._fees_period += float(getattr(f, "fee_paid_usdc", 0.0) or 0.0)
            self._rebates_period += float(getattr(f, "rebate_usdc", 0.0) or 0.0)
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

        # Resolutions: any market whose endDate passed → settle.
        # We iterate `executor._positions` and use the end_ts/strike/symbol
        # snapshotted on the Position itself (set when the position was first
        # opened). This is the ONLY reliable source: Polymarket drops markets
        # off the Gamma `active` list within seconds of endDate, so a
        # registry-based lookup would silently miss every resolution.
        for cid, pos in list(self.executor._positions.items()):  # noqa: SLF001
            # Prefer the snapshot stored on the Position; fall back to the
            # registry only for legacy positions opened before this code
            # landed (no end_ts set).
            end_ts = float(getattr(pos, "end_ts", 0.0) or 0.0)
            strike = float(getattr(pos, "strike_price", 0.0) or 0.0)
            symbol = pos.symbol
            slug = str(getattr(pos, "market_slug", "") or "")
            if end_ts <= 0.0 or strike <= 0.0:
                mkt = next(
                    (m for m in self._all_known_markets() if m.condition_id == cid),
                    None,
                )
                if mkt is None:
                    continue
                end_ts = mkt.end_ts
                strike = mkt.strike_price
                symbol = mkt.symbol
                slug = mkt.market_slug
            if end_ts > now:
                continue
            # YES wins on tie (Polymarket "Up or Down" convention). Settle off
            # the most recent Binance mid we have for this symbol.
            st = self.feed.get_state(symbol)
            if st is None or st.mid <= 0 or strike <= 0:
                yes_value = 0.5  # unknown — settle at midpoint (conservative)
            else:
                yes_value = 1.0 if st.mid >= strike else 0.0
            ev = await self.executor.resolve_market(cid, yes_value, ts=now)
            if ev is not None:
                self.risk.on_close(ev.realized_pnl_usdc)
                logger.info(
                    f"resolve {ev.symbol} {ev.condition_id[:12]}... pnl={ev.realized_pnl_usdc:+.3f}"
                )
                self._gross_pnl_period += float(ev.realized_pnl_usdc)
                self._closes_period += 1
                self._record_close(ev)
                if self.notifier is not None:
                    try:
                        self.notifier.notify_crypto_lag_close(
                            symbol=ev.symbol,
                            realized_pnl_usdc=ev.realized_pnl_usdc,
                            final_yes_price=ev.final_yes_price,
                            reason=ev.reason,
                            market_slug=slug,
                        )
                    except Exception as exc:
                        logger.debug(f"notify_crypto_lag_close failed: {exc}")

        # Drain any lingering close events the executor produced
        for ev in self.executor.drain_close_log():
            # already handled above for ones we triggered; this is a safety drain
            pass

        # Counter leak defense: cancel orders for any cid whose market has
        # ended (or is gone from the registry). _handle_market only runs for
        # markets active_for() returns, so dead-cid orders would otherwise
        # accumulate forever in engine._open and drive `open_orders_count`
        # above max_concurrent_orders, gating the bot globally. See
        # https://github.com/.../bot-prod-1 30h diagnostic for context.
        live_cids: set[str] = set()
        for sym in self.feed.symbols:
            for m in self.registry.active_for(sym, now):
                live_cids.add(m.condition_id)
        for cid in list(self.engine._open.keys()):  # noqa: SLF001
            if cid in live_cids:
                continue
            try:
                await self.engine.cancel_all_for(cid)
            except Exception as exc:
                logger.debug(f"orphan cancel {cid[:12]}...: {exc}")

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
        """At resolution: YES wins if Binance close >= strike at endDate.

        Polymarket convention: end ≥ start counts as UP (YES wins ties). We
        use the most recent Binance mid as a proxy for the close price (the
        cycle ticks every few seconds, so we'll sample within seconds of the
        actual close). For a more robust LIVE implementation, snap the close
        from Binance's REST `/klines` endpoint at endDate, or — better —
        from Chainlink Data Streams which is the actual oracle Polymarket
        uses (see plan F0.3).
        """
        st = self.feed.get_state(market.symbol)
        if st is None or st.mid <= 0 or market.strike_price <= 0:
            return 0.5  # unknown — settle at midpoint (conservative)
        # Tie convention matches Polymarket "Up or Down": end ≥ start = UP
        return 1.0 if st.mid >= market.strike_price else 0.0

    def _compose_sigma(self, symbol: str, feed_state, now: float) -> float:
        """Compose the rolling realized vol (EWMA), the bootstrapped historical
        klines vol, the Deribit IV (if available) and the GARCH(1,1) one-step
        forecast into a single σ-per-√second.

        For the *realized* leg we mirror the original n_rt blend so we don't
        over-weight a near-empty rolling history; for the *IV* and *GARCH*
        legs we just feed them in if present and let `blend_volatility`
        redistribute weights when one is missing.
        """
        rt_history = list(feed_state.price_history)
        rt_sigma = realized_vol_per_sqrt_s(
            rt_history,
            mode=self.realized_mode,
            ewma_lambda=self.ewma_lambda,
        )
        hist_sigma = self.feed.get_hist_sigma(symbol)
        n_rt = len(rt_history)
        if n_rt < 60:
            realized_leg = hist_sigma
        elif n_rt < 300:
            alpha = (n_rt - 60) / (300 - 60)
            realized_leg = (1.0 - alpha) * hist_sigma + alpha * rt_sigma
        else:
            realized_leg = rt_sigma
        realized_leg = max(realized_leg, 1e-5)

        # Deribit IV (annualized → per-√second already done in the provider).
        iv_leg: Optional[float] = None
        if self.deribit_iv is not None:
            try:
                iv_val = self.deribit_iv.get_sigma_per_sqrt_s(symbol)
                if iv_val is not None and iv_val > 0:
                    iv_leg = float(iv_val)
            except Exception:
                iv_leg = None

        # GARCH(1,1): the model is fit on per-bar (1m) returns, so its sigma
        # is per √minute. We convert to per-√second here.
        garch_leg: Optional[float] = None
        g = self._garch.get(symbol)
        if g is not None and g.fitted:
            try:
                sigma_per_sqrt_minute = g.sigma()
                if sigma_per_sqrt_minute > 0:
                    # 1 minute = 60 seconds → divide by √60 to go per-√second
                    garch_leg = float(sigma_per_sqrt_minute / math.sqrt(60.0))
            except Exception:
                garch_leg = None

        sigma = blend_volatility(
            realized=realized_leg,
            iv=iv_leg,
            garch=garch_leg,
            weights=self.vol_weights,
        )
        return max(sigma, 1e-5)

    def _refit_garch_if_due(self, symbol: str, now: float) -> None:
        """Initialize / refit the per-symbol GARCH(1,1) once per hour using
        proper 1-minute bars resampled from the feed's tick history.

        BUG FIX (May 2026): the previous implementation fed RAW ticks (sub-1s
        spaced) directly to `returns_from_prices`. Tick-to-tick log-returns on
        a $50k BTC barely moving $1/sec are O(1e-5), squared O(1e-10), which
        never moves the conditional variance off the GARCH long-run baseline
        ω/(1-β) ≈ 6.67e-7 → σ ≈ 8.16e-4. As a result every symbol converged
        to the *same* long-run σ regardless of the actual price action,
        breaking the IV-vs-realized-vs-GARCH blend for BNB/XRP/DOGE.

        Resampling to 1-minute bars (last price per minute bucket) recovers
        the volatility scale the GARCH was designed for. With deque maxlen
        of 600 ticks at ~1 tick/sec this typically yields 5-10 bars — short
        but enough to differentiate symbols; the recursion adapts as more
        data accumulates and the hourly refit refreshes the seed.
        """
        if symbol not in self.feed.symbols:
            return
        hour = int(now // 3600)
        last_hour = self._garch_last_refit_hour.get(symbol, -1)
        if hour == last_hour and symbol in self._garch:
            return
        st = self.feed.get_state(symbol)
        if st is None or len(st.price_history) < 30:
            return
        # Resample tick history to 1-minute bars: keep the LAST price seen in
        # each integer-minute bucket (UTC). Iterating the deque is ordered;
        # we trust the per-tick `ts` is monotonically non-decreasing.
        last_per_minute: dict[int, float] = {}
        for ts, p in st.price_history:
            try:
                if p is None or float(p) <= 0:
                    continue
                bucket = int(float(ts) // 60)
                last_per_minute[bucket] = float(p)
            except (TypeError, ValueError):
                continue
        # Sort by minute bucket so returns are computed in chronological order.
        bars = [last_per_minute[k] for k in sorted(last_per_minute.keys())]
        if len(bars) < 3:
            return
        rets = returns_from_prices(bars)
        if len(rets) < 2:
            return
        try:
            g = self._garch.get(symbol) or Garch11()
            g.fit(rets)
            self._garch[symbol] = g
            self._garch_last_refit_hour[symbol] = hour
            logger.info(
                f"GARCH({symbol}): refit on {len(rets)} 1m returns → "
                f"σ_next={g.sigma():.2e}/√bar (lr={g.long_run_sigma():.2e})"
            )
        except Exception as exc:
            logger.debug(f"GARCH({symbol}) refit failed: {exc}")

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
            self._safe_log_snapshot(snap)
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
            self._safe_log_snapshot(snap)
        except Exception as exc:
            logger.debug(f"heartbeat snapshot error: {exc}")

    def _maybe_log_snapshot(
        self, market, feed_state, sigma, out, poly_bid, poly_ask, decision, now,
        phase: str = "NORMAL",
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
                self._safe_log_snapshot(snap)
        except Exception:
            pass

    def _safe_log_snapshot(self, snap) -> None:
        """Log a snapshot tagged with this cycle's variant. Falls back to
        the variant-less call if the DB is on the older schema/method
        signature so an out-of-date deployment doesn't crash the cycle."""
        try:
            self.db.log_crypto_lag_snapshot(snap, variant=self.variant)
        except TypeError:
            self.db.log_crypto_lag_snapshot(snap)

    def _record_trade(self, fill, now) -> None:
        if self.db is None or not hasattr(self.db, "log_crypto_lag_fill"):
            return
        try:
            # Older Database revisions don't accept the variant kwarg; fall
            # back gracefully so we never crash the cycle on a logging call.
            try:
                self.db.log_crypto_lag_fill(fill, variant=self.variant)
            except TypeError:
                self.db.log_crypto_lag_fill(fill)
        except Exception:
            pass

    def _record_close(self, ev) -> None:
        if self.db is None or not hasattr(self.db, "log_crypto_lag_close"):
            return
        try:
            try:
                self.db.log_crypto_lag_close(ev, variant=self.variant)
            except TypeError:
                self.db.log_crypto_lag_close(ev)
        except Exception:
            pass

    # Period over which we accumulate variant stats before emitting the
    # CRYPTO_LAG_VARIANT_STATS event. 1h matches the GARCH refit cadence and
    # gives ~24 data points per day for offline analysis.
    _STATS_PERIOD_S: float = 3600.0

    def _emit_stats_if_due(self, now: float) -> None:
        """Once per `_STATS_PERIOD_S`, write a per-variant rollup to events.jsonl
        and reset the per-period counters. Cheap (<1 row write per hour per
        variant). The structured logger may not be importable in some test
        configurations; failures are swallowed to never fault the cycle."""
        elapsed = now - self._stats_period_start
        if elapsed < self._STATS_PERIOD_S:
            return
        # Snapshot before resetting so we don't race a fill during the write.
        placements = int(getattr(self.engine, "placements_period", 0) or 0)
        placements_taker = int(getattr(self.engine, "placements_period_taker", 0) or 0)
        fills = self._fills_period
        closes = self._closes_period
        gross_pnl = round(self._gross_pnl_period, 4)
        fees = round(self._fees_period, 4)
        rebates = round(self._rebates_period, 4)
        net_pnl = round(gross_pnl + rebates - fees, 4)
        fill_rate = round(fills / placements, 4) if placements > 0 else 0.0

        try:
            from structured_logger import get_logger
            slog = get_logger()
            slog.log(
                "CRYPTO_LAG_VARIANT_STATS",
                f"crypto_lag.{self.variant}",
                {
                    "variant": self.variant,
                    "quote_mode": getattr(self.engine, "quote_mode", "unknown"),
                    "period_seconds": round(elapsed, 1),
                    "placements": placements,
                    "placements_taker": placements_taker,
                    "fills": fills,
                    "closes": closes,
                    "fill_rate": fill_rate,
                    "gross_pnl_usdc": gross_pnl,
                    "fees_paid_usdc": fees,
                    "rebates_usdc": rebates,
                    "net_pnl_usdc": net_pnl,
                },
            )
        except Exception as exc:
            logger.debug(f"variant stats event write failed: {exc}")

        # Reset both sides of the counters so the next period is clean.
        self._stats_period_start = now
        self._fills_period = 0
        self._closes_period = 0
        self._gross_pnl_period = 0.0
        self._fees_period = 0.0
        self._rebates_period = 0.0
        try:
            self.engine.reset_period_counters()
        except Exception:
            pass


def _book_imbalance(bid_size: float, ask_size: float) -> float:
    """Top-of-book imbalance ∈ [-1, 1]. Used for Polymarket OBI.
    Returns 0 when the book is empty / both zero."""
    try:
        b = float(bid_size or 0.0)
        a = float(ask_size or 0.0)
    except (TypeError, ValueError):
        return 0.0
    denom = b + a
    if denom <= 0:
        return 0.0
    return (b - a) / denom
