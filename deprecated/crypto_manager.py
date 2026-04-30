"""
Crypto Manager - Orchestrates the full crypto trading cycle.
Runs independently from the Polymarket cycle on a 60-second interval.

v3 — Regime-Adaptive System:
- Market regime detection (ADX) via CryptoSignalEngine
- Multi-timeframe: fetches 5m + 15m candles; passes 15m RSI as filter
- Heat check: pauses all trading for 1h after 3 consecutive losses
- Entry regime tracked per position; passed to analyze_exit()
- ATR-based dynamic TP passed to place_buy_order()
"""

import logging
import time
import threading
from datetime import datetime, timezone
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from structured_logger import get_logger as get_structured_logger
from strategies.crypto_signals import compute_rsi

logger = logging.getLogger("polymarket_bot.crypto.manager")


class CryptoManager:
    def __init__(self, config: dict, trader, signal_engine, database=None, notifications=None):
        """
        Args:
            config: Full bot config dict
            trader: CryptoTrader instance
            signal_engine: CryptoSignalEngine instance
            database: Database instance (optional, for logging trades)
            notifications: NotificationManager instance (optional)
        """
        self.config = config
        self.trader = trader
        self.signal_engine = signal_engine
        self.db = database
        self.notifications = notifications

        crypto_cfg = config.get("crypto", {})
        self.pairs = crypto_cfg.get("pairs", ["BTCUSDT", "ETHUSDT"])
        self.max_concurrent_trades = crypto_cfg.get("risk", {}).get("max_concurrent_trades", 4)
        self.claude_review_every = crypto_cfg.get("claude_review_every_n_cycles", 30)

        self._cycle_count = 0
        self._market_bias = "NEUTRAL"
        self._total_pnl = 0.0
        self._trades_today = 0
        self.slog = get_structured_logger()

        # ── Heat check: pause after consecutive losses ──
        risk_cfg = config.get("crypto", {}).get("risk", {})
        self._heat_max_losses = risk_cfg.get("heat_check_consecutive_losses", 3)
        self._heat_pause_minutes = risk_cfg.get("heat_check_pause_minutes", 60)
        self._consecutive_losses = 0
        self._heat_pause_until = 0.0  # timestamp

        # ── Entry regime per position (for analyze_exit) ──
        self._position_entry_regime: dict = {}  # symbol -> regime string

        # ── Anti-double-instance lock ──
        self._is_running = False
        self._run_lock = threading.Lock()

        logger.info(
            f"CryptoManager initialized: pairs={self.pairs} "
            f"max_trades={self.max_concurrent_trades} "
            f"mode={self.trader.mode}"
        )

        # Restore OPEN DEMO trades on startup so they hit SL/TP and close properly
        if self.trader.mode == "DEMO" and self.db:
            try:
                cursor = self.db.conn.cursor()
                cursor.execute("SELECT * FROM trades WHERE module = 'crypto' AND status = 'OPEN'")
                open_trades = cursor.fetchall()
                for t in open_trades:
                    symbol = t["market_id"]
                    sl_pct = config.get("crypto", {}).get("risk", {}).get("stop_loss_pct", 0.008)
                    tp_pct = config.get("crypto", {}).get("risk", {}).get("take_profit_pct", 0.015)
                    entry_p = float(t["price_entry"])
                    self.trader._demo_positions[symbol] = {
                        "order_id": f"DEMO-{t['id']}",
                        "symbol": symbol,
                        "quantity": float(t["size_usdc"]) / entry_p,
                        "entry_price": entry_p,
                        "stop_loss": round(entry_p * (1 - sl_pct), 8),
                        "take_profit": round(entry_p * (1 + tp_pct), 8),
                        "usdt_invested": float(t["size_usdc"]),
                        "timestamp": t["timestamp"],
                        "entry_ts": time.time() - 60,  # Aproximación: llevan al menos 1 min
                        "status": "OPEN",
                    }
                if open_trades:
                    logger.info(f"Restored {len(open_trades)} OPEN demo trades from DB to RAM.")
            except Exception as e:
                logger.error(f"Failed to restore open demo trades: {e}")

    def run_cycle(self, claude_agent=None) -> dict:
        """
        Execute one crypto trading cycle:
        1. Anti-double-instance guard
        2. Tick cooldowns (decrement per pair)
        3. Check & close existing positions (SL/TP/time/signal)
        4. Get Claude bias update (every N cycles)
        5. Analyze signals for each pair
        6. Close open positions with active SELL signal
        7. Execute BUY signals sorted by strength (best first)

        Returns summary dict.
        """
        # ── Guard: evitar ejecuciones solapadas ──
        if not self._run_lock.acquire(blocking=False):
            logger.debug("Ciclo ya en ejecución, saltando este tick del scheduler")
            return {"cycle": self._cycle_count, "skipped": True}

        try:
            return self._run_cycle_impl(claude_agent)
        finally:
            self._run_lock.release()

    def _run_cycle_impl(self, claude_agent=None) -> dict:
        """Implementación real del ciclo (llamada dentro del lock)."""
        self._cycle_count += 1
        cycle_start = datetime.now(timezone.utc)
        summary = {
            "cycle": self._cycle_count,
            "timestamp": cycle_start.isoformat(),
            "closed_positions": [],
            "signals": {},
            "new_trades": [],
            "market_bias": self._market_bias,
            "capital_usdt": self.trader.get_available_capital(),
            "open_positions": len(self.trader.get_open_positions()),
        }

        if os.path.exists("data/bot_paused.lock"):
            if self._cycle_count % 10 == 0:
                logger.info("Bot is PAUSED via dashboard. Skipping cycle executions.")
            return summary

        if self._cycle_count % 10 == 0 or summary["open_positions"] > 0:
            logger.info(
                f"CRYPTO CYCLE #{self._cycle_count} | "
                f"Capital: ${summary['capital_usdt']:.2f} | "
                f"Open: {summary['open_positions']} | "
                f"P&L: ${self._total_pnl:+.2f}"
            )

        # ── Step 1: Tick cooldowns (una vez por ciclo) ──
        self.trader.tick_cooldowns()
        self.trader.tick_position_ages()

        # ── Step 2: Check stop-losses, take-profits y max_hold ──
        closed = self.trader.check_and_close_positions()
        for c in closed:
            pnl = c.get("pnl", 0)
            self._total_pnl += pnl
            summary["closed_positions"].append(c)

            # Heat check: track consecutive losses
            if pnl < 0:
                self._consecutive_losses += 1
                if self._consecutive_losses >= self._heat_max_losses:
                    pause_secs = self._heat_pause_minutes * 60
                    self._heat_pause_until = time.time() + pause_secs
                    logger.warning(
                        f"HEAT CHECK triggered: {self._consecutive_losses} consecutive losses "
                        f"→ pausing all new trades for {self._heat_pause_minutes} minutes"
                    )
                    self._consecutive_losses = 0  # Reset after triggering
            else:
                self._consecutive_losses = 0  # Reset on win

            # Clean up regime tracking for closed position
            self._position_entry_regime.pop(c.get("symbol", ""), None)
            logger.info(
                f"Position closed: {c.get('symbol')} "
                f"@ {c.get('price', '?')} | "
                f"P&L: ${pnl:+.2f} | Reason: {c.get('reason', '?')}"
            )
            entry_p = c.get("entry_price", 0) or 0
            exit_p = c.get("price", 0) or 0
            self.slog.log_trade_close(
                "crypto", c.get("trade_id", self._cycle_count),
                c.get("symbol", ""), "rsi_bb_mean_reversion", "BUY",
                entry_p, exit_p, pnl,
                (exit_p - entry_p) / entry_p if entry_p else 0,
                c.get("reason", ""),
            )
            if self.notifications:
                try:
                    self.notifications.send(
                        f"🔄 CRYPTO CLOSED: {c.get('symbol')} | "
                        f"P&L: ${pnl:+.2f} | {c.get('reason', '')}"
                    )
                except Exception:
                    pass
            if self.db:
                try:
                    self.db.log_crypto_trade_close(c)
                except Exception:
                    pass

        # ── Step 3: Claude bias update (every N cycles) ──
        if claude_agent and self._cycle_count % self.claude_review_every == 0:
            try:
                self._update_market_bias(claude_agent)
                summary["market_bias"] = self._market_bias
            except Exception as e:
                logger.warning(f"Claude bias update failed: {e}")

        # ── Step 4: Analizar señales de todos los pares ──
        open_positions = self.trader.get_open_positions()
        all_signals = {}  # pair -> (signal, df_5m)

        for pair in self.pairs:
            # Primary: 5m candles
            df = self.trader.get_klines(pair)
            if df is None:
                logger.warning(f"No data for {pair}")
                continue

            # Secondary: 15m candles for RSI filter (higher-timeframe context)
            rsi_15m = None
            try:
                df_15m = self.trader.get_klines(pair, interval="15m", limit=100)
                if df_15m is not None and len(df_15m) >= 20:
                    rsi_15m_series = compute_rsi(df_15m["close"], 14)
                    rsi_15m = float(rsi_15m_series.iloc[-1])
            except Exception as e:
                logger.debug(f"15m RSI fetch failed for {pair}: {e}")

            signal = self.signal_engine.analyze(df, pair, rsi_15m=rsi_15m)
            all_signals[pair] = (signal, df)
            summary["signals"][pair] = signal

        # ── Step 5: Cerrar posiciones abiertas con señal SELL activa ──
        # NUEVO: si hay posición y la señal dice SELL (técnica) o analyze_exit() lo recomienda
        open_positions = self.trader.get_open_positions()
        for pair, (signal, df) in list(all_signals.items()):
            if pair not in open_positions:
                continue

            pos = open_positions[pair]
            age_cycles = self.trader.get_position_age_cycles(pair)

            # 5a: Señal SELL del motor de señales (RSI>65 o precio>=BB_middle)
            if signal["signal"] == "SELL":
                # Sólo cerramos si llevamos al menos 2 ciclos abiertos (evitar cierres prematuros)
                if age_cycles >= 2:
                    result = self.trader.close_position_by_signal(pair, reason="sell_signal")
                    if result:
                        pnl = result.get("pnl", 0)
                        self._total_pnl += pnl
                        summary["closed_positions"].append(result)
                        self._position_entry_regime.pop(pair, None)
                        logger.info(
                            f"SELL SIGNAL EXIT: {pair} @ {result.get('price')} "
                            f"| P&L: ${pnl:+.2f} | RSI: {signal.get('rsi')}"
                        )
                        if self.db:
                            try:
                                self.db.log_crypto_trade_close(result)
                            except Exception:
                                pass
                        # Refrescar posiciones abiertas
                        open_positions = self.trader.get_open_positions()
                        continue  # No evaluar más este par

            # 5b: Señal de salida técnica avanzada (analyze_exit)
            if pair in open_positions:  # puede haber sido cerrado en 5a
                entry_regime = self._position_entry_regime.get(pair, "RANGING")
                exit_analysis = self.signal_engine.analyze_exit(
                    df, pair,
                    entry_price=pos["entry_price"],
                    age_cycles=age_cycles,
                    entry_regime=entry_regime,
                )
                if exit_analysis["should_exit"] and exit_analysis["urgency"] in ("HIGH", "MEDIUM"):
                    if age_cycles >= 2:  # misma protección anti-cierre prematuro
                        result = self.trader.close_position_by_signal(
                            pair, reason=f"technical_exit({exit_analysis['urgency']}): {exit_analysis['reason']}"
                        )
                        if result:
                            pnl = result.get("pnl", 0)
                            self._total_pnl += pnl
                            summary["closed_positions"].append(result)
                            self._position_entry_regime.pop(pair, None)
                            logger.info(
                                f"TECHNICAL EXIT ({exit_analysis['urgency']}): {pair} "
                                f"@ {result.get('price')} | P&L: ${pnl:+.2f} "
                                f"| Reason: {exit_analysis['reason']}"
                            )
                            if self.db:
                                try:
                                    self.db.log_crypto_trade_close(result)
                                except Exception:
                                    pass
                            open_positions = self.trader.get_open_positions()

        # ── Step 6: Ejecutar señales BUY (priorizadas por signal_strength) ──
        # NUEVO: ordenamos por fuerza de señal descendente para entrar primero en las mejores
        open_positions = self.trader.get_open_positions()
        buy_candidates = []

        # Heat check: block new BUYs if paused
        heat_paused = time.time() < self._heat_pause_until
        if heat_paused:
            remaining_min = (self._heat_pause_until - time.time()) / 60
            if self._cycle_count % 5 == 0:
                logger.info(f"HEAT CHECK pause active — {remaining_min:.0f} min remaining, no new trades")

        for pair, (signal, df) in all_signals.items():
            skip_reason = None

            if signal["signal"] != "BUY":
                skip_reason = f"signal={signal['signal']}"
            elif heat_paused:
                skip_reason = "heat_check_pause"
            elif pair in open_positions:
                skip_reason = "already_open"
            elif self.trader.is_on_cooldown(pair):
                skip_reason = f"cooldown_after_loss"
            elif len(open_positions) >= self.max_concurrent_trades:
                skip_reason = f"max_concurrent={self.max_concurrent_trades}"
            elif signal.get("signal_strength", 0) < 0.75:
                skip_reason = f"weak_signal({signal.get('signal_strength', 0):.2f}<0.75)"
            elif self._market_bias == "BEARISH":
                skip_reason = "bias=BEARISH"
                logger.info(f"SKIP {pair}: market bias is BEARISH, no new longs")

            self.slog.log_crypto_signal(
                pair, signal["signal"], signal.get("price") or 0,
                signal.get("rsi"), signal.get("bb_lower"), signal.get("bb_middle"),
                signal.get("bb_upper"), signal.get("volume_ratio"),
                signal.get("trend", "NEUTRAL"), signal.get("signal_strength", 0),
                signal.get("reasons", []),
                acted_on=False,
                skip_reason=skip_reason,
            )

            if skip_reason is not None:
                continue

            # Candidato válido: guardamos con su fuerza para priorizar
            buy_candidates.append((pair, signal, df))

        # Ordenar candidatos de mayor a menor strength
        buy_candidates.sort(key=lambda x: x[1].get("signal_strength", 0), reverse=True)

        for pair, signal, df in buy_candidates:
            # Re-comprobar slots: pueden haberse llenado con candidatos anteriores
            open_positions = self.trader.get_open_positions()
            if len(open_positions) >= self.max_concurrent_trades:
                logger.info(
                    f"SKIP {pair}: slots llenos ({len(open_positions)}/{self.max_concurrent_trades})"
                )
                break

            # También re-verificar par ya abierto (otro candidato lo pudo abrir)
            if pair in open_positions:
                continue

            available = self.trader.get_available_capital()
            usdt_to_invest = available * self.trader.max_position_pct
            if usdt_to_invest < 10:
                logger.info(f"SKIP {pair}: insufficient capital (${available:.2f})")
                self.slog.log_opportunity_skip(
                    "crypto", pair, pair, "rsi_bb_mean_reversion",
                    f"insufficient_capital=${available:.2f}"
                )
                continue

            current_price = signal["price"]
            atr_pct = signal.get("atr_pct")
            order = self.trader.place_buy_order(pair, usdt_to_invest, current_price, atr_pct=atr_pct)
            if order:
                self._trades_today += 1
                regime = signal.get("regime", "RANGING")
                # Store entry regime for use in analyze_exit later
                self._position_entry_regime[pair] = regime
                summary["new_trades"].append({
                    "symbol": pair,
                    "usdt": usdt_to_invest,
                    "price": current_price,
                    "signal_strength": signal["signal_strength"],
                    "regime": regime,
                    "reasons": signal["reasons"],
                })
                logger.info(
                    f"NEW TRADE: {pair} [{regime}] | ${usdt_to_invest:.2f} @ {current_price:.2f} | "
                    f"Strength: {signal['signal_strength']:.2f} | ADX: {signal.get('adx')} | "
                    f"RSI: {signal['rsi']} | Reasons: {signal['reasons']}"
                )
                sl_price = current_price * (1 - self.config.get("crypto", {}).get("risk", {}).get("stop_loss_pct", 0.008))
                # TP depends on ATR (dynamic) — approximate for logging
                tp_pct = min(0.030, max(0.010, (atr_pct or 0) * 2.0)) if atr_pct else self.config.get("crypto", {}).get("risk", {}).get("take_profit_pct", 0.015)
                tp_price = current_price * (1 + tp_pct)
                self.slog.log_trade_open(
                    "crypto", order.get("orderId", self._trades_today), pair,
                    f"regime_{regime.lower()}", "BUY",
                    current_price, usdt_to_invest,
                    stop_loss=round(sl_price, 2), take_profit=round(tp_price, 2),
                    confidence=signal.get("signal_strength"),
                    extra={
                        "rsi": signal.get("rsi"),
                        "rsi_slope": signal.get("rsi_slope"),
                        "adx": signal.get("adx"),
                        "regime": regime,
                        "trend": signal.get("trend"),
                        "reasons": signal.get("reasons", []),
                    },
                )
                if self.notifications:
                    try:
                        self.notifications.send(
                            f"📈 CRYPTO BUY: {pair} | ${usdt_to_invest:.2f} @ {current_price:.2f}\n"
                            f"RSI: {signal['rsi']} | {' | '.join(signal['reasons'][:2])}"
                        )
                    except Exception:
                        pass
                if self.db:
                    try:
                        self.db.log_crypto_trade_open(pair, usdt_to_invest, current_price, signal)
                    except Exception:
                        pass
                # Actualizar posiciones abiertas para el siguiente candidato
                open_positions = self.trader.get_open_positions()

        elapsed = (datetime.now(timezone.utc) - cycle_start).total_seconds()
        self.slog.log_crypto_cycle(
            self._cycle_count,
            signals_evaluated=len(self.pairs),
            trades_opened=len(summary["new_trades"]),
            positions_closed=len(summary["closed_positions"]),
            market_bias=self._market_bias,
            balance_usdt=self.trader.get_available_capital(),
            duration_seconds=elapsed,
        )
        if len(summary["new_trades"]) > 0 or len(summary["closed_positions"]) > 0:
            logger.info(
                f"Crypto cycle #{self._cycle_count} done in {elapsed:.1f}s | "
                f"New trades: {len(summary['new_trades'])} | "
                f"Closed: {len(summary['closed_positions'])}"
            )
        return summary

    def _update_market_bias(self, claude_agent) -> None:
        """Ask Claude for current crypto market bias."""
        open_pos = self.trader.get_open_positions()
        cap = self.trader.get_available_capital()

        indicator_lines = []
        for pair in self.pairs:
            df = self.trader.get_klines(pair, limit=50)
            if df is not None:
                sig = self.signal_engine.analyze(df, pair)
                indicator_lines.append(
                    f"  {pair}: RSI={sig['rsi']} Trend={sig['trend']} "
                    f"Price={sig['price']} vs BB[{sig['bb_lower']}/{sig['bb_middle']}/{sig['bb_upper']}]"
                )

        context = (
            f"Capital disponible: ${cap:.2f}\n"
            f"Posiciones abiertas: {len(open_pos)}\n"
            f"P&L total sesion: ${self._total_pnl:+.2f}\n"
            f"Indicadores actuales:\n" + "\n".join(indicator_lines)
        )

        result = claude_agent.analyze_crypto_context(context)
        if result and "bias" in result:
            old_bias = self._market_bias
            self._market_bias = result["bias"].upper()
            logger.info(
                f"Market bias updated: {old_bias} -> {self._market_bias} "
                f"(confidence: {result.get('confidence', '?')})"
            )
            bias_reasoning = (
                f"[confidence={result.get('confidence','?')} old={old_bias}] "
                + result.get("reason", "")
            )
            self.slog.log_claude_decision(
                "crypto",
                decision=f"BIAS_{self._market_bias}",
                reasoning=bias_reasoning[:300],
            )

    def get_status(self) -> dict:
        """Return current status summary."""
        return {
            "cycle_count": self._cycle_count,
            "market_bias": self._market_bias,
            "total_pnl": self._total_pnl,
            "trades_today": self._trades_today,
            "open_positions": self.trader.get_open_positions(),
            "available_capital": self.trader.get_available_capital(),
        }
