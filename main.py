"""
Polymarket Trading Bot - Weather-only orchestrator (v2).

Features:
  - Single-process PID lock (data/bot.lock) enforced via psutil.
  - Weather-only strategy loop (no crypto, bonding, arbitrage).
  - Multi-source ensemble + confidence engine (see strategies/weather_bot.py).
  - Trading windows filter (config.trading_windows_utc).
  - Shadow mode: EXECUTE_TRADES=false records would-be trades without ordering.
  - Structured JSONL logging for every skip/open/close.

Usage:
  python main.py --mode demo
  python main.py --mode live
  EXECUTE_TRADES=false python main.py --mode demo   # shadow mode
"""

import argparse
import asyncio
import atexit
import logging
import os
import signal
import sys
import time

# On Windows, the default ProactorEventLoop leaks "Event loop is closed" errors
# when aiohttp transports are GC'd after asyncio.run() ends. Selector loop avoids it.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import colorlog
import psutil
import schedule
import yaml
from dotenv import load_dotenv

from database import Database
from polymarket_api import PolymarketAPI
from claude_agent import ClaudeAgent
from risk_manager import RiskManager
from memory import MemorySystem
from notifications import TelegramNotifier
from strategies.weather_bot import WeatherBotStrategy
from strategies.weather_sources import build_sources
from structured_logger import get_logger as get_slog, Event, SkipReason
from reeval_engine import ReevalEngine
from state_reset import reset_state
from reports_scheduler import maybe_generate_report

LOCK_PATH = Path("data/bot.lock")


# ============================================================
# Logging
# ============================================================
def setup_logging(config: dict) -> logging.Logger:
    log_level = os.getenv("BOT_LOG_LEVEL") or config.get("logging", {}).get("level", "INFO")
    log_dir = config.get("logging", {}).get("log_dir", "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"bot_{datetime.now().strftime('%Y-%m-%d')}.log")

    console = colorlog.StreamHandler()
    console.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        log_colors={"DEBUG": "cyan", "INFO": "green", "WARNING": "yellow",
                    "ERROR": "red", "CRITICAL": "bold_red"},
    ))
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level))
    root.addHandler(console)
    root.addHandler(file_handler)
    return logging.getLogger("polymarket_bot")


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# PID lock — ensures only one bot instance runs at a time
# ============================================================
def acquire_lock(logger: logging.Logger, mode: str) -> None:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        try:
            old_pid = int(LOCK_PATH.read_text().strip())
        except (ValueError, OSError):
            old_pid = -1
        if old_pid > 0 and psutil.pid_exists(old_pid):
            logger.error(f"Another bot instance is already running (PID={old_pid}). Aborting.")
            get_slog().log_process_lock("rejected", old_pid, {"mode": mode})
            try:
                TelegramNotifier().notify_single_process_violation(old_pid)
            except Exception:
                pass
            sys.exit(1)
        else:
            logger.warning(f"Removing orphan lock (PID={old_pid} not alive).")
            try:
                LOCK_PATH.unlink()
            except OSError:
                pass

    pid = os.getpid()
    LOCK_PATH.write_text(str(pid))
    logger.info(f"Process lock acquired (PID={pid}).")
    get_slog().log_process_lock("acquired", pid, {"mode": mode})

    def _release(*_):
        try:
            if LOCK_PATH.exists() and LOCK_PATH.read_text().strip() == str(pid):
                LOCK_PATH.unlink()
                get_slog().log_process_lock("released", pid)
        except Exception:
            pass

    atexit.register(_release)
    try:
        signal.signal(signal.SIGTERM, lambda *_: (_release(), sys.exit(0)))
        signal.signal(signal.SIGINT, lambda *_: (_release(), sys.exit(0)))
    except (ValueError, OSError):
        # SIGTERM/SIGINT not available on all platforms (Windows subthread, etc.)
        pass


# ============================================================
# Trading windows
# ============================================================
def in_trading_window(windows: list, utc_hour: int) -> bool:
    if not windows:
        return True  # no restriction
    for pair in windows:
        try:
            start, end = int(pair[0]), int(pair[1])
        except (TypeError, ValueError, IndexError):
            continue
        if start <= utc_hour < end:
            return True
    return False


# ============================================================
# TradingBot
# ============================================================
class TradingBot:
    def __init__(self, config: dict, mode: str, shadow: bool = False):
        self.config = config
        self.mode = mode.upper()
        self.shadow = shadow
        self.cycle_count = 0
        self.initial_capital = config["bot"]["demo_capital"]
        self.trading_windows = config.get("trading_windows_utc", [])

        self.logger = logging.getLogger("polymarket_bot")
        self.slog = get_slog()

        self.db = Database(config.get("database", {}).get("path", "data/bot.db"))

        # Clean start: archive old logs + wipe runtime tables (trades, cycles, skips, ...)
        if bool(config.get("bot", {}).get("reset_on_start", False)):
            try:
                summary = reset_state(self.db, config)
                self.logger.info(f"reset_on_start → {summary}")
            except Exception as e:
                self.logger.error(f"reset_on_start failed: {e}")

        self.api = PolymarketAPI(config)
        self.risk = RiskManager(config)
        self.notifier = TelegramNotifier()

        # Build weather sources from config (skips any provider whose key is missing).
        sources_cfg = config.get("weather", {}).get("sources", {})
        self.sources = build_sources(sources_cfg)
        self.logger.info(f"Weather sources active: {[s.name for s in self.sources]}")

        self.weather_bot = WeatherBotStrategy(config, self.sources, self.db)
        self.reeval = ReevalEngine(config, self.db, self.api, self.weather_bot, self.risk)
        self.memory = MemorySystem(self.db, config)
        self._last_report_at = 0.0

        self.claude = None  # Claude removed from trading pipeline

        self.slog.log_system("bot_start", {
            "mode": self.mode, "shadow": self.shadow,
            "capital": self.initial_capital,
            "sources": [s.name for s in self.sources],
        })

    # ------------------------------------------------------------------
    def get_portfolio_state(self) -> dict:
        portfolio_value = self.db.get_portfolio_value(self.initial_capital)
        total_exposure = self.db.get_total_exposure()
        stats = self.db.get_statistics(self.initial_capital)
        return {
            "portfolio_value": round(portfolio_value + total_exposure, 2),
            "available_capital": round(portfolio_value, 2),
            "total_exposure": round(total_exposure, 2),
            "open_positions": stats["open_positions"],
            "total_pnl": round(stats["total_pnl"], 4),
            "win_rate": stats["win_rate"],
            "yes_wr": self.db.get_side_rolling_winrate("YES", 20),
            "no_wr": self.db.get_side_rolling_winrate("NO", 20),
            "initial_capital": self.initial_capital,
        }

    # ------------------------------------------------------------------
    def check_and_close_positions(self, markets_data: list) -> float:
        """
        Política HOLD-TO-RESOLUTION: las posiciones sólo se cierran cuando Polymarket
        resuelve el mercado (end_date pasado). No hay stop-loss, no take-profit, no
        auto-cierre por precio. En DEMO simulamos la resolución de Polymarket cuando
        end_date queda en el pasado, usando price_yes>0.5 como outcome proxy.
        """
        open_positions = self.db.get_open_positions()
        if not open_positions:
            return 0.0

        current_prices = {
            m["id"]: {"price_yes": m["price_yes"], "price_no": m["price_no"]}
            for m in markets_data
        }
        for pos in open_positions:
            mid = pos.get("market_id")
            if mid and mid not in current_prices:
                prices = self.api.get_market_prices(mid)
                if prices:
                    current_prices[mid] = prices

        pnl_from_closes = 0.0

        if self.mode != "DEMO":
            return pnl_from_closes

        # DEMO resolution simulation.
        # Política: NUNCA cerrar por precio de mercado intermedio.
        # Solo cerrar cuando Polymarket publica el outcome binario (price=0 o 1),
        # que es la única señal fiable de resolución real.
        # Safety fallback: si target_dt + 72h ya pasó y el mercado sigue sin
        # resolver en la API, cerrar con precio actual (mercado ilíquido/inactivo).
        for pos in open_positions:
            mid = pos.get("market_id")
            side = pos.get("side", "YES")
            entry_price = pos.get("price_entry", 0) or 0
            shares = pos.get("shares", 0) or 0

            # Fetch latest market data directly from API (not just scan cache)
            # to catch resolved markets that left the active scan.
            prices = current_prices.get(mid, {})
            if not prices:
                detail = self.api.get_market_detail(mid)
                if detail:
                    prices = detail
                    current_prices[mid] = detail

            p_yes = prices.get("price_yes", None)
            p_no  = prices.get("price_no",  None)

            # ── Resolution check 1: binary price published by Polymarket ──
            # When Polymarket resolves, outcomePrices becomes ["1","0"] or ["0","1"].
            # We only trust closes when price is definitively 0 or 1 (±1 cent tolerance).
            RESOLVED_THRESHOLD = 0.01
            if p_yes is not None and p_no is not None:
                yes_resolved = p_yes >= (1.0 - RESOLVED_THRESHOLD) or p_yes <= RESOLVED_THRESHOLD
                if yes_resolved:
                    # Actual binary outcome available — close now.
                    won = (p_yes > 0.5) if side == "YES" else (p_no > 0.5)
                    current_price = p_yes if side == "YES" else p_no
                    pnl = (1.0 - entry_price) * shares if won else -entry_price * shares
                    self.db.close_position(pos["id"], current_price, pnl, "SIMULATED")
                    pnl_from_closes += pnl
                    trade = self.db.get_trade_by_id(pos["id"])
                    if trade:
                        self._log_close(trade, current_price, pnl,
                                        "resolved_win" if won else "resolved_loss")
                    continue  # done with this position

            # ── Resolution check 2: safety timeout ──
            # If the market never reaches binary prices (illiquid / delisted),
            # fall back to closing after target_dt + 72h using current price.
            target_raw = pos.get("target_dt") or pos.get("target_date") or ""
            if not target_raw:
                continue  # no target date stored, cannot estimate timeout
            try:
                t = datetime.fromisoformat(str(target_raw).replace(" ", "T").replace("Z", "+00:00"))
                if t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
                safety_deadline = t + timedelta(hours=72)
            except Exception:
                continue

            if datetime.now(timezone.utc) < safety_deadline:
                continue  # within 72h window, keep holding

            # Safety timeout reached — close with current (intermediate) price
            if p_yes is None or p_no is None:
                continue  # no price data at all, skip
            won = (p_yes > 0.5) if side == "YES" else (p_no > 0.5)
            current_price = p_yes if side == "YES" else p_no
            pnl = (1.0 - entry_price) * shares if won else -entry_price * shares
            self.db.close_position(pos["id"], current_price, pnl, "SIMULATED")
            pnl_from_closes += pnl
            trade = self.db.get_trade_by_id(pos["id"])
            if trade:
                self._log_close(trade, current_price, pnl,
                                "timeout_win" if won else "timeout_loss")

        return pnl_from_closes

    def _log_close(self, trade: dict, exit_price: float, pnl: float, reason: str) -> None:
        entry = trade.get("price_entry", 0) or 0
        pnl_pct = (pnl / (trade.get("size_usdc", 1) or 1)) if trade.get("size_usdc") else 0
        self.slog.log_trade_close(
            trace_id=trade.get("trace_id"),
            trade_id=trade["id"],
            market_id=trade.get("market_id", ""),
            side=trade.get("side", ""),
            entry_price=entry,
            exit_price=exit_price,
            pnl_usdc=pnl,
            pnl_pct=pnl_pct,
            close_reason=reason,
        )
        # Actuals feedback loop — only for winners/losers that resolve at the
        # target_dt (not for reeval-driven CLOSE-by-risk, since those may
        # resolve days later and our forecast was legitimate at the time).
        if reason in ("expired_win", "expired_loss") and \
                (trade.get("strategy") or "") == "weather_bot":
            try:
                from weather_actuals import update_source_reliability_from_trade
                summary = update_source_reliability_from_trade(self.db, trade)
                if summary.get("updated", 0) > 0:
                    self.logger.info(
                        f"Source reliability updated from trade {trade['id']}: "
                        f"actual={summary.get('actual'):.2f} "
                        f"sources={summary.get('sources')}"
                    )
            except Exception as e:
                self.logger.debug(f"actuals feedback failed trade={trade.get('id')}: {e}")
        self.notifier.notify_trade_closed(trade, pnl, reason)
        if self.claude and self.memory.analysis_enabled:
            self.memory.analyze_closed_trade(trade, self.claude)

    # ------------------------------------------------------------------
    def execute_trade(self, opp: dict, decision: dict, cycle_id: int) -> bool:
        price = float(decision.get("price_entry") or opp.get("price") or 0)
        size = float(decision.get("suggested_size_usdc") or 0)
        side = decision.get("side", opp.get("side"))
        market_id = opp["market_id"]
        market_question = opp["market_question"]

        if price <= 0 or size <= 0:
            self.logger.warning(f"Invalid trade params: price={price} size={size}")
            return False

        open_same_side = [
            p for p in self.db.get_open_positions_for_market(market_id)
            if p.get("side") == side
        ]
        if open_same_side:
            self.slog.log_opportunity_skip(
                market_id=market_id,
                market_question=market_question,
                reason_code=SkipReason.RECENT_SL_SAME_MARKET,
                reason_detail=f"existing OPEN {side} position on same market (no pyramid)",
                confidence_score=opp.get("confidence_score"),
                trace_id=opp.get("trace_id"),
            )
            self.logger.info(
                f"Trade blocked [OPEN_SAME_SIDE]: {market_question[:60]} ({side})"
            )
            return False

        cooldown_hours = float(
            self.config.get("pyramiding", {}).get("cooldown_after_sl_hours", 0)
        )
        if cooldown_hours > 0 and self.db.has_recent_losing_close(market_id, side, cooldown_hours):
            self.slog.log_opportunity_skip(
                market_id=market_id,
                market_question=market_question,
                reason_code=SkipReason.RECENT_SL_SAME_MARKET,
                reason_detail=f"same market+side closed at a loss within {cooldown_hours}h",
                confidence_score=opp.get("confidence_score"),
                trace_id=opp.get("trace_id"),
            )
            self.logger.info(
                f"Trade blocked [RECENT_SL_SAME_MARKET]: {market_question[:60]} ({side})"
            )
            return False

        portfolio = self.get_portfolio_state()
        same_mkt_positions = self.db.get_open_positions_for_market(market_id)

        allowed, reason = self.risk.can_add_to_market(
            market_id=market_id, new_side=side,
            new_confidence=float(opp.get("confidence_score", 0)),
            new_size=size,
            portfolio_value=portfolio["portfolio_value"],
            existing_positions=same_mkt_positions,
            hours_to_resolution=float(opp.get("hours_to_resolution", 0)),
            new_edge=float(opp.get("edge", opp.get("ev_calculated", 0)) or 0),
            new_price=price,
        )
        self.slog.log_pyramid_eval(
            trace_id=opp.get("trace_id"), market_id=market_id,
            allowed=allowed, reason=reason,
            extra={"confidence": opp.get("confidence_score"),
                   "existing_positions": len(same_mkt_positions)},
        )
        if not allowed:
            self.logger.info(f"Trade blocked [{reason}]: {market_question[:60]}")
            return False

        if not self.risk.can_open_new_position(portfolio["portfolio_value"], portfolio["total_exposure"]):
            return False

        max_size = portfolio["portfolio_value"] * self.config["risk"]["max_position_pct"]
        size = min(size, max_size, portfolio["available_capital"])
        if size < 2.0:
            return False

        shares = size / price if price > 0 else 0

        # Orderbook depth gate — don't place an order against an empty/thin book.
        # We're BUYing the side we hold, which lifts the ask: if ask_size < 30%
        # of our shares we'll walk the book and pay way worse than `price`.
        # Only enforce in LIVE (DEMO fills are synthetic, not liquidity-limited)
        # but we DO log the check so DEMO analyses surface markets that would
        # have been un-executable.
        depth_cfg = float(self.config.get("risk", {}).get("min_orderbook_depth_frac", 0.30))
        token_for_side = opp.get("token_yes") if side == "YES" else opp.get("token_no")
        if token_for_side and shares > 0:
            book = self.api.get_orderbook(token_for_side)
            ask_size = (book or {}).get("ask_size", 0.0) or 0.0
            required = shares * depth_cfg
            if ask_size < required:
                self.logger.info(
                    f"Thin book [{market_question[:50]}]: ask_size={ask_size:.1f} "
                    f"< {required:.1f} ({depth_cfg:.0%} of {shares:.1f} shares)"
                )
                if self.mode == "LIVE":
                    self.slog.log_opportunity_skip(
                        market_id=market_id,
                        market_question=market_question,
                        reason_code=SkipReason.INSUFFICIENT_LIQUIDITY,
                        reason_detail=f"ask_size {ask_size:.1f} < {required:.1f}",
                        trace_id=opp.get("trace_id"),
                    )
                    return False

        is_shadow = self.shadow or os.getenv("EXECUTE_TRADES", "true").lower() == "false"

        if self.mode == "LIVE" and not is_shadow:
            private_key = os.getenv("POLYMARKET_PRIVATE_KEY", "")
            order = self.api.place_order_live(market_id, side, size, price, private_key,
                                              "GTC", market_question)
            if not order:
                self.logger.error(f"Live order failed: {market_question[:60]}")
                return False
        elif not is_shadow:
            self.api.place_order_demo(market_id, side, size, price, market_question)

        import json as _json
        cond_params = opp.get("condition_params") or {}
        clob_token_ids = opp.get("clob_token_ids") or ""

        trade_data = {
            "trace_id": opp.get("trace_id"),
            "cycle_id": cycle_id,
            "mode": self.mode,
            "action": "BUY",
            "market_id": market_id,
            "market_question": market_question,
            "strategy": "weather_bot",
            "side": side,
            "price_entry": price,
            "size_usdc": size,
            "shares": round(shares, 6),
            "prob_real_estimated": opp.get("prob_real_estimated"),
            "prob_market": opp.get("prob_market"),
            "ev_calculated": opp.get("ev_calculated"),
            "reasoning": decision.get("reasoning") or opp.get("reasoning", ""),
            "status": "OPEN",
            "confidence_score": opp.get("confidence_score"),
            "confidence_breakdown": opp.get("confidence_breakdown"),
            "ensemble_mean": opp.get("ensemble_mean"),
            "ensemble_std": opp.get("ensemble_std"),
            "ensemble_sources_used": opp.get("ensemble_sources_used"),
            "sources_json": opp.get("sources_json"),
            "vetos_triggered": opp.get("vetos_triggered"),
            "side_wr_at_entry": opp.get("side_wr_at_entry"),
            "is_shadow": 1 if is_shadow else 0,
            # v3 enrichment — required by reeval_engine
            "risk_score": opp.get("risk_score"),
            "prob_real": opp.get("prob_real"),
            "min_prob_required": opp.get("min_prob_required"),
            "days_ahead": opp.get("days_ahead"),
            "location": opp.get("location"),
            "lat": opp.get("lat"),
            "lon": opp.get("lon"),
            "target_date": opp.get("target_date"),
            "target_dt": opp.get("target_dt"),
            "condition_type": opp.get("weather_type"),
            "metric": opp.get("metric"),
            "condition_params": _json.dumps(cond_params) if isinstance(cond_params, dict) else str(cond_params),
            "clob_token_ids": clob_token_ids if isinstance(clob_token_ids, str) else _json.dumps(clob_token_ids),
            "token_yes": opp.get("token_yes", ""),
            "token_no":  opp.get("token_no",  ""),
        }
        trade_id = self.db.log_trade(trade_data)

        self.slog.log_trade_open(
            trace_id=opp.get("trace_id"),
            trade_id=trade_id,
            market_id=market_id,
            market_question=market_question,
            side=side,
            price=price,
            size_usdc=size,
            confidence_score=float(opp.get("confidence_score", 0)),
            ensemble_mean=opp.get("ensemble_mean"),
            ensemble_std=opp.get("ensemble_std"),
            sources_used=opp.get("ensemble_sources_used"),
            hours_to_resolution=opp.get("hours_to_resolution"),
            is_shadow=is_shadow,
        )
        if not is_shadow:
            trade_data["id"] = trade_id
            self.notifier.notify_trade(trade_data)

        self.logger.info(
            f"{'[SHADOW]' if is_shadow else '[TRADE]'} #{trade_id} {side} @ {price:.4f} "
            f"${size:.2f} | conf={opp.get('confidence_score',0):.0f} | {market_question[:55]}"
        )
        return True

    # ------------------------------------------------------------------
    async def run_weather_cycle_async(self) -> None:
        self.cycle_count += 1
        cycle_id = self.cycle_count
        self.slog.log_cycle_start(cycle_id, self.mode)
        cycle_start = time.time()

        utc_hour = datetime.now(timezone.utc).hour
        in_window = in_trading_window(self.trading_windows, utc_hour)
        self.logger.info(f"{'='*60}")
        if in_window:
            self.logger.info(f"CYCLE #{cycle_id}")
        else:
            self.logger.info(f"CYCLE #{cycle_id} — outside trading window (UTC {utc_hour:02d}h)")

        try:
            portfolio = self.get_portfolio_state()
            self.logger.info(
                f"Portfolio: ${portfolio['portfolio_value']:.2f} | "
                f"Exposure: ${portfolio['total_exposure']:.2f} | "
                f"PnL: ${portfolio['total_pnl']:+.4f} | "
                f"WR YES={portfolio['yes_wr']:.2f} NO={portfolio['no_wr']:.2f}"
            )

            wb_cfg = dict(self.config.get("strategies", {}).get("weather_bot", {}))
            wb_cfg.update(self.config.get("weather", {}) or {})
            min_vol = wb_cfg.get("min_volume", 500)
            min_liq = wb_cfg.get("min_liquidity", 200)
            scan_limit = wb_cfg.get("scan_limit", 1000)

            markets = self.api.scan_weather_markets(
                min_volume=min_vol, min_liquidity=min_liq, max_results=scan_limit,
            )

            pnl_cycle = self.check_and_close_positions(markets)

            # Re-evaluate every open position: mark-to-market, decide HOLD/AVG_UP/CLOSE.
            try:
                reeval_summary = await self.reeval.run_cycle(cycle_id=cycle_id)
                self.logger.info(f"Reeval: {reeval_summary}")
            except Exception as e:
                self.logger.warning(f"Reeval cycle failed: {e}")

            portfolio = self.get_portfolio_state()

            if not in_window:
                self.logger.info("Outside trading window — no new entries this cycle")
                self._log_cycle_end(cycle_id, cycle_start, len(markets), 0, 0, pnl_cycle, portfolio)
                return

            opportunities = await self.weather_bot.find_opportunities(markets, cycle_id=cycle_id)

            # Compute dollar-sized suggestion scaled by prob_real.
            wb_cfg2 = self.config.get("weather", {}) or {}
            entry_fraction = float(wb_cfg2.get("initial_entry_fraction", 0.50))
            weather_kelly = float(wb_cfg2.get("kelly_fraction", self.risk.kelly_fraction))
            max_pos_pct_w = float(wb_cfg2.get("max_position_pct", self.risk.max_position_pct))
            bankroll = float(portfolio.get("portfolio_value", 0) or 0)
            for opp in opportunities:
                prob_real = float(opp.get("prob_real") or opp.get("prob_real_estimated") or 0)
                min_prob = float(opp.get("min_prob_required") or 0.70)
                size = self.risk.calculate_position_size(
                    portfolio_value=bankroll,
                    prob_real=prob_real,
                    price_market=float(opp.get("price", 0)),
                    kelly_fraction=weather_kelly,
                    min_prob_entry=min_prob,
                    entry_fraction=entry_fraction,
                    max_position_pct=max_pos_pct_w,
                )
                opp["suggested_size_usdc"] = size

            # Pre-screen: drop opportunities that risk_manager will already block.
            # Saves Claude API cost on trades that can't be opened anyway.
            screened: list = []
            for opp in opportunities:
                same = self.db.get_open_positions_for_market(opp["market_id"])
                allowed, reason = self.risk.can_add_to_market(
                    market_id=opp["market_id"],
                    new_side=opp.get("side", "YES"),
                    new_confidence=float(opp.get("confidence_score", 0)),
                    new_size=float(opp.get("suggested_size_usdc", 0)),
                    portfolio_value=portfolio["portfolio_value"],
                    existing_positions=same,
                    hours_to_resolution=float(opp.get("hours_to_resolution", 0)),
                    new_edge=float(opp.get("edge", opp.get("ev_calculated", 0)) or 0),
                    new_price=float(opp.get("price", 0)),
                )
                if allowed:
                    screened.append(opp)
                else:
                    self.slog.log_pyramid_eval(
                        trace_id=opp.get("trace_id"),
                        market_id=opp["market_id"],
                        allowed=False, reason=reason,
                        extra={"pre_screen": True,
                               "confidence": opp.get("confidence_score"),
                               "existing_positions": len(same)},
                    )
                    self.logger.info(f"Pre-screened out [{reason}]: {opp.get('market_question','')[:55]}")
            opportunities = screened

            trades_executed = 0
            max_per_cycle = self.config["bot"].get("max_trades_per_cycle", 2)

            decisions_list: list = []
            if opportunities and self.claude:
                mem_ctx = self.memory.get_memory_prompt_section() if self.memory else ""
                claude_result = self.claude.analyze_weather_opportunities(
                    opportunities, portfolio, mem_ctx,
                )
                decisions_list = claude_result.get("decisions", []) or []
                self.slog.log_claude_decision(
                    decision="ANALYZED",
                    reasoning=(claude_result.get("self_assessment") or "")[:300],
                    trades_approved=sum(1 for d in decisions_list if d.get("action") == "BUY"),
                    trades_rejected=sum(1 for d in decisions_list if d.get("action") == "SKIP"),
                )

            # Fallback: if Claude unavailable, approve top-K directly with size from kelly
            if not decisions_list:
                for opp in opportunities[:max_per_cycle]:
                    decisions_list.append({
                        "market_id": opp["market_id"],
                        "action": "BUY",
                        "side": opp["side"],
                        "price_entry": opp["price"],
                        "suggested_size_usdc": opp.get("suggested_size_usdc", 0),
                        "reasoning": "auto-approve (claude offline)",
                    })

            opp_by_id = {o["market_id"]: o for o in opportunities}
            buys = [d for d in decisions_list if d.get("action") == "BUY"]
            buys.sort(key=lambda d: opp_by_id.get(d.get("market_id"), {}).get("confidence_score", 0),
                      reverse=True)

            # Cluster caps: avoid stacking correlated bets on the same day /
            # same condition type. Today-resolving markets all share weather
            # noise; 3 "temp_above_c" longs at the same city+day would blow up
            # together. Count currently-open positions plus buys this cycle.
            wb_cfg3 = self.config.get("weather", {}) or {}
            cap_per_date = int(wb_cfg3.get("max_trades_per_target_date", 2))
            cap_per_cond = int(wb_cfg3.get("max_trades_per_condition_type", 3))
            open_all = self.db.get_open_positions() or []
            date_counts: dict[str, int] = {}
            cond_counts: dict[str, int] = {}
            for p in open_all:
                td = (p.get("target_date") or p.get("target_dt") or "")[:10]
                ct = p.get("condition_type") or p.get("weather_type") or ""
                if td:
                    date_counts[td] = date_counts.get(td, 0) + 1
                if ct:
                    cond_counts[ct] = cond_counts.get(ct, 0) + 1

            filtered_buys = []
            for d in buys:
                opp = opp_by_id.get(d.get("market_id"), {})
                td = (opp.get("target_date") or "")[:10]
                ct = opp.get("weather_type") or opp.get("condition_type") or ""
                if td and date_counts.get(td, 0) >= cap_per_date:
                    self.logger.info(f"Cluster cap [target_date={td}]: skipping {opp.get('market_question','')[:55]}")
                    continue
                if ct and cond_counts.get(ct, 0) >= cap_per_cond:
                    self.logger.info(f"Cluster cap [condition={ct}]: skipping {opp.get('market_question','')[:55]}")
                    continue
                filtered_buys.append(d)
                if td:
                    date_counts[td] = date_counts.get(td, 0) + 1
                if ct:
                    cond_counts[ct] = cond_counts.get(ct, 0) + 1
            buys = filtered_buys

            for decision in buys[:max_per_cycle]:
                opp = opp_by_id.get(decision.get("market_id"))
                if not opp:
                    continue
                if self.execute_trade(opp, decision, cycle_id):
                    trades_executed += 1

            portfolio = self.get_portfolio_state()
            self._log_cycle_end(cycle_id, cycle_start, len(markets),
                                len(opportunities), trades_executed, pnl_cycle, portfolio)

            # Memory tasks (analyze closed trades, extract rules, etc.)
            self._run_memory_tasks()

            # Hourly reports
            try:
                self._last_report_at = maybe_generate_report(
                    self.config, self.db, self._last_report_at
                )
            except Exception as e:
                self.logger.debug(f"report generation failed: {e}")

        except Exception as e:
            self.logger.error(f"Cycle #{cycle_id} failed: {e}", exc_info=True)
            self.slog.log_error("weather", type(e).__name__, str(e), exc=e,
                                context={"cycle": cycle_id})
            self.notifier.notify_error(str(e), f"Cycle #{cycle_id}")

    def run_weather_cycle(self) -> None:
        """Sync wrapper for the scheduler."""
        try:
            asyncio.run(self.run_weather_cycle_async())
        except RuntimeError:
            # In case an event loop is already active (shouldn't happen under schedule)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.run_weather_cycle_async())
            loop.close()

    def _log_cycle_end(self, cycle_id, cycle_start, markets, opps, trades, pnl_cycle, portfolio):
        elapsed = time.time() - cycle_start
        cycle_data = {
            "mode": self.mode,
            "markets_scanned": markets,
            "weather_markets_found": markets,
            "opportunities_found": opps,
            "trades_executed": trades,
            "portfolio_value": portfolio["portfolio_value"],
            "pnl_cycle": pnl_cycle,
            "pnl_total": portfolio["total_pnl"],
            "capital_initial": self.initial_capital,
            "sources_available": len(self.sources),
            "duration_seconds": round(elapsed, 2),
        }
        self.db.log_cycle(cycle_data)
        self.slog.log_cycle_end(cycle_id, {
            "markets_scanned": markets,
            "opportunities": opps,
            "trades_executed": trades,
            "portfolio_value": portfolio["portfolio_value"],
            "pnl_cycle": pnl_cycle,
            "duration_seconds": round(elapsed, 2),
        })
        self.notifier.notify_cycle_summary(cycle_data)
        self.logger.info(
            f"CYCLE #{cycle_id} done in {elapsed:.1f}s | "
            f"markets={markets} opps={opps} trades={trades} | "
            f"PnL cycle=${pnl_cycle:+.4f} portfolio=${portfolio['portfolio_value']:.2f}"
        )

    def _run_memory_tasks(self) -> None:
        if not self.claude:
            return
        try:
            unanalyzed = self.memory.get_unanalyzed_trades()
            for trade in unanalyzed[:3]:
                self.memory.analyze_closed_trade(trade, self.claude)
        except Exception as e:
            self.logger.debug(f"Memory tasks failed: {e}")


# ============================================================
# Entry point
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Polymarket Weather Trading Bot")
    parser.add_argument("--mode", type=str, default="demo", choices=["demo", "live"])
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--shadow", action="store_true", help="Log trades without executing")
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)
    logger = setup_logging(config)
    mode = args.mode.upper()
    shadow = args.shadow or os.getenv("EXECUTE_TRADES", "true").lower() == "false"

    logger.info("=" * 60)
    logger.info(f"  Weather Trading Bot v2 — {mode}{' [SHADOW]' if shadow else ''}")
    logger.info(f"  Capital: ${config['bot']['demo_capital']:,.2f}")
    logger.info(f"  Model:   {config.get('claude', {}).get('model', 'n/a')}")
    logger.info("=" * 60)

    if mode == "LIVE" and not shadow:
        logger.warning("LIVE mode: real orders will be placed.")
        if not os.getenv("POLYMARKET_PRIVATE_KEY"):
            logger.error("POLYMARKET_PRIVATE_KEY not set. Aborting.")
            sys.exit(1)

    acquire_lock(logger, mode)
    bot = TradingBot(config, mode, shadow=shadow)

    interval_min = config.get("bot", {}).get("cycle_interval_minutes", 10)
    schedule.every(interval_min).minutes.do(bot.run_weather_cycle)
    logger.info(f"Scheduled: weather cycle every {interval_min} minutes.")

    # Optional: crypto-lag MAKER bot in a parallel daemon thread. No-op if
    # config.crypto_lag.enabled is false (default).
    crypto_lag_handle = None
    cl_enabled = bool((config.get("crypto_lag") or {}).get("enabled"))
    try:
        from crypto_lag_runner import start_crypto_lag, stop_crypto_lag
        crypto_lag_handle = start_crypto_lag(config, bot.db, logger, notifier=bot.notifier)
    except Exception as exc:
        logger.warning(f"crypto_lag failed to start (continuing without it): {exc}")

    # Daily summary at 23:55 UTC — runs in the main thread via `schedule`.
    def _send_daily_summary():
        try:
            stats = bot.db.get_statistics(bot.initial_capital)
            sources = bot.db.get_source_reliability()
            bot.notifier.notify_daily_summary(
                portfolio_now=stats.get("portfolio_value", bot.initial_capital),
                capital_initial=bot.initial_capital,
                pnl_24h=stats.get("pnl_24h", stats.get("total_pnl", 0.0)),
                sources_status=sources,
            )
        except Exception as exc:
            logger.warning(f"daily summary failed: {exc}")

    schedule.every().day.at("23:55").do(_send_daily_summary)

    # Bot-startup notification — single ping, includes the running config.
    try:
        weather_cfg = config.get("weather") or {}
        bot.notifier.notify_bot_startup(
            mode=mode + (" [SHADOW]" if shadow else ""),
            capital=float(config.get("bot", {}).get("demo_capital", 0.0)),
            modules={
                "weather": bool(weather_cfg.get("enabled", True)),
                "crypto_lag": cl_enabled,
                "circuit_breaker": bool(weather_cfg.get("circuit_breaker_enabled")),
            },
            config_summary={
                "cycle_min": interval_min,
                "min_prob": str(weather_cfg.get("min_prob_to_bet_by_days", "")),
                "min_abs_edge": weather_cfg.get("min_abs_edge"),
                "claude_model": (config.get("claude") or {}).get("model"),
            },
        )
    except Exception as exc:
        logger.debug(f"startup notify failed: {exc}")

    logger.info("Running first cycle...")
    bot.run_weather_cycle()

    logger.info("Press Ctrl+C to stop.")
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
        try:
            bot.notifier.notify_bot_shutdown(reason="manual (Ctrl+C)")
        except Exception:
            pass
        try:
            if crypto_lag_handle is not None:
                stop_crypto_lag(crypto_lag_handle)
        except Exception as exc:
            logger.warning(f"stop_crypto_lag: {exc}")
        bot.db.close()
        sys.exit(0)


if __name__ == "__main__":
    main()
