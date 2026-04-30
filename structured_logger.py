"""
Structured Event Logger — writes JSONL records to logs/events.jsonl.

Weather Bot v2. Every event has:
  {
    "timestamp":  ISO-8601 UTC with milliseconds,
    "trace_id":   short UUID — correlates related events (e.g. open/close of same trade),
    "cycle_id":   integer ciclo actual (None si no aplica),
    "type":       event_type (ver Event enum abajo),
    "module":     "weather" | "system" | "dashboard",
    "severity":   "DEBUG" | "INFO" | "WARN" | "ERROR",
    "data":       { ...payload }
  }

log_analyzer.py lee este fichero para generar reportes.
"""

import json
import logging
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("polymarket_bot.structured_logger")

LOG_DIR = Path("logs")
EVENTS_FILE = LOG_DIR / "events.jsonl"


# ============================================================
# Event types (enum-like constants)
# ============================================================
class Event:
    SYSTEM = "SYSTEM"
    PROCESS_LOCK = "PROCESS_LOCK"
    CYCLE_START = "CYCLE_START"
    CYCLE_END = "CYCLE_END"
    MARKETS_SCANNED = "MARKETS_SCANNED"
    ENSEMBLE_FORECAST = "ENSEMBLE_FORECAST"
    SOURCE_CALL = "SOURCE_CALL"
    SOURCE_FAILED = "SOURCE_FAILED"
    CONFIDENCE_EVAL = "CONFIDENCE_EVAL"
    OPPORTUNITY_FOUND = "OPPORTUNITY_FOUND"
    OPPORTUNITY_SKIP = "OPPORTUNITY_SKIP"
    CLAUDE_DECISION = "CLAUDE_DECISION"
    TRADE_OPEN = "TRADE_OPEN"
    TRADE_CLOSE = "TRADE_CLOSE"
    TRADE_SHADOW = "TRADE_SHADOW"           # would-be trade in shadow mode
    MARKET_RESOLVED = "MARKET_RESOLVED"
    SOURCE_RELIABILITY_UPDATE = "SOURCE_RELIABILITY_UPDATE"
    SIDE_WR_UPDATE = "SIDE_WR_UPDATE"
    PYRAMID_EVAL = "PYRAMID_EVAL"
    # v3 re-evaluation loop
    BET_REEVALUATED = "BET_REEVALUATED"
    BET_CLOSED_BY_RISK = "BET_CLOSED_BY_RISK"
    BET_AVERAGED_UP = "BET_AVERAGED_UP"
    POSITION_MARK = "POSITION_MARK"
    RESET_ON_START = "RESET_ON_START"
    CANCEL_ORDER_ATTEMPT = "CANCEL_ORDER_ATTEMPT"
    CANCEL_ORDER_RESULT = "CANCEL_ORDER_RESULT"
    SELL_POSITION_ATTEMPT = "SELL_POSITION_ATTEMPT"
    SELL_POSITION_RESULT = "SELL_POSITION_RESULT"
    ERROR = "ERROR"
    WARNING = "WARNING"


# ============================================================
# Skip reason codes (used by OPPORTUNITY_SKIP + DB.opportunity_skips.reason_code)
# ============================================================
class SkipReason:
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    HIGH_ENSEMBLE_STD = "HIGH_ENSEMBLE_STD"
    FEW_SOURCES = "FEW_SOURCES"
    EDGE_SUSPICIOUS = "EDGE_SUSPICIOUS"
    TOO_CLOSE_TO_RESOLUTION = "TOO_CLOSE_TO_RESOLUTION"
    TOO_FAR_FROM_RESOLUTION = "TOO_FAR_FROM_RESOLUTION"
    SIDE_WR_VETO = "SIDE_WR_VETO"
    OUTSIDE_TRADING_WINDOW = "OUTSIDE_TRADING_WINDOW"
    MARKET_EXPOSURE_LIMIT = "MARKET_EXPOSURE_LIMIT"
    TOTAL_EXPOSURE_LIMIT = "TOTAL_EXPOSURE_LIMIT"
    MAX_CONCURRENT = "MAX_CONCURRENT"
    MAX_TRADES_PER_CYCLE = "MAX_TRADES_PER_CYCLE"
    INSUFFICIENT_LIQUIDITY = "INSUFFICIENT_LIQUIDITY"
    PYRAMID_REJECTED = "PYRAMID_REJECTED"
    EV_BELOW_THRESHOLD = "EV_BELOW_THRESHOLD"
    CLAUDE_REJECTED = "CLAUDE_REJECTED"
    NO_ENSEMBLE_DATA = "NO_ENSEMBLE_DATA"
    LOW_LIQUIDITY = "LOW_LIQUIDITY"
    LOW_VOLUME = "LOW_VOLUME"
    NEAR_RESOLVED = "NEAR_RESOLVED"
    PARSE_FAILED = "PARSE_FAILED"
    PAST_DATE = "PAST_DATE"
    TOO_FAR_AHEAD = "TOO_FAR_AHEAD"
    UNSUPPORTED_METRIC = "UNSUPPORTED_METRIC"
    LOW_EDGE = "LOW_EDGE"
    RECENT_SL_SAME_MARKET = "RECENT_SL_SAME_MARKET"
    PRIOR_RISKY_VETO = "PRIOR_RISKY_VETO"


# ============================================================
# Logger
# ============================================================
class StructuredLogger:
    def __init__(self, log_file: Optional[Path] = None):
        self.log_file = log_file or EVENTS_FILE
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._current_cycle_id: Optional[int] = None

    def set_cycle_id(self, cycle_id: Optional[int]):
        self._current_cycle_id = cycle_id

    @staticmethod
    def new_trace_id() -> str:
        return uuid.uuid4().hex[:10]

    # ------------------------------------------------------------------
    def log(
        self,
        event_type: str,
        module: str,
        data: dict,
        severity: str = "INFO",
        trace_id: Optional[str] = None,
        cycle_id: Optional[int] = None,
    ) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "trace_id": trace_id,
            "cycle_id": cycle_id if cycle_id is not None else self._current_cycle_id,
            "type": event_type,
            "module": module,
            "severity": severity,
            "data": data,
        }
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            logger.error(f"StructuredLogger write failed: {e}")

    # ------------------------------------------------------------------
    # Cycle
    # ------------------------------------------------------------------
    def log_cycle_start(self, cycle_id: int, mode: str):
        self.set_cycle_id(cycle_id)
        self.log(Event.CYCLE_START, "weather", {"cycle_id": cycle_id, "mode": mode})

    def log_cycle_end(self, cycle_id: int, summary: dict):
        self.log(Event.CYCLE_END, "weather", {"cycle_id": cycle_id, **summary})

    # ------------------------------------------------------------------
    # Ensemble / sources
    # ------------------------------------------------------------------
    def log_ensemble_forecast(
        self,
        trace_id: str,
        market_id: str,
        metric: str,
        mean: float,
        std: float,
        sources_used: list,
        sources_failed: list,
        latency_ms: int,
    ):
        self.log(
            Event.ENSEMBLE_FORECAST,
            "weather",
            {
                "market_id": market_id,
                "metric": metric,
                "mean": round(mean, 3) if mean is not None else None,
                "std": round(std, 3) if std is not None else None,
                "sources_used": sources_used,
                "sources_failed": sources_failed,
                "latency_ms": latency_ms,
            },
            trace_id=trace_id,
        )

    def log_source_call(self, source: str, latency_ms: int, success: bool, error: Optional[str] = None):
        self.log(
            Event.SOURCE_CALL if success else Event.SOURCE_FAILED,
            "weather",
            {"source": source, "latency_ms": latency_ms, "success": success, "error": error},
            severity="INFO" if success else "WARN",
        )

    # ------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------
    def log_confidence_eval(
        self,
        trace_id: str,
        market_id: str,
        score: float,
        breakdown: dict,
        vetos: list,
        passed: bool,
    ):
        self.log(
            Event.CONFIDENCE_EVAL,
            "weather",
            {
                "market_id": market_id,
                "score": round(score, 1),
                "breakdown": breakdown,
                "vetos": vetos,
                "passed": passed,
            },
            trace_id=trace_id,
        )

    # ------------------------------------------------------------------
    # Opportunities
    # ------------------------------------------------------------------
    def log_opportunity_skip(
        self,
        market_id: str,
        market_question: str,
        reason_code: str,
        reason_detail: str = "",
        confidence_score: Optional[float] = None,
        ensemble_std: Optional[float] = None,
        sources_used: Optional[int] = None,
        prob_market: Optional[float] = None,
        prob_estimated: Optional[float] = None,
        edge: Optional[float] = None,
        trace_id: Optional[str] = None,
    ):
        self.log(
            Event.OPPORTUNITY_SKIP,
            "weather",
            {
                "market_id": market_id,
                "question": (market_question or "")[:120],
                "reason_code": reason_code,
                "reason_detail": reason_detail,
                "confidence_score": confidence_score,
                "ensemble_std": ensemble_std,
                "sources_used": sources_used,
                "prob_market": prob_market,
                "prob_estimated": prob_estimated,
                "edge": edge,
            },
            trace_id=trace_id,
        )

    def log_claude_decision(
        self,
        decision: str,
        reasoning: str,
        trades_approved: int = 0,
        trades_rejected: int = 0,
        latency_seconds: Optional[float] = None,
        trace_id: Optional[str] = None,
    ):
        self.log(
            Event.CLAUDE_DECISION,
            "weather",
            {
                "decision": decision,
                "reasoning": (reasoning or "")[:500],
                "trades_approved": trades_approved,
                "trades_rejected": trades_rejected,
                "latency_seconds": round(latency_seconds, 2) if latency_seconds else None,
            },
            trace_id=trace_id,
        )

    # ------------------------------------------------------------------
    # Trades
    # ------------------------------------------------------------------
    def log_trade_open(
        self,
        trace_id: str,
        trade_id: Any,
        market_id: str,
        market_question: str,
        side: str,
        price: float,
        size_usdc: float,
        confidence_score: float,
        ensemble_mean: Optional[float] = None,
        ensemble_std: Optional[float] = None,
        sources_used: Optional[int] = None,
        hours_to_resolution: Optional[float] = None,
        is_shadow: bool = False,
    ):
        self.log(
            Event.TRADE_SHADOW if is_shadow else Event.TRADE_OPEN,
            "weather",
            {
                "trade_id": trade_id,
                "market_id": market_id,
                "question": (market_question or "")[:120],
                "side": side,
                "price": round(price, 4),
                "size_usdc": round(size_usdc, 4),
                "confidence_score": round(confidence_score, 1),
                "ensemble_mean": round(ensemble_mean, 3) if ensemble_mean is not None else None,
                "ensemble_std": round(ensemble_std, 3) if ensemble_std is not None else None,
                "sources_used": sources_used,
                "hours_to_resolution": round(hours_to_resolution, 2) if hours_to_resolution is not None else None,
                "shadow": is_shadow,
            },
            trace_id=trace_id,
        )

    def log_trade_close(
        self,
        trace_id: Optional[str],
        trade_id: Any,
        market_id: str,
        side: str,
        entry_price: float,
        exit_price: float,
        pnl_usdc: float,
        pnl_pct: float,
        close_reason: str,
        duration_hours: Optional[float] = None,
    ):
        self.log(
            Event.TRADE_CLOSE,
            "weather",
            {
                "trade_id": trade_id,
                "market_id": market_id,
                "side": side,
                "entry_price": round(entry_price, 4),
                "exit_price": round(exit_price, 4),
                "pnl_usdc": round(pnl_usdc, 4),
                "pnl_pct": round(pnl_pct, 4),
                "close_reason": close_reason,
                "outcome": "WIN" if pnl_usdc > 0 else ("LOSS" if pnl_usdc < 0 else "BREAK_EVEN"),
                "duration_hours": round(duration_hours, 3) if duration_hours else None,
            },
            trace_id=trace_id,
        )

    def log_market_resolved(self, market_id: str, outcome: str, actual_value: Optional[float] = None):
        self.log(
            Event.MARKET_RESOLVED,
            "weather",
            {"market_id": market_id, "outcome": outcome, "actual_value": actual_value},
        )

    def log_source_reliability_update(self, reliability_rows: list):
        self.log(
            Event.SOURCE_RELIABILITY_UPDATE,
            "weather",
            {"sources": reliability_rows},
        )

    def log_side_wr_update(self, yes_wr: float, no_wr: float, sample_size: int):
        self.log(
            Event.SIDE_WR_UPDATE,
            "weather",
            {"yes_wr": round(yes_wr, 3), "no_wr": round(no_wr, 3), "n": sample_size},
        )

    def log_pyramid_eval(self, trace_id: str, market_id: str, allowed: bool, reason: str, extra: Optional[dict] = None):
        data = {"market_id": market_id, "allowed": allowed, "reason": reason}
        if extra:
            data.update(extra)
        self.log(Event.PYRAMID_EVAL, "weather", data, trace_id=trace_id)

    # ------------------------------------------------------------------
    # Errors / system
    # ------------------------------------------------------------------
    def log_error(self, module: str, error_type: str, message: str, exc: Optional[Exception] = None, context: Optional[dict] = None):
        data = {"error_type": error_type, "message": message}
        if exc is not None:
            data["traceback"] = traceback.format_exc()[-1000:]
        if context:
            data["context"] = context
        self.log(Event.ERROR, module, data, severity="ERROR")

    def log_warning(self, module: str, message: str, context: Optional[dict] = None):
        data = {"message": message}
        if context:
            data["context"] = context
        self.log(Event.WARNING, module, data, severity="WARN")

    def log_system(self, event: str, details: Optional[dict] = None):
        self.log(Event.SYSTEM, "system", {"event": event, **(details or {})})

    def log_process_lock(self, event: str, pid: int, details: Optional[dict] = None):
        self.log(
            Event.PROCESS_LOCK,
            "system",
            {"event": event, "pid": pid, **(details or {})},
            severity="WARN" if event != "acquired" else "INFO",
        )

    # ------------------------------------------------------------------
    # v3 — Re-evaluation, risk-based close, averaging up, mark-to-market
    # ------------------------------------------------------------------
    def log_bet_reevaluated(
        self,
        trace_id: Optional[str],
        trade_id: Any,
        market_id: str,
        prob_real: float,
        prob_real_entry: float,
        risk_score: float,
        action: str,
        vetos: Optional[list] = None,
        price_market: Optional[float] = None,
        unrealized_pnl: Optional[float] = None,
        hours_to_resolution: Optional[float] = None,
        notes: str = "",
    ):
        self.log(
            Event.BET_REEVALUATED,
            "weather",
            {
                "trade_id": trade_id,
                "market_id": market_id,
                "prob_real": round(prob_real, 4),
                "prob_real_entry": round(prob_real_entry, 4),
                "prob_delta": round(prob_real - prob_real_entry, 4),
                "risk_score": round(risk_score, 1),
                "action": action,
                "vetos": vetos or [],
                "price_market": round(price_market, 4) if price_market is not None else None,
                "unrealized_pnl": round(unrealized_pnl, 4) if unrealized_pnl is not None else None,
                "hours_to_resolution": round(hours_to_resolution, 2) if hours_to_resolution is not None else None,
                "notes": notes[:300],
            },
            trace_id=trace_id,
        )

    def log_bet_closed_by_risk(
        self,
        trace_id: Optional[str],
        trade_id: Any,
        market_id: str,
        reason: str,
        prob_real_entry: float,
        prob_real_now: float,
        exit_price: float,
        pnl_usdc: float,
        pnl_pct: float,
    ):
        self.log(
            Event.BET_CLOSED_BY_RISK,
            "weather",
            {
                "trade_id": trade_id,
                "market_id": market_id,
                "reason": reason,
                "prob_real_entry": round(prob_real_entry, 4),
                "prob_real_now": round(prob_real_now, 4),
                "exit_price": round(exit_price, 4),
                "pnl_usdc": round(pnl_usdc, 4),
                "pnl_pct": round(pnl_pct, 4),
            },
            trace_id=trace_id,
            severity="WARN",
        )

    def log_bet_averaged_up(
        self,
        trace_id: Optional[str],
        parent_trade_id: Any,
        new_trade_id: Any,
        market_id: str,
        side: str,
        add_price: float,
        add_size_usdc: float,
        prob_real_entry: float,
        prob_real_now: float,
    ):
        self.log(
            Event.BET_AVERAGED_UP,
            "weather",
            {
                "parent_trade_id": parent_trade_id,
                "new_trade_id": new_trade_id,
                "market_id": market_id,
                "side": side,
                "add_price": round(add_price, 4),
                "add_size_usdc": round(add_size_usdc, 4),
                "prob_real_entry": round(prob_real_entry, 4),
                "prob_real_now": round(prob_real_now, 4),
                "prob_delta": round(prob_real_now - prob_real_entry, 4),
            },
            trace_id=trace_id,
        )

    def log_position_mark(
        self,
        trade_id: Any,
        market_id: str,
        best_bid: Optional[float],
        best_ask: Optional[float],
        unrealized_pnl: float,
        value_if_closed_now: float,
        trace_id: Optional[str] = None,
    ):
        self.log(
            Event.POSITION_MARK,
            "weather",
            {
                "trade_id": trade_id,
                "market_id": market_id,
                "best_bid": round(best_bid, 4) if best_bid is not None else None,
                "best_ask": round(best_ask, 4) if best_ask is not None else None,
                "unrealized_pnl": round(unrealized_pnl, 4),
                "value_if_closed_now": round(value_if_closed_now, 4),
            },
            trace_id=trace_id,
            severity="DEBUG",
        )

    def log_reset_on_start(self, archived_to: Optional[str], wiped_tables: list, kept_rules: bool):
        self.log(
            Event.RESET_ON_START,
            "system",
            {
                "archived_to": archived_to,
                "wiped_tables": wiped_tables,
                "kept_rules": kept_rules,
            },
        )

    def log_cancel_order_attempt(self, order_id: str, market_id: str, mode: str):
        self.log(
            Event.CANCEL_ORDER_ATTEMPT,
            "weather",
            {"order_id": order_id, "market_id": market_id, "mode": mode},
        )

    def log_cancel_order_result(self, order_id: str, market_id: str, success: bool, error: Optional[str] = None):
        self.log(
            Event.CANCEL_ORDER_RESULT,
            "weather",
            {"order_id": order_id, "market_id": market_id, "success": success, "error": error},
            severity="INFO" if success else "WARN",
        )

    def log_sell_position_attempt(
        self,
        trade_id: Any,
        market_id: str,
        shares: float,
        min_price: float,
        mode: str,
    ):
        self.log(
            Event.SELL_POSITION_ATTEMPT,
            "weather",
            {
                "trade_id": trade_id,
                "market_id": market_id,
                "shares": round(shares, 4),
                "min_price": round(min_price, 4),
                "mode": mode,
            },
        )

    def log_sell_position_result(
        self,
        trade_id: Any,
        market_id: str,
        shares_sold: float,
        fill_price: float,
        gross_usdc: float,
        fees_usdc: float,
        success: bool,
        error: Optional[str] = None,
    ):
        self.log(
            Event.SELL_POSITION_RESULT,
            "weather",
            {
                "trade_id": trade_id,
                "market_id": market_id,
                "shares_sold": round(shares_sold, 4),
                "fill_price": round(fill_price, 4),
                "gross_usdc": round(gross_usdc, 4),
                "fees_usdc": round(fees_usdc, 4),
                "net_usdc": round(gross_usdc - fees_usdc, 4),
                "success": success,
                "error": error,
            },
            severity="INFO" if success else "WARN",
        )


# ============================================================
# Module-level singleton
# ============================================================
_instance: Optional[StructuredLogger] = None


def get_logger() -> StructuredLogger:
    global _instance
    if _instance is None:
        _instance = StructuredLogger()
    return _instance
