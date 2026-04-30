"""
Reports scheduler — generates reports/analysis_YYYYMMDD_HHMM.md every N minutes.

Called once per cycle (cheap no-op if less than interval minutes have elapsed).
Uses log_analyzer's helpers to build the Markdown report without shelling out.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import log_analyzer as la

logger = logging.getLogger("polymarket_bot.reports")


def maybe_generate_report(config: dict, db, last_ts: float) -> float:
    """
    Generate a report if `reports.interval_minutes` has elapsed since `last_ts`.
    Returns the new `last_ts` (epoch seconds). If generation is disabled or not
    due yet, returns `last_ts` unchanged.
    """
    rc = config.get("reports", {}) or {}
    if not rc.get("enabled", True):
        return last_ts
    interval_min = int(rc.get("interval_minutes", 60))
    now = time.time()
    if last_ts and (now - last_ts) < interval_min * 60:
        return last_ts

    try:
        out_dir = Path(rc.get("output_dir", "reports"))
        out_dir.mkdir(parents=True, exist_ok=True)
        db_path = config.get("database", {}).get("path", "data/bot.db")
        events_path = config.get("logging", {}).get("events_file", "logs/events.jsonl")
        last_hours = int(rc.get("window_hours", 24))

        cutoff = la._iso_cutoff(last_hours)
        cutoff_dt = datetime.fromisoformat(cutoff)
        conn = la._connect(db_path)
        try:
            data = {
                "global": la.global_pnl(conn, cutoff),
                "by_side": la.pnl_by_side(conn, cutoff),
                "by_hour": la.pnl_by_hour(conn, cutoff),
                "top_losers": la.top_losing_markets(conn, cutoff),
                "skip_reasons": la.top_skip_reasons(conn, cutoff),
                "sources": la.source_reliability(conn),
                "calibration": la.calibration_buckets(conn, cutoff),
                "confidence_dist": la.confidence_distribution(conn, cutoff),
                "events": la.events_summary(events_path, cutoff_dt),
            }
        finally:
            conn.close()

        md = la.render_markdown(last_hours, data)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        out_path = out_dir / f"analysis_{stamp}.md"
        out_path.write_text(md, encoding="utf-8")
        logger.info(f"Report written: {out_path}")
        return now
    except Exception as e:
        logger.warning(f"Report generation failed: {e}")
        return now  # still update ts to avoid tight-loop retries
