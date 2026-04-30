"""
State reset on startup — wipes DB runtime tables and archives prior logs.

Triggered by config `bot.reset_on_start: true`. Keeps the last N archived runs
under `logs/archive/run_<timestamp>/` (config `bot.keep_last_n_runs`).

Tables wiped: trades, cycles, trade_analyses, opportunity_skips,
              bet_evaluations, forecast_snapshots, weather_markets,
              market_resolutions.
Tables preserved by default: learned_rules, parameter_adjustments,
              source_reliability (can be opted-in via `bot.reset_rules: true`).
"""
from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from structured_logger import get_logger

logger = logging.getLogger("polymarket_bot.state_reset")


LOG_DIR = Path("logs")
ARCHIVE_DIR = LOG_DIR / "archive"


def _archive_logs(keep_last_n: int) -> Optional[str]:
    """Move active log files into logs/archive/run_<ts>/ and prune to last N."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = ARCHIVE_DIR / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    moved_any = False
    for item in LOG_DIR.iterdir():
        if item.is_file() and (item.name.startswith("bot_") or item.name == "events.jsonl"):
            try:
                shutil.move(str(item), str(run_dir / item.name))
                moved_any = True
            except Exception as e:
                logger.warning(f"Could not archive {item.name}: {e}")

    if not moved_any:
        try:
            run_dir.rmdir()
        except OSError:
            pass
        run_dir = None

    _prune_old_runs(keep_last_n)
    return str(run_dir) if run_dir else None


def _prune_old_runs(keep_last_n: int) -> None:
    if keep_last_n <= 0:
        return
    runs = sorted(
        (p for p in ARCHIVE_DIR.iterdir() if p.is_dir() and p.name.startswith("run_")),
        key=lambda p: p.name,
        reverse=True,
    )
    for old in runs[keep_last_n:]:
        try:
            shutil.rmtree(old, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Could not prune {old}: {e}")


def reset_state(db, config: dict) -> dict:
    """
    Perform the full reset pipeline. Returns summary dict for logging.

    Args:
        db: Database instance with `wipe_runtime_tables(reset_rules)` method.
        config: loaded config dict.
    """
    bot_conf = config.get("bot", {})
    keep_last_n = int(bot_conf.get("keep_last_n_runs", 3))
    reset_rules = bool(bot_conf.get("reset_rules", False))

    archived_to = _archive_logs(keep_last_n)

    wiped_counts: dict = {}
    try:
        result = db.wipe_runtime_tables(reset_rules=reset_rules)
        if isinstance(result, dict):
            wiped_counts = result
        elif isinstance(result, list):
            wiped_counts = {t: 0 for t in result}
    except Exception as e:
        logger.error(f"DB wipe failed: {e}")

    wiped_tables = list(wiped_counts.keys())
    summary = {
        "archived_to": archived_to,
        "wiped_tables": wiped_tables,
        "wiped_counts": wiped_counts,
        "kept_rules": not reset_rules,
    }

    try:
        get_logger().log_reset_on_start(
            archived_to=archived_to,
            wiped_tables=wiped_tables,
            kept_rules=not reset_rules,
        )
    except Exception as e:
        logger.debug(f"Could not emit RESET_ON_START event: {e}")

    logger.info(
        f"State reset complete — archived={archived_to}, wiped={wiped_counts}, kept_rules={not reset_rules}"
    )
    return summary
