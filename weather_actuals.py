"""
Actuals feedback loop — fetch observed weather at the target_dt of a closed
trade and update per-source reliability (rolling MAE / Brier) based on how
close each source's stored forecast_snapshot was.

Data source: Open-Meteo historical archive API (no key required). We use it as
the "ground truth" for all sources — imperfect (it's itself a reanalysis), but
consistent and cheap. Retail weather markets resolve against the same class of
gridded observation, so Open-Meteo archive is a reasonable proxy.

Rolling update: brier_score and mae are an EWMA with alpha=0.2 — recent
observations weigh more than old ones, which is what we want because source
quality drifts (models get re-tuned, stations go offline).
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import aiohttp

logger = logging.getLogger("polymarket_bot.actuals")

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Map our canonical metric → Open-Meteo hourly variable
_METRIC_TO_VAR = {
    "temperature_c":    "temperature_2m",
    "precipitation_mm": "precipitation",
    "wind_mph":         "wind_speed_10m",
    "wind_kph":         "wind_speed_10m",
    "snow_cm":          "snowfall",
}


async def fetch_actual(lat: float, lon: float, target_dt: datetime, metric: str) -> Optional[float]:
    """Fetch the actual observed value at the given hour. Returns None if
    the archive has no data yet (target_dt in the future or delayed)."""
    var = _METRIC_TO_VAR.get(metric)
    if not var:
        return None
    target_dt = target_dt.astimezone(timezone.utc)
    date_str = target_dt.strftime("%Y-%m-%d")

    # Archive trails real-time by ~2-5 days; if target is recent, fall back
    # to forecast endpoint which keeps a small past window of observations.
    now = datetime.now(timezone.utc)
    use_archive = (now - target_dt).days >= 5
    base = ARCHIVE_URL if use_archive else FORECAST_URL
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": var,
        "timezone": "UTC",
        "start_date": date_str,
        "end_date": date_str,
        "wind_speed_unit": "kmh",
    }
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as s:
            async with s.get(base, params=params) as r:
                if r.status != 200:
                    return None
                data = await r.json()
    except Exception as e:
        logger.debug(f"fetch_actual error: {e}")
        return None

    hours = data.get("hourly", {}).get("time", [])
    values = data.get("hourly", {}).get(var, [])
    target_key = target_dt.strftime("%Y-%m-%dT%H:00")
    for t, v in zip(hours, values):
        if t == target_key and v is not None:
            if metric == "wind_mph":
                return float(v) / 1.609344
            return float(v)
    return None


def update_source_reliability_from_trade(db, trade: dict, alpha: float = 0.2) -> dict:
    """For a closed trade, fetch the actual observation and update each source's
    rolling MAE (EWMA). Returns a summary dict."""
    metric = trade.get("metric") or ""
    lat = trade.get("lat")
    lon = trade.get("lon")
    target_dt_str = trade.get("target_dt") or ""
    trade_id = trade.get("id")
    if not (metric and lat is not None and lon is not None and target_dt_str):
        return {"updated": 0, "reason": "missing_fields"}

    try:
        target_dt = datetime.fromisoformat(target_dt_str.replace("Z", "+00:00"))
    except Exception:
        return {"updated": 0, "reason": "bad_target_dt"}

    actual = asyncio.run(fetch_actual(float(lat), float(lon), target_dt, metric))
    if actual is None:
        return {"updated": 0, "reason": "no_actual_available"}

    snaps = db.get_forecast_snapshots(trade_id, limit=500)
    if not snaps:
        return {"updated": 0, "reason": "no_snapshots", "actual": actual}

    # Deduplicate to the oldest forecast per source (its entry-time prediction)
    first_by_source: dict[str, float] = {}
    for s in sorted(snaps, key=lambda x: x.get("timestamp") or ""):
        src = s.get("source")
        val = s.get("forecast_value")
        if src and val is not None and src not in first_by_source:
            first_by_source[src] = float(val)

    updated = 0
    for src, forecast in first_by_source.items():
        err = abs(forecast - actual)
        # Pull existing row, compute EWMA
        rows = db.get_source_reliability()
        existing = next(
            (r for r in rows if r["source_name"] == src and r["metric"] == metric),
            None,
        )
        if existing and (existing.get("trades_used") or 0) > 0:
            new_mae = (1 - alpha) * float(existing.get("mae") or 0.0) + alpha * err
            # Brier-ish proxy normalised per metric scale (so temp vs precip comparable):
            scale = _metric_scale(metric)
            new_brier = (1 - alpha) * float(existing.get("brier_score") or 0.0) + alpha * min(1.0, err / scale) ** 2
            trades_used = int(existing.get("trades_used") or 0) + 1
        else:
            new_mae = err
            scale = _metric_scale(metric)
            new_brier = min(1.0, err / scale) ** 2
            trades_used = 1

        db.upsert_source_reliability(src, metric, {
            "trades_used": trades_used,
            "mae": new_mae,
            "brier_score": new_brier,
        })
        updated += 1

    return {
        "updated": updated,
        "actual": actual,
        "sources": list(first_by_source.keys()),
    }


def _metric_scale(metric: str) -> float:
    """Typical error scale per metric — used to normalise MAE into a 0-1 Brier-ish
    score so reliability weighting compares temperature sources and wind sources
    on the same axis."""
    return {
        "temperature_c":    10.0,   # 10 °C error = Brier 1.0 (terrible)
        "precipitation_mm": 20.0,
        "wind_mph":         15.0,
        "wind_kph":         25.0,
        "snow_cm":          15.0,
    }.get(metric, 10.0)
