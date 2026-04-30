"""
City-bias correction for weather ensemble forecasts.

Background: ensemble forecasts (GFS/ECMWF/etc) are systematically biased for some
city + season combinations. The 28-Apr-2026 diagnosis identified errors of 4–11°C
in Seoul/Jakarta/Lucknow during spring 2026 — exactly the cluster that produced
the Seoul trade collapse on 29-Apr.

This module exposes a small table of (city, metric, month) → signed bias offset
applied to ensemble.mean BEFORE computing the probability. A positive bias means
"the ensemble overestimates the metric" (subtract it from the mean to correct).

The seed table below is hand-tuned from the historical diagnostic. As more trades
close, `recompute_from_history` rewrites the table from observed (forecast vs
actual) deltas via the `market_resolutions` and `trades` DB tables.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

logger = logging.getLogger("polymarket_bot.city_bias")

# Default location of the persisted bias table. Falls back to in-memory if the
# file is missing or unreadable.
BIAS_TABLE_PATH = Path("data/city_bias.json")

# Hand-tuned seeds from the 28-Apr diagnostic. All values are in source units
# (temperature_c → °C, precipitation_mm → mm, etc). Months are 1–12.
# Format: {(city_lower, metric, month): signed_bias}
SEED_BIAS: dict[tuple[str, str, int], float] = {
    # Seoul — spring 2026 diagnostic: ensemble overestimated ~5 °C
    ("seoul",       "temperature_c", 4): 5.0,
    ("seoul",       "temperature_c", 5): 5.0,
    ("seoul",       "temperature_c", 6): 4.0,
    # Jakarta — diagnostic: ~6 °C overestimate
    ("jakarta",     "temperature_c", 4): 6.0,
    ("jakarta",     "temperature_c", 5): 6.0,
    ("jakarta",     "temperature_c", 6): 5.0,
    # Lucknow — diagnostic: ~8 °C overestimate (peak heat-bias)
    ("lucknow",     "temperature_c", 4): 8.0,
    ("lucknow",     "temperature_c", 5): 11.0,
    ("lucknow",     "temperature_c", 6): 9.0,
}

# Minimum number of resolved trades for a (city, metric, month) cell before
# `recompute_from_history` overwrites the seed value. Below this we trust the
# hand-tuned seed.
MIN_SAMPLES_FOR_OVERWRITE = 5


@dataclass(frozen=True)
class CityBiasKey:
    city: str
    metric: str
    month: int

    @classmethod
    def from_inputs(cls, city: str, metric: str, target_month: int) -> "CityBiasKey":
        return cls(city=city.strip().lower(), metric=metric, month=int(target_month))


class CityBiasTable:
    """Loads, queries, and persists the bias table.

    Always loads SEED_BIAS first, then overlays values from disk if present.
    Query is O(1) dict lookup. Missing keys return 0.0 (no bias correction).
    """

    def __init__(self, path: Path = BIAS_TABLE_PATH):
        self.path = path
        self._table: dict[tuple[str, str, int], float] = dict(SEED_BIAS)
        self._load_disk_overlay()

    def _load_disk_overlay(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            logger.warning(f"city_bias: could not read {self.path}: {e}")
            return
        for entry in raw.get("entries", []):
            try:
                k = (str(entry["city"]).lower(), str(entry["metric"]), int(entry["month"]))
                self._table[k] = float(entry["bias"])
            except (KeyError, ValueError, TypeError):
                continue
        logger.info(f"city_bias: loaded {len(self._table)} entries (seed + disk)")

    def get(self, city: str, metric: str, target_month: int) -> float:
        return self._table.get(
            (city.strip().lower(), metric, int(target_month)),
            0.0,
        )

    def adjust_mean(
        self, ensemble_mean: float, city: str, metric: str, target_month: int
    ) -> tuple[float, float]:
        """Return (adjusted_mean, bias_applied). Positive bias = overestimate → subtract."""
        bias = self.get(city, metric, target_month)
        return ensemble_mean - bias, bias

    def save(self, entries: list[dict]) -> None:
        """Persist a recomputed table (overwrites disk file)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"entries": entries}
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info(f"city_bias: saved {len(entries)} entries → {self.path}")


# Singleton accessor — cheap to import in hot path
_DEFAULT_TABLE: Optional[CityBiasTable] = None


def get_default_table() -> CityBiasTable:
    global _DEFAULT_TABLE
    if _DEFAULT_TABLE is None:
        _DEFAULT_TABLE = CityBiasTable()
    return _DEFAULT_TABLE


def reload_default_table() -> CityBiasTable:
    """Force-reload from disk (use after recompute_from_history)."""
    global _DEFAULT_TABLE
    _DEFAULT_TABLE = CityBiasTable()
    return _DEFAULT_TABLE


# ─────────────────────────────────────────────────────────────────
# Recompute from history
# ─────────────────────────────────────────────────────────────────
def recompute_from_history(db, write: bool = True) -> list[dict]:
    """Recalculate bias entries from `market_resolutions` joined with `trades`.

    For each closed trade with a recorded ensemble_mean and a resolved actual
    value, compute (ensemble_mean - actual_value) and average per (city, metric,
    month). Only cells with ≥ MIN_SAMPLES_FOR_OVERWRITE samples replace seeds.

    Returns the entries list (also written to disk if `write=True`).
    """
    c = db.conn.cursor()
    rows = c.execute(
        """
        SELECT t.city, t.metric, t.target_date, t.ensemble_mean,
               r.actual_value
        FROM trades t
        LEFT JOIN market_resolutions r ON r.market_id = t.market_id
        WHERE t.ensemble_mean IS NOT NULL
          AND r.actual_value IS NOT NULL
          AND t.city IS NOT NULL
          AND t.metric IS NOT NULL
        """
    ).fetchall()

    buckets: dict[tuple[str, str, int], list[float]] = {}
    for r in rows:
        city = (r["city"] or "").strip().lower()
        metric = r["metric"]
        target_date = r["target_date"]
        if not city or not metric or not target_date:
            continue
        try:
            month = date.fromisoformat(str(target_date)[:10]).month
        except ValueError:
            continue
        delta = float(r["ensemble_mean"]) - float(r["actual_value"])
        buckets.setdefault((city, metric, month), []).append(delta)

    # Start from seeds, overwrite cells with enough samples
    final = dict(SEED_BIAS)
    for k, deltas in buckets.items():
        if len(deltas) < MIN_SAMPLES_FOR_OVERWRITE:
            continue
        final[k] = sum(deltas) / len(deltas)

    entries = [
        {"city": k[0], "metric": k[1], "month": k[2],
         "bias": round(v, 3),
         "samples": len(buckets.get(k, []))}
        for k, v in sorted(final.items())
    ]

    if write:
        get_default_table().save(entries)
        reload_default_table()

    return entries
