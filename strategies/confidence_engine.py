"""
Confidence engine v3 — SOLO vetos duros, sin score compuesto.

El criterio principal de entrada es prob_real ≥ min_prob_to_bet_by_days (weather_bot).
Este módulo únicamente aplica vetos técnicos binarios que invalidan la estimación:
  * FEW_SOURCES               — < min_sources_required fuentes respondieron
  * HIGH_ENSEMBLE_STD         — std > umbral adaptativo por horizonte
  * EDGE_SUSPICIOUS           — |prob_real - prob_market| > max_edge_pts (posible bug)
  * TOO_CLOSE_TO_RESOLUTION   — horas < min_time_to_resolution_hours
  * NO_ENSEMBLE_DATA          — ensemble sin datos

risk_score = round(prob_real * 100). Solo se expone para display, no filtra.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ConfidenceBreakdown:
    final_score: float = 0.0       # = round(prob_real * 100) — display only
    prob_real: float = 0.0
    risk_score: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    vetos: list[str] = field(default_factory=list)
    passed: bool = False

    def to_dict(self) -> dict:
        return {
            "final_score": round(self.final_score, 1),
            "prob_real": round(self.prob_real, 4),
            "risk_score": round(self.risk_score, 1),
            "components": {k: round(v, 3) for k, v in self.components.items()},
            "vetos": self.vetos,
            "passed": self.passed,
        }


def _std_max_for_days(std_max_by_days: dict, metric: str, days_ahead: int, fallback: dict) -> Optional[float]:
    """Resolve std_max for given metric and horizon bucket (1-3, 4-5, 6-7)."""
    if days_ahead <= 3:
        bucket_key = "1-3"
    elif days_ahead <= 5:
        bucket_key = "4-5"
    else:
        bucket_key = "6-7"
    bucket = (std_max_by_days or {}).get(bucket_key) or {}
    return bucket.get(metric) or (fallback or {}).get(metric)


class ConfidenceEngine:
    def __init__(self, config: dict):
        conf = config.get("confidence", {})
        self.min_sources_required = conf.get("min_sources_required", 3)
        self.max_edge_pts = conf.get("max_edge_pts", 40)
        self.min_time_to_resolution_hours = conf.get("min_time_to_resolution_hours", 1)

        weather = config.get("weather", {})
        self.std_max_by_days = weather.get("ensemble_std_max_by_days") or {}
        self.std_max_fallback = weather.get("ensemble_std_max") or {
            "temperature_c": 5.0, "precipitation_mm": 12.0,
            "wind_mph": 10.0, "snow_cm": 8.0,
        }

    def evaluate(
        self,
        ensemble_result,
        prob_estimated: float,
        prob_market: float,
        liquidity: float = 0.0,
        volume_24h: float = 0.0,
        hours_to_resolution: float = 0.0,
        side: str = "YES",
        side_rolling_wr: Optional[float] = None,
        is_pyramid: bool = False,
        days_ahead: int = 0,
    ) -> ConfidenceBreakdown:
        br = ConfidenceBreakdown()
        br.prob_real = float(prob_estimated or 0.0)
        br.risk_score = round(br.prob_real * 100, 1)
        br.final_score = br.risk_score

        metric = ensemble_result.metric if ensemble_result else None

        if ensemble_result is None or ensemble_result.mean is None:
            br.vetos.append("NO_ENSEMBLE_DATA")
        else:
            if ensemble_result.n_used < self.min_sources_required:
                br.vetos.append("FEW_SOURCES")
            std_max = _std_max_for_days(self.std_max_by_days, metric, days_ahead, self.std_max_fallback)
            if std_max is not None and ensemble_result.std is not None and ensemble_result.std > std_max:
                br.vetos.append("HIGH_ENSEMBLE_STD")

        edge_pts = (br.prob_real - float(prob_market or 0.0)) * 100.0
        if abs(edge_pts) > self.max_edge_pts:
            br.vetos.append("EDGE_SUSPICIOUS")
        if hours_to_resolution < self.min_time_to_resolution_hours:
            br.vetos.append("TOO_CLOSE_TO_RESOLUTION")

        br.components = {
            "prob_real": br.prob_real,
            "prob_market": float(prob_market or 0.0),
            "edge_pts": round(edge_pts, 2),
            "ensemble_std": (ensemble_result.std if ensemble_result else None),
            "sources_used": (ensemble_result.n_used if ensemble_result else 0),
            "hours_to_resolution": hours_to_resolution,
            "days_ahead": days_ahead,
        }

        br.passed = not br.vetos
        return br
