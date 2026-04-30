"""
Weather ensemble: run N sources in parallel, compute mean/std, probability.

Designed so that:
 - slow/failing sources don't block the rest (asyncio.gather with return_exceptions)
 - outliers >3 std from group median are filtered before computing final stats
 - sources are weighted by recent reliability (Brier score) when available
 - probability of crossing a threshold is computed via normal CDF with ensemble mean/std
"""

from __future__ import annotations
import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean, pstdev, median
from typing import Optional

try:
    from scipy.stats import norm
    _HAS_SCIPY = True
except ImportError:  # graceful degradation
    import math
    _HAS_SCIPY = False

    class _NormFallback:
        @staticmethod
        def cdf(x):
            return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))

    norm = _NormFallback()

from .weather_sources import WeatherSource
from structured_logger import get_logger

logger = logging.getLogger("polymarket_bot.weather_ensemble")


@dataclass
class EnsembleResult:
    metric: str
    mean: Optional[float]
    std: Optional[float]
    median: Optional[float]
    sources_used: list[str] = field(default_factory=list)
    sources_failed: list[str] = field(default_factory=list)
    per_source: dict[str, float] = field(default_factory=dict)
    latency_ms: int = 0

    @property
    def n_used(self) -> int:
        return len(self.sources_used)

    def prob_over_threshold(self, threshold: float) -> Optional[float]:
        """P(ensemble value > threshold) under Normal(mean, std) assumption."""
        if self.mean is None or self.std is None or self.std <= 0:
            return None
        z = (threshold - self.mean) / self.std
        return float(1.0 - norm.cdf(z))

    def prob_in_range(self, low: float, high: float) -> Optional[float]:
        if self.mean is None or self.std is None or self.std <= 0:
            return None
        zh = (high - self.mean) / self.std
        zl = (low - self.mean) / self.std
        return float(norm.cdf(zh) - norm.cdf(zl))


# ============================================================
async def ensemble_forecast(
    sources: list[WeatherSource],
    lat: float,
    lon: float,
    target_dt: datetime,
    metric: str,
    trace_id: Optional[str] = None,
    market_id: Optional[str] = None,
    source_weights: Optional[dict[str, float]] = None,
    outlier_zscore: float = 3.0,
) -> EnsembleResult:
    """
    Fetch `metric` from all sources in parallel, return combined ensemble.

    `source_weights` is a {source_name: weight} dict from DB.source_reliability
    (weight = 1 / (1 + brier)). If None, uniform weights.
    """
    t0 = time.time()

    async def _call(src: WeatherSource) -> tuple[str, Optional[float]]:
        val = await src.forecast(lat, lon, target_dt, metric)
        return src.name, val

    results = await asyncio.gather(*[_call(s) for s in sources], return_exceptions=True)

    per_source: dict[str, float] = {}
    sources_used: list[str] = []
    sources_failed: list[str] = []

    for res in results:
        if isinstance(res, Exception):
            continue
        name, value = res
        if value is None:
            sources_failed.append(name)
            continue
        per_source[name] = float(value)
        sources_used.append(name)

    if not per_source:
        out = EnsembleResult(metric=metric, mean=None, std=None, median=None,
                             sources_used=[], sources_failed=sources_failed,
                             per_source={}, latency_ms=int((time.time() - t0) * 1000))
        get_logger().log_ensemble_forecast(
            trace_id or "-",
            market_id or "-",
            metric,
            mean=None, std=None,
            sources_used=[], sources_failed=sources_failed,
            latency_ms=out.latency_ms,
        )
        return out

    values = list(per_source.values())

    # ---- Outlier filter (preliminary median ± zscore * preliminary std) ----
    if len(values) >= 4:
        m0 = median(values)
        s0 = pstdev(values)
        if s0 > 0:
            kept = {n: v for n, v in per_source.items() if abs(v - m0) / s0 <= outlier_zscore}
            if len(kept) >= 3:
                per_source = kept

    # ---- Weighted mean ----
    if source_weights:
        w_sum = sum(source_weights.get(n, 1.0) for n in per_source)
        if w_sum > 0:
            m = sum(per_source[n] * source_weights.get(n, 1.0) for n in per_source) / w_sum
        else:
            m = mean(per_source.values())
    else:
        m = mean(per_source.values())

    s = pstdev(per_source.values()) if len(per_source) > 1 else 0.0
    med = median(per_source.values())

    out = EnsembleResult(
        metric=metric,
        mean=m,
        std=s,
        median=med,
        sources_used=list(per_source.keys()),
        sources_failed=sources_failed,
        per_source=per_source,
        latency_ms=int((time.time() - t0) * 1000),
    )

    get_logger().log_ensemble_forecast(
        trace_id or "-",
        market_id or "-",
        metric,
        mean=out.mean, std=out.std,
        sources_used=out.sources_used, sources_failed=out.sources_failed,
        latency_ms=out.latency_ms,
    )
    return out


# ============================================================
def blend_with_market(
    prob_ensemble: float,
    prob_market: float,
    ensemble_weight: float = 0.40,
) -> float:
    """
    Bayesian blend of ensemble probability and market-implied probability.
    Default: ensemble 40 %, market 60 % (the market is a strong baseline).
    Adjust `ensemble_weight` as confidence in the ensemble grows.
    """
    prob_ensemble = max(0.001, min(0.999, prob_ensemble))
    prob_market = max(0.001, min(0.999, prob_market))
    w = max(0.0, min(1.0, ensemble_weight))
    return w * prob_ensemble + (1.0 - w) * prob_market
