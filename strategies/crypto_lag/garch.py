"""Lightweight GARCH(1,1) volatility forecaster.

Used by the cycle to add a third leg to the volatility blend (alongside
realized EWMA and Deribit IV). We keep it dependency-free (no statsmodels,
no scipy.optimize) and intentionally simple — RiskMetrics-style fixed-α/β
defaults that need no MLE fit. The variance update is:

    σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}

with conventional crypto-friendly defaults ω=1e-7, α=0.10, β=0.85 (Chaim &
Laurini 2019 ballparks for BTC). For one-step-ahead forecasting that's all
we need; if we ever want a proper MLE fit we can add scipy later.

The class operates on per-bar log returns. The caller passes a series of
log returns (each one is r_i = ln(P_i/P_{i-1})) and gets back a sigma in the
SAME unit as the bar (e.g. per-minute if you fed 1m bars). Use
`probability_model.scale_sigma_to_period` to convert across horizons.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Iterable, Optional


class Garch11:
    """GARCH(1,1) one-step-ahead variance forecaster, no MLE.

    Usage:
        g = Garch11()
        g.fit(returns_iterable)        # warm-up over historical bars
        g.update(latest_return)        # roll forward one bar at a time
        sigma_per_bar = g.sigma()      # forecast σ for next bar
    """

    def __init__(
        self,
        omega: float = 1e-7,
        alpha: float = 0.10,
        beta: float = 0.85,
        max_history: int = 5000,
    ):
        if alpha < 0 or beta < 0 or omega < 0:
            raise ValueError("Garch11 params must be non-negative")
        if alpha + beta >= 1.0:
            raise ValueError(
                f"Garch11 not stationary: alpha+beta={alpha+beta:.3f} ≥ 1"
            )
        self.omega = float(omega)
        self.alpha = float(alpha)
        self.beta = float(beta)
        # Long-run variance under stationarity: ω / (1 - α - β)
        self._lr_var = self.omega / (1.0 - self.alpha - self.beta)
        self._var: float = self._lr_var
        self._returns: Deque[float] = deque(maxlen=int(max_history))
        self._fitted: bool = False

    # ─── public surface ────────────────────────────────────────
    def fit(self, returns: Iterable[float]) -> None:
        """Initialize variance from a historical sample. Seeds with sample
        variance (around mean=0), then rolls the GARCH recursion forward over
        every observation."""
        rs = [float(r) for r in returns if math.isfinite(float(r))]
        if not rs:
            self._var = self._lr_var
            self._fitted = True
            return
        sample_var = sum(r * r for r in rs) / len(rs)
        # Seed with sample var; let the recursion adjust as it walks through.
        self._var = max(sample_var, self._lr_var * 0.01)
        for r in rs:
            self._step(r)
            self._returns.append(r)
        self._fitted = True

    def update(self, ret: float) -> None:
        """Push one new observation and roll the variance forward by one bar."""
        if not math.isfinite(ret):
            return
        self._step(float(ret))
        self._returns.append(float(ret))
        self._fitted = True

    def variance(self) -> float:
        """Variance of the *next* bar (one-step-ahead forecast)."""
        return float(self._var)

    def sigma(self) -> float:
        """One-step-ahead σ (same units as the input returns)."""
        return math.sqrt(max(self._var, 0.0))

    def long_run_sigma(self) -> float:
        """Unconditional σ implied by (ω, α, β)."""
        return math.sqrt(max(self._lr_var, 0.0))

    @property
    def fitted(self) -> bool:
        return self._fitted

    # ─── internals ─────────────────────────────────────────────
    def _step(self, r: float) -> None:
        # σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}
        self._var = self.omega + self.alpha * (r * r) + self.beta * self._var
        # Numerical floor to avoid pathological zeros from a long quiet streak.
        if self._var <= 0:
            self._var = self._lr_var * 0.01


def returns_from_prices(prices: Iterable[float]) -> list[float]:
    """Compute log-returns from a price series. Skips non-finite / non-positive.
    """
    out: list[float] = []
    last: Optional[float] = None
    for p in prices:
        if p is None:
            continue
        try:
            pf = float(p)
        except (TypeError, ValueError):
            continue
        if pf <= 0 or not math.isfinite(pf):
            continue
        if last is not None and last > 0:
            out.append(math.log(pf / last))
        last = pf
    return out
