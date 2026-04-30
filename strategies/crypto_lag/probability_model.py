"""Probability model for "<symbol> Up or Down — N minutes" markets.

Given the strike price (locked at eventStartTime), the current Binance mid,
the realized volatility, and the time remaining, we compute P(close > strike)
under a Black-Scholes digital option assumption (drift = 0, no jumps).

Microstructure adjustments — small additive shifts capped at ±5%:
  - book imbalance (bid_qty vs ask_qty)
  - signed trade flow over last 5 seconds
  - blend with Polymarket mid (small weight, lets the lagged market correct
    us when we're obviously wrong about a regime change)

This module is deliberately PURE: no I/O, no logging, no time.time(). All
inputs come from the caller. Easy to backtest by replaying historical state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

try:
    from scipy.stats import norm
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

    class _NormFallback:
        @staticmethod
        def cdf(x: float) -> float:
            return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))

    norm = _NormFallback()


# Microstructure tunables — defaults are conservative. Cycle config overrides.
MAX_MICROSTRUCTURE_SHIFT = 0.05  # cap |adjustment| at 5 percentage points


@dataclass(frozen=True)
class ProbInputs:
    """Bundle of all inputs to the probability model."""
    spot_now: float                  # current Binance mid
    strike: float                    # price at eventStartTime
    sigma_per_sqrt_s: float          # realized vol per sqrt(second)
    t_remaining_s: float             # seconds until endDate
    book_imbalance: float = 0.0      # ∈ [-1, 1]
    trade_flow_5s: float = 0.0       # signed last-5s net buy quantity
    poly_mid: Optional[float] = None # current Polymarket YES mid (for blend)


@dataclass(frozen=True)
class ProbOutput:
    p_model_pure: float              # BS digital, no microstructure
    p_model: float                   # with microstructure adjustments
    p_blended: float                 # final after blending with poly_mid
    contributions: dict              # for logging/debugging


def realized_vol_per_sqrt_s(price_history: list[tuple[float, float]]) -> float:
    """Annualization-free realized volatility expressed as σ per √second.

    Input: list of (ts, price) sorted ascending. Uses log returns. Returns 0
    if fewer than 2 points or the time span is degenerate.
    """
    n = len(price_history)
    if n < 2:
        return 0.0
    log_rets = []
    for i in range(1, n):
        t_prev, p_prev = price_history[i - 1]
        t_now, p_now = price_history[i]
        dt = t_now - t_prev
        if dt <= 0 or p_prev <= 0 or p_now <= 0:
            continue
        # Standardize each return to per-sqrt-second
        r = math.log(p_now / p_prev)
        log_rets.append(r / math.sqrt(dt))
    if not log_rets:
        return 0.0
    # Sample stdev (assume zero mean — true on short windows)
    var = sum(r * r for r in log_rets) / len(log_rets)
    return math.sqrt(var)


def black_scholes_digital_up(
    spot: float, strike: float, sigma_per_sqrt_s: float, t_s: float
) -> float:
    """P(S_T > K) under GBM with r=0, q=0. Returns clamped to [0.001, 0.999]."""
    if t_s <= 0:
        return 1.0 if spot >= strike else 0.0
    if sigma_per_sqrt_s <= 0 or strike <= 0 or spot <= 0:
        return 0.5
    sigma_t = sigma_per_sqrt_s * math.sqrt(t_s)
    if sigma_t <= 0:
        return 1.0 if spot >= strike else 0.0
    d2 = (math.log(spot / strike) + 0.5 * sigma_t * sigma_t) / sigma_t
    p = float(norm.cdf(d2))
    return max(0.001, min(0.999, p))


def prob_up(
    inputs: ProbInputs,
    imbalance_alpha: float = 0.03,
    flow_alpha: float = 0.005,
    poly_blend_weight: float = 0.10,
) -> ProbOutput:
    """Full pipeline: BS digital + microstructure + Polymarket blend."""
    p_pure = black_scholes_digital_up(
        spot=inputs.spot_now,
        strike=inputs.strike,
        sigma_per_sqrt_s=inputs.sigma_per_sqrt_s,
        t_s=inputs.t_remaining_s,
    )

    # Microstructure: small additive shifts, clamped overall
    shift = imbalance_alpha * inputs.book_imbalance + flow_alpha * _signed_log1p(
        inputs.trade_flow_5s
    )
    shift = max(-MAX_MICROSTRUCTURE_SHIFT, min(MAX_MICROSTRUCTURE_SHIFT, shift))
    p_micro = max(0.001, min(0.999, p_pure + shift))

    # Polymarket blend: small weight; the lagged market mid is a partial sanity
    # check. If they sharply disagree (|diff| > 0.20), trust our model more.
    p_blended = p_micro
    if inputs.poly_mid is not None and 0.0 < inputs.poly_mid < 1.0:
        diff = abs(p_micro - inputs.poly_mid)
        # Reduce blend weight when the disagreement is large (we're either wrong
        # or the market is hopelessly stale; either way more weight to our model)
        decayed = poly_blend_weight * max(0.0, 1.0 - diff * 5.0)
        p_blended = (1.0 - decayed) * p_micro + decayed * inputs.poly_mid
        p_blended = max(0.001, min(0.999, p_blended))

    return ProbOutput(
        p_model_pure=p_pure,
        p_model=p_micro,
        p_blended=p_blended,
        contributions={
            "shift_imbalance": imbalance_alpha * inputs.book_imbalance,
            "shift_flow": flow_alpha * _signed_log1p(inputs.trade_flow_5s),
            "shift_total": shift,
            "poly_blend_weight_used": (
                poly_blend_weight * max(0.0, 1.0 - abs(p_micro - (inputs.poly_mid or p_micro)) * 5.0)
                if inputs.poly_mid is not None else 0.0
            ),
        },
    )


def _signed_log1p(x: float) -> float:
    """Squash large flow magnitudes; preserves sign."""
    return math.copysign(math.log1p(abs(x)), x)
