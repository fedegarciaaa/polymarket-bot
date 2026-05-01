"""Probability model for "<symbol> Up or Down — N minutes" markets.

Given the strike price (locked at eventStartTime), the current Binance mid,
the realized volatility, and the time remaining, we compute P(close > strike)
under a Black-Scholes digital option assumption (drift = 0, no jumps).

Microstructure adjustments — small additive shifts capped at ±5%:
  - book imbalance (Binance bid_qty vs ask_qty)
  - signed trade flow over last 5 seconds
  - Polymarket OBI tilt (own-book imbalance, separate from Binance imbalance)
  - blend with Polymarket mid (small weight, lets the lagged market correct
    us when we're obviously wrong about a regime change)

This module is deliberately PURE: no I/O, no logging, no time.time(). All
inputs come from the caller. Easy to backtest by replaying historical state.

v2 changes (May 2026 — F1.1, F1.3 of profitability plan):
  - `realized_vol_per_sqrt_s` now supports EWMA mode (λ=0.94 by default) for
    faster reaction to regime change while staying numerically stable on short
    histories. The legacy plain-variance path is preserved as `mode="plain"`
    so `binance_feed.bootstrap_historical_vol()` keeps producing the same
    sigma it always did.
  - New helper `blend_volatility(realized_pp, iv_pp, garch_pp, weights)` that
    composes the three vol estimates into one per-period sigma. The cycle
    still passes a per-√s sigma into `prob_up`; this is for the caller to
    compose first.
  - New input `poly_book_imbalance` separate from the Binance `book_imbalance`.
    Polymarket OBI is much noisier but can tilt the model when the Polymarket
    book is one-sided (Substack navnoorbawa: bid>65% predicts up-drift in
    15-30min windows). Capped at the same ±5% MAX_MICROSTRUCTURE_SHIFT.
  - Backwards compatible: all new fields are optional with safe defaults.
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

        @staticmethod
        def pdf(x: float) -> float:
            return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    norm = _NormFallback()


# Microstructure tunables — defaults are conservative. Cycle config overrides.
MAX_MICROSTRUCTURE_SHIFT = 0.05  # cap |adjustment| at 5 percentage points

# Default EWMA decay for realized vol. RiskMetrics 1996 constant.
EWMA_LAMBDA_DEFAULT = 0.94


@dataclass(frozen=True)
class ProbInputs:
    """Bundle of all inputs to the probability model."""
    spot_now: float                           # current Binance mid (or perp mark)
    strike: float                             # price at eventStartTime
    sigma_per_sqrt_s: float                   # realized vol per sqrt(second)
    t_remaining_s: float                      # seconds until endDate
    book_imbalance: float = 0.0               # Binance top-of-book imbalance ∈ [-1, 1]
    trade_flow_5s: float = 0.0                # signed last-5s net buy quantity (Binance)
    poly_mid: Optional[float] = None          # current Polymarket YES mid (for blend)
    poly_book_imbalance: float = 0.0          # Polymarket OBI ∈ [-1, 1]


@dataclass(frozen=True)
class ProbOutput:
    p_model_pure: float              # BS digital, no microstructure
    p_model: float                   # with microstructure adjustments
    p_blended: float                 # final after blending with poly_mid
    contributions: dict              # for logging/debugging


def realized_vol_per_sqrt_s(
    price_history: list[tuple[float, float]],
    mode: str = "plain",
    ewma_lambda: float = EWMA_LAMBDA_DEFAULT,
) -> float:
    """Annualization-free realized volatility expressed as σ per √second.

    Input: list of (ts, price) sorted ascending. Uses log returns. Returns 0
    if fewer than 2 points or the time span is degenerate.

    Modes:
      - "plain": sample variance (legacy behaviour, used by the historical
        klines bootstrap which is intrinsically stable over 24h).
      - "ewma":  exponentially weighted variance with decay λ. Reacts faster
        to regime change while staying smooth. Used inside the cycle on the
        rolling 5-min ticker history.
    """
    n = len(price_history)
    if n < 2:
        return 0.0
    log_rets: list[float] = []
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

    if mode == "plain":
        # Sample stdev (assume zero mean — true on short windows)
        var = sum(r * r for r in log_rets) / len(log_rets)
        return math.sqrt(var)

    if mode != "ewma":
        raise ValueError(f"realized_vol_per_sqrt_s: unknown mode {mode!r}")

    # EWMA: var_t = λ·var_{t-1} + (1-λ)·r_t² ; seed with first squared return.
    lam = max(0.0, min(0.999, ewma_lambda))
    var = log_rets[0] * log_rets[0]
    for r in log_rets[1:]:
        var = lam * var + (1.0 - lam) * (r * r)
    return math.sqrt(max(var, 0.0))


def blend_volatility(
    realized: float,
    iv: Optional[float] = None,
    garch: Optional[float] = None,
    weights: Optional[dict] = None,
) -> float:
    """Compose realized + implied + GARCH vol estimates into one number.

    All inputs and the output are in the SAME units (e.g. all per √second, or
    all per √period). The function only does a weighted average — it does
    NOT scale across horizons.

    `weights` defaults to ``{"realized": 0.5, "iv": 0.3, "garch": 0.2}``.
    Missing inputs (None or ≤0) get their weight redistributed to the rest.
    """
    w = dict(weights or {"realized": 0.5, "iv": 0.3, "garch": 0.2})
    candidates: list[tuple[float, float]] = []  # (value, weight)
    if realized > 0:
        candidates.append((float(realized), float(w.get("realized", 0.0))))
    if iv is not None and iv > 0:
        candidates.append((float(iv), float(w.get("iv", 0.0))))
    if garch is not None and garch > 0:
        candidates.append((float(garch), float(w.get("garch", 0.0))))
    if not candidates:
        return 0.0
    total_w = sum(wi for _, wi in candidates)
    if total_w <= 0:
        # Fall back to simple average if all weights ended up zero.
        return sum(v for v, _ in candidates) / len(candidates)
    return sum(v * wi for v, wi in candidates) / total_w


def scale_sigma_to_period(sigma_per_sqrt_s: float, t_s: float) -> float:
    """Convert per-√second sigma to a sigma for the full period of length t_s.

    Useful for constructing strike thresholds: a 1-σ move over T seconds is
    `spot · sigma_per_sqrt_s · √t_s`.
    """
    if sigma_per_sqrt_s <= 0 or t_s <= 0:
        return 0.0
    return float(sigma_per_sqrt_s) * math.sqrt(float(t_s))


def black_scholes_digital_up(
    spot: float, strike: float, sigma_per_sqrt_s: float, t_s: float
) -> float:
    """P(S_T > K) under GBM with r=0, q=0. Returns clamped to [0.001, 0.999].

    Tie convention matches Polymarket "Up or Down": end ≥ start counts as UP,
    so when spot equals strike at T=0 we return 1.0 (the YES side wins ties).
    """
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


def digital_delta_d_p_d_spot(
    spot: float, strike: float, sigma_per_sqrt_s: float, t_s: float
) -> float:
    """∂P(YES)/∂S under GBM. Used by the order_engine to compute fee-aware
    edges and reservation skews. Returns 0 outside the well-defined region.
    """
    if t_s <= 0 or sigma_per_sqrt_s <= 0 or strike <= 0 or spot <= 0:
        return 0.0
    sigma_t = sigma_per_sqrt_s * math.sqrt(t_s)
    if sigma_t <= 0:
        return 0.0
    d2 = (math.log(spot / strike) + 0.5 * sigma_t * sigma_t) / sigma_t
    return float(norm.pdf(d2)) / (spot * sigma_t)


def prob_up(
    inputs: ProbInputs,
    imbalance_alpha: float = 0.03,
    flow_alpha: float = 0.005,
    poly_blend_weight: float = 0.10,
    poly_obi_alpha: float = 0.02,
) -> ProbOutput:
    """Full pipeline: BS digital + microstructure + Polymarket blend.

    The microstructure shift is the SUM of three sources, capped at
    ±MAX_MICROSTRUCTURE_SHIFT (=5pp) overall:
      - Binance book imbalance × imbalance_alpha
      - Squashed Binance trade flow × flow_alpha
      - Polymarket book imbalance × poly_obi_alpha (own-book signal)
    """
    p_pure = black_scholes_digital_up(
        spot=inputs.spot_now,
        strike=inputs.strike,
        sigma_per_sqrt_s=inputs.sigma_per_sqrt_s,
        t_s=inputs.t_remaining_s,
    )

    # Microstructure: small additive shifts, clamped overall
    shift_imb = imbalance_alpha * inputs.book_imbalance
    shift_flow = flow_alpha * _signed_log1p(inputs.trade_flow_5s)
    shift_poly_obi = poly_obi_alpha * inputs.poly_book_imbalance
    shift_raw = shift_imb + shift_flow + shift_poly_obi
    shift = max(-MAX_MICROSTRUCTURE_SHIFT, min(MAX_MICROSTRUCTURE_SHIFT, shift_raw))
    p_micro = max(0.001, min(0.999, p_pure + shift))

    # Polymarket blend: small weight; the lagged market mid is a partial sanity
    # check. If they sharply disagree (|diff| > 0.20), trust our model more.
    p_blended = p_micro
    decayed_weight = 0.0
    if inputs.poly_mid is not None and 0.0 < inputs.poly_mid < 1.0:
        diff = abs(p_micro - inputs.poly_mid)
        # Reduce blend weight when the disagreement is large (we're either wrong
        # or the market is hopelessly stale; either way more weight to our model)
        decayed_weight = poly_blend_weight * max(0.0, 1.0 - diff * 5.0)
        p_blended = (1.0 - decayed_weight) * p_micro + decayed_weight * inputs.poly_mid
        p_blended = max(0.001, min(0.999, p_blended))

    return ProbOutput(
        p_model_pure=p_pure,
        p_model=p_micro,
        p_blended=p_blended,
        contributions={
            "shift_imbalance": shift_imb,
            "shift_flow": shift_flow,
            "shift_poly_obi": shift_poly_obi,
            "shift_raw": shift_raw,
            "shift_total": shift,
            "poly_blend_weight_used": decayed_weight,
        },
    )


def _signed_log1p(x: float) -> float:
    """Squash large flow magnitudes; preserves sign."""
    return math.copysign(math.log1p(abs(x)), x)
