"""Unit tests for the crypto_lag probability model.

Covers:
  - Black-Scholes digital under various spot/strike/sigma/T regimes.
  - EWMA vs plain realized vol behaviour (EWMA reacts faster to bursts).
  - Vol blending with missing legs (weights renormalize).
  - prob_up: BS + microstructure + Polymarket blend.
"""

from __future__ import annotations

import math
import unittest

from strategies.crypto_lag.probability_model import (
    EWMA_LAMBDA_DEFAULT,
    MAX_MICROSTRUCTURE_SHIFT,
    ProbInputs,
    black_scholes_digital_up,
    blend_volatility,
    digital_delta_d_p_d_spot,
    prob_up,
    realized_vol_per_sqrt_s,
    scale_sigma_to_period,
)


class BlackScholesDigitalTests(unittest.TestCase):
    def test_atm_returns_about_half(self):
        # At-the-money with positive sigma & T → ~0.5 (slight bias from the +0.5σ²T
        # term in d2 — Itô correction; clamped to [0.001, 0.999]).
        p = black_scholes_digital_up(spot=100.0, strike=100.0,
                                     sigma_per_sqrt_s=1e-3, t_s=300.0)
        self.assertGreater(p, 0.49)
        self.assertLess(p, 0.55)

    def test_deep_itm_returns_one(self):
        p = black_scholes_digital_up(spot=200.0, strike=100.0,
                                     sigma_per_sqrt_s=1e-3, t_s=300.0)
        self.assertGreaterEqual(p, 0.999)

    def test_deep_otm_returns_zero(self):
        p = black_scholes_digital_up(spot=50.0, strike=100.0,
                                     sigma_per_sqrt_s=1e-3, t_s=300.0)
        self.assertLessEqual(p, 0.001)

    def test_zero_t_resolves_at_strike_with_tie(self):
        # Polymarket tie convention: end ≥ start = UP.
        self.assertEqual(
            black_scholes_digital_up(spot=100.0, strike=100.0,
                                     sigma_per_sqrt_s=1e-3, t_s=0.0),
            1.0,
        )
        # Below strike at T=0 → resolves to 0
        self.assertEqual(
            black_scholes_digital_up(spot=99.99, strike=100.0,
                                     sigma_per_sqrt_s=1e-3, t_s=0.0),
            0.0,
        )

    def test_zero_sigma_with_positive_t_is_50_50(self):
        # σ=0 means the BS digital is degenerate; we return 0.5 as a safe
        # fallback rather than crashing on log(0).
        p = black_scholes_digital_up(spot=100.0, strike=100.0,
                                     sigma_per_sqrt_s=0.0, t_s=300.0)
        self.assertEqual(p, 0.5)

    def test_delta_positive_atm(self):
        d = digital_delta_d_p_d_spot(100.0, 100.0, 1e-3, 300.0)
        # ATM delta is high (binary digital peaks at strike)
        self.assertGreater(d, 0)


class RealizedVolTests(unittest.TestCase):
    def test_constant_price_returns_zero(self):
        hist = [(i, 100.0) for i in range(20)]
        self.assertEqual(realized_vol_per_sqrt_s(hist, mode="plain"), 0.0)
        self.assertEqual(realized_vol_per_sqrt_s(hist, mode="ewma"), 0.0)

    def test_short_history_returns_zero(self):
        self.assertEqual(realized_vol_per_sqrt_s([], mode="plain"), 0.0)
        self.assertEqual(realized_vol_per_sqrt_s([(0, 100.0)], mode="plain"), 0.0)

    def test_ewma_reacts_faster_than_plain(self):
        # Long calm period followed by a vol burst at the end. EWMA should
        # weigh the burst more heavily because of its decay; plain gives
        # equal weight to every observation.
        calm = [(i, 100.0 + math.sin(i * 0.01) * 0.01) for i in range(200)]
        burst = [(200 + i, 100.0 + (i % 3 - 1) * 5.0) for i in range(50)]
        v_plain = realized_vol_per_sqrt_s(calm + burst, mode="plain")
        v_ewma = realized_vol_per_sqrt_s(calm + burst, mode="ewma")
        # Both should be positive
        self.assertGreater(v_plain, 0)
        self.assertGreater(v_ewma, 0)
        # EWMA emphasises recent → strictly larger after a burst
        self.assertGreater(v_ewma, v_plain)

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            realized_vol_per_sqrt_s([(0, 1.0), (1, 2.0)], mode="lol")


class VolBlendTests(unittest.TestCase):
    def test_weighted_average_with_default_weights(self):
        v = blend_volatility(realized=0.001, iv=0.002, garch=0.0015)
        expected = (0.001 * 0.5 + 0.002 * 0.3 + 0.0015 * 0.2)
        self.assertAlmostEqual(v, expected, places=10)

    def test_missing_iv_redistributes_weight(self):
        v = blend_volatility(realized=0.001, iv=None, garch=0.0015)
        expected = (0.001 * 0.5 + 0.0015 * 0.2) / (0.5 + 0.2)
        self.assertAlmostEqual(v, expected, places=10)

    def test_only_realized(self):
        v = blend_volatility(realized=0.001)
        self.assertAlmostEqual(v, 0.001, places=10)

    def test_zero_inputs_return_zero(self):
        v = blend_volatility(realized=0.0, iv=0.0, garch=0.0)
        self.assertEqual(v, 0.0)


class ScaleSigmaTests(unittest.TestCase):
    def test_zero_inputs(self):
        self.assertEqual(scale_sigma_to_period(0.0, 100.0), 0.0)
        self.assertEqual(scale_sigma_to_period(1e-3, 0.0), 0.0)

    def test_one_minute_horizon(self):
        # 1m = 60s → factor = √60 ≈ 7.745
        v = scale_sigma_to_period(1e-3, 60.0)
        self.assertAlmostEqual(v, 1e-3 * math.sqrt(60.0), places=10)


class ProbUpTests(unittest.TestCase):
    def _inputs(self, **overrides):
        defaults = dict(
            spot_now=100.0, strike=100.0,
            sigma_per_sqrt_s=1e-3, t_remaining_s=300.0,
            book_imbalance=0.0, trade_flow_5s=0.0,
            poly_mid=None, poly_book_imbalance=0.0,
        )
        defaults.update(overrides)
        return ProbInputs(**defaults)

    def test_no_microstructure_atm_returns_about_half(self):
        out = prob_up(self._inputs())
        self.assertAlmostEqual(out.p_model, 0.5, delta=0.05)

    def test_book_imbalance_tilts_up(self):
        # Positive imbalance → modest upward shift
        out = prob_up(self._inputs(book_imbalance=1.0))
        self.assertGreater(out.p_model, 0.5)

    def test_negative_imbalance_tilts_down(self):
        out = prob_up(self._inputs(book_imbalance=-1.0))
        self.assertLess(out.p_model, 0.5)

    def test_microstructure_shift_capped(self):
        # Even with extreme inputs, total microstructure shift cannot exceed
        # ±MAX_MICROSTRUCTURE_SHIFT pp on top of p_pure (which itself can drift
        # slightly from 0.5 due to the Itô +0.5σ²T term in d2).
        out = prob_up(
            self._inputs(book_imbalance=1.0, poly_book_imbalance=1.0,
                         trade_flow_5s=1e9),
            imbalance_alpha=10.0,
            poly_obi_alpha=10.0,
            flow_alpha=10.0,
        )
        # p_model = clamp(p_pure + shift, 0.001, 0.999) where |shift| ≤ cap.
        self.assertLessEqual(
            out.p_model, out.p_model_pure + MAX_MICROSTRUCTURE_SHIFT + 1e-9
        )
        self.assertGreaterEqual(
            out.p_model, out.p_model_pure - MAX_MICROSTRUCTURE_SHIFT - 1e-9
        )

    def test_poly_blend_pulls_toward_market_when_close(self):
        # With small disagreement, blend does pull toward poly_mid.
        out_close = prob_up(self._inputs(poly_mid=0.51))
        # Same spot/strike, no microstructure → p_model ≈ 0.5
        # blend weight at diff=0.01 → ~0.10 × (1 - 0.05) ≈ 0.095
        # blended ≈ (1 - 0.095) * 0.5 + 0.095 * 0.51 ≈ 0.5009
        self.assertGreater(out_close.p_blended, out_close.p_model)

    def test_poly_blend_decays_when_far(self):
        # Big disagreement → blend weight decays to 0; p_blended ≈ p_model
        out_far = prob_up(self._inputs(poly_mid=0.95))
        self.assertAlmostEqual(out_far.p_blended, out_far.p_model, delta=0.01)


if __name__ == "__main__":
    unittest.main()
