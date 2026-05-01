"""Unit tests for the in-house GARCH(1,1) forecaster."""

from __future__ import annotations

import math
import unittest

from strategies.crypto_lag.garch import Garch11, returns_from_prices


class Garch11Tests(unittest.TestCase):
    def test_init_rejects_non_stationary_params(self):
        with self.assertRaises(ValueError):
            Garch11(omega=1e-7, alpha=0.5, beta=0.5)
        with self.assertRaises(ValueError):
            Garch11(omega=-1e-7)

    def test_long_run_sigma_matches_formula(self):
        g = Garch11(omega=2e-6, alpha=0.10, beta=0.85)
        # σ_lr² = ω / (1 - α - β) = 2e-6 / 0.05 = 4e-5
        # σ_lr = 0.006324
        self.assertAlmostEqual(g.long_run_sigma(), math.sqrt(4e-5), places=6)

    def test_pre_fit_sigma_equals_long_run(self):
        g = Garch11(omega=2e-6, alpha=0.10, beta=0.85)
        self.assertAlmostEqual(g.sigma(), g.long_run_sigma(), places=6)

    def test_fit_then_update_changes_sigma(self):
        g = Garch11()
        rets = [0.001 * (i % 5 - 2) for i in range(100)]
        g.fit(rets)
        s0 = g.sigma()
        # A burst of +5σ shocks should *raise* the forecast
        for _ in range(5):
            g.update(0.05)
        s1 = g.sigma()
        self.assertGreater(s1, s0)

    def test_returns_from_prices_skips_invalid(self):
        rets = returns_from_prices([100, 101, None, "junk", 102, 0, 103])
        # Valid transitions: 100→101, 101→102, 102→103
        self.assertEqual(len(rets), 3)
        for r in rets:
            self.assertTrue(math.isfinite(r))

    def test_empty_fit_seeds_with_long_run(self):
        g = Garch11()
        g.fit([])
        self.assertTrue(g.fitted)
        self.assertAlmostEqual(g.sigma(), g.long_run_sigma(), places=6)


if __name__ == "__main__":
    unittest.main()
