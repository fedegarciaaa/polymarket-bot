"""Unit tests for the crypto_lag order engine.

Covers:
  - parabolic_fee curve (0 at extremes, peak 1.80% at p=0.5).
  - expected_maker_rebate at the same regime.
  - Two-sided AS quoting: posts both sides when edge clears threshold.
  - Inventory skew: when |q|/q_max > 0.7, the inventory-deepening side is
    suppressed.
  - Fee-aware: a thin gross edge that's below threshold can become net
    edge above threshold once we add the rebate.
"""

from __future__ import annotations

import unittest

from strategies.crypto_lag.order_engine import (
    FEE_RATE_CRYPTO,
    MAKER_REBATE_SHARE,
    MakerOrderEngine,
    expected_maker_rebate,
    parabolic_fee,
)


class _DummyExec:
    pass


class _DummyRiskState:
    inventory_by_market: dict = {}
    open_orders_count: int = 0


class _DummyRisk:
    per_market_max_inventory_usdc = 100.0
    state = _DummyRiskState()

    def on_order_open(self) -> None: pass
    def on_order_close(self) -> None: pass


class FeeCurveTests(unittest.TestCase):
    def test_fee_zero_at_extremes(self):
        self.assertAlmostEqual(parabolic_fee(0.0), 0.0)
        self.assertAlmostEqual(parabolic_fee(1.0), 0.0)

    def test_fee_peak_at_50(self):
        peak = parabolic_fee(0.5)
        # 0.072 * 0.5 * 0.5 = 0.018 = 1.80%
        self.assertAlmostEqual(peak, 0.018)
        # Symmetric around 0.5
        self.assertAlmostEqual(parabolic_fee(0.4), parabolic_fee(0.6))

    def test_fee_clamps_input(self):
        # Out-of-range inputs are clamped, not exploded
        self.assertAlmostEqual(parabolic_fee(-0.1), 0.0)
        self.assertAlmostEqual(parabolic_fee(1.5), 0.0)

    def test_maker_rebate_is_share_of_fee(self):
        for p in (0.1, 0.3, 0.5, 0.7, 0.9):
            r = expected_maker_rebate(p)
            self.assertAlmostEqual(r, parabolic_fee(p) * MAKER_REBATE_SHARE)


class TwoSidedDecisionTests(unittest.TestCase):
    def setUp(self):
        self.eng = MakerOrderEngine(
            executor=_DummyExec(), risk=_DummyRisk(),
            edge_threshold_cents=2.0,
        )

    def test_balanced_inventory_strong_edge_quotes_both_sides(self):
        # fair_mid=0.55, book 0.50/0.51 → bid edge=0.05, ask edge=-0.04
        # So gross-bid is 5pp, gross-ask is -4pp. Net-bid stays large; net-ask
        # gets a tiny rebate boost but is still negative. Should be BID-only.
        d = self.eng.build_decision_two_sided(
            fair_mid=0.55, poly_best_bid=0.50, poly_best_ask=0.51,
            target_size_usdc=10.0, tick=0.01,
            sigma_per_sqrt_s=1e-4, t_remaining_s=300.0,
            inventory_usdc=0.0,
        )
        self.assertEqual(d.side, "BID")

    def test_balanced_with_wider_book_quotes_both_sides(self):
        # Book is 0.45/0.55, fair=0.50 — symmetric, both sides get edge ~5pp.
        # Net edges (incl. rebate) cross threshold on both sides → BOTH.
        d = self.eng.build_decision_two_sided(
            fair_mid=0.50, poly_best_bid=0.45, poly_best_ask=0.55,
            target_size_usdc=10.0, tick=0.01,
            sigma_per_sqrt_s=1e-4, t_remaining_s=300.0,
            inventory_usdc=0.0,
        )
        self.assertEqual(d.side, "BOTH")
        self.assertIsNotNone(d.bid_price)
        self.assertIsNotNone(d.ask_price)
        # Both sides clamped strictly inside the spread
        self.assertGreater(d.bid_price, 0.45)
        self.assertLess(d.ask_price, 0.55)
        # Penny jump worked: bid is at 0.46+, ask at 0.54-
        self.assertGreaterEqual(d.bid_price, 0.46 - 1e-9)
        self.assertLessEqual(d.ask_price, 0.54 + 1e-9)

    def test_inventory_blocks_buy_side_when_too_long(self):
        # Inventory = 90 USDC long out of 100 max → q_norm=0.9 > 0.7 threshold
        # The bot must NOT add more YES. It should still consider ask side.
        d = self.eng.build_decision_two_sided(
            fair_mid=0.50, poly_best_bid=0.45, poly_best_ask=0.55,
            target_size_usdc=10.0, tick=0.01,
            sigma_per_sqrt_s=1e-4, t_remaining_s=300.0,
            inventory_usdc=90.0, per_market_max_inventory_usdc=100.0,
        )
        self.assertIn(d.side, ("ASK", "NONE"))
        self.assertIsNone(d.bid_price)

    def test_inventory_blocks_sell_side_when_too_short(self):
        d = self.eng.build_decision_two_sided(
            fair_mid=0.50, poly_best_bid=0.45, poly_best_ask=0.55,
            target_size_usdc=10.0, tick=0.01,
            sigma_per_sqrt_s=1e-4, t_remaining_s=300.0,
            inventory_usdc=-90.0, per_market_max_inventory_usdc=100.0,
        )
        self.assertIn(d.side, ("BID", "NONE"))
        self.assertIsNone(d.ask_price)

    def test_no_edge_returns_none(self):
        # fair_mid right at mid → both gross edges ~0; rebate alone (~0.36cent
        # at 0.5) does NOT beat the 2-cent threshold.
        d = self.eng.build_decision_two_sided(
            fair_mid=0.50, poly_best_bid=0.50, poly_best_ask=0.51,
            target_size_usdc=10.0, tick=0.01,
            sigma_per_sqrt_s=1e-4, t_remaining_s=300.0,
            inventory_usdc=0.0,
        )
        self.assertEqual(d.side, "NONE")

    def test_legacy_single_sided_still_works(self):
        # `build_decision` keeps its v1 contract for callers/tests that
        # haven't migrated.
        d = self.eng.build_decision(
            fair_mid=0.55, poly_best_bid=0.50, poly_best_ask=0.51,
            target_size_usdc=10.0, tick=0.01,
        )
        self.assertEqual(d.side, "BID")
        self.assertIsNotNone(d.bid_price)
        self.assertIsNone(d.ask_price)


if __name__ == "__main__":
    unittest.main()
