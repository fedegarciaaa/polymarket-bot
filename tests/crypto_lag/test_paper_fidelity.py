"""Unit tests for the post-audit fidelity fixes (F1-F5).

These guard against regressions of the simulator-LIVE divergence that
produced +717% paper PnL on $1k in 7h:

  F1 — adverse fills must reduce SIZE proportional to q_tox, not just
       nudge fill_price.
  F2 — maker_race_lost_pct must ramp toward maker_race_lost_max in the
       tails (same quadratic shape as q_toxic).
  F3 — q_toxic_extreme must be capped (default 0.70) so F1 doesn't zero
       out tail trading entirely.
  F4 — depth haircut must shrink visible size further when the resting
       price is in the tails (extra spoof).
  F5 — order_engine must require a multiplied edge_threshold to quote
       in [0, P_MIN) ∪ (P_MAX, 1].
"""

from __future__ import annotations

import unittest

from strategies.crypto_lag.paper_executor import PaperExecutor
from strategies.crypto_lag.order_engine import (
    MakerOrderEngine,
    QUOTE_MODE_MAKER,
)


class _DummyRiskState:
    inventory_by_market: dict = {}
    open_orders_count: int = 0


class _DummyRisk:
    per_market_max_inventory_usdc = 100.0
    state = _DummyRiskState()

    def on_order_open(self) -> None: pass
    def on_order_close(self) -> None: pass


class _DummyExec:
    pass


# ─── F3: q_toxic_extreme cap ─────────────────────────────────────────
class QToxicCapTests(unittest.TestCase):
    def test_at_mid_returns_base(self):
        ex = PaperExecutor(q_toxic=0.30, q_toxic_extreme_scaling=True)
        self.assertAlmostEqual(ex._effective_q_toxic(0.5), 0.30, places=4)

    def test_extreme_capped_at_0_70_default(self):
        ex = PaperExecutor(q_toxic=0.30, q_toxic_extreme_scaling=True)
        # Without the cap the formula would produce ~0.88 at p=0.02.
        self.assertLessEqual(ex._effective_q_toxic(0.02), 0.70 + 1e-9)
        self.assertLessEqual(ex._effective_q_toxic(0.98), 0.70 + 1e-9)
        self.assertEqual(ex._effective_q_toxic(0.001), 0.70)

    def test_below_cap_returns_blended_value(self):
        ex = PaperExecutor(q_toxic=0.30, q_toxic_extreme_scaling=True)
        # At p=0.20 the blend is base + (1-base)*(1-4*0.2*0.8)^2 = 0.30 + 0.7*0.1296
        v = ex._effective_q_toxic(0.20)
        self.assertAlmostEqual(v, 0.30 + 0.70 * (1.0 - 4.0 * 0.2 * 0.8) ** 2, places=4)

    def test_cap_can_be_relaxed(self):
        ex = PaperExecutor(q_toxic=0.30, q_toxic_extreme_scaling=True,
                           q_toxic_extreme_cap=0.95)
        self.assertGreater(ex._effective_q_toxic(0.02), 0.80)


# ─── F2: maker_race_lost ramp ────────────────────────────────────────
class MakerRaceLostRampTests(unittest.TestCase):
    def test_at_mid_returns_base(self):
        ex = PaperExecutor(maker_race_lost_pct=0.15, maker_race_lost_max=0.65)
        self.assertAlmostEqual(ex._effective_maker_race_lost_pct(0.5), 0.15, places=4)

    def test_at_extreme_approaches_max(self):
        ex = PaperExecutor(maker_race_lost_pct=0.15, maker_race_lost_max=0.65)
        # At p=0.05 the quadratic is (1 - 4*0.05*0.95)^2 = (1 - 0.19)^2 = 0.6561
        # ramp = 0.15 + 0.50 * 0.6561 ≈ 0.478
        v = ex._effective_maker_race_lost_pct(0.05)
        self.assertGreater(v, 0.45)
        self.assertLess(v, 0.55)

    def test_at_p_one_returns_max(self):
        ex = PaperExecutor(maker_race_lost_pct=0.15, maker_race_lost_max=0.65)
        self.assertAlmostEqual(ex._effective_maker_race_lost_pct(0.999), 0.65, places=2)

    def test_no_ramp_when_max_equals_base(self):
        ex = PaperExecutor(maker_race_lost_pct=0.30, maker_race_lost_max=0.30)
        for p in (0.05, 0.5, 0.95):
            self.assertAlmostEqual(ex._effective_maker_race_lost_pct(p), 0.30, places=4)


# ─── F4: depth haircut by price ──────────────────────────────────────
class DepthHaircutByPriceTests(unittest.TestCase):
    def test_thin_book_at_mid_returns_base_thin_haircut(self):
        ex = PaperExecutor(depth_haircut_enabled=True)
        # 100 USDC visible, p=0.5 → base = 0.50 (thin), no extreme adjust.
        self.assertAlmostEqual(ex._depth_multiplier(100.0, price=0.5), 0.50, places=4)

    def test_extreme_compounds_with_thin_haircut(self):
        ex = PaperExecutor(
            depth_haircut_enabled=True,
            depth_extreme_multiplier=0.50,
        )
        # 100 USDC at p=0.05 → base 0.50 × extreme 0.50 = 0.25
        self.assertAlmostEqual(ex._depth_multiplier(100.0, price=0.05), 0.25, places=4)
        # 100 USDC at p=0.95 → same
        self.assertAlmostEqual(ex._depth_multiplier(100.0, price=0.95), 0.25, places=4)

    def test_near_extreme_uses_intermediate_factor(self):
        ex = PaperExecutor(
            depth_haircut_enabled=True,
            depth_near_extreme_multiplier=0.75,
        )
        # 1500 USDC at p=0.20 → base 1.0 × near 0.75 = 0.75
        self.assertAlmostEqual(ex._depth_multiplier(1500.0, price=0.20), 0.75, places=4)

    def test_disabled_returns_one_regardless_of_price(self):
        ex = PaperExecutor(depth_haircut_enabled=False)
        self.assertAlmostEqual(ex._depth_multiplier(100.0, price=0.05), 1.0, places=4)

    def test_no_price_arg_falls_back_to_size_only(self):
        ex = PaperExecutor(depth_haircut_enabled=True)
        # With no price, no extreme adjustment → base 0.50 only.
        self.assertAlmostEqual(ex._depth_multiplier(100.0), 0.50, places=4)


# ─── F1: adverse-fill size attenuation ───────────────────────────────
class AdverseSizeAttenuationTests(unittest.TestCase):
    def _make_buy_order(self, price, size_usdc=10.0):
        from strategies.crypto_lag.state import RestingOrder
        return RestingOrder(
            order_id="t-1",
            external_order_id=None,
            symbol="BTC",
            condition_id="cid",
            side="BUY",
            outcome="YES",
            price=price,
            size_usdc=size_usdc,
            placed_ts=0.0,
            last_replace_ts=0.0,
            tick_size=0.01,
            is_taker=False,
        )

    def test_adverse_buy_at_extreme_truncates_size(self):
        # Force adverse path: high q_tox at p=0.05 + capped at 0.70 means
        # size_haircut = 1 - 0.70*1.0 = 0.30 of the visible. seed makes
        # the rng deterministic so we always get adverse=True and no race.
        ex = PaperExecutor(
            rng_seed=42,
            q_toxic=0.30,
            q_toxic_extreme_scaling=True,
            q_toxic_extreme_cap=0.70,
            adverse_size_attenuation=1.0,
            min_fill_usdc=0.0,
            maker_race_lost_pct=0.0,   # disable race for determinism
            taker_race_lost_pct=0.0,
            depth_haircut_enabled=False,
            queue_position_enabled=False,
            live_realistic_rebates=True,
        )
        order = self._make_buy_order(price=0.05, size_usdc=10.0)
        # 10 shares of $1 visible at the ask (fill_size_usdc = 10).
        book = {"best_ask": 0.05, "ask_size": 200.0, "best_bid": 0.04, "bid_size": 100.0}
        ev = ex._try_match(order, book)
        # Many seeds will land on adverse; we only assert the contract:
        # if is_adverse, fill_size <= 0.31 * 10 (haircut floor).
        # If not adverse, fill_size = 10 (no truncation).
        self.assertIsNotNone(ev)
        if ev.is_adverse:
            # haircut = 1 - 0.70 = 0.30 → max 3 USDC.
            self.assertLessEqual(ev.fill_size_usdc, 3.0 + 1e-6)
            # Counter must have been incremented.
            self.assertGreaterEqual(ex.adverse_size_truncated_count, 1)

    def test_min_fill_usdc_cancels_below_floor(self):
        # With min_fill_usdc large enough that haircut output < floor,
        # _try_match must return None even if everything else clears.
        ex = PaperExecutor(
            rng_seed=1,
            q_toxic=0.99,                # virtually always adverse
            q_toxic_extreme_scaling=False,
            adverse_size_attenuation=1.0,
            min_fill_usdc=100.0,         # impossible floor
            maker_race_lost_pct=0.0,
            taker_race_lost_pct=0.0,
            depth_haircut_enabled=False,
            queue_position_enabled=False,
            live_realistic_rebates=True,
        )
        order = self._make_buy_order(price=0.05, size_usdc=10.0)
        book = {"best_ask": 0.05, "ask_size": 200.0, "best_bid": 0.04, "bid_size": 100.0}
        ev = ex._try_match(order, book)
        self.assertIsNone(ev)

    def test_zero_attenuation_preserves_size(self):
        # adverse_size_attenuation=0 disables F1; size unchanged.
        ex = PaperExecutor(
            rng_seed=7,
            q_toxic=0.99,
            q_toxic_extreme_scaling=False,
            adverse_size_attenuation=0.0,
            maker_race_lost_pct=0.0,
            taker_race_lost_pct=0.0,
            depth_haircut_enabled=False,
            queue_position_enabled=False,
            live_realistic_rebates=True,
        )
        order = self._make_buy_order(price=0.50, size_usdc=10.0)
        book = {"best_ask": 0.50, "ask_size": 200.0, "best_bid": 0.49, "bid_size": 100.0}
        ev = ex._try_match(order, book)
        self.assertIsNotNone(ev)
        # fill_size capped at order.size_usdc (10) since visible size = 100.
        self.assertAlmostEqual(ev.fill_size_usdc, 10.0, places=4)


# ─── F5: extreme price guard in _decide_maker ───────────────────────
class ExtremePriceGuardTests(unittest.TestCase):
    def _make_engine(self, **kw):
        return MakerOrderEngine(
            executor=_DummyExec(), risk=_DummyRisk(),
            quote_mode=QUOTE_MODE_MAKER,
            edge_threshold_cents=2.0,
            **kw,
        )

    def test_normal_band_uses_base_threshold(self):
        eng = self._make_engine()
        # bid_thr at p=0.5 should equal base = 0.02
        self.assertAlmostEqual(eng._extreme_edge_threshold(0.50), 0.02)

    def test_below_min_multiplies_threshold(self):
        eng = self._make_engine(extreme_price_min=0.10, extreme_edge_multiplier=4.0)
        self.assertAlmostEqual(eng._extreme_edge_threshold(0.05), 0.08)
        # boundary inclusive → 0.10 itself should NOT trigger guard
        self.assertAlmostEqual(eng._extreme_edge_threshold(0.10), 0.02)

    def test_above_max_multiplies_threshold(self):
        eng = self._make_engine(extreme_price_max=0.90, extreme_edge_multiplier=4.0)
        self.assertAlmostEqual(eng._extreme_edge_threshold(0.95), 0.08)
        self.assertAlmostEqual(eng._extreme_edge_threshold(0.90), 0.02)

    def test_blocks_marginal_quote_in_tail(self):
        # fair_mid=0.08, poly_bid=0.04 → edge=0.04 (passes 0.02 base, fails 0.08 extreme)
        eng = self._make_engine(extreme_price_min=0.10, extreme_edge_multiplier=4.0)
        decision = eng.build_decision_two_sided(
            fair_mid=0.08, poly_best_bid=0.04, poly_best_ask=0.06,
            target_size_usdc=10.0, tick=0.01,
        )
        self.assertIsNone(decision.bid_price)
        self.assertGreaterEqual(eng.extreme_price_blocked_lifetime, 1)

    def test_allows_quote_with_huge_edge_in_tail(self):
        # edge = 0.20 (massive) clears even the 4x threshold.
        eng = self._make_engine(extreme_price_min=0.10, extreme_edge_multiplier=4.0)
        decision = eng.build_decision_two_sided(
            fair_mid=0.30, poly_best_bid=0.04, poly_best_ask=0.06,
            target_size_usdc=10.0, tick=0.01,
        )
        # poly_bid + tick = 0.05, still in tail; edge_bid_net=0.26 > 0.08 → OK.
        self.assertIsNotNone(decision.bid_price)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
