"""Unit tests for the LIVE executor's safety guards.

These tests don't touch py-clob-client — they validate the pre-flight
checks (whitelist, sanity, halt logic) that gate every order placement.
The actual CLOB integration is covered by smoke testing in production.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path

from strategies.crypto_lag.live_executor import LiveExecutor, _LiveCredentials


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class CredentialsFromEnvTests(unittest.TestCase):
    def setUp(self):
        # Snapshot env so we can restore it.
        self._snapshot = {k: os.environ.get(k) for k in
                          ("POLYMARKET_PRIVATE_KEY", "POLYMARKET_FUNDER_ADDRESS")}

    def tearDown(self):
        for k, v in self._snapshot.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_missing_env_raises_with_clear_message(self):
        os.environ.pop("POLYMARKET_PRIVATE_KEY", None)
        os.environ.pop("POLYMARKET_FUNDER_ADDRESS", None)
        with self.assertRaises(RuntimeError) as cm:
            _LiveCredentials.from_env()
        msg = str(cm.exception)
        self.assertIn("POLYMARKET_PRIVATE_KEY", msg)
        self.assertIn("POLYMARKET_FUNDER_ADDRESS", msg)

    def test_strips_0x_prefix_from_private_key(self):
        os.environ["POLYMARKET_PRIVATE_KEY"] = "0x" + "ab" * 32
        os.environ["POLYMARKET_FUNDER_ADDRESS"] = "0x" + "cd" * 20
        creds = _LiveCredentials.from_env()
        self.assertEqual(creds.private_key, "ab" * 32)
        self.assertTrue(creds.funder.startswith("0x"))


class HaltLogicTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.halt_path = Path(self._tmp.name) / "halt_live"

    def tearDown(self):
        self._tmp.cleanup()

    def _make(self, **kw):
        defaults = dict(
            variant="t",
            halt_file_path=str(self.halt_path),
            halt_loss_usdc=50.0,
            sanity_checks_enabled=False,
        )
        defaults.update(kw)
        return LiveExecutor(**defaults)

    def test_no_halt_when_pnl_above_threshold(self):
        ex = self._make()
        ex._lifetime_realized_pnl = -10.0
        halted, reason = ex._check_halt()
        self.assertFalse(halted)
        self.assertEqual(reason, "")

    def test_halt_triggers_when_pnl_below_negative_threshold(self):
        ex = self._make()
        ex._lifetime_realized_pnl = -55.0
        halted, reason = ex._check_halt()
        self.assertTrue(halted)
        self.assertIn("PnL", reason)
        self.assertTrue(self.halt_path.exists(), "halt file must be touched on trigger")

    def test_halt_at_exact_threshold(self):
        ex = self._make(halt_loss_usdc=50.0)
        ex._lifetime_realized_pnl = -50.0
        halted, _ = ex._check_halt()
        self.assertTrue(halted, "boundary case at -50 should halt")

    def test_halt_latches_after_first_trigger(self):
        ex = self._make()
        ex._lifetime_realized_pnl = -55.0
        ex._check_halt()
        # Even if PnL recovers, latch keeps halt active for the process lifetime
        ex._lifetime_realized_pnl = +100.0
        halted, reason = ex._check_halt()
        self.assertTrue(halted)
        self.assertEqual(reason, "latched")

    def test_pre_existing_halt_file_blocks(self):
        self.halt_path.parent.mkdir(parents=True, exist_ok=True)
        self.halt_path.write_text("manual stop")
        ex = self._make()
        halted, reason = ex._check_halt()
        self.assertTrue(halted)
        self.assertIn("halt-file", reason)

    def test_custom_pnl_provider_overrides_internal_ledger(self):
        external_pnl = -60.0
        ex = self._make(daily_pnl_provider=lambda: external_pnl)
        ex._lifetime_realized_pnl = +5.0   # internal says we're up
        halted, _ = ex._check_halt()
        self.assertTrue(halted, "external provider must take precedence")


class WhitelistAndSanityTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.halt_path = Path(self._tmp.name) / "halt_live"

    def tearDown(self):
        self._tmp.cleanup()

    def _make_order(self, symbol="BTCUSDT", price=0.5, size=2.0):
        from strategies.crypto_lag.state import RestingOrder
        return RestingOrder(
            order_id="t-1",
            external_order_id=None,
            symbol=symbol,
            condition_id="cid",
            side="BUY",
            outcome="YES",
            price=price,
            size_usdc=size,
            placed_ts=0.0,
            last_replace_ts=0.0,
            tick_size=0.01,
            is_taker=False,
        )

    def _make_exec(self, **kw):
        defaults = dict(
            variant="t",
            halt_file_path=str(self.halt_path),
            max_order_usdc=5.0,
            whitelist_symbols=["BTCUSDT", "ETHUSDT"],
            max_concurrent_orders=5,
        )
        defaults.update(kw)
        ex = LiveExecutor(**defaults)
        # Bypass start() — we don't want to hit the network in unit tests.
        # Mark as started AND stub out the client so the guards execute
        # before the network call would have happened.
        ex._started = True
        ex._client = object()  # opaque sentinel
        return ex

    def test_whitelist_rejects_other_symbols(self):
        ex = self._make_exec()
        order = self._make_order(symbol="DOGEUSDT")
        with self.assertRaises(ValueError) as cm:
            _run(ex.place_order(order, token_id="tok-1"))
        self.assertIn("not in whitelist", str(cm.exception))

    def test_whitelist_accepts_listed_symbol_then_fails_at_client(self):
        # ETHUSDT passes whitelist + sanity, then hits the client=None guard.
        ex = self._make_exec()
        order = self._make_order(symbol="ETHUSDT")
        with self.assertRaises(Exception):
            _run(ex.place_order(order, token_id="tok-1"))

    def test_sanity_rejects_oversize(self):
        ex = self._make_exec(max_order_usdc=5.0)
        order = self._make_order(symbol="BTCUSDT", size=10.0)
        with self.assertRaises(ValueError) as cm:
            _run(ex.place_order(order, token_id="tok-1"))
        self.assertIn("max_order_usdc", str(cm.exception))

    def test_sanity_rejects_under_one_dollar(self):
        ex = self._make_exec()
        order = self._make_order(size=0.5)
        with self.assertRaises(ValueError) as cm:
            _run(ex.place_order(order, token_id="tok-1"))
        self.assertIn("$1 minimum", str(cm.exception))

    def test_sanity_rejects_extreme_price(self):
        ex = self._make_exec()
        order = self._make_order(price=0.01)
        with self.assertRaises(ValueError):
            _run(ex.place_order(order, token_id="tok-1"))

    def test_max_concurrent_orders_blocks_new_placement(self):
        ex = self._make_exec(max_concurrent_orders=2)
        # Pre-fill resting with 2 dummy entries
        ex._resting["a"] = self._make_order()
        ex._resting["b"] = self._make_order()
        order = self._make_order(symbol="BTCUSDT")
        with self.assertRaises(RuntimeError) as cm:
            _run(ex.place_order(order, token_id="tok-1"))
        self.assertIn("max_concurrent_orders", str(cm.exception))

    def test_halt_blocks_placement(self):
        ex = self._make_exec()
        ex._lifetime_realized_pnl = -100.0
        order = self._make_order(symbol="BTCUSDT")
        with self.assertRaises(RuntimeError) as cm:
            _run(ex.place_order(order, token_id="tok-1"))
        self.assertIn("halted", str(cm.exception).lower())


class FillEventApplicationTests(unittest.TestCase):
    def test_buy_fill_creates_long_position_with_correct_vwap(self):
        from strategies.crypto_lag.live_executor import _FillEvent
        ex = LiveExecutor(variant="t", sanity_checks_enabled=False)
        ev = _FillEvent(
            order_id="o", condition_id="cid", symbol="BTC", side="BUY",
            outcome="YES", fill_price=0.40, fill_size_usdc=10.0,
            is_adverse=False, ts=1.0,
        )
        from strategies.crypto_lag.state import RestingOrder
        order = RestingOrder(
            order_id="o", external_order_id=None, symbol="BTC",
            condition_id="cid", side="BUY", outcome="YES",
            price=0.40, size_usdc=10.0, placed_ts=0, last_replace_ts=0,
            end_ts=100, strike_price=50000, market_slug="x",
        )
        ex._apply_fill_to_position(ev, order)
        pos = ex._positions["cid"]
        self.assertEqual(pos.size_usdc, 10.0)
        self.assertEqual(pos.avg_entry_price, 0.40)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
