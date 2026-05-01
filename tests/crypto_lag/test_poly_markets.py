"""Unit tests for the poly_markets registry helpers (no network).

The module imports `aiohttp` at the top, so if the test environment doesn't
have it (some local dev machines don't) we skip the whole module rather than
fail. The real bot runs on a server with aiohttp installed.
"""

from __future__ import annotations

import unittest

try:
    import aiohttp  # noqa: F401
    _HAS_AIOHTTP = True
except ImportError:
    _HAS_AIOHTTP = False

if _HAS_AIOHTTP:
    from strategies.crypto_lag.poly_markets import (
        SLUG_RE,
        SYMBOL_TO_SLUG_PREFIXES,
        _parse_horizon_group,
    )
else:  # pragma: no cover — runtime-only branch
    SLUG_RE = None
    SYMBOL_TO_SLUG_PREFIXES = {}
    _parse_horizon_group = None


@unittest.skipUnless(_HAS_AIOHTTP, "aiohttp not installed; skipping poly_markets tests")
class _Base(unittest.TestCase):
    """Base mixin so the skip decorator applies cleanly to all classes below."""
    pass


class SlugParserTests(_Base):
    def test_5m_btc_slug_matches(self):
        m = SLUG_RE.match("btc-updown-5m-1234567890")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "btc")
        self.assertEqual(m.group(2), "5m")

    def test_15m_eth_slug_matches(self):
        m = SLUG_RE.match("eth-updown-15m-1234567890")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(2), "15m")

    def test_1h_sol_slug_matches(self):
        m = SLUG_RE.match("sol-updown-1h-1234567890")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(2), "1h")

    def test_doge_slug_matches(self):
        m = SLUG_RE.match("doge-updown-5m-1234567890")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "doge")

    def test_garbage_slug_does_not_match(self):
        self.assertIsNone(SLUG_RE.match("btc-up-or-down-5m-12345"))
        self.assertIsNone(SLUG_RE.match("btc-updown-5min-12345"))
        self.assertIsNone(SLUG_RE.match("trump-2024-winner"))


class ParseHorizonTests(_Base):
    def test_minutes(self):
        self.assertEqual(_parse_horizon_group("5m"), 5)
        self.assertEqual(_parse_horizon_group("15m"), 15)
        self.assertEqual(_parse_horizon_group("30m"), 30)

    def test_hours(self):
        self.assertEqual(_parse_horizon_group("1h"), 60)
        self.assertEqual(_parse_horizon_group("2h"), 120)

    def test_bare_number_is_minutes(self):
        self.assertEqual(_parse_horizon_group("90"), 90)

    def test_garbage_returns_none(self):
        self.assertIsNone(_parse_horizon_group(""))
        self.assertIsNone(_parse_horizon_group("xyz"))
        self.assertIsNone(_parse_horizon_group(None))


class SymbolPrefixesTests(_Base):
    def test_all_six_symbols_present(self):
        for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT", "XRPUSDT"):
            self.assertIn(sym, SYMBOL_TO_SLUG_PREFIXES)


if __name__ == "__main__":
    unittest.main()
