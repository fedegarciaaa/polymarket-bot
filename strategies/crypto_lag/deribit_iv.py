"""Deribit Implied Volatility puller.

Pulls ATM (or DVOL-style) implied volatility from Deribit for BTC/ETH/SOL and
exposes it as σ-per-√second so it can be blended directly with our realized
vol estimates.

Why we want this: realized vol on 5-min Binance ticks is dominated by
microstructure noise and undershoots the true forward-looking risk. IV from
Deribit short-dated options encodes the market's forward expectation,
including jumps and macro news, and historically tracks BTC realized at
roughly 1.0-1.3× over 1d horizons. Substack xniiinx documents that mixing
IV with realized boosts the signal-to-noise of the Black-Scholes digital
model meaningfully.

Endpoints (public, no auth):
  - https://www.deribit.com/api/v2/public/get_index_price?index_name=btc_usd
  - https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option
    → returns ALL option chains for the currency. We pick ATM near-dated and
      read the `mark_iv` field (annualized IV in percent points).

Conversion: Deribit `mark_iv` is annualized in % (e.g. 65 = 65%/year). To go
to σ-per-√second we divide by 100 (to fraction) and by √(seconds_per_year).
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Optional

try:
    import aiohttp
except ImportError:
    aiohttp = None  # type: ignore

logger = logging.getLogger("polymarket_bot.crypto_lag.deribit_iv")

_SECONDS_PER_YEAR = 365.0 * 86400.0
_INDEX_URL = "https://www.deribit.com/api/v2/public/get_index_price"
_BOOK_SUM_URL = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"

# Deribit-supported currencies with monthly+weekly options. SOL has options on
# Deribit since mid-2024; XRP/DOGE/BNB do NOT have options on Deribit, so for
# those we'll fall back to BTC IV scaled by an empirical multiplier.
_DERIBIT_CURRENCIES = {"BTC", "ETH", "SOL"}

# Empirical vol scaling for currencies WITHOUT Deribit options. Numbers come
# from rolling 30d realized-vol comparisons (April 2026). They are conservative
# (slightly above-market) so we don't underestimate risk.
_PROXY_FROM_BTC = {
    "BNB": 1.20,
    "DOGE": 2.00,
    "XRP": 1.50,
}

# Map Binance symbol → Deribit currency / proxy currency
def _binance_to_deribit(symbol: str) -> Optional[tuple[str, float]]:
    """Return (deribit_currency, scale_factor) or None if unknown.
    `scale_factor` multiplies the chosen currency's IV to estimate this
    symbol's IV.
    """
    s = symbol.upper().replace("USDT", "").replace("USD", "")
    if s in _DERIBIT_CURRENCIES:
        return s, 1.0
    if s in _PROXY_FROM_BTC:
        return "BTC", float(_PROXY_FROM_BTC[s])
    return None


class DeribitIVProvider:
    """Async pull-and-cache provider. Refresh interval is generous (every
    5 minutes by default) since IV doesn't move that fast and the public API
    rate-limits aggressive callers.
    """

    def __init__(
        self,
        symbols: list[str],
        refresh_seconds: float = 300.0,
        timeout_seconds: float = 10.0,
    ):
        self.symbols = [s.upper() for s in symbols]
        self.refresh_seconds = float(refresh_seconds)
        self.timeout = float(timeout_seconds)
        # symbol → (sigma_per_sqrt_s, ts_pulled)
        self._cache: dict[str, tuple[float, float]] = {}
        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    # ─── public surface ────────────────────────────────────────
    async def start(self) -> None:
        if aiohttp is None:
            logger.warning("DeribitIVProvider: aiohttp not installed; provider disabled")
            return
        if self._task is not None:
            return
        # Prime the cache once before background polling kicks in
        await self._refresh_all()
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()

    def get_sigma_per_sqrt_s(self, symbol: str) -> Optional[float]:
        """Latest IV as σ-per-√second, or None if we never pulled / failed.

        The caller should fall back to realized vol (or fallback constants)
        when this returns None — the cycle already handles missing inputs in
        `blend_volatility`.
        """
        rec = self._cache.get(symbol.upper())
        if rec is None:
            return None
        sigma, _ts = rec
        return float(sigma)

    def cache_age_seconds(self, symbol: str, now_ts: Optional[float] = None) -> float:
        """Seconds since this symbol's IV was last refreshed; +inf if never."""
        rec = self._cache.get(symbol.upper())
        if rec is None:
            return float("inf")
        return float((now_ts or time.time()) - rec[1])

    # ─── internals ─────────────────────────────────────────────
    async def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self._refresh_all()
            except Exception as exc:
                logger.warning(f"deribit IV refresh error: {exc}")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.refresh_seconds)
            except asyncio.TimeoutError:
                pass

    async def _refresh_all(self) -> None:
        if aiohttp is None:
            return
        # Group needed Deribit currencies (BTC/ETH/SOL) — we may need only a
        # subset depending on configured symbols.
        needed: dict[str, list[str]] = {}  # deribit_curr → list of symbols using it
        for sym in self.symbols:
            mapped = _binance_to_deribit(sym)
            if mapped is None:
                continue
            curr, _scale = mapped
            needed.setdefault(curr, []).append(sym)
        if not needed:
            return
        async with aiohttp.ClientSession() as session:
            for curr, syms in needed.items():
                try:
                    iv_annualized = await self._fetch_atm_mark_iv(session, curr)
                except Exception as exc:
                    logger.debug(f"deribit IV fetch {curr}: {exc}")
                    continue
                if iv_annualized is None or iv_annualized <= 0:
                    continue
                # Annualized fraction → per-√second
                sigma_per_sqrt_s = (iv_annualized / 100.0) / math.sqrt(_SECONDS_PER_YEAR)
                now = time.time()
                for sym in syms:
                    mapped = _binance_to_deribit(sym)
                    if mapped is None:
                        continue
                    _curr, scale = mapped
                    self._cache[sym] = (sigma_per_sqrt_s * scale, now)
                logger.info(
                    f"deribit IV {curr}: mark_iv={iv_annualized:.1f}% ann → "
                    f"σ={sigma_per_sqrt_s:.2e}/√s for {','.join(syms)}"
                )

    async def _fetch_atm_mark_iv(
        self, session: "aiohttp.ClientSession", currency: str
    ) -> Optional[float]:
        """Pull the mark_iv (annualized %) for the nearest-dated ATM option.

        We pick the option whose strike is closest to the index price, only
        considering options expiring within the next 7 days (the 5-min
        Polymarket markets care about short-dated implied vol).
        """
        # 1. Index price
        try:
            async with session.get(
                _INDEX_URL,
                params={"index_name": f"{currency.lower()}_usd"},
                timeout=self.timeout,
            ) as r:
                if r.status != 200:
                    return None
                data = await r.json()
            index_price = float(data.get("result", {}).get("index_price") or 0.0)
        except Exception:
            return None
        if index_price <= 0:
            return None

        # 2. Option chain summary
        try:
            async with session.get(
                _BOOK_SUM_URL,
                params={"currency": currency, "kind": "option"},
                timeout=self.timeout,
            ) as r:
                if r.status != 200:
                    return None
                data = await r.json()
            chain = data.get("result") or []
        except Exception:
            return None
        if not chain:
            return None

        now_ms = time.time() * 1000.0
        max_ms_ahead = 7.0 * 86400.0 * 1000.0
        best = None  # (distance_to_atm, mark_iv)
        for opt in chain:
            # Filter to short-dated by parsing expiry from instrument name.
            # Deribit names look like BTC-12MAY26-78000-C
            try:
                name = opt.get("instrument_name", "")
                parts = name.split("-")
                if len(parts) != 4:
                    continue
                strike = float(parts[2])
                mark_iv = float(opt.get("mark_iv") or 0.0)
                if mark_iv <= 0:
                    continue
                # We don't need to parse the expiry ourselves; Deribit gives us
                # `creation_timestamp` and stores expiry in the chain payload.
                expiry_ms = float(opt.get("expiration_timestamp") or 0.0)
                if expiry_ms <= 0:
                    continue
                if expiry_ms - now_ms > max_ms_ahead:
                    continue
                if expiry_ms - now_ms < 0:
                    continue
                dist = abs(strike - index_price) / index_price
                if best is None or dist < best[0]:
                    best = (dist, mark_iv)
            except (TypeError, ValueError):
                continue

        if best is None:
            return None
        return float(best[1])
