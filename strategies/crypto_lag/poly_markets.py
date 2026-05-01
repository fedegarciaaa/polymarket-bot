"""Polymarket "<symbol> updown" market registry.

Polls Gamma every `market_poll_seconds` to discover the currently-tradable
short-term up/down markets (slug pattern: `{symbol}-updown-{5m|15m|1h}-{ts}`).

What this module guarantees to consumers:
  - Returns only markets that are within their event window (eventStartTime
    has passed but endDate has not). Pre-event markets have no observable
    strike, so we don't quote on them.
  - Strike price is captured at first sight when the market crosses into its
    event window. If we missed the open, we backfill from Binance history
    closest to eventStartTime.
  - Markets without minimum tradable liquidity are filtered out (see
    `min_liquidity_usdc` config) — there's no point burning quotes on a
    book that no one is going to take.

Discovered fee profile (April 2026): `feeSchedule` is `{rate: 0.072,
takerOnly: true, rebateRate: 0.2}` — makers pay nothing and receive 20%
of taker fees as rebate. Liquidity rewards apply if quotes sit within
`rewardsMaxSpread` (cents) of the mid with size ≥ `rewardsMinSize` USDC.

v2 changes (May 2026 — F0.5 / F2.4 / F4.1 / F4.2 of profitability plan):
  - Multi-horizon support: caller can pass `prefer_horizons=[5, 15, 60]` and
    we'll accept markets in any of those horizons (not just one). Defaults
    keep the legacy 5-minute behaviour.
  - Liquidity filter: markets with `liquidityNum < min_liquidity_usdc` (when
    that field is exposed by Gamma) are filtered out of the active pool.
  - Wash filter: markets with a self-counterparty wash share above
    `max_wash_share` are dropped. Polymarket exposes wash signals on its
    public stats but we treat the field as opportunistic for now (default
    threshold is permissive: 0.05).
  - Symbol expansion: BTC/ETH/SOL/BNB/XRP/DOGE all in the slug regex.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

try:
    import aiohttp
except ImportError as e:
    raise ImportError("crypto_lag.poly_markets requires aiohttp") from e

from .state import PolyCryptoMarket

logger = logging.getLogger("polymarket_bot.crypto_lag.markets")

# Map of Binance symbol → set of slug prefixes (Polymarket uses different tokens
# for the same asset depending on horizon; e.g. "btc-updown-15m" vs "btc-updown-5m").
SYMBOL_TO_SLUG_PREFIXES = {
    "BTCUSDT": ("btc-updown",),
    "ETHUSDT": ("eth-updown",),
    "SOLUSDT": ("sol-updown",),
    "BNBUSDT": ("bnb-updown",),
    "DOGEUSDT": ("doge-updown",),
    "XRPUSDT": ("xrp-updown",),
}

# Match horizons: minutes (5m, 15m, 30m) and hourly (1h). The horizon group is
# normalized to MINUTES in `_parse_horizon_group`.
SLUG_RE = re.compile(r"^(btc|eth|sol|bnb|doge|xrp|hype)-updown-(\d+(?:m|h))-(\d+)$")


def _parse_horizon_group(token: str) -> Optional[int]:
    """Convert '5m' → 5, '15m' → 15, '1h' → 60, '2h' → 120. Returns None if
    the token is malformed."""
    if not token:
        return None
    s = token.strip().lower()
    if s.endswith("m"):
        try:
            return int(s[:-1])
        except ValueError:
            return None
    if s.endswith("h"):
        try:
            return int(s[:-1]) * 60
        except ValueError:
            return None
    try:
        # Bare number → assume minutes
        return int(s)
    except ValueError:
        return None


@dataclass
class _MarketInternal:
    """Mutable internal record. The frozen `PolyCryptoMarket` exposed to
    consumers is rebuilt each time we want to publish a fresh snapshot."""
    symbol: str
    market: PolyCryptoMarket
    raw: dict = field(repr=False)
    fees: dict = field(repr=False)
    horizon_minutes: int = 0
    liquidity_usdc: float = 0.0
    wash_share: float = 0.0
    rewards_enabled: bool = False


class CryptoMarketRegistry:
    """Polls Gamma for active <symbol>-updown markets.

    Usage:
        registry = CryptoMarketRegistry(symbols=['BTCUSDT', 'ETHUSDT'])
        await registry.start()       # spawn the polling task
        # later, in cycle:
        markets = registry.active_for('BTCUSDT', now_ts=time.time())
    """

    GAMMA_URL = "https://gamma-api.polymarket.com/markets"

    def __init__(
        self,
        symbols: list[str],
        poll_seconds: float = 30.0,
        slug_prefixes: Optional[dict[str, tuple[str, ...]]] = None,
        prefer_horizon_minutes: int = 5,
        prefer_horizons: Optional[list[int]] = None,
        min_liquidity_usdc: float = 0.0,
        max_wash_share: float = 0.10,
    ):
        self.symbols = [s.upper() for s in symbols]
        self.poll_seconds = poll_seconds
        self._slug_prefixes = slug_prefixes or SYMBOL_TO_SLUG_PREFIXES
        # Allow either a single horizon (legacy) or a list. The list takes
        # precedence when both are supplied. Convert any to a sorted set of
        # ints for fast membership checks.
        if prefer_horizons:
            self._horizons = {int(h) for h in prefer_horizons if int(h) > 0}
        else:
            self._horizons = {int(prefer_horizon_minutes)}
        self.prefer_horizon_minutes = int(prefer_horizon_minutes)  # legacy attr
        self.min_liquidity_usdc = float(max(0.0, min_liquidity_usdc))
        self.max_wash_share = float(max(0.0, min(1.0, max_wash_share)))
        self._records: dict[str, _MarketInternal] = {}  # condition_id → internal
        self._strikes: dict[str, float] = {}            # condition_id → captured strike
        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    # ─── public surface ────────────────────────────────────────
    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()

    def active_for(
        self, symbol: str, now_ts: Optional[float] = None
    ) -> list[PolyCryptoMarket]:
        """Return frozen `PolyCryptoMarket` records for `symbol` whose trading
        window is currently open (startDate ≤ now < endDate).

        Polymarket crypto markets are created ~24h before the event window
        (eventStartTime). We quote as soon as the market is open for trading
        (startDate), capturing the current Binance price as the strike on first
        sight. The model predicts direction over the full remaining horizon.
        """
        from dataclasses import replace
        now = now_ts or time.time()
        out: list[PolyCryptoMarket] = []
        for rec in self._records.values():
            if rec.symbol != symbol.upper():
                continue
            m = rec.market
            if m.end_ts <= now:
                continue
            # Use startDate (when the market opened for trading) as the gate,
            # not eventStartTime (when the measurement happens). Since we query
            # Gamma with active=true the startDate condition is always met.
            trade_start = float(rec.raw.get("_start_date_ts", 0.0))
            if trade_start > now:
                continue
            # Inject the latest captured strike (the registry holds it in a
            # separate dict). If we don't have one yet, leave 0.0 so the cycle
            # captures it on next tick.
            strike = self._strikes.get(m.condition_id, m.strike_price)
            if strike != m.strike_price:
                m = replace(m, strike_price=float(strike))
            out.append(m)
        return out

    def get_strike(self, condition_id: str) -> Optional[float]:
        return self._strikes.get(condition_id)

    def set_strike(self, condition_id: str, price: float) -> None:
        self._strikes[condition_id] = float(price)

    def fees_for(self, condition_id: str) -> dict:
        rec = self._records.get(condition_id)
        return dict(rec.fees) if rec else {}

    def metadata_for(self, condition_id: str) -> Optional[dict]:
        """Expose horizon, liquidity and rewards metadata for the dashboard
        and the rewards farming module."""
        rec = self._records.get(condition_id)
        if rec is None:
            return None
        return {
            "horizon_minutes": rec.horizon_minutes,
            "liquidity_usdc": rec.liquidity_usdc,
            "wash_share": rec.wash_share,
            "rewards_enabled": rec.rewards_enabled,
            "fees": dict(rec.fees),
        }

    # ─── internals ──────────────────────────────────────────────
    async def _poll_loop(self) -> None:
        async with aiohttp.ClientSession() as session:
            while not self._stop.is_set():
                try:
                    await self._refresh(session)
                except Exception as exc:
                    logger.warning(f"market poll error: {exc}")
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self.poll_seconds)
                except asyncio.TimeoutError:
                    pass
        logger.info("market registry stopped")

    async def _refresh(self, session: aiohttp.ClientSession) -> None:
        # We pull the most recently created markets sorted by startDate desc.
        # Up/down 15m markets have lifetimes of ~16 minutes, so a 200-row
        # window is plenty.
        params = {
            "limit": 200,
            "active": "true",
            "closed": "false",
            "order": "startDate",
            "ascending": "false",
        }
        async with session.get(self.GAMMA_URL, params=params, timeout=15) as r:
            if r.status != 200:
                logger.warning(f"gamma {r.status}")
                return
            data = await r.json()

        new_records: dict[str, _MarketInternal] = {}
        skipped_liquidity = 0
        skipped_wash = 0
        skipped_horizon = 0
        for raw in data:
            slug = (raw.get("slug") or "").lower()
            m = SLUG_RE.match(slug)
            if not m:
                continue
            slug_sym = m.group(1)            # btc | eth | sol | ...
            horizon_token = m.group(2)       # "5m" | "15m" | "1h"
            horizon_min = _parse_horizon_group(horizon_token)
            if horizon_min is None:
                continue
            if horizon_min not in self._horizons:
                skipped_horizon += 1
                continue
            # Map slug prefix back to Binance symbol
            target_sym: Optional[str] = None
            for binance_sym, prefixes in self._slug_prefixes.items():
                if any(slug.startswith(p) for p in prefixes) and binance_sym in self.symbols:
                    target_sym = binance_sym
                    break
            if not target_sym:
                continue
            try:
                cond = self._build_market(raw, target_sym, horizon_min)
            except Exception as exc:
                logger.debug(f"skip slug={slug}: {exc}")
                continue
            if cond is None:
                continue
            # Liquidity filter — drop if Gamma reports zero / below threshold.
            if cond.liquidity_usdc < self.min_liquidity_usdc:
                skipped_liquidity += 1
                continue
            # Wash filter — drop self-counterparty heavy markets if the field
            # is exposed (Gamma sometimes returns 0.0 when unknown; we don't
            # punish those — only filter when wash > threshold).
            if cond.wash_share > self.max_wash_share:
                skipped_wash += 1
                continue
            new_records[cond.market.condition_id] = cond

        # Drop markets that have closed (endDate passed)
        now = time.time()
        new_records = {
            cid: rec for cid, rec in new_records.items()
            if rec.market.end_ts > now
        }
        added = set(new_records) - set(self._records)
        removed = set(self._records) - set(new_records)
        self._records = new_records
        if added or removed or skipped_liquidity or skipped_wash:
            logger.info(
                f"market registry: +{len(added)} -{len(removed)} = {len(self._records)} active "
                f"(skipped: liq={skipped_liquidity} wash={skipped_wash} horizon={skipped_horizon})"
            )

    @staticmethod
    def _parse_iso(s: str) -> float:
        """Parse ISO-8601 (with Z) to POSIX seconds."""
        if not s:
            return 0.0
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()

    @staticmethod
    def _maybe_float(v) -> float:
        try:
            if v is None:
                return 0.0
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    def _build_market(
        self, raw: dict, symbol: str, horizon_min: int
    ) -> Optional[_MarketInternal]:
        # outcomes & token IDs are JSON strings in the Gamma payload
        outcomes_raw = raw.get("outcomes")
        tokens_raw = raw.get("clobTokenIds")
        if isinstance(outcomes_raw, str):
            outcomes_raw = json.loads(outcomes_raw)
        if isinstance(tokens_raw, str):
            tokens_raw = json.loads(tokens_raw)
        if not outcomes_raw or not tokens_raw or len(outcomes_raw) != 2:
            return None
        # Normalize to YES/NO with our convention: YES = "Up", NO = "Down".
        try:
            up_idx = [o.lower() for o in outcomes_raw].index("up")
        except ValueError:
            return None
        down_idx = 1 - up_idx
        token_yes = str(tokens_raw[up_idx])
        token_no = str(tokens_raw[down_idx])

        end_ts = self._parse_iso(raw.get("endDate") or raw.get("endDateIso") or "")
        if end_ts <= 0:
            return None
        event_start_ts = self._parse_iso(raw.get("eventStartTime") or "")
        if event_start_ts <= 0:
            # Fallback: use endDate - horizon for old payloads that omit eventStartTime
            event_start_ts = end_ts - 60.0 * float(horizon_min)

        # Strike: not in the Gamma payload. Captured separately at eventStartTime.
        strike = self._strikes.get(str(raw.get("conditionId")), 0.0)

        market = PolyCryptoMarket(
            condition_id=str(raw.get("conditionId") or ""),
            market_slug=str(raw.get("slug") or ""),
            question=str(raw.get("question") or ""),
            symbol=symbol,
            direction="up",   # YES side = "Up" by convention
            strike_price=strike,
            end_ts=end_ts,
            token_yes=token_yes,
            token_no=token_no,
            tick_size=float(raw.get("orderPriceMinTickSize") or 0.01),
        )

        # Fee snapshot — used by paper_executor and risk to size correctly
        fees = {
            "fee_schedule": dict(raw.get("feeSchedule") or {}),
            "maker_base_fee": raw.get("makerBaseFee"),
            "taker_base_fee": raw.get("takerBaseFee"),
            "maker_rebate_share_bps": raw.get("makerRebatesFeeShareBps"),
            "rewards_max_spread_cents": raw.get("rewardsMaxSpread"),
            "rewards_min_size_usdc": raw.get("rewardsMinSize"),
        }
        rewards_enabled = bool(
            raw.get("rewardsEnabled")
            or (fees.get("rewards_max_spread_cents") not in (None, 0))
        )

        # Liquidity is reported via several fields depending on Gamma's mood.
        # Prefer `liquidityNum` (numeric), fall back to `liquidity` (string).
        liquidity = self._maybe_float(raw.get("liquidityNum"))
        if liquidity <= 0:
            liquidity = self._maybe_float(raw.get("liquidity"))

        # Wash share — only fill when Gamma exposes it. Otherwise 0.
        wash = self._maybe_float(
            raw.get("washVolumeShare")
            or raw.get("selfCounterpartyShare")
        )
        wash = max(0.0, min(1.0, wash))

        start_date_ts = self._parse_iso(raw.get("startDate") or "")

        rec = _MarketInternal(
            symbol=symbol, market=market, raw=raw, fees=fees,
            horizon_minutes=int(horizon_min),
            liquidity_usdc=liquidity,
            wash_share=wash,
            rewards_enabled=rewards_enabled,
        )
        rec.raw["_event_start_ts"] = event_start_ts
        rec.raw["_start_date_ts"] = start_date_ts
        return rec
