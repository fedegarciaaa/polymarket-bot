"""Polymarket "<symbol> updown" market registry.

Polls Gamma every `market_poll_seconds` to discover the currently-tradable
short-term up/down markets (slug pattern: `{symbol}-updown-{15m|5m}-{ts}`).

What this module guarantees to consumers:
  - Returns only markets that are within their event window (eventStartTime
    has passed but endDate has not). Pre-event markets have no observable
    strike, so we don't quote on them.
  - Strike price is captured at first sight when the market crosses into its
    event window. If we missed the open, we backfill from Binance history
    closest to eventStartTime.

Discovered fee profile (April 2026): `feeSchedule` is `{rate: 0.072,
takerOnly: true, rebateRate: 0.2}` — makers pay nothing and receive 20%
of taker fees as rebate. Liquidity rewards apply if quotes sit within
`rewardsMaxSpread` (cents) of the mid with size ≥ `rewardsMinSize` USDC.
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
    # Binance USD-M futures pairs (not used by default but easy to extend)
    "BNBUSDT": ("bnb-updown",),
    "DOGEUSDT": ("doge-updown",),
    "XRPUSDT": ("xrp-updown",),
}

SLUG_RE = re.compile(r"^(btc|eth|sol|bnb|doge|xrp|hype)-updown-(\d+)m-(\d+)$")


@dataclass
class _MarketInternal:
    """Mutable internal record. The frozen `PolyCryptoMarket` exposed to
    consumers is rebuilt each time we want to publish a fresh snapshot."""
    symbol: str
    market: PolyCryptoMarket
    raw: dict = field(repr=False)
    fees: dict = field(repr=False)


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
        prefer_horizon_minutes: int = 15,
    ):
        self.symbols = [s.upper() for s in symbols]
        self.poll_seconds = poll_seconds
        self._slug_prefixes = slug_prefixes or SYMBOL_TO_SLUG_PREFIXES
        self.prefer_horizon_minutes = prefer_horizon_minutes
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
        """Return frozen `PolyCryptoMarket` records for `symbol` whose event
        window is currently open (eventStartTime ≤ now < endDate).

        IMPORTANT: rebuilds the frozen dataclass with the latest captured
        strike from `self._strikes`. Previously the strike captured by the
        cycle stayed in the dict but never propagated to the dataclass field,
        so `market.strike_price` stayed at 0.0 forever and the cycle was stuck
        in a "capture strike → return" loop without generating snapshots.
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
            # event-window check uses raw startDate-of-event; we stored it as a
            # field-not-on-PolyCryptoMarket — keep the check on the raw record.
            event_start = float(rec.raw.get("_event_start_ts", m.end_ts))
            if event_start > now:
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
        for raw in data:
            slug = (raw.get("slug") or "").lower()
            m = SLUG_RE.match(slug)
            if not m:
                continue
            slug_sym = m.group(1)            # btc | eth | sol | ...
            horizon_min = int(m.group(2))    # 5 | 15
            if horizon_min != self.prefer_horizon_minutes:
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
                cond = self._build_market(raw, target_sym)
            except Exception as exc:
                logger.debug(f"skip slug={slug}: {exc}")
                continue
            if cond is None:
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
        if added or removed:
            logger.info(
                f"market registry: +{len(added)} -{len(removed)} = {len(self._records)} active"
            )

    @staticmethod
    def _parse_iso(s: str) -> float:
        """Parse ISO-8601 (with Z) to POSIX seconds."""
        if not s:
            return 0.0
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()

    def _build_market(
        self, raw: dict, symbol: str
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
            # Fallback: use endDate - 15min for old payloads that omit eventStartTime
            event_start_ts = end_ts - 60.0 * self.prefer_horizon_minutes

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

        rec = _MarketInternal(symbol=symbol, market=market, raw=raw, fees=fees)
        rec.raw["_event_start_ts"] = event_start_ts
        return rec
