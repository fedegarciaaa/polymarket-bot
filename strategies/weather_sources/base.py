"""
Weather source base class + metric registry.

All sources return **SI-ish units**:
  - temperature_c:    degrees Celsius
  - precipitation_mm: millimeters over a 1-hour window ending at target_dt
  - wind_mph:         miles per hour (retail markets quote mph)
  - wind_kph:         kilometers per hour
  - snow_cm:          centimeters over the day containing target_dt

If a source returns a different unit natively, it converts before returning.
"""

from __future__ import annotations
import os
import time
import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional

import aiohttp

logger = logging.getLogger("polymarket_bot.weather_sources")


METRICS = {
    "temperature_c",
    "temperature_f",       # accepted as alias, converted to °C internally
    "temperature_max_c",   # daily high (°C) — the metric Polymarket markets actually resolve on
    "temperature_min_c",   # daily low  (°C)
    "precipitation_mm",
    "precipitation_in",    # alias, converted to mm
    "wind_mph",
    "wind_kph",
    "snow_cm",
    "snow_in",             # alias, converted to cm
}


def normalize_metric(metric: str) -> str:
    """Normalize aliases to canonical metric names."""
    return {
        "temperature_f": "temperature_c",
        "precipitation_in": "precipitation_mm",
        "snow_in": "snow_cm",
    }.get(metric, metric)


class WeatherSource(ABC):
    """Abstract base: one provider (Open-Meteo, NOAA, etc.)."""

    name: str = "base"
    requires_key: bool = False
    supported_metrics: set[str] = set()
    # Max lead time in hours for which the source is reliable
    max_lead_time_hours: int = 240

    def __init__(self):
        self._cache: dict = {}
        self._cache_ttl = int(os.environ.get("SOURCE_CACHE_TTL", "60"))
        self._timeout = int(os.environ.get("SOURCE_TIMEOUT", "10"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def forecast(
        self,
        lat: float,
        lon: float,
        target_dt: datetime,
        metric: str,
    ) -> Optional[float]:
        """Return forecasted value or None on any failure / unsupported metric."""
        metric = normalize_metric(metric)
        if metric not in self.supported_metrics:
            return None
        lead = (target_dt - datetime.now(timezone.utc)).total_seconds() / 3600
        if lead > self.max_lead_time_hours or lead < -1:
            return None

        key = self._cache_key(lat, lon, target_dt, metric)
        cached = self._cache.get(key)
        if cached and time.time() - cached[0] < self._cache_ttl:
            return cached[1]

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout),
                headers={"User-Agent": f"PolymarketWeatherBot/2.0 ({self.name})"},
            ) as session:
                value = await self._fetch(session, lat, lon, target_dt, metric)
            if value is not None:
                self._cache[key] = (time.time(), value)
            return value
        except asyncio.TimeoutError:
            logger.warning(f"{self.name} timeout for {metric} @ ({lat},{lon})")
            return None
        except Exception as e:
            logger.warning(f"{self.name} error: {e}")
            return None

    # ------------------------------------------------------------------
    @abstractmethod
    async def _fetch(
        self,
        session: aiohttp.ClientSession,
        lat: float,
        lon: float,
        target_dt: datetime,
        metric: str,
    ) -> Optional[float]:
        ...

    # ------------------------------------------------------------------
    def _cache_key(self, lat: float, lon: float, target_dt: datetime, metric: str) -> str:
        return f"{self.name}|{lat:.2f}|{lon:.2f}|{target_dt.isoformat()}|{metric}"

    # ------------------------------------------------------------------
    # Unit helpers
    # ------------------------------------------------------------------
    @staticmethod
    def f_to_c(f: float) -> float:
        return (f - 32.0) * 5.0 / 9.0

    @staticmethod
    def mph_to_kph(v: float) -> float:
        return v * 1.609344

    @staticmethod
    def kph_to_mph(v: float) -> float:
        return v / 1.609344

    @staticmethod
    def ms_to_mph(v: float) -> float:
        return v * 2.23694

    @staticmethod
    def ms_to_kph(v: float) -> float:
        return v * 3.6

    @staticmethod
    def in_to_mm(v: float) -> float:
        return v * 25.4

    @staticmethod
    def in_to_cm(v: float) -> float:
        return v * 2.54
