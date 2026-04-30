"""Open-Meteo - no API key. https://open-meteo.com/en/docs"""

from __future__ import annotations
from datetime import datetime, timezone, timedelta
from typing import Optional

import aiohttp

from .base import WeatherSource


class OpenMeteoSource(WeatherSource):
    name = "open_meteo"
    requires_key = False
    supported_metrics = {
        "temperature_c", "temperature_max_c", "temperature_min_c",
        "precipitation_mm", "wind_mph", "wind_kph", "snow_cm",
    }
    max_lead_time_hours = 240

    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    async def _fetch(self, session, lat, lon, target_dt, metric) -> Optional[float]:
        # Daily max/min temperature → use the `daily` endpoint with
        # timezone=auto so the aggregation window matches the *local* calendar
        # day the Polymarket question resolves on. Other metrics keep hourly.
        if metric in ("temperature_max_c", "temperature_min_c"):
            var = "temperature_2m_max" if metric == "temperature_max_c" else "temperature_2m_min"
            params = {
                "latitude": lat,
                "longitude": lon,
                "daily": var,
                "timezone": "auto",
                "forecast_days": 10,
            }
            async with session.get(self.BASE_URL, params=params) as r:
                if r.status != 200:
                    return None
                data = await r.json()
            days = data.get("daily", {}).get("time", [])
            values = data.get("daily", {}).get(var, [])
            target_day = target_dt.strftime("%Y-%m-%d")
            for d, v in zip(days, values):
                if d == target_day and v is not None:
                    return float(v)
            return None

        hourly_map = {
            "temperature_c": "temperature_2m",
            "precipitation_mm": "precipitation",
            "wind_mph": "wind_speed_10m",
            "wind_kph": "wind_speed_10m",
            "snow_cm": "snowfall",  # Open-Meteo returns cm
        }
        var = hourly_map[metric]
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": var,
            "timezone": "UTC",
            "forecast_days": 10,
            "wind_speed_unit": "kmh",
        }
        async with session.get(self.BASE_URL, params=params) as r:
            if r.status != 200:
                return None
            data = await r.json()

        hours = data.get("hourly", {}).get("time", [])
        values = data.get("hourly", {}).get(var, [])
        if not hours:
            return None

        target_hour = target_dt.strftime("%Y-%m-%dT%H:00")
        for t, v in zip(hours, values):
            if t == target_hour and v is not None:
                if metric == "wind_mph":
                    return self.kph_to_mph(v)
                return float(v)
        return None
