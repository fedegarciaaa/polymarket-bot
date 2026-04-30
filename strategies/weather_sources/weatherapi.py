"""WeatherAPI.com (free tier, requires WEATHERAPI_KEY)."""

from __future__ import annotations
import os
from datetime import datetime
from typing import Optional

from .base import WeatherSource


class WeatherApiSource(WeatherSource):
    name = "weatherapi"
    requires_key = True
    supported_metrics = {
        "temperature_c", "temperature_max_c", "temperature_min_c",
        "precipitation_mm", "wind_mph", "wind_kph", "snow_cm",
    }
    max_lead_time_hours = 336  # 14 days

    URL = "https://api.weatherapi.com/v1/forecast.json"

    def __init__(self):
        super().__init__()
        self.api_key = os.environ.get("WEATHERAPI_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError("WEATHERAPI_KEY not set")

    async def _fetch(self, session, lat, lon, target_dt, metric) -> Optional[float]:
        days = max(1, min(14, (target_dt.date() - datetime.utcnow().date()).days + 1))
        params = {
            "key": self.api_key,
            "q": f"{lat},{lon}",
            "days": days,
            "aqi": "no",
            "alerts": "no",
        }
        async with session.get(self.URL, params=params) as r:
            if r.status != 200:
                return None
            data = await r.json()

        date_str = target_dt.strftime("%Y-%m-%d")
        hour_str = target_dt.strftime("%Y-%m-%d %H:00")
        for d in data.get("forecast", {}).get("forecastday", []):
            if d.get("date") != date_str:
                continue
            # Daily aggregates (max/min) come from the `day` block, which is
            # pre-computed by WeatherAPI across the local day — no need to
            # scan hourly.
            if metric in ("temperature_max_c", "temperature_min_c"):
                day = d.get("day", {}) or {}
                return day.get("maxtemp_c" if metric == "temperature_max_c" else "mintemp_c")
            for h in d.get("hour", []):
                if h.get("time") == hour_str:
                    if metric == "temperature_c":
                        return h.get("temp_c")
                    if metric == "precipitation_mm":
                        return h.get("precip_mm")
                    if metric == "wind_kph":
                        return h.get("wind_kph")
                    if metric == "wind_mph":
                        return h.get("wind_mph")
                    if metric == "snow_cm":
                        return h.get("snow_cm")
        return None
