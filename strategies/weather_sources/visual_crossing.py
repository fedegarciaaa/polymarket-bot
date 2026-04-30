"""Visual Crossing Weather (requires VISUAL_CROSSING_KEY, free tier 1000 reqs/day)."""

from __future__ import annotations
import os
from datetime import datetime, timezone
from typing import Optional

from .base import WeatherSource


class VisualCrossingSource(WeatherSource):
    name = "visual_crossing"
    requires_key = True
    supported_metrics = {
        "temperature_c", "temperature_max_c", "temperature_min_c",
        "precipitation_mm", "wind_mph", "wind_kph", "snow_cm",
    }
    max_lead_time_hours = 360  # 15 days

    URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/{date}"

    def __init__(self):
        super().__init__()
        self.api_key = os.environ.get("VISUAL_CROSSING_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError("VISUAL_CROSSING_KEY not set")

    async def _fetch(self, session, lat, lon, target_dt, metric) -> Optional[float]:
        date = target_dt.strftime("%Y-%m-%dT%H:%M:%S")
        url = self.URL.format(lat=lat, lon=lon, date=date)
        params = {"key": self.api_key, "unitGroup": "metric", "include": "hours", "contentType": "json"}
        async with session.get(url, params=params) as r:
            if r.status != 200:
                return None
            data = await r.json()

        days = data.get("days", [])
        if not days:
            return None

        # VC's `days[0]` aggregates over the local day for this location —
        # that's the window Polymarket resolves on, so take the daily max/min
        # directly instead of scanning hourly for a proxy peak-hour value.
        if metric in ("temperature_max_c", "temperature_min_c"):
            return days[0].get("tempmax" if metric == "temperature_max_c" else "tempmin")

        target_hour = target_dt.strftime("%H:00:00")
        for h in days[0].get("hours", []):
            if h.get("datetime") == target_hour:
                if metric == "temperature_c":
                    return h.get("temp")
                if metric == "precipitation_mm":
                    return h.get("precip")
                if metric == "wind_kph":
                    return h.get("windspeed")   # kph in metric
                if metric == "wind_mph":
                    v = h.get("windspeed")
                    return self.kph_to_mph(v) if v is not None else None
                if metric == "snow_cm":
                    v = h.get("snow")           # cm in metric
                    return v
        return None
