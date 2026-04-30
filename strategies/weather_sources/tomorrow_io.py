"""Tomorrow.io (free tier, requires TOMORROW_IO_KEY)."""

from __future__ import annotations
import os
from datetime import datetime, timezone
from typing import Optional

from .base import WeatherSource


class TomorrowIoSource(WeatherSource):
    name = "tomorrow_io"
    requires_key = True
    supported_metrics = {
        "temperature_c", "temperature_max_c", "temperature_min_c",
        "precipitation_mm", "wind_mph", "wind_kph", "snow_cm",
    }
    max_lead_time_hours = 240  # 10 days

    URL = "https://api.tomorrow.io/v4/timelines"

    def __init__(self):
        super().__init__()
        self.api_key = os.environ.get("TOMORROW_IO_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError("TOMORROW_IO_KEY not set")

    async def _fetch(self, session, lat, lon, target_dt, metric) -> Optional[float]:
        from datetime import timedelta
        # Daily endpoint: Tomorrow.io exposes temperatureMax/Min pre-aggregated
        # per local day via `timesteps=1d`. Use it directly for max/min markets.
        if metric in ("temperature_max_c", "temperature_min_c"):
            field = "temperatureMax" if metric == "temperature_max_c" else "temperatureMin"
            params = {
                "apikey": self.api_key,
                "location": f"{lat},{lon}",
                "fields": field,
                "timesteps": "1d",
                "units": "metric",
                "startTime": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:00:00Z"),
                "endTime": (target_dt + timedelta(days=1)).strftime("%Y-%m-%dT%H:00:00Z"),
            }
            async with session.get(self.URL, params=params) as r:
                if r.status != 200:
                    return None
                data = await r.json()
            intervals = data.get("data", {}).get("timelines", [{}])[0].get("intervals", [])
            target_day = target_dt.strftime("%Y-%m-%d")
            for iv in intervals:
                start = iv.get("startTime", "")
                if start.startswith(target_day):
                    return (iv.get("values") or {}).get(field)
            return None

        fields_map = {
            "temperature_c": ["temperature"],
            "precipitation_mm": ["precipitationIntensity"],
            "wind_kph": ["windSpeed"],
            "wind_mph": ["windSpeed"],
            "snow_cm": ["snowAccumulation"],
        }
        fields = fields_map[metric]
        params = {
            "apikey": self.api_key,
            "location": f"{lat},{lon}",
            "fields": ",".join(fields),
            "timesteps": "1h",
            "units": "metric",
            "startTime": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:00:00Z"),
            "endTime": (target_dt + timedelta(hours=1)).strftime("%Y-%m-%dT%H:00:00Z"),
        }
        async with session.get(self.URL, params=params) as r:
            if r.status != 200:
                return None
            data = await r.json()

        intervals = data.get("data", {}).get("timelines", [{}])[0].get("intervals", [])
        target_iso = target_dt.strftime("%Y-%m-%dT%H:00:00Z")
        for iv in intervals:
            if iv.get("startTime") != target_iso:
                continue
            vals = iv.get("values", {})
            if metric == "temperature_c":
                return vals.get("temperature")
            if metric == "precipitation_mm":
                return vals.get("precipitationIntensity")  # mm/h
            if metric == "wind_kph":
                v = vals.get("windSpeed")  # m/s in metric
                return self.ms_to_kph(v) if v is not None else None
            if metric == "wind_mph":
                v = vals.get("windSpeed")
                return self.ms_to_mph(v) if v is not None else None
            if metric == "snow_cm":
                v = vals.get("snowAccumulation")  # mm
                return v / 10.0 if v is not None else None
        return None
