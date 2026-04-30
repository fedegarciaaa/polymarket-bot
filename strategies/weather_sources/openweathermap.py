"""OpenWeatherMap (free tier). Uses the /forecast/hourly endpoint (One Call v3 alt.)."""

from __future__ import annotations
import os
from datetime import datetime, timezone
from typing import Optional

from .base import WeatherSource


class OpenWeatherMapSource(WeatherSource):
    name = "openweathermap"
    requires_key = True
    supported_metrics = {
        "temperature_c", "temperature_max_c", "temperature_min_c",
        "precipitation_mm", "wind_mph", "wind_kph",
    }
    max_lead_time_hours = 120  # free tier: 5 days / 3-hourly

    URL = "https://api.openweathermap.org/data/2.5/forecast"

    def __init__(self):
        super().__init__()
        self.api_key = os.environ.get("OPENWEATHERMAP_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError("OPENWEATHERMAP_KEY not set")

    async def _fetch(self, session, lat, lon, target_dt, metric) -> Optional[float]:
        params = {"lat": lat, "lon": lon, "units": "metric", "appid": self.api_key}
        async with session.get(self.URL, params=params) as r:
            if r.status != 200:
                return None
            data = await r.json()

        entries = data.get("list", [])

        # Daily aggregate: slice every 3-hour entry whose timestamp falls in
        # the target local day (city timezone offset comes from `city.timezone`
        # in seconds) and take max / min of the `temp` field.
        if metric in ("temperature_max_c", "temperature_min_c"):
            tz_offset = int((data.get("city") or {}).get("timezone") or 0)
            target_local_day = target_dt.strftime("%Y-%m-%d")
            vals: list[float] = []
            for e in entries:
                dt_utc = e.get("dt")
                if dt_utc is None:
                    continue
                local_day = datetime.fromtimestamp(dt_utc + tz_offset, tz=timezone.utc).strftime("%Y-%m-%d")
                if local_day != target_local_day:
                    continue
                t = (e.get("main") or {}).get("temp")
                if t is not None:
                    vals.append(float(t))
            if not vals:
                return None
            return max(vals) if metric == "temperature_max_c" else min(vals)

        target_epoch = int(target_dt.astimezone(timezone.utc).timestamp())

        # Find nearest entry (3-hour grid) and interpolate linearly.
        best_before = None
        best_after = None
        for e in entries:
            dt = e.get("dt")
            if dt is None:
                continue
            if dt <= target_epoch and (best_before is None or dt > best_before["dt"]):
                best_before = e
            if dt >= target_epoch and (best_after is None or dt < best_after["dt"]):
                best_after = e

        if best_before is None and best_after is None:
            return None
        pick = best_before or best_after

        if metric == "temperature_c":
            return pick.get("main", {}).get("temp")
        if metric == "precipitation_mm":
            return (pick.get("rain", {}) or {}).get("3h", 0.0) / 3.0
        if metric == "wind_kph":
            v = pick.get("wind", {}).get("speed")  # m/s
            return self.ms_to_kph(v) if v is not None else None
        if metric == "wind_mph":
            v = pick.get("wind", {}).get("speed")
            return self.ms_to_mph(v) if v is not None else None
        return None
