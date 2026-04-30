"""
ECMWF Open Data (keyless since 2024).
We query the lightweight index at data.ecmwf.int/forecasts/ to discover the
latest cycle, then fetch the per-location JSON via the meteomatics-free mirror
`open-meteo.com/v1/ecmwf` which wraps the ECMWF Open Data in an easy JSON API.

This avoids downloading heavy GRIB2 files while still using the ECMWF numerical model.
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional

from .base import WeatherSource


class EcmwfOpenSource(WeatherSource):
    name = "ecmwf_open"
    requires_key = False
    supported_metrics = {
        "temperature_c", "temperature_max_c", "temperature_min_c",
        "precipitation_mm", "wind_mph", "wind_kph",
    }
    max_lead_time_hours = 240  # 10 days

    BASE_URL = "https://api.open-meteo.com/v1/ecmwf"

    async def _fetch(self, session, lat, lon, target_dt, metric) -> Optional[float]:
        # ECMWF via Open-Meteo exposes hourly 3-hourly steps only — no
        # pre-aggregated daily max/min. Pull the hourly series for the target
        # local day and reduce. timezone=auto means the `time` strings are
        # already in local time.
        if metric in ("temperature_max_c", "temperature_min_c"):
            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": "temperature_2m",
                "timezone": "auto",
                "forecast_days": 10,
            }
            async with session.get(self.BASE_URL, params=params) as r:
                if r.status != 200:
                    return None
                data = await r.json()
            hours = data.get("hourly", {}).get("time", [])
            values = data.get("hourly", {}).get("temperature_2m", [])
            target_day = target_dt.strftime("%Y-%m-%d")
            vals: list[float] = [
                float(v) for t, v in zip(hours, values)
                if t.startswith(target_day) and v is not None
            ]
            if not vals:
                return None
            return max(vals) if metric == "temperature_max_c" else min(vals)

        var_map = {
            "temperature_c": "temperature_2m",
            "precipitation_mm": "precipitation",
            "wind_mph": "wind_speed_10m",
            "wind_kph": "wind_speed_10m",
        }
        var = var_map[metric]
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
        target_hour = target_dt.strftime("%Y-%m-%dT%H:00")
        for t, v in zip(hours, values):
            if t == target_hour and v is not None:
                if metric == "wind_mph":
                    return self.kph_to_mph(v)
                return float(v)
        return None
