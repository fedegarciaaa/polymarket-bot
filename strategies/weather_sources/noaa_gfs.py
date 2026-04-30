"""
NOAA - api.weather.gov (no key). Only covers US coordinates.
https://www.weather.gov/documentation/services-web-api
"""

from __future__ import annotations
import re
from datetime import datetime, timezone, timedelta
from typing import Optional

from .base import WeatherSource


class NoaaGfsSource(WeatherSource):
    name = "noaa_gfs"
    requires_key = False
    supported_metrics = {
        "temperature_c", "temperature_max_c", "temperature_min_c",
        "precipitation_mm", "wind_mph", "snow_cm",
    }
    max_lead_time_hours = 168  # 7 days

    POINTS = "https://api.weather.gov/points/{lat},{lon}"

    # NOAA validTime duration like "PT1H" or "PT6H"
    DUR_RE = re.compile(r"PT(\d+)H")

    async def _fetch(self, session, lat, lon, target_dt, metric) -> Optional[float]:
        async with session.get(self.POINTS.format(lat=f"{lat:.4f}", lon=f"{lon:.4f}")) as r:
            if r.status != 200:
                return None
            meta = await r.json()
        grid_url = meta.get("properties", {}).get("forecastGridData")
        if not grid_url:
            return None

        async with session.get(grid_url) as r:
            if r.status != 200:
                return None
            grid = await r.json()

        props = grid.get("properties", {})
        # NOAA grid data exposes `maxTemperature` / `minTemperature` series
        # aggregated per forecast period (typically ~12h daytime / nighttime).
        # We pick the value whose period overlaps the target local day.
        if metric in ("temperature_max_c", "temperature_min_c"):
            key = "maxTemperature" if metric == "temperature_max_c" else "minTemperature"
            series = props.get(key, {}).get("values", [])
            if not series:
                return None
            tz_hours = round(lon / 15.0)
            day_start = datetime.combine(target_dt.date(), datetime.min.time(), tzinfo=timezone.utc) - timedelta(hours=tz_hours)
            day_end = day_start + timedelta(hours=24)
            candidates: list[float] = []
            for item in series:
                vt = item.get("validTime", "")
                if "/" not in vt:
                    continue
                start_iso, dur_str = vt.split("/", 1)
                try:
                    start = datetime.fromisoformat(start_iso).astimezone(timezone.utc)
                except Exception:
                    continue
                m = self.DUR_RE.match(dur_str)
                hours = int(m.group(1)) if m else 1
                end = start + timedelta(hours=hours)
                if end <= day_start or start >= day_end:
                    continue
                val = item.get("value")
                if val is not None:
                    candidates.append(float(val))
            if not candidates:
                return None
            return max(candidates) if metric == "temperature_max_c" else min(candidates)

        var_map = {
            "temperature_c": "temperature",               # °C native
            "precipitation_mm": "quantitativePrecipitation",  # mm
            "wind_mph": "windSpeed",                      # km/h native → convert
            "snow_cm": "snowfallAmount",                  # mm native → convert
        }
        series = props.get(var_map[metric], {}).get("values", [])
        if not series:
            return None

        target = target_dt.astimezone(timezone.utc)

        for item in series:
            vt = item.get("validTime", "")
            if "/" not in vt:
                continue
            start_iso, dur_str = vt.split("/", 1)
            try:
                start = datetime.fromisoformat(start_iso).astimezone(timezone.utc)
            except Exception:
                continue
            m = self.DUR_RE.match(dur_str)
            hours = int(m.group(1)) if m else 1
            end = start + timedelta(hours=hours)
            if start <= target < end:
                val = item.get("value")
                if val is None:
                    return None
                val = float(val)
                if metric == "wind_mph":
                    return self.kph_to_mph(val)
                if metric == "snow_cm":
                    return val / 10.0  # mm → cm
                return val
        return None
