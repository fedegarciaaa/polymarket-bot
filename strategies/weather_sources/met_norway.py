"""
MET Norway (no key, only User-Agent). https://api.met.no/weatherapi/locationforecast/2.0/
Global coverage, 9-day forecast, free.
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional

from .base import WeatherSource


class MetNorwaySource(WeatherSource):
    name = "met_norway"
    requires_key = False
    supported_metrics = {
        "temperature_c", "temperature_max_c", "temperature_min_c",
        "precipitation_mm", "wind_mph", "wind_kph",
    }
    max_lead_time_hours = 216  # 9 days

    URL = "https://api.met.no/weatherapi/locationforecast/2.0/compact"

    async def _fetch(self, session, lat, lon, target_dt, metric) -> Optional[float]:
        params = {"lat": f"{lat:.4f}", "lon": f"{lon:.4f}"}
        async with session.get(self.URL, params=params) as r:
            if r.status != 200:
                return None
            data = await r.json()

        series = data.get("properties", {}).get("timeseries", [])

        # MET Norway only exposes hourly timeseries in UTC — we approximate the
        # local day by taking the 24h window centered on target_dt 12:00 UTC
        # and reducing to max/min. Longitude offset corrects for extreme tz.
        if metric in ("temperature_max_c", "temperature_min_c"):
            from datetime import timedelta
            tz_hours = round(lon / 15.0)
            day_start_utc = datetime.combine(target_dt.date(), datetime.min.time(), tzinfo=timezone.utc) - timedelta(hours=tz_hours)
            day_end_utc = day_start_utc + timedelta(hours=24)
            vals: list[float] = []
            for entry in series:
                try:
                    t = datetime.fromisoformat(entry.get("time", "").replace("Z", "+00:00"))
                except Exception:
                    continue
                if not (day_start_utc <= t < day_end_utc):
                    continue
                v = (entry.get("data", {}).get("instant", {}).get("details", {}) or {}).get("air_temperature")
                if v is not None:
                    vals.append(float(v))
            if not vals:
                return None
            return max(vals) if metric == "temperature_max_c" else min(vals)

        target_iso = target_dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:00:00Z")
        for entry in series:
            t = entry.get("time")
            if t != target_iso:
                continue
            inst = entry.get("data", {}).get("instant", {}).get("details", {})
            one_h = entry.get("data", {}).get("next_1_hours", {}).get("details", {})
            if metric == "temperature_c":
                return inst.get("air_temperature")
            if metric == "precipitation_mm":
                return one_h.get("precipitation_amount")
            if metric == "wind_kph":
                v = inst.get("wind_speed")  # m/s
                return self.ms_to_kph(v) if v is not None else None
            if metric == "wind_mph":
                v = inst.get("wind_speed")
                return self.ms_to_mph(v) if v is not None else None
        return None
