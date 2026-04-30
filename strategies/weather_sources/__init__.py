"""
Weather data sources registry.

Each source implements `WeatherSource.forecast(lat, lon, target_dt, metric)`
and returns a numeric value or None if the source failed / doesn't support
the metric / lacks an API key.

The ensemble (strategies/weather_ensemble.py) calls all enabled sources
concurrently and combines them into a mean + std.
"""

from .base import WeatherSource, METRICS
from .open_meteo import OpenMeteoSource
from .noaa_gfs import NoaaGfsSource
from .ecmwf_open import EcmwfOpenSource
from .met_norway import MetNorwaySource
from .visual_crossing import VisualCrossingSource
from .openweathermap import OpenWeatherMapSource
from .weatherapi import WeatherApiSource
from .tomorrow_io import TomorrowIoSource


ALL_SOURCE_CLASSES: dict[str, type[WeatherSource]] = {
    "open_meteo": OpenMeteoSource,
    "noaa_gfs": NoaaGfsSource,
    "ecmwf_open": EcmwfOpenSource,
    "met_norway": MetNorwaySource,
    "visual_crossing": VisualCrossingSource,
    "openweathermap": OpenWeatherMapSource,
    "weatherapi": WeatherApiSource,
    "tomorrow_io": TomorrowIoSource,
}


def build_sources(config_sources: dict) -> list[WeatherSource]:
    """
    Given `config.weather.sources` dict (name -> bool), return instantiated enabled sources.
    Sources that need a key but don't have one are skipped with a warning.
    """
    import logging
    logger = logging.getLogger("polymarket_bot.weather_sources")
    instances: list[WeatherSource] = []
    for name, enabled in (config_sources or {}).items():
        if not enabled:
            continue
        cls = ALL_SOURCE_CLASSES.get(name)
        if cls is None:
            logger.warning(f"Unknown weather source in config: {name}")
            continue
        try:
            inst = cls()
        except RuntimeError as e:
            logger.warning(f"Source {name} disabled: {e}")
            continue
        instances.append(inst)
    return instances


__all__ = ["WeatherSource", "METRICS", "ALL_SOURCE_CLASSES", "build_sources"]
