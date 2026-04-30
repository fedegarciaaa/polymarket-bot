"""
Weather Bot Strategy — exploit meteorological edge on Polymarket weather markets,
now built on a multi-source ensemble + confidence engine.

Pipeline for each market:
  1. Filter: only weather markets (WEATHER_CONFIRM_PATTERNS, EXCLUDE_PATTERNS).
  2. Parse: extract location, target_date, condition_type, threshold/range.
  3. Ensemble forecast: fetch the metric from N weather sources in parallel.
  4. Probability: compute P(condition) from ensemble mean/std via normal CDF.
  5. Blend with market (Bayesian, market 60% / ensemble 40% by default).
  6. Confidence engine: score 0-100, apply vetos, decide pass/fail.
  7. Return only opportunities that pass the confidence threshold.

Every rejection is logged as OPPORTUNITY_SKIP with a standardized reason_code
and persisted to the `opportunity_skips` DB table for later analysis.
"""

from __future__ import annotations

import math
import re
import logging
import asyncio
import json
import requests
from datetime import datetime, date, time, timezone, timedelta
from typing import Optional

from .weather_ensemble import ensemble_forecast, blend_with_market, EnsembleResult
from .confidence_engine import ConfidenceEngine, ConfidenceBreakdown
from .city_bias import get_default_table as get_city_bias_table
from structured_logger import get_logger, Event, SkipReason

logger = logging.getLogger("polymarket_bot.weather")

# ─────────────────────────────────────────────────────────────────
# City coordinates (lat, lon)
# ─────────────────────────────────────────────────────────────────
KNOWN_CITIES: dict[str, tuple[float, float]] = {
    # Asia
    "hong kong":        (22.3193,  114.1694),
    "shanghai":         (31.2304,  121.4737),
    "beijing":          (39.9042,  116.4074),
    "tokyo":            (35.6762,  139.6503),
    "guangzhou":        (23.1291,  113.2644),
    "shenzhen":         (22.5431,  114.0579),
    "wuhan":            (30.5928,  114.3055),
    "chengdu":          (30.5728,  104.0668),
    "chongqing":        (29.5630,  106.5516),
    "nanjing":          (32.0603,  118.7969),
    "hangzhou":         (30.2741,  120.1551),
    "xian":             (34.3416,  108.9398),
    "tianjin":          (39.3434,  117.3616),
    "shenyang":         (41.7968,  123.4291),
    "harbin":           (45.8038,  126.5349),
    "kunming":          (25.0389,  102.7183),
    "taipei":           (25.0330,  121.5654),
    "osaka":            (34.6937,  135.5023),
    "seoul":            (37.5665,  126.9780),
    "bangkok":          (13.7563,  100.5018),
    "singapore":        (1.3521,   103.8198),
    "kuala lumpur":     (3.1390,   101.6869),
    "jakarta":          (-6.2088,  106.8456),
    "manila":           (14.5995,  120.9842),
    "mumbai":           (19.0760,   72.8777),
    "delhi":            (28.6139,   77.2090),
    "new delhi":        (28.6139,   77.2090),
    "kolkata":          (22.5726,   88.3639),
    "chennai":          (13.0827,   80.2707),
    "bangalore":        (12.9716,   77.5946),
    "karachi":          (24.8607,   67.0011),
    "dhaka":            (23.8103,   90.4125),
    "hanoi":            (21.0245,  105.8412),
    "ho chi minh":      (10.8231,  106.6297),
    "ulaanbaatar":      (47.8864,  106.9057),
    "yangon":           (16.8661,   96.1951),
    "colombo":          (6.9271,    79.8612),
    "dubai":            (25.2048,   55.2708),
    "abu dhabi":        (24.4539,   54.3773),
    "riyadh":           (24.7136,   46.6753),
    "tehran":           (35.6892,   51.3890),
    "istanbul":         (41.0082,   28.9784),
    "ankara":           (39.9334,   32.8597),
    "tel aviv":         (32.0853,   34.7818),
    # Europe
    "london":           (51.5074,   -0.1278),
    "paris":            (48.8566,    2.3522),
    "berlin":           (52.5200,   13.4050),
    "madrid":           (40.4168,   -3.7038),
    "rome":             (41.9028,   12.4964),
    "amsterdam":        (52.3676,    4.9041),
    "brussels":         (50.8503,    4.3517),
    "vienna":           (48.2082,   16.3738),
    "zurich":           (47.3769,    8.5417),
    "stockholm":        (59.3293,   18.0686),
    "oslo":             (59.9139,   10.7522),
    "copenhagen":       (55.6761,   12.5683),
    "dublin":           (53.3498,   -6.2603),
    "lisbon":           (38.7169,   -9.1395),
    "moscow":           (55.7558,   37.6173),
    "warsaw":           (52.2297,   21.0122),
    "budapest":         (47.4979,   19.0402),
    "barcelona":        (41.3851,    2.1734),
    "munich":           (48.1351,   11.5820),
    "frankfurt":        (50.1109,    8.6821),
    "milan":            (45.4654,    9.1859),
    "athens":           (37.9838,   23.7275),
    "prague":           (50.0755,   14.4378),
    "bucharest":        (44.4268,   26.1025),
    "sofia":            (42.6977,   23.3219),
    "zagreb":           (45.8150,   15.9819),
    "kyiv":             (50.4501,   30.5234),
    # Americas
    "new york":         (40.7128,  -74.0060),
    "new york city":    (40.7128,  -74.0060),
    "nyc":              (40.7128,  -74.0060),
    "los angeles":      (34.0522, -118.2437),
    "chicago":          (41.8781,  -87.6298),
    "houston":          (29.7604,  -95.3698),
    "miami":            (25.7617,  -80.1918),
    "dallas":           (32.7767,  -96.7970),
    "san francisco":    (37.7749, -122.4194),
    "seattle":          (47.6062, -122.3321),
    "boston":           (42.3601,  -71.0589),
    "denver":           (39.7392, -104.9903),
    "atlanta":          (33.7490,  -84.3880),
    "phoenix":          (33.4484, -112.0740),
    "philadelphia":     (39.9526,  -75.1652),
    "san diego":        (32.7157, -117.1611),
    "minneapolis":      (44.9778,  -93.2650),
    "portland":         (45.5231, -122.6765),
    "las vegas":        (36.1699, -115.1398),
    "nashville":        (36.1627,  -86.7816),
    "orlando":          (28.5383,  -81.3792),
    "tampa":            (27.9506,  -82.4572),
    "charlotte":        (35.2271,  -80.8431),
    "detroit":          (42.3314,  -83.0458),
    "baltimore":        (39.2904,  -76.6122),
    "washington":       (38.9072,  -77.0369),
    "new orleans":      (29.9511,  -90.0715),
    "kansas city":      (39.0997,  -94.5786),
    "salt lake city":   (40.7608, -111.8910),
    "austin":           (30.2672,  -97.7431),
    "toronto":          (43.6532,  -79.3832),
    "vancouver":        (49.2827, -123.1207),
    "montreal":         (45.5017,  -73.5673),
    "mexico city":      (19.4326,  -99.1332),
    "buenos aires":     (-34.6037, -58.3816),
    "sao paulo":        (-23.5505, -46.6333),
    "rio de janeiro":   (-22.9068, -43.1729),
    "bogota":           (4.7110,   -74.0721),
    "lima":             (-12.0464, -77.0428),
    "santiago":         (-33.4489, -70.6693),
    # Oceania
    "sydney":           (-33.8688,  151.2093),
    "melbourne":        (-37.8136,  144.9631),
    "brisbane":         (-27.4705,  153.0260),
    "perth":            (-31.9505,  115.8605),
    "auckland":         (-36.8485,  174.7633),
    "wellington":       (-41.2866,  174.7756),
    # Africa
    "cairo":            (30.0444,   31.2357),
    "lagos":            (6.5244,     3.3792),
    "nairobi":          (-1.2921,   36.8219),
    "johannesburg":     (-26.2041,  28.0473),
    "cape town":        (-33.9249,  18.4241),
    "casablanca":       (33.5731,   -7.5898),
    "accra":            (5.6037,    -0.1870),
    "dar es salaam":    (-6.7924,   39.2083),
}

WEATHER_CONFIRM_PATTERNS = [
    r'highest temperature',
    r'lowest temperature',
    r'high temperature',
    r'low temperature',
    r'temperature in \w',
    r'\d+\s*°\s*[cCfF]',
    r'\d+\s*degrees',
    r'will it rain',
    r'will there be rain',
    r'measurable precipitation',
    r'rainfall',
    r'snowfall',
    r'will it snow',
    r'precipitation in',
    r'hurricane',
    r'tornado',
    r'blizzard',
]

EXCLUDE_PATTERNS = [
    r'price of (bitcoin|ethereum|solana|bnb|xrp|eth|btc)',
    r'will (btc|eth|sol|bnb|xrp)\b',
    r'ceasefire',
    r'up or down',
    r'esport',
    r'eurovision',
    r'election',
    r'impeach',
]

OPEN_METEO_GEOCODING = "https://geocoding-api.open-meteo.com/v1/search"

_geocoding_cache: dict[str, Optional[tuple[str, float, float]]] = {}


# ─────────────────────────────────────────────────────────────────
# Main strategy
# ─────────────────────────────────────────────────────────────────
class WeatherBotStrategy:
    """
    Ensemble-driven weather strategy. `sources` is a list of WeatherSource
    instances built via strategies.weather_sources.build_sources(config).
    `db` is the Database instance (for rolling WR + skip logging + reliability).
    """

    def __init__(self, config: dict, sources: list, db):
        # Accept config either under `weather:` (current) or legacy
        # `strategies.weather_bot:`. `weather:` wins when both are present.
        wb = dict(config.get("strategies", {}).get("weather_bot", {}))
        wb.update(config.get("weather", {}) or {})
        self.enabled          = wb.get("enabled", True)
        self.min_edge         = wb.get("min_edge", 0.05)
        self.max_days_ahead   = wb.get("max_days_ahead", 7)
        self.min_liquidity    = wb.get("min_liquidity", 200)
        self.min_volume       = wb.get("min_volume", 500)
        self.kelly_fraction   = wb.get("kelly_fraction", 0.15)
        self.max_position_pct = wb.get("max_position_pct", 0.03)
        self.ensemble_weight  = wb.get("ensemble_weight", 0.50)
        # Default ensemble source weights from config — used as fallback when
        # `source_reliability` lacks ≥5 trades for a source. Mergeados con los
        # pesos de DB en el sitio de uso (ver loop principal).
        self.default_source_weights: dict[str, float] = dict(
            wb.get("default_source_weights", {}) or {}
        )

        # Thresholds adaptive by horizon bucket.
        self.min_prob_by_days = wb.get("min_prob_to_bet_by_days", {
            "1-3": 0.70, "4-5": 0.75, "6-7": 0.80,
        })
        self.near_resolved_low  = float(wb.get("near_resolved_low", 0.02))
        self.near_resolved_high = float(wb.get("near_resolved_high", 0.985))

        # Edge-quality gates — see _passes_edge_quality docstring.
        self.min_abs_edge   = float(wb.get("min_abs_edge",   0.025))
        self.min_kelly_edge = float(wb.get("min_kelly_edge", 0.20))

        # Hours-to-resolution window — specialize where the bot performs best.
        # Under `min_hours_to_resolution` the market enters a noise-dominated,
        # news-driven regime that our ensemble can't model (actual observations
        # leaking into the market, intraday weather reports). Over
        # `max_hours_to_resolution` the ensemble std is too wide and our prob
        # estimates get washed out. Defaults target the 12h–7d sweet spot.
        self.min_hours_to_resolution = float(wb.get("min_hours_to_resolution", 12.0))
        self.max_hours_to_resolution = float(wb.get("max_hours_to_resolution", 168.0))

        self.sources = sources
        self.db = db
        self.config = config
        self.confidence_engine = ConfidenceEngine(config)

        # In-memory scratchpad for the "N=2 veto confirmation" rule: track the
        # vetos emitted last cycle per market. An entry is only allowed if the
        # current cycle is clean AND the previous cycle was clean for that market.
        # This prevents opening a position on a flicker of good data, right before
        # the next ensemble pass flips a hard veto (the trade-4 / trade-22 failure).
        self._last_cycle_vetos: dict[str, list[str]] = {}
        self._pending_vetos: dict[str, list[str]] = {}
        self._risky_veto_codes = {
            "HIGH_ENSEMBLE_STD", "EDGE_SUSPICIOUS",
            "FEW_SOURCES", "NO_ENSEMBLE_DATA",
        }

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "WeatherBot/2.0"})

        logger.info(
            f"WeatherBotStrategy v3: enabled={self.enabled} sources={len(sources)} "
            f"min_edge={self.min_edge:.0%} min_prob_by_days={self.min_prob_by_days} "
            f"max_days={self.max_days_ahead}"
        )

    def _min_prob_for_days(self, days_ahead: int) -> float:
        if days_ahead <= 3:
            return float(self.min_prob_by_days.get("1-3", 0.70))
        if days_ahead <= 5:
            return float(self.min_prob_by_days.get("4-5", 0.75))
        return float(self.min_prob_by_days.get("6-7", 0.80))

    def _passes_edge_quality(self, prob_real: float, price: float) -> tuple[bool, str]:
        """
        Enforce BOTH an absolute edge and a minimum Kelly edge ratio.

        `kelly_edge = (prob_real - price) / (1 - price)` — the fraction of
        remaining upside captured as edge. This is what true Kelly sizes on.

        Requiring `kelly_edge >= min_kelly_edge` lets us accept high-price
        "safe bets" (buy @0.90 when prob=0.97 → kelly_edge=0.70) while
        rejecting low-quality trades (buy @0.87 when prob=0.91 → kelly_edge=0.31).
        Pair it with a small absolute floor so 0.50→0.51 noise doesn't qualify.
        """
        if price <= 0 or price >= 1:
            return False, "price_out_of_range"
        abs_edge = prob_real - price
        if abs_edge < self.min_abs_edge:
            return False, f"abs_edge {abs_edge:+.3f} < {self.min_abs_edge:.3f}"
        kelly_edge = abs_edge / (1.0 - price)
        if kelly_edge < self.min_kelly_edge:
            return False, f"kelly_edge {kelly_edge:+.3f} < {self.min_kelly_edge:.3f}"
        return True, f"abs={abs_edge:+.3f} kelly={kelly_edge:+.3f}"

    # ── Public ──────────────────────────────────────────────────

    async def find_opportunities(
        self, markets: list[dict], cycle_id: Optional[str] = None
    ) -> list[dict]:
        if not self.enabled:
            return []

        weather_markets: list[dict] = []
        for market in markets:
            q = market.get("question", "")
            if self._is_weather_market(q):
                weather_markets.append(market)

        slog = get_logger()
        slog.log(Event.MARKETS_SCANNED, "weather", {
            "total": len(markets),
            "weather": len(weather_markets),
        }, cycle_id=cycle_id)

        # Evaluate markets concurrently (ensemble fetches are I/O bound)
        tasks = [self._evaluate_market(m, cycle_id) for m in weather_markets]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        opportunities: list[dict] = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning(f"Market eval exception: {r}")
                continue
            if r is not None:
                opportunities.append(r)

        # Promote this cycle's veto-sightings into "last cycle" so the next
        # scan can compare. We swap atomically so only what we saw THIS pass
        # survives into the next round.
        self._last_cycle_vetos = self._pending_vetos
        self._pending_vetos = {}

        opportunities.sort(key=lambda x: x.get("risk_score", 0), reverse=True)

        logger.info(
            f"Weather Bot: {len(weather_markets)} weather markets → "
            f"{len(opportunities)} passed prob_real threshold"
        )
        for opp in opportunities[:10]:
            logger.info(
                f"  [{opp['weather_type']:<14}] {opp['side']} @ {opp['price']:.3f} | "
                f"edge={opp['edge']:+.1%}  risk={opp['risk_score']:.0f}  "
                f"prob={opp['prob_real_estimated']:.2f}  "
                f"sources={opp['ensemble_sources_used']} | "
                f"{opp['market_question'][:55]}"
            )
        return opportunities

    # ── Per-market evaluation ───────────────────────────────────

    async def _evaluate_market(
        self, market: dict, cycle_id: Optional[str]
    ) -> Optional[dict]:
        trace_id = get_logger().new_trace_id()
        q         = market.get("question", "")
        market_id = market.get("id", "")
        price_yes = float(market.get("price_yes", 0.5))
        price_no  = float(market.get("price_no",  0.5))
        liquidity = float(market.get("liquidity", 0))
        volume    = float(market.get("volume_24h", 0))

        def _skip(reason: str, extra: Optional[dict] = None) -> None:
            extra = extra or {}
            get_logger().log_opportunity_skip(
                market_id=market_id,
                market_question=q,
                reason_code=reason,
                reason_detail=str(extra.get("reason_detail", "")),
                confidence_score=extra.get("confidence_score"),
                ensemble_std=extra.get("ensemble_std"),
                sources_used=extra.get("sources_used_count"),
                prob_market=extra.get("prob_market"),
                prob_estimated=extra.get("prob_estimated"),
                edge=extra.get("edge"),
                trace_id=trace_id,
            )
            try:
                self.db.log_opportunity_skip({
                    "cycle_id": cycle_id,
                    "market_id": market_id,
                    "market_question": q,
                    "reason_code": reason,
                    "reason_detail": json.dumps(extra, default=str)[:500],
                    "confidence_score": extra.get("confidence_score"),
                    "ensemble_std": extra.get("ensemble_std"),
                    "sources_used": extra.get("sources_used_count"),
                    "prob_market": extra.get("prob_market"),
                    "prob_estimated": extra.get("prob_estimated"),
                    "edge": extra.get("edge"),
                })
            except Exception as exc:
                logger.debug(f"log_opportunity_skip failed: {exc}")

        if liquidity < self.min_liquidity:
            _skip(SkipReason.LOW_LIQUIDITY, {"liquidity": liquidity})
            return None
        if volume < self.min_volume:
            _skip(SkipReason.LOW_VOLUME, {"volume": volume})
            return None
        if price_yes < self.near_resolved_low or price_yes > self.near_resolved_high:
            _skip(SkipReason.NEAR_RESOLVED, {"price_yes": price_yes})
            return None

        parsed = self._parse_weather_question(q, market.get("end_date", ""))
        if not parsed:
            _skip(SkipReason.PARSE_FAILED)
            return None
        location_name, lat, lon, target_date, condition_type, params_dict = parsed

        today = datetime.now(timezone.utc).date()
        days_ahead = (target_date - today).days
        if days_ahead < 0:
            _skip(SkipReason.PAST_DATE)
            return None
        if days_ahead > self.max_days_ahead:
            _skip(SkipReason.TOO_FAR_AHEAD, {"days_ahead": days_ahead})
            return None

        # Target resolution = the whole calendar day in local time. We no
        # longer pick a proxy "peak hour" — that was the main v1 bias (sampling
        # 15:00 local missed real highs at 17:00, etc.). Sources now query
        # *daily* endpoints (max / min over the local day) and we just hand
        # them the target_date. We still pass a datetime so downstream helpers
        # keep their signature; midnight-UTC is a stable, neutral anchor.
        target_dt = datetime.combine(target_date, time(0, 0), tzinfo=timezone.utc)

        # For hours_to_resolution we use the market's actual end_date (the
        # moment Polymarket closes the market) rather than midnight UTC of the
        # target_date, which can be hours before the real close for non-UTC
        # cities (e.g. Houston CDT midnight = April+1 05:00 UTC).
        end_date_str = market.get("end_date", "")
        resolution_dt = target_dt
        if end_date_str:
            try:
                resolution_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass
        metric = self._metric_for(condition_type)
        if metric is None:
            _skip(SkipReason.UNSUPPORTED_METRIC, {"condition_type": condition_type})
            return None

        # Load source reliability weights (source → 1/(1+brier)) if available.
        # Merge: DB reliability weights wins per-source; sources without ≥5
        # trades fall back to `default_source_weights` from config (ECMWF 0.30,
        # GFS 0.20, etc. — accuracy histórica documentada de bots rentables).
        try:
            db_weights = self.db.get_source_reliability_weights(metric) or {}
        except Exception:
            db_weights = {}
        weights: Optional[dict[str, float]] = None
        if db_weights or self.default_source_weights:
            weights = dict(self.default_source_weights)
            weights.update(db_weights)
            weights = weights or None

        ensemble = await ensemble_forecast(
            self.sources, lat, lon, target_dt, metric,
            trace_id=trace_id, market_id=market_id,
            source_weights=weights,
        )

        # City-bias correction: ensemble forecasts have systematic errors for
        # certain (city, metric, month) cells. Subtract the historical signed
        # bias from ensemble.mean before computing the probability. The seed
        # table is hand-tuned from the 28-Apr diagnostic (Seoul/Jakarta/Lucknow
        # spring-2026) and gets overwritten by recompute_from_history once we
        # have ≥5 closed trades per cell. Applied only to numeric metrics; for
        # P(condition) the corrected mean is what we feed to the CDF.
        bias_applied = 0.0
        if ensemble.mean is not None and metric:
            try:
                adj_mean, bias_applied = get_city_bias_table().adjust_mean(
                    ensemble.mean, location_name, metric, target_date.month
                )
                if bias_applied != 0.0:
                    ensemble.mean = adj_mean
                    logger.debug(
                        f"city_bias applied: {location_name} {metric} m={target_date.month} "
                        f"bias={bias_applied:+.2f} → adjusted_mean={adj_mean:.2f}"
                    )
            except Exception as exc:
                logger.warning(f"city_bias lookup failed: {exc}")

        # Turn ensemble into a P(condition is TRUE) = P(YES)
        prob_ensemble = self._condition_probability(ensemble, condition_type, params_dict)
        if prob_ensemble is None:
            _skip(SkipReason.NO_ENSEMBLE_DATA, {
                "sources_used_count": ensemble.n_used,
                "sources_used_list": ensemble.sources_used,
                "sources_failed": ensemble.sources_failed,
            })
            return None

        # Bayesian blend with market baseline
        prob_yes = blend_with_market(prob_ensemble, price_yes, self.ensemble_weight)

        # Pick side by edge
        edge_yes = prob_yes - price_yes
        edge_no  = (1.0 - prob_yes) - price_no
        if edge_yes >= edge_no:
            side = "YES"; price = price_yes; our_prob = prob_yes
            prob_market = price_yes
        else:
            side = "NO";  price = price_no;  our_prob = 1.0 - prob_yes
            prob_market = price_no

        edge = our_prob - price

        # Hours to resolution (uses real market end_date, not midnight UTC anchor)
        hours_to_resolution = max(0.0, (resolution_dt - datetime.now(timezone.utc)).total_seconds() / 3600.0)

        # Specialization window — reject markets outside the band where our
        # ensemble has historical edge (see __init__ notes on sweet spot).
        if hours_to_resolution < self.min_hours_to_resolution:
            _skip(SkipReason.TOO_CLOSE_TO_RESOLUTION, {
                "hours_to_resolution": round(hours_to_resolution, 2),
                "min_hours_required": self.min_hours_to_resolution,
            })
            return None
        if hours_to_resolution > self.max_hours_to_resolution:
            _skip(SkipReason.TOO_FAR_FROM_RESOLUTION, {
                "hours_to_resolution": round(hours_to_resolution, 2),
                "max_hours_allowed": self.max_hours_to_resolution,
            })
            return None

        # Rolling side WR
        try:
            side_wr = self.db.get_side_rolling_winrate(side, last_n=20)
        except Exception:
            side_wr = 0.5  # neutral if DB missing

        # Hard vetos only (no composite score blocks entry anymore)
        breakdown: ConfidenceBreakdown = self.confidence_engine.evaluate(
            ensemble_result=ensemble,
            prob_estimated=our_prob,
            prob_market=prob_market,
            liquidity=liquidity,
            volume_24h=volume,
            hours_to_resolution=hours_to_resolution,
            side=side,
            side_rolling_wr=side_wr,
            is_pyramid=False,
            days_ahead=days_ahead,
        )

        get_logger().log_confidence_eval(
            trace_id=trace_id,
            market_id=market_id,
            score=breakdown.final_score,
            breakdown=breakdown.to_dict().get("components", {}),
            vetos=breakdown.vetos,
            passed=breakdown.passed,
        )

        # Record this cycle's vetos for the market so the NEXT scan can apply
        # the N=2 confirmation gate.
        if breakdown.vetos:
            self._pending_vetos[market_id] = list(breakdown.vetos)

        if not breakdown.passed:
            reason = breakdown.vetos[0] if breakdown.vetos else SkipReason.LOW_CONFIDENCE
            _skip(reason, {
                "confidence_score": breakdown.final_score,
                "vetos": breakdown.vetos,
                "missing_sources": ensemble.sources_failed,
            })
            return None

        # N=2 confirmation gate removed — added too much friction for minimal benefit
        # (only blocked 3 trades/day, but compounded with other fixes)

        # Primary entry criterion: prob_real >= min_prob_for_days(days_ahead)
        min_prob = self._min_prob_for_days(days_ahead)
        if our_prob < min_prob:
            _skip(SkipReason.LOW_CONFIDENCE, {
                "prob_estimated": round(our_prob, 4),
                "min_prob_required": round(min_prob, 4),
                "days_ahead": days_ahead,
            })
            return None

        # Edge-quality gate: absolute edge + Kelly-edge ratio. This is what
        # allows safe-bet-style entries (high price, high confidence, thin edge
        # but high kelly_edge) while blocking low-quality entries at mid prices.
        ok, detail = self._passes_edge_quality(our_prob, price)
        if not ok:
            _skip(SkipReason.LOW_EDGE, {
                "edge": round(edge, 4),
                "price": round(price, 4),
                "prob_real": round(our_prob, 4),
                "reason_detail": detail,
            })
            return None

        # Fractional Kelly
        kelly_f = 0.0
        if 0.0 < price < 1.0:
            b = (1.0 - price) / price
            kelly_f = max(0.0, (our_prob * b - (1.0 - our_prob)) / b) * self.kelly_fraction

        risk_score = round(our_prob * 100, 1)
        opp = {
            "market_id":             market_id,
            "market_question":       q,
            "strategy":              "weather_bot",
            "side":                  side,
            "price":                 round(price, 4),
            "price_yes":             round(price_yes, 4),
            "price_no":              round(price_no,  4),
            "prob_ensemble":         round(prob_ensemble, 4),
            "prob_blended":          round(prob_yes, 4),
            "prob_real_estimated":   round(our_prob, 4),
            "prob_real":             round(our_prob, 4),
            "risk_score":            risk_score,
            "min_prob_required":     round(min_prob, 4),
            "prob_market":           round(prob_market, 4),
            "edge":                  round(edge, 4),
            "ev_calculated":         round(edge, 4),
            "kelly_fraction":        round(kelly_f, 4),
            "weather_type":          condition_type,
            "metric":                metric,
            "condition_params":      params_dict,
            "location":              location_name,
            "lat":                   lat,
            "lon":                   lon,
            "target_date":           target_date.isoformat(),
            "target_dt":             target_dt.isoformat(),
            "days_ahead":            days_ahead,
            "days_to_expiry":        days_ahead,
            "hours_to_resolution":   round(hours_to_resolution, 2),
            "volume_24h":            volume,
            "liquidity":             liquidity,
            "end_date":              market.get("end_date", ""),
            "category":              market.get("category", "Weather"),
            "clob_token_ids":        market.get("clob_token_ids", ""),
            "token_yes":             market.get("token_yes", ""),
            "token_no":              market.get("token_no",  ""),
            "outcomes":              market.get("outcomes",  []),
            "confidence_score":      risk_score,   # back-compat: now mirrors risk_score
            "confidence_breakdown":  breakdown.to_dict(),
            "ensemble_mean":         ensemble.mean,
            "ensemble_std":          ensemble.std,
            "city_bias_applied":     round(bias_applied, 3),
            "ensemble_sources_used": ensemble.n_used,
            "sources_json":          json.dumps(ensemble.per_source),
            "sources_per_source":    ensemble.per_source,
            "sources_used":          list(ensemble.sources_used),
            "sources_failed":        list(ensemble.sources_failed),
            "vetos_triggered":       list(breakdown.vetos),
            "side_wr_at_entry":      round(side_wr, 3),
            "trace_id":              trace_id,
            "reasoning": (
                f"Ensemble {ensemble.n_used}-src {metric}={ensemble.mean:.2f}"
                f"±{(ensemble.std or 0):.2f} → P(YES)={prob_ensemble:.1%} "
                f"blended={prob_yes:.1%} vs market={price_yes:.1%}, edge={edge:+.1%}, "
                f"prob_real={our_prob:.2f} (min={min_prob:.2f}@{days_ahead}d) → risk={risk_score:.0f}."
            ),
        }

        get_logger().log(Event.OPPORTUNITY_FOUND, "weather", {
            "market_id": market_id,
            "side": side,
            "edge": round(edge, 4),
            "confidence_score": round(breakdown.final_score, 1),
            "sources_used": ensemble.n_used,
        }, trace_id=trace_id, cycle_id=cycle_id)

        return opp

    # ── Condition → metric / probability ────────────────────────

    @staticmethod
    def _metric_for(condition_type: str) -> Optional[str]:
        # Temperature markets resolve on the *daily* max (or min for
        # "below"/"coldest" questions) of the target local day, NOT on a
        # spot reading at an arbitrary hour. We route each condition to the
        # matching daily metric; the sources must query daily endpoints.
        if condition_type in ("temp_above_c", "temp_range_c"):
            return "temperature_max_c"
        if condition_type == "temp_below_c":
            return "temperature_min_c"
        if condition_type in ("precipitation", "rain"):
            return "precipitation_mm"
        if condition_type == "snow":
            return "snow_cm"
        if condition_type == "wind_mph":
            return "wind_mph"
        return None

    @staticmethod
    def _condition_probability(
        ensemble: EnsembleResult, condition_type: str, params: dict
    ) -> Optional[float]:
        """Translate ensemble (mean, std) into P(condition is TRUE)."""
        if ensemble.mean is None:
            return None
        std = ensemble.std or 0.0

        if condition_type == "temp_above_c":
            thr = float(params["threshold"])
            if std <= 0:
                return 1.0 if ensemble.mean >= thr else 0.0
            p = ensemble.prob_over_threshold(thr)
            return p
        if condition_type == "temp_below_c":
            thr = float(params["threshold"])
            if std <= 0:
                return 1.0 if ensemble.mean <= thr else 0.0
            p = ensemble.prob_over_threshold(thr)
            return 1.0 - p if p is not None else None
        if condition_type == "temp_range_c":
            lo = float(params["lo"]); hi = float(params["hi"])
            if std <= 0:
                return 1.0 if lo <= ensemble.mean <= hi else 0.0
            return ensemble.prob_in_range(lo, hi)
        if condition_type in ("precipitation", "rain"):
            # "measurable precipitation" ≈ > 0.1 mm
            thr = float(params.get("threshold", 0.1))
            if std <= 0:
                return 1.0 if ensemble.mean >= thr else 0.0
            return ensemble.prob_over_threshold(thr)
        if condition_type == "snow":
            thr = float(params.get("threshold", 0.1))
            if std <= 0:
                return 1.0 if ensemble.mean >= thr else 0.0
            return ensemble.prob_over_threshold(thr)
        return None

    # ── Weather filter / parser (unchanged from v1) ─────────────

    def _is_weather_market(self, question: str) -> bool:
        q = question.lower()
        if any(re.search(p, q) for p in EXCLUDE_PATTERNS):
            return False
        return any(re.search(p, q) for p in WEATHER_CONFIRM_PATTERNS)

    def _parse_weather_question(self, question: str, end_date_str: str) -> Optional[tuple]:
        q = question.lower()

        m = re.search(r'between\s+(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*°\s*([fFcC])', question)
        if m:
            lo, hi, unit = float(m.group(1)), float(m.group(2)), m.group(3).upper()
            if unit == 'F':
                lo_c = (lo - 32) * 5 / 9
                hi_c = (hi - 32) * 5 / 9
            else:
                lo_c, hi_c = lo, hi
            condition_type = "temp_range_c"
            params_dict = {"lo": lo_c, "hi": hi_c}

        elif re.search(r'\d+\s*°\s*F\s+or\s+below', question, re.IGNORECASE):
            m2 = re.search(r'(\d+(?:\.\d+)?)\s*°\s*F', question, re.IGNORECASE)
            threshold_c = (float(m2.group(1)) - 32) * 5 / 9
            condition_type = "temp_below_c"
            params_dict = {"threshold": threshold_c}

        elif re.search(r'\d+\s*°\s*C\s+or\s+below', question, re.IGNORECASE):
            m2 = re.search(r'(\d+(?:\.\d+)?)\s*°\s*C', question, re.IGNORECASE)
            condition_type = "temp_below_c"
            params_dict = {"threshold": float(m2.group(1))}

        elif re.search(r'\d+\s*°\s*C\s+or\s+(higher|above)', question, re.IGNORECASE):
            m2 = re.search(r'(\d+(?:\.\d+)?)\s*°\s*C', question, re.IGNORECASE)
            condition_type = "temp_above_c"
            params_dict = {"threshold": float(m2.group(1))}

        elif re.search(r'\d+\s*°\s*F\s+or\s+(higher|above)', question, re.IGNORECASE):
            m2 = re.search(r'(\d+(?:\.\d+)?)\s*°\s*F', question, re.IGNORECASE)
            condition_type = "temp_above_c"
            params_dict = {"threshold": (float(m2.group(1)) - 32) * 5 / 9}

        elif re.search(r'\d+\s*°\s*C', question, re.IGNORECASE):
            # No explicit direction — treat as exact target (±1°C range)
            m2 = re.search(r'(\d+(?:\.\d+)?)\s*°\s*C', question, re.IGNORECASE)
            thr = float(m2.group(1))
            condition_type = "temp_range_c"
            params_dict = {"lo": thr - 1.0, "hi": thr + 1.0}

        elif re.search(r'\d+\s*°\s*F', question, re.IGNORECASE):
            # No explicit direction — treat as exact target (±1°C range)
            m2 = re.search(r'(\d+(?:\.\d+)?)\s*°\s*F', question, re.IGNORECASE)
            threshold_c = (float(m2.group(1)) - 32) * 5 / 9
            condition_type = "temp_range_c"
            params_dict = {"lo": threshold_c - 1.0, "hi": threshold_c + 1.0}

        elif re.search(r'\d+\s+degrees?\s+or\s+(below|lower)', q):
            m2 = re.search(r'(\d+(?:\.\d+)?)\s+degrees?', question, re.IGNORECASE)
            condition_type = "temp_below_c"
            params_dict = {"threshold": float(m2.group(1))}

        elif re.search(r'(\d+)\s*degrees?', q):
            m2 = re.search(r'(\d+(?:\.\d+)?)\s*degrees?', question, re.IGNORECASE)
            condition_type = "temp_above_c"
            params_dict = {"threshold": float(m2.group(1))}

        elif any(p in q for p in ['rainfall', 'will it rain', 'rain in', 'precipitation', 'measurable precip']):
            condition_type = "precipitation"
            params_dict = {}

        elif any(p in q for p in ['snowfall', 'will it snow', 'snow in', 'blizzard']):
            condition_type = "snow"
            params_dict = {}

        elif 'hurricane' in q or 'tornado' in q:
            condition_type = "precipitation"
            params_dict = {}

        else:
            return None

        location_name, lat, lon = self._extract_location(question)
        if lat is None:
            return None

        target_date = self._extract_date(question, end_date_str)
        if target_date is None:
            return None

        return (location_name, lat, lon, target_date, condition_type, params_dict)

    def _extract_location(self, question: str) -> tuple[Optional[str], Optional[float], Optional[float]]:
        q = question.lower()
        for city in sorted(KNOWN_CITIES.keys(), key=len, reverse=True):
            if city in q:
                lat, lon = KNOWN_CITIES[city]
                return (city.title(), lat, lon)
        patterns = [
            r'temperature in ([A-Z][a-zA-Z\s\-]{2,25}?)(?:\s+be\b|\s+on\b|\s+exceed|\?)',
            r'\bin\s+([A-Z][a-zA-Z\s\-]{2,25}?)(?:\s+on\b|\s+this\b|\s+exceed|\s+be\b|\?)',
        ]
        for pat in patterns:
            m = re.search(pat, question)
            if m:
                candidate = m.group(1).strip()
                if len(candidate) >= 3:
                    result = self._geocode(candidate)
                    if result:
                        return result
        return (None, None, None)

    def _geocode(self, city_name: str) -> Optional[tuple[str, float, float]]:
        key = city_name.strip().lower()
        if key in _geocoding_cache:
            return _geocoding_cache[key]
        try:
            resp = self.session.get(
                OPEN_METEO_GEOCODING,
                params={"name": city_name, "count": 1, "language": "en", "format": "json"},
                timeout=8,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if results:
                r = results[0]
                result: tuple[str, float, float] = (r["name"], float(r["latitude"]), float(r["longitude"]))
                _geocoding_cache[key] = result
                return result
        except Exception as exc:
            logger.debug(f"Geocoding failed for '{city_name}': {exc}")
        _geocoding_cache[key] = None
        return None

    def _extract_date(self, question: str, end_date_str: str) -> Optional[date]:
        today = datetime.now(timezone.utc).date()
        MONTHS = {
            "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
            "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
            "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        }
        for month_name, month_num in MONTHS.items():
            m = re.search(
                rf'\b{month_name}\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:[,\s]+(\d{{4}}))?',
                question, re.IGNORECASE
            )
            if m:
                day = int(m.group(1))
                year = int(m.group(2)) if m.group(2) else today.year
                try:
                    d = date(year, month_num, day)
                    if d < today and (today - d).days > 30:
                        d = date(year + 1, month_num, day)
                    return d
                except ValueError:
                    continue

        m = re.search(r'(\d{4})-(\d{2})-(\d{2})', question)
        if m:
            try:
                return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                pass

        if end_date_str:
            try:
                end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                return end_dt.date()
            except (ValueError, TypeError):
                pass
        return None
