"""
Risk Manager - Position sizing (Kelly), exposure limits, averaging-up gate.

Política: HOLD-TO-RESOLUTION. No hay stop-loss ni take-profit: cada apuesta
se mantiene hasta que Polymarket resuelva. Este módulo sólo sizea entradas,
aplica límites de exposición y gobierna el averaging-up.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("polymarket_bot.risk")


class RiskManager:
    def __init__(self, config: dict):
        risk = config["risk"]
        self.max_position_pct = risk["max_position_pct"]
        self.max_total_exposure_pct = risk["max_total_exposure_pct"]
        self.max_exposure_per_market_pct = risk.get("max_exposure_per_market_pct", 0.15)
        self.min_ev_threshold = risk["min_ev_threshold"]
        self.kelly_fraction = risk.get("kelly_fraction", 0.25)
        self.min_bet = 2.0

        pyr = config.get("pyramiding", {})
        self.pyramid_enabled = pyr.get("enabled", True)
        self.pyramid_min_confidence = pyr.get("min_confidence", 80)
        self.pyramid_min_hours = pyr.get("min_hours_to_resolution", 2)
        self.pyramid_reject_loss_pct = pyr.get("reject_if_existing_loss_pct", 0.15)
        self.pyramid_min_cooldown_min = pyr.get("min_cooldown_minutes", 30)
        logger.info(
            f"RiskManager initialized: kelly={self.kelly_fraction}, "
            f"policy=HOLD-TO-RESOLUTION (stop/tp disabled)"
        )

    def calculate_ev(self, prob_real: float, price_market: float) -> float:
        ev = prob_real * (1 - price_market) - (1 - prob_real) * price_market
        return round(ev, 6)

    def calculate_position_size(
        self,
        portfolio_value: float,
        prob_real: float,
        price_market: float,
        kelly_fraction: float = None,
        min_prob_entry: float = 0.70,
        entry_fraction: float = 1.0,
        max_position_pct: Optional[float] = None,
    ) -> float:
        """
        True Kelly sizing for a binary prediction-market bet.

        For buying a share at `price_market` with true probability `prob_real`,
        the optimal fraction of bankroll is:

            f* = (prob_real - price_market) / (1 - price_market)

        (standard Kelly on a binary bet where payoff = 1.0 if correct, 0 otherwise.)
        This grows with the *edge / remaining upside* ratio — so when we're 97% sure
        at a market price of 0.90 the stake is larger in proportion to its edge,
        even though the absolute profit per share is small. That's the "safe bet"
        math the user asked for.

        We apply `kelly_fraction` (fractional Kelly) to tame variance, then
        `entry_fraction` lets callers reserve budget for averaging-up, and finally
        cap by `max_position_pct` (hard ceiling per trade).
        """
        kf = kelly_fraction if kelly_fraction is not None else self.kelly_fraction

        if price_market <= 0 or price_market >= 1:
            return 0.0
        if prob_real <= price_market:
            return 0.0  # no edge → no bet

        kelly_f = (prob_real - price_market) / (1.0 - price_market)
        kelly_f = max(0.0, min(1.0, kelly_f))

        sized = kelly_f * kf
        sized *= max(0.0, min(1.0, entry_fraction))
        if sized <= 0:
            return 0.0

        cap_pct = max_position_pct if max_position_pct is not None else self.max_position_pct
        max_size = portfolio_value * cap_pct
        size = min(sized * portfolio_value, max_size)
        if size < self.min_bet:
            return 0.0
        return round(size, 2)

    def can_open_new_position(self, portfolio_value: float, current_exposure: float) -> bool:
        if portfolio_value <= 0:
            return False
        exposure_pct = current_exposure / portfolio_value
        can_open = exposure_pct < self.max_total_exposure_pct
        if not can_open:
            logger.warning(
                f"Cannot open new position: exposure {exposure_pct:.2%} "
                f">= limit {self.max_total_exposure_pct:.2%}"
            )
        return can_open

    def can_add_to_market(
        self,
        market_id: str,
        new_side: str,
        new_confidence: float,
        new_size: float,
        portfolio_value: float,
        existing_positions: list[dict],
        hours_to_resolution: float,
        new_edge: float = 0.0,
        new_price: float = 0.0,
    ) -> tuple[bool, str]:
        """
        Decide whether to allow a (pyramid) entry on a market where we may already have
        exposure. Returns (allowed, reason_code).
        """
        same_market = [p for p in existing_positions if p.get("market_id") == market_id]

        # First entry on this market - normal checks only
        if not same_market:
            cap = portfolio_value * self.max_exposure_per_market_pct
            if new_size > cap:
                return False, "MARKET_EXPOSURE_LIMIT"
            return True, "FIRST_ENTRY"

        if not self.pyramid_enabled:
            return False, "PYRAMIDING_DISABLED"

        if hours_to_resolution < self.pyramid_min_hours:
            return False, "PYRAMID_TOO_CLOSE_TO_RESOLUTION"

        # Never average down on a losing position (use current market price if provided)
        for p in same_market:
            entry = p.get("price_entry") or 0.0
            current = new_price if new_price > 0 else (p.get("last_known_price") or entry)
            if entry > 0:
                loss_pct = (entry - current) / entry if p.get("side") == "YES" else (current - entry) / entry
                if loss_pct > self.pyramid_reject_loss_pct:
                    return False, "PYRAMID_EXISTING_LOSS"

        # Cap total exposure on this single market
        existing_exposure = sum((p.get("size_usdc") or 0.0) for p in same_market)
        cap = portfolio_value * self.max_exposure_per_market_pct
        if existing_exposure + new_size > cap:
            return False, "MARKET_EXPOSURE_LIMIT"

        # Cooldown: don't re-enter the exact same market within N minutes
        try:
            last_ts = max(
                datetime.fromisoformat(str(p.get("timestamp")).replace("Z", "+00:00"))
                for p in same_market if p.get("timestamp")
            )
            age_min = (datetime.now(timezone.utc) - last_ts).total_seconds() / 60.0
            if age_min < self.pyramid_min_cooldown_min:
                return False, "PYRAMID_COOLDOWN"
        except (ValueError, TypeError):
            pass

        # Thesis check: allow pyramid if ANY of these hold (vs. last entry on same side):
        #   (a) different side entirely (hedge)
        #   (b) confidence up >=3 pts
        #   (c) edge up >=2 pts AND confidence >= pyramid_min_confidence
        #   (d) exposure on this market still <40% of per-market cap (fresh entry space)
        last = same_market[0]
        last_side = last.get("side")
        last_conf = last.get("confidence_score") or 0.0
        last_edge = last.get("ev_calculated") or 0.0

        if last_side != new_side:
            return True, "PYRAMID_APPROVED"  # different side = hedge, always allow

        conf_improved = new_confidence >= last_conf + 3.0
        edge_improved = (new_edge - last_edge) >= 0.02 and new_confidence >= self.pyramid_min_confidence
        room_to_scale = existing_exposure < (cap * 0.40)

        if not (conf_improved or edge_improved or room_to_scale):
            return False, "PYRAMID_THESIS_UNCHANGED"

        # Still require the floor confidence to avoid adding on low-quality entries
        if new_confidence < self.pyramid_min_confidence - 5:
            return False, "PYRAMID_CONFIDENCE_TOO_LOW"

        return True, "PYRAMID_APPROVED"

    def validate_trade(self, ev: float, price: float, liquidity: float,
                       min_liquidity: float, volume_24h: float,
                       min_volume: float) -> tuple[bool, str]:
        if ev < self.min_ev_threshold:
            return False, f"EV {ev:.4f} below threshold {self.min_ev_threshold}"

        if price > 0.97:
            return False, f"Price {price:.4f} too high (>0.97)"

        if price < 0.03:
            return False, f"Price {price:.4f} too low (<0.03)"

        if liquidity < min_liquidity:
            return False, f"Liquidity ${liquidity:.0f} below minimum ${min_liquidity:.0f}"

        if volume_24h < min_volume:
            return False, f"Volume ${volume_24h:.0f} below minimum ${min_volume:.0f}"

        return True, "OK"
