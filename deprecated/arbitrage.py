"""
Value Betting Strategy - Find mispriced markets where Claude can estimate better than the market.
Targets uncertain markets (0.10-0.90) where information edge is most likely.

Replaces the original arbitrage strategy since Gamma API returns normalized prices
(YES + NO always = 1.0), making sum-to-one arbitrage impossible at the API level.
"""

import logging
from datetime import datetime, timezone

logger = logging.getLogger("polymarket_bot.strategies.value")


def _compute_days_to_expiry(end_date_str: str):
    """Returns (days_float, hours_float) or (None, None) if unparseable."""
    if not end_date_str:
        return None, None
    try:
        end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        delta = end_date - now
        total_seconds = delta.total_seconds()
        if total_seconds < 0:
            return None, None   # expired market — callers skip on None
        hours = total_seconds / 3600
        days = total_seconds / 86400
        return round(days, 2), round(hours, 1)
    except (ValueError, TypeError):
        return None, None


class ValueBettingStrategy:
    def __init__(self, config: dict):
        vb_cfg = config["strategies"].get("value_betting", {})
        self.enabled = vb_cfg.get("enabled", True)
        self.min_price = vb_cfg.get("min_price", 0.10)
        self.max_price = vb_cfg.get("max_price", 0.90)
        self.min_volume = vb_cfg.get("min_volume_24h", 5000)
        self.min_liquidity = vb_cfg.get("min_liquidity", 1000)
        self.max_days_to_expiry = vb_cfg.get("max_days_to_expiry", 5)
        self.prefer_categories = vb_cfg.get("prefer_categories", [])
        logger.info(
            f"ValueBettingStrategy initialized: enabled={self.enabled} "
            f"range=[{self.min_price}, {self.max_price}] "
            f"min_vol=${self.min_volume:,.0f} "
            f"max_days={self.max_days_to_expiry}"
        )

    def find_opportunities(self, markets: list[dict], config: dict) -> list[dict]:
        if not self.enabled:
            return []

        opportunities = []

        for market in markets:
            try:
                price_yes = market["price_yes"]
                price_no = market["price_no"]

                # We want uncertain markets - not near 0 or 1
                yes_in_range = self.min_price <= price_yes <= self.max_price
                no_in_range = self.min_price <= price_no <= self.max_price

                if not (yes_in_range or no_in_range):
                    continue

                volume = market.get("volume_24h", 0)
                liquidity = market.get("liquidity", 0)

                if volume < self.min_volume:
                    continue
                if liquidity < self.min_liquidity:
                    continue

                # Duration filter: only fast-resolving markets
                end_date_str = market.get("end_date", "")
                days_to_expiry, hours_to_expiry = _compute_days_to_expiry(end_date_str)
                if days_to_expiry is None:
                    continue
                if days_to_expiry > self.max_days_to_expiry:
                    continue

                # Uncertainty score: closer to 0.50 = more uncertain = more edge potential
                uncertainty = 1.0 - abs(price_yes - 0.5) * 2  # 1.0 at 0.50, 0.0 at 0/1

                # Time bonus: shorter duration = higher priority
                time_bonus = 1.0 + max(0, (self.max_days_to_expiry - days_to_expiry)) * 0.3
                opportunity_score = uncertainty * (volume / 100000) * time_bonus

                # Determine which side is more interesting to bet
                if yes_in_range:
                    primary_side = "YES"
                    primary_price = price_yes
                else:
                    primary_side = "NO"
                    primary_price = price_no

                opportunities.append({
                    "market_id": market["id"],
                    "market_question": market["question"],
                    "strategy": "value_betting",
                    "side": primary_side,
                    "price": primary_price,
                    "price_yes": price_yes,
                    "price_no": price_no,
                    "implied_prob_yes": price_yes,
                    "implied_prob_no": price_no,
                    "uncertainty_score": round(uncertainty, 4),
                    "opportunity_score": round(opportunity_score, 4),
                    "volume_24h": volume,
                    "liquidity": liquidity,
                    "end_date": end_date_str,
                    "days_to_expiry": days_to_expiry,
                    "hours_to_expiry": hours_to_expiry,
                    "category": market.get("category", ""),
                })

            except (KeyError, TypeError, ValueError) as e:
                logger.warning(f"Error analyzing value opportunity for {market.get('id', 'unknown')}: {e}")
                continue

        # Sort by opportunity score descending
        opportunities.sort(key=lambda x: x["opportunity_score"], reverse=True)
        top = opportunities[:10]

        if top:
            logger.info(f"Value Betting: found {len(opportunities)} opportunities, returning top {len(top)}")
            for opp in top:
                time_str = f"{opp['hours_to_expiry']:.0f}h" if opp['hours_to_expiry'] < 24 else f"{opp['days_to_expiry']:.1f}d"
                logger.info(
                    f"  YES={opp['price_yes']:.4f} NO={opp['price_no']:.4f} "
                    f"uncert={opp['uncertainty_score']:.2f} "
                    f"score={opp['opportunity_score']:.2f} "
                    f"vol=${opp['volume_24h']:,.0f} "
                    f"exp={time_str} "
                    f"- {opp['market_question'][:45]}"
                )
        else:
            logger.info("Value Betting: no opportunities found")

        return top


class MomentumStrategy:
    """
    Momentum Strategy - Identify markets with strong recent volume suggesting incoming resolution.
    Targets markets where volume spike indicates new information the price hasn't fully absorbed.
    """

    def __init__(self, config: dict):
        mom_cfg = config["strategies"].get("momentum", {})
        self.enabled = mom_cfg.get("enabled", True)
        self.min_volume = mom_cfg.get("min_volume_24h", 10000)
        self.price_range = (
            mom_cfg.get("min_price", 0.15),
            mom_cfg.get("max_price", 0.85),
        )
        self.max_days_to_expiry = mom_cfg.get("max_days_to_expiry", 5)
        logger.info(
            f"MomentumStrategy initialized: enabled={self.enabled} "
            f"min_vol=${self.min_volume:,.0f} "
            f"max_days={self.max_days_to_expiry}"
        )

    def find_opportunities(self, markets: list[dict], config: dict) -> list[dict]:
        if not self.enabled:
            return []

        opportunities = []

        for market in markets:
            try:
                price_yes = market["price_yes"]
                price_no = market["price_no"]
                volume = market.get("volume_24h", 0)
                liquidity = market.get("liquidity", 0)

                if volume < self.min_volume:
                    continue

                # Skip extreme prices
                if price_yes < self.price_range[0] or price_yes > self.price_range[1]:
                    continue

                # Duration filter: only fast-resolving markets
                end_date_str = market.get("end_date", "")
                days_to_expiry, hours_to_expiry = _compute_days_to_expiry(end_date_str)
                if days_to_expiry is None:
                    continue
                if days_to_expiry > self.max_days_to_expiry:
                    continue

                # Volume-to-liquidity ratio: high ratio = lots of action relative to depth
                vol_liq_ratio = volume / liquidity if liquidity > 0 else 0

                if vol_liq_ratio < 1.0:
                    continue

                # Directional signal
                if price_yes > 0.5:
                    momentum_side = "YES"
                    momentum_price = price_yes
                else:
                    momentum_side = "NO"
                    momentum_price = price_no

                # Time bonus: shorter duration = higher score
                time_bonus = 1.0 + max(0, (self.max_days_to_expiry - days_to_expiry)) * 0.4
                momentum_score = vol_liq_ratio * (1 - abs(price_yes - 0.5)) * time_bonus

                opportunities.append({
                    "market_id": market["id"],
                    "market_question": market["question"],
                    "strategy": "momentum",
                    "side": momentum_side,
                    "price": momentum_price,
                    "price_yes": price_yes,
                    "price_no": price_no,
                    "volume_24h": volume,
                    "liquidity": liquidity,
                    "vol_liq_ratio": round(vol_liq_ratio, 2),
                    "momentum_score": round(momentum_score, 4),
                    "end_date": end_date_str,
                    "days_to_expiry": days_to_expiry,
                    "hours_to_expiry": hours_to_expiry,
                    "category": market.get("category", ""),
                })

            except (KeyError, TypeError, ValueError) as e:
                logger.warning(f"Error analyzing momentum for {market.get('id', 'unknown')}: {e}")
                continue

        opportunities.sort(key=lambda x: x["momentum_score"], reverse=True)
        top = opportunities[:5]

        if top:
            logger.info(f"Momentum: found {len(opportunities)} opportunities, returning top {len(top)}")
            for opp in top:
                time_str = f"{opp['hours_to_expiry']:.0f}h" if opp['hours_to_expiry'] < 24 else f"{opp['days_to_expiry']:.1f}d"
                logger.info(
                    f"  {opp['side']} @ {opp['price']:.4f} "
                    f"vol/liq={opp['vol_liq_ratio']:.1f}x "
                    f"score={opp['momentum_score']:.2f} "
                    f"vol=${opp['volume_24h']:,.0f} "
                    f"exp={time_str} "
                    f"- {opp['market_question'][:45]}"
                )
        else:
            logger.info("Momentum: no opportunities found")

        return top
