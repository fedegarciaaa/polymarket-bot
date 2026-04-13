"""
Value Betting Strategy - Find mispriced markets where Claude can estimate better than the market.
Targets uncertain markets (0.10-0.90) where information edge is most likely.

Replaces the original arbitrage strategy since Gamma API returns normalized prices
(YES + NO always = 1.0), making sum-to-one arbitrage impossible at the API level.
"""

import logging

logger = logging.getLogger("polymarket_bot.strategies.value")


class ValueBettingStrategy:
    def __init__(self, config: dict):
        vb_cfg = config["strategies"].get("value_betting", {})
        self.enabled = vb_cfg.get("enabled", True)
        self.min_price = vb_cfg.get("min_price", 0.10)
        self.max_price = vb_cfg.get("max_price", 0.90)
        self.min_volume = vb_cfg.get("min_volume_24h", 20000)
        self.min_liquidity = vb_cfg.get("min_liquidity", 10000)
        self.prefer_categories = vb_cfg.get("prefer_categories", [])
        logger.info(
            f"ValueBettingStrategy initialized: enabled={self.enabled} "
            f"range=[{self.min_price}, {self.max_price}] "
            f"min_vol=${self.min_volume:,.0f}"
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
                # Check if YES side is in value range
                yes_in_range = self.min_price <= price_yes <= self.max_price
                # Check if NO side is in value range
                no_in_range = self.min_price <= price_no <= self.max_price

                if not (yes_in_range or no_in_range):
                    continue

                volume = market.get("volume_24h", 0)
                liquidity = market.get("liquidity", 0)

                if volume < self.min_volume:
                    continue
                if liquidity < self.min_liquidity:
                    continue

                # Calculate uncertainty score (closer to 0.50 = more uncertain = more edge potential)
                uncertainty = 1.0 - abs(price_yes - 0.5) * 2  # 1.0 at 0.50, 0.0 at 0/1

                # Volume-adjusted score (high volume + high uncertainty = best opportunity)
                opportunity_score = uncertainty * (volume / 100000)

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
                    "end_date": market.get("end_date", ""),
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
                logger.info(
                    f"  YES={opp['price_yes']:.4f} NO={opp['price_no']:.4f} "
                    f"uncert={opp['uncertainty_score']:.2f} "
                    f"score={opp['opportunity_score']:.2f} "
                    f"vol=${opp['volume_24h']:,.0f} "
                    f"- {opp['market_question'][:50]}"
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
        self.min_volume = mom_cfg.get("min_volume_24h", 50000)
        self.price_range = (
            mom_cfg.get("min_price", 0.15),
            mom_cfg.get("max_price", 0.85),
        )
        logger.info(
            f"MomentumStrategy initialized: enabled={self.enabled} "
            f"min_vol=${self.min_volume:,.0f}"
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

                # Skip extreme prices (already resolved or near-resolved)
                if price_yes < self.price_range[0] or price_yes > self.price_range[1]:
                    continue

                # Volume-to-liquidity ratio: high ratio = lots of action relative to depth
                # This suggests new information is being traded on
                vol_liq_ratio = volume / liquidity if liquidity > 0 else 0

                if vol_liq_ratio < 1.0:
                    continue

                # Directional signal: if YES > 0.5, momentum is bullish; if < 0.5, bearish
                if price_yes > 0.5:
                    momentum_side = "YES"
                    momentum_price = price_yes
                else:
                    momentum_side = "NO"
                    momentum_price = price_no

                momentum_score = vol_liq_ratio * (1 - abs(price_yes - 0.5))

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
                    "end_date": market.get("end_date", ""),
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
                logger.info(
                    f"  {opp['side']} @ {opp['price']:.4f} "
                    f"vol/liq={opp['vol_liq_ratio']:.1f}x "
                    f"score={opp['momentum_score']:.2f} "
                    f"vol=${opp['volume_24h']:,.0f} "
                    f"- {opp['market_question'][:50]}"
                )
        else:
            logger.info("Momentum: no opportunities found")

        return top
