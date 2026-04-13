"""
Bonding Strategy - Buy high-probability outcomes for near-guaranteed returns.
Uses annualized return as primary metric (not raw EV, which is always tiny for high-prob markets).
"""

import logging
from datetime import datetime, timezone

logger = logging.getLogger("polymarket_bot.strategies.bonding")


class BondingStrategy:
    def __init__(self, config: dict):
        bonding_cfg = config["strategies"]["bonding"]
        self.enabled = bonding_cfg.get("enabled", True)
        self.min_price = bonding_cfg.get("min_price", 0.80)
        self.max_price = bonding_cfg.get("max_price", 0.97)
        self.min_annualized_return = bonding_cfg.get("min_annualized_return", 0.15)
        self.max_days_to_resolution = bonding_cfg.get("max_days_to_resolution", 90)
        logger.info(
            f"BondingStrategy initialized: enabled={self.enabled} "
            f"range=[{self.min_price}, {self.max_price}] "
            f"min_ann_ret={self.min_annualized_return:.0%}"
        )

    def find_opportunities(self, markets: list[dict], config: dict) -> list[dict]:
        if not self.enabled:
            return []

        opportunities = []

        for market in markets:
            try:
                for side in ["YES", "NO"]:
                    price = market["price_yes"] if side == "YES" else market["price_no"]

                    if not (self.min_price <= price <= self.max_price):
                        continue

                    implied_prob = price

                    # Calculate days to resolution
                    end_date_str = market.get("end_date", "")
                    if not end_date_str:
                        continue

                    try:
                        end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                        now = datetime.now(timezone.utc)
                        delta = end_date - now
                        days_to_resolution = max(delta.days, 1)
                    except (ValueError, TypeError):
                        continue

                    if days_to_resolution > self.max_days_to_resolution:
                        continue

                    # Return per dollar: buy at price, get 1.0 if correct
                    return_per_dollar = (1.0 / price) - 1.0
                    daily_return = return_per_dollar / days_to_resolution
                    annualized_return = daily_return * 365

                    # Skip if annualized return is too low
                    if annualized_return < self.min_annualized_return:
                        continue

                    # For bonding, real prob is estimated slightly above market
                    # (these are near-certain outcomes)
                    prob_real_estimated = min(price + 0.02, 0.995)

                    # Raw EV (will be small, that's fine for bonding)
                    ev = prob_real_estimated * (1 - price) - (1 - prob_real_estimated) * price

                    opportunities.append({
                        "market_id": market["id"],
                        "market_question": market["question"],
                        "strategy": "bonding",
                        "side": side,
                        "price": price,
                        "implied_prob": implied_prob,
                        "prob_real_estimated": prob_real_estimated,
                        "ev": round(ev, 6),
                        "volume_24h": market["volume_24h"],
                        "liquidity": market["liquidity"],
                        "end_date": end_date_str,
                        "days_to_resolution": days_to_resolution,
                        "daily_return": round(daily_return, 6),
                        "annualized_return": round(annualized_return, 4),
                        "return_per_dollar": round(return_per_dollar, 6),
                        "category": market.get("category", ""),
                    })

            except (KeyError, TypeError, ValueError) as e:
                logger.warning(f"Error analyzing bonding opportunity for {market.get('id', 'unknown')}: {e}")
                continue

        # Sort by annualized return descending
        opportunities.sort(key=lambda x: x["annualized_return"], reverse=True)
        top = opportunities[:10]

        if top:
            logger.info(f"Bonding: found {len(opportunities)} opportunities, returning top {len(top)}")
            for opp in top:
                logger.info(
                    f"  {opp['side']} @ {opp['price']:.4f} "
                    f"ann_ret={opp['annualized_return']:.1%} "
                    f"days={opp['days_to_resolution']} "
                    f"ret/$ ={opp['return_per_dollar']:.4f} "
                    f"- {opp['market_question'][:55]}"
                )
        else:
            logger.info("Bonding: no opportunities found")

        return top
