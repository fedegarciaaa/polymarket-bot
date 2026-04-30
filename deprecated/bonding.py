"""
Bonding Strategy - Buy high-probability outcomes for near-guaranteed returns.
Fast-trading mode: focuses on markets resolving within 5 days.
For ultra-short markets (<24h), uses raw return instead of annualized return.
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
        self.max_days_to_resolution = bonding_cfg.get("max_days_to_resolution", 5)
        self.min_raw_return_ultra_short = bonding_cfg.get("min_raw_return_ultra_short", 0.03)
        logger.info(
            f"BondingStrategy initialized: enabled={self.enabled} "
            f"range=[{self.min_price}, {self.max_price}] "
            f"min_ann_ret={self.min_annualized_return:.0%} "
            f"max_days={self.max_days_to_resolution}"
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

                    # Calculate time to resolution with sub-day precision
                    end_date_str = market.get("end_date", "")
                    if not end_date_str:
                        continue

                    try:
                        end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                        now = datetime.now(timezone.utc)
                        delta = end_date - now
                        total_seconds = delta.total_seconds()
                        if total_seconds < 0:
                            continue
                        hours_to_resolution = total_seconds / 3600
                        days_to_resolution = max(total_seconds / 86400, 1/24)  # min 1 hour
                    except (ValueError, TypeError):
                        continue

                    if days_to_resolution > self.max_days_to_resolution:
                        continue

                    # Return per dollar: buy at price, get 1.0 if correct
                    return_per_dollar = (1.0 / price) - 1.0

                    # Ultra-short markets (<24h): use raw return instead of annualized
                    is_ultra_short = hours_to_resolution < 24
                    if is_ultra_short:
                        if return_per_dollar < self.min_raw_return_ultra_short:
                            continue
                        # Compute annualized only for display
                        daily_return = return_per_dollar / max(days_to_resolution, 1/24)
                        annualized_return = daily_return * 365
                    else:
                        daily_return = return_per_dollar / days_to_resolution
                        annualized_return = daily_return * 365
                        if annualized_return < self.min_annualized_return:
                            continue

                    # For bonding, real prob is estimated slightly above market
                    prob_real_estimated = min(price + 0.02, 0.995)

                    # Raw EV
                    ev = prob_real_estimated * (1 - price) - (1 - prob_real_estimated) * price

                    # Priority score: shorter duration = higher priority
                    # Ultra-short gets a big bonus to float above multi-day markets
                    if is_ultra_short:
                        priority_score = annualized_return * 2.0
                    else:
                        priority_score = annualized_return

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
                        "days_to_resolution": round(days_to_resolution, 2),
                        "hours_to_resolution": round(hours_to_resolution, 1),
                        "is_ultra_short": is_ultra_short,
                        "daily_return": round(daily_return, 6),
                        "annualized_return": round(annualized_return, 4),
                        "return_per_dollar": round(return_per_dollar, 6),
                        "priority_score": round(priority_score, 4),
                        "category": market.get("category", ""),
                    })

            except (KeyError, TypeError, ValueError) as e:
                logger.warning(f"Error analyzing bonding opportunity for {market.get('id', 'unknown')}: {e}")
                continue

        # Sort by priority score (ultra-short gets boosted score)
        opportunities.sort(key=lambda x: x["priority_score"], reverse=True)
        top = opportunities[:10]

        if top:
            logger.info(f"Bonding: found {len(opportunities)} opportunities, returning top {len(top)}")
            for opp in top:
                time_str = f"{opp['hours_to_resolution']:.0f}h" if opp['is_ultra_short'] else f"{opp['days_to_resolution']:.1f}d"
                logger.info(
                    f"  {opp['side']} @ {opp['price']:.4f} "
                    f"ann_ret={opp['annualized_return']:.1%} "
                    f"exp={time_str} "
                    f"ret/$ ={opp['return_per_dollar']:.4f} "
                    f"- {opp['market_question'][:55]}"
                )
        else:
            logger.info("Bonding: no opportunities found")

        return top
