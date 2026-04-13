"""
Risk Manager - Position sizing (Kelly), stop-losses, take-profits, exposure limits.
"""

import logging
from typing import Optional

logger = logging.getLogger("polymarket_bot.risk")


class RiskManager:
    def __init__(self, config: dict):
        risk = config["risk"]
        self.max_position_pct = risk["max_position_pct"]
        self.max_total_exposure_pct = risk["max_total_exposure_pct"]
        self.stop_loss_pct = risk["stop_loss_pct"]
        self.take_profit_partial_pct = risk["take_profit_partial_pct"]
        self.take_profit_sell_fraction = risk.get("take_profit_sell_fraction", 0.50)
        self.min_ev_threshold = risk["min_ev_threshold"]
        self.kelly_fraction = risk.get("kelly_fraction", 0.25)
        self.min_bet = 2.0
        logger.info(
            f"RiskManager initialized: kelly={self.kelly_fraction}, "
            f"stop_loss={self.stop_loss_pct}, take_profit={self.take_profit_partial_pct}"
        )

    def calculate_ev(self, prob_real: float, price_market: float) -> float:
        ev = prob_real * (1 - price_market) - (1 - prob_real) * price_market
        return round(ev, 6)

    def calculate_position_size(self, portfolio_value: float, prob_real: float,
                                price_market: float, kelly_fraction: float = None) -> float:
        kf = kelly_fraction if kelly_fraction is not None else self.kelly_fraction

        if price_market <= 0 or price_market >= 1:
            return 0.0

        b = (1 / price_market) - 1  # odds
        q = 1 - prob_real

        if b <= 0:
            return 0.0

        kelly_raw = (prob_real * b - q) / b
        kelly_sized = kelly_raw * kf

        if kelly_sized <= 0:
            return 0.0

        max_size = portfolio_value * self.max_position_pct
        size = min(kelly_sized * portfolio_value, max_size)
        size = max(size, self.min_bet)

        # Don't exceed max position
        size = min(size, max_size)

        # Don't exceed available capital
        size = min(size, portfolio_value * 0.5)

        return round(size, 2)

    def check_stop_losses(self, open_positions: list[dict],
                          current_prices: dict) -> list[dict]:
        to_close = []
        for pos in open_positions:
            market_id = pos.get("market_id")
            entry_price = pos.get("price_entry", 0)
            side = pos.get("side", "YES")

            price_data = current_prices.get(market_id)
            if not price_data:
                continue

            if side == "YES":
                current_price = price_data.get("price_yes", entry_price)
            else:
                current_price = price_data.get("price_no", entry_price)

            if entry_price <= 0:
                continue

            loss_pct = (entry_price - current_price) / entry_price
            if loss_pct >= self.stop_loss_pct:
                logger.warning(
                    f"STOP LOSS triggered: trade {pos['id']} "
                    f"entry={entry_price:.4f} current={current_price:.4f} "
                    f"loss={loss_pct:.2%}"
                )
                to_close.append({
                    "trade_id": pos["id"],
                    "reason": "stop_loss",
                    "current_price": current_price,
                    "loss_pct": loss_pct,
                    "profit_loss": (current_price - entry_price) * pos.get("shares", 0),
                })

        return to_close

    def check_take_profits(self, open_positions: list[dict],
                           current_prices: dict) -> list[dict]:
        to_partial_close = []
        for pos in open_positions:
            market_id = pos.get("market_id")
            entry_price = pos.get("price_entry", 0)
            side = pos.get("side", "YES")

            price_data = current_prices.get(market_id)
            if not price_data:
                continue

            if side == "YES":
                current_price = price_data.get("price_yes", entry_price)
            else:
                current_price = price_data.get("price_no", entry_price)

            if entry_price <= 0:
                continue

            gain_pct = (current_price - entry_price) / entry_price
            if gain_pct >= self.take_profit_partial_pct:
                shares_to_sell = pos.get("shares", 0) * self.take_profit_sell_fraction
                logger.info(
                    f"TAKE PROFIT triggered: trade {pos['id']} "
                    f"entry={entry_price:.4f} current={current_price:.4f} "
                    f"gain={gain_pct:.2%} selling {self.take_profit_sell_fraction:.0%}"
                )
                to_partial_close.append({
                    "trade_id": pos["id"],
                    "reason": "take_profit",
                    "current_price": current_price,
                    "gain_pct": gain_pct,
                    "shares_to_sell": shares_to_sell,
                    "profit_loss": (current_price - entry_price) * shares_to_sell,
                })

        return to_partial_close

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
