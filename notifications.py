"""
Telegram notifications for trades, errors and cycle summaries.
No-op if TELEGRAM_TOKEN is not configured.
"""

import os
import logging
import requests

logger = logging.getLogger("polymarket_bot.notifications")


class TelegramNotifier:
    def __init__(self):
        self.token = os.getenv("TELEGRAM_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.enabled = bool(self.token and self.chat_id)
        self.base_url = f"https://api.telegram.org/bot{self.token}" if self.token else ""

        if self.enabled:
            logger.info("Telegram notifications enabled")
        else:
            logger.info("Telegram notifications disabled (no token/chat_id configured)")

    def _send_message(self, text: str, parse_mode: str = "HTML"):
        if not self.enabled:
            return

        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
            }
            resp = requests.post(url, json=payload, timeout=10)
            resp.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")

    def notify_trade(self, trade_data: dict):
        action = trade_data.get("action", "?")
        side = trade_data.get("side", "?")
        strategy = trade_data.get("strategy", "?")
        price = trade_data.get("price_entry", 0)
        size = trade_data.get("size_usdc", 0)
        ev = trade_data.get("ev_calculated", 0)
        market = trade_data.get("market_question", "?")
        mode = trade_data.get("mode", "DEMO")
        confidence = trade_data.get("confidence", "?")

        emoji = "\U0001f7e2" if action == "BUY" else "\U0001f534"
        mode_tag = f"[{mode}]"

        text = (
            f"{emoji} <b>{mode_tag} {action} {side}</b>\n"
            f"\U0001f4ca <b>Market:</b> {market[:80]}\n"
            f"\U0001f3af Strategy: {strategy}\n"
            f"\U0001f4b0 Price: {price:.4f} | Size: ${size:.2f}\n"
            f"\U0001f4c8 EV: {ev:.4f} | Confidence: {confidence}\n"
        )
        self._send_message(text)

    def notify_trade_closed(self, trade_data: dict, profit_loss: float, reason: str):
        market = trade_data.get("market_question", "?")
        mode = trade_data.get("mode", "DEMO")
        emoji = "\u2705" if profit_loss > 0 else "\u274c"

        text = (
            f"{emoji} <b>[{mode}] CLOSED - {reason.upper()}</b>\n"
            f"\U0001f4ca {market[:80]}\n"
            f"\U0001f4b0 P&L: <b>${profit_loss:+.4f}</b>\n"
        )
        self._send_message(text)

    def notify_error(self, error_msg: str, context: str = ""):
        text = (
            f"\U0001f6a8 <b>ERROR</b>\n"
            f"{context}\n"
            f"<code>{error_msg[:500]}</code>"
        )
        self._send_message(text)

    def notify_cycle_summary(self, cycle_data: dict):
        markets = cycle_data.get("markets_scanned", 0)
        opps = cycle_data.get("opportunities_found", 0)
        trades = cycle_data.get("trades_executed", 0)
        portfolio = cycle_data.get("portfolio_value", 0)
        pnl_cycle = cycle_data.get("pnl_cycle", 0)
        pnl_total = cycle_data.get("pnl_total", 0)
        mode = cycle_data.get("mode", "DEMO")

        pnl_emoji = "\U0001f4c8" if pnl_cycle >= 0 else "\U0001f4c9"

        text = (
            f"\U0001f504 <b>[{mode}] Cycle Complete</b>\n"
            f"\U0001f50d Markets: {markets} | Opportunities: {opps}\n"
            f"\U0001f4dd Trades: {trades}\n"
            f"{pnl_emoji} Cycle P&L: ${pnl_cycle:+.4f}\n"
            f"\U0001f4b0 Total P&L: ${pnl_total:+.4f}\n"
            f"\U0001f3e6 Portfolio: ${portfolio:.2f}\n"
        )
        self._send_message(text)

    def notify_memory_update(self, update_type: str, details: str):
        text = (
            f"\U0001f9e0 <b>Memory Update: {update_type}</b>\n"
            f"{details[:300]}"
        )
        self._send_message(text)
