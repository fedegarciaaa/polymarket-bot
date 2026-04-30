"""Telegram notifier — rich messages with anti-spam and severity levels.

Design choices:
  - All `notify_*` methods are best-effort: they swallow errors and never
    raise. Notification failure must NEVER break the trading loop.
  - Per-event-type rate limiting via `_RateLimiter` so a flapping condition
    doesn't flood the chat.
  - HTML parse mode with consistent visual structure (severity emoji,
    section dividers, code blocks for numeric tables).
  - `silent` flag (-> Telegram disable_notification) for low-importance
    summaries so the user doesn't get pinged at 4am.
  - Two retries with backoff on network errors.

If TELEGRAM_TOKEN / TELEGRAM_CHAT_ID are not set, the notifier becomes a
no-op (every public method returns immediately).
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

import requests

logger = logging.getLogger("polymarket_bot.notifications")


# ============================================================
# Internals
# ============================================================
class _RateLimiter:
    """Per-key minimum interval limiter.

    Used both for type-level rate limits ("don't send more than 1 source-down
    alert per hour") and for sub-key limits ("don't notify the same trade
    closing twice in 30s").
    """

    def __init__(self) -> None:
        self._last: dict[str, float] = {}

    def allow(self, key: str, min_interval_s: float) -> bool:
        now = time.time()
        last = self._last.get(key, 0.0)
        if (now - last) < min_interval_s:
            return False
        self._last[key] = now
        return True


def _h(s: str | None) -> str:
    """HTML-escape a string for safe inclusion in Telegram HTML messages."""
    if s is None:
        return ""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _fmt_money(v: float | int | None) -> str:
    if v is None:
        return "—"
    sign = "+" if v >= 0 else ""
    return f"{sign}${v:.2f}"


def _fmt_pct(v: float | int | None, digits: int = 1) -> str:
    if v is None:
        return "—"
    return f"{v*100:.{digits}f}%"


def _fmt_short_market(question: str | None, max_len: int = 80) -> str:
    if not question:
        return "—"
    q = str(question).strip()
    return _h(q if len(q) <= max_len else q[: max_len - 1] + "…")


def _duration_human(seconds: float | int | None) -> str:
    if seconds is None or seconds <= 0:
        return "—"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s//60}m{s%60:02d}s"
    h = s // 3600
    m = (s % 3600) // 60
    return f"{h}h{m:02d}m"


# Severity → leading emoji. Keep this set tight — too many flavors and the
# user can't pattern-match at a glance.
_SEVERITY = {
    "info":     "🔵",
    "success":  "✅",
    "warn":     "⚠️",
    "error":    "🚨",
    "critical": "⛔",
    "trade":    "📥",
    "close":    "📤",
    "money":    "💰",
    "stats":    "📊",
}


# ============================================================
# Main class
# ============================================================
class TelegramNotifier:
    """Best-effort Telegram client with rate limiting and rich formatting."""

    SEND_URL_TPL = "https://api.telegram.org/bot{}/sendMessage"

    def __init__(self) -> None:
        self.token = os.getenv("TELEGRAM_TOKEN", "").strip()
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        self.enabled = bool(self.token and self.chat_id)
        self._rl = _RateLimiter()
        # Daily counters reset at UTC midnight; used by daily summary
        self._day_marker = self._utc_day(time.time())
        self._day_stats = self._fresh_day_stats()
        if self.enabled:
            logger.info("Telegram notifications enabled")
        else:
            logger.info("Telegram notifications disabled (no token/chat_id)")

    # ─── primitives ────────────────────────────────────────────
    def _send(
        self,
        text: str,
        *,
        silent: bool = False,
        rate_key: Optional[str] = None,
        rate_interval_s: float = 0.0,
    ) -> bool:
        """Low-level send. Returns True on success, False otherwise.

        `rate_key` + `rate_interval_s` enforces a min-interval between
        messages with that key (skip silently if too soon).
        """
        if not self.enabled:
            return False
        if rate_key and rate_interval_s > 0:
            if not self._rl.allow(rate_key, rate_interval_s):
                logger.debug(f"telegram: rate-limited {rate_key}")
                return False
        url = self.SEND_URL_TPL.format(self.token)
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
            "disable_notification": bool(silent),
        }
        # Two retries with exponential backoff
        for attempt in (0, 1, 2):
            try:
                r = requests.post(url, json=payload, timeout=10)
                if r.ok:
                    return True
                logger.warning(f"telegram non-200: {r.status_code} {r.text[:120]}")
            except requests.RequestException as exc:
                logger.warning(f"telegram send failed (attempt {attempt+1}): {exc}")
            time.sleep(1.5 * (attempt + 1))
        return False

    @staticmethod
    def _utc_day(ts: float) -> int:
        return int(ts // 86400)

    def _fresh_day_stats(self) -> dict:
        return {
            "trades_opened": 0, "trades_closed": 0,
            "wins": 0, "losses": 0,
            "pnl_realized_usdc": 0.0,
            "crypto_lag_fills": 0,
            "crypto_lag_pnl_usdc": 0.0,
            "errors": 0,
        }

    def _roll_day_if_needed(self) -> None:
        d = self._utc_day(time.time())
        if d != self._day_marker:
            self._day_marker = d
            self._day_stats = self._fresh_day_stats()

    # ─── lifecycle ────────────────────────────────────────────
    def notify_bot_startup(self, *, mode: str, capital: float,
                           modules: dict, config_summary: dict | None = None) -> None:
        modules_str = ", ".join(
            f"<b>{_h(k)}</b>" for k, v in (modules or {}).items() if v
        ) or "—"
        cfg = config_summary or {}
        body = (
            f"{_SEVERITY['success']} <b>Bot iniciado</b> [{_h(mode)}]\n"
            f"───────────────────────────\n"
            f"💵 Capital: <b>${capital:,.2f}</b>\n"
            f"🧩 Módulos: {modules_str}\n"
        )
        if cfg:
            rows = "\n".join(f"  • <code>{_h(k)}</code>: {_h(v)}" for k, v in cfg.items())
            body += f"⚙️ Config:\n{rows}\n"
        body += f"🕒 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        self._send(body)

    def notify_bot_shutdown(self, *, reason: str = "manual") -> None:
        self._send(
            f"{_SEVERITY['warn']} <b>Bot detenido</b>\n"
            f"Razón: <code>{_h(reason)}</code>\n"
            f"🕒 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )

    # ─── Weather Bot trades ───────────────────────────────────
    def notify_trade(self, trade_data: dict) -> None:
        """Trade opened (Weather Bot). Rich message with all the context."""
        self._roll_day_if_needed()
        self._day_stats["trades_opened"] += 1

        side = trade_data.get("side", "?")
        price = float(trade_data.get("price_entry") or 0)
        size = float(trade_data.get("size_usdc") or 0)
        ev = trade_data.get("ev_calculated")
        prob_real = trade_data.get("prob_real") or trade_data.get("prob_real_estimated")
        prob_market = trade_data.get("prob_market")
        ensemble_mean = trade_data.get("ensemble_mean")
        ensemble_std = trade_data.get("ensemble_std")
        market = trade_data.get("market_question")
        location = trade_data.get("location") or "—"
        target_date = trade_data.get("target_date") or "—"
        days_ahead = trade_data.get("days_ahead")
        hours_to_resolution = trade_data.get("hours_to_resolution")
        confidence = trade_data.get("confidence_score") or trade_data.get("confidence")
        kelly_f = trade_data.get("kelly_fraction")
        mode = trade_data.get("mode", "DEMO")

        side_emoji = "🟢" if str(side).upper() == "YES" else "🔴"
        edge = (prob_real - prob_market) if (prob_real is not None and prob_market is not None) else None

        body = (
            f"{_SEVERITY['trade']} <b>OPEN [{_h(mode)}]</b> {side_emoji} <code>{_h(side)}</code>\n"
            f"───────────────────────────\n"
            f"📍 <b>{_h(str(location).title())}</b> · {_h(target_date)}"
        )
        if days_ahead is not None:
            body += f" · <code>{int(days_ahead)}d</code>"
        if hours_to_resolution is not None:
            body += f" · <code>{float(hours_to_resolution):.1f}h</code>"
        body += "\n"
        body += f"❓ {_fmt_short_market(market, 90)}\n"
        body += "───────────────────────────\n"
        body += f"💵 Precio: <code>{price:.4f}</code> · Tamaño: <b>${size:.2f}</b>\n"

        if prob_real is not None or prob_market is not None:
            pr_str = f"{prob_real*100:.1f}%" if prob_real is not None else "—"
            pm_str = f"{prob_market*100:.1f}%" if prob_market is not None else "—"
            edge_str = f"{edge*100:+.1f}pp" if edge is not None else "—"
            body += f"📈 P(real)=<b>{pr_str}</b> · P(mkt)={pm_str} · edge=<b>{edge_str}</b>\n"
        if ensemble_mean is not None:
            std_str = f" ± {ensemble_std:.2f}" if ensemble_std is not None else ""
            body += f"🌡 Ensemble: <code>{ensemble_mean:.2f}{std_str}</code>\n"
        if confidence is not None:
            body += f"🎯 Confidence: <code>{float(confidence):.0f}/100</code>"
            if kelly_f is not None:
                body += f" · Kelly: <code>{float(kelly_f):.3f}</code>"
            body += "\n"
        if ev is not None:
            body += f"💎 EV: <code>{float(ev):+.4f}</code>\n"

        self._send(body)

    def notify_trade_closed(self, trade_data: dict, profit_loss: float, reason: str) -> None:
        self._roll_day_if_needed()
        self._day_stats["trades_closed"] += 1
        if profit_loss > 0:
            self._day_stats["wins"] += 1
        else:
            self._day_stats["losses"] += 1
        self._day_stats["pnl_realized_usdc"] += float(profit_loss or 0.0)

        market = trade_data.get("market_question")
        side = trade_data.get("side", "?")
        location = trade_data.get("location") or "—"
        size = float(trade_data.get("size_usdc") or 0)
        price_entry = float(trade_data.get("price_entry") or 0)
        mode = trade_data.get("mode", "DEMO")
        # Duration if we have created_at + closed_at-ish info
        opened_at = trade_data.get("timestamp") or trade_data.get("opened_at")
        closed_at = trade_data.get("closed_at")
        duration_s = None
        if opened_at and closed_at:
            try:
                t0 = datetime.fromisoformat(str(opened_at).replace("Z", "+00:00"))
                t1 = datetime.fromisoformat(str(closed_at).replace("Z", "+00:00"))
                duration_s = (t1 - t0).total_seconds()
            except Exception:
                pass

        roi_pct = (profit_loss / size) if size > 0 else None
        sev = _SEVERITY["success"] if profit_loss > 0 else _SEVERITY["error"]
        title = "WIN" if profit_loss > 0 else "LOSS"
        if "emergency" in reason.lower() or "circuit" in reason.lower():
            title = "EMERGENCY"
            sev = _SEVERITY["warn"]

        body = (
            f"{sev} <b>CLOSE {title} [{_h(mode)}]</b>\n"
            f"───────────────────────────\n"
            f"📍 <b>{_h(str(location).title())}</b> · <code>{_h(side)}</code> @ {price_entry:.4f}\n"
            f"❓ {_fmt_short_market(market, 90)}\n"
            f"───────────────────────────\n"
            f"💰 P&amp;L: <b>{_fmt_money(profit_loss)}</b>"
        )
        if roi_pct is not None:
            body += f"  ({_fmt_pct(roi_pct)} ROI)"
        body += "\n"
        if duration_s is not None:
            body += f"⏱ Duración: <code>{_duration_human(duration_s)}</code>\n"
        body += f"📌 Razón: <code>{_h(reason)}</code>\n"
        # Running day score
        d = self._day_stats
        body += f"📊 Hoy: <b>{d['wins']}W</b> / <b>{d['losses']}L</b> · {_fmt_money(d['pnl_realized_usdc'])}"
        self._send(body)

    # ─── Cycle/portfolio ──────────────────────────────────────
    def notify_cycle_summary(self, cycle_data: dict, *, force: bool = False) -> None:
        """Send a summary only if something interesting happened, OR if forced.

        Skipping no-trade no-change cycles eliminates ~95% of the spam from
        the previous version (one ping every 10 minutes) while still alerting
        on runs with actual activity.
        """
        trades = int(cycle_data.get("trades_executed") or 0)
        opps = int(cycle_data.get("opportunities_found") or 0)
        pnl_cycle = float(cycle_data.get("pnl_cycle") or 0.0)
        # Skip silent cycles unless caller forces
        if not force and trades == 0 and abs(pnl_cycle) < 1e-6 and opps == 0:
            return

        markets = cycle_data.get("markets_scanned", 0)
        portfolio = cycle_data.get("portfolio_value", 0)
        pnl_total = cycle_data.get("pnl_total", 0)
        mode = cycle_data.get("mode", "DEMO")

        emoji = "📈" if pnl_cycle >= 0 else "📉"
        body = (
            f"{_SEVERITY['stats']} <b>Cycle [{_h(mode)}]</b>\n"
            f"───────────────────────────\n"
            f"🔍 Mercados: <b>{markets}</b> · oportunidades: <b>{opps}</b> · trades: <b>{trades}</b>\n"
            f"{emoji} Cycle: <b>{_fmt_money(pnl_cycle)}</b>  ·  Total: {_fmt_money(pnl_total)}\n"
            f"🏦 Portfolio: <b>${float(portfolio):,.2f}</b>"
        )
        # Send silently — these are status, not alerts
        self._send(body, silent=True, rate_key="cycle_summary", rate_interval_s=60)

    def notify_daily_summary(self, *, portfolio_now: float, capital_initial: float,
                             pnl_24h: float, sources_status: list | None = None,
                             extra: dict | None = None) -> None:
        """Send once per day at the configured time."""
        d = self._day_stats
        wr = (d["wins"] / max(1, d["wins"] + d["losses"])) if (d["wins"] + d["losses"]) > 0 else None
        change_pct = ((portfolio_now - capital_initial) / capital_initial) if capital_initial else 0.0

        body = (
            f"{_SEVERITY['stats']} <b>Daily Summary</b> · "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n"
            f"───────────────────────────\n"
            f"🏦 Portfolio: <b>${portfolio_now:,.2f}</b>  "
            f"({_fmt_pct(change_pct, 2)} vs inicial)\n"
            f"📈 P&amp;L 24h: <b>{_fmt_money(pnl_24h)}</b>\n"
            f"───────────────────────────\n"
            f"📥 Trades abiertos: <b>{d['trades_opened']}</b>\n"
            f"📤 Trades cerrados: <b>{d['trades_closed']}</b> "
            f"({d['wins']}W / {d['losses']}L"
        )
        if wr is not None:
            body += f", WR <b>{wr*100:.0f}%</b>"
        body += ")\n"
        if d["crypto_lag_fills"] > 0 or d["crypto_lag_pnl_usdc"] != 0:
            body += (
                f"⚡ Crypto-Lag: <b>{d['crypto_lag_fills']}</b> fills · "
                f"{_fmt_money(d['crypto_lag_pnl_usdc'])}\n"
            )
        if d["errors"] > 0:
            body += f"🚨 Errores: <b>{d['errors']}</b>\n"
        if sources_status:
            down = [s for s in sources_status if s.get("consecutive_failures", 0) >= 3]
            if down:
                body += f"⚠️ Sources DOWN: {', '.join(_h(s['source_name']) for s in down)}\n"
            else:
                body += "✅ Todas las sources OK\n"
        if extra:
            for k, v in extra.items():
                body += f"  • <code>{_h(k)}</code>: {_h(v)}\n"
        self._send(body)
        # Reset day window after daily report
        self._day_stats = self._fresh_day_stats()

    # ─── Crypto-Lag events ────────────────────────────────────
    def notify_crypto_lag_fill(self, *, symbol: str, side: str, outcome: str,
                               fill_price: float, fill_size_usdc: float,
                               is_adverse: bool = False, market_slug: str = "") -> None:
        self._roll_day_if_needed()
        self._day_stats["crypto_lag_fills"] += 1
        emoji = "📥" if side.upper() == "BUY" else "📤"
        adv = "  <i>(adverse)</i>" if is_adverse else ""
        body = (
            f"{emoji} <b>⚡ Crypto-Lag FILL</b>{adv}\n"
            f"───────────────────────────\n"
            f"🪙 <b>{_h(symbol)}</b> · {_h(side)} {_h(outcome)} @ <code>{fill_price:.3f}</code>\n"
            f"💵 Size: <b>${fill_size_usdc:.2f}</b>\n"
            f"📊 Hoy: {self._day_stats['crypto_lag_fills']} fills · "
            f"{_fmt_money(self._day_stats['crypto_lag_pnl_usdc'])}"
        )
        if market_slug:
            body += f"\n🔗 <code>{_h(market_slug)}</code>"
        # Throttle to one per 10s per symbol — burst protection on choppy markets
        self._send(body, silent=True, rate_key=f"clag_fill_{symbol}", rate_interval_s=10)

    def notify_crypto_lag_close(self, *, symbol: str, realized_pnl_usdc: float,
                                final_yes_price: float | None, reason: str = "resolved",
                                market_slug: str = "") -> None:
        self._roll_day_if_needed()
        self._day_stats["crypto_lag_pnl_usdc"] += float(realized_pnl_usdc or 0.0)
        sev = _SEVERITY["success"] if realized_pnl_usdc > 0 else _SEVERITY["error"]
        body = (
            f"{sev} <b>⚡ Crypto-Lag CLOSE</b>\n"
            f"───────────────────────────\n"
            f"🪙 <b>{_h(symbol)}</b> · {_h(reason)}\n"
            f"💰 P&amp;L: <b>{_fmt_money(realized_pnl_usdc)}</b>"
        )
        if final_yes_price is not None:
            body += f" · resolved YES={float(final_yes_price):.2f}"
        body += f"\n📊 Total módulo hoy: <b>{_fmt_money(self._day_stats['crypto_lag_pnl_usdc'])}</b>"
        if market_slug:
            body += f"\n🔗 <code>{_h(market_slug)}</code>"
        self._send(body)

    def notify_crypto_lag_status(self, *, status: str, detail: str = "") -> None:
        """E.g. status='HALTED', 'WS_RECONNECT', 'CIRCUIT_BREAKER'."""
        body = (
            f"{_SEVERITY['warn']} <b>⚡ Crypto-Lag · {_h(status)}</b>\n"
            f"{_h(detail)}"
        )
        self._send(body, rate_key=f"clag_status_{status}", rate_interval_s=300)

    # ─── Risk events ──────────────────────────────────────────
    def notify_circuit_breaker(self, trade_data: dict, *, recovered_usdc: float,
                               loss_avoided_usdc: float = 0.0) -> None:
        market = trade_data.get("market_question")
        location = trade_data.get("location") or "—"
        body = (
            f"{_SEVERITY['warn']} <b>Circuit Breaker</b>\n"
            f"───────────────────────────\n"
            f"📍 <b>{_h(str(location).title())}</b>\n"
            f"❓ {_fmt_short_market(market, 90)}\n"
            f"💰 Recuperado: <b>{_fmt_money(recovered_usdc)}</b>\n"
        )
        if loss_avoided_usdc > 0:
            body += f"🛡 Pérdida evitada: <b>{_fmt_money(loss_avoided_usdc)}</b>"
        self._send(body)

    def notify_position_degraded(self, trade_data: dict, *, current_safety: float) -> None:
        market = trade_data.get("market_question")
        body = (
            f"{_SEVERITY['warn']} <b>Posición DEGRADED</b>\n"
            f"❓ {_fmt_short_market(market, 90)}\n"
            f"📉 Safety: <b>{current_safety*100:.0f}%</b> · re-apuestas bloqueadas"
        )
        self._send(body, silent=True, rate_key=f"degraded_{trade_data.get('id','?')}",
                   rate_interval_s=3600)

    # ─── Errors / warnings ────────────────────────────────────
    def notify_error(self, error_msg: str, context: str = "") -> None:
        self._roll_day_if_needed()
        self._day_stats["errors"] += 1
        body = (
            f"{_SEVERITY['error']} <b>ERROR</b>"
            + (f" · {_h(context)}" if context else "")
            + "\n<pre>" + _h(str(error_msg)[:600]) + "</pre>"
        )
        # Throttle similar errors so a recurring exception doesn't ping forever
        key = f"err_{(context or 'general')[:32]}"
        self._send(body, rate_key=key, rate_interval_s=120)

    def notify_source_down(self, source_name: str, consecutive_failures: int) -> None:
        body = (
            f"{_SEVERITY['warn']} <b>Source DOWN</b>\n"
            f"<code>{_h(source_name)}</code> · {consecutive_failures} fallos consecutivos\n"
            f"<i>Ensemble degradado.</i>"
        )
        # Once per source per hour
        self._send(body, rate_key=f"src_down_{source_name}", rate_interval_s=3600)

    def notify_confidence_anomaly(self, streak: int, markets_sample: list) -> None:
        sample = "\n".join(f" • {_fmt_short_market(q, 70)}" for q in (markets_sample or [])[:3])
        body = (
            f"{_SEVERITY['critical']} <b>Confidence anomaly</b>\n"
            f"{streak} pérdidas consecutivas con confidence ≥ 85:\n{sample}\n"
            f"<i>Considera subir min_confidence o revisar el ensemble.</i>"
        )
        self._send(body, rate_key="conf_anomaly", rate_interval_s=1800)

    def notify_single_process_violation(self, pid: int) -> None:
        body = (
            f"{_SEVERITY['critical']} <b>Single-process violation</b>\n"
            f"Otra instancia ya está corriendo (PID=<code>{pid}</code>).\n"
            f"Arranque abortado."
        )
        self._send(body)

    # ─── Memory / learning ────────────────────────────────────
    def notify_memory_update(self, update_type: str, details: str) -> None:
        body = (
            f"🧠 <b>Memory · {_h(update_type)}</b>\n"
            f"<i>{_h(str(details)[:400])}</i>"
        )
        # Group memory updates so we don't get a flood after a parameter sweep
        self._send(body, silent=True, rate_key=f"memory_{update_type}", rate_interval_s=300)
