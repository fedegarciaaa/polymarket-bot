"""Risk gates and circuit breakers for the crypto-lag module.

Single source of truth for "can we quote right now?" decisions. Every gate
exposes a method that returns (allow: bool, reason: str). The cycle calls
these in order and stops at the first deny.

Two kinds of state:
  - Persistent: bankroll, daily P&L, consecutive losses, halt flag (file).
  - Transient: WS staleness, realized-vol burst, in-flight inventory.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("polymarket_bot.crypto_lag.risk")


@dataclass
class RiskState:
    """In-memory tracking. Persisted to DB by the cycle every minute or so."""
    daily_pnl_usdc: float = 0.0
    consecutive_losses: int = 0
    last_loss_ts: float = 0.0
    inventory_by_market: dict[str, float] = field(default_factory=dict)  # cond_id → net usdc
    open_orders_count: int = 0


class CryptoLagRisk:
    def __init__(
        self,
        config: dict,
        get_bankroll_usdc,    # callable returning current bankroll (callable so we re-read each cycle)
    ):
        cfg = config.get("crypto_lag", {}) or {}
        self.capital_pct = float(cfg.get("capital_pct", 0.30))
        self.max_order_usdc = float(cfg.get("max_order_usdc", 25))
        self.per_market_max_inventory_usdc = float(
            cfg.get("per_market_max_inventory_usdc", 100)
        )
        self.max_concurrent_orders = int(
            cfg.get("max_concurrent_orders", 6)
        )

        cb = cfg.get("circuit_breakers", {}) or {}
        self.daily_max_loss_usdc = float(cb.get("daily_max_loss_usdc", 50.0))
        self.consecutive_loss_threshold = int(cb.get("consecutive_losses", 5))
        self.realized_vol_multiplier = float(cb.get("realized_vol_multiplier", 3.0))
        self.halt_file = Path(cb.get("halt_file", "data/crypto_lag.halt"))

        self.binance_stale_seconds = float(
            (cfg.get("binance") or {}).get("stale_seconds", 5.0)
        )

        self._bankroll_fn = get_bankroll_usdc
        self.state = RiskState()
        # Daily P&L is reset at the wall-clock day boundary (UTC).
        self._pnl_day = self._utc_day(time.time())

    # ─── pure gates (no side effects) ───────────────────────────
    def can_quote_globally(self, now_ts: Optional[float] = None) -> tuple[bool, str]:
        """Top-level: are we allowed to quote anywhere right now?"""
        now = now_ts or time.time()

        # 1. Halt file — operator override, highest priority
        if self.halt_file.exists():
            return False, "halt_file_present"

        # 2. Daily P&L roll
        if self._utc_day(now) != self._pnl_day:
            self.state.daily_pnl_usdc = 0.0
            self.state.consecutive_losses = 0
            self._pnl_day = self._utc_day(now)

        # 3. Daily loss circuit
        if self.state.daily_pnl_usdc <= -abs(self.daily_max_loss_usdc):
            return False, f"daily_loss_breached:{self.state.daily_pnl_usdc:.2f}"

        # 4. Streak of losses
        if self.state.consecutive_losses >= self.consecutive_loss_threshold:
            return False, f"consecutive_losses:{self.state.consecutive_losses}"

        return True, "ok"

    def can_quote_market(
        self,
        condition_id: str,
        feed_state,
        now_ts: Optional[float] = None,
        in_reconnect_freeze: bool = False,
    ) -> tuple[bool, str]:
        """Per-market gate."""
        now = now_ts or time.time()

        ok, reason = self.can_quote_globally(now)
        if not ok:
            return False, reason

        # 5. Binance feed must be fresh
        if feed_state is None or feed_state.last_update_ts == 0:
            return False, "feed_no_data"
        if feed_state.is_stale(now, self.binance_stale_seconds):
            return False, f"feed_stale:{(now - feed_state.last_update_ts):.1f}s"
        if in_reconnect_freeze:
            return False, "reconnect_freeze"

        # 6. Inventory cap per market
        net = abs(self.state.inventory_by_market.get(condition_id, 0.0))
        if net >= self.per_market_max_inventory_usdc:
            return False, f"inventory_capped:{net:.0f}usdc"

        # 7. Concurrent orders cap
        if self.state.open_orders_count >= self.max_concurrent_orders:
            return False, f"max_orders:{self.state.open_orders_count}"

        return True, "ok"

    # ─── sizing ─────────────────────────────────────────────────
    def order_size_usdc(self, kelly_size_usdc: float) -> float:
        """Clip kelly size to the configured per-order cap and to the remaining
        bankroll budget (capital_pct of the total). Always returns ≥ 0."""
        size = max(0.0, float(kelly_size_usdc))
        size = min(size, self.max_order_usdc)
        bankroll = max(0.0, float(self._bankroll_fn() or 0.0))
        budget = bankroll * self.capital_pct
        size = min(size, budget)
        return size

    # ─── state mutators (called by order_engine on fill events) ──
    def on_fill(self, condition_id: str, signed_usdc: float) -> None:
        """`signed_usdc` positive = added long YES, negative = added short."""
        cur = self.state.inventory_by_market.get(condition_id, 0.0)
        self.state.inventory_by_market[condition_id] = cur + float(signed_usdc)

    def on_close(self, pnl_usdc: float) -> None:
        self.state.daily_pnl_usdc += float(pnl_usdc)
        if pnl_usdc < 0:
            self.state.consecutive_losses += 1
            self.state.last_loss_ts = time.time()
        elif pnl_usdc > 0:
            self.state.consecutive_losses = 0

    def on_order_open(self) -> None:
        self.state.open_orders_count += 1

    def on_order_close(self) -> None:
        self.state.open_orders_count = max(0, self.state.open_orders_count - 1)

    @staticmethod
    def _utc_day(ts: float) -> int:
        return int(ts // 86400)
