"""Risk gates and circuit breakers for the crypto-lag module.

Single source of truth for "can we quote right now?" decisions. Every gate
exposes a method that returns (allow: bool, reason: str). The cycle calls
these in order and stops at the first deny.

This module owns the per-variant bankroll and capital accounting:
  - bankroll_usdc:                lifetime virtual capital, decrements with
                                  realized losses on `on_close`.
  - current_exposure_by_market:   signed USDC committed in OPEN positions,
                                  per condition_id.
  - gross_filled_by_market:       cumulative |USDC| ever filled per cid
                                  (rotation BUY+SELL counts both).
  - placements_in_flight_usdc:    sum of size of placed-but-unacked orders.

Gates layered on top:
  - daily_max_loss_usdc:          halt the variant if today's PnL ≤ -$X.
  - consecutive_losses:           halt after N losses in a row.
  - bankroll_floor_usdc:          halt if bankroll_usdc drops below floor.
  - per_market_max_gross_usdc:    cap rotation per cid (BUY+SELL).
  - variant_max_committed_usdc:   cap simultaneous open exposure across cids.
  - placements_per_minute_limit:  tripwire to detect runaway churn.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Optional, Tuple

logger = logging.getLogger("polymarket_bot.crypto_lag.risk")


@dataclass
class RiskState:
    """Per-variant in-memory state. Persisted to DB every minute by the cycle.

    Fase 1 refactor: split inventory_by_market (NET, kept for skew) from a new
    pair (gross_filled_by_market, current_exposure_by_market) so caps can be
    applied to gross flow rather than net inventory — the previous cap missed
    BUY+SELL rotation entirely (DOGE cid had $5,093 gross filled with $76 net).
    """
    daily_pnl_usdc: float = 0.0
    consecutive_losses: int = 0
    last_loss_ts: float = 0.0
    open_orders_count: int = 0

    # NET inventory (cond_id → signed usdc). Kept so the order_engine can
    # still apply inventory skew (lean against the pile-up direction).
    inventory_by_market: dict[str, float] = field(default_factory=dict)

    # GROSS filled per cid (cond_id → cumulative |usdc|). Purged on close.
    gross_filled_by_market: dict[str, float] = field(default_factory=dict)
    # Signed exposure per cid (cond_id → signed usdc, same as inventory_by_market
    # but cleared on close). Used for available_committed accounting.
    current_exposure_by_market: dict[str, float] = field(default_factory=dict)

    # Sum of size_usdc of orders placed-but-not-yet-acked or filled. Decremented
    # on fill or cancel. Used so a place flood can't bypass the committed cap
    # while orders are in transit.
    placements_in_flight_usdc: float = 0.0

    # Bankroll tracking (Fase 1).
    bankroll_usdc: float = 1000.0
    realized_pnl_lifetime_usdc: float = 0.0
    gross_filled_lifetime_usdc: float = 0.0

    # Recent placement timestamps for rate limiting (last 60s window).
    placement_timestamps: Deque[float] = field(default_factory=lambda: deque(maxlen=500))


class CryptoLagRisk:
    """Per-variant risk manager. Each variant has its own instance and its own
    isolated bankroll — variants don't share capital.

    Constructor signature changed in Fase 1: bankroll is now passed as a number
    (not a callable) and the new gross-cap parameters are first-class args.
    """

    def __init__(
        self,
        config: dict,
        bankroll_usdc: float = 1000.0,
        variant_name: str = "main",
    ):
        cfg = config.get("crypto_lag", {}) or {}
        # Per-order cap (clipping individual order sizes).
        self.max_order_usdc = float(cfg.get("max_order_usdc", 25))
        # Legacy NET inventory cap — kept as a soft guard, but the real cap
        # is per_market_max_gross_usdc below.
        self.per_market_max_inventory_usdc = float(
            cfg.get("per_market_max_inventory_usdc", 100)
        )
        self.max_concurrent_orders = int(cfg.get("max_concurrent_orders", 6))

        # Fase 1 caps.
        self.per_market_max_gross_usdc = float(
            cfg.get("per_market_max_gross_usdc", 50)
        )
        self.variant_max_committed_usdc = float(
            cfg.get("variant_max_committed_usdc", 200)
        )
        self.placements_per_minute_limit = int(
            cfg.get("placements_per_minute_limit", 100)
        )
        self.bankroll_floor_usdc = float(cfg.get("bankroll_floor_usdc", 100))
        # Soft cap relative to bankroll: order_size ≤ capital_pct × bankroll.
        # Kept for backwards compat — variant_max_committed_usdc is the hard cap.
        self.capital_pct = float(cfg.get("capital_pct", 0.30))

        cb = cfg.get("circuit_breakers", {}) or {}
        self.daily_max_loss_usdc = float(cb.get("daily_max_loss_usdc", 50.0))
        self.consecutive_loss_threshold = int(cb.get("consecutive_losses", 5))
        self.realized_vol_multiplier = float(cb.get("realized_vol_multiplier", 3.0))
        self.halt_file = Path(cb.get("halt_file", "data/crypto_lag.halt"))

        self.binance_stale_seconds = float(
            (cfg.get("binance") or {}).get("stale_seconds", 5.0)
        )

        self.variant_name = str(variant_name)
        self.state = RiskState(bankroll_usdc=float(bankroll_usdc))
        # Daily P&L is reset at the wall-clock day boundary (UTC).
        self._pnl_day = self._utc_day(time.time())

    # ─── pure gates (no side effects) ───────────────────────────
    def can_quote_globally(self, now_ts: Optional[float] = None) -> Tuple[bool, str]:
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

        # 5. Bankroll floor — halts the variant for the rest of the run.
        if self.state.bankroll_usdc <= self.bankroll_floor_usdc:
            return False, f"bankroll_floor:{self.state.bankroll_usdc:.2f}"

        # 6. Placement rate limiter (tripwire — protects against runaway churn).
        recent = self._placements_last_60s(now)
        if recent >= self.placements_per_minute_limit:
            return False, f"placement_rate_limited:{recent}/min"

        return True, "ok"

    def can_quote_market(
        self,
        condition_id: str,
        feed_state,
        now_ts: Optional[float] = None,
        in_reconnect_freeze: bool = False,
    ) -> Tuple[bool, str]:
        """Per-market gate."""
        now = now_ts or time.time()

        ok, reason = self.can_quote_globally(now)
        if not ok:
            return False, reason

        # 7. Binance feed must be fresh
        if feed_state is None or feed_state.last_update_ts == 0:
            return False, "feed_no_data"
        if feed_state.is_stale(now, self.binance_stale_seconds):
            return False, f"feed_stale:{(now - feed_state.last_update_ts):.1f}s"
        if in_reconnect_freeze:
            return False, "reconnect_freeze"

        # 8. Per-market GROSS cap (Fase 1) — prevents unbounded BUY+SELL rotation.
        gross = self.state.gross_filled_by_market.get(condition_id, 0.0)
        if gross >= self.per_market_max_gross_usdc:
            return False, f"cid_gross_capped:{gross:.0f}usdc"

        # 9. Per-market NET inventory cap (legacy soft guard).
        net = abs(self.state.inventory_by_market.get(condition_id, 0.0))
        if net >= self.per_market_max_inventory_usdc:
            return False, f"inventory_capped:{net:.0f}usdc"

        # 10. Concurrent orders cap (count of open resting orders).
        if self.state.open_orders_count >= self.max_concurrent_orders:
            return False, f"max_orders:{self.state.open_orders_count}"

        # 11. Variant-level committed cap (sum of |exposure| across all cids
        #     plus in-flight placements). This is the hard ceiling that turns
        #     "$1000 bankroll" into "I will never have more than $X actually
        #     committed at once".
        if self.available_committed_usdc() <= 0:
            return False, "variant_committed_capped"

        return True, "ok"

    # ─── sizing ─────────────────────────────────────────────────
    def order_size_usdc(self, kelly_size_usdc: float) -> float:
        """Clip the requested Kelly size to all caps. Returns 0 when no room
        remains (caller should treat as "skip placement").
        """
        size = max(0.0, float(kelly_size_usdc))
        size = min(size, self.max_order_usdc)
        # Soft cap relative to bankroll.
        size = min(size, self.state.bankroll_usdc * self.capital_pct)
        # Hard cap — don't commit more than the variant ceiling allows.
        size = min(size, max(0.0, self.available_committed_usdc()))
        return size

    def available_committed_usdc(self) -> float:
        """How much more USDC can this variant commit right now?

        committed = sum of |signed exposure| in open positions
                  + sum of placed-but-unacked order sizes
        available = variant_max_committed_usdc - committed
        """
        signed_exposure = sum(
            abs(v) for v in self.state.current_exposure_by_market.values()
        )
        committed = signed_exposure + max(0.0, self.state.placements_in_flight_usdc)
        return max(0.0, self.variant_max_committed_usdc - committed)

    # ─── state mutators ─────────────────────────────────────────
    def on_placement(self, size_usdc: float, now_ts: Optional[float] = None) -> None:
        """Order has been placed (acked or pending). Counts toward in-flight
        commitment until the fill/cancel arrives.
        """
        self.state.placements_in_flight_usdc += abs(float(size_usdc))
        self.state.placement_timestamps.append(now_ts or time.time())

    def on_fill(self, condition_id: str, signed_usdc: float, size_usdc: Optional[float] = None) -> None:
        """`signed_usdc` positive = added long YES, negative = added short.
        `size_usdc` = absolute placement size (defaults to |signed_usdc|).
        """
        signed = float(signed_usdc)
        gross = float(size_usdc) if size_usdc is not None else abs(signed)

        # Net inventory (legacy).
        cur = self.state.inventory_by_market.get(condition_id, 0.0)
        self.state.inventory_by_market[condition_id] = cur + signed

        # Signed exposure (Fase 1 — same as inventory but cleared on close).
        cur_exp = self.state.current_exposure_by_market.get(condition_id, 0.0)
        self.state.current_exposure_by_market[condition_id] = cur_exp + signed

        # Gross filled (monotonic per cid until close).
        cur_gross = self.state.gross_filled_by_market.get(condition_id, 0.0)
        self.state.gross_filled_by_market[condition_id] = cur_gross + abs(gross)
        self.state.gross_filled_lifetime_usdc += abs(gross)

        # Free the in-flight commitment for the filled portion.
        self.state.placements_in_flight_usdc = max(
            0.0, self.state.placements_in_flight_usdc - abs(gross)
        )

    def on_cancel(self, size_usdc: float) -> None:
        """A placed-but-unfilled order was canceled. Free its in-flight commit."""
        self.state.placements_in_flight_usdc = max(
            0.0, self.state.placements_in_flight_usdc - abs(float(size_usdc))
        )

    def on_close(self, condition_id: str, pnl_usdc: float) -> None:
        """A market resolved. Update bankroll/PnL counters and purge per-cid state."""
        pnl = float(pnl_usdc)
        self.state.daily_pnl_usdc += pnl
        self.state.bankroll_usdc += pnl
        self.state.realized_pnl_lifetime_usdc += pnl
        if pnl < 0:
            self.state.consecutive_losses += 1
            self.state.last_loss_ts = time.time()
        elif pnl > 0:
            self.state.consecutive_losses = 0

        # Purge per-cid state — the market is done.
        self.state.inventory_by_market.pop(condition_id, None)
        self.state.current_exposure_by_market.pop(condition_id, None)
        self.state.gross_filled_by_market.pop(condition_id, None)

    def on_order_open(self) -> None:
        self.state.open_orders_count += 1

    def on_order_close(self) -> None:
        self.state.open_orders_count = max(0, self.state.open_orders_count - 1)

    # ─── snapshot for persistence / dashboard ───────────────────
    def snapshot(self) -> dict:
        """Lightweight dict for logging / dashboard / DB persistence."""
        return {
            "variant": self.variant_name,
            "bankroll_usdc": round(self.state.bankroll_usdc, 4),
            "realized_pnl_lifetime_usdc": round(self.state.realized_pnl_lifetime_usdc, 4),
            "gross_filled_lifetime_usdc": round(self.state.gross_filled_lifetime_usdc, 2),
            "daily_pnl_usdc": round(self.state.daily_pnl_usdc, 4),
            "consecutive_losses": self.state.consecutive_losses,
            "open_orders_count": self.state.open_orders_count,
            "open_markets": len(self.state.current_exposure_by_market),
            "committed_now_usdc": round(
                sum(abs(v) for v in self.state.current_exposure_by_market.values())
                + self.state.placements_in_flight_usdc,
                2,
            ),
            "available_committed_usdc": round(self.available_committed_usdc(), 2),
        }

    # ─── helpers ────────────────────────────────────────────────
    def _placements_last_60s(self, now_ts: float) -> int:
        # Drop timestamps older than 60s. The deque is maxlen-bounded but this
        # keeps the count accurate when traffic is bursty.
        ts = self.state.placement_timestamps
        while ts and ts[0] < now_ts - 60.0:
            ts.popleft()
        return len(ts)

    @staticmethod
    def _utc_day(ts: float) -> int:
        return int(ts // 86400)
