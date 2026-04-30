"""In-memory state for the crypto-lag module.

These are the only structs shared between feed/model/order/risk. Everything
that crosses module boundaries flows through one of these dataclasses, so when
something looks wrong you check these first.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional


# ─────────────────────────────────────────────────────────────────
# Binance feed state
# ─────────────────────────────────────────────────────────────────
@dataclass
class MarketState:
    """Per-symbol rolling snapshot maintained by BinanceFeed.

    All timestamps are POSIX seconds (float). `last_update_ts` is the wall-clock
    time we received the most recent message — used by the staleness watchdog.
    """
    symbol: str
    best_bid: float = 0.0
    best_ask: float = 0.0
    best_bid_qty: float = 0.0
    best_ask_qty: float = 0.0
    last_trade_price: float = 0.0
    last_trade_qty: float = 0.0
    last_update_ts: float = 0.0
    # Rolling history of (ts, mid_price). Capped at ~5 minutes.
    price_history: Deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=600))
    # Rolling signed trade volume over last ~5s (positive = aggressive buys).
    trade_flow_5s: float = 0.0

    @property
    def mid(self) -> float:
        if self.best_bid > 0 and self.best_ask > 0:
            return 0.5 * (self.best_bid + self.best_ask)
        return self.last_trade_price

    @property
    def book_imbalance(self) -> float:
        """Top-of-book imbalance ∈ [-1, 1]. >0 means more bid size than ask."""
        denom = self.best_bid_qty + self.best_ask_qty
        if denom <= 0:
            return 0.0
        return (self.best_bid_qty - self.best_ask_qty) / denom

    def is_stale(self, now_ts: float, stale_seconds: float) -> bool:
        return (now_ts - self.last_update_ts) > stale_seconds


# ─────────────────────────────────────────────────────────────────
# Polymarket market metadata
# ─────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PolyCryptoMarket:
    """One active 15-min Up/Down market on Polymarket."""
    condition_id: str
    market_slug: str
    question: str
    symbol: str                    # BTCUSDT | ETHUSDT | SOLUSDT
    direction: str                 # "up" | "down"
    strike_price: float            # the reference price the market resolves against
    end_ts: float                  # POSIX seconds
    token_yes: str                 # CLOB token id for the YES outcome
    token_no: str
    tick_size: float = 0.01        # CLOB tick


# ─────────────────────────────────────────────────────────────────
# Resting orders & positions
# ─────────────────────────────────────────────────────────────────
@dataclass
class RestingOrder:
    """A maker order placed by the bot. `external_order_id` is None until the
    venue (or paper executor) acknowledges it."""
    order_id: str                  # local UUID
    external_order_id: Optional[str]
    symbol: str
    condition_id: str
    side: str                      # "BUY" | "SELL"
    outcome: str                   # "YES" | "NO"
    price: float                   # in [0.01, 0.99]
    size_usdc: float
    placed_ts: float
    last_replace_ts: float
    filled_size_usdc: float = 0.0
    status: str = "open"           # open | partially_filled | filled | canceled


@dataclass
class Position:
    """Aggregated position in a single market across maker fills."""
    condition_id: str
    symbol: str
    outcome: str                   # "YES" | "NO" — net direction
    size_usdc: float               # net long
    avg_entry_price: float
    opened_ts: float
    last_fill_ts: float
    realized_pnl_usdc: float = 0.0
    unrealized_pnl_usdc: float = 0.0


# ─────────────────────────────────────────────────────────────────
# Cycle snapshot — written to crypto_lag_state_snapshots for backtesting
# ─────────────────────────────────────────────────────────────────
@dataclass
class CryptoLagSnapshot:
    ts: float
    symbol: str
    binance_mid: float
    sigma_realized: float
    book_imbalance: float
    p_model: float
    poly_bid: float
    poly_ask: float
    poly_mid: float
    fair_mid: float
    edge_bid: float
    edge_ask: float
    decision: str                  # "QUOTE_BID" | "QUOTE_ASK" | "QUOTE_BOTH" | "WAIT"
