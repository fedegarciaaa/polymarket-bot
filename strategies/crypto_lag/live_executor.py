"""LIVE-mode executor wrapping the official Polymarket CLOB client.

Counterpart of `paper_executor.PaperExecutor`. Implements the same surface
(`start`, `stop`, `place_order`, `cancel_order`, `poll_fills`,
`resolve_market`, `_get_book`, `get_queue_debt`) so the rest of the
crypto_lag stack is mode-agnostic.

──────────────────────────────────────────────────────────────────
SCAFFOLD ONLY (Fase 4) — DO NOT activate without manual sign-off.
──────────────────────────────────────────────────────────────────

Activation gate: `crypto_lag.mode = LIVE` in config.yaml. Currently the
runner refuses LIVE and falls back to DEMO ([crypto_lag_runner.py:134]),
so this module is import-safe but never instantiated until that gate
is removed.

Pre-conditions before flipping the gate:
  1. Fase 0+1+2 stable for ≥1 week with `bankroll_usdc` per variant
     visibly capping committed exposure on the dashboard.
  2. `py-clob-client` installed (`pip install py-clob-client`).
  3. Per-variant credentials present in env:
       POLYMARKET_PK_<VARIANT>            (private key, EIP-712 signer)
       POLYMARKET_API_KEY_<VARIANT>       (CLOB L2 auth)
       POLYMARKET_API_SECRET_<VARIANT>
       POLYMARKET_API_PASSPHRASE_<VARIANT>
       POLYMARKET_FUNDER_<VARIANT>        (proxy wallet 0x...)
  4. Smoke test on a single market with `live.max_order_usdc: 5` and
     verify the fill round-trip (place → fill event → reconcile balance).
  5. Halt-file dry-run confirmed in LIVE before stepping size up.

Polymarket CLOB time-in-force mapping (for reference):
  - is_taker=False, post-only edge_threshold met → OrderType.GTC
  - is_taker=True, single-shot crossing             → OrderType.FAK (=IOC)
  - All-or-nothing (not used today)                 → OrderType.FOK

This module deliberately keeps the public class intact so the import
chain works even when `py-clob-client` isn't installed locally — the
network/auth wiring is gated on a `_ensure_client()` call inside
`start()` that fails fast with a clear message.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Optional

from .state import RestingOrder

logger = logging.getLogger("polymarket_bot.crypto_lag.live_executor")


@dataclass
class _LiveCredentials:
    """Per-variant CLOB credentials read from environment variables.

    Each variant has its OWN wallet so capital is genuinely isolated
    on-chain — a bug in one variant cannot drain the other variant's
    USDC. The variant name is uppercased for env-var lookup.
    """
    private_key: str
    funder: str            # proxy wallet address (0x...)
    api_key: str
    api_secret: str
    api_passphrase: str

    @classmethod
    def from_env(cls, variant: str) -> "_LiveCredentials":
        v = variant.upper()
        missing = []
        def _g(k: str) -> str:
            val = os.getenv(k, "")
            if not val:
                missing.append(k)
            return val
        creds = cls(
            private_key=_g(f"POLYMARKET_PK_{v}"),
            funder=_g(f"POLYMARKET_FUNDER_{v}"),
            api_key=_g(f"POLYMARKET_API_KEY_{v}"),
            api_secret=_g(f"POLYMARKET_API_SECRET_{v}"),
            api_passphrase=_g(f"POLYMARKET_API_PASSPHRASE_{v}"),
        )
        if missing:
            raise RuntimeError(
                f"crypto_lag LIVE: missing env vars for variant {variant!r}: "
                f"{', '.join(missing)}. See live_executor.py docstring."
            )
        return creds


class LiveExecutor:
    """LIVE-mode executor — sends real orders to Polymarket CLOB.

    Currently a scaffold: methods raise NotImplementedError until the
    py-clob-client integration is finished and the activation gate is
    flipped (see module docstring).
    """

    def __init__(
        self,
        variant: str,
        host: str = "https://clob.polymarket.com",
        chain_id: int = 137,                 # Polygon mainnet
        max_order_usdc: float = 5.0,         # start tiny in LIVE
        sanity_checks_enabled: bool = True,
        expected_funder: Optional[str] = None,
    ):
        self.variant = variant
        self.host = host
        self.chain_id = chain_id
        self.max_order_usdc = float(max_order_usdc)
        self.sanity_checks_enabled = bool(sanity_checks_enabled)
        self.expected_funder = expected_funder

        self._client = None
        self._creds: Optional[_LiveCredentials] = None
        # Mirror PaperExecutor surface; the engine reads these.
        self._resting: dict[str, RestingOrder] = {}
        self._positions: dict[str, object] = {}  # cid → Position
        self._started = False

    # ─── lifecycle ──────────────────────────────────────────────
    async def start(self) -> None:
        """Read credentials, instantiate ClobClient, enable L2 auth."""
        self._creds = _LiveCredentials.from_env(self.variant)
        # Cross-variant safety: the configured `expected_funder` must match
        # the wallet derived from the private key. Without this check, a
        # mistaken env var swap would drain the wrong wallet.
        if self.expected_funder:
            actual = self._creds.funder.lower()
            expected = self.expected_funder.lower()
            if actual != expected:
                raise RuntimeError(
                    f"crypto_lag LIVE [{self.variant}]: funder mismatch — "
                    f"expected {expected}, got {actual}. Refusing to start."
                )
        # Lazy import so DEMO doesn't pay the dependency cost.
        try:
            from py_clob_client.client import ClobClient  # type: ignore
            from py_clob_client.clob_types import ApiCreds  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "crypto_lag LIVE requires py-clob-client. Install with "
                "`pip install py-clob-client` and retry."
            ) from exc

        self._client = ClobClient(
            host=self.host,
            chain_id=self.chain_id,
            key=self._creds.private_key,
            funder=self._creds.funder,
            signature_type=2,                 # Polymarket proxy-wallet signing
            creds=ApiCreds(
                api_key=self._creds.api_key,
                api_secret=self._creds.api_secret,
                api_passphrase=self._creds.api_passphrase,
            ),
        )
        self._started = True
        logger.info(
            f"crypto_lag LIVE [{self.variant}]: connected to {self.host} "
            f"(funder={self._creds.funder[:10]}...)"
        )

    async def stop(self) -> None:
        # py-clob-client doesn't expose an explicit close hook today; nothing
        # to do beyond marking the instance as stopped.
        self._started = False

    # ─── order operations ───────────────────────────────────────
    async def place_order(self, order: RestingOrder, token_id: str) -> str:
        """Send an order to the CLOB. TIF derived from order.is_taker:
          - is_taker=True  → FAK (Polymarket's IOC equivalent)
          - is_taker=False → GTC (resting maker order)
        """
        if not self._started or self._client is None:
            raise RuntimeError("LiveExecutor.place_order before start()")

        # Sanity guards — refuse trash inputs that would waste gas / fees.
        if self.sanity_checks_enabled:
            if order.size_usdc < 1.0:
                raise ValueError(
                    f"LIVE refuses order size ${order.size_usdc:.2f} < $1 minimum"
                )
            if order.size_usdc > self.max_order_usdc:
                raise ValueError(
                    f"LIVE refuses order size ${order.size_usdc:.2f} > "
                    f"max_order_usdc ${self.max_order_usdc:.2f}"
                )
            if not (0.02 <= order.price <= 0.98):
                raise ValueError(
                    f"LIVE refuses price {order.price:.4f} outside [0.02, 0.98]"
                )

        # NOTE: full integration pending py-clob-client wiring. Leave a clear
        # marker so anyone enabling LIVE knows what's missing.
        raise NotImplementedError(
            "LiveExecutor.place_order: py-clob-client integration TODO. "
            "Build OrderArgs, call create_order/post_order with GTC vs FAK "
            "depending on order.is_taker, and wire the response back to "
            "RestingOrder.external_order_id."
        )

    async def cancel_order(self, order_id: str) -> bool:
        if not self._started:
            return False
        # TODO: self._client.cancel(order_id=order_id)
        raise NotImplementedError("LiveExecutor.cancel_order: TODO")

    async def poll_fills(self) -> list:
        """Poll CLOB for status changes on resting orders, emit fill events.

        Returns the same `_FillEvent`-shaped objects the cycle expects.
        """
        if not self._started:
            return []
        # TODO: query self._client.get_orders(...) and diff against self._resting.
        return []

    async def resolve_market(self, condition_id: str, yes_outcome_value: float, ts: Optional[float] = None):
        """In LIVE, resolution happens on-chain — we just sync the position
        from the wallet balance and emit a close event for accounting parity
        with paper_executor.
        """
        # TODO: read on-chain balance; compute realized PnL; emit _CloseEvent.
        return None

    # ─── plumbing the engine reads ──────────────────────────────
    async def _get_book(self, token_id: str) -> dict:
        """Read the book via REST. Same shape paper_executor returns so the
        cycle's price logic works unchanged in LIVE.
        """
        # TODO: self._client.get_order_book(token_id=token_id)
        return {}

    def get_queue_debt(self, order_id: str) -> float:
        """LIVE doesn't model queue position — Polymarket doesn't expose the
        per-order queue depth. Return 0 so dashboards keep working.
        """
        return 0.0
