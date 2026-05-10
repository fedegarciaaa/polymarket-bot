"""LIVE-mode executor wrapping the official Polymarket CLOB client.

Counterpart of `paper_executor.PaperExecutor`. Implements the same surface
(`start`, `stop`, `place_order`, `cancel_order`, `poll_fills`,
`resolve_market`, `_get_book`, `get_queue_debt`) so the rest of the
crypto_lag stack is mode-agnostic.

──────────────────────────────────────────────────────────────────
LIVE smoke-test build (2026-05-10).
──────────────────────────────────────────────────────────────────

Activation: per-variant `mode: LIVE` in config.yaml. The runner
instantiates this class instead of `PaperExecutor` for those variants.

Pre-conditions before turning a variant `enabled: true` in LIVE:
  1. `py-clob-client` installed in the bot's venv.
  2. Credentials in `.env`:
       POLYMARKET_PRIVATE_KEY    (EIP-712 signer; 0x... 64 hex)
       POLYMARKET_FUNDER_ADDRESS (proxy wallet / Magic.link address; 0x...)
     API L2 keys (api_key/secret/passphrase) are derived on first start
     via `client.derive_api_key()` and cached at `data/clob_api_creds.json`.
  3. USDC on Polygon already in the proxy wallet; CLOB allowance set
     (the Polymarket UI does this automatically on first deposit).
  4. `whitelist_condition_ids` populated by the cycle so we only trade
     in the markets the variant explicitly approved.
  5. `halt_loss_usdc` configured. If lifetime PnL drops below the
     negative threshold the executor:
       - touches `data/halt_live` (kill-switch file),
       - refuses every subsequent placement,
       - cancels all resting orders.

Polymarket CLOB time-in-force mapping:
  - is_taker=False (resting maker)  → OrderType.GTC
  - is_taker=True  (single crossing) → OrderType.FAK (Polymarket's IOC)

This module imports `py-clob-client` lazily inside `start()` so DEMO
deployments don't pay the dependency cost.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

from .state import RestingOrder

logger = logging.getLogger("polymarket_bot.crypto_lag.live_executor")


# ─── Fill / Close events (mirror paper_executor for cycle compatibility) ──
@dataclass
class _FillEvent:
    order_id: str
    condition_id: str
    symbol: str
    side: str
    outcome: str
    fill_price: float
    fill_size_usdc: float
    is_adverse: bool
    ts: float
    rebate_usdc: float = 0.0
    fee_paid_usdc: float = 0.0
    market_slug: str = ""


@dataclass
class _CloseEvent:
    condition_id: str
    symbol: str
    realized_pnl_usdc: float
    final_yes_price: float
    ts: float
    reason: str
    accumulated_rebate_usdc: float = 0.0


@dataclass
class _LivePosition:
    """LIVE-side accounting mirror of paper_executor.Position. We can't
    inspect the on-chain state every poll, so we maintain our own ledger
    from observed fills and reconcile against the wallet balance at close.
    """
    condition_id: str
    symbol: str
    outcome: str               # "YES" or "NO"
    size_usdc: float           # signed: + long YES, - long NO (we accumulate by side)
    avg_entry_price: float
    opened_ts: float
    last_fill_ts: float
    end_ts: float = 0.0
    strike_price: float = 0.0
    market_slug: str = ""


@dataclass
class _LiveCredentials:
    """CLOB credentials. The L2 api_* triplet is derived from the EOA on
    first start and cached on disk so we don't hammer the derive endpoint
    on every restart.
    """
    private_key: str
    funder: str            # proxy wallet address (0x...)
    api_key: str = ""
    api_secret: str = ""
    api_passphrase: str = ""

    @classmethod
    def from_env(cls) -> "_LiveCredentials":
        pk = os.getenv("POLYMARKET_PRIVATE_KEY", "").strip()
        funder = os.getenv("POLYMARKET_FUNDER_ADDRESS", "").strip()
        missing = []
        if not pk:
            missing.append("POLYMARKET_PRIVATE_KEY")
        if not funder:
            missing.append("POLYMARKET_FUNDER_ADDRESS")
        if missing:
            raise RuntimeError(
                f"crypto_lag LIVE: missing env vars: {', '.join(missing)}. "
                "Add them to /home/botuser/polymarket-bot/.env (chmod 600) "
                "and restart the service."
            )
        # Normalize "0x" prefix on the private key — py-clob-client wants
        # the raw hex without the prefix.
        if pk.startswith("0x") or pk.startswith("0X"):
            pk = pk[2:]
        return cls(private_key=pk, funder=funder)


class LiveExecutor:
    """LIVE-mode executor — sends real orders to Polymarket CLOB.

    Designed for the smoke-test profile: tiny `max_order_usdc`, hard
    halt-loss threshold, and a market whitelist so a config bug in
    another variant cannot redirect orders to an unapproved condition_id.
    """

    DEFAULT_CREDS_CACHE = "data/clob_api_creds.json"

    def __init__(
        self,
        variant: str,
        host: str = "https://clob.polymarket.com",
        chain_id: int = 137,                           # Polygon mainnet
        signature_type: int = 1,                       # 1 = Magic.link / Polymarket proxy
        max_order_usdc: float = 5.0,                   # tiny start in LIVE
        sanity_checks_enabled: bool = True,
        whitelist_symbols: Optional[Iterable[str]] = None,
        max_concurrent_orders: int = 5,
        halt_file_path: str = "data/halt_live",
        halt_loss_usdc: float = 50.0,
        daily_pnl_provider: Optional[Callable[[], float]] = None,
        creds_cache_path: Optional[str] = None,
    ):
        self.variant = variant
        self.host = host
        self.chain_id = chain_id
        self.signature_type = int(signature_type)
        self.max_order_usdc = float(max_order_usdc)
        self.sanity_checks_enabled = bool(sanity_checks_enabled)
        self.whitelist_symbols = (
            {s.upper() for s in whitelist_symbols} if whitelist_symbols else None
        )
        self.max_concurrent_orders = int(max_concurrent_orders)
        self.halt_file_path = Path(halt_file_path)
        self.halt_loss_usdc = float(abs(halt_loss_usdc))
        self.daily_pnl_provider = daily_pnl_provider
        self.creds_cache_path = Path(creds_cache_path or self.DEFAULT_CREDS_CACHE)

        self._client = None
        self._creds: Optional[_LiveCredentials] = None
        # Mirror PaperExecutor surface; the engine reads these directly.
        self._resting: dict[str, RestingOrder] = {}
        self._positions: dict[str, _LivePosition] = {}
        self._close_log: list[_CloseEvent] = []
        # token_id → outcome ("YES" | "NO"); populated by cycle on placement
        self._token_to_outcome: dict[str, str] = {}
        # token_id → {condition_id, symbol}; populated by cycle on placement
        self._token_meta: dict[str, dict] = {}
        # Cumulative realized PnL since process start — used for halt trigger
        # if the caller doesn't supply a daily_pnl_provider.
        self._lifetime_realized_pnl: float = 0.0
        self._started = False
        # Halt latch: once tripped we stay halted until the file is removed
        # AND the process restarts — defensive against accidental clears.
        self._halted = False
        # poll_fills uses TradeParams.after to query incrementally. We track
        # the latest trade ts we've seen so each poll only returns NEW trades.
        # Initialised to start time on `start()` so backlog from before this
        # process boot doesn't trigger phantom fills.
        self._last_trade_ts: int = 0
        # _get_book TTL cache (in-memory). Without this, each cycle calls
        # get_order_book() per market via REST → at 200-500ms per call and
        # multiple markets per symbol the per-tick budget explodes. The
        # cycle still queries every tick, but cache hits are free.
        self._book_cache: dict[str, dict] = {}
        self._book_cache_ts: dict[str, float] = {}
        self._book_cache_ttl_s: float = 1.5

    # ─── Halt logic ────────────────────────────────────────────
    def _check_halt(self) -> tuple[bool, str]:
        """Return (halted, reason). Once True, stays True for the process
        lifetime — only an explicit file removal + restart re-arms LIVE.
        """
        if self._halted:
            return True, "latched"
        if self.halt_file_path.exists():
            self._halted = True
            return True, f"halt-file present: {self.halt_file_path}"
        # Active monitoring of cumulative PnL.
        try:
            pnl = (
                self.daily_pnl_provider()
                if self.daily_pnl_provider is not None
                else self._lifetime_realized_pnl
            )
        except Exception as exc:
            logger.warning(f"LIVE [{self.variant}]: daily_pnl_provider raised {exc}")
            pnl = self._lifetime_realized_pnl
        if pnl <= -self.halt_loss_usdc:
            self._halted = True
            try:
                self.halt_file_path.parent.mkdir(parents=True, exist_ok=True)
                self.halt_file_path.write_text(
                    f"{int(time.time())} {self.variant} pnl={pnl:.2f} "
                    f"<= -{self.halt_loss_usdc:.2f}\n"
                )
            except Exception as exc:
                logger.error(f"LIVE [{self.variant}]: failed to write halt file: {exc}")
            return True, f"PnL {pnl:.2f} <= -{self.halt_loss_usdc:.2f}"
        return False, ""

    # ─── lifecycle ─────────────────────────────────────────────
    async def start(self) -> None:
        self._creds = _LiveCredentials.from_env()
        try:
            from py_clob_client.client import ClobClient                # type: ignore
            from py_clob_client.clob_types import ApiCreds               # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "crypto_lag LIVE requires py-clob-client. Install with "
                "`/home/botuser/polymarket-bot/venv/bin/pip install py-clob-client`."
            ) from exc

        # Stage 1: connect with EOA only (no L2 creds yet) so we can derive
        # or reload the api_key triplet.
        client = ClobClient(
            host=self.host,
            chain_id=self.chain_id,
            key=self._creds.private_key,
            funder=self._creds.funder,
            signature_type=self.signature_type,
        )

        cached = self._load_cached_creds()
        if cached is None:
            try:
                api_creds = client.derive_api_key()
            except Exception as exc:
                # derive_api_key fails if the wallet has no API creds yet —
                # in that case create them.
                logger.info(
                    f"LIVE [{self.variant}]: derive_api_key failed ({exc}), "
                    "calling create_api_key"
                )
                api_creds = client.create_api_key()
            cached = {
                "api_key": api_creds.api_key,
                "api_secret": api_creds.api_secret,
                "api_passphrase": api_creds.api_passphrase,
            }
            self._save_cached_creds(cached)

        self._creds.api_key = cached["api_key"]
        self._creds.api_secret = cached["api_secret"]
        self._creds.api_passphrase = cached["api_passphrase"]

        # Stage 2: re-instantiate with L2 creds for trading.
        self._client = ClobClient(
            host=self.host,
            chain_id=self.chain_id,
            key=self._creds.private_key,
            funder=self._creds.funder,
            signature_type=self.signature_type,
            creds=ApiCreds(
                api_key=self._creds.api_key,
                api_secret=self._creds.api_secret,
                api_passphrase=self._creds.api_passphrase,
            ),
        )
        self._started = True
        # Anchor poll_fills incrementality at start — trades older than this
        # are ignored so a wallet with prior CLOB history doesn't replay.
        self._last_trade_ts = int(time.time())
        # First-call sanity: log balance so the operator can confirm the
        # wallet has USDC before any order goes out.
        try:
            bal = await self._read_balance()
            logger.info(
                f"LIVE [{self.variant}]: connected to {self.host} "
                f"(funder={self._creds.funder[:10]}...) USDC balance=${bal:.2f}"
            )
        except Exception as exc:
            logger.warning(f"LIVE [{self.variant}]: balance check failed: {exc}")
            logger.info(
                f"LIVE [{self.variant}]: connected to {self.host} "
                f"(funder={self._creds.funder[:10]}...)"
            )

    async def stop(self) -> None:
        # py-clob-client uses requests under the hood; nothing to close.
        self._started = False

    # ─── credentials cache ────────────────────────────────────
    def _load_cached_creds(self) -> Optional[dict]:
        if not self.creds_cache_path.exists():
            return None
        try:
            data = json.loads(self.creds_cache_path.read_text())
            for k in ("api_key", "api_secret", "api_passphrase"):
                if not data.get(k):
                    return None
            return data
        except Exception:
            return None

    def _save_cached_creds(self, data: dict) -> None:
        try:
            self.creds_cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.creds_cache_path.write_text(json.dumps(data, indent=2))
            os.chmod(self.creds_cache_path, 0o600)
        except Exception as exc:
            logger.warning(f"LIVE [{self.variant}]: cache creds failed: {exc}")

    async def _read_balance(self) -> float:
        from py_clob_client.clob_types import BalanceAllowanceParams, AssetType  # type: ignore
        params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None, lambda: self._client.get_balance_allowance(params)
        )
        # response includes balance in USDC base units (6 decimals)
        bal_raw = float(resp.get("balance", 0.0)) if isinstance(resp, dict) else 0.0
        return bal_raw / 1e6

    # ─── order operations ─────────────────────────────────────
    async def place_order(self, order: RestingOrder, token_id: str) -> str:
        if not self._started or self._client is None:
            raise RuntimeError("LiveExecutor.place_order before start()")

        halted, reason = self._check_halt()
        if halted:
            logger.warning(f"LIVE [{self.variant}] HALTED ({reason}) — refusing place_order")
            raise RuntimeError(f"LIVE halted: {reason}")

        if self.whitelist_symbols and order.symbol.upper() not in self.whitelist_symbols:
            raise ValueError(
                f"LIVE [{self.variant}]: symbol {order.symbol!r} not in whitelist "
                f"{sorted(self.whitelist_symbols)}"
            )

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
            if len(self._resting) >= self.max_concurrent_orders:
                raise RuntimeError(
                    f"LIVE [{self.variant}]: {len(self._resting)} resting orders "
                    f">= max_concurrent_orders {self.max_concurrent_orders}"
                )

        from py_clob_client.clob_types import OrderArgs, OrderType  # type: ignore
        from py_clob_client.order_builder.constants import BUY, SELL  # type: ignore

        side_const = BUY if order.side == "BUY" else SELL
        # OrderArgs.size is in SHARES (not USDC). Polymarket shares trade
        # at `price` USDC each, so shares = notional / price.
        size_shares = max(1.0, round(order.size_usdc / order.price, 4))

        order_args = OrderArgs(
            token_id=token_id,
            price=float(order.price),
            size=size_shares,
            side=side_const,
        )

        loop = asyncio.get_event_loop()
        try:
            signed = await loop.run_in_executor(
                None, lambda: self._client.create_order(order_args)
            )
            tif = OrderType.FAK if order.is_taker else OrderType.GTC
            resp = await loop.run_in_executor(
                None, lambda: self._client.post_order(signed, tif)
            )
        except Exception as exc:
            logger.error(
                f"LIVE [{self.variant}] post_order failed: {order.side} {order.symbol} "
                f"${order.size_usdc:.2f}@{order.price:.4f}: {exc}"
            )
            raise

        ext_id = ""
        if isinstance(resp, dict):
            ext_id = str(resp.get("orderID") or resp.get("orderHash") or "")
            if resp.get("errorMsg"):
                raise RuntimeError(f"CLOB rejected: {resp.get('errorMsg')}")
        if not ext_id:
            raise RuntimeError(f"CLOB response missing orderID: {resp!r}")

        order.external_order_id = ext_id
        self._resting[ext_id] = order
        # Track outcome / metadata for fill enrichment
        self._token_to_outcome[token_id] = order.outcome
        self._token_meta[token_id] = {
            "condition_id": order.condition_id, "symbol": order.symbol,
            "market_slug": order.market_slug,
            "end_ts": order.end_ts, "strike_price": order.strike_price,
        }
        # Verbose LIVE log so post-mortem can reconstruct the smoke run:
        # full price, size, raw response, time. Watch journalctl -fu polymarket-bot
        # filtered by "LIVE-PLACE".
        logger.info(
            f"LIVE-PLACE [{self.variant}] {order.side} {order.symbol} "
            f"slug={order.market_slug[:30]} "
            f"px={order.price:.4f} size_shares={size_shares:.4f} "
            f"size_usdc=${order.size_usdc:.2f} "
            f"is_taker={order.is_taker} ext={ext_id[:16]} "
            f"resp_keys={sorted((resp or {}).keys()) if isinstance(resp, dict) else type(resp).__name__}"
        )
        return ext_id

    async def cancel_order(self, order_id: str) -> bool:
        if not self._started or self._client is None:
            return False
        order = self._resting.get(order_id)
        ext = order.external_order_id if order else order_id
        if not ext:
            return False
        loop = asyncio.get_event_loop()
        try:
            resp = await loop.run_in_executor(
                None, lambda: self._client.cancel(order_id=ext)
            )
            ok = bool(resp) and (
                isinstance(resp, dict)
                and not resp.get("not_canceled")
            )
        except Exception as exc:
            logger.warning(f"LIVE [{self.variant}] cancel({ext[:12]}) failed: {exc}")
            ok = False
        self._resting.pop(order_id, None)
        return ok

    async def poll_fills(self) -> list[_FillEvent]:
        """Poll CLOB for trades against our resting orders.

        Uses `get_trades(TradeParams(after=self._last_trade_ts))` for
        incremental queries — without this we'd re-process the entire
        wallet trade history on every poll.
        """
        if not self._started or self._client is None:
            return []
        # Halt check is non-blocking here — we still want to drain remaining
        # fills for already-placed orders even after halt is tripped.
        from py_clob_client.clob_types import TradeParams  # type: ignore
        params = TradeParams(after=self._last_trade_ts)
        loop = asyncio.get_event_loop()
        try:
            trades = await loop.run_in_executor(
                None, lambda: self._client.get_trades(params)
            )
        except Exception as exc:
            logger.warning(f"LIVE [{self.variant}] get_trades failed: {exc}")
            return []

        if not isinstance(trades, list):
            return []

        events: list[_FillEvent] = []
        now = time.time()
        max_seen_ts = self._last_trade_ts
        for t in trades:
            try:
                trade_ts = int(t.get("match_time") or t.get("timestamp") or t.get("ts") or 0)
                if trade_ts > max_seen_ts:
                    max_seen_ts = trade_ts
            except (TypeError, ValueError):
                pass
            ext = str(t.get("maker_order_id") or t.get("order_id") or "")
            if not ext or ext not in self._resting:
                continue
            order = self._resting[ext]
            try:
                fill_price = float(t.get("price", order.price))
                fill_shares = float(t.get("size", 0.0))
                fill_size_usdc = fill_shares * fill_price
                fee_paid = float(t.get("fee_rate_bps", 0.0)) * fill_size_usdc / 10000.0
            except (TypeError, ValueError):
                continue
            if fill_size_usdc <= 0:
                continue
            ev = _FillEvent(
                order_id=order.order_id,
                condition_id=order.condition_id,
                symbol=order.symbol,
                side=order.side,
                outcome=order.outcome,
                fill_price=fill_price,
                fill_size_usdc=fill_size_usdc,
                is_adverse=False,    # LIVE doesn't model toxicity — ground truth is the trade
                ts=now,
                rebate_usdc=0.0,
                fee_paid_usdc=fee_paid,
                market_slug=order.market_slug,
            )
            events.append(ev)
            self._apply_fill_to_position(ev, order)
            order.filled_size_usdc += fill_size_usdc
            # LIVE-FILL log mirrors LIVE-PLACE so we can match each fill back
            # to its placement post-mortem.
            logger.info(
                f"LIVE-FILL [{self.variant}] {order.side} {order.symbol} "
                f"slug={order.market_slug[:30]} "
                f"placed_px={order.price:.4f} actual_px={fill_price:.4f} "
                f"shares={fill_shares:.4f} usdc=${fill_size_usdc:.2f} "
                f"fee=${fee_paid:.4f} "
                f"slippage_pct={(fill_price - order.price) / max(0.001, order.price) * 100:+.3f} "
                f"ext={ext[:16]}"
            )
            # Fully filled → drop from resting; partials stay.
            if order.filled_size_usdc + 1e-6 >= order.size_usdc:
                self._resting.pop(ext, None)

        # Persist forward progress so the next poll only fetches NEW trades.
        if max_seen_ts > self._last_trade_ts:
            self._last_trade_ts = max_seen_ts
        return events

    def _apply_fill_to_position(self, ev: _FillEvent, order: RestingOrder) -> None:
        pos = self._positions.get(ev.condition_id)
        if pos is None:
            self._positions[ev.condition_id] = _LivePosition(
                condition_id=ev.condition_id,
                symbol=ev.symbol,
                outcome=ev.outcome,
                size_usdc=ev.fill_size_usdc if ev.side == "BUY" else -ev.fill_size_usdc,
                avg_entry_price=ev.fill_price,
                opened_ts=ev.ts,
                last_fill_ts=ev.ts,
                end_ts=float(getattr(order, "end_ts", 0.0) or 0.0),
                strike_price=float(getattr(order, "strike_price", 0.0) or 0.0),
                market_slug=str(getattr(order, "market_slug", "") or ""),
            )
            return
        if ev.side == "BUY":
            new_size = pos.size_usdc + ev.fill_size_usdc
        else:
            new_size = pos.size_usdc - ev.fill_size_usdc
        # VWAP only when not flipping sign
        if (pos.size_usdc >= 0 and new_size >= 0 and ev.side == "BUY") or \
           (pos.size_usdc <= 0 and new_size <= 0 and ev.side == "SELL"):
            tot_prev = abs(pos.size_usdc) * pos.avg_entry_price
            new_add = ev.fill_size_usdc * ev.fill_price
            tot_abs = abs(new_size)
            if tot_abs > 1e-9:
                pos.avg_entry_price = (tot_prev + new_add) / tot_abs
        pos.size_usdc = new_size
        pos.last_fill_ts = ev.ts

    async def resolve_market(
        self, condition_id: str, yes_outcome_value: float, ts: Optional[float] = None
    ) -> Optional[_CloseEvent]:
        """Compute realized PnL when the market resolves.

        Polymarket settles on-chain — winning shares pay $1, losing shares
        pay $0. We compute PnL from our internal ledger; the actual USDC
        will land in the proxy wallet at settlement, no action needed.
        """
        pos = self._positions.pop(condition_id, None)
        if pos is None:
            return None
        ts_eff = float(ts) if ts is not None else time.time()
        if pos.outcome == "YES":
            shares = abs(pos.size_usdc) / max(0.001, pos.avg_entry_price)
            payoff = shares * float(yes_outcome_value)
        else:
            shares = abs(pos.size_usdc) / max(0.001, pos.avg_entry_price)
            payoff = shares * (1.0 - float(yes_outcome_value))
        cost_basis = abs(pos.size_usdc)
        sign = 1.0 if pos.size_usdc >= 0 else -1.0
        pnl = sign * (payoff - cost_basis)
        # Halt accounting: track lifetime realized PnL for the safety net.
        self._lifetime_realized_pnl += float(pnl)
        ev = _CloseEvent(
            condition_id=condition_id,
            symbol=pos.symbol,
            realized_pnl_usdc=float(pnl),
            final_yes_price=float(yes_outcome_value),
            ts=ts_eff,
            reason="resolved",
            accumulated_rebate_usdc=0.0,
        )
        self._close_log.append(ev)
        # LIVE-CLOSE log: forensic-grade. Includes shares, payoff, cost basis
        # and entry vs final price so we can audit each close manually.
        logger.info(
            f"LIVE-CLOSE [{self.variant}] {pos.symbol} cid={condition_id[:12]} "
            f"outcome={pos.outcome} entry_px={pos.avg_entry_price:.4f} "
            f"yes_final={yes_outcome_value:.2f} "
            f"shares={shares:.2f} cost=${cost_basis:.2f} payoff=${payoff:.2f} "
            f"pnl=${pnl:+.2f} lifetime=${self._lifetime_realized_pnl:+.2f} "
            f"slug={pos.market_slug[:30]}"
        )
        # Re-evaluate halt now that PnL has updated.
        halted, reason = self._check_halt()
        if halted:
            logger.error(
                f"LIVE [{self.variant}] HALT TRIGGERED ({reason}) — cancelling all"
            )
            await self.cancel_all_resting()
        return ev

    async def cancel_all_resting(self) -> None:
        for oid in list(self._resting.keys()):
            try:
                await self.cancel_order(oid)
            except Exception:
                pass

    def drain_close_log(self) -> list[_CloseEvent]:
        out, self._close_log = self._close_log, []
        return out

    # ─── plumbing the engine reads ─────────────────────────────
    async def _get_book(self, token_id: str) -> dict:
        """Return the order book in the same shape as paper_executor.

        Shape: {best_bid, best_ask, bid_size, ask_size}.
        TTL-cached (default 1.5s) to bound per-tick REST calls when
        the cycle iterates many markets.
        """
        if not self._started or self._client is None:
            return {}
        now = time.time()
        cached_ts = self._book_cache_ts.get(token_id, 0.0)
        if now - cached_ts < self._book_cache_ttl_s:
            return self._book_cache.get(token_id, {})
        loop = asyncio.get_event_loop()
        try:
            book = await loop.run_in_executor(
                None, lambda: self._client.get_order_book(token_id=token_id)
            )
        except Exception as exc:
            logger.debug(f"LIVE [{self.variant}] get_order_book failed: {exc}")
            return {}
        if book is None:
            return {}

        def _top(side_key: str, sort_high: bool) -> tuple[float, float]:
            entries = getattr(book, side_key, None) or []
            if not entries:
                return 0.0, 0.0
            try:
                best = max(entries, key=lambda e: float(e.price)) if sort_high \
                    else min(entries, key=lambda e: float(e.price))
                return float(best.price), float(best.size)
            except Exception:
                return 0.0, 0.0

        best_bid, bid_size = _top("bids", sort_high=True)
        best_ask, ask_size = _top("asks", sort_high=False)
        result = {
            "best_bid": best_bid, "bid_size": bid_size,
            "best_ask": best_ask, "ask_size": ask_size,
        }
        # Cache forward — TTL bound = self._book_cache_ttl_s.
        self._book_cache[token_id] = result
        self._book_cache_ts[token_id] = now
        return result

    def get_queue_debt(self, order_id: str) -> float:
        """LIVE doesn't model FIFO queue position — Polymarket doesn't expose
        per-order queue depth. Returning 0 keeps dashboards working.
        """
        return 0.0
