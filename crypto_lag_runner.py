"""Background runner for the crypto-lag MAKER bot.

Spawns ONE daemon thread (one asyncio event loop) that runs N "variants" of
the crypto-lag cycle concurrently via `asyncio.gather`. Each variant has its
own simulator (PaperExecutor), engine (MakerOrderEngine), risk state and
cycle, so we can run e.g. a strict simulator (`main`) and an optimistic
simulator (`permissive`) side-by-side with completely separate KPIs.

Resources that are EXPENSIVE or duplicate-unfriendly are SHARED across
variants:
  - BinanceFeed (single websocket connection, multi-subscriber)
  - CryptoMarketRegistry (single Gamma poll loop)
  - DeribitIVProvider (single REST puller)

Variants are configured under `crypto_lag.variants:` in config.yaml. If that
key is absent the runner falls back to a single implicit variant called
`main` with the legacy `crypto_lag.*` config — backwards compatible.

Example:

    crypto_lag:
      enabled: true
      mode: DEMO
      symbols: [...]
      paper:
        queue_position_enabled: true
      edge_threshold_cents: 2
      variants:
        main:        {}                           # uses defaults
        permissive:
          edge_threshold_cents: 1
          paper:
            queue_position_enabled: false
            q_toxic: 0.10

Usage from `main.py`:

    from crypto_lag_runner import start_crypto_lag, stop_crypto_lag

    handle = start_crypto_lag(config, db, logger)
    # … main loop runs …
    stop_crypto_lag(handle)

`start_crypto_lag` is a no-op if `config.crypto_lag.enabled` is false.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import sys
import threading
from dataclasses import dataclass, field
from typing import Optional


def _deep_merge(base: dict, override: dict) -> dict:
    """Return a deep-merged copy of `base` with `override` taking precedence.

    Lists / scalars in `override` REPLACE those in `base`. Dicts merge
    recursively. Used to combine the global `crypto_lag.*` config with
    per-variant overrides.
    """
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


@dataclass
class VariantHandle:
    """One crypto-lag variant running inside the shared event loop."""
    name: str
    cycle: object             # CryptoLagCycle
    executor: object          # PaperExecutor
    engine: object            # MakerOrderEngine
    risk: object              # CryptoLagRisk


@dataclass
class CryptoLagHandle:
    thread: threading.Thread
    loop: asyncio.AbstractEventLoop
    feed: object              # BinanceFeed (shared)
    registry: object          # CryptoMarketRegistry (shared)
    deribit_iv: object = None # DeribitIVProvider | None (shared)
    variants: dict = field(default_factory=dict)  # variant_name -> VariantHandle

    # Back-compat: expose the *first* variant's `cycle` / `executor` / `engine`
    # under their old names so callers that haven't been updated keep working
    # (the legacy single-bot setup is a 1-variant case).
    @property
    def cycle(self):
        return next(iter(self.variants.values())).cycle if self.variants else None

    @property
    def executor(self):
        return next(iter(self.variants.values())).executor if self.variants else None

    @property
    def engine(self):
        return next(iter(self.variants.values())).engine if self.variants else None


def _resolve_variants(cfg: dict) -> dict[str, dict]:
    """Return the {variant_name: overrides_dict} mapping to spawn.

    Falls back to {"main": {}} if no `variants:` block is present, so legacy
    deployments keep working unchanged.
    """
    variants = cfg.get("variants")
    if not variants:
        return {"main": {}}
    if not isinstance(variants, dict):
        return {"main": {}}
    out: dict[str, dict] = {}
    for name, overrides in variants.items():
        out[str(name)] = dict(overrides or {})
    if not out:
        return {"main": {}}
    return out


def start_crypto_lag(config: dict, db, logger: logging.Logger,
                     notifier=None) -> Optional[CryptoLagHandle]:
    cfg = config.get("crypto_lag") or {}
    if not cfg.get("enabled"):
        logger.info("crypto_lag: disabled (config.crypto_lag.enabled=false)")
        return None
    mode = (cfg.get("mode") or config.get("bot", {}).get("mode", "DEMO")).upper()
    if mode != "DEMO":
        logger.warning(
            "crypto_lag: only DEMO mode is wired today; ignoring requested mode "
            f"{mode!r} and running in DEMO."
        )

    # Late imports so we don't pay startup cost when the module is disabled.
    from strategies.crypto_lag.binance_feed import BinanceFeed
    from strategies.crypto_lag.poly_markets import CryptoMarketRegistry
    from strategies.crypto_lag.paper_executor import PaperExecutor
    from strategies.crypto_lag.order_engine import (
        MakerOrderEngine, FEE_RATE_CRYPTO, MAKER_REBATE_SHARE,
    )
    from strategies.crypto_lag.risk import CryptoLagRisk
    from strategies.crypto_lag.cycle import CryptoLagCycle
    from strategies.crypto_lag.deribit_iv import DeribitIVProvider

    symbols = [s["binance"] for s in cfg.get("symbols", []) if s.get("binance")]
    if not symbols:
        logger.warning("crypto_lag: no symbols configured; skipping start")
        return None

    # ─── shared resources ──────────────────────────────────────
    binance_cfg = cfg.get("binance") or {}
    feed = BinanceFeed(
        symbols=symbols,
        ws_url=binance_cfg.get("ws_url", "wss://stream.binance.com/stream"),
        ws_url_fallback=binance_cfg.get(
            "ws_url_fallback", "wss://data-stream.binance.vision/stream"
        ),
        reconnect_initial_seconds=float(binance_cfg.get("reconnect_initial_seconds", 1.0)),
        reconnect_max_seconds=float(binance_cfg.get("reconnect_max_seconds", 30.0)),
    )
    prefer_horizons = cfg.get("prefer_horizons")
    if prefer_horizons is None:
        prefer_horizons = [int(cfg.get("prefer_horizon_minutes", 5))]
    registry = CryptoMarketRegistry(
        symbols=symbols,
        poll_seconds=float(cfg.get("market_poll_seconds", 30.0)),
        prefer_horizon_minutes=int(cfg.get("prefer_horizon_minutes", 5)),
        prefer_horizons=[int(h) for h in prefer_horizons],
        min_liquidity_usdc=float(cfg.get("min_liquidity_usdc", 0.0)),
        max_wash_share=float(cfg.get("max_wash_share", 0.10)),
    )
    deribit_cfg = cfg.get("deribit") or {}
    deribit_iv = None
    if bool(deribit_cfg.get("enabled", True)):
        try:
            deribit_iv = DeribitIVProvider(
                symbols=symbols,
                refresh_seconds=float(deribit_cfg.get("refresh_seconds", 300.0)),
            )
        except Exception as exc:
            logger.warning(f"crypto_lag: DeribitIVProvider init failed: {exc}")
            deribit_iv = None

    def _bankroll() -> float:
        try:
            if hasattr(db, "get_bankroll"):
                return float(db.get_bankroll() or 0.0)
        except Exception:
            pass
        return float(config.get("bot", {}).get("demo_capital", 1000.0))

    # ─── per-variant stacks ────────────────────────────────────
    variants_cfg = _resolve_variants(cfg)
    variants: dict[str, VariantHandle] = {}
    for vname, voverrides in variants_cfg.items():
        # Build the effective config for this variant: deep-merge global
        # crypto_lag with per-variant overrides.
        eff = _deep_merge(cfg, voverrides)
        # Risk needs the merged crypto_lag block at config["crypto_lag"].
        variant_config_for_risk = dict(config)
        variant_config_for_risk["crypto_lag"] = eff
        paper_cfg_v = eff.get("paper") or {}
        executor = PaperExecutor(
            q_toxic=float(paper_cfg_v.get("q_toxic", 0.30)),
            adverse_haircut_pct=float(paper_cfg_v.get("adverse_haircut_pct", 0.015)),
            fee_rate=float(eff.get("fee_rate", FEE_RATE_CRYPTO)),
            maker_rebate_share=float(eff.get("maker_rebate_share", MAKER_REBATE_SHARE)),
            queue_position_enabled=bool(paper_cfg_v.get("queue_position_enabled", True)),
        )
        risk = CryptoLagRisk(variant_config_for_risk, get_bankroll_usdc=_bankroll)

        # Placement logger: persist every acknowledged order to crypto_lag_quotes
        # tagged with this variant. Closure captures `db` and `vname` so the
        # engine doesn't need to know either. No-op if the DB doesn't expose
        # the method (older deployments).
        def _make_placement_logger(_db, _variant: str):
            if _db is None or not hasattr(_db, "log_crypto_lag_placement"):
                return None
            def _log(order, queue_debt):
                try:
                    _db.log_crypto_lag_placement(order, queue_debt, variant=_variant)
                except Exception as exc:
                    logger.debug(f"placement logger ({_variant}) failed: {exc}")
            return _log
        placement_logger = _make_placement_logger(db, vname)

        engine = MakerOrderEngine(
            executor=executor,
            risk=risk,
            edge_threshold_cents=float(eff.get("edge_threshold_cents", 2.0)),
            replace_threshold_cents=float(eff.get("replace_threshold_cents", 1.0)),
            max_order_age_seconds=float(eff.get("max_order_age_seconds", 30.0)),
            gamma=float(eff.get("gamma", 0.10)),
            arrival_intensity_k=float(eff.get("arrival_intensity_k", 1.5)),
            inventory_skew_threshold=float(eff.get("inventory_skew_threshold", 0.7)),
            fee_rate=float(eff.get("fee_rate", FEE_RATE_CRYPTO)),
            maker_rebate_share=float(eff.get("maker_rebate_share", MAKER_REBATE_SHARE)),
            quote_mode=str(eff.get("quote_mode", "maker")),
            cross_threshold_ticks=float(eff.get("cross_threshold_ticks", 4.0)),
            placement_logger=placement_logger,
        )
        cycle = CryptoLagCycle(
            config=config, feed=feed, registry=registry, executor=executor,
            engine=engine, risk=risk, db=db, notifier=notifier,
            deribit_iv=deribit_iv,
            variant=vname,
            variant_overrides=voverrides,
        )
        variants[vname] = VariantHandle(
            name=vname, cycle=cycle, executor=executor, engine=engine, risk=risk,
        )
        # Operator-friendly summary line
        logger.info(
            f"crypto_lag variant {vname!r}: "
            f"mode={eff.get('quote_mode', 'maker')} "
            f"cross_thr={eff.get('cross_threshold_ticks', 4.0)}t "
            f"queue_pos={paper_cfg_v.get('queue_position_enabled', True)} "
            f"edge_thr={eff.get('edge_threshold_cents', 2)}c "
            f"q_toxic={paper_cfg_v.get('q_toxic', 0.30)}"
        )

    loop = asyncio.new_event_loop()

    async def _bootstrap_and_run():
        # Bootstrap shared resources first (REST historical vol, then start
        # the WS feed and Gamma poller). We only do this ONCE for all variants.
        await feed.bootstrap_historical_vol()
        await feed.start()
        await registry.start()
        if deribit_iv is not None:
            try:
                await deribit_iv.start()
            except Exception as exc:
                logger.warning(f"crypto_lag: deribit_iv.start failed: {exc}")
        # Each variant has its own executor session; start them all then run
        # all cycles concurrently. asyncio.gather propagates the first
        # exception, so a crash in one variant tears them all down (so the
        # supervisor / systemd unit can restart cleanly).
        for vh in variants.values():
            await vh.executor.start()
        try:
            await asyncio.gather(
                *(vh.cycle.run_forever() for vh in variants.values())
            )
        finally:
            for vh in variants.values():
                vh.cycle.stop()
                try:
                    await vh.engine.cancel_all()
                except Exception:
                    pass
                try:
                    await vh.executor.stop()
                except Exception:
                    pass
            await registry.stop()
            await feed.stop()
            if deribit_iv is not None:
                try:
                    await deribit_iv.stop()
                except Exception:
                    pass

    def _thread_main():
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                new_loop.run_until_complete(_bootstrap_and_run())
            finally:
                new_loop.close()
        else:
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_bootstrap_and_run())
            finally:
                loop.close()

    th = threading.Thread(
        target=_thread_main, name="crypto_lag", daemon=True
    )
    th.start()
    logger.info(
        f"crypto_lag: started (DEMO, {len(symbols)} symbols, "
        f"{len(variants)} variant(s): {list(variants.keys())})"
    )
    return CryptoLagHandle(
        thread=th, loop=loop, feed=feed, registry=registry,
        deribit_iv=deribit_iv, variants=variants,
    )


def stop_crypto_lag(handle: Optional[CryptoLagHandle], timeout: float = 10.0) -> None:
    if handle is None:
        return
    # Stop every variant cycle so asyncio.gather unwinds cleanly.
    for vh in handle.variants.values():
        try:
            vh.cycle.stop()  # type: ignore[attr-defined]
        except Exception:
            pass
    handle.thread.join(timeout=timeout)
