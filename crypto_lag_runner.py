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

    Variants with `enabled: false` are skipped. Falls back to {"main": {}} if
    no `variants:` block is present, so legacy deployments keep working.
    If every variant in the block is disabled, returns {} (caller should treat
    as "nothing to run" and not start the module).
    """
    variants = cfg.get("variants")
    if not variants:
        return {"main": {}}
    if not isinstance(variants, dict):
        return {"main": {}}
    out: dict[str, dict] = {}
    for name, overrides in variants.items():
        ov = dict(overrides or {})
        # Default enabled=true; explicit `enabled: false` skips this variant.
        if ov.get("enabled", True) is False:
            continue
        out[str(name)] = ov
    if not out and variants:
        # Every variant was disabled — return {} so the caller can refuse to start.
        return {}
    if not out:
        return {"main": {}}
    return out


def start_crypto_lag(config: dict, db, logger: logging.Logger,
                     notifier=None) -> Optional[CryptoLagHandle]:
    cfg = config.get("crypto_lag") or {}
    if not cfg.get("enabled"):
        logger.info("crypto_lag: disabled (config.crypto_lag.enabled=false)")
        return None
    global_mode = (cfg.get("mode") or config.get("bot", {}).get("mode", "DEMO")).upper()
    # Per-variant mode override is read inside the loop below — DEMO globally
    # but a single variant can opt into LIVE via `mode: LIVE` in its block.
    # We no longer warn on global LIVE: the per-variant gate handles activation.

    # Late imports so we don't pay startup cost when the module is disabled.
    from strategies.crypto_lag.binance_feed import BinanceFeed
    from strategies.crypto_lag.poly_markets import CryptoMarketRegistry
    from strategies.crypto_lag.paper_executor import PaperExecutor
    from strategies.crypto_lag.live_executor import LiveExecutor
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

    # Default bankroll per variant — variants override via `bankroll_usdc`.
    default_bankroll_usdc = float(
        cfg.get("bankroll_usdc_default",
                config.get("bot", {}).get("demo_capital", 1000.0))
    )
    # On reset_on_start, wipe persisted variant state so we boot from defaults.
    reset_on_start = bool(config.get("bot", {}).get("reset_on_start", False))
    if reset_on_start and db is not None and hasattr(db, "reset_crypto_lag_variant_states"):
        try:
            db.reset_crypto_lag_variant_states()
            logger.info("crypto_lag: cleared persisted variant state (reset_on_start=true)")
        except Exception as exc:
            logger.warning(f"crypto_lag: reset_crypto_lag_variant_states failed: {exc}")

    # ─── per-variant stacks ────────────────────────────────────
    variants_cfg = _resolve_variants(cfg)
    if not variants_cfg:
        logger.warning("crypto_lag: every variant is disabled (enabled=false); skipping start")
        return None
    variants: dict[str, VariantHandle] = {}
    for vname, voverrides in variants_cfg.items():
        # Build the effective config for this variant: deep-merge global
        # crypto_lag with per-variant overrides.
        eff = _deep_merge(cfg, voverrides)
        # Risk needs the merged crypto_lag block at config["crypto_lag"].
        variant_config_for_risk = dict(config)
        variant_config_for_risk["crypto_lag"] = eff
        paper_cfg_v = eff.get("paper") or {}
        # Per-variant mode override. Default is the global crypto_lag mode.
        # When mode=LIVE, the variant uses LiveExecutor (real Polymarket CLOB)
        # instead of PaperExecutor. The first LIVE variant per session must
        # have credentials in env vars or start() will fail fast.
        variant_mode = str(eff.get("mode", global_mode)).upper()
        live_cfg = eff.get("live") or {}
        if variant_mode == "LIVE":
            logger.warning(
                f"crypto_lag [{vname}]: mode=LIVE — instantiating LiveExecutor. "
                "Real money on Polymarket CLOB."
            )
            executor = LiveExecutor(
                variant=vname,
                host=str(live_cfg.get("host", "https://clob.polymarket.com")),
                chain_id=int(live_cfg.get("chain_id", 137)),
                signature_type=int(live_cfg.get("signature_type", 1)),
                max_order_usdc=float(live_cfg.get("max_order_usdc", 5.0)),
                sanity_checks_enabled=bool(live_cfg.get("sanity_checks_enabled", True)),
                whitelist_symbols=live_cfg.get("whitelist_symbols"),
                max_concurrent_orders=int(live_cfg.get("max_concurrent_orders", 5)),
                halt_file_path=str(live_cfg.get("halt_file_path", "data/halt_live")),
                halt_loss_usdc=float(live_cfg.get("halt_loss_usdc", 50.0)),
                creds_cache_path=str(live_cfg.get("creds_cache_path", "data/clob_api_creds.json")),
            )
        else:
            executor = PaperExecutor(
                q_toxic=float(paper_cfg_v.get("q_toxic", 0.30)),
                adverse_haircut_pct=float(paper_cfg_v.get("adverse_haircut_pct", 0.015)),
                fee_rate=float(eff.get("fee_rate", FEE_RATE_CRYPTO)),
                maker_rebate_share=float(eff.get("maker_rebate_share", MAKER_REBATE_SHARE)),
                queue_position_enabled=bool(paper_cfg_v.get("queue_position_enabled", True)),
                # LIVE-realism knobs (default: realistic). Tunable per variant via
                # `paper.live_realistic_rebates / taker_race_lost_pct / q_toxic_extreme_scaling`.
                live_realistic_rebates=bool(paper_cfg_v.get("live_realistic_rebates", True)),
                taker_race_lost_pct=float(paper_cfg_v.get("taker_race_lost_pct", 0.25)),
                q_toxic_extreme_scaling=bool(paper_cfg_v.get("q_toxic_extreme_scaling", True)),
                depth_haircut_enabled=bool(paper_cfg_v.get("depth_haircut_enabled", True)),
                maker_race_lost_pct=float(paper_cfg_v.get("maker_race_lost_pct", 0.15)),
                queue_advance_credit_pct=float(paper_cfg_v.get("queue_advance_credit_pct", 0.50)),
                # Post-audit fidelity fixes (2026-05-09). Defaults match the
                # PaperExecutor constructor's so behavior is unchanged unless
                # the variant explicitly overrides under `paper:`.
                adverse_size_attenuation=float(paper_cfg_v.get("adverse_size_attenuation", 1.0)),
                min_fill_usdc=float(paper_cfg_v.get("min_fill_usdc", 0.50)),
                maker_race_lost_max=float(paper_cfg_v.get("maker_race_lost_max", 0.65)),
                q_toxic_extreme_cap=float(paper_cfg_v.get("q_toxic_extreme_cap", 0.70)),
                depth_extreme_multiplier=float(paper_cfg_v.get("depth_extreme_multiplier", 0.50)),
                depth_near_extreme_multiplier=float(paper_cfg_v.get("depth_near_extreme_multiplier", 0.75)),
            )

        # Per-variant bankroll: prefer the per-variant override, fall back to
        # the global default. If reset_on_start is false AND we have persisted
        # state, restore it to survive crashes/restarts.
        configured_bankroll = float(eff.get("bankroll_usdc", default_bankroll_usdc))
        initial_bankroll = configured_bankroll
        if not reset_on_start and db is not None and hasattr(db, "get_crypto_lag_variant_state"):
            try:
                persisted = db.get_crypto_lag_variant_state(vname)
                if persisted is not None:
                    initial_bankroll = float(persisted["bankroll_usdc"])
                    logger.info(
                        f"crypto_lag variant {vname!r}: restored bankroll "
                        f"${initial_bankroll:.2f} from persisted state"
                    )
            except Exception as exc:
                logger.debug(f"variant state restore ({vname}) failed: {exc}")

        risk = CryptoLagRisk(
            variant_config_for_risk,
            bankroll_usdc=initial_bankroll,
            variant_name=vname,
        )

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
            min_replace_interval_seconds=float(eff.get("min_replace_interval_seconds", 5.0)),
            gamma=float(eff.get("gamma", 0.10)),
            arrival_intensity_k=float(eff.get("arrival_intensity_k", 1.5)),
            inventory_skew_threshold=float(eff.get("inventory_skew_threshold", 0.7)),
            fee_rate=float(eff.get("fee_rate", FEE_RATE_CRYPTO)),
            maker_rebate_share=float(eff.get("maker_rebate_share", MAKER_REBATE_SHARE)),
            quote_mode=str(eff.get("quote_mode", "maker")),
            cross_threshold_ticks=float(eff.get("cross_threshold_ticks", 4.0)),
            placement_logger=placement_logger,
            min_order_usdc=float(eff.get("min_order_usdc", 5.0)),
            # F5 — extreme price guard. Per-variant overrides go under
            # `extreme_price_min / extreme_price_max / extreme_edge_multiplier`.
            extreme_price_min=float(eff.get("extreme_price_min", 0.10)),
            extreme_price_max=float(eff.get("extreme_price_max", 0.90)),
            extreme_edge_multiplier=float(eff.get("extreme_edge_multiplier", 4.0)),
        )
        cycle = CryptoLagCycle(
            config=config, feed=feed, registry=registry, executor=executor,
            engine=engine, risk=risk, db=db, notifier=notifier,
            deribit_iv=deribit_iv,
            variant=vname,
            variant_overrides=voverrides,
            mode=variant_mode,
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
