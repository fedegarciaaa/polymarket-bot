"""Background runner for the crypto-lag MAKER bot.

Spawns a daemon thread that runs the asyncio cycle. Designed to coexist with
the existing weather scheduler in `main.py` — they share the `Database` and
the bankroll, but have independent ticking cadence.

Usage from `main.py`:

    from crypto_lag_runner import start_crypto_lag, stop_crypto_lag

    handle = start_crypto_lag(config, db, logger)
    # … main loop runs …
    stop_crypto_lag(handle)

`start_crypto_lag` is a no-op if `config.crypto_lag.enabled` is false.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
from dataclasses import dataclass
from typing import Optional


@dataclass
class CryptoLagHandle:
    thread: threading.Thread
    loop: asyncio.AbstractEventLoop
    cycle: object             # CryptoLagCycle
    feed: object              # BinanceFeed
    registry: object          # CryptoMarketRegistry
    executor: object          # PaperExecutor
    engine: object            # MakerOrderEngine


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
    from strategies.crypto_lag.order_engine import MakerOrderEngine
    from strategies.crypto_lag.risk import CryptoLagRisk
    from strategies.crypto_lag.cycle import CryptoLagCycle

    symbols = [s["binance"] for s in cfg.get("symbols", []) if s.get("binance")]
    if not symbols:
        logger.warning("crypto_lag: no symbols configured; skipping start")
        return None

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
    registry = CryptoMarketRegistry(
        symbols=symbols,
        poll_seconds=float(cfg.get("market_poll_seconds", 30.0)),
        prefer_horizon_minutes=int(cfg.get("prefer_horizon_minutes", 15)),
    )
    paper_cfg = cfg.get("paper") or {}
    executor = PaperExecutor(
        q_toxic=float(paper_cfg.get("q_toxic", 0.30)),
        adverse_haircut_pct=float(paper_cfg.get("adverse_haircut_pct", 0.015)),
    )

    def _bankroll() -> float:
        try:
            if hasattr(db, "get_bankroll"):
                return float(db.get_bankroll() or 0.0)
        except Exception:
            pass
        return float(config.get("bot", {}).get("demo_capital", 1000.0))

    risk = CryptoLagRisk(config, get_bankroll_usdc=_bankroll)
    engine = MakerOrderEngine(
        executor=executor,
        risk=risk,
        edge_threshold_cents=float(cfg.get("edge_threshold_cents", 2.0)),
        replace_threshold_cents=float(cfg.get("replace_threshold_cents", 1.0)),
        max_order_age_seconds=float(cfg.get("max_order_age_seconds", 30.0)),
    )
    cycle = CryptoLagCycle(
        config=config, feed=feed, registry=registry, executor=executor,
        engine=engine, risk=risk, db=db, notifier=notifier,
    )

    loop = asyncio.new_event_loop()

    async def _bootstrap_and_run():
        # Fetch 24h of 1-minute klines from Binance REST before starting the WS
        # so the probability model has real realized vol from tick 1.
        await feed.bootstrap_historical_vol()
        await feed.start()
        await registry.start()
        await executor.start()
        try:
            await cycle.run_forever()
        finally:
            cycle.stop()
            await engine.cancel_all()
            await registry.stop()
            await feed.stop()
            await executor.stop()

    def _thread_main():
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            # Re-create the loop AFTER setting the policy so we get a SelectorEventLoop
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
        f"capital_pct={cfg.get('capital_pct', 0.30):.0%})"
    )
    return CryptoLagHandle(
        thread=th, loop=loop, cycle=cycle, feed=feed,
        registry=registry, executor=executor, engine=engine,
    )


def stop_crypto_lag(handle: Optional[CryptoLagHandle], timeout: float = 10.0) -> None:
    if handle is None:
        return
    handle.cycle.stop()  # type: ignore[attr-defined]
    handle.thread.join(timeout=timeout)
