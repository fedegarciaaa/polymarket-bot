"""
Crypto Trader - Binance SPOT trading client with DEMO mode support.
Uses python-binance for market data. Implements RSI + Bollinger Bands strategy.

FIXES v2:
- Cooldown real tras pérdida por SL (N ciclos configurables)
- Cierre preciso al precio exacto de TP/SL sin drift de get_current_price()
- Función is_on_cooldown() operativa
"""

import os
import time
import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

logger = logging.getLogger("polymarket_bot.crypto")

try:
    from binance.client import Client as BinanceClient
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    logger.warning("python-binance not installed. Crypto module will run in DEMO-only mode.")


class CryptoTrader:
    def __init__(self, config: dict):
        self.config = config
        crypto_cfg = config.get("crypto", {})
        self.mode = crypto_cfg.get("mode", "DEMO").upper()
        self.pairs = crypto_cfg.get("pairs", ["BTCUSDT", "ETHUSDT"])
        self.timeframe = crypto_cfg.get("timeframe", "15m")
        self.candles_lookback = crypto_cfg.get("candles_lookback", 100)
        self.demo_capital = crypto_cfg.get("demo_capital", 500.0)

        risk_cfg = crypto_cfg.get("risk", {})
        self.fee_rate = risk_cfg.get("fee_rate", 0.001)
        self.stop_loss_pct = risk_cfg.get("stop_loss_pct", 0.008)
        self.take_profit_pct = risk_cfg.get("take_profit_pct", 0.015)
        self.max_position_pct = risk_cfg.get("max_position_pct", 0.10)
        self.max_concurrent_trades = risk_cfg.get("max_concurrent_trades", 4)
        self.max_trade_time_seconds = risk_cfg.get("max_trade_time_seconds", 2700)
        self.trailing_activate_pct = risk_cfg.get("trailing_activate_pct", 0.007)
        self.trailing_distance_pct = risk_cfg.get("trailing_distance_pct", 0.004)
        # Número de ciclos de cooldown después de pérdida por SL
        self.cooldown_cycles_after_loss = risk_cfg.get("cooldown_cycles_after_loss", 3)

        # Binance client (None in DEMO mode or if keys not set)
        self.client = None
        if self.mode == "LIVE" and BINANCE_AVAILABLE:
            api_key = os.getenv("BINANCE_API_KEY", "")
            api_secret = os.getenv("BINANCE_API_SECRET", "")
            if api_key and api_secret:
                try:
                    self.client = BinanceClient(api_key, api_secret)
                    logger.info("Binance client initialized (LIVE mode)")
                except Exception as e:
                    logger.error(f"Failed to initialize Binance client: {e}")
                    self.client = None
            else:
                logger.warning("BINANCE_API_KEY/SECRET not set. Crypto running in DEMO.")
                self.mode = "DEMO"
        else:
            logger.info(f"CryptoTrader initialized in {self.mode} mode")

        # Partial exit / risk management settings
        self._partial_exit_trigger_pct = risk_cfg.get("partial_exit_trigger_pct", 0.008)
        self._partial_exit_fraction = risk_cfg.get("partial_exit_fraction", 0.5)

        # DEMO state: simulated portfolio
        self._demo_balance_usdt = self.demo_capital
        self._demo_positions: dict = {}  # symbol -> {qty, entry_price, stop_loss, take_profit, usdt_invested, timestamp}
        self._demo_trade_counter = 0
        # Cooldown: symbol -> remaining cycles before re-entry allowed
        self._cooldown_remaining: dict = {}  # symbol -> int (cycles left)
        self._best_price: dict = {}  # symbol -> best price seen since entry (for trailing stop)
        # Track age of open positions in cycles (for signal-based exit)
        self._position_age_cycles: dict = {}  # symbol -> int
        # Partial exit tracking: symbol -> bool (True = already took partial profit)
        self._partial_exit_done: dict = {}  # symbol -> bool

    # ──────────────────────────────────────────────
    # Market Data
    # ──────────────────────────────────────────────

    def get_klines(self, symbol: str, interval: str = None, limit: int = None) -> Optional[pd.DataFrame]:
        """Fetch OHLCV candles and return as DataFrame."""
        interval = interval or self.timeframe
        limit = limit or self.candles_lookback

        try:
            if self.client:
                raw = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
                df = pd.DataFrame(raw, columns=[
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "quote_asset_volume", "num_trades",
                    "taker_buy_base", "taker_buy_quote", "ignore"
                ])
                df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = df[col].astype(float)
                df = df.set_index("open_time")
                return df[["open", "high", "low", "close", "volume"]]
            else:
                # DEMO: fetch via public REST without auth
                import urllib.request
                import json
                url = (
                    f"https://api.binance.com/api/v3/klines"
                    f"?symbol={symbol}&interval={interval}&limit={limit}"
                )
                with urllib.request.urlopen(url, timeout=10) as resp:
                    raw = json.loads(resp.read())
                df = pd.DataFrame(raw, columns=[
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "quote_asset_volume", "num_trades",
                    "taker_buy_base", "taker_buy_quote", "ignore"
                ])
                df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = df[col].astype(float)
                df = df.set_index("open_time")
                return df[["open", "high", "low", "close", "volume"]]
        except Exception as e:
            logger.error(f"Failed to fetch klines for {symbol}: {e}")
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a symbol."""
        try:
            if self.client:
                ticker = self.client.get_symbol_ticker(symbol=symbol)
                return float(ticker["price"])
            else:
                import urllib.request
                import json
                url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
                with urllib.request.urlopen(url, timeout=5) as resp:
                    data = json.loads(resp.read())
                return float(data["price"])
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return None

    # ──────────────────────────────────────────────
    # Order Execution
    # ──────────────────────────────────────────────

    def get_available_capital(self) -> float:
        """Return available USDT balance."""
        if self.mode == "DEMO":
            return self._demo_balance_usdt
        try:
            balance = self.client.get_asset_balance(asset="USDT")
            return float(balance["free"])
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return 0.0

    def get_open_positions(self) -> dict:
        """Return currently open positions."""
        if self.mode == "DEMO":
            return dict(self._demo_positions)
        return {}

    def tick_cooldowns(self):
        """
        Decrementa el contador de cooldown de todos los pares.
        Debe llamarse UNA vez por ciclo del manager (antes de analizar señales).
        """
        expired = []
        for symbol, remaining in self._cooldown_remaining.items():
            if remaining <= 1:
                expired.append(symbol)
            else:
                self._cooldown_remaining[symbol] = remaining - 1
        for symbol in expired:
            del self._cooldown_remaining[symbol]
            logger.info(f"Cooldown expired for {symbol} — re-entry allowed")

    def tick_position_ages(self):
        """Incrementa la edad en ciclos de cada posición abierta."""
        for symbol in list(self._demo_positions.keys()):
            self._position_age_cycles[symbol] = self._position_age_cycles.get(symbol, 0) + 1

    def get_position_age_cycles(self, symbol: str) -> int:
        """Devuelve la edad en ciclos de la posición abierta."""
        return self._position_age_cycles.get(symbol, 0)

    def place_buy_order(
        self,
        symbol: str,
        usdt_amount: float,
        current_price: float,
        atr_pct: float = None,
    ) -> Optional[dict]:
        """Buy symbol with usdt_amount. Returns order dict or None.

        En DEMO restamos la comisión Binance taker de la cantidad efectiva que compra
        moneda. usdt_invested queda = usdt_amount (coste total descontado del balance).

        atr_pct: if provided, use dynamic TP = 2*ATR (clamped to [1%, 3%]).
        """
        fee_buy = usdt_amount * self.fee_rate
        effective_usdt = usdt_amount - fee_buy
        quantity = effective_usdt / current_price
        stop_loss_price = round(current_price * (1 - self.stop_loss_pct), 8)

        # Dynamic TP based on ATR: 2×ATR, clamped to [1%, 3%]
        if atr_pct and atr_pct > 0:
            dynamic_tp = max(0.010, min(0.030, atr_pct * 2.0))
        else:
            dynamic_tp = self.take_profit_pct
        take_profit_price = round(current_price * (1 + dynamic_tp), 8)

        if self.mode == "DEMO":
            self._demo_trade_counter += 1
            self._demo_balance_usdt -= usdt_amount
            self._demo_positions[symbol] = {
                "order_id": f"DEMO-{self._demo_trade_counter}",
                "symbol": symbol,
                "quantity": round(quantity, 8),
                "entry_price": current_price,
                "stop_loss": stop_loss_price,
                "take_profit": take_profit_price,
                "usdt_invested": usdt_amount,
                "fee_paid_buy": round(fee_buy, 6),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "entry_ts": time.time(),
                "status": "OPEN",
            }
            self._best_price[symbol] = current_price
            self._position_age_cycles[symbol] = 0
            self._partial_exit_done[symbol] = False  # Reset partial exit state
            logger.info(
                f"[DEMO] BUY {symbol}: ${usdt_amount:.2f} (fee ${fee_buy:.3f}) @ {current_price:.2f} "
                f"| SL: {stop_loss_price:.2f} | TP: {take_profit_price:.2f} (tp_pct={dynamic_tp*100:.2f}%)"
            )
            return self._demo_positions[symbol]

        # LIVE mode
        try:
            order = self.client.order_market_buy(
                symbol=symbol,
                quoteOrderQty=usdt_amount,
            )
            filled_price = float(order.get("fills", [{}])[0].get("price", current_price))
            logger.info(f"[LIVE] BUY {symbol}: ${usdt_amount:.2f} @ {filled_price:.2f}")
            return order
        except Exception as e:
            logger.error(f"BUY order failed for {symbol}: {e}")
            return None

    def place_sell_order(self, symbol: str, quantity: float, reason: str = "signal",
                         forced_price: float = None) -> Optional[dict]:
        """
        Sell all of symbol position.
        forced_price: si se especifica, usa ese precio exacto para el P&L (TP/SL hit).
        """
        if self.mode == "DEMO":
            pos = self._demo_positions.pop(symbol, None)
            if pos is None:
                return None

            # Usar precio forzado (exacto de TP/SL) o precio de mercado actual
            if forced_price is not None:
                current_price = forced_price
            else:
                current_price = self.get_current_price(symbol) or pos["entry_price"]

            gross = quantity * current_price
            fee_sell = gross * self.fee_rate
            proceeds = gross - fee_sell
            self._demo_balance_usdt += proceeds
            pnl = proceeds - pos["usdt_invested"]
            pnl_pct = (pnl / pos["usdt_invested"]) * 100
            logger.info(
                f"[DEMO] SELL {symbol}: {quantity:.6f} @ {current_price:.2f} "
                f"(fee ${fee_sell:.3f}) | P&L: ${pnl:+.3f} ({pnl_pct:+.3f}%) | Reason: {reason}"
            )
            # Limpiar estado del par
            self._best_price.pop(symbol, None)
            self._position_age_cycles.pop(symbol, None)
            self._partial_exit_done.pop(symbol, None)
            return {"symbol": symbol, "price": current_price, "pnl": pnl, "reason": reason,
                    "fee_sell": round(fee_sell, 6), "fee_buy": pos.get("fee_paid_buy", 0.0)}

        # LIVE mode
        try:
            order = self.client.order_market_sell(symbol=symbol, quantity=quantity)
            logger.info(f"[LIVE] SELL {symbol}: {quantity} | Reason: {reason}")
            return order
        except Exception as e:
            logger.error(f"SELL order failed for {symbol}: {e}")
            return None

    def check_and_close_positions(self) -> list:
        """
        Comprueba posiciones abiertas para SL/TP/time expiry/trailing.
        
        FIX: Usa el precio exacto del TP/SL calculado en la entrada (no get_current_price())
        cuando el precio de mercado lo cruza, evitando P&L negativo en cierres por TP.
        
        Returns list of closed position dicts.
        """
        closed = []
        positions = self.get_open_positions()

        for symbol, pos in positions.items():
            current_price = self.get_current_price(symbol)
            if current_price is None:
                continue

            reason = None
            forced_price = None  # Precio exacto para el cálculo de P&L
            entry_price = pos["entry_price"]
            pnl_pct = (current_price - entry_price) / entry_price

            # Update best price seen (for trailing stop)
            if symbol not in self._best_price or current_price > self._best_price.get(symbol, 0):
                self._best_price[symbol] = current_price

            # 1. Take Profit hit — usar precio exacto del TP para evitar P&L negativo
            if current_price >= pos["take_profit"]:
                reason = "take_profit"
                forced_price = pos["take_profit"]  # Precio exacto del nivel TP

            # 2. Stop Loss hit — usar precio exacto del SL + activar cooldown
            elif current_price <= pos["stop_loss"]:
                reason = "stop_loss"
                forced_price = pos["stop_loss"]  # Precio exacto del nivel SL
                # Activar cooldown para evitar re-entrada inmediata
                self._cooldown_remaining[symbol] = self.cooldown_cycles_after_loss
                logger.info(f"SL hit on {symbol} — cooldown activado: {self.cooldown_cycles_after_loss} ciclos")

            else:
                entry_ts = pos.get("entry_ts")
                if entry_ts is None:
                    try:
                        ts_str = pos.get("timestamp", "").replace("+00:00", "").replace("Z", "")
                        entry_ts = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc).timestamp()
                    except Exception:
                        entry_ts = time.time()

                age_seconds = time.time() - entry_ts

                # 3. Partial exit: close fraction at +partial_exit_trigger_pct, move SL to breakeven
                if not self._partial_exit_done.get(symbol, False):
                    if pnl_pct >= self._partial_exit_trigger_pct:
                        fraction = self._partial_exit_fraction
                        partial_qty = pos["quantity"] * fraction
                        partial_usdt = partial_qty * current_price
                        fee_partial = partial_usdt * self.fee_rate
                        net_partial = partial_usdt - fee_partial
                        pnl_partial = net_partial - pos["usdt_invested"] * fraction
                        self._demo_balance_usdt += net_partial
                        # Update remaining position
                        pos["quantity"] = round(pos["quantity"] * (1 - fraction), 8)
                        pos["usdt_invested"] = pos["usdt_invested"] * (1 - fraction)
                        # Move SL to breakeven + fees
                        pos["stop_loss"] = round(entry_price * (1 + self.fee_rate * 2), 8)
                        self._partial_exit_done[symbol] = True
                        logger.info(
                            f"[DEMO] PARTIAL EXIT {symbol}: {fraction*100:.0f}% @ {current_price:.4f} "
                            f"| Partial P&L: ${pnl_partial:+.3f} | SL moved to breakeven"
                        )

                # 4. Trailing stop: activa tras +trailing_activate_pct, sigue a trailing_distance_pct
                best = self._best_price.get(symbol, entry_price)
                best_pnl = (best - entry_price) / entry_price
                if best_pnl >= self.trailing_activate_pct:
                    drawdown_from_best = (best - current_price) / best
                    if drawdown_from_best >= self.trailing_distance_pct:
                        reason = f"trailing_stop(peak={best_pnl*100:+.2f}%,now={pnl_pct*100:+.2f}%)"
                        logger.info(
                            f"TRAILING STOP {symbol}: peaked at {best_pnl*100:+.2f}%, "
                            f"now {pnl_pct*100:+.2f}%"
                        )

                # 5. Hard cap: despues de max_trade_time_seconds cerramos sea cual sea el P&L
                if reason is None and age_seconds >= self.max_trade_time_seconds:
                    reason = f"max_hold({age_seconds:.0f}s,pnl={pnl_pct*100:+.2f}%)"
                    logger.info(
                        f"MAX HOLD {symbol}: {age_seconds:.0f}s | P&L {pnl_pct*100:+.2f}%"
                    )

            if reason:
                result = self.place_sell_order(symbol, pos["quantity"], reason=reason,
                                               forced_price=forced_price)
                if result:
                    result["entry_price"] = pos["entry_price"]
                    result["usdt_invested"] = pos["usdt_invested"]
                    result["forced_price"] = forced_price
                    closed.append(result)

        return closed

    def close_position_by_signal(self, symbol: str, reason: str = "signal_exit") -> Optional[dict]:
        """
        Cierra una posición abierta por señal técnica (SELL signal del motor).
        Usa precio de mercado actual (no hay nivel exacto predefinido).
        """
        pos = self._demo_positions.get(symbol)
        if pos is None:
            return None

        result = self.place_sell_order(symbol, pos["quantity"], reason=reason)
        if result:
            result["entry_price"] = pos["entry_price"]
            result["usdt_invested"] = pos["usdt_invested"]
        self._partial_exit_done.pop(symbol, None)
        return result

    def is_on_cooldown(self, symbol: str) -> bool:
        """
        Devuelve True si el par está en período de cooldown tras pérdida.
        El cooldown se decrementa cada ciclo via tick_cooldowns().
        """
        remaining = self._cooldown_remaining.get(symbol, 0)
        if remaining > 0:
            logger.debug(f"{symbol} en cooldown: {remaining} ciclos restantes")
            return True
        return False
