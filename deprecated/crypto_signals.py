"""
Crypto Signals — Adaptive Regime-Based Strategy.

Detects market regime via ADX and applies the correct strategy:
  RANGING       → Mean Reversion (RSI+BB) with strict hard requirements
  TRENDING_UP   → Trend Following (pullback to EMA21)
  TRENDING_DOWN → No longs (block all BUY signals)
  UNCERTAIN     → No trades (ADX in ambiguous zone)

Exit logic also adapted per regime.
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional

from strategies.market_regime import (
    detect_regime,
    compute_rsi_slope,
)

logger = logging.getLogger("polymarket_bot.crypto.signals")


# ─── Indicator helpers ────────────────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    return middle + std_dev * std, middle, middle - std_dev * std


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


# ─── Signal Engine ────────────────────────────────────────────────────────────

class CryptoSignalEngine:
    """
    Regime-adaptive signal engine.

    Selects strategy based on ADX:
      RANGING      → Mean Reversion with hard filters
      TRENDING_UP  → Trend Following (pullback entries)
      TRENDING_DOWN / UNCERTAIN → HOLD (no new longs)
    """

    def __init__(self, config: dict):
        strategy_cfg = config.get("crypto", {}).get("strategy", {})
        risk_cfg = config.get("crypto", {}).get("risk", {})

        self.rsi_period = strategy_cfg.get("rsi_period", 14)
        self.rsi_oversold = strategy_cfg.get("rsi_oversold", 28)
        self.rsi_overbought = strategy_cfg.get("rsi_overbought", 60)
        self.bb_period = strategy_cfg.get("bb_period", 20)
        self.bb_std = strategy_cfg.get("bb_std", 2.0)
        self.ema_fast = strategy_cfg.get("ema_fast", 9)
        self.ema_slow = strategy_cfg.get("ema_slow", 21)
        self.volume_multiplier = strategy_cfg.get("volume_multiplier", 1.5)
        self._risk_tp_pct = risk_cfg.get("take_profit_pct", 0.015)
        self._sell_signal_rsi_exit = risk_cfg.get("sell_signal_rsi_exit", 60)

        # Regime thresholds
        self.adx_period = strategy_cfg.get("adx_period", 14)
        self.adx_trending = strategy_cfg.get("adx_trending_threshold", 25.0)
        self.adx_ranging = strategy_cfg.get("adx_ranging_threshold", 20.0)
        self.rsi_slope_lookback = strategy_cfg.get("rsi_slope_lookback", 5)

    # ── Indicator computation ──────────────────────────────────────────────────

    def _compute_indicators(self, df: pd.DataFrame) -> Optional[dict]:
        try:
            close = df["close"]
            volume = df["volume"]

            rsi_series = compute_rsi(close, self.rsi_period)
            bb_upper, bb_middle, bb_lower = compute_bollinger_bands(close, self.bb_period, self.bb_std)
            ema_f = compute_ema(close, self.ema_fast)
            ema_s = compute_ema(close, self.ema_slow)

            current_price = float(close.iloc[-1])
            current_rsi = float(rsi_series.iloc[-1])
            current_bb_upper = float(bb_upper.iloc[-1])
            current_bb_middle = float(bb_middle.iloc[-1])
            current_bb_lower = float(bb_lower.iloc[-1])
            current_ema_f = float(ema_f.iloc[-1])
            current_ema_s = float(ema_s.iloc[-1])

            vol_avg = float(volume.rolling(20).mean().iloc[-1])
            current_vol = float(volume.iloc[-1])
            volume_ratio = current_vol / vol_avg if vol_avg > 0 else 1.0

            if current_ema_f > current_ema_s * 1.001:
                trend = "BULLISH"
            elif current_ema_f < current_ema_s * 0.999:
                trend = "BEARISH"
            else:
                trend = "NEUTRAL"

            prev_close = float(close.iloc[-2])
            prev2_close = float(close.iloc[-3])
            current_open = float(df["open"].iloc[-1])
            prev_open = float(df["open"].iloc[-2])

            high = df["high"]
            low = df["low"]
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs()
            ], axis=1).max(axis=1)
            atr = float(tr.rolling(14).mean().iloc[-1])
            atr_pct = atr / current_price if current_price > 0 else 0

            rsi_slope = compute_rsi_slope(rsi_series, self.rsi_slope_lookback)

            return {
                "price": current_price,
                "rsi": current_rsi,
                "rsi_slope": rsi_slope,
                "bb_upper": current_bb_upper,
                "bb_middle": current_bb_middle,
                "bb_lower": current_bb_lower,
                "ema_f": current_ema_f,
                "ema_s": current_ema_s,
                "volume_ratio": volume_ratio,
                "trend": trend,
                "prev_close": prev_close,
                "prev2_close": prev2_close,
                "prev_open": prev_open,
                "current_open": current_open,
                "atr_pct": atr_pct,
                "rsi_series": rsi_series,
            }
        except Exception as e:
            logger.error(f"_compute_indicators failed: {e}")
            return None

    # ── Mean Reversion (RANGING regime) ───────────────────────────────────────

    def _analyze_mean_reversion(self, ind: dict, rsi_15m: Optional[float]) -> tuple[str, float, list]:
        """
        Mean reversion entry for RANGING markets.
        Hard requirements: ALL must pass or return HOLD immediately.
        """
        price = ind["price"]
        rsi = ind["rsi"]
        rsi_slope = ind["rsi_slope"]
        volume_ratio = ind["volume_ratio"]
        trend = ind["trend"]
        bb_lower = ind["bb_lower"]
        bb_middle = ind["bb_middle"]
        bb_upper = ind["bb_upper"]
        prev_close = ind["prev_close"]
        prev2_close = ind["prev2_close"]
        current_open = ind["current_open"]
        atr_pct = ind["atr_pct"]

        reasons = []

        # ── Hard requirements (any failure → HOLD) ────────────────────────────
        if rsi >= self.rsi_oversold:
            return "HOLD", 0.0, [f"MR: RSI {rsi:.1f} >= {self.rsi_oversold} (not oversold enough)"]

        if volume_ratio < self.volume_multiplier:
            return "HOLD", 0.0, [f"MR: volume {volume_ratio:.2f}x < {self.volume_multiplier}x (mandatory)"]

        if trend == "BEARISH":
            return "HOLD", 0.0, [f"MR: BEARISH trend blocked (hard filter)"]

        if rsi_slope <= 0:
            return "HOLD", 0.0, [f"MR: RSI slope {rsi_slope:.2f} <= 0 (RSI still falling)"]

        if rsi_15m is not None and rsi_15m > 55:
            return "HOLD", 0.0, [f"MR: 15m RSI {rsi_15m:.1f} > 55 (overbought on higher TF)"]

        # ── All hard requirements passed ──────────────────────────────────────
        depth = self.rsi_oversold - rsi
        reasons.append(f"MR: RSI oversold ({rsi:.1f}, -{depth:.1f} below {self.rsi_oversold})")
        reasons.append(f"MR: volume {volume_ratio:.2f}x confirmed")
        reasons.append(f"MR: RSI slope +{rsi_slope:.2f} (rising)")

        # ── Confirmation scoring ──────────────────────────────────────────────
        score = 0.0

        # Price at/below BB lower
        bb_pen = (bb_lower - price) / bb_lower
        if bb_pen >= 0.0:
            score += 0.40 + min(bb_pen * 50, 0.15)
            reasons.append(f"MR: at/below BB lower (pen={bb_pen*100:.3f}%)")
        elif price <= bb_lower * 1.005:
            score += 0.25
            bb_pct = (price - bb_lower) / bb_lower * 100
            reasons.append(f"MR: near BB lower ({bb_pct:+.3f}%)")
        else:
            score += 0.05  # below BB middle zone
            reasons.append(f"MR: below BB mid")

        # Bullish candle
        if price > current_open:
            body_pct = (price - current_open) / current_open * 100
            score += 0.25
            reasons.append(f"MR: bullish candle (+{body_pct:.3f}%)")

        # Bounce pattern (prev down, now up)
        if price > prev_close and prev_close < prev2_close:
            score += 0.30
            reasons.append("MR: bounce detected")

        # EMA trend
        if trend == "BULLISH":
            score += 0.20
            reasons.append("MR: BULLISH trend")
        elif trend == "NEUTRAL":
            score += 0.10
            reasons.append("MR: NEUTRAL trend")

        # ATR supports TP
        if atr_pct >= self._risk_tp_pct * 1.2:
            score += 0.15
            reasons.append(f"MR: ATR supports TP ({atr_pct*100:.3f}% >= {self._risk_tp_pct*100:.2f}%*1.2)")

        # RSI extra depth bonus
        if depth > 5:
            score += min(depth * 0.01, 0.10)
            reasons.append(f"MR: deep oversold bonus (depth={depth:.1f})")

        threshold = 0.80  # raised from 0.75
        if score >= threshold:
            return "BUY", min(score, 1.0), reasons
        else:
            reasons.append(f"MR: score {score:.2f} < {threshold} threshold")
            return "HOLD", 0.0, reasons

    # ── Momentum Breakout (RANGING regime, mid-RSI with strong volume) ────────

    def _analyze_momentum_breakout(self, ind: dict) -> tuple[str, float, list]:
        """
        Catches breakout moves from neutral RSI territory (45-62) with strong volume.
        Complements mean reversion — fires when price breaks above BB middle with momentum.
        Example: XRP RSI 45→73 with 5x volume — mean reversion never triggers.
        """
        price = ind["price"]
        rsi = ind["rsi"]
        rsi_slope = ind["rsi_slope"]
        volume_ratio = ind["volume_ratio"]
        bb_middle = ind["bb_middle"]
        bb_upper = ind["bb_upper"]
        ema_f = ind["ema_f"]
        ema_s = ind["ema_s"]
        prev_close = ind["prev_close"]
        current_open = ind["current_open"]

        # ── Hard requirements ─────────────────────────────────────────────────
        if not (45 <= rsi <= 63):
            return "HOLD", 0.0, [f"MB: RSI {rsi:.1f} outside breakout zone [45-63]"]

        if rsi_slope < 6:
            return "HOLD", 0.0, [f"MB: RSI slope {rsi_slope:.1f} < 6 (not fast enough)"]

        if volume_ratio < 1.5:
            return "HOLD", 0.0, [f"MB: volume {volume_ratio:.2f}x < 1.5x required"]

        if price < bb_middle:
            return "HOLD", 0.0, [f"MB: price below BB middle (no breakout yet)"]

        # ── All hard requirements passed ──────────────────────────────────────
        reasons = [
            f"MB: RSI breakout zone ({rsi:.1f})",
            f"MB: RSI slope +{rsi_slope:.1f} (fast momentum)",
            f"MB: volume {volume_ratio:.2f}x (strong conviction)",
            f"MB: price above BB middle",
        ]

        score = 0.0

        # Bullish candle (current green)
        if price > current_open:
            score += 0.30
            reasons.append("MB: bullish candle")

        # Continuation (prev candle also up)
        if price > prev_close:
            score += 0.20
            reasons.append("MB: price above prev close")

        # EMA aligned bullish
        if ema_f > ema_s:
            score += 0.20
            reasons.append("MB: EMA bullish alignment")

        # Extra volume strength
        if volume_ratio >= 4.0:
            score += 0.15
            reasons.append(f"MB: very strong volume {volume_ratio:.2f}x")

        # Price approaching BB upper (momentum still has room)
        bb_range = bb_upper - bb_middle
        pct_into_range = (price - bb_middle) / bb_range if bb_range > 0 else 0
        if pct_into_range < 0.6:  # Still has room to BB upper
            score += 0.15
            reasons.append(f"MB: room to BB upper ({pct_into_range*100:.0f}% of range used)")

        threshold = 0.75
        if score >= threshold:
            return "BUY", min(score, 1.0), reasons
        else:
            reasons.append(f"MB: score {score:.2f} < {threshold}")
            return "HOLD", 0.0, reasons

    # ── Trend Following (TRENDING_UP or UNCERTAIN regime) ─────────────────────

    def _analyze_trend_following(self, ind: dict, regime: dict) -> tuple[str, float, list]:
        """
        Trend-following entry: buy pullbacks to EMA21 in uptrends.
        """
        price = ind["price"]
        rsi = ind["rsi"]
        rsi_slope = ind["rsi_slope"]
        volume_ratio = ind["volume_ratio"]
        ema_f = ind["ema_f"]
        ema_s = ind["ema_s"]
        prev_close = ind["prev_close"]
        prev_open = ind["prev_open"]
        current_open = ind["current_open"]
        adx = regime["adx"]

        reasons = []

        # ── Hard requirements ─────────────────────────────────────────────────

        # Price must be near EMA21 (pulling back to dynamic support)
        ema_distance_pct = abs(price - ema_s) / ema_s
        if ema_distance_pct > 0.008:
            return "HOLD", 0.0, [f"TF: price {ema_distance_pct*100:.3f}% from EMA21 (need <= 0.8%)"]

        # RSI in momentum zone (not overbought, not crashed)
        if not (42 <= rsi <= 62):
            return "HOLD", 0.0, [f"TF: RSI {rsi:.1f} outside momentum zone [42-62]"]

        # RSI slope must be neutral or rising
        if rsi_slope < -2:
            return "HOLD", 0.0, [f"TF: RSI slope {rsi_slope:.2f} falling too fast"]

        # Volume confirmation
        if volume_ratio < 1.3:
            return "HOLD", 0.0, [f"TF: volume {volume_ratio:.2f}x < 1.3x required"]

        # ── All hard requirements passed ──────────────────────────────────────
        reasons.append(f"TF: ADX={adx:.1f} confirmed uptrend")
        reasons.append(f"TF: pullback to EMA21 ({ema_distance_pct*100:.3f}% away)")
        reasons.append(f"TF: RSI momentum zone ({rsi:.1f})")
        reasons.append(f"TF: volume {volume_ratio:.2f}x")

        # ── Confirmation scoring ──────────────────────────────────────────────
        score = 0.0

        # Previous candle was bearish (pullback), current is bullish (recovery)
        prev_bearish = prev_close < prev_open
        curr_bullish = price > current_open
        if prev_bearish and curr_bullish:
            score += 0.35
            reasons.append("TF: pullback candle confirmed (prev red, now green)")

        # Price above EMA fast (momentum maintained)
        if price >= ema_f:
            score += 0.25
            reasons.append(f"TF: above EMA9 ({ema_f:.2f})")

        # Strong volume
        if volume_ratio >= 2.0:
            score += 0.20
            reasons.append(f"TF: strong volume {volume_ratio:.2f}x")

        # Strong ADX
        if adx >= 30:
            score += 0.20
            reasons.append(f"TF: strong ADX {adx:.1f}")

        # RSI rising slope
        if rsi_slope > 1:
            score += 0.10
            reasons.append(f"TF: RSI slope +{rsi_slope:.2f}")

        threshold = 0.75
        if score >= threshold:
            return "BUY", min(score, 1.0), reasons
        else:
            reasons.append(f"TF: score {score:.2f} < {threshold} threshold")
            return "HOLD", 0.0, reasons

    # ── Exit analysis ──────────────────────────────────────────────────────────

    def analyze_exit(
        self,
        df: pd.DataFrame,
        symbol: str,
        entry_price: float,
        age_cycles: int,
        entry_regime: str = "RANGING",
    ) -> dict:
        """
        Analyze whether an open position should be closed on technical signal.
        entry_regime: the regime active when position was opened.
        """
        if df is None or len(df) < max(self.bb_period, self.rsi_period) + 5:
            return {"should_exit": False, "reason": "insufficient_data", "urgency": "LOW"}

        try:
            ind = self._compute_indicators(df)
            if ind is None:
                return {"should_exit": False, "reason": "indicator_error", "urgency": "LOW"}

            price = ind["price"]
            rsi = ind["rsi"]
            bb_middle = ind["bb_middle"]
            bb_upper = ind["bb_upper"]
            trend = ind["trend"]
            pnl_pct = (price - entry_price) / entry_price * 100

            # Current regime
            regime_info = detect_regime(
                df,
                adx_trending_threshold=self.adx_trending,
                adx_ranging_threshold=self.adx_ranging,
                adx_period=self.adx_period,
            )
            current_regime = regime_info["regime"]

            reasons = []
            urgency = "LOW"

            # EXIT: RSI overbought
            if rsi > self.rsi_overbought:
                reasons.append(f"RSI overbought ({rsi:.1f} > {self.rsi_overbought})")
                urgency = "HIGH"

            # EXIT: RSI moderately high + above BB middle
            elif rsi > self._sell_signal_rsi_exit and price >= bb_middle:
                reasons.append(f"RSI={rsi:.1f} > {self._sell_signal_rsi_exit} above BB_mid — take profit")
                urgency = "MEDIUM"

            # EXIT: Near BB upper with profit
            if price >= bb_upper * 0.998 and pnl_pct > 0.3:
                reasons.append(f"Near BB upper, booking profit ({pnl_pct:.2f}%)")
                urgency = "HIGH"

            # EXIT: Mean reversion position but regime turned trending (thesis invalidated)
            if entry_regime == "RANGING" and current_regime in ("TRENDING_DOWN", "UNCERTAIN"):
                reasons.append(f"Regime changed from RANGING to {current_regime} — exit MR position")
                urgency = "HIGH"

            # EXIT: Bearish trend + old + losing
            if trend == "BEARISH" and age_cycles >= 6 and pnl_pct < 0:
                reasons.append(f"BEARISH trend, {age_cycles} cycles, P&L={pnl_pct:.2f}%")
                if urgency == "LOW":
                    urgency = "MEDIUM"

            # EXIT: Trend following position but trend reversed
            if entry_regime == "TRENDING_UP" and current_regime == "TRENDING_DOWN":
                reasons.append("Trend reversed: TRENDING_UP → TRENDING_DOWN")
                urgency = "HIGH"

            should_exit = len(reasons) > 0
            return {
                "should_exit": should_exit,
                "reason": " | ".join(reasons) if reasons else "hold",
                "urgency": urgency,
                "rsi": round(rsi, 2),
                "price": price,
                "pnl_pct": round(pnl_pct, 3),
                "regime": current_regime,
            }

        except Exception as e:
            logger.error(f"analyze_exit failed for {symbol}: {e}")
            return {"should_exit": False, "reason": f"error: {e}", "urgency": "LOW"}

    # ── Main entry point ───────────────────────────────────────────────────────

    def analyze(self, df: pd.DataFrame, symbol: str, rsi_15m: Optional[float] = None) -> dict:
        """
        Analyze market and return signal dict.

        Returns:
            {
              "symbol": str,
              "signal": "BUY" | "SELL" | "HOLD",
              "price": float,
              "rsi": float,
              "rsi_slope": float,
              "bb_upper/middle/lower": float,
              "ema_fast/slow": float,
              "volume_ratio": float,
              "trend": str,
              "regime": str,
              "adx": float,
              "signal_strength": float,
              "reasons": list[str],
              "atr_pct": float,
            }
        """
        if df is None or len(df) < max(self.bb_period, self.rsi_period, self.adx_period * 2) + 5:
            return self._no_signal(symbol, "Insufficient candle data")

        try:
            ind = self._compute_indicators(df)
            if ind is None:
                return self._no_signal(symbol, "Indicator computation failed")

            # Detect market regime
            regime_info = detect_regime(
                df,
                adx_trending_threshold=self.adx_trending,
                adx_ranging_threshold=self.adx_ranging,
                adx_period=self.adx_period,
            )
            regime = regime_info["regime"]
            adx = regime_info["adx"]

            price = ind["price"]
            rsi = ind["rsi"]

            # ── Route to correct strategy ──────────────────────────────────────
            if regime == "RANGING":
                # Try mean reversion first, then momentum breakout if no signal
                signal, strength, reasons = self._analyze_mean_reversion(ind, rsi_15m)
                if signal == "HOLD":
                    signal, strength, reasons = self._analyze_momentum_breakout(ind)

            elif regime in ("TRENDING_UP", "UNCERTAIN"):
                # Trend following in both confirmed uptrend and ambiguous zone
                # UNCERTAIN uses stricter internal scoring (same function, same threshold)
                signal, strength, reasons = self._analyze_trend_following(ind, regime_info)

            elif regime == "TRENDING_DOWN":
                signal = "HOLD"
                strength = 0.0
                reasons = [f"TRENDING_DOWN (ADX={adx:.1f}) — no longs in downtrend"]

            else:
                signal = "HOLD"
                strength = 0.0
                reasons = [f"Unknown regime — HOLD"]

            # ── SELL signal (independent of regime — for open positions) ───────
            sell_score = 0.0
            sell_reasons = []
            bb_middle = ind["bb_middle"]

            if rsi > self.rsi_overbought:
                sell_reasons.append(f"RSI overbought ({rsi:.1f} > {self.rsi_overbought})")
                sell_score += 0.6

            if price >= bb_middle:
                sell_reasons.append(f"Price at/above BB middle")
                sell_score += 0.4

            if sell_score >= 0.5 and signal != "BUY":
                signal = "SELL"
                strength = min(sell_score, 1.0)
                reasons = sell_reasons

            # ── Build result ───────────────────────────────────────────────────
            result = {
                "symbol": symbol,
                "signal": signal,
                "price": price,
                "rsi": round(rsi, 2),
                "rsi_slope": round(ind["rsi_slope"], 2),
                "bb_upper": round(ind["bb_upper"], 4),
                "bb_middle": round(ind["bb_middle"], 4),
                "bb_lower": round(ind["bb_lower"], 4),
                "ema_fast": round(ind["ema_f"], 4),
                "ema_slow": round(ind["ema_s"], 4),
                "volume_ratio": round(ind["volume_ratio"], 2),
                "trend": ind["trend"],
                "regime": regime,
                "adx": round(adx, 2),
                "signal_strength": round(strength, 3),
                "reasons": reasons,
                "atr_pct": round(ind["atr_pct"], 6),
            }

            emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(signal, "")
            logger.info(
                f"{emoji} {symbol} {signal} [{regime}] | "
                f"Price:{price:.4f} RSI:{rsi:.1f}(slope:{ind['rsi_slope']:+.1f}) "
                f"ADX:{adx:.1f} Vol:{ind['volume_ratio']:.2f}x"
            )

            return result

        except Exception as e:
            logger.error(f"Signal analysis failed for {symbol}: {e}")
            return self._no_signal(symbol, str(e))

    def _no_signal(self, symbol: str, reason: str) -> dict:
        return {
            "symbol": symbol,
            "signal": "HOLD",
            "price": None,
            "rsi": None,
            "rsi_slope": None,
            "bb_upper": None,
            "bb_middle": None,
            "bb_lower": None,
            "ema_fast": None,
            "ema_slow": None,
            "volume_ratio": None,
            "trend": "NEUTRAL",
            "regime": "UNCERTAIN",
            "adx": None,
            "signal_strength": 0.0,
            "reasons": [f"No signal: {reason}"],
            "atr_pct": None,
        }
