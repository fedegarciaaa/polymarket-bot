"""
Market Regime Detection — ADX + RSI Slope.

Detects whether the market is:
  TRENDING_UP   — ADX > 25 and +DI > -DI
  TRENDING_DOWN — ADX > 25 and -DI > +DI
  RANGING       — ADX < 20 (mean-reversion friendly)
  UNCERTAIN     — ADX 20-25 (avoid trading)

Used by CryptoSignalEngine to select the right strategy mode.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("polymarket_bot.crypto.regime")


def compute_adx(df: pd.DataFrame, period: int = 14) -> tuple[float, float, float, str]:
    """
    Compute ADX, +DI, -DI from OHLCV DataFrame.

    Returns:
        (adx, plus_di, minus_di, direction)
        direction: 'UP' if +DI > -DI, else 'DOWN'
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # True Range
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    # Directional Movement
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm_series = pd.Series(plus_dm, index=df.index)
    minus_dm_series = pd.Series(minus_dm, index=df.index)

    # Wilder smoothing (EWM with com = period - 1)
    atr_smooth = tr.ewm(com=period - 1, min_periods=period).mean()
    plus_dm_smooth = plus_dm_series.ewm(com=period - 1, min_periods=period).mean()
    minus_dm_smooth = minus_dm_series.ewm(com=period - 1, min_periods=period).mean()

    plus_di = 100 * (plus_dm_smooth / atr_smooth.replace(0, np.nan))
    minus_di = 100 * (minus_dm_smooth / atr_smooth.replace(0, np.nan))

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(com=period - 1, min_periods=period).mean()

    adx_val = float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 0.0
    plus_di_val = float(plus_di.iloc[-1]) if not np.isnan(plus_di.iloc[-1]) else 0.0
    minus_di_val = float(minus_di.iloc[-1]) if not np.isnan(minus_di.iloc[-1]) else 0.0

    direction = "UP" if plus_di_val >= minus_di_val else "DOWN"

    return adx_val, plus_di_val, minus_di_val, direction


def compute_rsi_slope(rsi_series: pd.Series, lookback: int = 5) -> float:
    """
    RSI slope: current RSI minus RSI N candles ago.
    Positive = RSI rising (momentum improving).
    """
    if len(rsi_series) < lookback + 1:
        return 0.0
    current = float(rsi_series.iloc[-1])
    past = float(rsi_series.iloc[-(lookback + 1)])
    return current - past


def detect_regime(
    df: pd.DataFrame,
    adx_trending_threshold: float = 25.0,
    adx_ranging_threshold: float = 20.0,
    adx_period: int = 14,
) -> dict:
    """
    Detect market regime from a 5m OHLCV DataFrame.

    Args:
        df: OHLCV DataFrame with columns [open, high, low, close, volume]
        adx_trending_threshold: ADX above this → trending
        adx_ranging_threshold:  ADX below this → ranging
        adx_period: period for ADX computation

    Returns:
        {
          "regime": "TRENDING_UP" | "TRENDING_DOWN" | "RANGING" | "UNCERTAIN",
          "adx": float,
          "plus_di": float,
          "minus_di": float,
          "adx_direction": "UP" | "DOWN",
        }
    """
    min_candles = adx_period * 2 + 5
    if df is None or len(df) < min_candles:
        return {
            "regime": "UNCERTAIN",
            "adx": 0.0,
            "plus_di": 0.0,
            "minus_di": 0.0,
            "adx_direction": "UP",
        }

    try:
        adx_val, plus_di, minus_di, direction = compute_adx(df, period=adx_period)

        if adx_val >= adx_trending_threshold:
            regime = "TRENDING_UP" if direction == "UP" else "TRENDING_DOWN"
        elif adx_val < adx_ranging_threshold:
            regime = "RANGING"
        else:
            regime = "UNCERTAIN"

        logger.debug(
            f"Regime: {regime} | ADX={adx_val:.1f} | +DI={plus_di:.1f} | -DI={minus_di:.1f}"
        )

        return {
            "regime": regime,
            "adx": round(adx_val, 2),
            "plus_di": round(plus_di, 2),
            "minus_di": round(minus_di, 2),
            "adx_direction": direction,
        }

    except Exception as e:
        logger.error(f"detect_regime error: {e}")
        return {
            "regime": "UNCERTAIN",
            "adx": 0.0,
            "plus_di": 0.0,
            "minus_di": 0.0,
            "adx_direction": "UP",
        }
