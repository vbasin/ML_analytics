"""Candle structure features: body/range ratio, CLV, volatility regime.

v3: Uses bar_windows(). Removed cs_5m multitimeframe (redundant with 128-bar
    sequence). Removed cs_clv_bullish_count (redundant with cs_clv_ma).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.bar_windows import bars, bars_m


def compute(
    candles: pd.DataFrame,
) -> pd.DataFrame:
    """Compute candle structure features from OHLCV candle data.

    Parameters
    ----------
    candles : DataFrame with columns: open, high, low, close, volume.
    """
    # CLV rolling windows: ~20s, 30s, 50s, 100s in real time
    w20s = bars(20)
    w30s = bars(30)
    w50s = bars(50)
    w100s = bars(100)
    clv_rolling = [w20s, w30s, w50s, w100s]

    w1m  = bars_m(1)
    w5m  = bars_m(5)
    w10m = bars_m(10)

    df = candles.copy()
    features = pd.DataFrame(index=df.index)

    candle_range = df["high"] - df["low"]
    body = (df["close"] - df["open"]).abs()

    features["cs_body_range_ratio"] = body / candle_range.replace(0, np.nan)
    features["cs_body_direction"] = np.sign(df["close"] - df["open"])

    # Close Location Value (CLV): where close sits in the bar's range
    features["cs_clv"] = (2 * df["close"] - df["high"] - df["low"]) / candle_range.replace(0, np.nan)

    # Rolling CLV
    for w in clv_rolling:
        features[f"cs_clv_ma_{w}"] = features["cs_clv"].rolling(w).mean()

    # Upper / lower shadow ratios
    features["cs_upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1)) / candle_range.replace(0, np.nan)
    features["cs_lower_shadow"] = (df[["open", "close"]].min(axis=1) - df["low"]) / candle_range.replace(0, np.nan)

    # Volatility regime: rolling range relative to longer-term
    for w in (w1m, w5m, w10m):
        features[f"cs_range_ma_{w}"] = candle_range.rolling(w).mean()
    features["cs_vol_regime"] = candle_range / features[f"cs_range_ma_{w5m}"].replace(0, np.nan)

    # Range expansion/contraction
    features["cs_range_chg"] = candle_range / candle_range.shift(1).replace(0, np.nan) - 1

    return features.astype(np.float32)


def compute_multitimeframe(
    candles_1m: pd.DataFrame,
    timeframes: list[str] | None = None,
) -> pd.DataFrame:
    """Resample 1-min candles to 15m and 30m and compute body_range + CLV.

    5m timeframe removed (redundant with 128-bar × 5s sequence coverage).
    """
    if timeframes is None:
        timeframes = ["15min", "30min"]

    features = pd.DataFrame(index=candles_1m.index)

    for tf in timeframes:
        label = tf.replace("min", "m")
        ohlcv = candles_1m.resample(tf).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna(subset=["open"])

        rng = ohlcv["high"] - ohlcv["low"]
        body = (ohlcv["close"] - ohlcv["open"]).abs()
        ohlcv[f"cs_{label}_body_range"] = body / rng.replace(0, np.nan)
        ohlcv[f"cs_{label}_clv"] = (
            (2 * ohlcv["close"] - ohlcv["high"] - ohlcv["low"]) / rng.replace(0, np.nan)
        )

        # Forward-fill back to 1-min resolution
        aligned = ohlcv[[f"cs_{label}_body_range", f"cs_{label}_clv"]].reindex(
            candles_1m.index, method="ffill"
        )
        features = features.join(aligned)

    return features.astype(np.float32)
