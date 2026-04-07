"""Higher-timeframe context: daily trend, position-in-range, gap, ATR ratio.

Provides multi-day structural context that is constant within each intraday
session but varies day-to-day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute(
    daily_ohlc: pd.DataFrame,
    target_date: str | pd.Timestamp,
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Compute higher-timeframe features and broadcast to intraday index.

    Parameters
    ----------
    daily_ohlc : DataFrame with columns: date, open, high, low, close.
                 Must include at least 20 prior trading days.
    target_date : The date being processed (used to select prior days).
    index : Intraday DatetimeIndex to broadcast features onto.
    """
    features = pd.DataFrame(index=index)

    if daily_ohlc.empty or len(daily_ohlc) < 2:
        features["ht_daily_trend_10"] = np.nan
        features["ht_daily_trend_20"] = np.nan
        features["ht_daily_pos_in_range_20"] = np.nan
        features["ht_gap_pct"] = np.nan
        features["ht_daily_atr_ratio"] = np.nan
        return features.astype(np.float32)

    # Ensure sorted by date
    df = daily_ohlc.sort_values("date").copy()
    target = pd.Timestamp(target_date).date() if not isinstance(target_date, str) else pd.Timestamp(target_date).date()

    # Filter to days before target (strictly prior)
    prior = df[df["date"] < target].tail(20)
    if len(prior) < 2:
        features["ht_daily_trend_10"] = np.nan
        features["ht_daily_trend_20"] = np.nan
        features["ht_daily_pos_in_range_20"] = np.nan
        features["ht_gap_pct"] = np.nan
        features["ht_daily_atr_ratio"] = np.nan
        return features.astype(np.float32)

    closes = prior["close"].values
    highs = prior["high"].values
    lows = prior["low"].values

    # ── Trend: slope of linear regression on closes, normalized by ATR ──
    atr_20 = np.mean(highs - lows) if len(highs) > 0 else 1.0
    atr_20 = max(atr_20, 0.01)

    def _trend_slope(closes_subset: np.ndarray) -> float:
        n = len(closes_subset)
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=np.float64)
        slope = np.polyfit(x, closes_subset, 1)[0]
        return float(slope / atr_20)

    ht_trend_10 = _trend_slope(closes[-10:]) if len(closes) >= 10 else _trend_slope(closes)
    ht_trend_20 = _trend_slope(closes)

    # ── Position in range: (yesterday_close - 20d_low) / (20d_high - 20d_low) ──
    range_high = highs.max()
    range_low = lows.min()
    range_width = range_high - range_low
    yesterday_close = closes[-1]
    ht_pos = float((yesterday_close - range_low) / range_width) if range_width > 0 else 0.5

    # ── Gap: today's open vs yesterday's close ──
    today_data = df[df["date"] == target]
    if not today_data.empty:
        today_open = today_data["open"].iloc[0]
        ht_gap_pct = float((today_open - yesterday_close) / yesterday_close) if yesterday_close != 0 else 0.0
    else:
        ht_gap_pct = 0.0

    # ── ATR ratio: 5-day ATR / 20-day ATR ──
    tr = highs - lows  # simplified true range
    atr_5 = np.mean(tr[-5:]) if len(tr) >= 5 else np.mean(tr)
    ht_atr_ratio = float(atr_5 / atr_20)

    # Broadcast to all intraday bars
    features["ht_daily_trend_10"] = np.float32(ht_trend_10)
    features["ht_daily_trend_20"] = np.float32(ht_trend_20)
    features["ht_daily_pos_in_range_20"] = np.float32(ht_pos)
    features["ht_gap_pct"] = np.float32(ht_gap_pct)
    features["ht_daily_atr_ratio"] = np.float32(ht_atr_ratio)

    return features.astype(np.float32)
