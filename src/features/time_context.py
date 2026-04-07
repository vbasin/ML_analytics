"""Time context features: session blocks, minutes into session, opening range.

v3: Uses bar_windows(). Added 30-minute opening range (Gap #20).
    Removed overnight/premarket blocks (always 0 with 8-11am filter).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute(
    index: pd.DatetimeIndex,
    candles: pd.DataFrame | None = None,
    close_series: pd.Series | None = None,
    trading_start: str = "08:00",
    trading_end: str = "11:00",
    timezone: str = "America/Chicago",
) -> pd.DataFrame:
    """Compute time context features.

    Parameters
    ----------
    index : DatetimeIndex at the output timestep resolution.
    candles : Optional OHLCV data for opening-range features.
    close_series : Price series aligned to *index* for breakout signals.
    trading_start, trading_end : RTH session boundaries (local time).
    timezone : Timezone for session-block encoding.
    """
    ts = index.tz_convert(timezone) if index.tz is not None else index.tz_localize("UTC").tz_convert(timezone)
    features = pd.DataFrame(index=index)

    # Minutes since session open
    start_h, start_m = (int(x) for x in trading_start.split(":"))
    session_start_minutes = start_h * 60 + start_m
    bar_minutes = ts.hour * 60 + ts.minute
    features["tc_minutes_into_session"] = (bar_minutes - session_start_minutes).astype(np.float32)

    # Cyclical time encoding (sine/cosine)
    minutes_in_day = ts.hour * 60 + ts.minute
    features["tc_sin_time"] = np.sin(2 * np.pi * minutes_in_day / 1440).astype(np.float32)
    features["tc_cos_time"] = np.cos(2 * np.pi * minutes_in_day / 1440).astype(np.float32)

    # Day of week (0=Mon, 4=Fri)
    features["tc_dow"] = ts.dayofweek.astype(np.float32)
    features["tc_sin_dow"] = np.sin(2 * np.pi * ts.dayofweek / 5).astype(np.float32)
    features["tc_cos_dow"] = np.cos(2 * np.pi * ts.dayofweek / 5).astype(np.float32)

    # Session block one-hot (only blocks relevant to 8-11am CST)
    blocks = {
        "open_5m":    (8 * 60, 8 * 60 + 5),
        "open_15m":   (8 * 60, 8 * 60 + 15),
        "morning":    (9 * 60, 11 * 60),
    }
    for name, (start, end) in blocks.items():
        features[f"tc_block_{name}"] = ((bar_minutes >= start) & (bar_minutes < end)).astype(np.float32)

    # Opening range features (5-min, 15-min, 30-min)
    if candles is not None and "high" in candles.columns:
        candles_tz = candles.copy()
        if candles_tz.index.tz is None:
            candles_tz.index = candles_tz.index.tz_localize("UTC").tz_convert(timezone)
        else:
            candles_tz.index = candles_tz.index.tz_convert(timezone)

        candle_minutes = candles_tz.index.hour * 60 + candles_tz.index.minute
        session_open = 8 * 60  # 08:00 CT

        for label, duration in [("5", 5), ("15", 15), ("30", 30)]:
            or_mask = (candle_minutes >= session_open) & (candle_minutes < session_open + duration)
            or_data = candles_tz[or_mask]
            if not or_data.empty:
                or_daily = or_data.groupby(or_data.index.date).agg(
                    or_high=("high", "max"), or_low=("low", "min"),
                )
                or_daily.index = pd.to_datetime(or_daily.index).tz_localize(timezone)
                or_aligned = or_daily.reindex(candles_tz.index, method="ffill")
                or_aligned.index = index[: len(or_aligned)]  # re-align safely
                features[f"tc_or{label}_high"] = or_aligned["or_high"].reindex(index).astype(np.float32)
                features[f"tc_or{label}_low"] = or_aligned["or_low"].reindex(index).astype(np.float32)

        # Breakout signals require a price series
        if close_series is not None:
            px = close_series.reindex(index)
            for label in ("5", "15", "30"):
                h_col = f"tc_or{label}_high"
                l_col = f"tc_or{label}_low"
                if h_col in features.columns:
                    features[f"tc_or{label}_breakout_up"] = (px > features[h_col]).astype(np.float32)
                    features[f"tc_or{label}_breakout_down"] = (px < features[l_col]).astype(np.float32)

    # Session progress — fraction of session elapsed (0 at open, 1 at close)
    start_h, start_m = (int(x) for x in trading_start.split(":"))
    end_h, end_m = (int(x) for x in trading_end.split(":"))
    session_len = (end_h * 60 + end_m) - (start_h * 60 + start_m)
    if session_len > 0:
        features["tc_session_progress"] = (
            features["tc_minutes_into_session"].clip(0, session_len) / session_len
        ).astype(np.float32)

    return features
