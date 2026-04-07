"""Volume profile features: POC, Value Area, and migration.

Gap #13: Point of Control, Value Area high/low, and POC migration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.bar_windows import bars_m, get_timestep


def compute(
    tbbo: pd.DataFrame,
    underlying_prices: pd.Series,
    timestep: str | None = None,
    tick_size: float = 0.25,
    va_pct: float = 0.70,
) -> pd.DataFrame:
    """Compute rolling volume profile features.

    Parameters
    ----------
    tbbo : DataFrame with columns: ts_event, price, size.
    underlying_prices : Resampled price series (for normalization).
    timestep : Resample interval.
    tick_size : NQ tick size (0.25 points).
    va_pct : Value area percentage (default 70%).
    """
    timestep = timestep or get_timestep()
    w30m = bars_m(30)
    w2m  = bars_m(2)

    df = tbbo.sort_values("ts_event").set_index("ts_event")

    # Round prices to tick level
    df["price_tick"] = (df["price"] / tick_size).round() * tick_size

    # Resample to timestep — carry forward POC/VA computations
    und = underlying_prices.resample(timestep).last().ffill()

    features = pd.DataFrame(index=und.index)
    features["vp_poc_dist"] = np.nan
    features["vp_va_high_dist"] = np.nan
    features["vp_va_low_dist"] = np.nan
    features["vp_in_value_area"] = np.nan

    # Compute volume profile in rolling windows
    # For efficiency, compute per-bar volume at each price level
    bar_profiles: list[dict] = []
    for ts in und.index:
        window_start = ts - pd.Timedelta(seconds=w30m * 5)  # w30m bars * 5s each
        window_data = df[(df.index > window_start) & (df.index <= ts)]
        if window_data.empty:
            bar_profiles.append({})
            continue

        # Build volume-by-price
        vbp = window_data.groupby("price_tick")["size"].sum()
        if vbp.empty:
            bar_profiles.append({})
            continue

        # POC = price with highest volume
        poc = vbp.idxmax()

        # Value Area: expand outward from POC until va_pct of total volume
        total_vol = vbp.sum()
        sorted_prices = vbp.sort_values(ascending=False)
        cum_vol = sorted_prices.cumsum()
        va_prices = sorted_prices[cum_vol <= total_vol * va_pct].index
        if len(va_prices) == 0:
            va_prices = pd.Index([poc])
        va_high = va_prices.max()
        va_low = va_prices.min()

        current_price = und.get(ts, np.nan)
        if np.isnan(current_price) or current_price == 0:
            bar_profiles.append({})
            continue

        bar_profiles.append({
            "poc_dist": (current_price - poc) / current_price,
            "va_high_dist": (current_price - va_high) / current_price,
            "va_low_dist": (current_price - va_low) / current_price,
            "in_va": 1.0 if va_low <= current_price <= va_high else 0.0,
            "poc": poc,
        })

    # Assign to features
    for i, (ts, prof) in enumerate(zip(und.index, bar_profiles)):
        if prof:
            features.loc[ts, "vp_poc_dist"] = prof["poc_dist"]
            features.loc[ts, "vp_va_high_dist"] = prof["va_high_dist"]
            features.loc[ts, "vp_va_low_dist"] = prof["va_low_dist"]
            features.loc[ts, "vp_in_value_area"] = prof["in_va"]

    # POC slope: direction of POC migration
    poc_series = pd.Series(
        [p.get("poc", np.nan) for p in bar_profiles],
        index=und.index,
    )
    features["vp_poc_slope"] = poc_series.diff(w2m) / poc_series.shift(w2m).replace(0, np.nan)

    return features.astype(np.float32)
