"""Trade location features: where trades execute relative to the spread.

Gap #8: trade_location, effective_spread, aggressive_pct.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.bar_windows import get_timestep


def compute(
    tbbo: pd.DataFrame,
    timestep: str | None = None,
) -> pd.DataFrame:
    """Compute trade location features from TBBO data.

    Parameters
    ----------
    tbbo : DataFrame with columns: ts_event, price, size, side, bid_px, ask_px.
    timestep : Resample interval.
    """
    timestep = timestep or get_timestep()

    df = tbbo.copy()
    df["mid"] = (df["bid_px"] + df["ask_px"]) / 2
    df["half_spread"] = (df["ask_px"] - df["bid_px"]) / 2

    # Trade location: -1 = at bid, +1 = at ask, 0 = at mid
    df["trade_location"] = (df["price"] - df["mid"]) / df["half_spread"].replace(0, np.nan)

    # Effective spread: 2 * |trade - mid|
    df["effective_spread"] = 2 * (df["price"] - df["mid"]).abs()

    # Aggressive: trade beyond far side of spread (crossing)
    df["aggressive"] = (df["trade_location"].abs() >= 0.9).astype(np.float32)

    # Resample to timestep
    bars = df.set_index("ts_event").resample(timestep).agg(
        tl_trade_location=("trade_location", "mean"),
        tl_effective_spread=("effective_spread", "mean"),
        tl_aggressive_pct=("aggressive", "mean"),
    )

    return bars[["tl_trade_location", "tl_effective_spread", "tl_aggressive_pct"]].astype(np.float32)
