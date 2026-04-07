"""Dealer GEX (Gamma Exposure) estimation from NQ options.

Gap #25: Estimates net dealer gamma from 0DTE + monthly options.
Requires: NQ.OPT BBO-1s, definitions, underlying prices.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.common.bar_windows import bars_m, get_timestep

logger = logging.getLogger("vtech.features.dealer_gex")


def _bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes gamma (same for puts and calls)."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))


def compute(
    bbo_opts: pd.DataFrame,
    definitions: pd.DataFrame,
    underlying_prices: pd.Series,
    timestep: str | None = None,
    risk_free: float = 0.05,
    contract_multiplier: float = 20.0,
) -> pd.DataFrame:
    """Compute dealer GEX features from options data.

    Focuses on 0DTE + monthly expirations for cost efficiency.

    Parameters
    ----------
    bbo_opts : BBO-1s DataFrame for NQ options.
    definitions : Definitions DataFrame with OI if available.
    underlying_prices : Series with NQ futures mid prices.
    timestep : Resample interval.
    contract_multiplier : NQ options multiplier ($20/point).
    """
    timestep = timestep or get_timestep()
    w2m = bars_m(2)

    und = underlying_prices.resample(timestep).last().ffill()

    features = pd.DataFrame(index=und.index)
    features["gx_dealer_gex"] = np.nan
    features["gx_gex_per_strike_max"] = np.nan
    features["gx_gex_flip_dist"] = np.nan

    # Merge options with definitions
    defs_cols = ["instrument_id", "strike_price", "expiration", "instrument_class"]
    if "open_interest" in definitions.columns:
        defs_cols.append("open_interest")
    merged_defs = definitions[defs_cols].copy()

    for ts in und.index:
        window_start = ts - pd.Timedelta(timestep)
        snap = bbo_opts[(bbo_opts["ts_event"] > window_start) & (bbo_opts["ts_event"] <= ts)]
        if snap.empty:
            continue

        snap_last = snap.sort_values("ts_event").groupby("instrument_id").last().reset_index()
        price = und.get(ts, np.nan)
        if np.isnan(price):
            continue

        merged = snap_last.merge(merged_defs, on="instrument_id", how="inner")
        if merged.empty:
            continue

        merged["mid"] = (merged["bid_px"] + merged["ask_px"]) / 2
        merged["dte"] = (merged["expiration"] - merged["ts_event"]).dt.total_seconds() / (365.25 * 86400)
        merged = merged[merged["dte"] > 0].copy()

        if merged.empty:
            continue

        # Filter to 0DTE + monthly (DTE < 1 day or nearest monthly)
        dte_0 = merged[merged["dte"] < 1 / 365.25]
        monthly = merged[merged["dte"] >= 20 / 365.25]
        if not monthly.empty:
            nearest_monthly_exp = monthly["expiration"].min()
            monthly = monthly[merged["expiration"] == nearest_monthly_exp]
        selected = pd.concat([dte_0, monthly])

        if selected.empty:
            continue

        # Estimate IV for gamma computation (use simple approximation)
        # Use 30% default IV if we can't compute it
        selected["iv"] = 0.30

        # Compute gamma per option
        selected["gamma"] = selected.apply(
            lambda r: _bs_gamma(price, r["strike_price"], r["dte"], risk_free, r["iv"]),
            axis=1,
        )

        # Use bid_sz + ask_sz as OI proxy if OI not available
        if "open_interest" in selected.columns:
            selected["oi"] = selected["open_interest"].fillna(selected["bid_sz"] + selected["ask_sz"])
        else:
            selected["oi"] = selected["bid_sz"] + selected["ask_sz"]

        # Dealer GEX: assume dealers are short options (market makers)
        # GEX = gamma * OI * contractMultiplier * spotPrice * 0.01
        selected["gex"] = (
            selected["gamma"] * selected["oi"] * contract_multiplier * price * 0.01
        )
        # Calls have positive gamma for dealer (if short), puts negative
        selected["dealer_gex"] = np.where(
            selected["instrument_class"] == "C",
            -selected["gex"],  # short call → negative gamma when price rises
            selected["gex"],   # short put → positive gamma when price falls
        )

        total_gex = selected["dealer_gex"].sum()
        max_strike_gex = selected.groupby("strike_price")["dealer_gex"].sum().abs().max()

        # GEX flip point: where dealer gamma changes sign
        strike_gex = selected.groupby("strike_price")["dealer_gex"].sum().sort_index()
        if len(strike_gex) > 1:
            signs = np.sign(strike_gex.values)
            flips = np.where(np.diff(signs) != 0)[0]
            if len(flips) > 0:
                flip_strike = strike_gex.index[flips[0]]
                gex_flip_dist = (price - flip_strike) / price
            else:
                gex_flip_dist = np.nan
        else:
            gex_flip_dist = np.nan

        features.loc[ts, "gx_dealer_gex"] = total_gex
        features.loc[ts, "gx_gex_per_strike_max"] = max_strike_gex
        features.loc[ts, "gx_gex_flip_dist"] = gex_flip_dist

    # GEX rate of change
    features[f"gx_gex_chg_{w2m}"] = features["gx_dealer_gex"].diff(w2m)

    return features.astype(np.float32)
