"""Options IV surface features computed from BBO-1s options + definitions.

Includes: ATM IV, put-call skew, term structure slope, butterfly spread,
IV velocity, GEX proxy (gamma exposure), 25-delta risk reversal.

v3: Uses bar_windows(). Added iv_25d_risk_reversal, iv_term_slope,
    iv_25d_butterfly. Doubled all rolling windows.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.common.bar_windows import bars, bars_m, get_timestep

logger = logging.getLogger("vtech.features.options_surface")

# ── Black-Scholes IV inversion ──────────────────────────────────


def _bs_price(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if is_call:
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _bs_delta(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    """Compute Black-Scholes delta."""
    if T <= 0 or sigma <= 0:
        return 1.0 if is_call else -1.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return float(norm.cdf(d1)) if is_call else float(norm.cdf(d1) - 1)


def _implied_vol(
    mid: float, S: float, K: float, T: float, r: float, is_call: bool,
    tol: float = 1e-6, max_iter: int = 50,
) -> float | None:
    """Newton-Raphson IV solver. Returns None on failure."""
    if T <= 0 or mid <= 0:
        return None
    sigma = 0.3  # initial guess
    for _ in range(max_iter):
        price = _bs_price(S, K, T, r, sigma, is_call)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        vega = S * np.sqrt(T) * norm.pdf(d1)
        if vega < 1e-12:
            return None
        sigma -= (price - mid) / vega
        if sigma <= 0:
            return None
        if abs(price - mid) < tol:
            return sigma
    return None


# ── Surface builder ─────────────────────────────────────────────


def _build_surface(
    bbo_opts: pd.DataFrame,
    defs: pd.DataFrame,
    underlying_price: float,
    risk_free: float = 0.05,
    max_spread_pct: float = 0.25,
    min_price: float = 0.25,
) -> pd.DataFrame:
    """Compute IV + delta for each option instrument at a single timestamp."""
    merged = bbo_opts.merge(
        defs[["instrument_id", "strike_price", "expiration", "instrument_class", "multiplier"]],
        on="instrument_id",
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame()

    merged["mid"] = (merged["bid_px"] + merged["ask_px"]) / 2
    merged["spread_pct"] = (merged["ask_px"] - merged["bid_px"]) / merged["mid"].replace(0, np.nan)

    # Quality filter
    mask = (
        (merged["spread_pct"] <= max_spread_pct)
        & (merged["mid"] >= min_price)
        & (merged["strike_price"] > 0)
    )
    merged = merged.loc[mask].copy()
    if merged.empty:
        return pd.DataFrame()

    # Compute time to expiry in years
    merged["dte"] = (merged["expiration"] - merged["ts_event"]).dt.total_seconds() / (365.25 * 86400)
    merged = merged[merged["dte"] > 0].copy()

    # Compute IV
    merged["iv"] = merged.apply(
        lambda r: _implied_vol(
            r["mid"], underlying_price, r["strike_price"], r["dte"],
            risk_free, r["instrument_class"] == "C",
        ),
        axis=1,
    )
    merged = merged.dropna(subset=["iv"])
    merged["moneyness"] = np.log(merged["strike_price"] / underlying_price)

    # Compute delta for each option
    merged["delta"] = merged.apply(
        lambda r: _bs_delta(
            underlying_price, r["strike_price"], r["dte"],
            risk_free, r["iv"], r["instrument_class"] == "C",
        ),
        axis=1,
    )

    return merged


# ── Aggregated features ─────────────────────────────────────────


def compute(
    bbo_opts: pd.DataFrame,
    definitions: pd.DataFrame,
    underlying_prices: pd.Series,
    timestep: str | None = None,
    risk_free: float = 0.05,
) -> pd.DataFrame:
    """Compute options IV surface features resampled to *timestep*.

    Parameters
    ----------
    bbo_opts : BBO-1s DataFrame for NQ options.
    definitions : Definitions DataFrame.
    underlying_prices : Series with NQ futures mid prices.
    timestep : Output resample interval.
    """
    timestep = timestep or get_timestep()
    w30s = bars(30)
    w1m  = bars_m(1)
    w2m  = bars_m(2)

    # Resample underlying to per-timestep
    und = underlying_prices.resample(timestep).last().ffill()
    ts_index = und.index

    records: list[dict] = []

    for ts in ts_index:
        window_start = ts - pd.Timedelta(timestep)
        snap = bbo_opts[(bbo_opts["ts_event"] > window_start) & (bbo_opts["ts_event"] <= ts)]
        if snap.empty:
            records.append({"ts": ts})
            continue

        # Use last quote per instrument in this window
        snap_last = snap.sort_values("ts_event").groupby("instrument_id").last().reset_index()
        price = und.get(ts, np.nan)
        if np.isnan(price):
            records.append({"ts": ts})
            continue

        surface = _build_surface(snap_last, definitions, price, risk_free)
        if surface.empty:
            records.append({"ts": ts})
            continue

        # 0DTE filter (DTE < 1 day)
        dte_0 = surface[surface["dte"] < 1 / 365.25]
        all_exp = surface

        row: dict = {"ts": ts}

        # ATM IV (nearest-to-money)
        atm = all_exp.iloc[(all_exp["moneyness"].abs()).argsort()[:5]]
        row["iv_atm_mean"] = atm["iv"].mean()

        # Put-call skew (25-delta wing approximation via moneyness)
        calls = all_exp[all_exp["instrument_class"] == "C"]
        puts = all_exp[all_exp["instrument_class"] == "P"]
        otm_puts = puts[puts["moneyness"] < -0.02]
        otm_calls = calls[calls["moneyness"] > 0.02]
        row["iv_skew"] = otm_puts["iv"].mean() - otm_calls["iv"].mean() if len(otm_puts) and len(otm_calls) else np.nan

        # Butterfly (wings vs ATM)
        wings_iv = pd.concat([otm_puts, otm_calls])["iv"].mean() if len(otm_puts) + len(otm_calls) else np.nan
        row["iv_butterfly"] = wings_iv - row["iv_atm_mean"] if not np.isnan(wings_iv) else np.nan

        # 0DTE specific
        if not dte_0.empty:
            atm_0 = dte_0.iloc[(dte_0["moneyness"].abs()).argsort()[:3]]
            row["iv_0dte_atm"] = atm_0["iv"].mean()
        else:
            row["iv_0dte_atm"] = np.nan

        # ── 25-delta risk reversal (Gap #28) ──
        # Find options nearest to ±0.25 delta
        puts_25d = puts[(puts["delta"] > -0.35) & (puts["delta"] < -0.15)]
        calls_25d = calls[(calls["delta"] > 0.15) & (calls["delta"] < 0.35)]
        if len(puts_25d) and len(calls_25d):
            row["iv_25d_risk_reversal"] = calls_25d["iv"].mean() - puts_25d["iv"].mean()
            row["iv_25d_butterfly"] = (calls_25d["iv"].mean() + puts_25d["iv"].mean()) / 2 - row["iv_atm_mean"]
        else:
            row["iv_25d_risk_reversal"] = np.nan
            row["iv_25d_butterfly"] = np.nan

        # ── Term structure slope ──
        near = all_exp[all_exp["dte"] < 5 / 365.25]
        far = all_exp[all_exp["dte"] >= 5 / 365.25]
        if not near.empty and not far.empty:
            near_atm = near.iloc[(near["moneyness"].abs()).argsort()[:3]]
            far_atm = far.iloc[(far["moneyness"].abs()).argsort()[:3]]
            row["iv_term_slope"] = far_atm["iv"].mean() - near_atm["iv"].mean()
        else:
            row["iv_term_slope"] = np.nan

        records.append(row)

    features = pd.DataFrame(records).set_index("ts")

    # IV velocity (rate of change) — doubled windows
    for w in (w30s, w1m, w2m):
        features[f"iv_atm_chg_{w}"] = features["iv_atm_mean"].diff(w)
        features[f"iv_atm_zscore_{w}"] = (
            (features["iv_atm_mean"] - features["iv_atm_mean"].rolling(w).mean())
            / features["iv_atm_mean"].rolling(w).std().replace(0, np.nan)
        )

    return features.astype(np.float32)
