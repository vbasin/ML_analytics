"""Feature engine v3: orchestrates all feature modules.

Loads source data from TimescaleDB, runs each feature module, concatenates
the result into a single feature matrix, and caches to Parquet.

v3 changes:
  - 5s timestep (configurable via VTECH_FEATURES_TIMESTEP)
  - 8:00–11:00 CST time filter applied to output
  - Removed macro_sentiment (VIXY) and equity_context (NVDA/TSLA/XLK/SMH)
  - Added 10 new feature modules
  - Uses bar_windows() for all window conversions

Run as:
    python -m src.features.engine --date=2025-07-15
    python -m src.features.engine --start=2025-04-01 --end=2025-07-01
"""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

from src.common.bar_windows import get_timestep
from src.common.db import get_dsn
from src.features import (
    book_pressure,
    candle_structure,
    cross_asset,
    daily_context,
    dealer_gex,
    economic_calendar,
    hawkes_clustering,
    higher_timeframe,
    large_trade_cvd,
    microstructure,
    options_surface,
    order_flow,
    realized_vol,
    same_side,
    sub_bar_dynamics,
    time_context,
    trade_arrival,
    trade_location,
    volume_profile,
    vwap,
    wavelets,
)

logger = logging.getLogger("vtech.features.engine")


class FeatureEngine:
    """Orchestrates feature computation from TimescaleDB sources."""

    def __init__(
        self,
        timestep: str | None = None,
        data_dir: str | None = None,
        timezone: str | None = None,
        trading_start: str | None = None,
        trading_end: str | None = None,
    ) -> None:
        self.timestep = timestep or get_timestep()
        self.data_dir = Path(data_dir or os.environ.get("VTECH_DATA_DIR", "/opt/vtech/data"))
        self.timezone = timezone or os.environ.get("VTECH_TIMEZONE", "America/Chicago")
        self.trading_start = trading_start or os.environ.get("VTECH_TRADING_HOURS_START", "08:00")
        self.trading_end = trading_end or os.environ.get("VTECH_TRADING_HOURS_END", "11:00")
        self._dsn = get_dsn()

    # ── public API ───────────────────────────────────────────────

    def build_date(self, target_date: date) -> Path:
        """Build all features for a single trading day and save to Parquet."""
        logger.info("build_date_start", extra={"date": str(target_date)})

        tbbo = self._load_tbbo(target_date)
        bbo_all = self._load_bbo_1s(target_date)
        defs = self._load_definitions(target_date)
        stats = self._load_statistics(target_date)
        candles_1m = self._load_candles_1min(target_date)

        # Separate futures vs options BBO using definitions table
        opt_ids = set(defs["instrument_id"]) if not defs.empty else set()
        if not bbo_all.empty and opt_ids:
            bbo_fut = bbo_all[~bbo_all["instrument_id"].isin(opt_ids)]
            bbo_opt = bbo_all[bbo_all["instrument_id"].isin(opt_ids)]
        else:
            bbo_fut = bbo_all
            bbo_opt = pd.DataFrame()

        # Filter candles to primary instrument (most-traded)
        if not candles_1m.empty and "instrument_id" in candles_1m.columns:
            vol_by_id = candles_1m.groupby("instrument_id")["volume"].sum()
            primary_id = vol_by_id.idxmax()
            candles_1m = candles_1m[candles_1m["instrument_id"] == primary_id].copy()

        # Underlying (futures) mid price series for IV/GEX/vol computation
        und_prices = pd.Series(dtype=np.float64)
        if not bbo_fut.empty:
            und_series = bbo_fut.set_index("ts_event").sort_index()
            und_prices = ((und_series["bid_px"] + und_series["ask_px"]) / 2).resample(self.timestep).last().ffill()

        all_features: list[pd.DataFrame] = []

        # ── P1: Book pressure (liquidity withdrawal + fleeting + velocity) ──
        if not bbo_fut.empty:
            all_features.append(book_pressure.compute(bbo_fut, self.timestep))

        # ── P2: Order flow (CVD + divergence) ──
        if not tbbo.empty:
            all_features.append(order_flow.compute(tbbo, self.timestep))

        # ── P2b: Trade location ──
        if not tbbo.empty:
            all_features.append(trade_location.compute(tbbo, self.timestep))

        # ── P2c: Same-side runs ──
        if not tbbo.empty:
            all_features.append(same_side.compute(tbbo, self.timestep))

        # ── P2d: Large-trade CVD ──
        if not tbbo.empty:
            all_features.append(large_trade_cvd.compute(tbbo, self.timestep))

        # ── P2e: Trade arrival rate + acceleration ──
        if not tbbo.empty:
            all_features.append(trade_arrival.compute(tbbo, self.timestep))

        # ── P3: Options IV surface ──
        if not bbo_opt.empty and not defs.empty and not und_prices.empty:
            all_features.append(
                options_surface.compute(bbo_opt, defs, und_prices, self.timestep)
            )

        # ── P3b: Dealer GEX ──
        if not bbo_opt.empty and not defs.empty and not und_prices.empty:
            all_features.append(
                dealer_gex.compute(bbo_opt, defs, und_prices, self.timestep)
            )

        # ── P4: Daily context ──
        if not stats.empty and not und_prices.empty:
            all_features.append(daily_context.compute(stats, und_prices, self.timestep))

        # ── P5: Microstructure (VPIN, Kyle's Lambda, Amihud) ──
        if not tbbo.empty:
            all_features.append(microstructure.compute(tbbo, self.timestep))

        # ── P5b: Realized volatility + vol-of-vol ──
        if not und_prices.empty:
            all_features.append(realized_vol.compute(und_prices, self.timestep))

        # ── P5c: Sub-bar dynamics ──
        if not bbo_fut.empty or not tbbo.empty:
            all_features.append(sub_bar_dynamics.compute(
                bbo_fut if not bbo_fut.empty else pd.DataFrame(),
                tbbo if not tbbo.empty else pd.DataFrame(),
                self.timestep,
            ))

        # ── P5d: Volume profile ──
        if not tbbo.empty and not und_prices.empty:
            all_features.append(volume_profile.compute(tbbo, und_prices, self.timestep))

        # ── P5e: Hawkes event clustering ──
        if not und_prices.empty:
            all_features.append(hawkes_clustering.compute(und_prices, self.timestep))

        # ── P6: Wavelets (need >= 128 bars at 5s) ──
        if not candles_1m.empty and len(candles_1m) >= 64:
            all_features.append(wavelets.compute(candles_1m))

        # ── P7: Candle structure ──
        if not candles_1m.empty:
            all_features.append(candle_structure.compute(candles_1m))

        # ── P7b: Multi-timeframe candle structure (15m, 30m) ──
        if not candles_1m.empty and len(candles_1m) >= 15:
            all_features.append(candle_structure.compute_multitimeframe(candles_1m))

        # ── P8: VWAP ──
        if not tbbo.empty:
            all_features.append(vwap.compute(tbbo, self.timestep))

        # ── P9: Cross-asset (NQ vs ES) ──
        es_candles = self._load_es_candles_1min(target_date)
        if not candles_1m.empty and not es_candles.empty:
            nq_resampled = candles_1m[["close"]].resample(self.timestep).last().ffill()
            es_resampled = es_candles[["close"]].resample(self.timestep).last().ffill()
            ca_feats = cross_asset.compute(nq_resampled, es_resampled, self.timestep)
            if not ca_feats.empty:
                all_features.append(ca_feats)

        # ── Time context (always available) ──
        if all_features:
            idx = all_features[0].index
            close_ser = None
            if not candles_1m.empty and "close" in candles_1m.columns:
                close_ser = candles_1m["close"].resample(self.timestep).last().ffill().reindex(idx)
            all_features.append(
                time_context.compute(
                    idx, candles_1m if not candles_1m.empty else None,
                    close_series=close_ser,
                    trading_start=self.trading_start,
                    trading_end=self.trading_end,
                )
            )

        # ── Economic calendar ──
        if all_features:
            idx = all_features[0].index
            all_features.append(economic_calendar.compute(idx, self.timezone))

        # ── Higher timeframe context ──
        if all_features:
            idx = all_features[0].index
            daily_ohlc = self._load_daily_ohlc(target_date)
            all_features.append(higher_timeframe.compute(daily_ohlc, target_date, idx))

        if not all_features:
            logger.warning("no_features", extra={"date": str(target_date)})
            return Path()

        # Concatenate and align
        deduped = [f.loc[~f.index.duplicated(keep="first")] for f in all_features]
        combined = pd.concat(deduped, axis=1)
        combined = combined.loc[~combined.index.duplicated(keep="first")]
        combined = combined.sort_index()

        # ── Apply 8:00–11:00 CST time filter ──
        combined = self._filter_trading_hours(combined)

        # Include close price for label generation (prefixed with _ to mark as non-feature)
        if not candles_1m.empty and "close" in candles_1m.columns:
            close_resampled = candles_1m["close"].resample(self.timestep).last().ffill()
            combined["_close"] = close_resampled.reindex(combined.index)

        # Save to Parquet
        out_path = self.data_dir / "parquet" / f"features_{target_date}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(out_path, engine="pyarrow", compression="zstd")

        logger.info("build_date_done", extra={
            "date": str(target_date),
            "features": combined.shape[1],
            "rows": len(combined),
            "nan_pct": float(combined.isna().mean().mean()),
            "path": str(out_path),
        })
        return out_path

    def build_range(self, start: date, end: date) -> list[Path]:
        """Build features for every trading day in [start, end)."""
        paths = []
        current = start
        while current < end:
            if current.weekday() < 5:  # Mon-Fri
                paths.append(self.build_date(current))
            current += timedelta(days=1)
        return [p for p in paths if p != Path()]

    # ── time filter ──────────────────────────────────────────────

    def _filter_trading_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to only rows within trading_start–trading_end in local timezone."""
        if df.empty:
            return df

        idx = df.index
        if idx.tz is None:
            idx_local = idx.tz_localize("UTC").tz_convert(self.timezone)
        else:
            idx_local = idx.tz_convert(self.timezone)

        start_h, start_m = (int(x) for x in self.trading_start.split(":"))
        end_h, end_m = (int(x) for x in self.trading_end.split(":"))

        bar_minutes = idx_local.hour * 60 + idx_local.minute
        start_minutes = start_h * 60 + start_m
        end_minutes = end_h * 60 + end_m

        mask = (bar_minutes >= start_minutes) & (bar_minutes < end_minutes)
        return df.loc[mask]

    # ── data loaders ─────────────────────────────────────────────

    def _query(self, sql: str, params: tuple) -> pd.DataFrame:
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                cols = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
        return pd.DataFrame(rows, columns=cols)

    def _load_tbbo(self, d: date) -> pd.DataFrame:
        return self._query(
            "SELECT * FROM databento.tbbo WHERE ts_event::date = %s "
            "AND symbol NOT LIKE 'ES%%' ORDER BY ts_event",
            (d,),
        )

    def _load_bbo_1s(self, d: date) -> pd.DataFrame:
        return self._query(
            "SELECT * FROM databento.bbo_1s WHERE ts_event::date = %s ORDER BY ts_event",
            (d,),
        )

    def _load_definitions(self, d: date) -> pd.DataFrame:
        return self._query(
            "SELECT * FROM databento.definitions WHERE ts_event::date = %s",
            (d,),
        )

    def _load_statistics(self, d: date) -> pd.DataFrame:
        return self._query(
            "SELECT * FROM databento.statistics WHERE ts_event::date = %s ORDER BY ts_event",
            (d,),
        )

    def _load_candles_1min(self, d: date) -> pd.DataFrame:
        df = self._query(
            "SELECT * FROM databento.tbbo_1min WHERE bucket::date = %s "
            "AND symbol NOT LIKE 'ES%%' ORDER BY bucket",
            (d,),
        )
        if not df.empty:
            df = df.set_index("bucket").sort_index()
        return df

    def _load_es_candles_1min(self, d: date) -> pd.DataFrame:
        """Load ES futures 1-minute candles for cross-asset features."""
        df = self._query(
            "SELECT * FROM databento.tbbo_1min WHERE bucket::date = %s "
            "AND symbol LIKE 'ES%%' AND symbol NOT LIKE 'ES%%-%%' ORDER BY bucket",
            (d,),
        )
        if not df.empty:
            if "instrument_id" in df.columns:
                vol_by_id = df.groupby("instrument_id")["volume"].sum()
                primary_id = vol_by_id.idxmax()
                df = df[df["instrument_id"] == primary_id].copy()
            df = df.set_index("bucket").sort_index()
        return df

    def _load_daily_ohlc(self, d: date) -> pd.DataFrame:
        """Load ~20 prior trading days of daily OHLC for higher-timeframe features.

        Built from tbbo_1min aggregation (no separate daily table needed).
        """
        start = d - timedelta(days=35)  # ~25 business days
        df = self._query(
            "SELECT bucket::date AS date, "
            "  (array_agg(open ORDER BY bucket))[1] AS open, "
            "  max(high) AS high, "
            "  min(low) AS low, "
            "  (array_agg(close ORDER BY bucket DESC))[1] AS close "
            "FROM databento.tbbo_1min "
            "WHERE bucket::date BETWEEN %s AND %s "
            "  AND symbol NOT LIKE 'ES%%' "
            "GROUP BY bucket::date "
            "ORDER BY date",
            (start, d),
        )
        return df


# ── CLI ──────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Build feature matrices")
    parser.add_argument("--date", help="Single date (YYYY-MM-DD) or 'today'")
    parser.add_argument("--start", help="Start date for range build")
    parser.add_argument("--end", help="End date for range build")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    engine = FeatureEngine()

    if args.date:
        d = date.today() if args.date == "today" else datetime.strptime(args.date, "%Y-%m-%d").date()
        engine.build_date(d)
    elif args.start and args.end:
        engine.build_range(
            datetime.strptime(args.start, "%Y-%m-%d").date(),
            datetime.strptime(args.end, "%Y-%m-%d").date(),
        )
    else:
        parser.error("Provide --date or --start/--end")


if __name__ == "__main__":
    main()
