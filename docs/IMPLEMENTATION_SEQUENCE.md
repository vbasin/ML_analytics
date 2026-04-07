# Implementation Sequence & Roadmap

> **Date**: 2026-04-06
> **Companion to**: [DATABENTO_VM_ARCHITECTURE_v2.md](DATABENTO_VM_ARCHITECTURE_v2.md) §16 (original sequence), [DATABENTO_VM_ARCHITECTURE_v3.md](DATABENTO_VM_ARCHITECTURE_v3.md) (current architecture)

This document tracks the full implementation history from v2's Phase 1–12 plan through the current state, and defines the remaining roadmap for the ML orchestration pipeline.

---

## Completed Phases (from Architecture v2 §16)

### Phase 1: Foundation ✅

- Provisioned Oracle Cloud VM — Ubuntu 22.04, 4 vCPU, 31 GB RAM, 49 GB root + 500 GB data disk
- Installed Python 3.12 (venv), PostgreSQL 18.3 + TimescaleDB 2.26.1
- Created `vtech` user, directory structure at `/opt/vtech/`
- Configured `.env` with Databento API key + PG credentials
- Verified Databento connectivity

### Phase 2: Database Setup ✅

- Created `vtech_market` database with `databento` schema
- Created hypertables: `tbbo`, `bbo_1s`, `definitions`, `statistics`
- Set compression policies on `tbbo` and `bbo_1s` (30-day interval)
- Created `tbbo_1min` continuous aggregate with auto-refresh policy

### Phase 3: Historical Backfill — NQ Futures ✅

- Downloaded NQ.FUT TBBO via `timeseries.get_range` (2025-04-01 → 2026-04-02)
- Ingested TBBO into TimescaleDB (122M+ rows initial)
- BBO-1s populated from live stream (not backfilled independently)

### Phase 4: Historical Backfill — Options ⚠️ PARTIAL

- NQ.OPT BBO-1s batch job completed (GLBX-20260402-S4EMNFKM8K)
- **Downloaded** 313 daily `.dbn.zst` files (53 GB total on disk)
- **NOT ingested** — files sitting in `data/raw/backfill/nq_opt/`
- NQ.OPT Definition — not downloaded (needed for IV/GEX features)
- **Enables**: `iv_surface` (13 features) + `dealer_gex` (4 features) once ingested

### Phase 5: Live Streaming ✅

- Implemented `LiveStreamer` class subscribing to NQ.FUT tbbo, NQ.FUT bbo_1s, NQ.OPT bbo_1s, NQ.OPT definition, NQ.FUT statistics
- Deployed as systemd service (currently **inactive**)
- Raw DBN archive saving to `data/raw/live/YYYY-MM-DD/`
- Live data present for: 2026-04-02, 2026-04-05, 2026-04-06

### Phase 6: Ingestion Pipeline ✅

- Built `src/ingestion/loader.py` — DBN → DataFrame → batch INSERT / COPY
- Dedup via `ON CONFLICT DO NOTHING`
- Built `src/ingestion/quality.py` — gap detection, record counts
- Scripts: `databento_backfill.py`, `databento_cost_check.py`, `databento_verify.py`

### Phase 7: Feature Engineering ✅ (v2: 12 modules → v3: 23 modules)

- **v2 (12 modules, 138 features)**: book_pressure, order_flow, microstructure, wavelets, candle_structure, time_context, vwap, cross_asset, macro_sentiment, equity_context, options_surface (disabled), daily_context (disabled)
- **v3 (23 modules, ~182 features)**: Added trade_location, same_side, large_trade_cvd, trade_arrival, realized_vol, sub_bar_dynamics, volume_profile, dealer_gex, economic_calendar, hawkes_clustering, higher_timeframe; updated all existing modules with new features; removed equity_context and macro_sentiment from active pipeline
- Engine rewritten with 21 active modules, 08:00–11:00 CST time filter
- Feature resolution changed from 10s to 5s bars, sequence length 64 → 128

### Phase 8: ML Training Pipeline ✅

- Built config, labels, data_pipeline, attention_lstm, trainer, evaluate modules
- 3-class labeling (BIG_DOWN / CONSOLIDATION / BIG_UP) with 50pt threshold
- Attention-LSTM architecture: (128, 182) input, 4-head attention
- Deployed trainer service + timer (weekly Sunday 12:00 UTC)

### Phase 9: Cross-Asset — ES Futures ✅

- Downloaded ES.FUT TBBO (1 year, 2025-04-01 → 2026-04-02)
- Ingested into `databento.tbbo` alongside NQ data
- Built `cross_asset.py` — 12 features (NQ/ES correlation, lead-lag, spread, beta)
- Retrained model v002 with cross-asset features

### Phase 10: Hyperparameter Optimization ✅

- Built `scripts/hpo_search.py` with Optuna integration
- 9 searchable feature groups, architecture + training + label parameters
- Optuna DB at `data/optuna.db` (study: `nq_momentum_v1`)
- 11 HPO trials completed

### Phase 11: Cross-Asset — Equities & Macro ✅

- Downloaded DBEQ.BASIC OHLCV-1m for NVDA, TSLA, XLK, SMH, VIXY ($1.45 cost)
- Created `databento.equity_ohlcv` hypertable (469K rows)
- Built `macro_sentiment.py` (10 features) + `equity_context.py` (23 features)
- Trained model v003 (138 features, 12 groups)
- **Note**: Equity/macro modules later removed from v3 active pipeline (daily resolution too coarse for 5s bars)

---

## Post-v2 Phases (implemented after Architecture v2 was written)

### Phase 12A: Architecture v3 — Resolution & Feature Overhaul ✅

_Documented in [DATABENTO_VM_ARCHITECTURE_v3.md](DATABENTO_VM_ARCHITECTURE_v3.md)_

- Changed feature timestep from 10s → **5s**
- Changed sequence length from 64 → **128** bars (same 10.7 min real-time coverage)
- Fixed forward_windows bug: `[5,10,15,30]` bars → `[60,120,180,360]` bars (now 5/10/15/30 min)
- Added 08:00–11:00 CST session filter
- Created `src/common/bar_windows.py` — central timestep→bars helper
- **11 new feature modules**: trade_location, same_side, large_trade_cvd, trade_arrival, realized_vol, sub_bar_dynamics, volume_profile, dealer_gex, economic_calendar, hawkes_clustering, higher_timeframe
- Updated all existing modules with new features (spread_vol, fleeting liquidity, Amihud, VWAP slope, 25d risk reversal, etc.)
- Feature count: 138 → **~182**

### Phase 12B: Blueprint Gap Analysis ✅

_Documented in [blueprint vs implementation.md](blueprint%20vs%20implementation.md)_

- Identified and categorized 23 gaps between the original blueprint and implementation
- Tier 1 (7 critical gaps): forward window fix, close session, confidence threshold, probability calibration, walk-forward validation, realized vol, adaptive labels
- Tier 2 (7 high-impact gaps): trade location, fleeting liquidity, Amihud, same-side runs, large-trade CVD, volume profile, multi-resolution
- Tier 3 (9 medium-impact gaps): XGBoost baseline, SHAP, economic calendar, trade arrival, VWAP slope, 30m OR, spread vol, book pressure velocity, multi-horizon ensemble
- Most Tier 1–2 gaps addressed in Phase 12A feature overhaul

### Phase 12C: NQ.FUT Definitions & Statistics Backfill ✅

- Downloaded NQ.FUT definition (337 KB, 2025-04-06 → 2026-04-06)
- Downloaded NQ.FUT statistics (22 MB, 2025-04-06 → 2026-04-06)
- Ingested into PostgreSQL: 1,197 definitions rows + 1,028,670 statistics rows
- 42 distinct NQ symbols (16 outrights + 26 spreads) across the period
- 10 stat_types present (settlement, high, low, OI, etc.)
- **Enables**: `daily_context` (4 features) module activation

### Phase 12D: NQ.OPT BBO-1s Batch Download ⚠️ Downloaded, not ingested

- Batch job `GLBX-20260402-S4EMNFKM8K`: NQ.OPT BBO-1s, 2025-04-01 → 2026-04-02
- 313 daily `.dbn.zst` files downloaded (53 GB on disk)
- Contains NQ options quotes: calls, puts, and UD spreads across all strikes/expirations
- **NOT YET INGESTED** into `databento.bbo_1s`
- **Enables**: `iv_surface` (13 features) + `dealer_gex` (4 features) once ingested

---

## Current State Summary (2026-04-06)

### Database

| Table | Rows | Date Range | Status |
|-------|------|-----------|--------|
| `databento.tbbo` | 207,993,433 | 2025-04-01 → 2026-04-06 | ✅ NQ + ES futures, live updating |
| `databento.bbo_1s` | 18,478,651 | 2025-04-06 → 2026-04-06 | ✅ NQ futures BBO from live stream |
| `databento.definitions` | 1,197 | 2025-04-06 → 2026-04-05 | ✅ NQ futures contracts |
| `databento.statistics` | 1,028,670 | 2025-04-06 → 2026-04-05 | ✅ NQ settlement/hi/lo/OI |
| `databento.equity_ohlcv` | 469,178 | 2025-04-01 → 2026-04-02 | ⚠️ Static (no live feed) |
| `tbbo_1min` (continuous agg) | — | derived from tbbo | ✅ Auto-refreshing |

### Raw Data on Disk (500 GB data volume, 73 GB used / 394 GB free)

| Path | Size | Contents |
|------|------|----------|
| `data/raw/backfill/nq_fut/` | 2.8 GB | NQ.FUT tbbo, bbo-1s, definitions, statistics |
| `data/raw/backfill/nq_opt/` | 53 GB | NQ.OPT BBO-1s batch (313 daily files) — **not ingested** |
| `data/raw/live/` | 584 MB | Live DBN archives (3 days) |

### Models

| Checkpoint | Features | Description |
|------------|----------|-------------|
| `model_v001_baseline` | 93 | Base model, no cross-asset |
| `model_best_v001` | — | Best from initial HPO |
| `model_best_v002` | ~105 | + ES cross-asset features |
| `model_best_v003` | 138 | + equity/macro (v2 feature set) |
| `hpo/` | — | 11 Optuna trial checkpoints |

> **Note**: No v3 model trained yet (182 features). v003 used the v2 feature set.

### Services (all currently **inactive**)

| Service | Purpose | Schedule |
|---------|---------|----------|
| `vtech-live-streamer` | Databento → TimescaleDB (NQ + options) | 24/7 |
| `vtech-feature-builder` | engine.py → Parquet (5s bars, 8–11am) | Daily 21:05 UTC |
| `vtech-trainer` | Train Attention-LSTM | Weekly Sun 12:00 UTC |

---

## TODO: Remaining Roadmap

### Phase 13: Options Data Ingestion ❌ NOT STARTED

> **Priority**: HIGH — unblocks 17 features (iv_surface + dealer_gex)

- [ ] Build (or extend) ingestion pipeline for NQ.OPT BBO-1s batch files
  - 313 files × avg 170 MB each = 53 GB compressed
  - Need chunked/streaming ingestion (files are too large for in-memory `to_df()`)
  - Columns: bid_px, ask_px, bid_sz, ask_sz per option strike
- [ ] Download NQ.OPT definition data (instrument specs — strike, expiry, put/call flag)
  - Required to map instrument_id → strike/expiry for IV calculation
- [ ] Ingest NQ.OPT definitions into `databento.definitions`
- [ ] Ingest NQ.OPT BBO-1s into `databento.bbo_1s` (expect billions of rows)
  - May need partitioning strategy or separate table for options vs futures
  - Estimate storage: ~50–100 GB uncompressed in TimescaleDB
- [ ] Verify `iv_surface` and `dealer_gex` modules activate with options data present
- [ ] Assess disk capacity (currently 394 GB free — should be sufficient)

### Phase 14: Daily Context & Statistics Activation ❌ NOT STARTED

> **Priority**: HIGH — data already ingested, just need to verify feature module works

- [ ] Test `daily_context.py` against the now-populated `databento.statistics` table
- [ ] Verify 4 `dc_` features appear in Parquet output
- [ ] Wire into v3 engine if not already connected (check engine.py module list)
- [ ] Run feature builder to confirm end-to-end

### Phase 15: V3 Model Training ❌ NOT STARTED

> **Priority**: HIGH — no model has been trained on the v3 feature set

- [ ] Rebuild Parquet feature cache with v3 engine (5s bars, 182 features, 08–11am filter)
  - Needs several weeks of accumulated data for meaningful training
- [ ] Train first v3 model: Attention-LSTM (128, 182)
- [ ] Compare v3 vs v2 model performance
- [ ] Run HPO with v3 feature set (update `hpo_search.py` for new feature groups)
- [ ] Save as `model_best_v004`

### Phase 16: Validation & Diagnostics ❌ NOT STARTED

> **Priority**: HIGH — current models predict 100% CONSOLIDATION (trivial majority class)

- [ ] **Walk-forward validation**: expanding/sliding window retraining (monthly retrain, test on next month)
- [ ] **XGBoost baseline**: train a non-sequential model on same features for diagnostic comparison
- [ ] **SHAP feature importance**: identify which feature groups actually drive predictions
  - Prune bottom 20–30% features → reduce from ~182 to ~140
- [ ] **Probability calibration**: Platt scaling or isotonic regression on validation set
- [ ] **Confidence thresholding**: only emit predictions when softmax > configurable threshold (e.g., 0.85)
  - Target: 1–2 high-confidence trades per day
- [ ] **Adaptive label threshold**: scale 50pt threshold by ATR (fixed points → relative to volatility)
- [ ] Debug why all models predict CONSOLIDATION (likely class imbalance + insufficient non-consolidation samples)

### Phase 17: Live Data Pipeline Gaps ❌ NOT STARTED

> **Priority**: MEDIUM — features decay without fresh data

- [ ] **Restart live streamer service** — currently inactive, NQ live data stopped
- [ ] **Live ES futures pipeline**: add ES.FUT subscription to live_streamer (or daily batch download)
  - Without this, `cross_asset` features stale after 2026-04-02
- [ ] **Live equity pipeline**: daily DBEQ.BASIC OHLCV fetch ($0.006/day)
  - Only needed if equity_context / macro_sentiment modules re-enabled
  - Currently removed from v3 pipeline — low priority unless restored
- [ ] **BBO-1s backfill**: the backfill table only has data from live stream (2025-04-06 onward)
  - Consider whether historical BBO-1s backfill (streaming API) is needed for deeper training

### Phase 18: Inference Service ❌ NOT STARTED

> **Priority**: MEDIUM — trained model never produces live predictions

- [ ] Build `src/inference/predictor.py` — load model, run on latest feature row
- [ ] Build `src/inference/confidence_filter.py` — apply calibrated probability threshold
- [ ] Create inference systemd service (runs after feature-builder, or in real-time loop)
- [ ] Define output format: prediction + confidence + timestamp + metadata
- [ ] Output to: log file / database table / webhook (TBD)
- [ ] Paper-trade mode: log predictions without execution

### Phase 19: Production Hardening ❌ NOT STARTED

> **Priority**: LOW (until model shows real signal)

- [ ] **Monitoring**: health check cron, systemd watchdog, service status dashboard
- [ ] **Alerting**: email/webhook on service failure, data gap, or disk >80%
- [ ] **Automated backups**: `pg_dump` cron → cloud storage (Oracle Object Storage or S3)
- [ ] **Disk management**: automated cleanup of old live raw archives
- [ ] **Log rotation**: configure journald limits for service logs
- [ ] **Documentation**: operational runbook for common tasks (restart services, re-ingest, retrain)

### Phase 20: Advanced Model Improvements ❌ NOT STARTED

> **Priority**: LOW (research / iteration after core pipeline is validated)

- [ ] **Multi-horizon ensemble**: separate models per forward window → fuse predictions
- [ ] **Close session training**: add 14:00–15:00 CT window alongside morning session
- [ ] **Multi-resolution features**: 1s microstructure path alongside 5s feature bars
- [ ] **Transformer architecture**: replace LSTM with temporal transformer (if LSTM proves insufficient)
- [ ] **Online learning**: incremental model updates from daily data (vs weekly full retrain)
- [ ] **Regime detection**: unsupervised clustering to identify market regimes → conditional models
