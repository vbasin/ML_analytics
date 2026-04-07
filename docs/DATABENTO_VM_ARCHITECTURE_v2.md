# Databento VM Architecture v2 — NQ Momentum ML Platform

> **Updated**: April 7, 2026 — Added /dev/sdb data disk (500 GB), fixed ingest_file() dispatcher, safe-delete guard.
>
> Purpose-built VM for Databento data acquisition, storage, feature engineering, and ML training for NQ 50-point momentum prediction.

---

## Changelog from v1

| Area | v1 (Original Plan) | v2 (Actual as of 2026-04-06) |
|---|---|---|
| **VM specs** | 4–8 vCPU, 32 GB RAM, 500 GB NVMe | 4 vCPU, 31 GB RAM, 49 GB root + 500 GB data disk (`/dev/sdb` → `/opt/vtech/data`) |
| **OS** | Ubuntu 22.04, Python 3.12 via deadsnakes | Ubuntu 22.04.5, Python 3.10 system + Python 3.12.13 in venv |
| **PostgreSQL** | PostgreSQL 16 + TimescaleDB 2.x | PostgreSQL 18.3 + TimescaleDB 2.26.1 |
| **Venv path** | `/opt/vtech/venv` | `/opt/vtech/venv` (note: systemd units reference `.venv`) |
| **Feature modules** | 10 planned | 12 implemented (+macro_sentiment, +equity_context) |
| **Feature count** | ~93 implied | 138 features across 10 active groups |
| **Cross-asset** | ES.FUT only (Phase 9) | ES.FUT + VIXY + NVDA/TSLA/XLK/SMH (Phase 11) |
| **Database tables** | 4 hypertables + 1 continuous agg | 5 hypertables + 1 continuous aggregate (+equity_ohlcv) |
| **Ingestion** | `loader.py`, `transforms.py`, `dedup.py` | `loader.py`, `quality.py` only |
| **Models dir** | `attention_lstm.py` + `registry.py` | `attention_lstm.py` only (no registry) |
| **Symbology** | `symbology.py` planned | Not implemented (parent symbology used directly) |
| **Services** | 3 systemd units | 5 systemd units (+trainer.service, +trainer.timer) |
| **Feature timer** | 22:00 UTC | 21:05 UTC (17:05 ET) |
| **ML framework** | TensorFlow 2.15 | TensorFlow 2.21.0 / Keras 3.13.2 |
| **HPO** | Not in original plan | Optuna 4.8.0 with `hpo_search.py` |
| **Model versions** | Single model path | 4 checkpoints: baseline, v001, v002, v003 + 11 HPO trials |
| **Docker** | "Optional for ancillary" | Not used |
| **Rust/dbn-cli** | Planned | Not installed (Python SDK sufficient) |
| **Monitoring** | Prometheus + health check | **Gap**: not deployed |
| **Backups** | pg_dump + rsync | **Gap**: not automated |
| **Live equity pipeline** | Not planned | **Gap**: equity data is one-time backfill only |
| **Inference service** | Implied | **Gap**: not implemented |

---

## Table of Contents

1. [Design Principles](#1-design-principles)
2. [VM Specifications](#2-vm-specifications)
3. [OS & Runtime Stack](#3-os--runtime-stack)
4. [Directory Layout](#4-directory-layout)
5. [Storage Architecture](#5-storage-architecture)
6. [Data Acquisition Layer](#6-data-acquisition-layer)
7. [Database Schema Design](#7-database-schema-design)
8. [Live Streaming Service](#8-live-streaming-service)
9. [Feature Engineering Pipeline](#9-feature-engineering-pipeline)
10. [ML Training & Inference](#10-ml-training--inference)
11. [Service Orchestration](#11-service-orchestration)
12. [Configuration & Secrets](#12-configuration--secrets)
13. [Monitoring & Observability](#13-monitoring--observability)
14. [Network & Security](#14-network--security)
15. [Backup & Recovery](#15-backup--recovery)
16. [Implementation Sequence](#16-implementation-sequence)
17. [Known Issues & Gaps](#17-known-issues--gaps)

---

## 1. Design Principles

| Principle | Rationale | Status |
|---|---|---|
| **DBN-native storage** | Keep raw `.dbn.zst` files as immutable archive | ✅ Raw files retained on 500 GB data disk (`/dev/sdb`); only deleted via `--cleanup` after confirmed ingestion |
| **Three-tier data** | Raw DBN → TimescaleDB → Parquet (ML-ready) | ✅ Implemented (Tier 1 is transient, Tier 2 is primary, Tier 3 is feature cache) |
| **Schema-per-concern** | Separate DB schemas per concern | ⚠️ Single `databento` schema for all tables (adequate at scale) |
| **Batch for backfill, live for incremental** | Databento batch for >5GB; live for real-time | ✅ Historical API for backfill, Live API for NQ streaming |
| **No Docker** | systemd services avoid container I/O overhead | ✅ All services are systemd-managed |
| **Environment-driven config** | All secrets via `.env` | ✅ `/opt/vtech/.env` |
| **Idempotent ingestion** | Safe to re-run | ✅ `ON CONFLICT DO NOTHING` in all inserts |

---

## 2. VM Specifications

### Actual (Deployed)

| Resource | Spec | Notes |
|---|---|---|
| **Hostname** | `vtech-chi-vm1-2025` | Oracle Cloud |
| **CPU** | 4 vCPU (AMD EPYC) | Sufficient for feature compute + CPU training |
| **RAM** | 31 GB | ~1.5 GB used, ~26 GB available |
| **Root disk** | 49 GB SSD (`/dev/sda1`) | 31% used (15 GB), 34 GB free |
| **Data disk** | 500 GB Block Volume (`/dev/sdb` → `/opt/vtech/data`) | <1% used (510 MB), 466 GB free |
| **GPU** | None | CPU-only training (~5 min/model) |
| **Kernel** | `6.8.0-1047-oracle` | |

### Storage Layout

The root disk (`/dev/sda1`, 49 GB) holds the OS, packages, Python venv, and PostgreSQL data directory.
A separate 500 GB block volume (`/dev/sdb`) is mounted at `/opt/vtech/data` and holds all raw DBN archives, Parquet feature cache, and model checkpoints.

- **Raw DBN files are retained** after ingestion (466 GB available for archives)
- NQ.OPT BBO-1s batch (~56 GB compressed) fits comfortably on the data disk
- TimescaleDB compression is aggressive (tbbo: 3.1 GB for 124M rows)

---

## 3. OS & Runtime Stack

```
Ubuntu 22.04.5 LTS (kernel 6.8.0-1047-oracle)
├── Python 3.10.12 (system /usr/bin/python3)
├── Python 3.12.13 (venv: /opt/vtech/venv)
│   └── See dependency table below
├── PostgreSQL 18.3 + TimescaleDB 2.26.1
├── systemd (5 service/timer units)
└── No Rust toolchain, no CUDA/GPU stack
```

### Core Python Dependencies

| Package | Version | Purpose |
|---|---|---|
| databento | 0.74.1 | Historical + Live client |
| databento-dbn | 0.52.1 | DBN file format library |
| tensorflow | 2.21.0 | ML training backend |
| keras | 3.13.2 | High-level ML API (Keras 3) |
| optuna | 4.8.0 | Hyperparameter optimization |
| psycopg | 3.3.3 | PostgreSQL driver (psycopg3) |
| psycopg-pool | 3.3.0 | Connection pooling |
| pandas | 2.3.3 | DataFrame operations |
| numpy | 2.4.4 | Numerical compute |
| scipy | 1.17.1 | Signal processing |
| scikit-learn | 1.8.0 | Preprocessing, metrics |
| PyWavelets | 1.9.0 | Wavelet transforms |
| structlog | 25.5.0 | Structured logging |
| python-dotenv | 1.2.2 | `.env` file loading |
| pyarrow | (installed) | Parquet I/O |

---

## 4. Directory Layout

```
/opt/vtech/                              # Application root
├── venv/                                # Python 3.12 virtual environment
├── .env                                 # Environment variables (chmod 600)
├── ingest_tbbo_chunked.py               # Root-level ingest script (legacy)
├── src/
│   ├── acquisition/                     # Data download & streaming
│   │   ├── __init__.py
│   │   ├── historical.py                # Batch backfill (Historical API)
│   │   ├── live_streamer.py             # Real-time Live API (NQ only)
│   │   └── schemas.py                   # Column mappings, BACKFILL_JOBS
│   ├── ingestion/                       # DBN → TimescaleDB loading
│   │   ├── __init__.py
│   │   ├── loader.py                    # DBN → DataFrame → batch INSERT
│   │   └── quality.py                   # Gap detection, record counts
│   ├── features/                        # Feature engineering (12 modules + engine)
│   │   ├── __init__.py
│   │   ├── engine.py                    # Orchestrator — loads data, runs all modules
│   │   ├── book_pressure.py             # bp_ prefix — 14 features
│   │   ├── order_flow.py                # of_ prefix — 18 features
│   │   ├── options_surface.py           # iv_ prefix — disabled (no options data)
│   │   ├── daily_context.py             # dc_ prefix — disabled (no statistics data)
│   │   ├── microstructure.py            # ms_ prefix — 4 features
│   │   ├── wavelets.py                  # wv_ prefix — 14 features
│   │   ├── candle_structure.py          # cs_ prefix — 24 features
│   │   ├── time_context.py              # tc_ prefix — 12 features
│   │   ├── vwap.py                      # vw_ prefix — 7 features
│   │   ├── cross_asset.py               # ca_ prefix — 12 features
│   │   ├── macro_sentiment.py           # mx_ prefix — 10 features
│   │   └── equity_context.py            # eq_ prefix — 23 features
│   ├── training/                        # ML training pipeline
│   │   ├── __init__.py
│   │   ├── config.py                    # MLConfig dataclass hierarchy
│   │   ├── data_pipeline.py             # Parquet load, split, scale, sequences
│   │   ├── labels.py                    # 3-class / 5-class labeling
│   │   ├── trainer.py                   # 10-step training orchestrator
│   │   └── evaluate.py                  # Metrics, confusion matrix
│   ├── models/
│   │   ├── __init__.py
│   │   └── attention_lstm.py            # Multi-head attention + stacked LSTM
│   └── common/
│       ├── __init__.py
│       ├── config.py                    # AppConfig frozen dataclass
│       ├── db.py                        # psycopg_pool singleton
│       └── logging.py                   # structlog JSON/console setup
├── scripts/
│   ├── databento_backfill.py            # Download + ingest orchestrator
│   ├── databento_cost_check.py          # Cost estimation
│   ├── databento_verify.py              # Data quality verification
│   ├── hpo_search.py                    # Optuna HPO search
│   └── ingest_tbbo_chunked.py           # Chunked TBBO ingestion
├── data/
│   ├── raw/
│   │   ├── backfill/                    # Dirs exist but empty (files deleted post-ingest)
│   │   └── live/                        # Live stream DBN captures (daily rotation)
│   │       └── YYYY-MM-DD/
│   ├── parquet/                         # Feature cache
│   │   └── features_YYYYMMDD.parquet
│   ├── checkpoints/                     # Model checkpoints
│   │   ├── model_v001_baseline/
│   │   ├── model_best_v001/
│   │   ├── model_best_v002/
│   │   ├── model_best_v003/
│   │   └── hpo/                         # Trial checkpoints
│   └── optuna.db                        # Optuna SQLite study storage
├── logs/
├── systemd/                             # 5 service/timer unit files
└── docs/
    └── DATABENTO_VM_ARCHITECTURE.md
```

### Files Not Implemented (vs v1 Plan)

| Planned File | Status | Reason |
|---|---|---|
| `src/acquisition/symbology.py` | Not needed | Parent symbology (`NQ.FUT`, `ES.FUT`) works directly |
| `src/ingestion/transforms.py` | Not needed | Transforms are inline in `loader.py` |
| `src/ingestion/dedup.py` | Not needed | `ON CONFLICT DO NOTHING` in SQL sufficient |
| `src/models/registry.py` | Not needed | Manual directory-based versioning |
| `scripts/migrate_db.py` | Not created | Schema managed via ad-hoc SQL |


---

## 5. Storage Architecture

### Three-Tier Model (Modified)

```
┌──────────────────────────────────────────────────────────────┐
│            TIER 1: Raw DBN Archive (PERSISTENT)              │
│  /opt/vtech/data/raw/**/*.dbn.zst  (on /dev/sdb, 500 GB)     │
│  Downloaded during backfill, RETAINED after ingestion         │
│  Deleted only via --cleanup flag after confirmed ingestion    │
│  Live streams saved to data/raw/live/YYYY-MM-DD/             │
│  Retention: indefinite (466 GB free)                         │
└──────────────────────┬───────────────────────────────────────┘
                       │ DBNStore.to_df() → transform → INSERT
                       ▼
┌──────────────────────────────────────────────────────────────┐
│             TIER 2: TimescaleDB (PRIMARY STORE)              │
│  PostgreSQL 18.3 + TimescaleDB 2.26.1                        │
│  5 hypertables + 1 continuous aggregate                      │
│  Compression enabled on: tbbo, bbo_1s, equity_ohlcv          │
│  Total DB size: ~3.2 GB (dominated by tbbo at 3.1 GB)       │
│  Retention: indefinite (compression makes this feasible)     │
└──────────────────────┬───────────────────────────────────────┘
                       │ Feature engine → to_parquet()
                       ▼
┌──────────────────────────────────────────────────────────────┐
│           TIER 3: Parquet Feature Cache (ML-Ready)           │
│  /opt/vtech/data/parquet/features_*.parquet                  │
│  ~1 MB per trading day (138 features × ~8,500 rows)         │
│  Retention: regenerable (delete & recompute from Tier 2)     │
└──────────────────────────────────────────────────────────────┘
```

### Actual Storage Budget (as of 2026-04-06)

| Component | On-Disk Size | Rows | Notes |
|---|---|---|---|
| `databento.tbbo` | 3,136 MB | 123,945,223 | Compressed, 317 chunks |
| `databento.bbo_1s` | 13 MB | 54,635 | Compressed, 9 chunks |
| `databento.equity_ohlcv` | 21 MB | 469,178 | Compressed, 54 chunks |
| `databento.definitions` | 48 KB | 0 | Empty (schema exists) |
| `databento.statistics` | 24 KB | 0 | Empty (schema exists) |
| `databento.tbbo_1min` | ~0 | — | Continuous aggregate (auto-refreshed) |
| Live DBN files | ~450 MB | — | 3 days of captures |
| Parquet cache | ~2.5 MB | — | 4 feature files |
| Model checkpoints | ~12 MB | — | 4 models + 11 HPO trials |
| Optuna DB | 135 KB | — | SQLite |
| **Total** | **~3.6 GB** | | **<1% of 500 GB data disk + 31% of 49 GB root** |

### TimescaleDB Configuration

No custom tuning has been applied (using PostgreSQL defaults). The v1 plan specified:

```ini
# v1 planned (NOT applied):
# shared_buffers = 8GB, effective_cache_size = 24GB, work_mem = 256MB
# maintenance_work_mem = 2GB, wal_buffers = 64MB

# Actual: PostgreSQL 18.3 defaults
# Adequate at current data volume (~3 GB)
```

---

## 6. Data Acquisition Layer

### 6.1 Dataset Coverage

| Dataset | Symbols | Schema | Rows | Date Range | Status |
|---|---|---|---|---|---|
| `GLBX.MDP3` | NQ.FUT (parent) | `tbbo` | 123.9M | 2025-04-01 → 2026-04-06 | ✅ Backfilled + live streaming |
| `GLBX.MDP3` | NQ.FUT | `bbo-1s` | 54.6K | — | ✅ Live streaming (sparse) |
| `GLBX.MDP3` | NQ.OPT | `bbo-1s` | 0 | — | ❌ Not backfilled (cost/disk) |
| `GLBX.MDP3` | NQ.OPT | `definition` | 0 | — | ❌ Not backfilled |
| `GLBX.MDP3` | NQ.FUT | `statistics` | 0 | — | ❌ Not backfilled |
| `GLBX.MDP3` | ES.FUT | `tbbo` | (in tbbo) | 2025-04-01 → 2026-04-02 | ✅ Backfilled (within tbbo table) |
| `DBEQ.BASIC` | NVDA,TSLA,XLK,SMH,VIXY | `ohlcv-1m` | 469K | 2025-04-01 → 2026-04-02 | ✅ One-time backfill |

### 6.2 Historical Backfill

```python
# src/acquisition/historical.py
# Functions: cost_check(), download_range(), submit_batch(), download_batch()
# All hardcoded to DATASET = "GLBX.MDP3" (from schemas.py)
# Equity data (DBEQ.BASIC) was downloaded via ad-hoc script, not through this module

# BACKFILL_JOBS (from schemas.py):
# P0: NQ.FUT tbbo, NQ.FUT bbo-1s (parent)
# P1: NQ.OPT bbo-1s, NQ.OPT definition (parent)
# P2: NQ.FUT statistics, ES.FUT tbbo (parent)
```

### 6.3 Cost Summary

| Download | Cost | Date |
|---|---|---|
| NQ.FUT TBBO (1 year) | ~$5-10 | Phase 6 |
| NQ.OPT data | ~$15-30 est. | Not downloaded |
| ES.FUT TBBO (1 year) | ~$5-10 | Phase 11 |
| DBEQ.BASIC equities (1 year, 5 symbols) | $1.45 | Phase 11.3 |
| NQ live streaming | Included in subscription | Ongoing |

### 6.4 Contract Stitching

Databento's parent symbology (`NQ.FUT`, `ES.FUT`) with `stype_in="parent"` handles contract rolling automatically. The v1 plan discussed continuous symbology (`NQ.c.0`, `NQ.v.0`) but parent symbology proved sufficient — it subscribes to all active contracts, and `engine.py` selects the most-traded one at query time.


---

## 7. Database Schema Design

### 7.1 Schema Organization

```sql
-- Single schema (v1 planned separate schemas per concern)
CREATE SCHEMA IF NOT EXISTS databento;
CREATE EXTENSION IF NOT EXISTS timescaledb;
```

### 7.2 Actual Tables

```sql
-- ============================================================
-- TBBO: Trades with Best Bid/Offer (NQ.FUT + ES.FUT)
-- 123.9M rows, 3.1 GB, 317 chunks, compression ON
-- ============================================================
CREATE TABLE databento.tbbo (
    ts_event       TIMESTAMPTZ    NOT NULL,
    ts_recv        TIMESTAMPTZ    NOT NULL,
    instrument_id  INTEGER        NOT NULL,
    symbol         TEXT           NOT NULL,   -- e.g., NQM5, ESZ5
    price          DOUBLE PRECISION NOT NULL,
    size           INTEGER        NOT NULL,
    side           CHAR(1),                   -- 'A'sk, 'B'id, 'N'one
    action         CHAR(1),                   -- 'T'rade
    flags          SMALLINT       DEFAULT 0,
    sequence       BIGINT,
    bid_px         DOUBLE PRECISION,
    ask_px         DOUBLE PRECISION,
    bid_sz         INTEGER,
    ask_sz         INTEGER,
    bid_ct         INTEGER,
    ask_ct         INTEGER,
    ts_in_delta    INTEGER,                   -- Matching engine latency (ns)
    dataset        TEXT           DEFAULT 'GLBX.MDP3',
    UNIQUE (instrument_id, ts_event, sequence)
);

SELECT create_hypertable('databento.tbbo', 'ts_event',
    chunk_time_interval => INTERVAL '1 day');

-- Compression: enabled, policy = 30 days
ALTER TABLE databento.tbbo SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'instrument_id',
    timescaledb.compress_orderby = 'ts_event ASC'
);
SELECT add_compression_policy('databento.tbbo', INTERVAL '30 days');


-- ============================================================
-- BBO_1S: Best Bid/Offer sampled every 1 second
-- 54.6K rows, 13 MB, 9 chunks, compression ON
-- ============================================================
CREATE TABLE databento.bbo_1s (
    ts_event       TIMESTAMPTZ    NOT NULL,
    instrument_id  INTEGER        NOT NULL,
    symbol         TEXT           NOT NULL,
    open           DOUBLE PRECISION,
    high           DOUBLE PRECISION,
    low            DOUBLE PRECISION,
    close          DOUBLE PRECISION,
    volume         BIGINT,
    bid_px         DOUBLE PRECISION,
    ask_px         DOUBLE PRECISION,
    bid_sz         INTEGER,
    ask_sz         INTEGER,
    spread         DOUBLE PRECISION GENERATED ALWAYS AS (ask_px - bid_px) STORED,
    dataset        TEXT           DEFAULT 'GLBX.MDP3',
    UNIQUE (instrument_id, ts_event)
);

SELECT create_hypertable('databento.bbo_1s', 'ts_event',
    chunk_time_interval => INTERVAL '1 day');

-- Note: live_streamer.py inserts only bid_px/ask_px/bid_sz/ask_sz columns
-- (OHLCV fields from bbo-1s schema are not populated by live stream)


-- ============================================================
-- EQUITY_OHLCV: Equity/ETF 1-minute bars (NEW in v2)
-- 469K rows, 21 MB, 54 chunks, compression ON
-- Source: DBEQ.BASIC ohlcv-1m
-- ============================================================
CREATE TABLE databento.equity_ohlcv (
    ts_event       TIMESTAMPTZ    NOT NULL,
    symbol         TEXT           NOT NULL,   -- NVDA, TSLA, XLK, SMH, VIXY
    open           DOUBLE PRECISION,
    high           DOUBLE PRECISION,
    low            DOUBLE PRECISION,
    close          DOUBLE PRECISION,
    volume         BIGINT,
    UNIQUE (ts_event, symbol)
);

SELECT create_hypertable('databento.equity_ohlcv', 'ts_event',
    chunk_time_interval => INTERVAL '7 days');

ALTER TABLE databento.equity_ohlcv SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol',
    timescaledb.compress_orderby = 'ts_event ASC'
);


-- ============================================================
-- DEFINITIONS: Instrument definitions (schema exists, no data)
-- ============================================================
CREATE TABLE databento.definitions (
    ts_event           TIMESTAMPTZ  NOT NULL,
    instrument_id      INTEGER      NOT NULL,
    symbol             TEXT         NOT NULL,
    instrument_class   CHAR(1),
    strike_price       DOUBLE PRECISION,
    expiration         TIMESTAMPTZ,
    underlying         TEXT,
    exchange           TEXT,
    currency           TEXT         DEFAULT 'USD',
    min_price_increment DOUBLE PRECISION,
    multiplier         DOUBLE PRECISION,
    trading_reference_price DOUBLE PRECISION,
    settlement_price   DOUBLE PRECISION,
    open_interest      BIGINT,
    dataset            TEXT         DEFAULT 'GLBX.MDP3',
    raw_record         JSONB,
    UNIQUE (instrument_id, ts_event)
);

SELECT create_hypertable('databento.definitions', 'ts_event',
    chunk_time_interval => INTERVAL '1 week');


-- ============================================================
-- STATISTICS: Daily settlement, hi/lo, volume, OI (no data)
-- ============================================================
CREATE TABLE databento.statistics (
    ts_event       TIMESTAMPTZ    NOT NULL,
    ts_recv        TIMESTAMPTZ    NOT NULL,
    instrument_id  INTEGER        NOT NULL,
    symbol         TEXT           NOT NULL,
    stat_type      SMALLINT       NOT NULL,
    price          DOUBLE PRECISION,
    quantity       BIGINT,
    sequence       BIGINT,
    ts_ref         TIMESTAMPTZ,
    update_action  SMALLINT,
    stat_flags     SMALLINT,
    dataset        TEXT           DEFAULT 'GLBX.MDP3',
    UNIQUE (instrument_id, ts_event, stat_type, sequence)
);

SELECT create_hypertable('databento.statistics', 'ts_event',
    chunk_time_interval => INTERVAL '1 week');


-- ============================================================
-- Continuous Aggregate: 1-minute OHLCV from TBBO
-- ============================================================
CREATE MATERIALIZED VIEW databento.tbbo_1min
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', ts_event) AS bucket,
    instrument_id,
    symbol,
    first(price, ts_event)  AS open,
    max(price)              AS high,
    min(price)              AS low,
    last(price, ts_event)   AS close,
    sum(size)               AS volume,
    sum(CASE WHEN side = 'B' THEN size ELSE 0 END) AS buy_volume,
    sum(CASE WHEN side = 'A' THEN size ELSE 0 END) AS sell_volume,
    count(*)                AS trade_count,
    last(bid_px, ts_event)  AS bid_px,
    last(ask_px, ts_event)  AS ask_px,
    last(bid_sz, ts_event)  AS bid_sz,
    last(ask_sz, ts_event)  AS ask_sz
FROM databento.tbbo
GROUP BY bucket, instrument_id, symbol
WITH NO DATA;

SELECT add_continuous_aggregate_policy('databento.tbbo_1min',
    start_offset    => INTERVAL '3 hours',
    end_offset      => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute'
);
```

### 7.3 Hypertable Summary

| Table | Chunks | Compression | Size | Rows |
|---|---|---|---|---|
| `tbbo` | 317 | ✅ Enabled (30d policy) | 3,136 MB | 123.9M |
| `bbo_1s` | 9 | ✅ Enabled | 13 MB | 54.6K |
| `equity_ohlcv` | 54 | ✅ Enabled | 21 MB | 469K |
| `definitions` | 0 | ❌ | 48 KB | 0 |
| `statistics` | 0 | ❌ | 24 KB | 0 |
| `tbbo_1min` (cont. agg) | — | — | ~0 | Auto-refreshed |


---

## 8. Live Streaming Service

### 8.1 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Live Streaming Service                      │
│  src/acquisition/live_streamer.py                            │
│                                                             │
│  db.Live()                                                  │
│  ├── subscribe(GLBX.MDP3, tbbo, NQ.FUT, parent)           │
│  ├── subscribe(GLBX.MDP3, bbo-1s, NQ.FUT, parent)         │
│  ├── subscribe(GLBX.MDP3, bbo-1s, NQ.OPT, parent)         │
│  ├── subscribe(GLBX.MDP3, definition, NQ.OPT, parent)     │
│  └── subscribe(GLBX.MDP3, statistics, NQ.FUT, parent)     │
│                                                             │
│  Record Routing:                                            │
│  ├── MboMsg (action=Trade) → databento.tbbo INSERT         │
│  ├── Mbp1Msg → databento.bbo_1s INSERT                     │
│  ├── InstrumentDefMsgV2 → databento.definitions UPSERT     │
│  ├── StatMsg → databento.statistics INSERT                 │
│  └── SystemMsg → log only                                  │
│                                                             │
│  Raw archive: data/raw/live/YYYY-MM-DD/*.dbn                │
│                                                             │
│  ⚠️ NOT subscribed to: ES.FUT, DBEQ.BASIC equities         │
│  ⚠️ Each DB insert opens a new psycopg.connect() (no pool) │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 Key Implementation Details

- **Reconnect**: `on-failure` via systemd (not Databento client-level reconnect)
- **Rate limiting**: `StartLimitBurst=5` within `StartLimitIntervalSec=600`
- **Memory cap**: `MemoryMax=2G`
- **Log output**: systemd journal (`SyslogIdentifier=vtech-live-streamer`)
- **Price scaling**: `price / FIXED_PRICE_SCALE` (1e9) applied in `_px()` helper

### 8.3 Coverage Gaps

| Data | Live Subscribed | Impact |
|---|---|---|
| NQ.FUT TBBO | ✅ Yes | Core trading data |
| NQ.FUT BBO-1s | ✅ Yes | Book snapshots |
| NQ.OPT BBO-1s | ✅ Yes | Options quotes (but definitions needed to use) |
| NQ.OPT Definition | ✅ Yes | Contract specs |
| NQ.FUT Statistics | ✅ Yes | Settlement/hi/lo |
| **ES.FUT** | ❌ No | `ca_` features will go stale after 2026-04-02 |
| **DBEQ.BASIC equities** | ❌ No | `mx_` and `eq_` features will go stale after 2026-04-02 |


---

## 9. Feature Engineering Pipeline

### 9.1 Data Flow

```
TimescaleDB (Tier 2)
    │
    ├── tbbo ──────────────────────── ▶ Order flow (of_), VPIN (ms_), VWAP (vw_)
    │
    ├── tbbo_1min (continuous agg) ── ▶ Candle structure (cs_), Wavelets (wv_)
    │
    ├── bbo_1s ────────────────────── ▶ Book pressure (bp_)
    │
    ├── tbbo (ES contracts) ───────── ▶ Cross-asset NQ/ES (ca_)
    │
    ├── equity_ohlcv (VIXY) ───────── ▶ Macro sentiment (mx_)
    │
    ├── equity_ohlcv (NVDA/TSLA/     ▶ Equity context (eq_)
    │   XLK/SMH)
    │
    ├── bbo_1s (options) + defs ───── ▶ [DISABLED] IV surface (iv_)
    │
    └── statistics ────────────────── ▶ [DISABLED] Daily context (dc_)
    │
    ▼
Feature Engine (src/features/engine.py)
    │
    ▼
Parquet cache: data/parquet/features_YYYYMMDD.parquet
    │
    ▼
ML Training Pipeline
```

### 9.2 Feature Groups (138 Total Features)

| Priority | Module | Prefix | Count | Source | Status |
|---|---|---|---|---|---|
| P1 | `book_pressure.py` | `bp_` | 14 | `bbo_1s` (NQ futures) | ✅ Active |
| P2 | `order_flow.py` | `of_` | 18 | `tbbo` (NQ futures) | ✅ Active |
| P3 | `options_surface.py` | `iv_` | — | `bbo_1s` (options) + `definitions` | ❌ Disabled (no data) |
| P4 | `daily_context.py` | `dc_` | — | `statistics` | ❌ Disabled (no data) |
| P5 | `microstructure.py` | `ms_` | 4 | `tbbo` | ✅ Active |
| P6 | `wavelets.py` | `wv_` | 14 | `tbbo_1min` | ✅ Active |
| P7 | `candle_structure.py` | `cs_` | 24 | `tbbo_1min` (multi-timeframe) | ✅ Active |
| P7b | (multitimeframe) | `cs_` | (included above) | `tbbo_1min` (1m, 5m, 15m) | ✅ Active |
| P8 | `vwap.py` | `vw_` | 7 | `tbbo` | ✅ Active |
| P9 | `cross_asset.py` | `ca_` | 12 | `tbbo` (ES contracts) | ✅ Active |
| P10a | `macro_sentiment.py` | `mx_` | 10 | `equity_ohlcv` (VIXY) | ✅ Active |
| P10b | `equity_context.py` | `eq_` | 23 | `equity_ohlcv` (NVDA/TSLA/XLK/SMH) | ✅ Active |
| — | `time_context.py` | `tc_` | 12 | Derived from index + candles | ✅ Active (always on) |

### 9.3 Feature Computation Details

**Resolution**: 10-second timesteps (configurable via `VTECH_FEATURES_TIMESTEP`)

**Engine pipeline order** (in `engine.py`):
1. Load TBBO data (NQ only, `symbol NOT LIKE 'ES%%'`)
2. Load BBO-1s data (NQ only)
3. Load definitions + statistics (currently empty)
4. Load 1-min candles from continuous aggregate
5. Load ES 1-min candles (most-traded ES contract)
6. Load equity OHLCV (VIXY, NVDA, TSLA, XLK, SMH)
7. Resample all to 10s bars
8. Run enabled feature modules in order (P1→P10b)
9. Append time_context features (always enabled)
10. Include `_close` column for label generation
11. Save to Parquet

**Trigger**: Daily at 21:05 UTC via systemd timer, or manual `--date` flag.

### 9.4 Feature Detail by Module

#### `book_pressure.py` — 14 features (bp_)
- Spread: mean, max, z-scores (6/30/60 periods)
- Imbalance: raw, MA-6
- Depth: total, min, change (6/30), z-scores (6/30)
- Bid/ask ratio

#### `order_flow.py` — 18 features (of_)
- Delta: raw, %, buy ratio, trade count
- CVD: change at windows (3/6/12/30)
- Delta: sum + z-score at windows (3/6/12/30)
- Volume: MA-30, ratio

#### `microstructure.py` — 4 features (ms_)
- VPIN, Kyle's Lambda (30-period)
- VPIN z-scores (30/60)

#### `wavelets.py` — 14 features (wv_)
- Haar decomposition at 6 scales
- Detail + energy per scale
- Approximation coefficient + raw close

#### `candle_structure.py` — 24 features (cs_)
- Per candle: body ratio, upper/lower shadow, CLV, vol (with z-scores)
- Multi-timeframe: 5m, 15m aggregations

#### `vwap.py` — 7 features (vw_)
- VWAP, deviation, z-score, upper/lower bands, cumulative delta/ratio

#### `cross_asset.py` — 12 features (ca_)
- NQ/ES: spread (raw + z-score), correlation (30/60/180), beta (30/60), return diff, lead-lag (30/60), relative strength (30)

#### `macro_sentiment.py` — 10 features (mx_)
- VIXY: return, momentum (30/60), z-scores (30/60/180)
- NQ/VIXY: correlation (30/60)
- VIXY: acceleration, fear spike

#### `equity_context.py` — 23 features (eq_)
- Per symbol (NVDA/TSLA/XLK/SMH): momentum (30/60), lag-1 return, NQ correlation (60), relative strength (60)
- SMH/XLK: ratio, z-score (60)
- Breadth vs NQ

#### `time_context.py` — 12 features (tc_)
- Session block indicators, minutes since open/close
- Opening range high/low/width
- Breakout signals, time-of-day cyclical encoding


---

## 10. ML Training & Inference

### 10.1 Model Architecture

```
Input: (batch, sequence_length=64, num_features=138)
    │
    ▼
┌─── Attention-LSTM ───────────────────────────────┐
│  Stacked LSTM (configurable: [64,32] or [128,64])│
│  → Multi-Head Self-Attention (4 heads)           │
│  → LayerNorm                                     │
│  → Global Average Pooling                        │
│  → Dense → Dropout → Softmax (3 classes)         │
└──────────────────────────────────────────────────┘
    │
    ▼
Output: [BIG_DOWN, CONSOLIDATION, BIG_UP]
        50-point NQ threshold
        Forward windows: [5, 10, 15, 30] minutes
```

**Implementation**: `src/models/attention_lstm.py`
- Custom `AttentionLayer(keras.layers.Layer)` with Q/K/V projections + scaled dot-product
- Uses Keras 3 `add_weight()` syntax
- Compiled with Adam optimizer, `sparse_categorical_crossentropy`

### 10.2 Training Pipeline (10 Steps)

```
src/training/trainer.py — class Trainer

Step 1:  Load Parquet feature files for date range
Step 2:  Generate labels (3-class, 50pt threshold, max forward move)
Step 3:  Filter features by enabled_groups config
Step 4:  Temporal train/val/test split (70/15/15)
Step 5:  Preprocess (clip outliers z>5, RobustScaler)
Step 6:  Create sequences (length=64, stride=1)
Step 7:  Compute class weights (handle imbalance)
Step 8:  Build model (from config)
Step 9:  Train with early stopping (patience=10, lr reduction)
Step 10: Evaluate on test set, save checkpoint
```

**Feature Group Filtering**: `trainer.py` uses `FEATURE_GROUP_PREFIXES` dict to map enabled_groups → column prefixes:
```
bp_ → book_pressure    of_ → order_flow     iv_ → iv_surface
dc_ → daily_context    ms_ → vpin           wv_ → wavelets
cs_ → candle_structure  tc_ → time_context   vw_ → vwap
ca_ → cross_asset      mx_ → macro_sentiment eq_ → equity_context
```

### 10.3 HPO (Optuna)

`scripts/hpo_search.py` — Optuna-based hyperparameter optimization.

**Search space**:
- Architecture: LSTM units, attention heads, dense units, dropout
- Training: learning rate, batch size, scaler type
- Labels: threshold, forward windows, classification type (3/5 class)
- **Feature groups**: 9 toggleable groups (all except time_context)

**Storage**: SQLite at `/opt/vtech/data/optuna.db`, study name `nq_momentum_v1`

### 10.4 Model Versions

| Version | Features | LSTM Config | Key Change | Test Result |
|---|---|---|---|---|
| `model_v001_baseline` | 93 | [64,32] | Initial model | 100% CONSOLIDATION |
| `model_best_v001` | 93 | [64,32] | Best of initial HPO | 100% CONSOLIDATION |
| `model_best_v002` | ~105 | varies | +ES cross-asset | 100% CONSOLIDATION |
| `model_best_v003` | 138 | [128,64] | +VIXY +equities | 100% CONSOLIDATION |

**⚠️ All models predict 100% CONSOLIDATION** — zero BIG_DOWN or BIG_UP samples in test data. This is expected with only ~1 day of training data and a 50-point threshold. Meaningful class distribution requires weeks/months of data accumulation.

### 10.5 Checkpoint Format

```
model_best_v003/
├── model.keras           # 2.5 MB — Keras 3 saved model
├── features.json         # Feature list + metadata
└── training_result.json  # Metrics, confusion matrix, training history
```

### 10.6 Inference Service

**⚠️ NOT IMPLEMENTED** — No inference/prediction service exists. The trained model is not loaded for real-time predictions. This is a planned future phase.


---

## 11. Service Orchestration

### 11.1 Deployed systemd Services

```ini
# /opt/vtech/systemd/vtech-live-streamer.service
[Unit]
Description=VTech Databento Live Streamer
After=network-online.target postgresql.service
Wants=network-online.target
StartLimitIntervalSec=600
StartLimitBurst=5

[Service]
Type=simple
User=vtech
Group=vtech
WorkingDirectory=/opt/vtech
EnvironmentFile=/opt/vtech/.env
ExecStart=/opt/vtech/.venv/bin/python -m src.acquisition.live_streamer
Restart=on-failure
RestartSec=30
StandardOutput=journal
StandardError=journal
SyslogIdentifier=vtech-live-streamer
MemoryMax=2G

[Install]
WantedBy=multi-user.target
```

```ini
# /opt/vtech/systemd/vtech-feature-builder.service
[Unit]
Description=VTech Feature Builder Service
After=postgresql.service
Wants=postgresql.service

[Service]
Type=oneshot
User=vtech
Group=vtech
WorkingDirectory=/opt/vtech
EnvironmentFile=/opt/vtech/.env
ExecStart=/opt/vtech/.venv/bin/python -m src.features.engine --date today
StandardOutput=journal
StandardError=journal
SyslogIdentifier=vtech-feature-builder
MemoryMax=4G

[Install]
WantedBy=multi-user.target
```

```ini
# /opt/vtech/systemd/vtech-feature-builder.timer
[Unit]
Description=VTech Feature Builder Timer (daily at 17:05 ET)

[Timer]
OnCalendar=*-*-* 21:05:00 UTC
Persistent=true
RandomizedDelaySec=60

[Install]
WantedBy=timers.target
```

```ini
# /opt/vtech/systemd/vtech-trainer.service
[Unit]
Description=VTech Model Trainer Service
After=postgresql.service
Wants=postgresql.service

[Service]
Type=oneshot
User=vtech
Group=vtech
WorkingDirectory=/opt/vtech
EnvironmentFile=/opt/vtech/.env
ExecStart=/opt/vtech/.venv/bin/python -m src.training.trainer --start rolling --end today
StandardOutput=journal
StandardError=journal
SyslogIdentifier=vtech-trainer
MemoryMax=16G

[Install]
WantedBy=multi-user.target
```

```ini
# /opt/vtech/systemd/vtech-trainer.timer
[Unit]
Description=VTech Model Trainer Timer (weekly Sunday 08:00 ET)

[Timer]
OnCalendar=Sun *-*-* 12:00:00 UTC
Persistent=true
RandomizedDelaySec=300

[Install]
WantedBy=timers.target
```

### 11.2 Service Topology

```
systemd
├── postgresql.service                    # TimescaleDB (always running)
├── vtech-live-streamer.service           # Databento live NQ feed (persistent)
├── vtech-feature-builder.timer           # Daily at 21:05 UTC (17:05 ET)
│   └── vtech-feature-builder.service     # Oneshot — build features for today
├── vtech-trainer.timer                   # Weekly Sunday 12:00 UTC (08:00 ET)
│   └── vtech-trainer.service             # Oneshot — retrain model
└── (no monitoring/alerting services)
```

### 11.3 Known Issue: Venv Path Mismatch

The systemd services reference `/opt/vtech/.venv/bin/python` but the actual venv is at `/opt/vtech/venv/bin/python`. This works if `.venv` is a symlink to `venv`, or if `.venv` is the actual path and `venv` is the symlink. Verify with `ls -la /opt/vtech/.venv /opt/vtech/venv`.

### 11.4 Operational Commands

```bash
# Service management
sudo systemctl start vtech-live-streamer
sudo systemctl status vtech-live-streamer
journalctl -u vtech-live-streamer -f

# Manual feature build
/opt/vtech/venv/bin/python -m src.features.engine --date 2026-04-05

# Manual training
/opt/vtech/venv/bin/python -m src.training.trainer --start rolling --end today

# HPO search
/opt/vtech/venv/bin/python scripts/hpo_search.py --n-trials 20

# Cost check
/opt/vtech/venv/bin/python scripts/databento_cost_check.py

# Data verification
/opt/vtech/venv/bin/python scripts/databento_verify.py
```


---

## 12. Configuration & Secrets

### 12.1 Environment File

```bash
# /opt/vtech/.env (chmod 600)

# ─── Databento ───
DATABENTO_API_KEY=db-XXXXXXXX...          # Redacted

# ─── PostgreSQL / TimescaleDB ───
PGHOST=127.0.0.1
PGPORT=5432
PGDATABASE=vtech_market
PGUSER=vtech
PGPASSWORD=<redacted>

# ─── Application ───
VTECH_LOG_LEVEL=INFO
VTECH_DATA_DIR=/opt/vtech/data
VTECH_FEATURES_TIMESTEP=10s
VTECH_MODEL_CHECKPOINT_DIR=/opt/vtech/data/checkpoints
VTECH_ENV=production

# ─── ML ───
VTECH_LABEL_THRESHOLD=50
VTECH_SEQUENCE_LENGTH=64
VTECH_FORWARD_WINDOWS=5,10,15,30
VTECH_TRADING_HOURS_START=08:00
VTECH_TRADING_HOURS_END=11:00
VTECH_TIMEZONE=America/Chicago
```

### 12.2 Configuration Loader

```python
# src/common/config.py — AppConfig frozen dataclass with from_env() classmethod
# Loads all env vars, constructs pg_dsn, sets defaults
# Used by feature engine and training pipeline

# src/training/config.py — MLConfig hierarchy:
#   FeatureConfig: timestep, lookback, enabled_groups (12 groups)
#   DataConfig: train/val/test splits, scaler
#   LabelConfig: threshold, windows, classification type
#   ModelConfig: LSTM units, attention, dense, dropout, num_classes
#   TrainingConfig: batch_size, epochs, lr, early stopping
#   MLConfig: combines all above
```

### 12.3 Enabled Feature Groups (config.py defaults)

| Group | Default | Prefix |
|---|---|---|
| `book_pressure` | `True` | `bp_` |
| `order_flow` | `True` | `of_` |
| `iv_surface` | `False` | `iv_` |
| `daily_context` | `False` | `dc_` |
| `vpin` | `True` | `ms_` |
| `wavelets` | `True` | `wv_` |
| `candle_structure` | `True` | `cs_` |
| `time_context` | `True` | `tc_` |
| `vwap` | `True` | `vw_` |
| `cross_asset` | `False` | `ca_` |
| `macro_sentiment` | `False` | `mx_` |
| `equity_context` | `False` | `eq_` |

> cross_asset, macro_sentiment, equity_context default to False but are enabled via HPO or manual override during training.

---

## 13. Monitoring & Observability

### 13.1 Current State

**⚠️ No monitoring is deployed.** The v1 plan included Prometheus, node exporter, and health check scripts. None have been implemented.

### 13.2 What Exists

- **systemd journal**: All services log to journal (viewable via `journalctl`)
- **structlog**: Application-level structured logging (JSON format)
- **Manual checks**: `systemctl status`, ad-hoc SQL queries

### 13.3 Planned (from v1, not yet implemented)

| Metric | Source | Alert Threshold |
|---|---|---|
| Records ingested/second | Live streamer counter | < 1 during market hours |
| DB insert latency (p99) | Ingestion timer | > 100ms |
| Live stream heartbeat age | Last heartbeat timestamp | > 45s |
| Disk usage | Node exporter | > 85% |
| Feature build duration | Timer service log | > 30 min |
| Model training loss | Training log | val_loss not decreasing |

---

## 14. Network & Security

### 14.1 Current State

- PostgreSQL listens on `127.0.0.1` only (local connections)
- SSH access (key-based assumed)
- `.env` file contains secrets (should be `chmod 600`)
- Dedicated `vtech` system user for services
- No explicit firewall rules configured via `ufw` (cloud security group assumed)

### 14.2 Security Checklist

- [x] `DATABENTO_API_KEY` in `.env` file — not in code
- [x] PostgreSQL on localhost only
- [x] Dedicated `vtech` system user
- [ ] `.env` file permissions verified as 600
- [ ] SSH key-only auth confirmed
- [ ] Unattended security updates enabled
- [ ] Firewall (ufw) configured
- [ ] Credential rotation policy

---

## 15. Backup & Recovery

### 15.1 Current State

**⚠️ No automated backups.** The v1 plan included daily pg_dump and rsync to cloud storage. Not implemented.

### 15.2 Recovery Considerations

| Scenario | Recovery Path | Impact |
|---|---|---|
| Live streamer crash | systemd auto-restart (30s delay) | Brief gap in TBBO data |
| Database corruption | No backup exists — would need re-download from Databento | High (re-download costs money + time) |
| VM total loss | No offsite backup — full rebuild required | Critical |
| Disk full | Compress old chunks, delete live DBN archives | Recoverable |

### 15.3 Recommended (Not Yet Implemented)

```bash
# Daily pg_dump at 23:00 UTC
pg_dump -Fc vtech_market > /opt/vtech/data/backups/vtech_$(date +%Y%m%d).dump

# Model checkpoints to cloud
rsync -avz /opt/vtech/data/checkpoints/ <cloud-storage>:/vtech/checkpoints/

# Retention: 7 daily, 4 weekly
```


---

## 16. Implementation Sequence

### Phase 1: Foundation ✅ COMPLETED

- [x] Provision VM — Oracle Cloud, Ubuntu 22.04, 4 vCPU, 31 GB RAM, 49 GB root + 500 GB data disk
- [x] Install Python 3.12 (venv), PostgreSQL 18.3 + TimescaleDB 2.26.1
- [x] Create `vtech` user, directory structure at `/opt/vtech/`
- [x] Configure `.env` with Databento API key + PG credentials
- [x] Verify Databento connectivity

### Phase 2: Database Setup ✅ COMPLETED

- [x] Create `vtech_market` database
- [x] Create `databento` schema + all table DDL
- [x] Create hypertables (tbbo, bbo_1s, definitions, statistics)
- [x] Set compression policies on tbbo and bbo_1s (30-day interval)
- [x] Create `tbbo_1min` continuous aggregate with auto-refresh policy

### Phase 3: Historical Backfill — NQ Futures ✅ COMPLETED

- [x] Cost check for NQ.FUT TBBO (1 year)
- [x] Download NQ.FUT TBBO via `timeseries.get_range` (2025-04-01 → 2026-04-02)
- [x] Ingest TBBO into TimescaleDB (122M+ rows initial)
- [x] Verify record counts and date coverage
- [ ] ~~Download NQ.FUT BBO-1s (separate backfill)~~ — BBO-1s populated from live stream only (54.6K rows)

### Phase 4: Historical Backfill — Options ❌ NOT COMPLETED

- [ ] Download NQ.OPT BBO-1s — batch job completed (GLBX-20260402-S4EMNFKM8K, 56 GB), awaiting download
- [ ] Download NQ.OPT Definition — pending (streaming API)
- [ ] Download NQ.FUT Statistics — pending (streaming API)
- [ ] Ingest into respective tables

> **Impact**: `iv_surface` (options IV) and `daily_context` (settlement/hi-lo) feature groups are disabled due to missing data. These represent potentially high-signal features that are deferred.

### Phase 5: Live Streaming ✅ COMPLETED

- [x] Implement `LiveStreamer` class in `src/acquisition/live_streamer.py`
- [x] Subscribe to: NQ.FUT tbbo, NQ.FUT bbo-1s, NQ.OPT bbo-1s, NQ.OPT definition, NQ.FUT statistics
- [x] Deploy as systemd service (`vtech-live-streamer.service`)
- [x] Verify live data flowing into TimescaleDB
- [x] Raw DBN archive saving to `data/raw/live/YYYY-MM-DD/`
- [ ] ~~Monitor overnight reconnect~~ — Using systemd `Restart=on-failure` instead of client-level reconnect

### Phase 6: Ingestion Pipeline ✅ COMPLETED (Simplified)

- [x] Build `src/ingestion/loader.py` — DBN → DataFrame → batch INSERT
- [x] Implement dedup via `ON CONFLICT DO NOTHING` (inline, not separate module)
- [x] Build `src/ingestion/quality.py` — gap detection, record counts
- [ ] ~~Build `transforms.py`~~ — Transforms inline in loader
- [ ] ~~Build `dedup.py`~~ — Dedup via SQL constraint, not separate module
- [x] Scripts: `databento_backfill.py`, `databento_cost_check.py`, `databento_verify.py`

### Phase 7: Feature Engineering ✅ COMPLETED (12 Modules)

- [x] Build `book_pressure.py` — 14 features (bp_) from bbo_1s
- [x] Build `order_flow.py` — 18 features (of_) from tbbo
- [x] Build `options_surface.py` — IV surface module (exists but disabled — no data)
- [x] Build `daily_context.py` — Settlement/hi-lo module (exists but disabled — no data)
- [x] Build `microstructure.py` — 4 features (ms_) VPIN + Kyle's Lambda
- [x] Build `wavelets.py` — 14 features (wv_) Haar decomposition
- [x] Build `candle_structure.py` — 24 features (cs_) + multitimeframe
- [x] Build `time_context.py` — 12 features (tc_) session context
- [x] Build `vwap.py` — 7 features (vw_) VWAP deviation
- [x] Build `engine.py` — Feature orchestrator with 12-step pipeline
- [x] Deploy feature builder service + timer (daily 21:05 UTC)
- [x] Validate feature matrix (138 features, ~8,500 rows/day at 10s resolution)

### Phase 8: ML Training Pipeline ✅ COMPLETED

- [x] Build `config.py` — MLConfig dataclass hierarchy (Feature/Data/Label/Model/Training configs)
- [x] Build `labels.py` — 3-class + 5-class labeling (50pt threshold)
- [x] Build `data_pipeline.py` — Parquet load, temporal split, RobustScaler, sequence creation
- [x] Build `attention_lstm.py` — Multi-head attention + stacked LSTM (Keras 3)
- [x] Build `trainer.py` — 10-step training orchestrator
- [x] Build `evaluate.py` — Metrics, confusion matrix, classification report
- [x] Train initial model (v001_baseline) on available data
- [x] Deploy trainer service + timer (weekly Sunday 12:00 UTC)

### Phase 9: Cross-Asset — ES Futures ✅ COMPLETED

- [x] Download ES.FUT TBBO (1 year, 2025-04-01 → 2026-04-02)
- [x] Ingest into `databento.tbbo` table (co-located with NQ data)
- [x] Build `cross_asset.py` — 12 features (ca_) NQ/ES correlation, lead-lag, spread, beta
- [x] Wire into engine.py (`_load_es_candles_1min()` selects most-traded ES contract)
- [x] Retrain model v002 with cross-asset features
- [x] Run HPO with cross_asset toggle

### Phase 10: Hyperparameter Optimization ✅ COMPLETED

- [x] Build `scripts/hpo_search.py` — Optuna integration
- [x] Configure search space: architecture, training, labels, feature groups
- [x] Run initial HPO trials (5 trials, ~1 hr total)
- [x] Save best model as v001 (from HPO)
- [x] Optuna DB at `data/optuna.db` (study: `nq_momentum_v1`)

### Phase 11: Cross-Asset — Equities & Macro ✅ COMPLETED

- [x] Cost assessment: NVDA, TSLA, XLK, SMH, VIXY via DBEQ.BASIC ($1.45/year)
- [x] Download DBEQ.BASIC OHLCV-1m for 5 symbols (2025-04-01 → 2026-04-02)
- [x] Create `databento.equity_ohlcv` hypertable (7-day chunk interval)
- [x] Ingest 469K rows (with dedup), enable compression
- [x] Build `macro_sentiment.py` — 10 features (mx_) from VIXY
- [x] Build `equity_context.py` — 23 features (eq_) from NVDA/TSLA/XLK/SMH
- [x] Wire into engine.py, config.py, trainer.py, hpo_search.py
- [x] Run 5-trial HPO with all new feature groups
- [x] Train and save model v003 (138 features)
- [x] Delete raw equity download to save disk

### Phase 12: Production Hardening ❌ NOT STARTED

- [ ] Monitoring dashboards (Prometheus + node exporter)
- [ ] Health check cron job
- [ ] Automated backup (pg_dump + rsync)
- [ ] Live equity/ES data pipeline (daily fetch or streaming)
- [ ] Inference/prediction service
- [ ] Disk expansion plan (for options data)
- [ ] Documentation of operational runbook
- [ ] Load test: simulate elevated message rates


---

## 17. Known Issues & Gaps

### 17.1 Critical Gaps

| Gap | Impact | Effort to Fix |
|---|---|---|
| **No live equity data pipeline** | `mx_` and `eq_` features go stale after 2026-04-02 | Medium — need daily DBEQ.BASIC fetch (~$0.006/day) |
| **No live ES data pipeline** | `ca_` features go stale after 2026-04-02 | Medium — add ES.FUT to live_streamer subscriptions |
| **No inference service** | Trained model never makes predictions | High — requires new module + service |
| **100% CONSOLIDATION predictions** | Models trivially predict majority class | Expected — need more data accumulation over weeks |
| **No automated backups** | Data loss risk if VM fails | Low — add pg_dump cron + cloud sync |
| **No monitoring/alerting** | Silent failures go unnoticed | Medium — add health checks + alerts |

### 17.2 Code Issues

| Issue | File | Description |
|---|---|---|
| Stray SQL in root script | `ingest_tbbo_chunked.py` | Has raw SQL appended after Python code — would error if executed |
| Duplicate `get_dsn()` | `src/ingestion/loader.py` | Defines own `get_dsn()` instead of importing from `src.common.db` |
| ~~Missing `ingest_file()`~~ | `scripts/databento_backfill.py` | ✅ Fixed — `ingest_file()` dispatcher added to `loader.py` (supports tbbo, bbo-1s, definition, statistics) |
| Venv path mismatch | systemd units | Services use `.venv` but actual venv may be at `venv` |
| BBO-1s partial insert | `live_streamer.py` | Inserts only 7 of 14 columns (bid/ask only, no OHLCV) |

### 17.3 Data Gaps

| Table | Expected | Actual | Impact |
|---|---|---|---|
| `definitions` | NQ.OPT contract specs | 0 rows | Cannot compute IV surface features |
| `statistics` | NQ.FUT settlement/hi/lo | 0 rows | Cannot compute daily context features |
| `equity_ohlcv` | Ongoing daily updates | Static (ends 2026-04-02) | Features decay without refresh |
| `tbbo` (ES) | Ongoing ES data | Static (ends 2026-04-02) | Cross-asset features decay |

---

## Appendix A: Databento Client Quick Reference

```python
import databento as db

# Authentication (env var preferred)
client_hist = db.Historical()    # Uses DATABENTO_API_KEY
client_live = db.Live()

# Cost check (always free)
cost = client_hist.metadata.get_cost(
    dataset="GLBX.MDP3", symbols="NQ.FUT",
    schema="tbbo", stype_in="parent",
    start="2025-04-01", end="2026-04-01",
)

# Stream to file
data = client_hist.timeseries.get_range(
    dataset="GLBX.MDP3", symbols="NQ.FUT",
    schema="tbbo", stype_in="parent",
    start="2025-04-01", end="2025-04-02",
    path="nq_tbbo.dbn.zst",
)

# Read DBN file
store = db.DBNStore.from_file("nq_tbbo.dbn.zst")
df = store.to_df(pretty_ts=True, map_symbols=True)

# Price scaling
from databento_dbn import FIXED_PRICE_SCALE  # 1e9
actual_price = raw_price / FIXED_PRICE_SCALE

# Live streaming
client_live = db.Live()
client_live.subscribe(dataset="GLBX.MDP3", schema="tbbo",
                      symbols="NQ.FUT", stype_in="parent")
client_live.add_callback(handler)
client_live.add_stream("archive.dbn")
client_live.start()
client_live.block_for_close()
```

## Appendix B: Actual Feature List (138 features, model v003)

```
book_pressure (14):  bp_spread_mean, bp_spread_max, bp_spread_zscore_6,
                     bp_spread_zscore_30, bp_spread_zscore_60, bp_imbalance,
                     bp_imbalance_ma6, bp_depth_total, bp_depth_min,
                     bp_depth_chg_6, bp_depth_zscore_6, bp_depth_chg_30,
                     bp_depth_zscore_30, bp_bid_ask_ratio

order_flow (18):     of_delta, of_delta_pct, of_buy_ratio, of_trade_count,
                     of_cvd_chg_3, of_delta_sum_3, of_delta_zscore_3,
                     of_cvd_chg_6, of_delta_sum_6, of_delta_zscore_6,
                     of_cvd_chg_12, of_delta_sum_12, of_delta_zscore_12,
                     of_cvd_chg_30, of_delta_sum_30, of_delta_zscore_30,
                     of_volume_ma_30, of_volume_ratio

microstructure (4):  ms_vpin, ms_kyle_lambda_30, ms_vpin_zscore_30,
                     ms_vpin_zscore_60

wavelets (14):       wv_detail_1..6, wv_energy_1..6, wv_approx, wv_close

candle_struct (24):  cs_body_ratio, cs_upper_shadow, cs_lower_shadow,
                     cs_clv, cs_vol, cs_body_ratio_zscore_6..30,
                     cs_vol_zscore_6..30, cs_5m_*, cs_15m_*

vwap (7):            vw_vwap, vw_deviation, vw_zscore, vw_upper_band,
                     vw_lower_band, vw_cum_delta, vw_cum_ratio

cross_asset (12):    ca_spread, ca_spread_zscore_30, ca_corr_30, ca_corr_60,
                     ca_corr_180, ca_beta_30, ca_beta_60, ca_ret_diff,
                     ca_lead_lag_30, ca_lead_lag_60, ca_rel_strength_30

macro_sent (10):     mx_vixy_ret, mx_vixy_mom_30, mx_vixy_mom_60,
                     mx_vixy_zscore_30, mx_vixy_zscore_60, mx_vixy_zscore_180,
                     mx_nq_vixy_corr_30, mx_nq_vixy_corr_60,
                     mx_vixy_accel, mx_vixy_spike

equity_ctx (23):     eq_nvda_mom_30..60, eq_nvda_ret_lag1, eq_nvda_corr_60,
                     eq_nvda_rel_str_60, eq_tsla_*, eq_xlk_*, eq_smh_*,
                     eq_smh_xlk_ratio, eq_smh_xlk_zscore_60, eq_breadth_vs_nq

time_context (12):   tc_session_*, tc_minutes_*, tc_or_*, tc_breakout_*,
                     tc_sin_*, tc_cos_*
```

## Appendix C: Actual Storage Budget (2026-04-06)

| Component | Size | Notes |
|---|---|---|
| TimescaleDB (all tables) | ~3.2 GB | Dominated by tbbo (3.1 GB, 124M rows) |
| Live DBN archives | ~450 MB | 3 days of streaming |
| Parquet feature cache | ~2.5 MB | 4 days computed |
| Model checkpoints | ~12 MB | 4 models + HPO trials |
| Optuna DB | 135 KB | |
| Python venv | ~2 GB est. | TensorFlow is largest |
| OS + packages | ~8 GB est. | |
| **Root disk used** | **~15 GB** | **31% of 49 GB root** |
| **Data disk used** | **~510 MB** | **<1% of 500 GB data** |
| **Root available** | **34 GB** | |
| **Data available** | **466 GB** | |

### Projection

At current growth rate (~400 MB/day live archives + ~3 MB/day TimescaleDB after compression):
- **30 days**: +12 GB live + 90 MB DB = ~13 GB on data disk (~3% of 500 GB)
- **90 days**: +36 GB live + 270 MB DB = ~37 GB on data disk (~8% of 500 GB)
- **With NQ.OPT backfill** (~56 GB): ~93 GB used (~19% of 500 GB)
- **1 year projection**: ~146 GB live + 1 GB DB + 56 GB backfill = ~203 GB (~41% of 500 GB)
