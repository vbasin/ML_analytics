# Databento VM Architecture — NQ Momentum ML Platform

> Purpose-built VM for Databento data acquisition, storage, feature engineering, and ML training for NQ 50-point momentum prediction.

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

---

## 1. Design Principles

| Principle | Rationale |
|---|---|
| **DBN-native storage** | Keep raw `.dbn.zst` files as immutable archive; Databento's zero-copy format (19.1M msg/sec) is the canonical source of truth |
| **Three-tier data** | Raw DBN → TimescaleDB (queryable) → Parquet/numpy (ML-ready feature cache) |
| **Schema-per-concern** | Separate DB schemas for futures vs options vs definitions — clean boundaries, independent retention |
| **Batch for backfill, live for incremental** | Databento recommends batch for >5GB (free re-download 30 days); live streaming for real-time incremental |
| **No Docker for data pipeline** | systemd services — eliminates container overhead on I/O-critical path; Docker optional for ancillary services |
| **Environment-driven config** | All secrets/endpoints via env vars or `.env` files; zero hardcoded values |
| **Idempotent ingestion** | Every pipeline stage is safe to re-run; dedup on `(instrument_id, ts_event, schema)` |

---

## 2. VM Specifications

### Recommended (Primary Target)

| Resource | Spec | Rationale |
|---|---|---|
| **CPU** | 4–8 vCPU (AMD EPYC / Intel Xeon) | DBN decode is CPU-light; feature compute + ML training are the bottleneck |
| **RAM** | 32 GB | TimescaleDB shared_buffers (8GB) + pandas/numpy working memory for feature computation + ML training |
| **Root disk** | 50 GB SSD | OS, packages, code |
| **Data disk** | 500 GB NVMe SSD | Raw DBN archive (~20-40GB/yr), TimescaleDB (~150-200GB), Parquet cache (~50GB), growth headroom |
| **GPU** | Optional — 1× T4 or L4 | Only needed for GPU-accelerated training; CPU training works for LSTM at this data scale |
| **Network** | 1 Gbps+ | Databento streams over TCP; batch downloads benefit from bandwidth |
| **OS** | Ubuntu 22.04 LTS | Mature, well-supported, matches existing VM environment |

### Minimum Viable

| Resource | Spec |
|---|---|
| **CPU** | 2 vCPU |
| **RAM** | 16 GB |
| **Data disk** | 250 GB SSD |
| **GPU** | None (CPU training) |

> **Databento's own guidance**: "Very low minimum system requirements. No special hardware needed." Most customers do not need sub-50μs latency. TCP transport, no special NIC required.

---

## 3. OS & Runtime Stack

```
Ubuntu 22.04 LTS
├── Python 3.12 (via deadsnakes PPA or pyenv)
│   └── venv: /opt/vtech/venv
├── PostgreSQL 16 + TimescaleDB 2.x (apt repo)
├── systemd (service management)
├── Rust toolchain (for dbn-cli transcoding utility)
└── Optional: CUDA 12.x + cuDNN (if GPU present)
```

### Core Python Dependencies

```
databento>=0.43.0        # Databento client (Historical + Live)
pandas>=2.1              # DataFrame operations
numpy>=1.26              # Numerical compute
psycopg[binary]>=3.1     # PostgreSQL async driver (psycopg3)
sqlalchemy>=2.0          # ORM/connection pooling (optional)
pyarrow>=14.0            # Parquet I/O
tensorflow>=2.15         # ML training (or torch>=2.1)
scikit-learn>=1.4        # Preprocessing, metrics
scipy>=1.12              # Wavelets, signal processing
pywt>=1.6                # Discrete wavelet transforms
python-dotenv>=1.0       # .env file loading
structlog>=24.1          # Structured logging
prometheus-client>=0.20  # Metrics export
```

### System Packages

```bash
sudo apt update && sudo apt install -y \
  build-essential python3.12 python3.12-venv python3.12-dev \
  postgresql-16 postgresql-16-timescaledb \
  libpq-dev zstd lz4 \
  curl git htop iotop tmux
```

### Databento CLI (Rust)

```bash
# For DBN file inspection and transcoding (DBN → CSV/JSON)
cargo install dbn-cli
```

---

## 4. Directory Layout

```
/opt/vtech/                         # Application root
├── venv/                           # Python virtual environment
├── .env                            # Environment variables (600 permissions)
├── src/                            # Application source code
│   ├── acquisition/                # Data download & streaming
│   │   ├── __init__.py
│   │   ├── historical.py           # Batch backfill from Databento Historical API
│   │   ├── live_streamer.py        # Real-time Live API streaming service
│   │   ├── schemas.py              # Schema definitions & field mappings
│   │   └── symbology.py            # Symbol resolution & contract stitching
│   ├── ingestion/                  # DBN → TimescaleDB transform & load
│   │   ├── __init__.py
│   │   ├── loader.py               # DBN file → DataFrame → DB insert
│   │   ├── transforms.py           # Price scaling, timestamp normalization
│   │   └── dedup.py                # Idempotent insert logic
│   ├── features/                   # Feature engineering (evolved from ml/features/)
│   │   ├── __init__.py
│   │   ├── engine.py               # Feature orchestrator
│   │   ├── microstructure.py       # VPIN, Kyle's Lambda, Amihud
│   │   ├── order_flow.py           # CVD, delta, absorption
│   │   ├── book_pressure.py        # Bid/ask imbalance, depth analysis
│   │   ├── options_surface.py      # IV surface, skew, GEX
│   │   ├── wavelets.py             # Multi-scale decomposition
│   │   ├── candle_structure.py     # Body ratio, CLV, volatility regime
│   │   ├── time_context.py         # Session blocks, opening range
│   │   ├── vwap.py                 # VWAP deviation features
│   │   ├── daily_context.py        # Settlement, high/low from Statistics
│   │   └── cross_asset.py          # ES correlation, cross-market signals
│   ├── training/                   # ML training pipeline
│   │   ├── __init__.py
│   │   ├── config.py               # All ML configuration dataclasses
│   │   ├── data_pipeline.py        # DB → feature matrix → sequences
│   │   ├── labels.py               # 3-class labeling (50pt threshold)
│   │   ├── trainer.py              # Training orchestrator
│   │   └── evaluate.py             # Metrics, confusion matrix, reporting
│   ├── models/                     # Model definitions
│   │   ├── __init__.py
│   │   ├── attention_lstm.py       # Attention-LSTM architecture
│   │   └── registry.py             # Model versioning & checkpoint management
│   └── common/                     # Shared utilities
│       ├── __init__.py
│       ├── db.py                   # Database connection pool
│       ├── config.py               # Environment config loader
│       └── logging.py              # Structured logging setup
├── data/                           # Data storage root
│   ├── raw/                        # Immutable DBN archive
│   │   ├── backfill/               # Historical batch downloads
│   │   │   ├── nq_futures/         # TBBO + BBO-1s for NQ futures
│   │   │   │   ├── GLBX-20240701-XXXXX/  # Batch job directories
│   │   │   │   │   └── *.dbn.zst
│   │   │   │   └── ...
│   │   │   ├── nq_options/         # BBO-1s + Definition for NQ options
│   │   │   └── es_futures/         # TBBO for ES (cross-asset)
│   │   └── live/                   # Live stream captures (daily rotation)
│   │       └── YYYY-MM-DD/
│   │           ├── nq_tbbo.dbn
│   │           └── nq_opts_bbo1s.dbn
│   ├── parquet/                    # ML-ready feature cache
│   │   └── features_YYYYMMDD.parquet
│   └── checkpoints/                # Model checkpoints
│       └── model_v001_YYYYMMDD/
├── logs/                           # Application logs
├── scripts/                        # Operational scripts
│   ├── backfill.py                 # One-time historical data pull
│   ├── verify_data.py              # Data quality checks
│   ├── migrate_db.py               # Schema migrations
│   └── cost_check.py               # Pre-download cost estimation
└── systemd/                        # Service unit files
    ├── vtech-live-streamer.service
    ├── vtech-feature-builder.service
    └── vtech-timescaledb.conf      # PostgreSQL tuning overrides
```

---

## 5. Storage Architecture

### Three-Tier Model

```
┌──────────────────────────────────────────────────────────────┐
│                    TIER 1: Raw DBN Archive                    │
│  /opt/vtech/data/raw/**/*.dbn.zst                            │
│  Immutable, compressed (zstd), self-describing metadata      │
│  ~20-40 GB/year for all schemas combined                     │
│  Retention: indefinite (cheap at this scale)                 │
│  Purpose: reprocessing, audit trail, schema evolution        │
└──────────────────────┬───────────────────────────────────────┘
                       │ DBNStore.to_df() → transform → INSERT
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                 TIER 2: TimescaleDB (Queryable)              │
│  PostgreSQL 16 + TimescaleDB hypertables                     │
│  ~150-200 GB/year (futures + 0DTE options BBO-1s)            │
│  Retention: 1 year hot, compress chunks >30 days             │
│  Purpose: feature queries, ad-hoc analysis, backtesting      │
└──────────────────────┬───────────────────────────────────────┘
                       │ Feature engine → to_parquet()
                       ▼
┌──────────────────────────────────────────────────────────────┐
│              TIER 3: Parquet Feature Cache (ML-Ready)        │
│  /opt/vtech/data/parquet/features_*.parquet                  │
│  ~5-15 GB for 1 year of pre-computed feature matrices        │
│  Retention: regenerable (delete & recompute)                 │
│  Purpose: fast ML training data loading                      │
└──────────────────────────────────────────────────────────────┘
```

### TimescaleDB Tuning

```ini
# /opt/vtech/systemd/vtech-timescaledb.conf
# Append to postgresql.conf or use ALTER SYSTEM

shared_buffers = 8GB
effective_cache_size = 24GB
work_mem = 256MB
maintenance_work_mem = 2GB
max_connections = 50
wal_buffers = 64MB

# TimescaleDB-specific
timescaledb.max_background_workers = 8
timescaledb.last_tuned = 'vtech-databento'

# Compression
timescaledb.compress_segmentby = 'instrument_id'
timescaledb.compress_orderby = 'ts_event ASC'
```

---

## 6. Data Acquisition Layer

### 6.1 Historical Backfill (Batch API)

The primary mechanism for initial data population. Uses `batch.submit_job` for large requests (>5GB), `timeseries.get_range` with `path=` for smaller ones.

```python
# src/acquisition/historical.py — Core backfill pattern

import databento as db
from pathlib import Path
from datetime import date, timedelta

def backfill_schema(
    dataset: str,
    symbols: str | list[str],
    schema: str,
    start: str,
    end: str,
    stype_in: str = "parent",
    output_dir: Path = Path("/opt/vtech/data/raw/backfill"),
) -> Path:
    """Download historical data via streaming API, save as DBN.
    
    For requests >5GB, use batch.submit_job instead.
    Uses DATABENTO_API_KEY env var for authentication.
    """
    client = db.Historical()
    
    # Pre-check cost
    cost = client.metadata.get_cost(
        dataset=dataset,
        symbols=symbols,
        schema=schema,
        start=start,
        end=end,
        stype_in=stype_in,
    )
    print(f"Estimated cost: ${cost:.4f}")
    
    # Stream directly to file
    out_path = output_dir / f"{dataset}-{schema}-{start}-{end}.dbn.zst"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = client.timeseries.get_range(
        dataset=dataset,
        symbols=symbols,
        schema=schema,
        start=start,
        end=end,
        stype_in=stype_in,
        path=str(out_path),  # Stream directly to disk
    )
    
    return out_path


def batch_backfill_large(
    dataset: str,
    symbols: str | list[str],
    schema: str,
    start: str,
    end: str,
    stype_in: str = "parent",
    output_dir: Path = Path("/opt/vtech/data/raw/backfill"),
) -> list[Path]:
    """Submit batch job for large historical requests (>5GB).
    
    Batch jobs are processed asynchronously. Files can be re-downloaded
    for free within 30 days.
    """
    client = db.Historical()
    
    job = client.batch.submit_job(
        dataset=dataset,
        symbols=symbols,
        schema=schema,
        start=start,
        end=end,
        stype_in=stype_in,
        encoding="dbn",
        compression="zstd",
        split_duration="day",        # One file per day
        split_symbols=False,
    )
    
    print(f"Batch job submitted: {job['id']} (state: {job['state']})")
    print(f"Monitor at: https://databento.com/portal/download-center")
    
    # Poll and download when ready (or check portal)
    # client.batch.download(job_id=job["id"], output_dir=str(output_dir))
    return job["id"]
```

### 6.2 Schema-Specific Backfill Jobs

| Job | Dataset | Symbols | Schema | stype_in | Priority | Est. Size/Year |
|---|---|---|---|---|---|---|
| NQ futures trades+BBO | `GLBX.MDP3` | `NQ.FUT` | `tbbo` | `parent` | **P0** | ~3-5 GB |
| NQ futures quotes 1s | `GLBX.MDP3` | `NQ.FUT` | `bbo-1s` | `parent` | **P0** | ~2-4 GB |
| NQ 0DTE options quotes | `GLBX.MDP3` | `NQ.OPT` | `bbo-1s` | `parent` | **P1** | ~6-14 GB |
| NQ options definitions | `GLBX.MDP3` | `NQ.OPT` | `definition` | `parent` | **P1** | ~0.5 GB |
| NQ futures statistics | `GLBX.MDP3` | `NQ.FUT` | `statistics` | `parent` | **P2** | ~0.2 GB |
| ES futures trades+BBO | `GLBX.MDP3` | `ES.FUT` | `tbbo` | `parent` | **P2** | ~3-5 GB |

### 6.3 Contract Stitching via Continuous Symbology

Databento's continuous contract symbology eliminates manual contract stitching:

```python
# Calendar-based continuous: always maps to front month
# NQ.c.0 = front month, NQ.c.1 = second month
data = client.timeseries.get_range(
    dataset="GLBX.MDP3",
    symbols="NQ.c.0",           # Automatic rolling front month
    schema="tbbo",
    stype_in="continuous",      # KEY: use continuous symbology
    start="2024-07-01",
    end="2025-07-01",
)

# Volume-based continuous: maps to highest volume contract
data = client.timeseries.get_range(
    dataset="GLBX.MDP3",
    symbols="NQ.v.0",           # Highest volume NQ contract
    schema="tbbo",
    stype_in="continuous",
    start="2024-07-01",
)
```

Symbology mappings are embedded in DBN metadata — `DBNStore.to_df(map_symbols=True)` automatically adds a `symbol` column mapping `instrument_id` to the raw symbol for every record.

---

## 7. Database Schema Design

### 7.1 Schema Organization

```sql
-- Separate PostgreSQL schemas for clean boundaries
CREATE SCHEMA IF NOT EXISTS databento;
CREATE SCHEMA IF NOT EXISTS features;
CREATE SCHEMA IF NOT EXISTS ml;

-- Enable TimescaleDB
CREATE EXTENSION IF NOT EXISTS timescaledb;
```

### 7.2 Core Tables

```sql
-- ============================================================
-- TBBO: Trades with Best Bid/Offer at time of trade
-- Source: Databento TBBO schema (NQ.FUT, ES.FUT)
-- ============================================================
CREATE TABLE databento.tbbo (
    ts_event       TIMESTAMPTZ    NOT NULL,  -- Exchange event timestamp (nanoseconds → timestamptz)
    ts_recv        TIMESTAMPTZ    NOT NULL,  -- Databento capture timestamp
    instrument_id  INTEGER        NOT NULL,  -- Venue-assigned instrument ID
    symbol         TEXT           NOT NULL,  -- Raw symbol (e.g., NQU5, ESZ5)
    price          DOUBLE PRECISION NOT NULL,-- Trade price
    size           INTEGER        NOT NULL,  -- Trade size
    side           CHAR(1),                  -- Aggressor side: 'A'sk, 'B'id, 'N'one
    action         CHAR(1),                  -- 'T'rade
    flags          SMALLINT       DEFAULT 0, -- Event flags bitfield
    sequence       BIGINT,                   -- Exchange sequence number
    -- BBO snapshot at time of trade
    bid_px         DOUBLE PRECISION,
    ask_px         DOUBLE PRECISION,
    bid_sz         INTEGER,
    ask_sz         INTEGER,
    bid_ct         INTEGER,                  -- Bid order count
    ask_ct         INTEGER,                  -- Ask order count
    ts_in_delta    INTEGER,                  -- Matching engine latency (ns)
    -- Metadata
    dataset        TEXT           DEFAULT 'GLBX.MDP3',
    UNIQUE (instrument_id, ts_event, sequence)
);

SELECT create_hypertable('databento.tbbo', 'ts_event',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX idx_tbbo_symbol_ts ON databento.tbbo (symbol, ts_event DESC);
CREATE INDEX idx_tbbo_instrument_ts ON databento.tbbo (instrument_id, ts_event DESC);


-- ============================================================
-- BBO_1S: Best Bid/Offer sampled every 1 second
-- Source: Databento BBO-1s schema (NQ.FUT, NQ.OPT, ES.FUT)
-- ============================================================
CREATE TABLE databento.bbo_1s (
    ts_event       TIMESTAMPTZ    NOT NULL,  -- Interval open timestamp
    instrument_id  INTEGER        NOT NULL,
    symbol         TEXT           NOT NULL,
    -- OHLCV for the 1-second interval
    open           DOUBLE PRECISION,
    high           DOUBLE PRECISION,
    low            DOUBLE PRECISION,
    close          DOUBLE PRECISION,
    volume         BIGINT,
    -- BBO at interval close
    bid_px         DOUBLE PRECISION,
    ask_px         DOUBLE PRECISION,
    bid_sz         INTEGER,
    ask_sz         INTEGER,
    -- Computed spread
    spread         DOUBLE PRECISION GENERATED ALWAYS AS (ask_px - bid_px) STORED,
    dataset        TEXT           DEFAULT 'GLBX.MDP3',
    UNIQUE (instrument_id, ts_event)
);

SELECT create_hypertable('databento.bbo_1s', 'ts_event',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX idx_bbo1s_symbol_ts ON databento.bbo_1s (symbol, ts_event DESC);


-- ============================================================
-- DEFINITIONS: Instrument definitions (options contract specs)
-- Source: Databento Definition schema (NQ.OPT)
-- ============================================================
CREATE TABLE databento.definitions (
    ts_event           TIMESTAMPTZ  NOT NULL,
    instrument_id      INTEGER      NOT NULL,
    symbol             TEXT         NOT NULL,
    -- Contract spec fields
    instrument_class   CHAR(1),               -- 'C'all, 'P'ut, 'F'uture
    strike_price       DOUBLE PRECISION,
    expiration         TIMESTAMPTZ,
    underlying         TEXT,                   -- Root symbol (NQ)
    exchange           TEXT,
    currency           TEXT         DEFAULT 'USD',
    min_price_increment DOUBLE PRECISION,
    multiplier         DOUBLE PRECISION,
    trading_reference_price DOUBLE PRECISION,
    settlement_price   DOUBLE PRECISION,
    open_interest      BIGINT,
    -- Metadata
    dataset            TEXT         DEFAULT 'GLBX.MDP3',
    raw_record         JSONB,                 -- Full definition record for fields we didn't explicitly extract
    UNIQUE (instrument_id, ts_event)
);

SELECT create_hypertable('databento.definitions', 'ts_event',
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE
);

CREATE INDEX idx_def_symbol ON databento.definitions (symbol);
CREATE INDEX idx_def_expiration ON databento.definitions (expiration);
CREATE INDEX idx_def_underlying_class ON databento.definitions (underlying, instrument_class);


-- ============================================================
-- STATISTICS: Daily settlement, hi/lo, volume, OI
-- Source: Databento Statistics schema
-- ============================================================
CREATE TABLE databento.statistics (
    ts_event       TIMESTAMPTZ    NOT NULL,
    ts_recv        TIMESTAMPTZ    NOT NULL,
    instrument_id  INTEGER        NOT NULL,
    symbol         TEXT           NOT NULL,
    stat_type      SMALLINT       NOT NULL,  -- Enum: settlement, hi, lo, volume, OI
    price          DOUBLE PRECISION,
    quantity       BIGINT,
    sequence       BIGINT,
    ts_ref         TIMESTAMPTZ,              -- Reference date for the stat
    update_action  SMALLINT,                 -- 1=New, 2=Delete
    stat_flags     SMALLINT,
    dataset        TEXT           DEFAULT 'GLBX.MDP3',
    UNIQUE (instrument_id, ts_event, stat_type, sequence)
);

SELECT create_hypertable('databento.statistics', 'ts_event',
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE
);


-- ============================================================
-- Compression policies (chunks older than 30 days)
-- ============================================================
ALTER TABLE databento.tbbo SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'instrument_id',
    timescaledb.compress_orderby = 'ts_event ASC'
);
SELECT add_compression_policy('databento.tbbo', INTERVAL '30 days');

ALTER TABLE databento.bbo_1s SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'instrument_id',
    timescaledb.compress_orderby = 'ts_event ASC'
);
SELECT add_compression_policy('databento.bbo_1s', INTERVAL '30 days');


-- ============================================================
-- Continuous aggregates for common query patterns
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
    -- BBO at last trade
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

### 7.3 Contract Mapping View

```sql
-- Active NQ options with DTE calculation
CREATE VIEW databento.active_nq_options AS
SELECT
    d.instrument_id,
    d.symbol,
    d.instrument_class,
    d.strike_price,
    d.expiration,
    EXTRACT(DAY FROM d.expiration - CURRENT_TIMESTAMP) AS dte,
    d.min_price_increment,
    d.multiplier,
    d.open_interest
FROM databento.definitions d
WHERE d.underlying = 'NQ'
  AND d.expiration > CURRENT_TIMESTAMP
  AND d.instrument_class IN ('C', 'P')
ORDER BY d.expiration, d.strike_price;
```

---

## 8. Live Streaming Service

### 8.1 Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                  Live Streaming Service                       │
│                                                              │
│  db.Live(reconnect_policy="reconnect")                       │
│  ├── subscribe(GLBX.MDP3, tbbo, NQ.FUT, parent)            │
│  ├── subscribe(GLBX.MDP3, bbo-1s, NQ.FUT, parent)          │
│  ├── subscribe(GLBX.MDP3, bbo-1s, NQ.OPT, parent)          │
│  ├── subscribe(GLBX.MDP3, definition, NQ.OPT, parent)      │
│  └── subscribe(GLBX.MDP3, statistics, NQ.FUT, parent)      │
│                                                              │
│  Callbacks:                                                  │
│  ├── record_handler() → route by record type                 │
│  │   ├── TradeMsg/MBP1Msg → databento.tbbo INSERT           │
│  │   ├── OHLCVMsg → databento.bbo_1s INSERT                 │
│  │   ├── InstrumentDefMsg → databento.definitions UPSERT    │
│  │   ├── StatMsg → databento.statistics INSERT              │
│  │   └── SystemMsg → log (heartbeat/warnings)               │
│  └── add_stream("live/YYYY-MM-DD/all.dbn") → raw archive    │
│                                                              │
│  Reconnection: automatic with intraday replay from last      │
│  ts_event to fill gaps                                       │
└──────────────────────────────────────────────────────────────┘
```

### 8.2 Implementation Pattern

```python
# src/acquisition/live_streamer.py

import databento as db
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("vtech.live")

class LiveStreamer:
    """Persistent live data streaming service with auto-reconnect."""
    
    def __init__(self, db_pool):
        self.db_pool = db_pool
        self.last_ts_event: dict[str, int] = {}  # schema → last ts_event (for recovery)
    
    def start(self):
        today = datetime.utcnow().strftime("%Y-%m-%d")
        archive_path = Path(f"/opt/vtech/data/raw/live/{today}/all.dbn")
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        
        client = db.Live(
            reconnect_policy="reconnect",
            heartbeat_interval_s=15,
        )
        
        # Multiple subscriptions on single session (same dataset)
        client.subscribe(dataset="GLBX.MDP3", schema="tbbo",
                         symbols="NQ.FUT", stype_in="parent")
        client.subscribe(dataset="GLBX.MDP3", schema="bbo-1s",
                         symbols="NQ.FUT", stype_in="parent")
        client.subscribe(dataset="GLBX.MDP3", schema="bbo-1s",
                         symbols="NQ.OPT", stype_in="parent")
        client.subscribe(dataset="GLBX.MDP3", schema="definition",
                         symbols="NQ.OPT", stype_in="parent")
        client.subscribe(dataset="GLBX.MDP3", schema="statistics",
                         symbols="NQ.FUT", stype_in="parent")
        
        # Dual output: callback for DB writes + file stream for raw archive
        client.add_callback(
            record_callback=self._handle_record,
            exception_callback=self._handle_error,
        )
        client.add_stream(str(archive_path))
        
        # Track reconnection gaps
        client.add_reconnect_callback(
            reconnect_callback=self._handle_reconnect,
        )
        
        client.start()
        logger.info("Live streaming started for GLBX.MDP3")
        client.block_for_close()
    
    def _handle_record(self, record: db.DBNRecord):
        """Route records by type to appropriate DB table."""
        if isinstance(record, db.TradeMsg):
            self._insert_tbbo(record)
        elif isinstance(record, db.OHLCVMsg):
            self._insert_bbo_1s(record)
        elif isinstance(record, db.InstrumentDefMsgV2):
            self._upsert_definition(record)
        elif isinstance(record, db.StatMsg):
            self._insert_statistics(record)
        elif isinstance(record, db.SystemMsg):
            if record.is_heartbeat:
                pass  # Expected
            else:
                logger.info(f"System message: {record.msg} (code={record.code})")
        elif isinstance(record, db.ErrorMsg):
            logger.error(f"Gateway error: {record.err} (code={record.code})")
    
    def _handle_error(self, exc: Exception):
        logger.exception(f"Callback error: {exc}")
    
    def _handle_reconnect(self, start, end):
        logger.warning(f"Reconnection gap: {start} → {end}")
    
    # Insert methods use self.db_pool — batch inserts for efficiency
    # Implementation details in src/ingestion/loader.py
```

### 8.3 Connection Limits (Standard Plan)

| Constraint | Limit |
|---|---|
| Simultaneous connections per dataset per team | 10 |
| Subscription rate | 10/second (throttled, not rejected) |
| Incoming connections per second from same IP | 5 |
| Heartbeat interval (minimum) | 5 seconds |

Our design uses **1 connection** with **5 subscriptions** — well within limits.

---

## 9. Feature Engineering Pipeline

### 9.1 Data Flow

```
TimescaleDB (Tier 2)
    │
    ├── tbbo table ──────────────── ▶ Order flow features
    │                                  (CVD, delta, absorption, VPIN)
    ├── bbo_1s (futures) ────────── ▶ Microstructure features
    │                                  (spread dynamics, book pressure)
    ├── bbo_1s (options) ────────── ▶ IV surface features
    │   + definitions                  (ATM IV, skew, butterfly, GEX)
    ├── statistics ──────────────── ▶ Daily context features
    │                                  (settlement, hi/lo, OI change)
    └── tbbo_1min (continuous agg)─ ▶ Candle structure, wavelets, VWAP
                                       (body ratio, CLV, volatility regime)
    │
    ▼
Feature Engine (src/features/engine.py)
    │
    ▼
Parquet cache (Tier 3) → ML Training
```

### 9.2 Feature Priority (from Master Blueprint)

| Priority | Feature Group | Source Schema | New Module |
|---|---|---|---|
| P1 | Liquidity withdrawal (bid/ask size drop) | `bbo_1s` (futures) | `book_pressure.py` |
| P2 | Aggressive trade flow (CVD, delta) | `tbbo` | `order_flow.py` (evolved) |
| P3 | Options IV acceleration | `bbo_1s` (options) + `definitions` | `options_surface.py` |
| P4 | Spread dynamics (widening = stress) | `bbo_1s` (futures) | `book_pressure.py` |
| P5 | Daily context (settlement, range) | `statistics` | `daily_context.py` |
| P6 | VPIN (toxicity proxy) | `tbbo` | `microstructure.py` |
| P7 | Wavelet decomposition | `tbbo_1min` | `wavelets.py` (evolved) |
| P8 | Cross-asset ES correlation | `tbbo` (ES) | `cross_asset.py` |
| P9 | GEX (gamma exposure proxy) | `definitions` + `bbo_1s` (options) | `options_surface.py` |

### 9.3 Feature Computation Pattern

```python
# Triggered by: cron / systemd timer after market close
# Or: real-time in sliding window during live session

# 1. Query date range from TimescaleDB
# 2. Compute features in pandas (vectorized operations)
# 3. Cache to Parquet for ML consumption

def build_features_for_date(target_date: str) -> Path:
    """Build all features for a single trading day."""
    
    engine = FeatureEngine(db_pool)
    
    # Load source data (parallel queries)
    tbbo_df = engine.load_tbbo(target_date)
    bbo_fut_df = engine.load_bbo_1s(target_date, symbol_filter="NQ%")
    bbo_opt_df = engine.load_bbo_1s(target_date, symbol_filter="NQ%OPT%")
    defs_df = engine.load_definitions(target_date)
    stats_df = engine.load_statistics(target_date)
    
    # Resample to 10s timesteps (model input resolution)
    features = engine.compute_all(
        tbbo=tbbo_df,
        bbo_futures=bbo_fut_df,
        bbo_options=bbo_opt_df,
        definitions=defs_df,
        statistics=stats_df,
        timestep="10s",
    )
    
    # Save to Parquet
    out_path = Path(f"/opt/vtech/data/parquet/features_{target_date}.parquet")
    features.to_parquet(out_path, engine="pyarrow", compression="zstd")
    return out_path
```

---

## 10. ML Training & Inference

### 10.1 Model Architecture (Inherited + Enhanced)

```
Input: Feature matrix (sequence_length=64 × num_features)
    │
    ▼
┌─── Attention-LSTM ───────────────────────┐
│  Multi-head self-attention (4 heads)     │
│  → LSTM (hidden=128, layers=2)           │
│  → Dropout (0.3)                         │
│  → Dense → Softmax (3 classes)           │
└──────────────────────────────────────────┘
    │
    ▼
Output: [BIG_UP, CONSOLIDATION, BIG_DOWN]
        (50-point threshold, forward windows [5,10,15,30] min)
```

### 10.2 Training Pipeline

```python
# Loads from Parquet cache (Tier 3)
# Same 10-step pipeline as existing ml/training/trainer.py:
#
# 1. Load feature parquets for date range
# 2. Compute features (if not cached)
# 3. Generate labels (3-class, 50pt threshold)
# 4. Temporal train/val/test split (70/15/15)
# 5. Normalize/standardize features
# 6. Build sequences (length=64, stride=1)
# 7. Compute class weights (handle imbalance)
# 8. Build model
# 9. Train with early stopping
# 10. Evaluate on test set
```

### 10.3 Checkpoint Management

```
/opt/vtech/data/checkpoints/
├── model_v001_20250715/
│   ├── model.keras              # Saved model weights
│   ├── config.json              # Training config snapshot
│   ├── feature_stats.json       # Mean/std for normalization
│   ├── metrics.json             # Test set performance
│   └── training_log.csv         # Epoch-by-epoch metrics
└── model_v002_20250801/
    └── ...
```

---

## 11. Service Orchestration

### 11.1 systemd Services

```ini
# /opt/vtech/systemd/vtech-live-streamer.service
[Unit]
Description=VTech Databento Live Streamer
After=postgresql.service
Wants=postgresql.service

[Service]
Type=simple
User=vtech
Group=vtech
WorkingDirectory=/opt/vtech
EnvironmentFile=/opt/vtech/.env
ExecStart=/opt/vtech/venv/bin/python -m src.acquisition.live_streamer
Restart=always
RestartSec=5
StandardOutput=append:/opt/vtech/logs/live-streamer.log
StandardError=append:/opt/vtech/logs/live-streamer.err

# Hardening
NoNewPrivileges=true
ProtectSystem=strict
ReadWritePaths=/opt/vtech/data /opt/vtech/logs

[Install]
WantedBy=multi-user.target
```

```ini
# /opt/vtech/systemd/vtech-feature-builder.service
# Runs as a oneshot triggered by timer for post-market feature computation
[Unit]
Description=VTech Feature Builder (post-market)
After=postgresql.service

[Service]
Type=oneshot
User=vtech
Group=vtech
WorkingDirectory=/opt/vtech
EnvironmentFile=/opt/vtech/.env
ExecStart=/opt/vtech/venv/bin/python -m src.features.engine --date=today
StandardOutput=append:/opt/vtech/logs/feature-builder.log
StandardError=append:/opt/vtech/logs/feature-builder.err
```

```ini
# /opt/vtech/systemd/vtech-feature-builder.timer
[Unit]
Description=Run feature builder after NQ close (daily)

[Timer]
# 17:15 UTC = 12:15 PM CT (after NQ RTH close at 16:00 CT / 21:00 UTC)
# Using 22:00 UTC to ensure all data is landed
OnCalendar=*-*-* 22:00:00 UTC
Persistent=true

[Install]
WantedBy=timers.target
```

### 11.2 Service Topology

```
systemd
├── postgresql.service                 # TimescaleDB (always running)
├── vtech-live-streamer.service        # Databento live feed (market hours)
├── vtech-feature-builder.timer        # Triggers post-market feature build
│   └── vtech-feature-builder.service  # Oneshot feature computation
└── prometheus-node-exporter.service   # System metrics (optional)
```

### 11.3 Operational Commands

```bash
# Service management
sudo systemctl start vtech-live-streamer
sudo systemctl status vtech-live-streamer
sudo journalctl -u vtech-live-streamer -f   # Live logs

# Manual feature build
/opt/vtech/venv/bin/python -m src.features.engine --date=2025-07-15

# Manual backfill
/opt/vtech/venv/bin/python scripts/backfill.py \
    --schema=tbbo --symbols=NQ.FUT --start=2024-07-01 --end=2025-07-01

# Cost check before download
/opt/vtech/venv/bin/python scripts/cost_check.py \
    --schema=tbbo --symbols=NQ.FUT --start=2024-07-01 --end=2025-07-01

# Data verification
/opt/vtech/venv/bin/python scripts/verify_data.py --date=2025-07-15
```

---

## 12. Configuration & Secrets

### 12.1 Environment File

```bash
# /opt/vtech/.env (chmod 600, owned by vtech user)

# ─── Databento ───
DATABENTO_API_KEY=db-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# ─── PostgreSQL / TimescaleDB ───
PGHOST=localhost
PGPORT=5432
PGDATABASE=vtech_market
PGUSER=vtech
PGPASSWORD=<generated-secure-password>

# ─── Application ───
VTECH_LOG_LEVEL=INFO
VTECH_DATA_DIR=/opt/vtech/data
VTECH_FEATURES_TIMESTEP=10s
VTECH_MODEL_CHECKPOINT_DIR=/opt/vtech/data/checkpoints

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
# src/common/config.py
from dataclasses import dataclass, field
from os import environ

@dataclass(frozen=True)
class AppConfig:
    databento_api_key: str = field(repr=False)  # Never log
    pg_dsn: str = field(repr=False)
    data_dir: str = "/opt/vtech/data"
    log_level: str = "INFO"
    features_timestep: str = "10s"
    label_threshold: int = 50
    sequence_length: int = 64
    forward_windows: list[int] = field(default_factory=lambda: [5, 10, 15, 30])
    trading_hours_start: str = "08:00"
    trading_hours_end: str = "11:00"
    timezone: str = "America/Chicago"

    @classmethod
    def from_env(cls) -> "AppConfig":
        fwd = environ.get("VTECH_FORWARD_WINDOWS", "5,10,15,30")
        return cls(
            databento_api_key=environ["DATABENTO_API_KEY"],
            pg_dsn=f"postgresql://{environ['PGUSER']}:{environ['PGPASSWORD']}"
                   f"@{environ.get('PGHOST','localhost')}:{environ.get('PGPORT','5432')}"
                   f"/{environ.get('PGDATABASE','vtech_market')}",
            data_dir=environ.get("VTECH_DATA_DIR", "/opt/vtech/data"),
            log_level=environ.get("VTECH_LOG_LEVEL", "INFO"),
            features_timestep=environ.get("VTECH_FEATURES_TIMESTEP", "10s"),
            label_threshold=int(environ.get("VTECH_LABEL_THRESHOLD", "50")),
            sequence_length=int(environ.get("VTECH_SEQUENCE_LENGTH", "64")),
            forward_windows=[int(x) for x in fwd.split(",")],
            trading_hours_start=environ.get("VTECH_TRADING_HOURS_START", "08:00"),
            trading_hours_end=environ.get("VTECH_TRADING_HOURS_END", "11:00"),
            timezone=environ.get("VTECH_TIMEZONE", "America/Chicago"),
        )
```

---

## 13. Monitoring & Observability

### 13.1 Key Metrics

| Metric | Source | Alert Threshold |
|---|---|---|
| Records ingested/second | Live streamer counter | < 1 during market hours |
| DB insert latency (p99) | Ingestion timer | > 100ms |
| Live stream heartbeat age | Last heartbeat timestamp | > 45s (heartbeat_interval + 10s buffer) |
| Disk usage (data partition) | Node exporter | > 85% |
| TimescaleDB chunk count | PG system catalog | > 500 uncompressed |
| Feature build duration | Timer service log | > 30 min |
| Model training loss | Training log | val_loss not decreasing for 10 epochs |

### 13.2 Logging Standard

```python
# Structured JSON logging via structlog
import structlog

logger = structlog.get_logger("vtech.live")

logger.info("record_ingested",
    schema="tbbo",
    symbol="NQU5",
    ts_event="2025-07-15T14:30:01.123456789Z",
    price=21450.25,
    size=3,
)
```

### 13.3 Health Check Script

```bash
#!/bin/bash
# scripts/healthcheck.sh — run via cron every 5 min during market hours

# Check live streamer is running
systemctl is-active --quiet vtech-live-streamer || echo "ALERT: live streamer down"

# Check last record timestamp
LAST_TS=$(psql -d vtech_market -tAc "
    SELECT EXTRACT(EPOCH FROM now() - max(ts_recv))
    FROM databento.tbbo
    WHERE ts_recv > now() - INTERVAL '1 hour'
")
if (( $(echo "$LAST_TS > 60" | bc -l) )); then
    echo "ALERT: no records in last ${LAST_TS}s"
fi

# Check disk space
DISK_PCT=$(df /opt/vtech/data --output=pcent | tail -1 | tr -d ' %')
if (( DISK_PCT > 85 )); then
    echo "ALERT: data disk at ${DISK_PCT}%"
fi
```

---

## 14. Network & Security

### 14.1 Firewall Rules

```bash
# Only PostgreSQL and SSH
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow from 127.0.0.1 to any port 5432  # Local Postgres only
sudo ufw enable
```

### 14.2 Security Checklist

- [ ] `DATABENTO_API_KEY` in `.env` file with `chmod 600` — never in code/logs/git
- [ ] PostgreSQL listens on `localhost` only (`listen_addresses = 'localhost'`)
- [ ] Dedicated `vtech` system user with minimal privileges
- [ ] SSH key-only authentication (no password)
- [ ] Unattended security updates enabled (`unattended-upgrades`)
- [ ] TimescaleDB credentials rotated quarterly
- [ ] No sensitive data in application logs (API key masked in config repr)

---

## 15. Backup & Recovery

### 15.1 Strategy

| Layer | Method | Frequency | Retention |
|---|---|---|---|
| Raw DBN files | Already immutable archive; rsync to cloud storage | Daily | Indefinite |
| TimescaleDB | `pg_dump` compressed | Daily at 23:00 UTC | 7 daily, 4 weekly |
| Parquet cache | Regenerable — no backup needed | — | — |
| Model checkpoints | rsync to cloud storage | After each training run | Last 5 versions |
| Configuration | Git-tracked (excluding secrets) | Every change | Git history |

### 15.2 Recovery Scenarios

| Scenario | Recovery Method | RTO |
|---|---|---|
| Live streamer crash | systemd auto-restart + intraday replay from last `ts_event` | < 30 seconds |
| Database corruption | Restore from `pg_dump` + re-ingest from raw DBN archive | < 2 hours |
| VM total loss | New VM setup + restore DBN archive from cloud + re-ingest | < 4 hours |
| Databento API outage | No new data; existing data unaffected; retry after outage | Depends on Databento |

---

## 16. Implementation Sequence

### Phase 1: Foundation (Day 1-2)

- [ ] Provision VM (Ubuntu 22.04, specs per Section 2)
- [ ] Install Python 3.12, PostgreSQL 16 + TimescaleDB
- [ ] Create `vtech` user, directory structure (Section 4)
- [ ] Configure `.env` with Databento API key
- [ ] `pip install databento` — verify connectivity with `metadata.list_datasets()`

### Phase 2: Database Setup (Day 2-3)

- [ ] Create `vtech_market` database
- [ ] Run schema DDL from Section 7
- [ ] Apply TimescaleDB tuning (Section 5)
- [ ] Test hypertable creation and compression policies

### Phase 3: Historical Backfill — Futures (Day 3-5)

- [ ] Run `cost_check.py` for NQ.FUT TBBO + BBO-1s (1 year)
- [ ] Download NQ.FUT TBBO via `timeseries.get_range` (or batch if >5GB)
- [ ] Download NQ.FUT BBO-1s
- [ ] Ingest DBN files → `databento.tbbo` and `databento.bbo_1s`
- [ ] Verify record counts and date coverage

### Phase 4: Historical Backfill — Options (Day 5-8)

- [ ] Download NQ.OPT BBO-1s (largest dataset — use `batch.submit_job`)
- [ ] Download NQ.OPT Definition
- [ ] Download NQ.FUT Statistics
- [ ] Ingest into respective tables
- [ ] Verify option chain completeness for sample dates

### Phase 5: Live Streaming (Day 8-10)

- [ ] Implement `LiveStreamer` class (Section 8)
- [ ] Test with limit of 100 records to verify callback routing
- [ ] Deploy as systemd service
- [ ] Monitor overnight — verify reconnect after Sunday maintenance window
- [ ] Validate DBN raw archive files are being written

### Phase 6: Ingestion Pipeline (Day 10-12)

- [ ] Build `src/ingestion/loader.py` — DBN → DataFrame → DB batch insert
- [ ] Implement dedup logic (ON CONFLICT DO NOTHING)
- [ ] Add data quality checks: gap detection, price sanity, symbol mapping
- [ ] Backfill from live DBN archive files to validate end-to-end

### Phase 7: Feature Engineering (Day 12-18)

- [ ] Port `order_flow.py` — adapt from candles table to `databento.tbbo` (richer data)
- [ ] Build `book_pressure.py` — new, using `databento.bbo_1s` bid/ask sizes
- [ ] Build `options_surface.py` — IV computation from options BBO + definitions
- [ ] Build `daily_context.py` — settlement / hi-lo from `databento.statistics`
- [ ] Build `microstructure.py` — VPIN, Kyle's Lambda
- [ ] Port remaining: `wavelets.py`, `candle_structure.py`, `time_context.py`, `vwap.py`
- [ ] Build `FeatureEngine.compute_all()` orchestrator
- [ ] Validate feature matrix shapes and NaN rates

### Phase 8: ML Training (Day 18-22)

- [ ] Port `config.py`, `labels.py`, `trainer.py` adapted for new data pipeline
- [ ] Port `attention_lstm.py` model
- [ ] Run initial training on 6-month feature set
- [ ] Evaluate: confusion matrix, per-class precision/recall
- [ ] Checkpoint management (Section 10.3)

### Phase 9: Cross-Asset (Day 22-25)

- [ ] Download ES.FUT TBBO (1 year)
- [ ] Build `cross_asset.py` — NQ/ES correlation, lead-lag
- [ ] Retrain model with cross-asset features
- [ ] Compare metrics: with vs without ES features

### Phase 10: Production Hardening (Day 25-30)

- [ ] Monitoring dashboards (Section 13)
- [ ] Backup automation (Section 15)
- [ ] Health check cron job
- [ ] Documentation of operational runbook
- [ ] Load test: simulate 5× message rate to verify headroom

---

## Appendix A: Databento Client Quick Reference

```python
import databento as db

# ─── Authentication ───
# Preferred: set DATABENTO_API_KEY env var
client_hist = db.Historical()           # Historical API
client_live = db.Live()                 # Live API

# ─── Cost check (free, always do before download) ───
cost = client_hist.metadata.get_cost(
    dataset="GLBX.MDP3", symbols="NQ.FUT",
    schema="tbbo", stype_in="parent",
    start="2024-07-01", end="2025-07-01",
)

# ─── Stream to file ───
data = client_hist.timeseries.get_range(
    dataset="GLBX.MDP3", symbols="NQ.FUT",
    schema="tbbo", stype_in="parent",
    start="2024-07-01", end="2024-07-02",
    path="nq_tbbo_20240701.dbn.zst",     # Saves directly to disk
)

# ─── Read back from file ───
store = db.DBNStore.from_file("nq_tbbo_20240701.dbn.zst")
df = store.to_df(pretty_ts=True, map_symbols=True)

# ─── Batch download for large requests ───
job = client_hist.batch.submit_job(
    dataset="GLBX.MDP3", symbols="NQ.OPT",
    schema="bbo-1s", stype_in="parent",
    start="2024-07-01", end="2025-07-01",
    encoding="dbn", compression="zstd",
    split_duration="day",
)
# Later: client_hist.batch.download(job_id=job["id"], output_dir="data/raw/")

# ─── Live streaming ───
client_live = db.Live(reconnect_policy="reconnect")
client_live.subscribe(dataset="GLBX.MDP3", schema="tbbo",
                      symbols="NQ.FUT", stype_in="parent")
client_live.add_callback(my_handler)
client_live.add_stream("live_archive.dbn")
client_live.start()
client_live.block_for_close()

# ─── Key output methods ───
# store.to_df()       → pandas DataFrame
# store.to_ndarray()  → numpy array (zero-copy)
# store.to_parquet()  → Apache Parquet
# store.to_csv()      → CSV
# store.to_file()     → DBN (native, recommended for archive)
```

## Appendix B: Databento → Existing ML Pipeline Field Mapping

| Existing Candle Field | Databento Source | Schema | Field Path |
|---|---|---|---|
| `open` | `tbbo_1min` continuous agg | `tbbo` → agg | `first(price)` |
| `high` | `tbbo_1min` continuous agg | `tbbo` → agg | `max(price)` |
| `low` | `tbbo_1min` continuous agg | `tbbo` → agg | `min(price)` |
| `close` | `tbbo_1min` continuous agg | `tbbo` → agg | `last(price)` |
| `volume` | `tbbo_1min` continuous agg | `tbbo` → agg | `sum(size)` |
| `bid_volume` | `tbbo_1min` continuous agg | `tbbo` → agg | `sum(size) WHERE side='B'` |
| `ask_volume` | `tbbo_1min` continuous agg | `tbbo` → agg | `sum(size) WHERE side='A'` |
| `vwap` | Compute from tick data | `tbbo` | `sum(price*size)/sum(size)` |
| `imp_volatility` | **NEW**: Compute from options BBO | `bbo_1s` (options) | Black-Scholes inversion |
| `bid_px` / `ask_px` | Direct from TBBO record | `tbbo` | `bid_px`, `ask_px` |
| `bid_sz` / `ask_sz` | Direct from TBBO record | `tbbo` | `bid_sz`, `ask_sz` |

**Key upgrade**: Existing pipeline had scalar `imp_volatility` from dxfeed that was loaded but never consumed. New pipeline computes full IV surface from actual options BBO data → enables `options_surface.py` features that were previously stubbed out.

## Appendix C: Storage Budget (1 Year Projection)

| Component | Compressed (DBN) | PostgreSQL (Uncompressed) | PostgreSQL (30d+ Compressed) |
|---|---|---|---|
| NQ.FUT TBBO | ~3-5 GB | ~20-30 GB | ~8-12 GB |
| NQ.FUT BBO-1s | ~2-4 GB | ~15-20 GB | ~6-8 GB |
| NQ.OPT BBO-1s (0DTE) | ~6-14 GB | ~60-150 GB | ~25-60 GB |
| NQ.OPT Definition | ~0.5 GB | ~3-5 GB | ~1-2 GB |
| NQ.FUT Statistics | ~0.2 GB | ~1-2 GB | ~0.5 GB |
| ES.FUT TBBO | ~3-5 GB | ~20-30 GB | ~8-12 GB |
| **Total** | **~15-29 GB** | **~119-237 GB** | **~49-95 GB** |

With TimescaleDB compression (typically 10-20× on time-series data), the effective on-disk footprint for data >30 days old drops dramatically. A 500 GB NVMe SSD provides ample headroom.
