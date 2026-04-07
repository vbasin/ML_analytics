#!/usr/bin/env python3
"""Parallel direct-COPY ingestion of NQ.OPT bbo-1s daily files.

Designed for one-time bulk loading of batch-downloaded daily .dbn.zst files.
Uses 4 parallel workers with direct COPY (no staging table, no ON CONFLICT).

Usage:
    # Dry run — show what would be ingested
    python scripts/ingest_nq_opt.py --dry-run

    # Full ingest with 4 workers
    python scripts/ingest_nq_opt.py

    # Custom parallelism
    python scripts/ingest_nq_opt.py --workers 2

Prerequisites:
    - Files downloaded via scripts/download_nq_opt.py
    - Table databento.bbo_1s must exist
"""
from __future__ import annotations

import argparse
import csv
import io
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import databento as db
import psycopg
from dotenv import load_dotenv

load_dotenv("/opt/vtech/.env")

DATA_DIR = Path("/opt/vtech/data/raw/backfill/nq_opt/GLBX-20260402-S4EMNFKM8K")
TABLE = "databento.bbo_1s"
CHUNK_SIZE = 500_000

# Columns to COPY — excludes 'spread' (GENERATED) and 'dataset' (DEFAULT)
COPY_COLS = [
    "ts_event", "instrument_id", "symbol",
    "open", "high", "low", "close", "volume",
    "bid_px", "ask_px", "bid_sz", "ask_sz",
]

# DataFrame renames from DBN column names to DB column names
DF_RENAME = {"bid_px_00": "bid_px", "ask_px_00": "ask_px",
             "bid_sz_00": "bid_sz", "ask_sz_00": "ask_sz"}

logger = logging.getLogger("ingest_nq_opt")


def get_dsn() -> str:
    return (
        f"postgresql://{os.environ['PGUSER']}:{os.environ['PGPASSWORD']}"
        f"@{os.environ.get('PGHOST', 'localhost')}:{os.environ.get('PGPORT', '5432')}"
        f"/{os.environ.get('PGDATABASE', 'vtech_market')}"
    )


def ingest_one_file(path: str) -> tuple[str, int, float]:
    """Ingest a single daily .dbn.zst file via direct COPY. Returns (filename, rows, seconds)."""
    load_dotenv("/opt/vtech/.env")
    fname = os.path.basename(path)
    t0 = time.time()

    store = db.DBNStore.from_file(path)
    chunks = store.to_df(
        pretty_ts=True, map_symbols=True, price_type="float", count=CHUNK_SIZE,
    )

    total_rows = 0
    cols_sql = ", ".join(COPY_COLS)

    with psycopg.connect(get_dsn()) as conn:
        cur = conn.cursor()
        cur.execute("SET synchronous_commit = off")
        cur.execute("SET work_mem = '256MB'")

        for df in chunks:
            df = df.reset_index().rename(columns=DF_RENAME)
            df = df[df["ts_event"].notna()].copy()
            if len(df) == 0:
                continue

            prepped = df[COPY_COLS]

            # Vectorized CSV generation
            buf = io.BytesIO()
            prepped.to_csv(buf, header=False, index=False, na_rep="",
                           quoting=csv.QUOTE_MINIMAL, encoding="utf-8")
            buf.seek(0)

            # Direct COPY into target — no staging table
            with cur.copy(
                f"COPY {TABLE} ({cols_sql}) FROM STDIN WITH (FORMAT csv, NULL '')"
            ) as copy:
                while block := buf.read(1 << 16):
                    copy.write(block)

            conn.commit()
            total_rows += len(prepped)

    elapsed = time.time() - t0
    return fname, total_rows, elapsed


def drop_indexes(dsn: str) -> list[str]:
    """Drop non-essential indexes for bulk load speed. Returns DDL to recreate them."""
    recreate = []
    with psycopg.connect(dsn) as conn:
        cur = conn.cursor()

        # Get all indexes
        cur.execute("""
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE schemaname = 'databento' AND tablename = 'bbo_1s'
        """)
        indexes = cur.fetchall()

        for name, defn in indexes:
            # Keep the unique constraint index (needed for ON CONFLICT if rerun)
            # Drop the others for speed
            if "UNIQUE" in defn.upper():
                logger.info(f"  Dropping unique constraint index: {name}")
                # Find and drop the constraint first
                cur.execute("""
                    SELECT conname FROM pg_constraint
                    WHERE conrelid = 'databento.bbo_1s'::regclass
                    AND contype = 'u'
                """)
                for (conname,) in cur.fetchall():
                    recreate.append(defn)
                    recreate.append(
                        f"ALTER TABLE {TABLE} ADD CONSTRAINT {conname} "
                        f"UNIQUE (instrument_id, ts_event)"
                    )
                    cur.execute(f"ALTER TABLE {TABLE} DROP CONSTRAINT {conname}")
                    logger.info(f"    Dropped constraint: {conname}")
            else:
                recreate.append(defn)
                cur.execute(f"DROP INDEX IF EXISTS databento.{name}")
                logger.info(f"  Dropped index: {name}")

        conn.commit()
    return recreate


def rebuild_indexes(dsn: str, ddl_list: list[str]):
    """Rebuild indexes from saved DDL."""
    with psycopg.connect(dsn) as conn:
        cur = conn.cursor()
        for ddl in ddl_list:
            logger.info(f"  Rebuilding: {ddl[:80]}...")
            cur.execute(ddl)
        conn.commit()


def tune_pg_for_bulk(dsn: str):
    """Temporarily tune PG for bulk loading."""
    with psycopg.connect(dsn) as conn:
        cur = conn.cursor()
        cur.execute("ALTER SYSTEM SET max_wal_size = '4GB'")
        cur.execute("ALTER SYSTEM SET checkpoint_completion_target = 0.9")
        cur.execute("ALTER SYSTEM SET wal_buffers = '64MB'")
        cur.execute("SELECT pg_reload_conf()")
        conn.commit()
    logger.info("  PG tuned for bulk load")


def reset_pg_tuning(dsn: str):
    """Reset PG tuning to defaults."""
    with psycopg.connect(dsn) as conn:
        cur = conn.cursor()
        cur.execute("ALTER SYSTEM RESET max_wal_size")
        cur.execute("ALTER SYSTEM RESET checkpoint_completion_target")
        cur.execute("ALTER SYSTEM RESET wal_buffers")
        cur.execute("SELECT pg_reload_conf()")
        conn.commit()
    logger.info("  PG settings reset")


def main():
    parser = argparse.ArgumentParser(description="Parallel NQ.OPT bbo-1s ingestion")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers (default: 4)")
    parser.add_argument("--dry-run", action="store_true", help="List files only")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR),
                        help="Directory containing .dbn.zst files")
    parser.add_argument("--skip-index-drop", action="store_true",
                        help="Don't drop indexes (slower but safer for re-runs)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("glbx-mdp3-*.bbo-1s.dbn.zst"))

    if not files:
        logger.error(f"No .dbn.zst files found in {data_dir}")
        sys.exit(1)

    total_bytes = sum(f.stat().st_size for f in files)
    logger.info(f"Found {len(files)} files ({total_bytes / 1e9:.1f} GB compressed)")

    if args.dry_run:
        for f in files:
            print(f"  {f.name:60s}  {f.stat().st_size / 1e6:.1f} MB")
        print(f"\nWould ingest {len(files)} files with {args.workers} workers")
        return

    dsn = get_dsn()

    # Phase 1: Prep
    logger.info("Phase 1: Preparing database for bulk load...")
    tune_pg_for_bulk(dsn)

    recreate_ddl = []
    if not args.skip_index_drop:
        logger.info("Dropping indexes and constraints...")
        recreate_ddl = drop_indexes(dsn)

    # Phase 2: Parallel ingest
    logger.info(f"Phase 2: Ingesting {len(files)} files with {args.workers} workers...")
    t0 = time.time()
    total_rows = 0
    completed = 0
    file_paths = [str(f) for f in files]

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(ingest_one_file, p): p for p in file_paths}

        for future in as_completed(futures):
            completed += 1
            try:
                fname, rows, elapsed = future.result()
                total_rows += rows
                rate = rows / elapsed if elapsed > 0 else 0
                logger.info(
                    f"  [{completed}/{len(files)}] {fname}: "
                    f"{rows:,} rows in {elapsed:.0f}s ({rate:,.0f} rows/s)"
                )
            except Exception as e:
                logger.error(f"  [{completed}/{len(files)}] FAILED {futures[future]}: {e}")

    wall_time = time.time() - t0
    m, s = divmod(int(wall_time), 60)
    logger.info(f"\nIngestion complete: {total_rows:,} rows in {m}m{s:02d}s")

    # Phase 3: Rebuild
    if recreate_ddl:
        logger.info("Phase 3: Rebuilding indexes and constraints...")
        rebuild_indexes(dsn, recreate_ddl)

    # Deduplicate if needed (belt-and-suspenders)
    logger.info("Phase 3b: Deduplicating any overlap with existing data...")
    with psycopg.connect(dsn) as conn:
        cur = conn.cursor()
        cur.execute(f"""
            DELETE FROM {TABLE} a USING {TABLE} b
            WHERE a.instrument_id = b.instrument_id
              AND a.ts_event = b.ts_event
              AND a.ctid > b.ctid
        """)
        dupes = cur.rowcount
        conn.commit()
        if dupes > 0:
            logger.info(f"  Removed {dupes:,} duplicate rows")
        else:
            logger.info("  No duplicates found")

    # Phase 4: Cleanup
    logger.info("Phase 4: Compressing and cleaning up...")
    reset_pg_tuning(dsn)

    with psycopg.connect(dsn) as conn:
        cur = conn.cursor()
        cur.execute("ANALYZE databento.bbo_1s")
        conn.commit()

    logger.info("Done!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    main()
