"""DBN file → TimescaleDB loader.

Optimized for large files (100M+ rows):
- COPY-based bulk loading via staging table (10-50x faster than executemany)
- Vectorized pandas → CSV (no Python row loops)
- Chunked DataFrameIterator: never loads full file into memory
- ON CONFLICT DO NOTHING for idempotent re-runs
- Live progress bar with rows/s and ETA
"""
from __future__ import annotations

import csv
import io
import os
import sys
import time
import logging
from pathlib import Path

import databento as db
import psycopg
from dotenv import load_dotenv

load_dotenv("/opt/vtech/.env")
logger = logging.getLogger("vtech.loader")

_DF_CHUNK = 500_000

# Empirical compressed bytes-per-row for progress estimation
_BYTES_PER_ROW_EST = {"bbo-1s": 12, "tbbo": 25}


def get_dsn():
    return (
        f"postgresql://{os.environ['PGUSER']}:{os.environ['PGPASSWORD']}"
        f"@{os.environ.get('PGHOST','localhost')}:{os.environ.get('PGPORT','5432')}"
        f"/{os.environ.get('PGDATABASE','vtech_market')}"
    )


def _estimate_total(path: str, schema: str) -> int | None:
    """Estimate total rows from compressed file size."""
    bpr = _BYTES_PER_ROW_EST.get(schema)
    if bpr:
        return os.path.getsize(path) // bpr
    return None


def _progress_bar(current: int, total: int | None, t0: float,
                  new_rows: int, width: int = 40) -> str:
    """Format a live terminal progress bar."""
    elapsed = time.time() - t0
    rate = current / elapsed if elapsed > 0 else 0

    if total and total > 0:
        pct = min(current / total, 1.0)
        filled = int(width * pct)
        bar = '█' * filled + '░' * (width - filled)
        eta = (total - current) / rate if rate > 0 else 0
        eta_m, eta_s = divmod(int(eta), 60)
        return (
            f"\r  [{bar}] {pct:6.1%}  "
            f"{current:>12,} / ~{total:,}  "
            f"({new_rows:,} new)  "
            f"{rate:,.0f} rows/s  ETA {eta_m}m{eta_s:02d}s"
        )
    else:
        m, s = divmod(int(elapsed), 60)
        return (
            f"\r  {current:>12,} rows  "
            f"({new_rows:,} new)  "
            f"{rate:,.0f} rows/s  {m}m{s:02d}s elapsed"
        )


def _copy_chunk(cur, conn, table: str, db_cols: list[str], df_prepped) -> int:
    """COPY a DataFrame chunk into target via staging table.

    10-50x faster than executemany(): uses PG bulk COPY protocol
    with vectorized pandas CSV generation (no Python row loops).
    """
    cols_sql = ", ".join(db_cols)
    cur.execute("DROP TABLE IF EXISTS _stg")
    # CREATE TABLE AS SELECT ... WHERE false → same types, ZERO constraints
    cur.execute(
        f"CREATE TEMP TABLE _stg AS SELECT {cols_sql} FROM {table} WHERE false"
    )

    # Vectorized CSV generation (pandas C engine)
    buf = io.BytesIO()
    df_prepped.to_csv(
        buf, header=False, index=False, na_rep='',
        quoting=csv.QUOTE_MINIMAL, encoding='utf-8',
    )
    buf.seek(0)

    # Bulk COPY into staging
    with cur.copy(
        f"COPY _stg ({cols_sql}) FROM STDIN WITH (FORMAT csv, NULL '')"
    ) as copy:
        while block := buf.read(1 << 16):  # 64 KB blocks
            copy.write(block)

    # Dedup upsert: staging → target
    cur.execute(
        f"INSERT INTO {table} ({cols_sql}) "
        f"SELECT {cols_sql} FROM _stg ON CONFLICT DO NOTHING"
    )
    inserted = cur.rowcount
    conn.commit()
    return inserted


# ── Schema-specific column prep (vectorized, no iterrows) ──────

_TBBO_RENAME = {
    'bid_px_00': 'bid_px', 'ask_px_00': 'ask_px',
    'bid_sz_00': 'bid_sz', 'ask_sz_00': 'ask_sz',
    'bid_ct_00': 'bid_ct', 'ask_ct_00': 'ask_ct',
}
_TBBO_DB_COLS = [
    'ts_event', 'ts_recv', 'instrument_id', 'symbol', 'price', 'size',
    'side', 'action', 'flags', 'sequence',
    'bid_px', 'ask_px', 'bid_sz', 'ask_sz', 'bid_ct', 'ask_ct', 'ts_in_delta',
]


def _prep_tbbo(df):
    """Vectorized TBBO DataFrame preparation."""
    df = df.reset_index().rename(columns=_TBBO_RENAME)
    df = df[df['ts_event'].notna()].copy()
    if 'side' in df.columns:
        df['side'] = df['side'].str[:1]
    if 'action' in df.columns:
        df['action'] = df['action'].str[:1]
    if 'ts_in_delta' not in df.columns:
        df['ts_in_delta'] = None
    return df[_TBBO_DB_COLS]


_BBO_RENAME = {
    'bid_px_00': 'bid_px', 'ask_px_00': 'ask_px',
    'bid_sz_00': 'bid_sz', 'ask_sz_00': 'ask_sz',
}
_BBO_DB_COLS = [
    'ts_event', 'instrument_id', 'symbol',
    'bid_px', 'ask_px', 'bid_sz', 'ask_sz',
]


def _prep_bbo_1s(df):
    """Vectorized BBO-1s DataFrame preparation."""
    df = df.reset_index().rename(columns=_BBO_RENAME)
    df = df[df['ts_event'].notna()].copy()
    return df[_BBO_DB_COLS]


# ── Generic COPY-based ingestion engine ─────────────────────────

def _ingest_chunked(dbn_path: str, table: str, db_cols: list[str],
                    prep_fn, label: str, schema: str,
                    total_hint: int | None = None) -> int:
    """COPY-based chunked ingestion with live progress bar."""
    store = db.DBNStore.from_file(dbn_path)
    total_est = total_hint or _estimate_total(dbn_path, schema)
    chunks = store.to_df(
        pretty_ts=True, map_symbols=True, price_type="float", count=_DF_CHUNK,
    )

    total_read = 0
    total_inserted = 0
    t0 = time.time()

    est_str = f" (~{total_est:,} est)" if total_est else ""
    logger.info(f"Starting {label} ingestion{est_str}: {dbn_path}")
    print(f"\n  {label} ingestion{est_str}")

    with psycopg.connect(get_dsn()) as conn:
        cur = conn.cursor()
        cur.execute("SET synchronous_commit = off")
        cur.execute("SET work_mem = '256MB'")

        for chunk_num, df in enumerate(chunks):
            prepped = prep_fn(df)
            if len(prepped) == 0:
                total_read += len(df)
                continue

            inserted = _copy_chunk(cur, conn, table, db_cols, prepped)
            total_read += len(df)
            total_inserted += inserted

            bar = _progress_bar(total_read, total_est, t0, total_inserted)
            sys.stdout.write(bar)
            sys.stdout.flush()

    elapsed = time.time() - t0
    m, s = divmod(int(elapsed), 60)
    rate = total_read / elapsed if elapsed > 0 else 0
    sys.stdout.write('\n')
    summary = (
        f"{label} done: {total_read:,} read, {total_inserted:,} new "
        f"in {m}m{s:02d}s ({rate:,.0f} rows/s)"
    )
    logger.info(summary)
    print(f"  {summary}")
    return total_inserted


def ingest_tbbo(dbn_path: str, batch_size: int = 10_000,
                total_hint: int | None = None) -> int:
    """Load a TBBO .dbn.zst file into databento.tbbo."""
    return _ingest_chunked(
        dbn_path, 'databento.tbbo', _TBBO_DB_COLS,
        _prep_tbbo, 'TBBO', 'tbbo', total_hint,
    )


def ingest_bbo_1s(dbn_path: str, batch_size: int = 10_000,
                  total_hint: int | None = None) -> int:
    """Load a BBO-1s .dbn.zst file into databento.bbo_1s."""
    return _ingest_chunked(
        dbn_path, 'databento.bbo_1s', _BBO_DB_COLS,
        _prep_bbo_1s, 'BBO-1s', 'bbo-1s', total_hint,
    )


def ingest_definition(dbn_path: str, batch_size: int = 5000) -> int:
    """Load a Definition .dbn.zst file into databento.definitions table."""
    store = db.DBNStore.from_file(dbn_path)
    df = store.to_df(pretty_ts=True, map_symbols=True, price_type="float")

    logger.info(f"Loaded {len(df):,} definition records from {dbn_path}")
    df = df.reset_index()

    insert_sql = """
        INSERT INTO databento.definitions
            (ts_event, instrument_id, symbol, instrument_class, strike_price,
             expiration, underlying, exchange, currency, min_price_increment,
             multiplier, trading_reference_price, settlement_price, open_interest)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
    """

    with psycopg.connect(get_dsn()) as conn:
        with conn.cursor() as cur:
            rows = []
            for i, row in df.iterrows():
                rows.append((
                    row.get('ts_event'),
                    int(row.get('instrument_id', 0)),
                    str(row.get('symbol', '')),
                    str(row.get('instrument_class', ''))[:1] if row.get('instrument_class') else None,
                    float(row['strike_price']) if row.get('strike_price') else None,
                    row.get('expiration'),
                    str(row['underlying']) if row.get('underlying') else None,
                    str(row['exchange']) if row.get('exchange') else None,
                    str(row.get('currency', 'USD')),
                    float(row['min_price_increment']) if row.get('min_price_increment') else None,
                    float(row['multiplier']) if row.get('multiplier') else None,
                    float(row['trading_reference_price']) if row.get('trading_reference_price') else None,
                    float(row['settlement_price']) if row.get('settlement_price') else None,
                    int(row['open_interest']) if row.get('open_interest') else None,
                ))

                if len(rows) >= batch_size:
                    cur.executemany(insert_sql, rows)
                    conn.commit()
                    logger.info(f"  Inserted batch ({i+1:,} / {len(df):,})")
                    rows = []

            if rows:
                cur.executemany(insert_sql, rows)
                conn.commit()

    logger.info(f"Ingestion complete: {len(df):,} definition records")
    return len(df)


def ingest_statistics(dbn_path: str, batch_size: int = 5000) -> int:
    """Load a Statistics .dbn.zst file into databento.statistics table."""
    store = db.DBNStore.from_file(dbn_path)
    df = store.to_df(pretty_ts=True, map_symbols=True, price_type="float")

    logger.info(f"Loaded {len(df):,} statistics records from {dbn_path}")
    df = df.reset_index()

    insert_sql = """
        INSERT INTO databento.statistics
            (ts_event, ts_recv, instrument_id, symbol, stat_type, price,
             quantity, sequence, ts_ref, update_action, stat_flags)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
    """

    with psycopg.connect(get_dsn()) as conn:
        with conn.cursor() as cur:
            rows = []
            for i, row in df.iterrows():
                rows.append((
                    row.get('ts_event'),
                    row.get('ts_recv'),
                    int(row.get('instrument_id', 0)),
                    str(row.get('symbol', '')),
                    int(row.get('stat_type', 0)),
                    float(row['price']) if row.get('price') else None,
                    int(row['quantity']) if row.get('quantity') else None,
                    int(row['sequence']) if row.get('sequence') else None,
                    row.get('ts_ref'),
                    int(row['update_action']) if row.get('update_action') is not None else None,
                    int(row['stat_flags']) if row.get('stat_flags') is not None else None,
                ))

                if len(rows) >= batch_size:
                    cur.executemany(insert_sql, rows)
                    conn.commit()
                    logger.info(f"  Inserted batch ({i+1:,} / {len(df):,})")
                    rows = []

            if rows:
                cur.executemany(insert_sql, rows)
                conn.commit()

    logger.info(f"Ingestion complete: {len(df):,} statistics records")
    return len(df)


# ── Dispatcher ──────────────────────────────────────────────────

_INGEST_MAP = {
    "tbbo":       ingest_tbbo,
    "bbo-1s":     ingest_bbo_1s,
    "definition": ingest_definition,
    "statistics": ingest_statistics,
}


def ingest_file(path: str | Path, schema: str, batch_size: int = 5000,
                total_hint: int | None = None) -> int:
    """Dispatch ingestion to the correct function based on schema name.

    Returns the number of rows ingested.
    """
    func = _INGEST_MAP.get(schema)
    if func is None:
        raise ValueError(f"Unknown schema '{schema}'. Expected one of: {sorted(_INGEST_MAP)}")
    kwargs: dict = {"batch_size": batch_size}
    if total_hint is not None and schema in ("tbbo", "bbo-1s"):
        kwargs["total_hint"] = total_hint
    return func(str(path), **kwargs)


if __name__ == "__main__":
    import argparse as _ap
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    p = _ap.ArgumentParser(description="Ingest a .dbn.zst file into TimescaleDB")
    p.add_argument("schema", choices=sorted(_INGEST_MAP))
    p.add_argument("path", help="Path to .dbn.zst file")
    p.add_argument("--total", type=int, default=None,
                   help="Known total row count (improves progress ETA)")
    args = p.parse_args()

    n = ingest_file(args.path, args.schema, total_hint=args.total)
    print(f"\nIngested {n:,} rows")
