#!/usr/bin/env python3
"""Historical backfill script — download + ingest for all priority groups.

Usage:
    # Cost check only (no download)
    python scripts/databento_backfill.py --dry-run

    # Download + ingest P0 only (NQ.FUT tbbo + bbo-1s)
    python scripts/databento_backfill.py --priority 0 --start 2025-04-06 --end 2026-04-06

    # Download + ingest all priorities
    python scripts/databento_backfill.py --start 2025-04-06 --end 2026-04-06

    # Only specific symbols/schema
    python scripts/databento_backfill.py --symbols NQ.FUT --schema tbbo

NOTE: Raw .dbn.zst files are ALWAYS retained after ingestion.
      Cleanup is a manual operation — this script never deletes data.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

import databento as db

from src.acquisition.historical import cost_check, download_range, submit_batch
from src.acquisition.schemas import BACKFILL_JOBS, DATASET
from src.ingestion.loader import ingest_file

logger = logging.getLogger("vtech.backfill")

# Schemas >5 GB typically → use batch API
BATCH_SCHEMAS = {"NQ.OPT:bbo-1s"}


def _get_billable_size(symbols: str, schema: str, start: str, end: str, stype_in: str) -> int:
    """Return estimated download size in bytes."""
    client = db.Historical()
    return client.metadata.get_billable_size(
        dataset=DATASET, symbols=symbols, schema=schema,
        stype_in=stype_in, start=start, end=end,
    )


def main() -> None:
    today = date.today()
    default_start = (today - timedelta(days=365)).isoformat()
    default_end = today.isoformat()

    parser = argparse.ArgumentParser(description="Databento historical backfill")
    parser.add_argument("--start", default=default_start)
    parser.add_argument("--end", default=default_end)
    parser.add_argument("--priority", type=int, default=None, help="Only run jobs at this priority level")
    parser.add_argument("--symbols", help="Override symbols (skip BACKFILL_JOBS)")
    parser.add_argument("--schema", help="Override schema (requires --symbols)")
    parser.add_argument("--stype-in", default="parent", help="Symbology type: parent, continuous")
    parser.add_argument("--output-dir", default="/opt/vtech/data/raw/backfill")
    parser.add_argument("--dry-run", action="store_true", help="Cost + size check only, no download")
    parser.add_argument("--skip-ingest", action="store_true")
    args = parser.parse_args()

    if args.symbols and args.schema:
        jobs = [{"symbols": args.symbols, "schema": args.schema,
                 "stype_in": args.stype_in, "priority": 0}]
    else:
        jobs = BACKFILL_JOBS
        if args.priority is not None:
            jobs = [j for j in jobs if j["priority"] == args.priority]

    for job in jobs:
        sym, schema = job["symbols"], job["schema"]
        stype = job.get("stype_in", "parent")
        logger.info(f"--- {sym} / {schema} (P{job['priority']}) ---")

        cost = cost_check(sym, schema, args.start, args.end, stype)
        size = _get_billable_size(sym, schema, args.start, args.end, stype)
        size_gb = size / 1e9
        print(f"  Cost: ${cost:.2f}  |  Download: {size_gb:.2f} GB")

        if args.dry_run:
            continue

        key = f"{sym}:{schema}"
        subdir = sym.lower().replace(".", "_")
        out_dir = Path(args.output_dir) / subdir

        if key in BATCH_SCHEMAS:
            job_id = submit_batch(sym, schema, args.start, args.end, stype)
            print(f"  Batch job submitted: {job_id}")
            print(f"  Monitor: https://databento.com/portal/download-center")
            continue

        path = download_range(sym, schema, args.start, args.end, out_dir, stype)
        print(f"  Downloaded: {path} ({path.stat().st_size / 1e9:.2f} GB)")

        if not args.skip_ingest:
            n = ingest_file(path, schema)
            print(f"  Ingested {n:,} rows")
            print(f"  Raw file retained: {path}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    main()
