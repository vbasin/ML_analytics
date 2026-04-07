#!/usr/bin/env python3
"""Download NQ.OPT bbo-1s batch files from Databento Download Center.

Usage:
    python scripts/download_nq_opt.py                    # download all files
    python scripts/download_nq_opt.py --list             # list files only
    python scripts/download_nq_opt.py --resume            # skip already-downloaded files
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import databento as db
from dotenv import load_dotenv

load_dotenv("/opt/vtech/.env")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("download_nq_opt")

JOB_ID = "GLBX-20260402-S4EMNFKM8K"
OUTPUT_DIR = Path("/opt/vtech/data/raw/backfill/nq_opt")


def main():
    parser = argparse.ArgumentParser(description="Download NQ.OPT batch files")
    parser.add_argument("--list", action="store_true", help="List files only, no download")
    parser.add_argument("--resume", action="store_true", help="Skip files already on disk")
    args = parser.parse_args()

    client = db.Historical()

    # Verify job is still available
    jobs = client.batch.list_jobs(states=["done"])
    job = next((j for j in jobs if j["id"] == JOB_ID), None)
    if not job:
        logger.error(f"Job {JOB_ID} not found or not in 'done' state. May have expired.")
        sys.exit(1)
    logger.info(f"Job {JOB_ID}: state={job['state']}, expires={job.get('ts_expiration')}")

    # List files
    files = client.batch.list_files(job_id=JOB_ID)
    data_files = [f for f in files if f["filename"].endswith(".dbn.zst")]
    support_files = [f for f in files if not f["filename"].endswith(".dbn.zst")]

    total_size = sum(f["size"] for f in data_files)
    logger.info(f"Data files: {len(data_files)}, total: {total_size / 1e9:.1f} GB")

    if args.list:
        for f in data_files:
            print(f"  {f['filename']:60s}  {f['size'] / 1e9:.3f} GB")
        return

    # Download
    out = OUTPUT_DIR / JOB_ID
    out.mkdir(parents=True, exist_ok=True)

    # Download support files first
    for f in support_files:
        dest = out / f["filename"]
        if args.resume and dest.exists():
            continue
        logger.info(f"Downloading {f['filename']}...")
        client.batch.download(
            job_id=JOB_ID,
            output_dir=str(OUTPUT_DIR),
            filename_to_download=f["filename"],
        )

    # Download data files
    downloaded = 0
    skipped = 0
    for i, f in enumerate(data_files, 1):
        dest = out / f["filename"]
        if args.resume and dest.exists() and dest.stat().st_size == f["size"]:
            skipped += 1
            continue

        logger.info(f"[{i}/{len(data_files)}] {f['filename']} ({f['size'] / 1e9:.3f} GB)")
        client.batch.download(
            job_id=JOB_ID,
            output_dir=str(OUTPUT_DIR),
            filename_to_download=f["filename"],
        )
        downloaded += 1

    logger.info(f"Done: {downloaded} downloaded, {skipped} skipped (already on disk)")
    logger.info(f"Files at: {out}/")


if __name__ == "__main__":
    main()
