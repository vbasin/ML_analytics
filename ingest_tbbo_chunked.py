"""Chunked TBBO ingestion — never loads full file into memory."""
import databento as db, psycopg, os, sys
from dotenv import load_dotenv
load_dotenv("/opt/vtech/.env")

DSN = f"postgresql://{os.environ['PGUSER']}:{os.environ['PGPASSWORD']}@localhost:5432/{os.environ.get('PGDATABASE','vtech_market')}"
DBN = sys.argv[1] if len(sys.argv) > 1 else "/opt/vtech/data/raw/backfill/nq_futures/nq_fut_tbbo_20250401_20260401.dbn.zst"
BATCH = 10_000
SQL = """INSERT INTO databento.tbbo
  (ts_event,ts_recv,instrument_id,symbol,price,size,side,action,flags,sequence,
   bid_px,ask_px,bid_sz,ask_sz,bid_ct,ask_ct,ts_in_delta)
  VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
  ON CONFLICT DO NOTHING"""

store = db.DBNStore.from_file(DBN)
n = 0
with psycopg.connect(DSN) as conn:
    cur = conn.cursor()
    batch = []
    for rec in store:
        row = (rec.ts_event, rec.ts_recv, rec.instrument_id,
               rec.pretty_symbol if hasattr(rec,'pretty_symbol') else str(rec.instrument_id),
               rec.price/1e9, rec.size, chr(rec.side) if rec.side else None,
               chr(rec.action) if rec.action else None, rec.flags, rec.sequence,
               rec.levels[0].bid_px/1e9, rec.levels[0].ask_px/1e9,
               rec.levels[0].bid_sz, rec.levels[0].ask_sz,
               rec.levels[0].bid_ct, rec.levels[0].ask_ct, rec.ts_in_delta)
        batch.append(row)
        if len(batch) >= BATCH:
            cur.executemany(SQL, batch)
            conn.commit()
            n += len(batch)
            print(f"\r{n:>12,} rows inserted", end="", flush=True)
            batch = []
    if batch:
        cur.executemany(SQL, batch)
        conn.commit()
        n += len(batch)
print(f"\nDone: {n:,} rows inserted")
