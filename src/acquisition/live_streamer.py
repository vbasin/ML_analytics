"""Real-time Databento live streaming service.

Subscribes to GLBX.MDP3 for NQ futures + options, routes records to
TimescaleDB via callbacks, and archives raw DBN to disk.

Run as: python -m src.acquisition.live_streamer
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import databento as db
import psycopg
from databento_dbn import FIXED_PRICE_SCALE

from src.acquisition.schemas import DATASET
from src.common.db import get_dsn

logger = logging.getLogger("vtech.live")

_PX_SCALE = float(FIXED_PRICE_SCALE)


def _ns_to_dt(ns: int) -> datetime | None:
    """Convert nanosecond unix timestamp to timezone-aware datetime."""
    if ns is None or ns == 0:
        return None
    return datetime.fromtimestamp(ns / 1_000_000_000, tz=timezone.utc)


def _px(raw: int | float) -> float | None:
    """Convert Databento fixed-point price to float."""
    if raw is None:
        return None
    return float(raw) / _PX_SCALE


class LiveStreamer:
    """Persistent live streaming service with auto-reconnect."""

    def __init__(self) -> None:
        self._dsn = get_dsn()
        self._record_count = 0
        self._data_dir = Path(os.environ.get("VTECH_DATA_DIR", "/opt/vtech/data"))
        self._client: db.Live | None = None

    # ── public ───────────────────────────────────────────────────

    def start(self) -> None:
        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")
        archive_dir = self._data_dir / "raw" / "live" / today
        archive_dir.mkdir(parents=True, exist_ok=True)
        ts = now.strftime("%H%M%S")
        archive_path = archive_dir / f"stream_{ts}.dbn"

        client = db.Live(
            reconnect_policy="reconnect",
            heartbeat_interval_s=15,
        )
        self._client = client

        # Multiple subscriptions on a single TCP session
        client.subscribe(dataset=DATASET, schema="tbbo",
                         symbols="NQ.FUT", stype_in="parent")
        client.subscribe(dataset=DATASET, schema="bbo-1s",
                         symbols="NQ.FUT", stype_in="parent")
        client.subscribe(dataset=DATASET, schema="bbo-1s",
                         symbols="NQ.OPT", stype_in="parent")
        client.subscribe(dataset=DATASET, schema="definition",
                         symbols="NQ.OPT", stype_in="parent")
        client.subscribe(dataset=DATASET, schema="statistics",
                         symbols="NQ.FUT", stype_in="parent")

        client.add_callback(
            record_callback=self._handle_record,
            exception_callback=self._handle_error,
        )
        client.add_stream(str(archive_path))
        client.add_reconnect_callback(reconnect_callback=self._handle_reconnect)

        logger.info("live_start", extra={"dataset": DATASET, "archive": str(archive_path)})
        client.start()
        client.block_for_close()

    # ── callbacks ────────────────────────────────────────────────

    def _handle_record(self, record: db.DBNRecord) -> None:
        self._record_count += 1
        if self._record_count % 10_000 == 0:
            logger.info("live_progress", extra={"records": self._record_count})

        if isinstance(record, db.TBBOMsg):
            self._insert_tbbo(record)
        elif isinstance(record, db.BBOMsg):
            self._insert_bbo_1s(record)
        elif isinstance(record, db.InstrumentDefMsg):
            self._upsert_definition(record)
        elif isinstance(record, db.StatMsg):
            self._insert_statistics(record)
        elif isinstance(record, db.SystemMsg):
            if not record.is_heartbeat:
                logger.info("system_msg", extra={"msg": record.msg, "code": record.code})
        elif isinstance(record, db.ErrorMsg):
            logger.error("gateway_error", extra={"err": record.err, "code": record.code})

    def _handle_error(self, exc: Exception) -> None:
        logger.exception("callback_error", exc_info=exc)

    def _handle_reconnect(self, start, end) -> None:
        logger.warning("reconnect_gap", extra={"start": str(start), "end": str(end)})

    # ── helpers ──────────────────────────────────────────────────

    def _resolve_symbol(self, instrument_id: int) -> str:
        """Resolve instrument_id to symbol via the live client's symbology map."""
        if self._client is None:
            return ""
        sym = self._client.symbology_map.get(instrument_id, "")
        return str(sym) if sym else ""

    # ── database inserts ─────────────────────────────────────────

    def _insert_tbbo(self, rec: db.TBBOMsg) -> None:
        sql = """
            INSERT INTO databento.tbbo
                (ts_event, ts_recv, instrument_id, symbol, price, size,
                 side, action, flags, sequence,
                 bid_px, ask_px, bid_sz, ask_sz, bid_ct, ask_ct, ts_in_delta)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT DO NOTHING
        """
        lvl = rec.levels[0] if rec.levels else None
        with psycopg.connect(self._dsn) as conn:
            conn.execute(sql, (
                _ns_to_dt(rec.ts_event), _ns_to_dt(rec.ts_recv), rec.instrument_id,
                self._resolve_symbol(rec.instrument_id),
                _px(rec.price), int(rec.size),
                str(rec.side) if hasattr(rec, "side") else None,
                str(rec.action) if hasattr(rec, "action") else None,
                int(getattr(rec, "flags", 0)), getattr(rec, "sequence", None),
                _px(lvl.bid_px) if lvl else None,
                _px(lvl.ask_px) if lvl else None,
                int(lvl.bid_sz) if lvl else None,
                int(lvl.ask_sz) if lvl else None,
                int(lvl.bid_ct) if lvl else None,
                int(lvl.ask_ct) if lvl else None,
                int(getattr(rec, "ts_in_delta", 0)),
            ))

    def _insert_bbo_1s(self, rec: db.BBOMsg) -> None:
        sql = """
            INSERT INTO databento.bbo_1s
                (ts_event, instrument_id, symbol,
                 bid_px, ask_px, bid_sz, ask_sz)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT DO NOTHING
        """
        lvl = rec.levels[0] if rec.levels else None
        with psycopg.connect(self._dsn) as conn:
            conn.execute(sql, (
                _ns_to_dt(rec.ts_event), rec.instrument_id,
                self._resolve_symbol(rec.instrument_id),
                _px(lvl.bid_px) if lvl else None,
                _px(lvl.ask_px) if lvl else None,
                int(lvl.bid_sz) if lvl else None,
                int(lvl.ask_sz) if lvl else None,
            ))

    def _upsert_definition(self, rec: db.InstrumentDefMsg) -> None:
        sql = """
            INSERT INTO databento.definitions
                (ts_event, instrument_id, symbol, instrument_class, strike_price,
                 expiration, underlying, exchange, currency, min_price_increment,
                 multiplier, trading_reference_price, settlement_price, open_interest)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (instrument_id, ts_event) DO UPDATE SET
                settlement_price = EXCLUDED.settlement_price,
                open_interest = EXCLUDED.open_interest
        """
        with psycopg.connect(self._dsn) as conn:
            conn.execute(sql, (
                _ns_to_dt(rec.ts_event), rec.instrument_id,
                self._resolve_symbol(rec.instrument_id),
                getattr(rec, "instrument_class", None),
                _px(rec.strike_price) if hasattr(rec, "strike_price") else None,
                _ns_to_dt(getattr(rec, "expiration", None)),
                getattr(rec, "underlying", None),
                getattr(rec, "exchange", None),
                getattr(rec, "currency", "USD"),
                _px(rec.min_price_increment) if hasattr(rec, "min_price_increment") else None,
                _px(rec.multiplier) if hasattr(rec, "multiplier") else None,
                _px(rec.trading_reference_price) if hasattr(rec, "trading_reference_price") else None,
                _px(rec.settlement_price) if hasattr(rec, "settlement_price") else None,
                int(rec.open_interest) if hasattr(rec, "open_interest") else None,
            ))

    def _insert_statistics(self, rec: db.StatMsg) -> None:
        sql = """
            INSERT INTO databento.statistics
                (ts_event, ts_recv, instrument_id, symbol, stat_type,
                 price, quantity, sequence, ts_ref, update_action, stat_flags)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT DO NOTHING
        """
        with psycopg.connect(self._dsn) as conn:
            conn.execute(sql, (
                _ns_to_dt(rec.ts_event), _ns_to_dt(rec.ts_recv), rec.instrument_id,
                self._resolve_symbol(rec.instrument_id),
                int(rec.stat_type),
                _px(rec.price) if hasattr(rec, "price") else None,
                int(rec.quantity) if hasattr(rec, "quantity") else None,
                getattr(rec, "sequence", None),
                _ns_to_dt(getattr(rec, "ts_ref", None)),
                str(rec.update_action) if hasattr(rec, "update_action") else None,
                int(rec.stat_flags) if hasattr(rec, "stat_flags") else None,
            ))


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    from src.common.logging import setup_logging
    setup_logging()
    LiveStreamer().start()


if __name__ == "__main__":
    main()
