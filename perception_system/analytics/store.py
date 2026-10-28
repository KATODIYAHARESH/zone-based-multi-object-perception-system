"""SQLite event store: WAL journal, batched background writer, retention-aware.

The perception loop never blocks on disk I/O: writes are queued and committed
in batches by a dedicated thread (bounded queue; overflow is counted and logged,
never allowed to stall the video loop).
"""
from __future__ import annotations

import json
import logging
import queue
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from core.types import Alert, ZoneEvent

log = logging.getLogger(__name__)

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, type TEXT NOT NULL,
  zone_id TEXT, zone_name TEXT, track_id INTEGER, category TEXT, label TEXT,
  dwell_s REAL, x REAL, y REAL, reason TEXT);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE TABLE IF NOT EXISTS alerts (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, rule_id TEXT NOT NULL, rule_type TEXT,
  severity TEXT NOT NULL, message TEXT NOT NULL, zone_id TEXT, zone_name TEXT, track_id INTEGER,
  category TEXT, snapshot TEXT, data TEXT);
CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(ts);
CREATE TABLE IF NOT EXISTS counts (
  ts REAL NOT NULL, zone_id TEXT NOT NULL, category TEXT NOT NULL, count INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS idx_counts_ts ON counts(ts);
CREATE TABLE IF NOT EXISTS emotions (
  ts REAL NOT NULL, zone_id TEXT NOT NULL, emotion TEXT NOT NULL, count INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS idx_emotions_ts ON emotions(ts);
"""
TABLES = ("events", "alerts", "counts", "emotions")
Write = Tuple[str, Union[Sequence[Any], List[Sequence[Any]]]]


class EventStore:
    def __init__(self, db_path: Union[str, Path], snapshots_dir: Union[str, Path], batch_size: int = 200,
                 flush_interval_s: float = 0.5, queue_size: int = 20000) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir = Path(snapshots_dir)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size, self.flush_interval_s = batch_size, flush_interval_s
        init = sqlite3.connect(self.db_path, timeout=10)
        init.executescript(_SCHEMA)
        init.close()
        self._q: "queue.Queue[Write]" = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._read_lock = threading.Lock()
        self._reader = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10)
        self._reader.row_factory = sqlite3.Row
        self.dropped = 0
        self._writer = threading.Thread(target=self._write_loop, name="event-store", daemon=True)
        self._writer.start()

    # ------------------------------------------------------------ writes
    def _put(self, item: Write) -> None:
        try:
            self._q.put_nowait(item)
        except queue.Full:
            self.dropped += 1
            if self.dropped % 100 == 1:
                log.warning("Event store queue full - %d writes dropped so far", self.dropped)

    def add_zone_event(self, ev: ZoneEvent) -> None:
        x, y = ev.anchor_norm if ev.anchor_norm else (None, None)
        self._put(("INSERT INTO events (ts,type,zone_id,zone_name,track_id,category,label,dwell_s,x,y,reason) "
                   "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                   (ev.timestamp, ev.type.value, ev.zone_id, ev.zone_name, ev.track_id, ev.category.value,
                    ev.label, ev.dwell_s, x, y, ev.reason)))

    def add_alert(self, a: Alert) -> None:
        self._put(("INSERT INTO alerts (ts,rule_id,rule_type,severity,message,zone_id,zone_name,track_id,category,"
                   "snapshot,data) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                   (a.timestamp, a.rule_id, a.rule_type, a.severity.value, a.message, a.zone_id, a.zone_name,
                    a.track_id, a.category, a.snapshot_path, json.dumps(a.data, default=str))))

    def add_counts(self, ts: float, counts: Mapping[str, Mapping[str, int]]) -> None:
        rows = [(ts, z, c, int(n)) for z, per in counts.items() for c, n in per.items()]
        if rows:
            self._put(("INSERT INTO counts (ts,zone_id,category,count) VALUES (?,?,?,?)", rows))

    def add_emotions(self, ts: float, emotions: Mapping[str, Mapping[str, int]]) -> None:
        rows = [(ts, z, e, int(n)) for z, per in emotions.items() for e, n in per.items() if n]
        if rows:
            self._put(("INSERT INTO emotions (ts,zone_id,emotion,count) VALUES (?,?,?,?)", rows))

    def _write_loop(self) -> None:
        conn = sqlite3.connect(self.db_path, timeout=10)
        pending: List[Write] = []
        last = time.monotonic()
        while not (self._stop.is_set() and self._q.empty() and not pending):
            try:
                pending.append(self._q.get(timeout=self.flush_interval_s))
            except queue.Empty:
                pass
            due = len(pending) >= self.batch_size or time.monotonic() - last >= self.flush_interval_s or self._stop.is_set()
            if pending and due:
                try:
                    with conn:
                        for sql, params in pending:
                            if isinstance(params, list):
                                conn.executemany(sql, params)
                            else:
                                conn.execute(sql, params)
                except sqlite3.Error:
                    log.exception("SQLite batch write failed (%d statements lost)", len(pending))
                pending.clear()
                last = time.monotonic()
        conn.close()

    # ------------------------------------------------------------- reads
    def query(self, sql: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
        with self._read_lock:
            return [dict(r) for r in self._reader.execute(sql, params).fetchall()]

    def recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self.query("SELECT * FROM events ORDER BY ts DESC LIMIT ?", (limit,))

    def recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self.query("SELECT * FROM alerts ORDER BY ts DESC LIMIT ?", (limit,))

    def counts_timeseries(self, since_ts: float, bucket_s: int = 60) -> List[Dict[str, Any]]:
        return self.query("SELECT CAST(ts / ? AS INTEGER) * ? AS bucket, zone_id, category, AVG(count) AS avg_count, "
                          "MAX(count) AS max_count FROM counts WHERE ts >= ? GROUP BY bucket, zone_id, category "
                          "ORDER BY bucket", (bucket_s, bucket_s, since_ts))

    def zone_summary(self, since_ts: float) -> List[Dict[str, Any]]:
        return self.query("SELECT zone_id, zone_name, SUM(type='entry') AS entries, SUM(type='exit') AS exits, "
                          "AVG(CASE WHEN type='exit' THEN dwell_s END) AS avg_dwell_s, "
                          "MAX(CASE WHEN type='exit' THEN dwell_s END) AS max_dwell_s "
                          "FROM events WHERE ts >= ? GROUP BY zone_id, zone_name ORDER BY entries DESC", (since_ts,))

    def alert_summary(self, since_ts: float) -> List[Dict[str, Any]]:
        return self.query("SELECT severity, rule_id, COUNT(*) AS n FROM alerts WHERE ts >= ? "
                          "GROUP BY severity, rule_id ORDER BY n DESC", (since_ts,))

    def emotion_summary(self, since_ts: float) -> List[Dict[str, Any]]:
        return self.query("SELECT emotion, SUM(count) AS n FROM emotions WHERE ts >= ? GROUP BY emotion "
                          "ORDER BY n DESC", (since_ts,))

    # --------------------------------------------------------- snapshots
    def save_snapshot(self, jpeg: bytes, alert: Alert) -> str:
        day = time.strftime("%Y%m%d", time.gmtime(alert.timestamp))
        folder = self.snapshots_dir / day
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{int(alert.timestamp * 1000)}_{alert.rule_id}.jpg"
        path.write_bytes(jpeg)
        return str(path)

    # --------------------------------------------------------- retention
    def purge_before(self, cutoff_ts: float) -> Dict[str, int]:
        report: Dict[str, int] = {}
        with sqlite3.connect(self.db_path, timeout=30) as conn:
            for table in TABLES:
                report[table] = conn.execute(f"DELETE FROM {table} WHERE ts < ?", (cutoff_ts,)).rowcount
        removed = 0
        for f in self.snapshots_dir.rglob("*.jpg"):
            try:
                if f.stat().st_mtime < cutoff_ts:
                    f.unlink()
                    removed += 1
            except OSError as exc:
                log.warning("Could not delete snapshot %s: %s", f, exc)
        for d in sorted(self.snapshots_dir.glob("*"), reverse=True):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
        report["snapshots"] = removed
        return report

    def flush(self, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while not self._q.empty() and time.monotonic() < deadline:
            time.sleep(0.05)
        time.sleep(self.flush_interval_s + 0.1)

    def close(self) -> None:
        self._stop.set()
        self._writer.join(timeout=10)
        with self._read_lock:
            self._reader.close()
