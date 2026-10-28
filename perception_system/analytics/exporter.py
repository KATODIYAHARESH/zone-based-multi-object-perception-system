"""CSV / JSON export of stored analytics."""
from __future__ import annotations

import csv
import io
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from analytics.store import TABLES, EventStore
from core.types import iso_utc


def fetch_rows(store: EventStore, table: str, since_ts: Optional[float] = None) -> List[Dict[str, Any]]:
    if table not in TABLES:  # whitelist: the table name is interpolated into SQL
        raise ValueError(f"unknown table {table!r}; expected one of {TABLES}")
    rows = store.query(f"SELECT * FROM {table} WHERE ts >= ? ORDER BY ts", (since_ts or 0.0,))
    for r in rows:
        r["time_iso"] = iso_utc(float(r["ts"]))
    return rows


def rows_to_csv(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def export_table(store: EventStore, table: str, fmt: str, out_dir: Union[str, Path],
                 since_ts: Optional[float] = None) -> Path:
    if fmt not in ("csv", "json"):
        raise ValueError("format must be 'csv' or 'json'")
    rows = fetch_rows(store, table, since_ts)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{table}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.{fmt}"
    if fmt == "csv":
        path.write_text(rows_to_csv(rows), encoding="utf-8")
    else:
        path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    return path
