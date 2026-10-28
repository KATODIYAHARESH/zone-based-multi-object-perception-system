"""Consent-gated face gallery: ArcFace embeddings ONLY (never images).

* Every identity carries a documented consent reference, purpose, operator and
  optional expiry; enrollment without them raises ``PermissionError``.
* Revocation deletes the embeddings immediately; expiry is enforced on load
  and by the retention purge.
* Every change is recorded in an append-only ``audit`` table.
* Matching is conservative: cosine similarity >= threshold AND a margin over
  the best *different* identity - a false identification is worse than "unknown".
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

log = logging.getLogger(__name__)

_SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS identities (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,
  consent_reference TEXT NOT NULL,
  consent_at REAL NOT NULL,
  purpose TEXT NOT NULL,
  operator TEXT NOT NULL,
  expires_at REAL,
  revoked INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS embeddings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  identity_id INTEGER NOT NULL REFERENCES identities(id) ON DELETE CASCADE,
  vector BLOB NOT NULL,
  dim INTEGER NOT NULL,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL NOT NULL,
  action TEXT NOT NULL,
  name TEXT,
  detail TEXT
);
"""


class FaceGallery:
    def __init__(self, path: Union[str, Path], margin: float = 0.05) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.margin = margin
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=10)
        self._conn.executescript(_SCHEMA)
        self._matrix: Optional[np.ndarray] = None
        self._owners: List[str] = []
        self.reload()

    def _audit(self, action: str, name: str, detail: str = "") -> None:
        self._conn.execute("INSERT INTO audit (ts, action, name, detail) VALUES (?, ?, ?, ?)",
                           (time.time(), action, name, detail))

    def enroll(self, name: str, embeddings: Sequence[np.ndarray], *, consent_reference: str, purpose: str,
               operator: str, expires_days: Optional[int] = None) -> int:
        name = name.strip()
        if not name:
            raise ValueError("name is required")
        if not consent_reference or not consent_reference.strip():
            raise PermissionError("enrollment requires a documented consent reference")
        if not purpose or not purpose.strip():
            raise PermissionError("enrollment requires a stated purpose")
        vecs = [np.asarray(e, np.float32).ravel() for e in embeddings]
        if not vecs:
            raise ValueError("no embeddings supplied")
        dim = vecs[0].size
        if any(v.size != dim for v in vecs):
            raise ValueError("inconsistent embedding dimensions")
        vecs = [v / (np.linalg.norm(v) + 1e-9) for v in vecs]
        now = time.time()
        expires = now + expires_days * 86400.0 if expires_days else None
        with self._lock, self._conn:
            row = self._conn.execute("SELECT id FROM identities WHERE name = ?", (name,)).fetchone()
            if row:
                ident = int(row[0])
                self._conn.execute("UPDATE identities SET consent_reference=?, consent_at=?, purpose=?, operator=?, "
                                   "expires_at=?, revoked=0 WHERE id=?",
                                   (consent_reference, now, purpose, operator, expires, ident))
                self._conn.execute("DELETE FROM embeddings WHERE identity_id=?", (ident,))
            else:
                cur = self._conn.execute("INSERT INTO identities (name, consent_reference, consent_at, purpose, operator, "
                                         "expires_at) VALUES (?, ?, ?, ?, ?, ?)",
                                         (name, consent_reference, now, purpose, operator, expires))
                ident = int(cur.lastrowid)
            self._conn.executemany("INSERT INTO embeddings (identity_id, vector, dim, created_at) VALUES (?, ?, ?, ?)",
                                   [(ident, v.tobytes(), dim, now) for v in vecs])
            self._audit("enroll", name, f"samples={len(vecs)} consent={consent_reference} purpose={purpose} operator={operator}")
        self.reload()
        return ident

    def revoke(self, name: str, operator: str = "unknown") -> bool:
        with self._lock, self._conn:
            row = self._conn.execute("SELECT id FROM identities WHERE name = ?", (name.strip(),)).fetchone()
            if not row:
                return False
            self._conn.execute("DELETE FROM embeddings WHERE identity_id=?", (row[0],))
            self._conn.execute("UPDATE identities SET revoked=1 WHERE id=?", (row[0],))
            self._audit("revoke", name, f"operator={operator}")
        self.reload()
        return True

    def purge_expired(self) -> int:
        now = time.time()
        with self._lock, self._conn:
            rows = self._conn.execute("SELECT id, name FROM identities WHERE revoked=0 AND expires_at IS NOT NULL "
                                      "AND expires_at <= ?", (now,)).fetchall()
            for ident, name in rows:
                self._conn.execute("DELETE FROM embeddings WHERE identity_id=?", (ident,))
                self._conn.execute("UPDATE identities SET revoked=1 WHERE id=?", (ident,))
                self._audit("expire", name, "consent expired")
        if rows:
            self.reload()
        return len(rows)

    def list_identities(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT i.name, i.consent_reference, i.consent_at, i.purpose, i.operator, i.expires_at, i.revoked, "
                "COUNT(e.id) FROM identities i LEFT JOIN embeddings e ON e.identity_id = i.id "
                "GROUP BY i.id ORDER BY i.name").fetchall()
        return [{"name": r[0], "consent_reference": r[1], "consent_at": r[2], "purpose": r[3], "operator": r[4],
                 "expires_at": r[5], "revoked": bool(r[6]), "samples": int(r[7])} for r in rows]

    def reload(self) -> None:
        now = time.time()
        with self._lock:
            rows = self._conn.execute(
                "SELECT i.name, e.vector FROM embeddings e JOIN identities i ON i.id = e.identity_id "
                "WHERE i.revoked = 0 AND (i.expires_at IS NULL OR i.expires_at > ?)", (now,)).fetchall()
            if rows:
                self._owners = [r[0] for r in rows]
                self._matrix = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
            else:
                self._owners, self._matrix = [], None
        log.info("Face gallery: %d embeddings / %d active identities", len(self._owners), len(set(self._owners)))

    def match(self, embedding: Optional[np.ndarray], threshold: float) -> Tuple[Optional[str], float]:
        with self._lock:
            matrix, owners = self._matrix, self._owners
        if matrix is None or embedding is None:
            return None, 0.0
        e = np.asarray(embedding, np.float32).ravel()
        if e.size != matrix.shape[1]:
            return None, 0.0
        sims = matrix @ (e / (np.linalg.norm(e) + 1e-9))
        order = np.argsort(-sims)
        best_name, best = owners[int(order[0])], float(sims[order[0]])
        second = next((float(sims[j]) for j in order[1:] if owners[int(j)] != best_name), -1.0)
        if best >= threshold and best - second >= self.margin:
            return best_name, best
        return None, best

    @property
    def size(self) -> int:
        return len(set(self._owners))

    def close(self) -> None:
        with self._lock:
            self._conn.close()
