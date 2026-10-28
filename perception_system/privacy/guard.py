"""Privacy controls: face anonymisation and data-retention purge.

* Anonymisation uses coarse **pixelation** (irreversible at 12x down-sampling,
  unlike a weak Gaussian blur, which can be partially inverted).
* When a person has no face measurement yet, a conservative head-region
  estimate is masked, so privacy fails closed rather than open.
* Retention purge deletes events, alerts, snapshots, rotated logs and expired
  consent records older than ``privacy.retention_days``.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from core.boxes import project_relative
from core.config import PrivacyConfig
from core.types import BBox, Category, Track

log = logging.getLogger(__name__)


def purge_old_files(directory: Union[str, Path], cutoff_ts: float, patterns: Iterable[str]) -> int:
    d = Path(directory)
    if not d.is_dir():
        return 0
    removed = 0
    for pattern in patterns:
        for f in d.rglob(pattern):
            try:
                if f.is_file() and f.stat().st_mtime < cutoff_ts:
                    f.unlink()
                    removed += 1
            except OSError as exc:
                log.warning("Could not delete %s: %s", f, exc)
    return removed


class PrivacyGuard:
    def __init__(self, cfg: PrivacyConfig) -> None:
        self.cfg = cfg
        self._last_purge = 0.0

    @staticmethod
    def face_box(track: Track) -> BBox:
        if track.face is not None:
            return project_relative(track.face.bbox_rel, track.bbox)
        x1, y1, x2, y2 = track.bbox
        bw, bh = x2 - x1, y2 - y1
        return (x1 + 0.2 * bw, y1, x2 - 0.2 * bw, y1 + 0.22 * bh)  # fail-closed head estimate

    def boxes_to_blur(self, tracks: Sequence[Track], force: bool = False) -> List[BBox]:
        """Face regions to anonymise. ``force`` masks everyone (alert snapshots)."""
        if not (self.cfg.blur_faces or force):
            return []
        out: List[BBox] = []
        for t in tracks:
            if t.category is not Category.HUMAN:
                continue
            if t.identity and not force and not self.cfg.blur_enrolled:
                continue
            out.append(self.face_box(t))
        return out

    @staticmethod
    def anonymize(frame: np.ndarray, boxes: Sequence[BBox], pad: float = 0.15, block: int = 12) -> None:
        import cv2

        H, W = frame.shape[:2]
        for x1, y1, x2, y2 in boxes:
            pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
            a, b = int(max(0, x1 - pw)), int(max(0, y1 - ph))
            c, d = int(min(W, x2 + pw)), int(min(H, y2 + ph))
            if c - a < 2 or d - b < 2:
                continue
            roi = frame[b:d, a:c]
            small = cv2.resize(roi, (max(1, (c - a) // block), max(1, (d - b) // block)), interpolation=cv2.INTER_LINEAR)
            frame[b:d, a:c] = cv2.resize(small, (c - a, d - b), interpolation=cv2.INTER_NEAREST)

    def maybe_purge(self, now: float, store: Any, log_dir: Union[str, Path], gallery: Any = None) -> Optional[Dict[str, Any]]:
        if now - self._last_purge < self.cfg.purge_interval_s:
            return None
        return self.purge(now, store, log_dir, gallery)

    def purge(self, now: float, store: Any, log_dir: Union[str, Path], gallery: Any = None) -> Dict[str, Any]:
        self._last_purge = now
        cutoff = now - self.cfg.retention_days * 86400.0
        report: Dict[str, Any] = {"cutoff": cutoff, "retention_days": self.cfg.retention_days}
        try:
            if store is not None:
                report.update(store.purge_before(cutoff))
            report["log_files"] = purge_old_files(log_dir, cutoff, ("*.log.*", "*.jsonl.*"))
            if gallery is not None:
                report["expired_identities"] = gallery.purge_expired()
            log.info("Retention purge: %s", report)
        except Exception:
            log.exception("Retention purge failed")
            report["error"] = True
        return report
