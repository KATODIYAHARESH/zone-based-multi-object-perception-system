"""Foreground-motion analysis that surfaces objects no detector can name.

Blobs of persistent foreground motion that are *not* explained by any detection
become ``unclassified`` detections. This implements the requirement to flag
unknown objects instead of guessing a class for them.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from core.boxes import intersection_over_area
from core.config import MotionConfig
from core.types import Category, Detection, Point, bbox_bottom_center

_CELL = 48  # px - coarse grid used by the persistence filter


class MotionUnclassifiedDetector:
    def __init__(self, cfg: MotionConfig) -> None:
        self.cfg = cfg
        self._bg = cv2.createBackgroundSubtractorMOG2(history=cfg.history, varThreshold=cfg.var_threshold,
                                                      detectShadows=True)
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._persist: Dict[Tuple[int, int], int] = {}

    def detect(self, frame: np.ndarray, known: Sequence[Detection],
               in_zone: Optional[Callable[[Point], bool]] = None) -> List[Detection]:
        s = self.cfg.downscale
        small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1.0 else frame
        mask = self._bg.apply(small)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)  # drop shadow pixels (127)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = float(frame.shape[0] * frame.shape[1])
        persist: Dict[Tuple[int, int], int] = {}
        out: List[Detection] = []
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            box = (x / s, y / s, (x + bw) / s, (y + bh) / s)
            frac = ((box[2] - box[0]) * (box[3] - box[1])) / frame_area
            if not (self.cfg.min_area_frac <= frac <= self.cfg.max_area_frac):
                continue
            if any(intersection_over_area(box, d.bbox) >= self.cfg.covered_frac for d in known):
                continue  # explained by a detector
            anchor = bbox_bottom_center(box)
            if self.cfg.only_in_zones and in_zone is not None and not in_zone(anchor):
                continue
            cell = (int(anchor[0] // _CELL), int(anchor[1] // _CELL))
            prev = max(self._persist.get((cell[0] + dx, cell[1] + dy), 0) for dx in (-1, 0, 1) for dy in (-1, 0, 1))
            persist[cell] = max(persist.get(cell, 0), prev + 1)
            if persist[cell] >= self.cfg.persist_frames:
                out.append(Detection(bbox=box, confidence=self.cfg.confidence, label="unclassified",
                                     category=Category.UNCLASSIFIED, source="motion"))
        self._persist = persist
        return out
