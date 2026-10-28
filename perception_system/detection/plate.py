"""Licence-plate *region* localisation (no OCR by default - privacy by design).

Uses a dedicated plate detector when ``detection.plates.model`` is configured,
otherwise a classical morphology heuristic (black-hat + Scharr gradient +
geometric filtering) that runs in about 1 ms per vehicle on CPU.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from core.config import PlateConfig
from core.types import BBox

log = logging.getLogger(__name__)


class PlateLocator:
    def __init__(self, cfg: PlateConfig, device: str) -> None:
        self.cfg = cfg
        self.device = device
        self._model = None
        if cfg.model:
            try:
                from ultralytics import YOLO
                self._model = YOLO(cfg.model)
                log.info("Plate detector loaded: %s", cfg.model)
            except Exception as exc:
                log.warning("Plate model unavailable (%s) - using heuristic", exc)

    def locate(self, frame: np.ndarray, vehicle_bbox: BBox) -> Optional[BBox]:
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(vehicle_bbox[0])), max(0, int(vehicle_bbox[1]))
        x2, y2 = min(w, int(vehicle_bbox[2])), min(h, int(vehicle_bbox[3]))
        if y2 - y1 < self.cfg.min_vehicle_height_px or x2 - x1 < 40:
            return None
        top = y1 + int((y2 - y1) * 0.35)  # plates live in the lower part of a vehicle
        crop = frame[top:y2, x1:x2]
        if crop.size == 0:
            return None
        box = self._from_model(crop) if self._model is not None else self._heuristic(crop)
        if box is None:
            return None
        return (float(x1 + box[0]), float(top + box[1]), float(x1 + box[2]), float(top + box[3]))

    def _from_model(self, crop: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        res = self._model.predict(source=crop, conf=self.cfg.confidence, device=self.device, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return None
        i = int(res.boxes.conf.argmax())
        x1, y1, x2, y2 = (float(v) for v in res.boxes.xyxy[i].tolist())
        return (x1, y1, x2, y2)

    @staticmethod
    def _heuristic(crop: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        ch, cw = gray.shape
        rect_k = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_k)  # dark glyphs on bright plate
        grad = np.absolute(cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=-1))
        lo, hi = float(grad.min()), float(grad.max())
        if hi - lo < 1e-6:
            return None
        grad = ((grad - lo) / (hi - lo) * 255).astype("uint8")
        grad = cv2.morphologyEx(cv2.GaussianBlur(grad, (5, 5), 0), cv2.MORPH_CLOSE, rect_k)
        _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        thresh = cv2.dilate(cv2.erode(thresh, None, iterations=2), None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best, best_score, crop_area = None, 0.0, float(ch * cw)
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            if bh == 0:
                continue
            aspect, frac = bw / float(bh), (bw * bh) / crop_area
            if not (2.0 <= aspect <= 6.5 and 0.004 <= frac <= 0.12):
                continue
            fill = cv2.contourArea(c) / float(bw * bh)
            centrality = 1.0 - abs((x + bw / 2.0) / cw - 0.5)
            score = fill * centrality * min(1.0, frac * 40.0)
            if score > best_score:
                best, best_score = (x, y, x + bw, y + bh), score
        return best if best_score >= 0.15 else None
