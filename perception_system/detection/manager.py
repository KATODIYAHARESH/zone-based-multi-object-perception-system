"""Detection orchestration: YOLO primary + open-vocab fallback + motion "unclassified" + plates.

Unclassified policy (never guess):
1. Detector scores in [unclassified_floor, confidence) only *continue* existing
   tracks (ByteTrack stage 2); they never create labelled objects.
2. Open-vocabulary labels refine low-confidence primary boxes or add objects
   COCO lacks; phrases outside the taxonomy become ``unclassified``.
3. Persistent foreground motion that no detection explains becomes ``unclassified``.
4. At track level, a label is committed only with a weighted-vote consensus;
   otherwise the track is shown as ``unclassified`` (see byte_tracker).
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Dict, List, Optional

import numpy as np

from core.boxes import iou
from core.config import AppConfig
from core.types import Category, Detection, Point

log = logging.getLogger(__name__)


def merge_open_vocab(primary: List[Detection], extra: List[Detection], merge_iou: float,
                     confident: float, vote_weight: float) -> List[Detection]:
    """Fuse open-vocabulary detections into the primary set."""
    merged = list(primary)
    for ov in extra:
        ov.vote_weight = vote_weight
        best_i, best = -1, 0.0
        for i, p in enumerate(merged):
            v = iou(ov.bbox, p.bbox)
            if v > best:
                best_i, best = i, v
        if best >= merge_iou:
            p = merged[best_i]
            if p.confidence < confident and ov.category is not Category.UNCLASSIFIED:
                merged[best_i] = Detection(bbox=p.bbox, confidence=max(p.confidence, ov.confidence),
                                           label=ov.label, category=ov.category, source="open_vocab",
                                           raw_label=ov.raw_label, vote_weight=vote_weight)
        else:
            merged.append(ov)
    return merged


def category_nms(dets: List[Detection], iou_thr: float) -> List[Detection]:
    """Suppress duplicate boxes with *different labels in the same category* (car vs truck)."""
    if iou_thr >= 1.0:
        return list(dets)
    keep: List[Detection] = []
    for d in sorted(dets, key=lambda x: x.confidence, reverse=True):
        if any(k.category is d.category and k.label != d.label and iou(k.bbox, d.bbox) >= iou_thr for k in keep):
            continue
        keep.append(d)
    return keep


class DetectionManager:
    def __init__(self, cfg: AppConfig, device: str) -> None:
        from detection.motion import MotionUnclassifiedDetector
        from detection.open_vocab import OpenVocabDetector
        from detection.plate import PlateLocator
        from detection.taxonomy import Taxonomy
        from detection.yolo_detector import YoloDetector

        dc = cfg.detection
        self.cfg = dc
        self.taxonomy = Taxonomy(dc.open_vocab.prompts or None)
        self.primary = YoloDetector(dc, cfg.performance, device, self.taxonomy)
        self.open_vocab = None
        if dc.open_vocab.enabled:
            try:
                self.open_vocab = OpenVocabDetector(dc.open_vocab, device, cfg.performance.imgsz, self.taxonomy)
            except Exception as exc:
                log.warning("Open-vocabulary fallback disabled: %s", exc)
        self.motion = None
        if dc.motion.enabled:
            try:
                self.motion = MotionUnclassifiedDetector(dc.motion)
            except Exception as exc:
                log.warning("Motion analysis disabled: %s", exc)
        self.plates = PlateLocator(dc.plates, device) if dc.plates.enabled else None
        self._calls = 0
        self.timings_ms: Dict[str, float] = {}

    def warmup(self, width: int, height: int) -> None:
        self.primary.warmup(width, height)

    def detect(self, frame: np.ndarray, in_zone: Optional[Callable[[Point], bool]] = None) -> List[Detection]:
        t0 = time.perf_counter()
        dets = self.primary.detect(frame)
        self.timings_ms["yolo"] = (time.perf_counter() - t0) * 1000.0
        if self.open_vocab is not None and self._calls % self.cfg.open_vocab.interval_frames == 0:
            t1 = time.perf_counter()
            try:
                extra = self.open_vocab.detect(frame)
            except Exception:
                log.exception("Open-vocabulary pass failed")
                extra = []
            dets = merge_open_vocab(dets, extra, self.cfg.open_vocab.merge_iou, self.cfg.confidence,
                                    float(self.cfg.open_vocab.interval_frames))
            self.timings_ms["open_vocab"] = (time.perf_counter() - t1) * 1000.0
        dets = category_nms(dets, self.cfg.category_nms_iou)
        if self.motion is not None:
            t2 = time.perf_counter()
            dets.extend(self.motion.detect(frame, dets, in_zone))
            self.timings_ms["motion"] = (time.perf_counter() - t2) * 1000.0
        if self.plates is not None:
            for d in dets:
                if d.category is Category.VEHICLE and d.confidence >= self.cfg.confidence:
                    d.plate_bbox = self.plates.locate(frame, d.bbox)
        self._calls += 1
        return dets
