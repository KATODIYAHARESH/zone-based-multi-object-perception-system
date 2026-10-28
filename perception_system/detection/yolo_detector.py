"""Primary closed-set detector built on Ultralytics YOLO (YOLO26 by default)."""
from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

from core.config import DetectionConfig, PerformanceConfig
from core.types import Detection
from detection.taxonomy import Taxonomy

log = logging.getLogger(__name__)


class YoloDetector:
    """Runs YOLO and maps every box through the taxonomy.

    Inference uses ``conf=unclassified_floor`` so that low-score boxes are
    available to ByteTrack's second association stage; they never create
    labelled objects on their own (see ``tracking.byte_tracker``).
    """

    def __init__(self, cfg: DetectionConfig, perf: PerformanceConfig, device: str, taxonomy: Taxonomy) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - environment issue
            raise RuntimeError("ultralytics is not installed - pip install -r requirements.txt") from exc
        self.cfg = cfg
        self.device = device
        self.imgsz = perf.imgsz
        self.half = bool(perf.half_precision and device.startswith("cuda"))
        self.taxonomy = taxonomy
        log.info("Loading primary detector %s on %s (fp16=%s)", cfg.model, device, self.half)
        self.model = YOLO(cfg.model)
        names = self.model.names
        self.names: Dict[int, str] = dict(names) if isinstance(names, dict) else dict(enumerate(names))

    def warmup(self, width: int, height: int, iterations: int = 2) -> None:
        dummy = np.zeros((height, width, 3), dtype=np.uint8)
        for _ in range(iterations):
            self.detect(dummy)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        result = self.model.predict(
            source=frame, conf=self.cfg.unclassified_floor, iou=self.cfg.iou, imgsz=self.imgsz,
            device=self.device, half=self.half, max_det=self.cfg.max_det, classes=self.cfg.classes,
            agnostic_nms=self.cfg.agnostic_nms, verbose=False)[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        out: List[Detection] = []
        for (x1, y1, x2, y2), score, k in zip(xyxy, conf, cls):
            raw = self.names.get(int(k), str(int(k)))
            out.append(Detection(
                bbox=(float(x1), float(y1), float(x2), float(y2)), confidence=float(score),
                label=self.taxonomy.display(raw), category=self.taxonomy.category_of(raw),
                class_id=int(k), source="yolo", raw_label=raw))
        return out
