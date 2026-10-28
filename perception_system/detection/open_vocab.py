"""Open-vocabulary fallback detector for classes outside COCO-80.

Back-ends: YOLO-World (default, fastest), YOLOE, or Grounding DINO
(transformers). It runs every ``interval_frames`` detection frames; its labels
refine or extend the primary detections (see ``detection.manager``).
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np

from core.config import OpenVocabConfig
from core.types import Category, Detection
from detection.taxonomy import Taxonomy

log = logging.getLogger(__name__)


class OpenVocabDetector:
    def __init__(self, cfg: OpenVocabConfig, device: str, imgsz: int, taxonomy: Taxonomy) -> None:
        self.cfg = cfg
        self.device = device
        self.imgsz = imgsz
        self.taxonomy = taxonomy
        self.prompts = taxonomy.open_vocab_prompts
        if not self.prompts:
            raise ValueError("open_vocab.prompts is empty")
        self.backend = cfg.backend
        if self.backend == "yolo_world":
            from ultralytics import YOLOWorld
            self.model = YOLOWorld(cfg.model)
            self.model.set_classes(self.prompts)
        elif self.backend == "yoloe":
            from ultralytics import YOLOE
            self.model = YOLOE(cfg.model)
            try:
                self.model.set_classes(self.prompts, self.model.get_text_pe(self.prompts))
            except (TypeError, AttributeError):
                self.model.set_classes(self.prompts)
        else:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
            model_id = cfg.model if "/" in cfg.model else "IDEA-Research/grounding-dino-tiny"
            self._torch = torch
            self.processor = AutoProcessor.from_pretrained(model_id)
            target = "cpu" if device == "mps" else device
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(target)
            self._text = ". ".join(self.prompts) + "."
        log.info("Open-vocabulary fallback ready: %s (%d prompts)", self.backend, len(self.prompts))

    def _make(self, box, score: float, phrase: str) -> Detection:
        cat = self.taxonomy.category_of(phrase)
        label = self.taxonomy.display(phrase) if cat is not Category.UNCLASSIFIED else "unclassified"
        return Detection(bbox=tuple(float(v) for v in box), confidence=float(score), label=label,
                         category=cat, source="open_vocab", raw_label=phrase)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if self.backend in ("yolo_world", "yoloe"):
            res = self.model.predict(source=frame, conf=self.cfg.confidence, imgsz=self.imgsz,
                                     device=self.device, verbose=False)[0]
            if res.boxes is None or len(res.boxes) == 0:
                return []
            names = res.names if isinstance(res.names, dict) else dict(enumerate(res.names))
            return [self._make(b, s, names.get(int(c), "unclassified"))
                    for b, s, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(),
                                       res.boxes.cls.cpu().numpy())]
        return self._detect_grounding_dino(frame)

    def _detect_grounding_dino(self, frame: np.ndarray) -> List[Detection]:
        import cv2
        from PIL import Image

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=image, text=self._text, return_tensors="pt").to(self.model.device)
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        sizes = [image.size[::-1]]
        try:  # transformers >= 4.51 renamed box_threshold -> threshold
            res = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=self.cfg.confidence,
                text_threshold=self.cfg.confidence, target_sizes=sizes)[0]
        except TypeError:
            res = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, box_threshold=self.cfg.confidence,
                text_threshold=self.cfg.confidence, target_sizes=sizes)[0]
        labels = res.get("text_labels", res.get("labels", []))
        return [self._make(b, s, str(lbl)) for b, s, lbl in zip(res["boxes"].tolist(), res["scores"].tolist(), labels)]
