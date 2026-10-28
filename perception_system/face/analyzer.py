"""Face detection + embedding back-ends with an automatic fallback chain.

* ``insightface`` - RetinaFace-family SCRFD detector + ArcFace (buffalo_l) +
  age head in one pass. Gender output is discarded (data minimisation).
* ``yunet`` - OpenCV's built-in YuNet (CPU friendly); ArcFace embeddings when
  ``face.arcface_model`` points to an ArcFace ONNX file.
* ``mediapipe`` - BlazeFace detector (detection + expression only).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from core.config import FaceConfig
from core.device import onnx_providers

log = logging.getLogger(__name__)
Resolver = Callable[[str], Path]


@dataclass
class RawFace:
    bbox: Tuple[float, float, float, float]  # in the coordinates of the analysed image
    score: float
    kps: Optional[np.ndarray] = None          # 5x2 landmarks (eyes, nose, mouth corners)
    embedding: Optional[np.ndarray] = None    # L2-normalised ArcFace vector
    age: Optional[float] = None


def _l2(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    return v / (np.linalg.norm(v) + 1e-9)


class _InsightFaceBackend:
    name = "insightface"

    def __init__(self, cfg: FaceConfig, device: str, resolve: Resolver) -> None:
        from insightface.app import FaceAnalysis

        modules = ["detection"]
        if cfg.recognition.enabled:
            modules.append("recognition")
        if cfg.age.enabled:
            modules.append("genderage")
        self.app = FaceAnalysis(name=cfg.insightface_model, allowed_modules=modules, providers=onnx_providers(device))
        self.app.prepare(ctx_id=0 if device.startswith("cuda") else -1, det_size=(cfg.det_size, cfg.det_size))
        self.provides_embeddings = cfg.recognition.enabled

    def detect(self, image: np.ndarray) -> List[RawFace]:
        out: List[RawFace] = []
        for f in self.app.get(image):
            emb = getattr(f, "normed_embedding", None)
            age = getattr(f, "age", None)
            out.append(RawFace(bbox=tuple(float(v) for v in f.bbox[:4]), score=float(f.det_score),
                               kps=getattr(f, "kps", None), embedding=None if emb is None else _l2(emb),
                               age=None if age is None else float(age)))
        return out


class _YuNetBackend:
    name = "yunet"

    def __init__(self, cfg: FaceConfig, device: str, resolve: Resolver) -> None:
        import cv2

        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError("OpenCV >= 4.5.4 is required for YuNet")
        model = resolve(cfg.yunet_model)
        if not model.is_file():
            raise FileNotFoundError(f"YuNet model not found at {model} (download face_detection_yunet_2023mar.onnx)")
        self._det = cv2.FaceDetectorYN.create(str(model), "", (320, 320), cfg.min_det_score, 0.3, 50)
        self._rec = None
        if cfg.recognition.enabled and cfg.arcface_model:
            from insightface.model_zoo import get_model

            self._rec = get_model(str(resolve(cfg.arcface_model)), providers=onnx_providers(device))
            self._rec.prepare(ctx_id=0 if device.startswith("cuda") else -1)
        self.provides_embeddings = self._rec is not None

    def detect(self, image: np.ndarray) -> List[RawFace]:
        h, w = image.shape[:2]
        self._det.setInputSize((w, h))
        _, faces = self._det.detect(image)
        out: List[RawFace] = []
        if faces is None:
            return out
        for row in faces:
            x, y, bw, bh = (float(v) for v in row[:4])
            kps = row[4:14].reshape(5, 2).astype(np.float32)
            emb = None
            if self._rec is not None:
                from insightface.utils import face_align

                aligned = face_align.norm_crop(image, landmark=kps, image_size=112)
                emb = _l2(self._rec.get_feat(aligned))
            out.append(RawFace((x, y, x + bw, y + bh), float(row[14]), kps, emb))
        return out


class _MediaPipeBackend:
    name = "mediapipe"
    provides_embeddings = False

    def __init__(self, cfg: FaceConfig, device: str, resolve: Resolver) -> None:
        import mediapipe as mp

        self._fd = mp.solutions.face_detection.FaceDetection(model_selection=1,
                                                             min_detection_confidence=cfg.min_det_score)

    def detect(self, image: np.ndarray) -> List[RawFace]:
        import cv2

        h, w = image.shape[:2]
        res = self._fd.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        out: List[RawFace] = []
        for d in res.detections or []:
            rb = d.location_data.relative_bounding_box
            x1, y1 = rb.xmin * w, rb.ymin * h
            out.append(RawFace((x1, y1, x1 + rb.width * w, y1 + rb.height * h), float(d.score[0])))
        return out


_BACKENDS = {"insightface": _InsightFaceBackend, "yunet": _YuNetBackend, "mediapipe": _MediaPipeBackend}


class FaceAnalyzer:
    """Tries the configured detector, then ``fallback_order``; raises only if all fail."""

    def __init__(self, cfg: FaceConfig, device: str, resolve: Resolver = Path) -> None:
        self.cfg = cfg
        order = [cfg.detector] + [b for b in cfg.fallback_order if b != cfg.detector]
        errors: List[str] = []
        self._backend = None
        for name in order:
            try:
                self._backend = _BACKENDS[name](cfg, device, resolve)
                log.info("Face detector online: %s (embeddings=%s)", name, self._backend.provides_embeddings)
                break
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                log.warning("Face backend %s unavailable: %s", name, exc)
        if self._backend is None:
            raise RuntimeError("no face backend available - " + "; ".join(errors))
        self.name = self._backend.name
        self.provides_embeddings = bool(self._backend.provides_embeddings)

    def detect(self, image: np.ndarray) -> List[RawFace]:
        return [f for f in self._backend.detect(image)
                if f.score >= self.cfg.min_det_score and (f.bbox[2] - f.bbox[0]) >= self.cfg.min_face_px]
