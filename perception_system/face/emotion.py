"""Facial-expression recognition, normalised to 7 canonical classes.

Canonical set: happy, sad, angry, neutral, surprise, fear, disgust.
HSEmotion's 8-class models add *contempt*, which we fold into *disgust* (the
closest class by valence and arousal); the mapping is documented in the README.

NOTE: an expression is not an emotion. Facial-expression output must never be
treated as ground truth about a person's internal state (see README limitations).
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

from core.config import EmotionConfig
from core.types import EMOTIONS

log = logging.getLogger(__name__)

CANONICAL: Dict[str, str] = {
    "happiness": "happy", "happy": "happy", "sadness": "sad", "sad": "sad", "anger": "angry",
    "angry": "angry", "neutral": "neutral", "surprise": "surprise", "fear": "fear",
    "disgust": "disgust", "contempt": "disgust",
}
_HSE_8 = {0: "Anger", 1: "Contempt", 2: "Disgust", 3: "Fear", 4: "Happiness", 5: "Neutral", 6: "Sadness", 7: "Surprise"}


def canonicalize(scores: Dict[str, float]) -> Dict[str, float]:
    """Map back-end labels onto the canonical 7 and renormalise to sum to 1."""
    out = {e: 0.0 for e in EMOTIONS}
    for key, value in scores.items():
        c = CANONICAL.get(str(key).strip().lower())
        if c is not None:
            out[c] += max(0.0, float(value))
    total = sum(out.values())
    return {k: v / total for k, v in out.items()} if total > 0 else out


class EmotionRecognizer:
    def __init__(self, cfg: EmotionConfig) -> None:
        self.cfg = cfg
        self.backend = cfg.backend if cfg.enabled else "none"
        self._impl = None
        self._classes: Dict[int, str] = _HSE_8
        if self.backend == "none":
            return
        try:
            if self.backend == "hsemotion":
                from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

                self._impl = HSEmotionRecognizer(model_name=cfg.model)
                self._classes = dict(getattr(self._impl, "idx_to_class", None) or _HSE_8)
            elif self.backend == "deepface":
                from deepface import DeepFace

                self._impl = DeepFace
            log.info("Emotion recognizer online: %s", self.backend)
        except Exception as exc:
            log.warning("Emotion backend %s unavailable (%s) - expressions disabled", self.backend, exc)
            self.backend, self._impl = "none", None

    @property
    def available(self) -> bool:
        return self._impl is not None

    def predict(self, face_bgr: Optional[np.ndarray]) -> Optional[Dict[str, float]]:
        """Return canonical emotion probabilities for one face crop, or None."""
        if self._impl is None or face_bgr is None or face_bgr.size == 0 or min(face_bgr.shape[:2]) < 16:
            return None
        if self.backend == "hsemotion":
            import cv2

            _, scores = self._impl.predict_emotions(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB), logits=False)
            raw = {self._classes.get(i, str(i)): float(s) for i, s in enumerate(np.asarray(scores).ravel())}
        else:
            res = self._impl.analyze(img_path=face_bgr, actions=["emotion"], enforce_detection=False,
                                     detector_backend="skip", silent=True)
            res = res[0] if isinstance(res, list) else res
            raw = {k: float(v) for k, v in res["emotion"].items()}
        return canonicalize(raw)
