"""Per-track face analysis: detect -> expression -> age group -> consent-gated identity.

The pipeline searches only the head/shoulder region of each person track, which
is cheaper than full-frame face detection and associates face -> track
implicitly. :class:`AsyncFaceWorker` runs it on a background thread so face
analysis can never stall the video loop.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from core.config import FaceConfig
from core.types import Category, FaceAttributes, Track
from face.age import AgeGrouper
from face.analyzer import FaceAnalyzer
from face.emotion import EmotionRecognizer
from face.gallery import FaceGallery

log = logging.getLogger(__name__)


class FacePipeline:
    def __init__(self, cfg: FaceConfig, device: str, gallery: Optional[FaceGallery],
                 resolve: Callable[[str], Path] = Path) -> None:
        self.cfg = cfg
        self.analyzer = FaceAnalyzer(cfg, device, resolve)
        self.emotion = EmotionRecognizer(cfg.emotion)
        self.ages = AgeGrouper(cfg.age.groups) if cfg.age.enabled else None
        self.gallery = gallery if (cfg.recognition.enabled and self.analyzer.provides_embeddings) else None
        self._state: Dict[int, FaceAttributes] = {}
        self._lock = threading.Lock()

    def analyze_track(self, frame: np.ndarray, track: Track, now: float) -> Optional[FaceAttributes]:
        x1, y1, x2, y2 = track.bbox
        bw, bh = x2 - x1, y2 - y1
        if bw <= 2 or bh <= 2:
            return None
        H, W = frame.shape[:2]
        rx1, rx2 = int(max(0, x1 - 0.1 * bw)), int(min(W, x2 + 0.1 * bw))
        ry1, ry2 = int(max(0, y1 - 0.05 * bh)), int(min(H, y1 + 0.6 * bh))
        region = frame[ry1:ry2, rx1:rx2]
        if region.size == 0 or min(region.shape[:2]) < self.cfg.min_face_px:
            return None
        faces = self.analyzer.detect(region)
        if not faces:
            return None
        f = max(faces, key=lambda r: r.score * (r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1]))
        fx1, fy1, fx2, fy2 = f.bbox[0] + rx1, f.bbox[1] + ry1, f.bbox[2] + rx1, f.bbox[3] + ry1
        attrs = FaceAttributes(bbox_rel=((fx1 - x1) / bw, (fy1 - y1) / bh, (fx2 - x1) / bw, (fy2 - y1) / bh),
                               det_score=f.score, updated_at=now)
        with self._lock:
            prev = self._state.get(track.track_id)
        crop = frame[int(max(0, fy1)):int(min(H, fy2)), int(max(0, fx1)):int(min(W, fx2))]
        scores = self.emotion.predict(crop) if self.emotion.available else None
        if scores:
            a = self.cfg.emotion.smoothing
            if prev is not None and prev.emotion_scores and a > 0:
                scores = {k: a * prev.emotion_scores.get(k, 0.0) + (1 - a) * v for k, v in scores.items()}
            top = max(scores, key=scores.get)
            attrs.emotion, attrs.emotion_conf, attrs.emotion_scores = top, float(scores[top]), scores
            same = prev is not None and prev.emotion == top and prev.emotion_since is not None
            attrs.emotion_since = prev.emotion_since if same else now
        elif prev is not None:
            attrs.emotion, attrs.emotion_conf = prev.emotion, prev.emotion_conf
            attrs.emotion_scores, attrs.emotion_since = prev.emotion_scores, prev.emotion_since
        if self.ages is not None and f.age is not None:
            age = f.age if prev is None or prev.age is None else 0.7 * prev.age + 0.3 * f.age
            attrs.age, attrs.age_group = age, self.ages.group(age)
        elif prev is not None:
            attrs.age, attrs.age_group = prev.age, prev.age_group
        if self.gallery is not None and f.embedding is not None:
            attrs.identity, attrs.identity_score = self.gallery.match(f.embedding, self.cfg.recognition.threshold)
        with self._lock:
            self._state[track.track_id] = attrs
        return attrs

    def process(self, frame: np.ndarray, tracks: Sequence[Track], now: float) -> Dict[int, FaceAttributes]:
        out: Dict[int, FaceAttributes] = {}
        for t in tracks:
            try:
                attrs = self.analyze_track(frame, t, now)
                if attrs is not None:
                    out[t.track_id] = attrs
            except Exception:
                log.exception("Face analysis failed for track %d", t.track_id)
        return out

    def forget(self, alive_ids: Iterable[int]) -> None:
        alive = set(alive_ids)
        with self._lock:
            for tid in [t for t in self._state if t not in alive]:
                del self._state[tid]


class AsyncFaceWorker:
    """Background face analysis with stale-job dropping and per-track refresh cadence."""

    def __init__(self, pipeline: FacePipeline, every_n_frames: int, max_per_cycle: int) -> None:
        self.pipeline = pipeline
        self.every = every_n_frames
        self.max_per_cycle = max_per_cycle
        self._jobs: "queue.Queue[Tuple[np.ndarray, List[Track], float]]" = queue.Queue(maxsize=1)
        self._results: Dict[int, FaceAttributes] = {}
        self._last_submit: Dict[int, int] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.latency_ms = 0.0
        self._thread = threading.Thread(target=self._run, name="face-worker", daemon=True)
        self._thread.start()

    def submit(self, frame: np.ndarray, tracks: Sequence[Track], frame_idx: int, now: float) -> None:
        due = [t for t in tracks if t.category is Category.HUMAN and not t.is_coasting
               and frame_idx - self._last_submit.get(t.track_id, -10 ** 9) >= self.every]
        if not due:
            return
        due.sort(key=lambda t: self._last_submit.get(t.track_id, -10 ** 9))  # most stale first
        batch = due[: self.max_per_cycle]
        try:
            self._jobs.put_nowait((frame, batch, now))
        except queue.Full:
            return  # worker busy: retry these tracks on a later frame
        for t in batch:
            self._last_submit[t.track_id] = frame_idx

    def attach(self, tracks: Sequence[Track]) -> None:
        with self._lock:
            for t in tracks:
                if t.category is Category.HUMAN:
                    t.face = self._results.get(t.track_id)

    def prune(self, alive_ids: Iterable[int]) -> None:
        alive = set(alive_ids)
        with self._lock:
            for tid in [t for t in self._results if t not in alive]:
                del self._results[tid]
        for tid in [t for t in self._last_submit if t not in alive]:
            del self._last_submit[tid]
        self.pipeline.forget(alive)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                frame, batch, now = self._jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            t0 = time.perf_counter()
            results = self.pipeline.process(frame, batch, now)
            self.latency_ms = 0.8 * self.latency_ms + 0.2 * (time.perf_counter() - t0) * 1000.0
            with self._lock:
                self._results.update(results)

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._thread.join(timeout)
