"""ByteTrack multi-object tracker (Zhang et al., ECCV 2022) - self-contained implementation.

Why in-house? Track IDs drive dwell time and alerts, so the tracker is
safety-relevant: we keep it dependency-light, deterministic and unit-testable,
and extend the reference algorithm with
  * class-aware gating (never associate a dog with a person),
  * confidence-weighted label voting with a consensus threshold
    (-> "unclassified" instead of guessing when labels disagree),
  * contiguous public IDs issued only on confirmation (no ID gaps from noise),
  * miss-free coasting on frames where detection was skipped.
"""
from __future__ import annotations

import itertools
from enum import IntEnum
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from core.boxes import project_relative, to_relative
from core.types import BBox, Category, Detection, Track
from tracking.kalman import KalmanBoxFilter, xyah_to_xyxy, xyxy_to_xyah

if TYPE_CHECKING:  # pragma: no cover
    from core.config import TrackingConfig

try:
    from scipy.optimize import linear_sum_assignment as _lsa
except Exception:  # pragma: no cover - scipy optional
    _lsa = None

_PLATE_TTL = 15  # detection frames a plate region survives without re-observation


class TrackState(IntEnum):
    TENTATIVE = 0
    CONFIRMED = 1
    LOST = 2
    REMOVED = 3


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N,4) and (M,4) xyxy arrays."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    ax1, ay1, ax2, ay2 = (a[:, i:i + 1] for i in range(4))
    bx1, by1, bx2, by2 = (b[:, i] for i in range(4))
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def linear_assignment(cost: np.ndarray, max_cost: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Optimal (Hungarian) assignment with gating; greedy fallback without scipy."""
    n, m = cost.shape
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))
    matches: List[Tuple[int, int]] = []
    if _lsa is not None:
        rows, cols = _lsa(cost)
        matches = [(int(r), int(c)) for r, c in zip(rows, cols) if cost[r, c] <= max_cost]
    else:
        used_r: Set[int] = set()
        used_c: Set[int] = set()
        for flat in np.argsort(cost, axis=None):
            r, c = divmod(int(flat), m)
            if cost[r, c] > max_cost:
                break
            if r not in used_r and c not in used_c:
                used_r.add(r)
                used_c.add(c)
                matches.append((r, c))
    mr = {r for r, _ in matches}
    mc = {c for _, c in matches}
    return matches, [i for i in range(n) if i not in mr], [j for j in range(m) if j not in mc]


class _STrack:
    kf = KalmanBoxFilter()

    def __init__(self, det: Detection, ts: float, cfg: "TrackingConfig") -> None:
        self.cfg = cfg
        self.mean, self.cov = self.kf.initiate(xyxy_to_xyah(det.bbox))
        self.track_id = -1
        self.state = TrackState.TENTATIVE
        self.hits = 1
        self.time_since_update = 0
        self.first_seen = self.last_seen = ts
        self.score = det.confidence
        self.source = det.source
        self.category = det.category          # gating category, fixed at birth
        self.votes: Dict[Tuple[str, Category], float] = {}
        self.plate_rel: Optional[Tuple[float, float, float, float]] = None
        self.plate_age = 0
        self._observe_plate(det)
        self._vote(det)
        if det.source == "open_vocab" or cfg.min_hits <= 1:
            self.state = TrackState.CONFIRMED

    # ---------------------------------------------------------------- labels
    def _vote(self, det: Detection) -> None:
        for key in self.votes:
            self.votes[key] *= self.cfg.vote_decay
        key = (det.label, det.category)
        self.votes[key] = self.votes.get(key, 0.0) + det.confidence * det.vote_weight

    @property
    def label_category(self) -> Tuple[str, Category]:
        if not self.votes:
            return "unclassified", Category.UNCLASSIFIED
        (label, cat), weight = max(self.votes.items(), key=lambda kv: kv[1])
        total = sum(self.votes.values())
        if cat is Category.UNCLASSIFIED or total <= 0 or weight / total < self.cfg.label_consensus:
            return "unclassified", Category.UNCLASSIFIED
        return label, cat

    def _observe_plate(self, det: Detection) -> None:
        if det.plate_bbox is not None:
            self.plate_rel, self.plate_age = to_relative(det.plate_bbox, det.bbox), 0
        else:
            self.plate_age += 1
            if self.plate_age > _PLATE_TTL:
                self.plate_rel = None

    # ---------------------------------------------------------------- motion
    @property
    def bbox(self) -> BBox:
        return xyah_to_xyxy(self.mean)

    def predict(self) -> None:
        mean = self.mean.copy()
        if self.state == TrackState.LOST:
            mean[7] = 0.0  # freeze height velocity while unobserved
        self.mean, self.cov = self.kf.predict(mean, self.cov)

    def update(self, det: Detection, ts: float, vote: bool = True) -> None:
        self.mean, self.cov = self.kf.update(self.mean, self.cov, xyxy_to_xyah(det.bbox))
        self.hits += 1
        self.time_since_update = 0
        self.last_seen = ts
        self.score = det.confidence
        self._observe_plate(det)
        if vote:
            self._vote(det)
        if self.state == TrackState.LOST:
            self.state = TrackState.CONFIRMED
        elif self.state == TrackState.TENTATIVE and self.hits >= self.cfg.min_hits:
            self.state = TrackState.CONFIRMED

    def mark_missed(self) -> None:
        self.time_since_update += 1
        if self.state == TrackState.CONFIRMED:
            self.state = TrackState.LOST
        elif self.state == TrackState.TENTATIVE:
            self.state = TrackState.REMOVED

    def snapshot(self, coasting: bool) -> Track:
        label, cat = self.label_category
        box = self.bbox
        plate = project_relative(self.plate_rel, box) if (self.plate_rel and cat is Category.VEHICLE) else None
        return Track(track_id=self.track_id, bbox=box, confidence=self.score, label=label, category=cat,
                     source=self.source, first_seen=self.first_seen, last_seen=self.last_seen, hits=self.hits,
                     is_coasting=coasting or self.time_since_update > 0, plate_bbox=plate)


class ByteTracker:
    """Two-stage (high / low score) association tracker with persistent IDs."""

    def __init__(self, cfg: "TrackingConfig", high_thresh: float, low_thresh: float, frame_rate: float = 30.0) -> None:
        self.cfg = cfg
        self.high = cfg.high_thresh if cfg.high_thresh is not None else high_thresh
        self.low = cfg.low_thresh if cfg.low_thresh is not None else low_thresh
        self.max_lost = max(1, int(round(cfg.track_buffer * frame_rate / 30.0)))
        self._tracks: List[_STrack] = []
        self._ids = itertools.count(1)
        self.frame_idx = 0

    def reset(self) -> None:
        self._tracks.clear()
        self._ids = itertools.count(1)
        self.frame_idx = 0

    def _cost(self, tracks: Sequence[_STrack], dets: Sequence[Detection]) -> np.ndarray:
        if not tracks or not dets:
            return np.zeros((len(tracks), len(dets)), dtype=np.float64)
        ious = iou_matrix(np.array([t.bbox for t in tracks], dtype=np.float64),
                          np.array([d.bbox for d in dets], dtype=np.float64))
        if self.cfg.class_aware:
            tc = np.array([t.category.value for t in tracks])[:, None]
            dc = np.array([d.category.value for d in dets])[None, :]
            ious = np.where(tc == dc, ious, 0.0)
        return 1.0 - ious

    def update(self, detections: Sequence[Detection], timestamp: float) -> List[Track]:
        """Associate one frame of detections and return confirmed, visible tracks."""
        self.frame_idx += 1
        for t in self._tracks:
            t.predict()
        high = [d for d in detections if d.confidence >= self.high or d.source == "open_vocab"]
        low = [d for d in detections if self.low <= d.confidence < self.high and d.source != "open_vocab"]
        pool = [t for t in self._tracks if t.state in (TrackState.CONFIRMED, TrackState.LOST)]
        tentative = [t for t in self._tracks if t.state == TrackState.TENTATIVE]

        # Stage 1: confirmed + lost tracks <-> high-score detections.
        m1, u_trk, u_det = linear_assignment(self._cost(pool, high), self.cfg.match_thresh)
        for ti, di in m1:
            pool[ti].update(high[di], timestamp)
        # Stage 2: still-tracked leftovers <-> low-score detections (occlusion recovery, no label vote).
        tracked_left = [pool[i] for i in u_trk if pool[i].state == TrackState.CONFIRMED]
        m2, u_trk2, _ = linear_assignment(self._cost(tracked_left, low), self.cfg.second_match_thresh)
        for ti, di in m2:
            tracked_left[ti].update(low[di], timestamp, vote=False)
        for i in u_trk2:
            tracked_left[i].mark_missed()
        for i in u_trk:
            if pool[i].state == TrackState.LOST and pool[i] not in tracked_left:
                pool[i].mark_missed()
        # Stage 3: tentative tracks <-> remaining high-score detections.
        rest = [high[i] for i in u_det]
        m3, u_tent, u_new = linear_assignment(self._cost(tentative, rest), 0.7)
        for ti, di in m3:
            tentative[ti].update(rest[di], timestamp)
        for i in u_tent:
            tentative[i].mark_missed()
        # Births.
        for i in u_new:
            d = rest[i]
            if d.confidence >= self.cfg.new_track_thresh or d.source == "open_vocab":
                self._tracks.append(_STrack(d, timestamp, self.cfg))
        # Housekeeping: expire lost tracks, issue IDs on confirmation.
        for t in self._tracks:
            if t.state == TrackState.LOST and t.time_since_update > self.max_lost:
                t.state = TrackState.REMOVED
            if t.state == TrackState.CONFIRMED and t.track_id < 0:
                t.track_id = next(self._ids)
        self._tracks = [t for t in self._tracks if t.state != TrackState.REMOVED]
        return self._visible(coasting=False)

    def coast(self, timestamp: float) -> List[Track]:
        """Advance motion on a skipped frame without penalising tracks."""
        for t in self._tracks:
            t.predict()
        return self._visible(coasting=True)

    def _visible(self, coasting: bool) -> List[Track]:
        return [t.snapshot(coasting) for t in self._tracks if t.state == TrackState.CONFIRMED and t.track_id > 0]

    @property
    def alive_ids(self) -> Set[int]:
        """IDs of tracks that still exist (visible or temporarily lost)."""
        return {t.track_id for t in self._tracks
                if t.track_id > 0 and t.state in (TrackState.CONFIRMED, TrackState.LOST)}
