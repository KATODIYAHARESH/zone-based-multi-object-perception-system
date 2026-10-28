"""Shared, dependency-free data types.

These dataclasses are the contract between subsystems
(detection -> tracking -> zones -> alerts -> analytics -> UI). They deliberately
avoid NumPy so the zone and alert logic can be unit-tested with the standard
library alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional, Tuple

BBox = Tuple[float, float, float, float]
"""Axis-aligned box ``(x1, y1, x2, y2)`` in pixel coordinates."""

Point = Tuple[float, float]

EMOTIONS: Tuple[str, ...] = ("happy", "sad", "angry", "neutral", "surprise", "fear", "disgust")


def iso_utc(ts: float) -> str:
    """Format a UNIX timestamp as ISO-8601 UTC with millisecond precision."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class Category(str, Enum):
    """Top-level semantic category of a perceived object."""

    HUMAN = "human"
    ANIMAL = "animal"
    VEHICLE = "vehicle"
    FOOD = "food"
    OBJECT = "object"
    UNCLASSIFIED = "unclassified"

    @classmethod
    def parse(cls, value: "str | Category") -> "Category":
        if isinstance(value, Category):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            valid = ", ".join(c.value for c in cls)
            raise ValueError(f"unknown category {value!r}; expected one of: {valid}") from exc


class Severity(str, Enum):
    """Alert severity, ordered by :pyattr:`rank`."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return {"info": 0, "warning": 1, "critical": 2}[self.value]

    @classmethod
    def parse(cls, value: "str | Severity") -> "Severity":
        if isinstance(value, Severity):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            raise ValueError(f"unknown severity {value!r}; expected info, warning or critical") from exc


class EventType(str, Enum):
    ENTRY = "entry"
    EXIT = "exit"


def bbox_bottom_center(bbox: BBox) -> Point:
    """Ground-contact anchor of a box: the midpoint of its bottom edge."""
    x1, _y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)


@dataclass(slots=True)
class Detection:
    """A single-frame detector output."""

    bbox: BBox
    confidence: float
    label: str
    category: Category
    class_id: int = -1
    source: str = "yolo"              # yolo | open_vocab | motion
    raw_label: Optional[str] = None   # model's original class name (audit only)
    plate_bbox: Optional[BBox] = None
    vote_weight: float = 1.0          # label-vote multiplier (sparse open-vocab passes)

    @property
    def anchor(self) -> Point:
        return bbox_bottom_center(self.bbox)


@dataclass(slots=True)
class FaceAttributes:
    """Face-derived attributes attached to a human track (never stores pixels)."""

    bbox_rel: Tuple[float, float, float, float]  # face box relative to the person box (0..1)
    det_score: float = 0.0
    emotion: Optional[str] = None
    emotion_conf: float = 0.0
    emotion_scores: Dict[str, float] = field(default_factory=dict)
    emotion_since: Optional[float] = None
    age: Optional[float] = None
    age_group: Optional[str] = None
    identity: Optional[str] = None   # ONLY set for enrolled + consenting people
    identity_score: float = 0.0
    updated_at: float = 0.0


@dataclass(slots=True)
class Track:
    """Public per-frame snapshot of a tracked object."""

    track_id: int
    bbox: BBox
    confidence: float
    label: str
    category: Category
    source: str = "yolo"
    first_seen: float = 0.0
    last_seen: float = 0.0
    hits: int = 0
    is_coasting: bool = False
    plate_bbox: Optional[BBox] = None
    face: Optional[FaceAttributes] = None

    @property
    def anchor(self) -> Point:
        return bbox_bottom_center(self.bbox)

    @property
    def identity(self) -> Optional[str]:
        return self.face.identity if self.face is not None else None

    @property
    def display_name(self) -> str:
        """Privacy-preserving label: enrolled name, else ``Person #ID``."""
        if self.category is Category.HUMAN:
            return self.identity or f"Person #{self.track_id}"
        if self.category is Category.UNCLASSIFIED:
            return f"UNCLASSIFIED #{self.track_id}"
        return f"{self.label} #{self.track_id}"


@dataclass(slots=True)
class ZoneEvent:
    """A debounced ENTRY / EXIT transition of a track across a zone boundary."""

    type: EventType
    zone_id: str
    zone_name: str
    track_id: int
    category: Category
    label: str
    timestamp: float
    dwell_s: float = 0.0
    anchor_norm: Optional[Point] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value, "zone_id": self.zone_id, "zone_name": self.zone_name,
            "track_id": self.track_id, "category": self.category.value, "label": self.label,
            "timestamp": self.timestamp, "time_iso": iso_utc(self.timestamp),
            "dwell_s": round(self.dwell_s, 2), "anchor_norm": self.anchor_norm, "reason": self.reason,
        }


@dataclass(slots=True)
class Alert:
    """A fired alert (after cooldown / de-duplication)."""

    rule_id: str
    rule_type: str
    severity: Severity
    message: str
    timestamp: float
    zone_id: Optional[str] = None
    zone_name: Optional[str] = None
    track_id: Optional[int] = None
    category: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    snapshot_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id, "rule_type": self.rule_type, "severity": self.severity.value,
            "message": self.message, "timestamp": self.timestamp, "time_iso": iso_utc(self.timestamp),
            "zone_id": self.zone_id, "zone_name": self.zone_name, "track_id": self.track_id,
            "category": self.category, "data": dict(self.data), "snapshot_path": self.snapshot_path,
        }
