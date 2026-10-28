"""Zone model and thread-safe zone manager (pure Python; YAML I/O imported lazily)."""
from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Set

from core.types import BBox, Category, Point
from zones.geometry import bbox_anchor, bounding_box, point_in_polygon, validate_polygon

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,63}$")
ZONE_COLORS = ("amber", "green", "red", "white")  # palette restriction: no blue
_FIELDS = {"id", "name", "polygon", "mode", "categories", "max_occupancy", "dwell_limit_s", "color", "enabled"}
_HEADER = (
    "# Zone definitions - normalised coordinates (0..1), origin top-left.\n"
    "#   mode: allowed    -> `categories` is a whitelist (anything else = violation)\n"
    "#   mode: restricted -> `categories` is a blacklist (empty list = nothing may enter)\n"
)


class ZoneMode(str, Enum):
    ALLOWED = "allowed"
    RESTRICTED = "restricted"


class HasBBox(Protocol):
    track_id: int
    bbox: BBox


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return (s or "zone")[:64]


@dataclass
class Zone:
    """A polygonal region of interest with its own policy.

    Coordinates are normalised (0..1) so zones survive resolution changes.
    """

    id: str
    name: str
    polygon: List[Point]
    mode: ZoneMode = ZoneMode.RESTRICTED
    categories: List[str] = field(default_factory=list)
    max_occupancy: Optional[int] = None
    dwell_limit_s: Optional[float] = None
    color: str = "amber"
    enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not _ID_RE.match(self.id):
            raise ValueError(f"invalid zone id {self.id!r} (lowercase letters, digits, '_' or '-')")
        if not str(self.name).strip():
            raise ValueError("zone name must not be empty")
        try:
            self.mode = ZoneMode(self.mode.value if isinstance(self.mode, ZoneMode) else str(self.mode).lower())
        except ValueError as exc:
            raise ValueError(f"zone mode must be 'allowed' or 'restricted', got {self.mode!r}") from exc
        self.polygon = validate_polygon(self.polygon, normalized=True)
        self.categories = sorted({Category.parse(c).value for c in (self.categories or [])})
        if self.max_occupancy is not None:
            self.max_occupancy = int(self.max_occupancy)
            if self.max_occupancy < 0:
                raise ValueError("max_occupancy must be >= 0")
        if self.dwell_limit_s is not None:
            self.dwell_limit_s = float(self.dwell_limit_s)
            if self.dwell_limit_s <= 0:
                raise ValueError("dwell_limit_s must be > 0")
        if self.color not in ZONE_COLORS:
            raise ValueError(f"zone color must be one of {ZONE_COLORS}")
        self._bounds = bounding_box(self.polygon)

    def contains(self, point_norm: Point) -> bool:
        x, y = point_norm
        bx1, by1, bx2, by2 = self._bounds
        if x < bx1 or x > bx2 or y < by1 or y > by2:  # cheap rejection first
            return False
        return point_in_polygon(point_norm, self.polygon)

    def is_violation(self, category: "str | Category") -> bool:
        cat = Category.parse(category).value
        if self.mode is ZoneMode.ALLOWED:
            return cat not in self.categories
        return (not self.categories) or cat in self.categories

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "mode": self.mode.value, "categories": list(self.categories),
                "polygon": [[round(x, 4), round(y, 4)] for x, y in self.polygon],
                "max_occupancy": self.max_occupancy, "dwell_limit_s": self.dwell_limit_s,
                "color": self.color, "enabled": self.enabled}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Zone":
        unknown = set(data) - _FIELDS
        if unknown:
            raise ValueError(f"unknown zone field(s): {sorted(unknown)}")
        if "name" not in data or "polygon" not in data:
            raise ValueError("zone requires 'name' and 'polygon'")
        return cls(id=str(data.get("id") or slugify(str(data["name"]))), name=str(data["name"]),
                   polygon=[(p[0], p[1]) for p in data["polygon"]], mode=data.get("mode", "restricted"),
                   categories=list(data.get("categories") or []), max_occupancy=data.get("max_occupancy"),
                   dwell_limit_s=data.get("dwell_limit_s"), color=data.get("color", "amber"),
                   enabled=bool(data.get("enabled", True)))


class ZoneManager:
    """Thread-safe zone registry plus bottom-centre point-in-polygon assignment."""

    def __init__(self, zones: Iterable[Zone] = (), anchor: str = "bottom_center") -> None:
        bbox_anchor((0, 0, 1, 1), anchor)  # validates the anchor mode
        self.anchor = anchor
        self._zones: Dict[str, Zone] = {}
        self._lock = threading.RLock()
        self.version = 0
        for z in zones:
            self.add(z)

    def add(self, zone: Zone, replace: bool = False) -> None:
        with self._lock:
            if zone.id in self._zones and not replace:
                raise KeyError(f"zone {zone.id!r} already exists")
            self._zones[zone.id] = zone
            self.version += 1

    def remove(self, zone_id: str) -> bool:
        with self._lock:
            removed = self._zones.pop(zone_id, None) is not None
            self.version += int(removed)
            return removed

    def get(self, zone_id: str) -> Optional[Zone]:
        with self._lock:
            return self._zones.get(zone_id)

    def list(self, include_disabled: bool = True) -> List[Zone]:
        with self._lock:
            return [z for z in self._zones.values() if include_disabled or z.enabled]

    def names(self) -> Dict[str, str]:
        return {z.id: z.name for z in self.list(include_disabled=False)}

    def unique_id(self, name: str) -> str:
        base = slugify(name)
        candidate, i = base, 2
        with self._lock:
            while candidate in self._zones:
                suffix = f"_{i}"
                candidate, i = base[: 64 - len(suffix)] + suffix, i + 1
        return candidate

    def zones_at(self, point_norm: Point) -> Set[str]:
        return {z.id for z in self.list(include_disabled=False) if z.contains(point_norm)}

    def assign(self, tracks: Sequence[HasBBox], frame_w: int, frame_h: int) -> Dict[int, Set[str]]:
        """Map each track ID to the set of enabled zones containing its anchor point."""
        if frame_w <= 0 or frame_h <= 0:
            raise ValueError("frame dimensions must be positive")
        active = self.list(include_disabled=False)
        out: Dict[int, Set[str]] = {}
        for t in tracks:
            ax, ay = bbox_anchor(t.bbox, self.anchor)
            p = (ax / frame_w, ay / frame_h)
            out[t.track_id] = {z.id for z in active if z.contains(p)}
        return out

    def to_dicts(self) -> List[Dict[str, Any]]:
        return [z.to_dict() for z in self.list()]

    @classmethod
    def from_dicts(cls, items: Iterable[Mapping[str, Any]], anchor: str = "bottom_center") -> "ZoneManager":
        return cls([Zone.from_dict(d) for d in items], anchor)

    @classmethod
    def load(cls, path: "str | os.PathLike[str]", anchor: str = "bottom_center") -> "ZoneManager":
        import yaml

        p = Path(path)
        if not p.exists():
            return cls([], anchor)
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        items = data.get("zones", []) if isinstance(data, dict) else data
        return cls.from_dicts(items or [], anchor)

    def save(self, path: "str | os.PathLike[str]") -> None:
        """Atomic write (temp file + rename) so a crash never leaves a half-written file."""
        import yaml

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        body = yaml.safe_dump({"zones": self.to_dicts()}, sort_keys=False, default_flow_style=None, allow_unicode=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(_HEADER + body, encoding="utf-8")
        os.replace(tmp, p)
