"""Bounding-box helpers (pure Python)."""
from __future__ import annotations

from typing import Tuple

from core.types import BBox


def area(b: BBox) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection(a: BBox, b: BBox) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def iou(a: BBox, b: BBox) -> float:
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def intersection_over_area(a: BBox, b: BBox) -> float:
    """Fraction of box ``a`` covered by box ``b``."""
    ar = area(a)
    return intersection(a, b) / ar if ar > 0 else 0.0


def clip(b: BBox, width: int, height: int) -> BBox:
    return (min(max(0.0, b[0]), width), min(max(0.0, b[1]), height),
            min(max(0.0, b[2]), width), min(max(0.0, b[3]), height))


def to_relative(b: BBox, ref: BBox) -> Tuple[float, float, float, float]:
    """Express ``b`` in the normalised frame of ``ref`` (0..1 inside ``ref``)."""
    rw, rh = max(1e-6, ref[2] - ref[0]), max(1e-6, ref[3] - ref[1])
    return ((b[0] - ref[0]) / rw, (b[1] - ref[1]) / rh, (b[2] - ref[0]) / rw, (b[3] - ref[1]) / rh)


def project_relative(rel: Tuple[float, float, float, float], ref: BBox) -> BBox:
    """Inverse of :func:`to_relative`: map a relative box onto a (moved) reference box."""
    rw, rh = ref[2] - ref[0], ref[3] - ref[1]
    return (ref[0] + rel[0] * rw, ref[1] + rel[1] * rh, ref[0] + rel[2] * rw, ref[1] + rel[3] * rh)
