"""Planar geometry primitives for the zone engine (pure Python, no NumPy)."""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

Point = Tuple[float, float]
Polygon = Sequence[Point]
EPS = 1e-9


def polygon_area(polygon: Polygon) -> float:
    """Signed shoelace area."""
    n = len(polygon)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s / 2.0


def validate_polygon(polygon: Polygon, *, normalized: bool = False, tol: float = 1e-6) -> List[Point]:
    """Return a cleaned vertex list or raise ``ValueError``.

    Removes consecutive duplicates and an explicit closing vertex, rejects
    non-finite coordinates, fewer than 3 vertices and zero-area polygons.
    """
    try:
        pts = [(float(p[0]), float(p[1])) for p in polygon]
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"malformed polygon: {exc}") from exc
    if any(not (math.isfinite(x) and math.isfinite(y)) for x, y in pts):
        raise ValueError("polygon contains non-finite coordinates")
    clean: List[Point] = []
    for p in pts:
        if not clean or abs(p[0] - clean[-1][0]) > EPS or abs(p[1] - clean[-1][1]) > EPS:
            clean.append(p)
    if len(clean) > 1 and abs(clean[0][0] - clean[-1][0]) <= EPS and abs(clean[0][1] - clean[-1][1]) <= EPS:
        clean.pop()
    if len(clean) < 3:
        raise ValueError("polygon needs at least 3 distinct vertices")
    if abs(polygon_area(clean)) <= EPS:
        raise ValueError("polygon is degenerate (zero area)")
    if normalized and any(not (-tol <= x <= 1 + tol and -tol <= y <= 1 + tol) for x, y in clean):
        raise ValueError("normalised polygon vertices must lie within [0, 1]")
    return clean


def point_on_segment(p: Point, a: Point, b: Point, eps: float = 1e-9) -> bool:
    (px, py), (ax, ay), (bx, by) = p, a, b
    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    if abs(cross) > eps * max(1.0, abs(bx - ax) + abs(by - ay)):
        return False
    return (min(ax, bx) - eps <= px <= max(ax, bx) + eps) and (min(ay, by) - eps <= py <= max(ay, by) + eps)


def point_in_polygon(point: Point, polygon: Polygon, include_boundary: bool = True) -> bool:
    """Even-odd (ray casting) test with explicit boundary handling - O(n).

    Points exactly on an edge or vertex return ``include_boundary``. The
    half-open rule ``(yi > y) != (yj > y)`` ensures a ray through a vertex is
    counted exactly once, so concave and self-touching shapes behave correctly.
    """
    n = len(polygon)
    if n < 3:
        raise ValueError("polygon needs at least 3 vertices")
    x, y = point
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if point_on_segment((x, y), (xj, yj), (xi, yi)):
            return include_boundary
        if (yi > y) != (yj > y):
            x_cross = xi + (y - yi) * (xj - xi) / (yj - yi)
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def bounding_box(polygon: Polygon) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return (min(xs), min(ys), max(xs), max(ys))


def bbox_anchor(bbox: Sequence[float], mode: str = "bottom_center") -> Point:
    """Reference point of a box used for zone membership."""
    x1, y1, x2, y2 = bbox
    if mode == "bottom_center":
        return ((x1 + x2) / 2.0, y2)
    if mode == "center":
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    raise ValueError(f"unknown anchor mode {mode!r}")


def normalize_polygon(polygon: Polygon, width: float, height: float) -> List[Point]:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    return [(x / width, y / height) for x, y in polygon]


def denormalize_polygon(polygon: Polygon, width: float, height: float) -> List[Point]:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    return [(x * width, y * height) for x, y in polygon]


def polygon_centroid(polygon: Polygon) -> Point:
    a = polygon_area(polygon)
    if abs(a) <= EPS:
        return (sum(p[0] for p in polygon) / len(polygon), sum(p[1] for p in polygon) / len(polygon))
    cx = cy = 0.0
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        f = x1 * y2 - x2 * y1
        cx += (x1 + x2) * f
        cy += (y1 + y2) * f
    return (cx / (6.0 * a), cy / (6.0 * a))
