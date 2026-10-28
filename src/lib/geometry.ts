import type { Point } from "./types";

/** TypeScript port of perception_system/zones/geometry.py (identical semantics). */
export const EPS = 1e-9;

export function polygonArea(poly: Point[]): number {
  if (poly.length < 3) return 0;
  let s = 0;
  for (let i = 0; i < poly.length; i++) {
    const [x1, y1] = poly[i];
    const [x2, y2] = poly[(i + 1) % poly.length];
    s += x1 * y2 - x2 * y1;
  }
  return s / 2;
}

export function pointOnSegment(p: Point, a: Point, b: Point, eps = 1e-9): boolean {
  const cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
  if (Math.abs(cross) > eps * Math.max(1, Math.abs(b[0] - a[0]) + Math.abs(b[1] - a[1]))) return false;
  return (
    p[0] >= Math.min(a[0], b[0]) - eps &&
    p[0] <= Math.max(a[0], b[0]) + eps &&
    p[1] >= Math.min(a[1], b[1]) - eps &&
    p[1] <= Math.max(a[1], b[1]) + eps
  );
}

/** Even-odd ray casting with explicit boundary handling. */
export function pointInPolygon(p: Point, poly: Point[], includeBoundary = true): boolean {
  if (poly.length < 3) return false;
  const [x, y] = p;
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if (pointOnSegment(p, poly[j], poly[i])) return includeBoundary;
    if (yi > y !== yj > y) {
      const xCross = xi + ((y - yi) * (xj - xi)) / (yj - yi);
      if (x < xCross) inside = !inside;
    }
  }
  return inside;
}

/** Validate a normalised polygon; throws Error with a human-readable message. */
export function validatePolygon(input: unknown): Point[] {
  if (!Array.isArray(input)) throw new Error("polygon must be an array of [x, y] points");
  const pts: Point[] = [];
  for (const raw of input) {
    if (!Array.isArray(raw) || raw.length < 2) throw new Error("each vertex must be [x, y]");
    const x = Number(raw[0]);
    const y = Number(raw[1]);
    if (!Number.isFinite(x) || !Number.isFinite(y)) throw new Error("vertex coordinates must be finite numbers");
    if (x < -1e-6 || x > 1 + 1e-6 || y < -1e-6 || y > 1 + 1e-6) throw new Error("vertices must be normalised to [0, 1]");
    const prev = pts[pts.length - 1];
    if (!prev || Math.abs(prev[0] - x) > EPS || Math.abs(prev[1] - y) > EPS) pts.push([x, y]);
  }
  if (pts.length > 1) {
    const [f, l] = [pts[0], pts[pts.length - 1]];
    if (Math.abs(f[0] - l[0]) <= EPS && Math.abs(f[1] - l[1]) <= EPS) pts.pop();
  }
  if (pts.length < 3) throw new Error("polygon needs at least 3 distinct vertices");
  if (pts.length > 64) throw new Error("polygon may have at most 64 vertices");
  if (Math.abs(polygonArea(pts)) <= EPS) throw new Error("polygon is degenerate (zero area)");
  return pts.map(([x, y]) => [Math.round(x * 10000) / 10000, Math.round(y * 10000) / 10000]);
}

export function slugify(name: string): string {
  const s = name
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return (s || "zone").slice(0, 64);
}
