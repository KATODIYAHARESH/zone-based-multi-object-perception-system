"""OpenCV overlay renderer - mission-control HUD (amber / green / red / white only)."""
from __future__ import annotations

import time
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

from core.types import Alert, BBox, Category, Track
from ui.theme import (AMBER, AMBER_DIM, BG, CATEGORY_COLORS, GREEN, GREY, RED, RED_DIM, SEVERITY_COLORS, WHITE,
                      ZONE_COLORS, BGR)
from zones.geometry import denormalize_polygon
from zones.zone import Zone

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX
_CAT_ABBR = (("human", "H"), ("animal", "A"), ("vehicle", "V"), ("food", "F"), ("object", "O"), ("unclassified", "U"))


def put_text(img: np.ndarray, text: str, org: Tuple[int, int], color: BGR = WHITE, scale: float = 0.45,
             thickness: int = 1, bg: Optional[BGR] = BG, pad: int = 3, font: int = FONT) -> Tuple[int, int]:
    """Draw text on a solid background plate for legibility; returns (w, h) of the plate."""
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    x, y = int(org[0]), int(org[1])
    if bg is not None:
        cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + base + pad - 1), bg, -1)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    return tw + 2 * pad, th + base + 2 * pad


def draw_corner_box(img: np.ndarray, bbox: BBox, color: BGR, thickness: int = 2, frac: float = 0.22) -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    lx, ly = max(6, int((x2 - x1) * frac)), max(6, int((y2 - y1) * frac))
    for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * lx, cy), color, thickness, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * ly), color, thickness, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_4)


def draw_dashed_rect(img: np.ndarray, bbox: BBox, color: BGR, dash: int = 6) -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    for xa, ya, xb, yb in ((x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)):
        length = max(abs(xb - xa), abs(yb - ya))
        for s in range(0, length, dash * 2):
            t0, t1 = s / max(1, length), min(1.0, (s + dash) / max(1, length))
            p0 = (int(xa + (xb - xa) * t0), int(ya + (yb - ya) * t0))
            p1 = (int(xa + (xb - xa) * t1), int(ya + (yb - ya) * t1))
            cv2.line(img, p0, p1, color, 1, cv2.LINE_AA)


def draw_zone(img: np.ndarray, zone: Zone, w: int, h: int, count: Optional[int] = None, violated: bool = False,
              highlight: bool = False, fill_alpha: float = 0.12) -> None:
    """Translucent zone polygon with outline and name tag (blends only the zone's ROI)."""
    pts = np.array(denormalize_polygon(zone.polygon, w, h), dtype=np.int32)
    color = RED if violated else ZONE_COLORS.get(zone.color, AMBER)
    if fill_alpha > 0:
        x, y, bw, bh = cv2.boundingRect(pts)
        x0, y0, x1, y1 = max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)
        if x1 > x0 and y1 > y0:
            roi = img[y0:y1, x0:x1]
            layer = roi.copy()
            cv2.fillPoly(layer, [pts - np.array([x0, y0], dtype=np.int32)], color)
            cv2.addWeighted(layer, fill_alpha, roi, 1 - fill_alpha, 0, dst=roi)
    cv2.polylines(img, [pts], True, color, 3 if highlight else 2, cv2.LINE_AA)
    tag = f"{zone.name.upper()} [{zone.mode.value.upper()}]" + (f"  {count}" if count is not None else "")
    top = pts[int(np.argmin(pts[:, 1]))]
    put_text(img, tag, (int(top[0]) + 4, max(16, int(top[1]) - 8)), color, 0.45)


class OverlayRenderer:
    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    def render(self, img: np.ndarray, *, zones: Sequence[Zone], tracks: Sequence[Track],
               members: Mapping[int, Set[str]], counts: Mapping[str, Mapping[str, int]], violated: Set[str],
               alerts: Sequence[Alert], hud: Mapping[str, Any]) -> np.ndarray:
        h, w = img.shape[:2]
        for z in zones:
            if z.enabled:
                draw_zone(img, z, w, h, sum(counts.get(z.id, {}).values()), z.id in violated)
        for t in tracks:
            self._track(img, t, bool(members.get(t.track_id)))
        if self.cfg.ui.show_hud:
            self._hud(img, hud)
            self._alerts(img, alerts)
            self._counts(img, zones, counts)
        self._indicator(img)  # always drawn - mandatory, cannot be disabled
        return img

    def _track(self, img: np.ndarray, t: Track, in_zone: bool) -> None:
        color = CATEGORY_COLORS.get(t.category, WHITE)
        draw_corner_box(img, t.bbox, color, 1 if t.is_coasting else 2)
        x1, y1, x2, y2 = t.bbox
        lines = [f"{t.display_name}  {t.confidence:.2f}"]
        if t.face is not None and t.face.emotion:
            extra = f"{t.face.emotion} {int(t.face.emotion_conf * 100)}%"
            if t.face.age_group:
                extra += f" | {t.face.age_group}"
            lines.append(extra)
        y = int(y1) - 6
        for text in reversed(lines):
            put_text(img, text, (int(x1) + 2, max(14, y)), color, 0.42)
            y -= 17
        cv2.circle(img, (int((x1 + x2) / 2), int(y2)), 4, AMBER if in_zone else GREY, -1, cv2.LINE_AA)
        if t.plate_bbox is not None:
            draw_dashed_rect(img, t.plate_bbox, AMBER)
            put_text(img, "PLATE", (int(t.plate_bbox[0]), int(t.plate_bbox[3]) + 14), AMBER, 0.35)

    def _hud(self, img: np.ndarray, hud: Mapping[str, Any]) -> None:
        fps, target = float(hud.get("fps", 0.0)), float(self.cfg.performance.target_fps)
        fps_color = GREEN if fps >= target else (AMBER if fps >= 0.6 * target else RED)
        put_text(img, f"{self.cfg.system.name} // {hud.get('source', '-').upper()}", (10, 22), AMBER, 0.5)
        put_text(img, f"FPS {fps:5.1f}", (10, 44), fps_color, 0.48)
        put_text(img, f"LAT {hud.get('latency_ms', 0.0):5.0f} ms   SKIP {hud.get('skip', 1)}   TRACKS {hud.get('tracks', 0)}",
                 (110, 44), WHITE, 0.48)
        put_text(img, f"DEV {hud.get('device', '-')}   FACE {hud.get('face', 'off')}", (10, 66), WHITE, 0.45)

    def _indicator(self, img: np.ndarray) -> None:
        h, w = img.shape[:2]
        text = self.cfg.privacy.indicator_text
        (tw, th), _ = cv2.getTextSize(text, FONT_BOLD, 0.5, 1)
        x, y = w - tw - 36, 26
        cv2.rectangle(img, (x - 12, y - th - 9), (w - 8, y + 8), BG, -1)
        cv2.rectangle(img, (x - 12, y - th - 9), (w - 8, y + 8), RED, 1)
        blink = int(time.time() * 2) % 2 == 0
        cv2.circle(img, (x + 1, y - th // 2), 6, RED if blink else RED_DIM, -1, cv2.LINE_AA)
        cv2.putText(img, text, (x + 14, y), FONT_BOLD, 0.5, WHITE, 1, cv2.LINE_AA)

    def _alerts(self, img: np.ndarray, alerts: Sequence[Alert]) -> None:
        h = img.shape[0]
        y = h - 12
        for a in alerts[:5]:
            stamp = time.strftime("%H:%M:%S", time.localtime(a.timestamp))
            put_text(img, f"{stamp} {a.severity.value.upper():8s} {a.message[:90]}", (10, y),
                     SEVERITY_COLORS.get(a.severity, WHITE), 0.42)
            y -= 19

    def _counts(self, img: np.ndarray, zones: Sequence[Zone], counts: Mapping[str, Mapping[str, int]]) -> None:
        w = img.shape[1]
        y = 58
        for z in zones:
            if not z.enabled:
                continue
            per = counts.get(z.id, {})
            text = f"{z.name[:16]:16s} " + " ".join(f"{abbr}{per.get(cat, 0)}" for cat, abbr in _CAT_ABBR)
            (tw, _), _ = cv2.getTextSize(text, FONT, 0.42, 1)
            put_text(img, text, (w - tw - 14, y), ZONE_COLORS.get(z.color, AMBER), 0.42)
            y += 19

    @staticmethod
    def no_signal(width: int, height: int, message: str = "NO SIGNAL") -> np.ndarray:
        img = np.full((height, width, 3), BG, dtype=np.uint8)
        for x in range(0, width, 40):
            cv2.line(img, (x, 0), (x, height), (14, 18, 22), 1)
        for y in range(0, height, 40):
            cv2.line(img, (0, y), (width, y), (14, 18, 22), 1)
        (tw, th), _ = cv2.getTextSize(message, FONT_BOLD, 1.0, 2)
        cv2.putText(img, message, ((width - tw) // 2, (height + th) // 2), FONT_BOLD, 1.0, AMBER, 2, cv2.LINE_AA)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        cv2.putText(img, stamp, (12, height - 14), FONT, 0.45, AMBER_DIM, 1, cv2.LINE_AA)
        return img
