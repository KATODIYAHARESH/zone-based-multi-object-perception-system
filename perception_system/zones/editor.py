"""Interactive OpenCV polygon zone editor (the mouse-click UI).

Controls
--------
  Left click ........ add vertex             Right click / Enter .. close polygon, then type a name
  Backspace / u ..... undo last vertex       Tab .................. select next zone
  m ................. toggle allowed / restricted on the selected zone
  1-6 ............... toggle category: 1 human 2 animal 3 vehicle 4 food 5 object 6 unclassified
  c ................. cycle colour           d .................... delete selected zone
  s ................. save zones.yaml        q / Esc .............. quit
"""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from capture.video_source import VideoSource
from core.types import Category
from ui.overlay import OverlayRenderer, draw_zone, put_text
from ui.theme import AMBER, GREY, RED, WHITE
from zones.geometry import normalize_polygon
from zones.zone import ZONE_COLORS, Zone, ZoneManager, ZoneMode

log = logging.getLogger(__name__)
CATEGORY_KEYS = [c.value for c in Category]  # order matches keys 1..6


class ZoneEditor:
    def __init__(self, source: VideoSource, manager: ZoneManager, save_path: Path, size: Tuple[int, int],
                 window: str = "ZBMOPS // Zone Editor") -> None:
        self.source = source
        self.manager = manager
        self.save_path = Path(save_path)
        self.size = size
        self.window = window
        self.points: List[Tuple[int, int]] = []
        self.naming = False
        self.name_buf = ""
        zones = manager.list()
        self.selected: Optional[str] = zones[0].id if zones else None
        self.dirty = False
        self.status = "Left-click to add vertices; Enter or right-click closes the polygon."
        self._cursor = (0, 0)

    # ----------------------------------------------------------- interaction
    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        self._cursor = (x, y)
        if self.naming:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._close()

    def _close(self) -> None:
        if len(self.points) < 3:
            self.status = "Need at least 3 vertices."
            return
        self.naming, self.name_buf = True, ""
        self.status = "Type a zone name, Enter to confirm, Esc to cancel."

    def _commit(self) -> None:
        w, h = self.size
        name = self.name_buf.strip() or f"Zone {len(self.manager.list()) + 1}"
        try:
            zone = Zone(id=self.manager.unique_id(name), name=name,
                        polygon=normalize_polygon(self.points, w, h), mode=ZoneMode.RESTRICTED)
            self.manager.add(zone)
            self.selected, self.dirty = zone.id, True
            self.status = f"Created '{name}' (restricted, no categories = nothing allowed). m: mode, 1-6: categories."
        except ValueError as exc:
            self.status = f"Rejected: {exc}"
        self.points.clear()
        self.naming = False

    def _current(self) -> Optional[Zone]:
        return self.manager.get(self.selected) if self.selected else None

    def _mutate(self, **changes: object) -> None:
        zone = self._current()
        if zone is None:
            self.status = "No zone selected (press Tab)."
            return
        try:
            self.manager.add(dataclasses.replace(zone, **changes), replace=True)
            self.dirty = True
        except ValueError as exc:
            self.status = f"Rejected: {exc}"

    def _key(self, key: int) -> bool:
        """Handle one key press; returns False when the editor should quit."""
        if self.naming:
            if key in (13, 10):
                self._commit()
            elif key == 27:
                self.naming = False
                self.status = "Naming cancelled (vertices kept)."
            elif key in (8, 127):
                self.name_buf = self.name_buf[:-1]
            elif 32 <= key <= 126 and len(self.name_buf) < 48:
                self.name_buf += chr(key)
            return True
        zone = self._current()
        ids = [z.id for z in self.manager.list()]
        if key in (ord("q"), 27):
            return False
        if key in (13, 10):
            self._close()
        elif key in (8, 127, ord("u")):
            if self.points:
                self.points.pop()
        elif key == 9 and ids:
            self.selected = ids[(ids.index(self.selected) + 1) % len(ids)] if self.selected in ids else ids[0]
        elif key == ord("m") and zone:
            self._mutate(mode=ZoneMode.RESTRICTED if zone.mode is ZoneMode.ALLOWED else ZoneMode.ALLOWED)
        elif ord("1") <= key <= ord("6") and zone:
            cats = set(zone.categories) ^ {CATEGORY_KEYS[key - ord("1")]}
            self._mutate(categories=sorted(cats))
        elif key == ord("c") and zone:
            self._mutate(color=ZONE_COLORS[(ZONE_COLORS.index(zone.color) + 1) % len(ZONE_COLORS)])
        elif key == ord("d") and zone:
            self.manager.remove(zone.id)
            remaining = [z.id for z in self.manager.list()]
            self.selected, self.dirty = (remaining[0] if remaining else None), True
            self.status = f"Deleted '{zone.name}'."
        elif key == ord("s"):
            self.manager.save(self.save_path)
            self.dirty = False
            self.status = f"Saved {len(self.manager.list())} zone(s) to {self.save_path}"
        return True

    # ------------------------------------------------------------- rendering
    def _draw(self, img: np.ndarray) -> None:
        h, w = img.shape[:2]
        for z in self.manager.list():
            draw_zone(img, z, w, h, count=None, violated=False, highlight=(z.id == self.selected))
        if self.points:
            cv2.polylines(img, [np.array(self.points, np.int32)], False, AMBER, 2, cv2.LINE_AA)
            cv2.line(img, self.points[-1], self._cursor, GREY, 1, cv2.LINE_AA)
            for p in self.points:
                cv2.circle(img, p, 4, AMBER, -1, cv2.LINE_AA)
        sel = self._current()
        lines = ["ZONE EDITOR | L-click vertex  R-click/Enter close  u undo  Tab select  m mode  1-6 categories  c colour  d delete  s save  q quit"]
        if sel:
            lines.append(f"SELECTED: {sel.name} [{sel.mode.value}] categories={','.join(sel.categories) or '-'} colour={sel.color}")
        if self.naming:
            lines.append(f"NAME> {self.name_buf}_")
        lines.append(self.status + ("   * UNSAVED *" if self.dirty else ""))
        y = 22
        for i, text in enumerate(lines):
            put_text(img, text, (10, y), AMBER if i == 0 else (RED if "UNSAVED" in text else WHITE), 0.45)
            y += 20

    def run(self) -> None:
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, *self.size)
        cv2.setMouseCallback(self.window, self._on_mouse)
        frame: Optional[np.ndarray] = None
        try:
            while True:
                pkt = self.source.read(timeout=0.03)
                if pkt is not None:
                    frame = pkt.frame
                canvas = frame.copy() if frame is not None else OverlayRenderer.no_signal(*self.size, "WAITING FOR SIGNAL")
                self._draw(canvas)
                cv2.imshow(self.window, canvas)
                key = cv2.waitKey(20)
                if key != -1 and not self._key(key & 0xFF):
                    break
                if cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            cv2.destroyWindow(self.window)
            if self.dirty:
                log.warning("Zone editor closed with UNSAVED changes (press 's' to save).")
