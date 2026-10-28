"""Occupancy heatmap: short-term (decaying) and long-term (cumulative) grids.

Rendered with ``cv2.COLORMAP_HOT`` (black -> red -> amber -> white). This
deliberately avoids JET / TURBO, whose cold end is blue (house palette rule).
"""
from __future__ import annotations

from typing import Iterable, Tuple

import cv2
import numpy as np


class Heatmap:
    def __init__(self, width: int, height: int, cell_px: int = 8, decay: float = 0.997) -> None:
        self.width, self.height, self.cell, self.decay = width, height, max(1, cell_px), decay
        self.gw, self.gh = max(1, width // self.cell), max(1, height // self.cell)
        self.live = np.zeros((self.gh, self.gw), np.float32)
        self.total = np.zeros((self.gh, self.gw), np.float32)

    def add(self, points: Iterable[Tuple[float, float]]) -> None:
        self.live *= self.decay
        for x, y in points:
            gx, gy = int(x // self.cell), int(y // self.cell)
            if 0 <= gx < self.gw and 0 <= gy < self.gh:
                self.live[gy, gx] += 1.0
                self.total[gy, gx] += 1.0

    def _colorize(self, grid: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
        g = cv2.GaussianBlur(grid, (0, 0), sigmaX=1.5)
        peak = float(g.max())
        norm = (g / peak) if peak > 0 else g
        heat = cv2.resize((norm * 255).astype(np.uint8), (width, height), interpolation=cv2.INTER_LINEAR)
        return cv2.applyColorMap(heat, cv2.COLORMAP_HOT), heat

    def render(self, frame: np.ndarray, alpha: float = 0.45, cumulative: bool = False) -> np.ndarray:
        grid = self.total if cumulative else self.live
        if float(grid.max()) <= 0:
            return frame
        colored, heat = self._colorize(grid, frame.shape[1], frame.shape[0])
        blended = cv2.addWeighted(frame, 1 - alpha, colored, alpha, 0)
        return np.where((heat > 8)[..., None], blended, frame)

    def image(self, width: int = 640, height: int = 360, cumulative: bool = True) -> np.ndarray:
        colored, _ = self._colorize(self.total if cumulative else self.live, width, height)
        return colored

    def reset(self) -> None:
        self.live[:] = 0
        self.total[:] = 0
