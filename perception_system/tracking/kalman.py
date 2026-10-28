"""Constant-velocity Kalman filter in (cx, cy, aspect, height) space.

The SORT / DeepSORT / ByteTrack parameterisation: noise scales with box height
so the filter behaves consistently for near and far objects.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from core.types import BBox


def xyxy_to_xyah(b: BBox) -> np.ndarray:
    x1, y1, x2, y2 = b
    w, h = max(1e-3, x2 - x1), max(1e-3, y2 - y1)
    return np.array([x1 + w / 2.0, y1 + h / 2.0, w / h, h], dtype=np.float64)


def xyah_to_xyxy(m: np.ndarray) -> BBox:
    cx, cy, a, h = (float(v) for v in m[:4])
    h = max(1e-3, h)
    w = max(1e-3, a * h)
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


class KalmanBoxFilter:
    _w_pos = 1.0 / 20.0
    _w_vel = 1.0 / 160.0

    def __init__(self) -> None:
        ndim = 4
        self._F = np.eye(2 * ndim)
        for i in range(ndim):
            self._F[i, ndim + i] = 1.0
        self._H = np.eye(ndim, 2 * ndim)

    def initiate(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = z[3]
        std = [2 * self._w_pos * h, 2 * self._w_pos * h, 1e-2, 2 * self._w_pos * h,
               10 * self._w_vel * h, 10 * self._w_vel * h, 1e-5, 10 * self._w_vel * h]
        return np.r_[z, np.zeros(4)], np.diag(np.square(std))

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        std = [self._w_pos * h, self._w_pos * h, 1e-2, self._w_pos * h,
               self._w_vel * h, self._w_vel * h, 1e-5, self._w_vel * h]
        return self._F @ mean, self._F @ cov @ self._F.T + np.diag(np.square(std))

    def project(self, mean: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        std = [self._w_pos * h, self._w_pos * h, 1e-1, self._w_pos * h]
        return self._H @ mean, self._H @ cov @ self._H.T + np.diag(np.square(std))

    def update(self, mean: np.ndarray, cov: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        proj_mean, proj_cov = self.project(mean, cov)
        # K = P H^T S^-1, computed with a linear solve (no explicit inverse).
        gain = np.linalg.solve(proj_cov, (cov @ self._H.T).T).T
        return mean + gain @ (z - proj_mean), cov - gain @ proj_cov @ gain.T
