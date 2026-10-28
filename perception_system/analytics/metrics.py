"""Performance metrics: smoothed FPS and per-stage latency statistics."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional


class FPSMeter:
    """Exponentially smoothed frames-per-second."""

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self.fps = 0.0
        self._last: Optional[float] = None

    def tick(self) -> float:
        now = time.perf_counter()
        if self._last is not None:
            dt = now - self._last
            if dt > 0:
                inst = 1.0 / dt
                self.fps = inst if self.fps == 0.0 else (1 - self.alpha) * self.fps + self.alpha * inst
        self._last = now
        return self.fps


class LatencyStats:
    """Rolling window of per-stage latencies (ms) with mean / p50 / p95."""

    def __init__(self, window: int = 300) -> None:
        self._data: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=window))

    def add(self, stage: str, ms: float) -> None:
        self._data[stage].append(float(ms))

    def last(self, stage: str) -> float:
        d = self._data.get(stage)
        return d[-1] if d else 0.0

    def mean(self, stage: str) -> float:
        d = self._data.get(stage)
        return sum(d) / len(d) if d else 0.0

    def summary(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for stage, values in self._data.items():
            if not values:
                continue
            s = sorted(values)
            out[stage] = {"mean": round(sum(s) / len(s), 2), "p50": round(s[len(s) // 2], 2),
                          "p95": round(s[min(len(s) - 1, int(len(s) * 0.95))], 2), "n": len(s)}
        return out
