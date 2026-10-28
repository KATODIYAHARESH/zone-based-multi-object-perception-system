"""Threaded, self-healing video capture.

Design notes
------------
* A dedicated reader thread pulls frames as fast as the source delivers them and
  keeps **only the latest** one (latest-frame-wins). The processing loop never
  works on stale frames, which bounds end-to-end latency under load.
* Live sources (USB / RTSP / HTTP) that fail or stall are reopened with
  exponential back-off; video files loop or signal end-of-stream.
* All OpenCV capture calls are confined to the reader thread
  (``cv2.VideoCapture`` is not thread-safe).
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from core.config import SourceConfig

log = logging.getLogger(__name__)
_STREAM_SCHEMES = ("rtsp://", "rtsps://", "rtmp://", "http://", "https://", "udp://", "tcp://")


class SourceStatus(str, Enum):
    CONNECTING = "connecting"
    LIVE = "live"
    RECONNECTING = "reconnecting"
    ENDED = "ended"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass(slots=True)
class FramePacket:
    frame: np.ndarray
    frame_id: int
    timestamp: float  # wall clock (time.time()) at capture
    mono: float       # time.monotonic() at capture - for latency measurement


def parse_uri(uri: str) -> Union[int, str]:
    s = str(uri).strip()
    return int(s) if s.isdigit() else s


def is_file_source(uri: Union[int, str]) -> bool:
    return isinstance(uri, str) and not uri.lower().startswith(_STREAM_SCHEMES) and Path(uri).suffix != ""


class VideoSource:
    """Background frame grabber exposing the most recent frame via :meth:`read`."""

    def __init__(self, cfg: SourceConfig) -> None:
        self.cfg = cfg
        self._uri = parse_uri(cfg.uri)
        self._is_file = is_file_source(self._uri)
        self._cond = threading.Condition()
        self._latest: Optional[FramePacket] = None
        self._last_returned = -1
        self._next_id = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.status = SourceStatus.CONNECTING
        self.reconnects = 0
        self.source_fps = float(cfg.fps_hint)

    # ------------------------------------------------------------------ API
    def start(self) -> "VideoSource":
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="capture", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread:
            self._thread.join(timeout)
        self.status = SourceStatus.STOPPED

    def read(self, timeout: float = 1.0) -> Optional[FramePacket]:
        """Block until a frame newer than the previously returned one exists."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while not self._stop.is_set():
                pkt = self._latest
                if pkt is not None and pkt.frame_id != self._last_returned:
                    self._last_returned = pkt.frame_id
                    return pkt
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
        return None

    @property
    def ended(self) -> bool:
        return self.status in (SourceStatus.ENDED, SourceStatus.FAILED)

    @property
    def is_file(self) -> bool:
        return self._is_file

    # ------------------------------------------------------------- internals
    def _open(self) -> Optional[cv2.VideoCapture]:
        uri = self._uri
        if isinstance(uri, str) and not self._is_file:
            params = []
            if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                ms = int(self.cfg.read_timeout_s * 1000)
                params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, ms, cv2.CAP_PROP_READ_TIMEOUT_MSEC, ms]
            cap = cv2.VideoCapture(uri, cv2.CAP_FFMPEG, params) if params else cv2.VideoCapture(uri, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(uri)
        if not cap.isOpened():
            cap.release()
            return None
        if isinstance(uri, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
            cap.set(cv2.CAP_PROP_FPS, self.cfg.fps_hint)
        if not self._is_file:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps and 1.0 <= fps <= 240.0:
            self.source_fps = float(fps)
        return cap

    def _publish(self, frame: np.ndarray) -> None:
        if self.cfg.resize and (frame.shape[1] != self.cfg.width or frame.shape[0] != self.cfg.height):
            frame = cv2.resize(frame, (self.cfg.width, self.cfg.height), interpolation=cv2.INTER_AREA)
        pkt = FramePacket(frame, self._next_id, time.time(), time.monotonic())
        self._next_id += 1
        with self._cond:
            self._latest = pkt
            self._cond.notify_all()

    def _run(self) -> None:
        rc = self.cfg.reconnect
        delay, attempts, consecutive_fail = rc.initial_delay_s, 0, 0
        cap: Optional[cv2.VideoCapture] = None
        next_due = time.monotonic()
        try:
            while not self._stop.is_set():
                if cap is None:
                    self.status = SourceStatus.RECONNECTING if (self.reconnects or attempts) else SourceStatus.CONNECTING
                    cap = self._open()
                    if cap is None:
                        attempts += 1
                        exhausted = rc.max_attempts and attempts >= rc.max_attempts
                        if self._is_file or not rc.enabled or exhausted:
                            self.status = SourceStatus.FAILED
                            log.error("Cannot open source %r (attempt %d) - giving up", self.cfg.uri, attempts)
                            return
                        log.warning("Source %r unavailable (attempt %d); retry in %.1fs", self.cfg.uri, attempts, delay)
                        self._stop.wait(delay)
                        delay = min(delay * 2.0, rc.max_delay_s)
                        continue
                    log.info("Source %r opened (%.1f fps)", self.cfg.uri, self.source_fps)
                    self.status = SourceStatus.LIVE
                    attempts, delay, consecutive_fail = 0, rc.initial_delay_s, 0
                    next_due = time.monotonic()

                ok, frame = cap.read()
                if not ok or frame is None:
                    consecutive_fail += 1
                    if self._is_file:
                        if self.cfg.loop_video and consecutive_fail < 3:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        self.status = SourceStatus.ENDED if consecutive_fail < 3 or not self.cfg.loop_video else SourceStatus.FAILED
                        log.info("Video source finished (%s)", self.status.value)
                        return
                    log.warning("Frame grab failed on live source - reconnecting")
                    cap.release()
                    cap = None
                    self.reconnects += 1
                    continue
                consecutive_fail = 0
                self._publish(frame)
                if self._is_file and self.cfg.realtime_playback:
                    next_due += 1.0 / max(1.0, self.source_fps)
                    sleep = next_due - time.monotonic()
                    if sleep > 0:
                        self._stop.wait(sleep)
                    elif sleep < -1.0:  # fell far behind (e.g. debugger pause) - resync
                        next_due = time.monotonic()
        except Exception:  # pragma: no cover - defensive: never die silently
            log.exception("Capture thread crashed")
            self.status = SourceStatus.FAILED
        finally:
            if cap is not None:
                cap.release()
