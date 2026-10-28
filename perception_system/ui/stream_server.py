"""Tiny MJPEG + JSON telemetry server (standard library only).

Endpoints: ``/stream.mjpg``  ``/snapshot.jpg``  ``/api/state``  ``/healthz``.
Frames are JPEG-encoded lazily (only when a client asks), so the server costs
nothing when nobody is watching. CORS is open because the data is read-only
telemetry that the web console embeds; bind to 127.0.0.1 to restrict access.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)


class FrameHub:
    """Thread-safe holder of the latest annotated frame and engine state."""

    def __init__(self, jpeg_quality: int = 80) -> None:
        self._cond = threading.Condition()
        self._frame: Optional[np.ndarray] = None
        self._state: Dict[str, Any] = {}
        self.seq = 0
        self._jpeg: Optional[bytes] = None
        self._jpeg_seq = -1
        self.quality = jpeg_quality

    def publish(self, frame: np.ndarray, state: Dict[str, Any]) -> None:
        with self._cond:
            self._frame, self._state = frame, state
            self.seq += 1
            self._cond.notify_all()

    def state(self) -> Dict[str, Any]:
        with self._cond:
            return dict(self._state)

    def frame(self) -> Optional[np.ndarray]:
        with self._cond:
            return self._frame

    def jpeg(self) -> Tuple[Optional[bytes], int]:
        with self._cond:
            frame, seq = self._frame, self.seq
            if seq == self._jpeg_seq:
                return self._jpeg, seq
        if frame is None:
            return None, seq
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            return None, seq
        data = buf.tobytes()
        with self._cond:
            if seq >= self._jpeg_seq:
                self._jpeg, self._jpeg_seq = data, seq
        return data, seq

    def wait_new(self, last_seq: int, timeout: float = 1.0) -> int:
        with self._cond:
            if self.seq == last_seq:
                self._cond.wait(timeout)
            return self.seq


class StreamServer:
    def __init__(self, hub: FrameHub, host: str = "0.0.0.0", port: int = 8765, max_fps: float = 25.0) -> None:
        self.hub, self.host, self.port, self.max_fps = hub, host, port, max_fps
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._stop = threading.Event()

    def start(self) -> bool:
        hub, stop, min_dt = self.hub, self._stop, 1.0 / self.max_fps

        class Handler(BaseHTTPRequestHandler):
            server_version = "ZBMOPS/1.0"

            def log_message(self, fmt: str, *args: Any) -> None:  # route to logging, not stderr
                log.debug("%s - " + fmt, self.address_string(), *args)

            def _send(self, code: int, ctype: str, body: bytes) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802 - stdlib naming
                path = self.path.split("?", 1)[0]
                try:
                    if path == "/stream.mjpg":
                        self._stream()
                    elif path == "/snapshot.jpg":
                        data, _ = hub.jpeg()
                        self._send(200, "image/jpeg", data) if data else self._send(503, "text/plain", b"no frame yet")
                    elif path == "/api/state":
                        self._send(200, "application/json", json.dumps(hub.state(), default=str).encode())
                    elif path == "/healthz":
                        self._send(200, "application/json", json.dumps({"status": "ok", "frames": hub.seq}).encode())
                    else:
                        self._send(404, "text/plain", b"not found")
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def _stream(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                last = -1
                while not stop.is_set():
                    if hub.wait_new(last, 1.0) == last:
                        continue
                    data, last = hub.jpeg()
                    if not data:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
                    time.sleep(min_dt)

        try:
            self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        except OSError as exc:
            log.error("Stream server disabled - cannot bind %s:%d (%s)", self.host, self.port, exc)
            return False
        self._httpd.daemon_threads = True
        threading.Thread(target=self._httpd.serve_forever, name="stream-server", daemon=True).start()
        log.info("Stream server: http://%s:%d/stream.mjpg | /snapshot.jpg | /api/state | /healthz", self.host, self.port)
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
