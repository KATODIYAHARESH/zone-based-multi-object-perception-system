"""PerceptionEngine - capture -> detect -> track -> zones -> face -> alerts -> analytics -> UI.

Threads
-------
    capture ----> engine loop ----> face-worker      (async face analysis)
                       |----------> alert-dispatch   (network I/O, retries)
                       |----------> event-store      (batched SQLite writes)
                       '----------> stream-server    (HTTP clients)

The engine loop is the only writer of tracker / zone state; other threads talk
to it through bounded queues or lock-protected snapshots. One bad frame is
logged and skipped - it can never kill the loop (fault isolation).
"""
from __future__ import annotations

import logging
import threading
import time
from collections import Counter, deque
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional, Set

import cv2
import numpy as np

from alerts.engine import AlertEngine
from alerts.notifiers import AlertDispatcher, build_notifiers, zone_event_to_web
from alerts.rules import RuleContext, TrackView, build_rules
from analytics.heatmap import Heatmap
from analytics.metrics import FPSMeter, LatencyStats
from analytics.store import EventStore
from capture.video_source import FramePacket, VideoSource
from core.config import AppConfig
from core.device import describe_device, select_device
from core.types import Alert, Category, Point, Severity, Track, ZoneEvent, iso_utc
from privacy.guard import PrivacyGuard
from tracking.byte_tracker import ByteTracker
from tracking.dwell import DwellTracker, TrackInfo
from ui.overlay import OverlayRenderer
from ui.stream_server import FrameHub, StreamServer
from zones.zone import Zone, ZoneManager

log = logging.getLogger(__name__)


class PerceptionEngine:
    def __init__(self, cfg: AppConfig, zones: Optional[ZoneManager] = None, *, enable_face: Optional[bool] = None) -> None:
        from detection.manager import DetectionManager  # heavy (torch) import deferred

        self.cfg = cfg
        self.device = select_device(cfg.performance.device)
        self.device_label = describe_device(self.device)
        self.zones = zones or ZoneManager.load(cfg.path(cfg.zones.file), cfg.zones.anchor)
        self.source = VideoSource(cfg.source)
        self.detector = DetectionManager(cfg, self.device)
        self.tracker = ByteTracker(cfg.tracking, cfg.detection.confidence, cfg.detection.unclassified_floor,
                                   cfg.source.fps_hint)
        self.dwell = DwellTracker(cfg.zones.enter_frames, cfg.zones.exit_frames)
        self.alerts = AlertEngine(build_rules(cfg.alerts.rules), cfg.alerts.default_cooldown_s)
        notifiers, self.overlay_feed, self.webhook = build_notifiers(cfg.alerts, cfg.path)
        self.dispatcher = AlertDispatcher(notifiers)
        self.store = EventStore(cfg.path(cfg.analytics.db_path), cfg.path(cfg.analytics.snapshots_dir))
        self.privacy = PrivacyGuard(cfg.privacy)
        self.renderer = OverlayRenderer(cfg)
        hm = cfg.analytics.heatmap
        self.heatmap = Heatmap(cfg.source.width, cfg.source.height, hm.cell_px, hm.decay) if hm.enabled else None
        ss = cfg.ui.stream_server
        self.hub = FrameHub(ss.jpeg_quality)
        self.server = StreamServer(self.hub, ss.host, ss.port, ss.max_fps) if ss.enabled else None
        self.gallery: Any = None
        self.face_worker: Any = None
        if cfg.face.enabled if enable_face is None else enable_face:
            self._init_face()
        self.fps = FPSMeter()
        self.lat = LatencyStats()
        self.skip = cfg.performance.detect_every_n_frames
        self.frame_idx = 0
        self.show_heatmap = cfg.ui.show_heatmap
        self.paused = False
        self._proc_window: Deque[float] = deque(maxlen=60)
        self._stop = threading.Event()
        self._finished = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._display: Optional[np.ndarray] = None
        self._clean: Optional[np.ndarray] = None
        self._state: Dict[str, Any] = {"source": "connecting", "fps": 0.0}
        self._recent_events: Deque[ZoneEvent] = deque(maxlen=100)
        self._web_events: List[Dict[str, Any]] = []
        self._last_counts = self._last_web_flush = self._last_heartbeat = 0.0

    def _init_face(self) -> None:
        try:
            from face.gallery import FaceGallery
            from face.pipeline import AsyncFaceWorker, FacePipeline

            fc = self.cfg.face
            if fc.recognition.enabled:
                self.gallery = FaceGallery(self.cfg.path(fc.recognition.gallery_db))
            pipeline = FacePipeline(fc, self.device, self.gallery, self.cfg.path)
            self.face_worker = AsyncFaceWorker(pipeline, fc.every_n_frames, fc.max_faces_per_cycle)
        except Exception as exc:
            log.error("Face pipeline disabled: %s", exc)
            self.face_worker = None

    # ================================================================ lifecycle
    def start(self) -> "PerceptionEngine":
        if self._thread and self._thread.is_alive():
            return self
        self.source.start()
        if self.server is not None:
            self.server.start()
        if self.cfg.performance.warmup:
            try:
                self.detector.warmup(self.cfg.source.width, self.cfg.source.height)
            except Exception:
                log.exception("Detector warm-up failed (continuing)")
        self.privacy.purge(time.time(), self.store, self.cfg.path(self.cfg.system.log_dir), self.gallery)
        self._stop.clear()
        self._finished.clear()
        self._thread = threading.Thread(target=self._loop, name="engine", daemon=True)
        self._thread.start()
        log.info("%s online on %s - %s", self.cfg.system.name, self.device_label, self.cfg.privacy.indicator_text)
        return self

    def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                if self.paused:
                    time.sleep(0.05)
                    continue
                pkt = self.source.read(timeout=0.5)
                if pkt is None:
                    if self.source.ended:
                        self._publish_no_signal("END OF STREAM" if self.source.status.value == "ended" else "SOURCE FAILED")
                        log.info("Source finished (%s) - engine loop exiting", self.source.status.value)
                        break
                    self._publish_no_signal(f"NO SIGNAL - {self.source.status.value.upper()}")
                    continue
                try:
                    self.step(pkt)
                except Exception:
                    log.exception("Frame %d failed - skipped", self.frame_idx)
                    self.frame_idx += 1
        finally:
            self._finished.set()

    def run_forever(self, show_window: bool) -> None:
        """Blocking run. The OpenCV window (if any) is serviced on the calling (main) thread."""
        self.start()
        win = self.cfg.ui.window_name
        try:
            while not self._stop.is_set() and not self._finished.is_set():
                if not show_window:
                    time.sleep(0.25)
                    continue
                frame = self.latest_frame()
                if frame is not None:
                    cv2.imshow(win, frame)
                key = cv2.waitKey(15) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("h"):
                    self.show_heatmap = not self.show_heatmap
                elif key == ord("b"):
                    self.update_runtime(blur_faces=not self.cfg.privacy.blur_faces)
                elif key == ord("p"):
                    self.paused = not self.paused
                elif key == ord("s"):
                    log.info("Manual snapshot saved: %s", self.snapshot_now())
        except KeyboardInterrupt:
            log.info("Interrupted by operator")
        finally:
            self.stop()
            if show_window:
                cv2.destroyAllWindows()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.source.stop()
        if self.face_worker is not None:
            self.face_worker.stop()
        self.dispatcher.stop()
        self.store.close()
        if self.server is not None:
            self.server.stop()
        log.info("%s offline", self.cfg.system.name)

    # ================================================================ per frame
    def _zone_probe(self, w: int, h: int) -> Callable[[Point], bool]:
        return lambda p: bool(self.zones.zones_at((p[0] / w, p[1] / h)))

    def step(self, pkt: FramePacket) -> np.ndarray:
        frame, now = pkt.frame, pkt.timestamp
        h, w = frame.shape[:2]
        t0 = time.perf_counter()
        detect_now = self.frame_idx % self.skip == 0
        if detect_now:
            dets = self.detector.detect(frame, self._zone_probe(w, h))
            t1 = time.perf_counter()
            self.lat.add("detect", (t1 - t0) * 1000.0)
            tracks = self.tracker.update(dets, now)
        else:
            t1 = time.perf_counter()
            tracks = self.tracker.coast(now)
        alive = self.tracker.alive_ids
        if self.face_worker is not None:
            if detect_now:
                self.face_worker.submit(frame, tracks, self.frame_idx, now)
            self.face_worker.attach(tracks)
            if self.frame_idx % 30 == 0:
                self.face_worker.prune(alive)
        t2 = time.perf_counter()
        self.lat.add("track", (t2 - t1) * 1000.0)

        raw = self.zones.assign(tracks, w, h)
        infos = {t.track_id: TrackInfo(t.category, t.label, (t.anchor[0] / w, t.anchor[1] / h)) for t in tracks}
        events = self.dwell.update(now, raw, infos, self.zones.names(), alive)
        members = {t.track_id: self.dwell.zones_of(t.track_id) for t in tracks}  # hysteresis-confirmed
        active = {z.id: z for z in self.zones.list(include_disabled=False)}
        ctx = RuleContext(timestamp=now, tracks=[TrackView.from_track(t) for t in tracks], memberships=members,
                          zones=active, events=events, dwell=lambda tid, zid: self.dwell.dwell_seconds(tid, zid, now))
        fired = self.alerts.evaluate(ctx)
        t3 = time.perf_counter()
        self.lat.add("zones_alerts", (t3 - t2) * 1000.0)

        counts = self._zone_counts(tracks, members, active)
        violated = {zid for t in tracks for zid in members.get(t.track_id, ())
                    if zid in active and active[zid].is_violation(t.category)}
        display = frame.copy()
        blur = self.privacy.boxes_to_blur(tracks)
        if blur:
            self.privacy.anonymize(display, blur)  # privacy applied before anything leaves the engine
        if self.heatmap is not None:
            if detect_now:
                self.heatmap.add([t.anchor for t in tracks])
            if self.show_heatmap:
                display = self.heatmap.render(display, self.cfg.analytics.heatmap.alpha)
        clean = display.copy() if self.frame_idx % 10 == 0 else None
        zones_list = list(active.values())
        self.renderer.render(display, zones=zones_list, tracks=tracks, members=members, counts=counts,
                             violated=violated, alerts=self.overlay_feed.recent(5) if self.overlay_feed else [],
                             hud=self._hud(len(tracks)))
        self.lat.add("render", (time.perf_counter() - t3) * 1000.0)

        for alert in fired:
            self._handle_alert(alert, frame, tracks, zones_list, members, counts, violated)
        for ev in events:
            self._handle_event(ev)
        self._periodic(now, counts, tracks, members)

        proc_ms = (time.perf_counter() - t0) * 1000.0
        self.lat.add("process", proc_ms)
        e2e = (time.monotonic() - pkt.mono) * 1000.0
        self.lat.add("e2e", e2e)
        fps = self.fps.tick()
        self._adapt(proc_ms)
        self._publish(display, clean, tracks, counts, fps, e2e)
        self.frame_idx += 1
        return display

    # ================================================================ helpers
    @staticmethod
    def _zone_counts(tracks: List[Track], members: Mapping[int, Set[str]], active: Mapping[str, Zone]) -> Dict[str, Dict[str, int]]:
        counts = {zid: {c.value: 0 for c in Category} for zid in active}
        for t in tracks:
            for zid in members.get(t.track_id, ()):
                if zid in counts:
                    counts[zid][t.category.value] += 1
        return counts

    @staticmethod
    def _emotion_totals(tracks: List[Track]) -> Counter:
        return Counter(t.face.emotion for t in tracks if t.face is not None and t.face.emotion)

    @staticmethod
    def _zone_emotions(tracks: List[Track], members: Mapping[int, Set[str]]) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Counter] = {}
        for t in tracks:
            if t.face is not None and t.face.emotion:
                for zid in members.get(t.track_id) or {"_outside"}:
                    out.setdefault(zid, Counter())[t.face.emotion] += 1
        return {k: dict(v) for k, v in out.items()}

    def _hud(self, ntracks: int) -> Dict[str, Any]:
        return {"fps": self.fps.fps, "latency_ms": self.lat.last("e2e"), "skip": self.skip, "tracks": ntracks,
                "device": self.device_label, "source": self.source.status.value,
                "face": self.face_worker.pipeline.analyzer.name if self.face_worker else "off"}

    def _handle_alert(self, alert: Alert, frame: np.ndarray, tracks: List[Track], zones: List[Zone],
                      members: Mapping[int, Set[str]], counts: Mapping[str, Mapping[str, int]], violated: Set[str]) -> None:
        jpeg: Optional[bytes] = None
        if self.cfg.alerts.snapshot_on_alert:
            try:
                snap = frame.copy()
                boxes = self.privacy.boxes_to_blur(tracks, force=self.cfg.privacy.blur_snapshots)
                if boxes:
                    self.privacy.anonymize(snap, boxes)
                self.renderer.render(snap, zones=zones, tracks=tracks, members=members, counts=counts,
                                     violated=violated, alerts=[alert], hud=self._hud(len(tracks)))
                ok, buf = cv2.imencode(".jpg", snap, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    jpeg = buf.tobytes()
                    alert.snapshot_path = self.store.save_snapshot(jpeg, alert)
            except Exception:
                log.exception("Snapshot failed for alert %s", alert.rule_id)
        self.store.add_alert(alert)
        self.dispatcher.dispatch(alert, jpeg)
        log.log(logging.WARNING if alert.severity.rank >= 1 else logging.INFO,
                "ALERT [%s] %s", alert.severity.value.upper(), alert.message)

    def _handle_event(self, ev: ZoneEvent) -> None:
        self.store.add_zone_event(ev)
        self._recent_events.appendleft(ev)
        if self.webhook is not None and self.cfg.alerts.channels.webhook.forward_zone_events:
            self._web_events.append(zone_event_to_web(ev))

    def _periodic(self, now: float, counts: Dict[str, Dict[str, int]], tracks: List[Track],
                  members: Mapping[int, Set[str]]) -> None:
        if now - self._last_counts >= self.cfg.analytics.counts_interval_s:
            self._last_counts = now
            self.store.add_counts(now, counts)
            self.store.add_emotions(now, self._zone_emotions(tracks, members))
        hook = self.webhook
        if hook is not None:
            if self._web_events and now - self._last_web_flush >= 1.0:
                batch, self._web_events = self._web_events[:500], self._web_events[500:]
                self._last_web_flush = now
                self.dispatcher.submit(lambda b=batch: hook.post({"source": "engine", "events": b}))
            if hook.heartbeat_url and now - self._last_heartbeat >= 2.0:
                self._last_heartbeat = now
                payload = {"source": "engine", "fps": round(self.fps.fps, 1), "latencyMs": round(self.lat.last("e2e"), 1),
                           "device": self.device_label, "frameSkip": self.skip, "tracks": len(tracks),
                           "counts": counts, "emotions": dict(self._emotion_totals(tracks))}
                self.dispatcher.submit(lambda p=payload: hook.post(p, hook.heartbeat_url))
        self.privacy.maybe_purge(now, self.store, self.cfg.path(self.cfg.system.log_dir), self.gallery)

    def _adapt(self, proc_ms: float) -> None:
        """Adaptive frame skipping with hysteresis (amortised per-frame cost vs budget)."""
        perf = self.cfg.performance
        if not perf.adaptive_skip:
            self.skip = perf.detect_every_n_frames
            return
        self._proc_window.append(proc_ms)
        if len(self._proc_window) < (self._proc_window.maxlen or 60) or self.frame_idx % 30:
            return
        budget = 1000.0 / perf.target_fps
        avg = sum(self._proc_window) / len(self._proc_window)
        if avg > 0.95 * budget and self.skip < perf.max_skip:
            self.skip += 1
            log.info("Adaptive skip -> %d (amortised %.1f ms > budget %.1f ms)", self.skip, avg, budget)
            self._proc_window.clear()
        elif self.skip > perf.detect_every_n_frames and avg * self.skip / max(1, self.skip - 1) < 0.8 * budget:
            self.skip -= 1
            log.info("Adaptive skip -> %d (amortised %.1f ms)", self.skip, avg)
            self._proc_window.clear()

    def _publish(self, display: np.ndarray, clean: Optional[np.ndarray], tracks: List[Track],
                 counts: Dict[str, Dict[str, int]], fps: float, e2e: float) -> None:
        now = time.time()
        state = {
            "system": self.cfg.system.name, "site": self.cfg.system.site, "ts": now, "time_iso": iso_utc(now),
            "frame_idx": self.frame_idx, "fps": round(fps, 1), "latency_ms": round(e2e, 1),
            "stages": self.lat.summary(), "device": self.device_label, "skip": self.skip,
            "source": self.source.status.value, "reconnects": self.source.reconnects, "tracks": len(tracks),
            "counts": counts, "totals": dict(Counter(t.category.value for t in tracks)),
            "emotions": dict(self._emotion_totals(tracks)),
            "track_list": [{"id": t.track_id, "name": t.display_name, "category": t.category.value, "label": t.label,
                            "confidence": round(t.confidence, 2),
                            "emotion": t.face.emotion if t.face else None,
                            "age_group": t.face.age_group if t.face else None} for t in tracks[:50]],
            "face_latency_ms": round(self.face_worker.latency_ms, 1) if self.face_worker else None,
            "alerts": [a.to_dict() for a in (self.overlay_feed.recent(20) if self.overlay_feed else [])],
            "events": [e.to_dict() for e in list(self._recent_events)[:30]],
            "monitoring": self.cfg.privacy.indicator_text,
            "privacy": {"blur_faces": self.cfg.privacy.blur_faces, "retention_days": self.cfg.privacy.retention_days,
                        "enrolled_identities": self.gallery.size if self.gallery else 0},
            "dropped": {"store": self.store.dropped, "dispatch": self.dispatcher.dropped},
        }
        with self._lock:
            self._display, self._state = display, state
            if clean is not None:
                self._clean = clean
        self.hub.publish(display, state)

    def _publish_no_signal(self, message: str) -> None:
        frame = OverlayRenderer.no_signal(self.cfg.source.width, self.cfg.source.height, message)
        self.renderer._indicator(frame)
        with self._lock:
            self._display = frame
            self._state = {**self._state, "source": self.source.status.value, "fps": 0.0}
            state = self._state
        self.hub.publish(frame, state)

    # ================================================================ public API
    def latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._display

    def latest_clean_frame(self) -> Optional[np.ndarray]:
        """Privacy-filtered frame without overlays (zone-editor background)."""
        with self._lock:
            return self._clean if self._clean is not None else self._display

    def state(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def update_runtime(self, **changes: Any) -> Dict[str, Any]:
        """Apply validated runtime changes (raises on invalid key or value)."""
        applied: Dict[str, Any] = {}
        for key, value in changes.items():
            if key == "confidence":
                self.cfg.detection.confidence = float(value)
                if self.cfg.tracking.high_thresh is None:
                    self.tracker.high = float(value)
            elif key == "iou":
                self.cfg.detection.iou = float(value)
            elif key == "detect_every_n_frames":
                self.cfg.performance.detect_every_n_frames = int(value)
                self.skip = int(value)
            elif key == "adaptive_skip":
                self.cfg.performance.adaptive_skip = bool(value)
            elif key == "face_every_n_frames":
                self.cfg.face.every_n_frames = int(value)
                if self.face_worker is not None:
                    self.face_worker.every = int(value)
            elif key == "blur_faces":
                self.cfg.privacy.blur_faces = bool(value)
            elif key == "retention_days":
                self.cfg.privacy.retention_days = int(value)
            elif key == "show_heatmap":
                self.show_heatmap = bool(value)
            else:
                raise KeyError(f"unsupported runtime setting {key!r}")
            applied[key] = value
        log.info("Runtime settings updated: %s", applied)
        return applied

    def reload_rules(self, specs: List[Dict[str, Any]]) -> None:
        rules = build_rules(specs)  # validate fully before swapping
        self.alerts.set_rules(rules)
        self.cfg.alerts.rules = specs

    def save_zones(self) -> None:
        self.zones.save(self.cfg.path(self.cfg.zones.file))

    def purge_now(self) -> Dict[str, Any]:
        return self.privacy.purge(time.time(), self.store, self.cfg.path(self.cfg.system.log_dir), self.gallery)

    def snapshot_now(self) -> Optional[str]:
        frame = self.latest_frame()
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            return None
        manual = Alert(rule_id="manual", rule_type="manual", severity=Severity.INFO,
                       message="Manual snapshot", timestamp=time.time())
        return self.store.save_snapshot(buf.tobytes(), manual)
