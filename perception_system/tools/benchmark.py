"""FPS / latency benchmark at the configured processing resolution (default 1280x720).

    python main.py benchmark --source sample.mp4 --frames 300 [--no-face]

Frames are read as fast as possible (no real-time pacing). The full chain runs
synchronously per frame and per-stage mean / p50 / p95 latency plus sustained
throughput are reported. In production the face stage runs asynchronously, so
``--no-face`` throughput approximates the live loop. The JSON report is
written to ``analytics.export_dir``.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict

log = logging.getLogger(__name__)


def run_benchmark(cfg: Any, frames: int = 300, warmup: int = 20, enable_face: bool = True) -> Dict[str, Any]:
    import cv2

    from analytics.metrics import LatencyStats
    from capture.video_source import parse_uri
    from core.device import describe_device, select_device
    from core.types import Category
    from detection.manager import DetectionManager
    from tracking.byte_tracker import ByteTracker
    from tracking.dwell import DwellTracker, TrackInfo
    from ui.overlay import OverlayRenderer
    from zones.zone import ZoneManager

    device = select_device(cfg.performance.device)
    detector = DetectionManager(cfg, device)
    tracker = ByteTracker(cfg.tracking, cfg.detection.confidence, cfg.detection.unclassified_floor, cfg.source.fps_hint)
    zones = ZoneManager.load(cfg.path(cfg.zones.file), cfg.zones.anchor)
    dwell = DwellTracker(cfg.zones.enter_frames, cfg.zones.exit_frames)
    renderer = OverlayRenderer(cfg)
    face = None
    if enable_face and cfg.face.enabled:
        try:
            from face.pipeline import FacePipeline

            face = FacePipeline(cfg.face, device, None, cfg.path)
        except Exception as exc:
            log.warning("Face stage excluded from benchmark: %s", exc)
    cap = cv2.VideoCapture(parse_uri(cfg.source.uri))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open source {cfg.source.uri!r}")
    W, H = cfg.source.width, cfg.source.height
    stats = LatencyStats(window=frames)
    n, started = 0, time.perf_counter()
    try:
        while n < frames + warmup:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                if not ok:
                    break
            frame = cv2.resize(frame, (W, H))
            if n == warmup:
                started, stats = time.perf_counter(), LatencyStats(window=frames)
            t0 = time.perf_counter()
            dets = detector.detect(frame)
            t1 = time.perf_counter()
            tracks = tracker.update(dets, time.time())
            t2 = time.perf_counter()
            raw = zones.assign(tracks, W, H)
            dwell.update(time.time(), raw, {t.track_id: TrackInfo(t.category, t.label) for t in tracks}, zones.names(),
                         tracker.alive_ids)
            t3 = time.perf_counter()
            if face is not None:
                humans = [t for t in tracks if t.category is Category.HUMAN][: cfg.face.max_faces_per_cycle]
                face.process(frame, humans, time.time())
            t4 = time.perf_counter()
            renderer.render(frame, zones=zones.list(), tracks=tracks, members=raw, counts={}, violated=set(), alerts=[],
                            hud={"fps": 0.0})
            t5 = time.perf_counter()
            for stage, ms in (("detect", t1 - t0), ("track", t2 - t1), ("zones", t3 - t2), ("face", t4 - t3),
                              ("render", t5 - t4), ("total", t5 - t0)):
                stats.add(stage, ms * 1000.0)
            n += 1
    finally:
        cap.release()
    measured = max(0, n - warmup)
    elapsed = max(1e-9, time.perf_counter() - started)
    throughput = measured / elapsed if measured else 0.0
    report = {
        "device": describe_device(device), "model": cfg.detection.model, "imgsz": cfg.performance.imgsz,
        "resolution": f"{W}x{H}", "frames": measured, "throughput_fps": round(throughput, 2),
        "target_fps": cfg.performance.target_fps, "meets_target": throughput >= cfg.performance.target_fps,
        "face_stage": face is not None, "open_vocab": detector.open_vocab is not None, "stages_ms": stats.summary(),
    }
    out_dir = cfg.path(cfg.analytics.export_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"benchmark_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json").write_text(json.dumps(report, indent=2))
    return report
