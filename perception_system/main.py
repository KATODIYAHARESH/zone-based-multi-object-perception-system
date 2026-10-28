#!/usr/bin/env python3
"""ZBMOPS command-line entry point.

    python main.py run [--source 0|file.mp4|rtsp://...] [--headless] [--no-face] [--device cpu]
    python main.py dashboard [--port 8501]
    python main.py zones [--source video.mp4]
    python main.py enroll --name "Asha" --images ./asha --consent-ref FORM-2026-014 --i-confirm-consent
    python main.py unenroll --name "Asha"
    python main.py identities
    python main.py export --table alerts --format csv [--since-hours 24]
    python main.py purge
    python main.py benchmark --source sample.mp4 --frames 300 [--no-face]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import AppConfig, load_config  # noqa: E402
from core.logging_setup import setup_logging  # noqa: E402

log = logging.getLogger("zbmops")


def _load(args: argparse.Namespace) -> AppConfig:
    cfg = load_config(args.config)
    if getattr(args, "source", None) is not None:
        cfg.source.uri = str(args.source)
    if getattr(args, "device", None):
        cfg.performance.device = args.device
    setup_logging(cfg.system.log_level, cfg.path(cfg.system.log_dir))
    return cfg


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _load(args)
    from pipeline.engine import PerceptionEngine

    if args.headless:
        cfg.ui.mode = "headless"
    engine = PerceptionEngine(cfg, enable_face=False if args.no_face else None)
    engine.run_forever(show_window=cfg.ui.mode == "window")
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    cmd = [sys.executable, "-m", "streamlit", "run", str(ROOT / "ui" / "dashboard.py"), "--server.port", str(args.port),
           "--", "--config", str(Path(args.config).resolve())]
    return subprocess.call(cmd, cwd=str(ROOT))


def cmd_zones(args: argparse.Namespace) -> int:
    cfg = _load(args)
    from capture.video_source import VideoSource
    from zones.editor import ZoneEditor
    from zones.zone import ZoneManager

    path = cfg.path(cfg.zones.file)
    manager = ZoneManager.load(path, cfg.zones.anchor)
    source = VideoSource(cfg.source).start()
    try:
        ZoneEditor(source, manager, path, (cfg.source.width, cfg.source.height)).run()
    finally:
        source.stop()
    return 0


def _gallery(cfg: AppConfig):
    from face.gallery import FaceGallery

    return FaceGallery(cfg.path(cfg.face.recognition.gallery_db))


def cmd_enroll(args: argparse.Namespace) -> int:
    if not args.i_confirm_consent:
        print("Refusing to enroll: obtain documented, informed consent first, then pass --i-confirm-consent.",
              file=sys.stderr)
        return 2
    cfg = _load(args)
    from core.device import select_device
    from face.analyzer import FaceAnalyzer
    from face.enrollment import enroll_from_directory

    analyzer = FaceAnalyzer(cfg.face, select_device(cfg.performance.device), cfg.path)
    report = enroll_from_directory(_gallery(cfg), analyzer, args.name, Path(args.images),
                                   consent_reference=args.consent_ref, purpose=args.purpose,
                                   operator=args.operator or os.environ.get("USER", "unknown"),
                                   expires_days=args.expires_days)
    print(json.dumps(report, indent=2))
    return 0


def cmd_unenroll(args: argparse.Namespace) -> int:
    cfg = _load(args)
    ok = _gallery(cfg).revoke(args.name, operator=os.environ.get("USER", "unknown"))
    print(f"{'Revoked' if ok else 'Not found'}: {args.name}")
    return 0 if ok else 1


def cmd_identities(args: argparse.Namespace) -> int:
    cfg = _load(args)
    print(json.dumps(_gallery(cfg).list_identities(), indent=2))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    cfg = _load(args)
    from analytics.exporter import export_table
    from analytics.store import EventStore

    store = EventStore(cfg.path(cfg.analytics.db_path), cfg.path(cfg.analytics.snapshots_dir))
    try:
        since = time.time() - args.since_hours * 3600 if args.since_hours else None
        print(export_table(store, args.table, args.format, cfg.path(cfg.analytics.export_dir), since))
    finally:
        store.close()
    return 0


def cmd_purge(args: argparse.Namespace) -> int:
    cfg = _load(args)
    from analytics.store import EventStore
    from privacy.guard import PrivacyGuard

    store = EventStore(cfg.path(cfg.analytics.db_path), cfg.path(cfg.analytics.snapshots_dir))
    try:
        gallery = _gallery(cfg) if cfg.face.recognition.enabled else None
        print(json.dumps(PrivacyGuard(cfg.privacy).purge(time.time(), store, cfg.path(cfg.system.log_dir), gallery), indent=2))
    finally:
        store.close()
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    cfg = _load(args)
    from tools.benchmark import run_benchmark

    report = run_benchmark(cfg, frames=args.frames, enable_face=not args.no_face)
    print(json.dumps(report, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="zbmops", description="Zone-Based Multi-Object Perception System")
    p.add_argument("--config", default=str(ROOT / "config.yaml"), help="path to config.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run the perception engine")
    r.add_argument("--source", help="override source.uri (webcam index, file, RTSP URL)")
    r.add_argument("--device", help="override performance.device (auto, cpu, cuda:0, mps)")
    r.add_argument("--headless", action="store_true", help="no OpenCV window (stream server only)")
    r.add_argument("--no-face", action="store_true", help="disable the face pipeline")
    r.set_defaults(func=cmd_run)

    d = sub.add_parser("dashboard", help="launch the Streamlit mission-control dashboard")
    d.add_argument("--port", type=int, default=8501)
    d.set_defaults(func=cmd_dashboard)

    z = sub.add_parser("zones", help="interactive OpenCV zone editor")
    z.add_argument("--source")
    z.set_defaults(func=cmd_zones)

    e = sub.add_parser("enroll", help="enroll a CONSENTING person (embeddings only)")
    e.add_argument("--name", required=True)
    e.add_argument("--images", required=True, help="folder with 3-20 photos of the person")
    e.add_argument("--consent-ref", required=True, help="reference to the signed consent record")
    e.add_argument("--purpose", default="site access recognition")
    e.add_argument("--operator")
    e.add_argument("--expires-days", type=int, default=365)
    e.add_argument("--i-confirm-consent", action="store_true")
    e.add_argument("--device")
    e.set_defaults(func=cmd_enroll)

    u = sub.add_parser("unenroll", help="revoke consent and delete embeddings")
    u.add_argument("--name", required=True)
    u.set_defaults(func=cmd_unenroll)

    sub.add_parser("identities", help="list enrolled identities (no biometric data)").set_defaults(func=cmd_identities)

    x = sub.add_parser("export", help="export analytics to CSV / JSON")
    x.add_argument("--table", choices=["events", "alerts", "counts", "emotions"], default="events")
    x.add_argument("--format", choices=["csv", "json"], default="csv")
    x.add_argument("--since-hours", type=float)
    x.set_defaults(func=cmd_export)

    sub.add_parser("purge", help="apply the data-retention policy now").set_defaults(func=cmd_purge)

    b = sub.add_parser("benchmark", help="FPS / latency benchmark")
    b.add_argument("--source")
    b.add_argument("--device")
    b.add_argument("--frames", type=int, default=300)
    b.add_argument("--no-face", action="store_true")
    b.set_defaults(func=cmd_benchmark)
    return p


def main(argv: "list[str] | None" = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # configuration errors are reported cleanly
        if type(exc).__name__ == "ValidationError":
            print(f"configuration error:\n{exc}", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    sys.exit(main())
