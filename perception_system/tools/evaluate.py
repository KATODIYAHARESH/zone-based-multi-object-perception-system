"""Offline evaluation harness.

  python -m tools.evaluate detection --data data/site.yaml [--model yolo26s.pt] [--imgsz 640]
  python -m tools.evaluate emotion --dir datasets/expressions/test     # one sub-folder per class
  python -m tools.evaluate zones --gt annotations/zone_events.json --db data/events.db [--tolerance 1.0]

Zone ground truth format::

  {"run_start_ts": 1767225600.0,
   "events": [{"t": 12.4, "zone_id": "zone_a", "type": "entry", "category": "human"}, ...]}
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.types import EMOTIONS  # noqa: E402


def classification_report(confusion: Mapping[str, Mapping[str, int]]) -> Dict[str, Any]:
    labels = list(confusion)
    total = sum(sum(row.values()) for row in confusion.values())
    per: Dict[str, Dict[str, float]] = {}
    for lbl in labels:
        tp = confusion[lbl].get(lbl, 0)
        fp = sum(confusion[t].get(lbl, 0) for t in labels) - tp
        fn = sum(confusion[lbl].values()) - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        per[lbl] = {"precision": round(p, 4), "recall": round(r, 4),
                    "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0, "support": tp + fn}
    supported = [v for v in per.values() if v["support"]]
    correct = sum(confusion[lbl].get(lbl, 0) for lbl in labels)
    return {"samples": total, "accuracy": round(correct / total, 4) if total else 0.0,
            "macro_f1": round(sum(v["f1"] for v in supported) / len(supported), 4) if supported else 0.0,
            "per_class": per, "confusion": {k: dict(v) for k, v in confusion.items()}}


def match_events(gt: Sequence[Mapping[str, Any]], pred: Sequence[Mapping[str, Any]], tolerance_s: float) -> Dict[str, Any]:
    """Greedy one-to-one matching on (zone, type, category) within +/- tolerance seconds."""
    used, errors, tp = set(), [], 0
    for g in sorted(gt, key=lambda e: e["t"]):
        best, best_dt = None, None
        for i, p in enumerate(pred):
            if i in used or (p["zone_id"], p["type"], p["category"]) != (g["zone_id"], g["type"], g["category"]):
                continue
            dt = abs(p["t"] - g["t"])
            if dt <= tolerance_s and (best_dt is None or dt < best_dt):
                best, best_dt = i, dt
        if best is not None:
            used.add(best)
            errors.append(best_dt)
            tp += 1
    fp, fn = len(pred) - tp, len(gt) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
            "mean_abs_timing_error_s": round(sum(errors) / len(errors), 3) if errors else None}


def eval_detection(model: str, data: str, imgsz: int, device: str) -> Dict[str, Any]:
    from ultralytics import YOLO

    m = YOLO(model).val(data=data, imgsz=imgsz, device=device, verbose=False)
    names = m.names if isinstance(m.names, dict) else dict(enumerate(m.names))
    return {"mAP50": float(m.box.map50), "mAP50_95": float(m.box.map), "precision": float(m.box.mp),
            "recall": float(m.box.mr), "per_class_mAP50_95": {names[i]: float(v) for i, v in enumerate(m.box.maps)}}


def eval_emotion(directory: Path, config_path: str) -> Dict[str, Any]:
    import cv2

    from core.config import load_config
    from face.emotion import CANONICAL, EmotionRecognizer

    rec = EmotionRecognizer(load_config(config_path).face.emotion)
    if not rec.available:
        raise RuntimeError("emotion backend unavailable")
    confusion: Dict[str, Dict[str, int]] = {t: {p: 0 for p in EMOTIONS} for t in EMOTIONS}
    for class_dir in sorted(Path(directory).iterdir()):
        truth = CANONICAL.get(class_dir.name.lower())
        if not class_dir.is_dir() or truth is None:
            continue
        for img_path in class_dir.iterdir():
            img = cv2.imread(str(img_path))
            scores = rec.predict(img) if img is not None else None
            if scores:
                confusion[truth][max(scores, key=scores.get)] += 1
    return classification_report(confusion)


def eval_zone_events(gt_path: Path, db_path: Path, tolerance_s: float) -> Dict[str, Any]:
    gt_doc = json.loads(Path(gt_path).read_text(encoding="utf-8"))
    start = float(gt_doc["run_start_ts"])
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT ts, zone_id, type, category FROM events WHERE ts >= ? ORDER BY ts", (start,)).fetchall()
    pred: List[Dict[str, Any]] = [{"t": ts - start, "zone_id": z, "type": ty, "category": c} for ts, z, ty, c in rows]
    return match_events(gt_doc["events"], pred, tolerance_s)


def main(argv: "List[str] | None" = None) -> int:
    p = argparse.ArgumentParser(description="ZBMOPS evaluation harness")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detection")
    d.add_argument("--data", required=True)
    d.add_argument("--model", default="yolo26s.pt")
    d.add_argument("--imgsz", type=int, default=640)
    d.add_argument("--device", default="")
    e = sub.add_parser("emotion")
    e.add_argument("--dir", required=True)
    e.add_argument("--config", default=str(ROOT / "config.yaml"))
    z = sub.add_parser("zones")
    z.add_argument("--gt", required=True)
    z.add_argument("--db", default=str(ROOT / "data" / "events.db"))
    z.add_argument("--tolerance", type=float, default=1.0)
    args = p.parse_args(argv)
    if args.cmd == "detection":
        result = eval_detection(args.model, args.data, args.imgsz, args.device)
    elif args.cmd == "emotion":
        result = eval_emotion(Path(args.dir), args.config)
    else:
        result = eval_zone_events(Path(args.gt), Path(args.db), args.tolerance)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
