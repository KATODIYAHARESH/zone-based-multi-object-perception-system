# Evaluation Plan

All metrics are reported with dataset version, model hash, config hash and hardware. Acceptance thresholds are
**targets to be validated on site data**, not claims.

## 1. Detection (mAP, precision / recall)

| Dataset | Purpose |
|---|---|
| COCO val2017 | Regression baseline for the 80 closed-set classes |
| IDD (India Driving Dataset) + site footage | auto-rickshaw, bike, cycle, cattle on roads |
| Open Images V7 subset | rare animals, fruits, everyday objects (open-vocabulary prompts) |
| **Site set**: ≥ 2,000 frames, stratified by zone, day/night/rain/glare, labelled by 2 annotators (κ ≥ 0.8) | Operational truth |

* `python -m tools.evaluate detection --data data/site.yaml --model yolo26s.pt` reports mAP@0.5, mAP@0.5:0.95,
  P, R and per-class AP.
* Precision/recall are also reported **at the operating threshold** (`detection.confidence`), with recall broken
  down by object size (S/M/L) and per zone.
* **Open-set behaviour:** on a held-out set of objects outside the taxonomy, measure the fraction flagged
  `unclassified` versus confidently mislabelled.
* Targets: person recall ≥ 0.95 inside zones (M/L objects); vehicle mAP@0.5 ≥ 0.80 on site; unknown objects
  flagged ≥ 80% with ≤ 5% confident wrong labels.

## 2. Tracking

* MOT17 / MOT20 plus annotated site sequences: **MOTA, IDF1, HOTA**, ID switches per 100 tracks (`motmetrics`).
* Dwell-time absolute error (s) against hand-timed ground truth.
* Targets: IDF1 ≥ 0.70 on site sequences; ID switches ≤ 3 per 100 tracks; dwell MAE ≤ 1.0 s.

## 3. Zones & alerts (event level)

* Ground truth: ENTRY/EXIT annotations per zone. Run
  `python -m tools.evaluate zones --gt zone_events.json --db data/events.db --tolerance 1.0`, which gives
  precision, recall, F1 and timing error.
* Alert precision/recall per rule, false alarms per camera-hour, and time-to-alert p95 (frame timestamp → channel
  delivery).
* Targets: zone-event F1 ≥ 0.95; false alarms ≤ 1 per camera-hour for critical rules; time-to-alert p95 ≤ 2 s
  (overlay/log) and ≤ 5 s (Telegram).
* Unit tests (`tests/`) cover the geometry edge cases (boundaries, vertices, concave shapes), hysteresis, every
  rule type, cooldowns and fault isolation.

## 4. Face pipeline

| Task | Data | Metrics |
|---|---|---|
| Face detection | WIDER FACE (easy/medium), site crops | Recall vs face height (24/48/96 px) |
| Expression | RAF-DB, AffectNet-7 test, site set (consented actors) | Accuracy, **macro-F1**, per-class recall, confusion matrix (`tools/evaluate.py emotion`) |
| Age group | UTKFace, Adience | Bucket accuracy, ±1-bucket accuracy, MAE |
| Recognition | LFW / IJB-C protocols + site enrolled set | TAR@FAR=1e-4, FAR at the operating threshold, rank-1 |

* **Fairness audit:** every face metric is broken down by perceived skin tone (Monk scale), age band and gender
  presentation. A gap of more than 5 percentage points blocks release.
* Targets: expression macro-F1 is reported transparently (≈ 0.55–0.65 is typical in the wild, so it is used only as
  an aggregate or assistive signal). Recognition FAR ≤ 0.1% at threshold, which is why the conservative margin
  rule exists.

## 5. Performance (FPS benchmarks)

* Protocol: `python main.py benchmark --source bench_720p.mp4 --frames 300`. Warm up for 20 frames, then do 5 runs
  per configuration. Report mean ± std throughput and p50/p95 per stage (detect, track, zones, face, render, total).
* Matrix: {YOLO26n, YOLO26s, YOLO26s-TensorRT-FP16} × {face on, off} × {open-vocab on, off} ×
  {RTX 4070, RTX 3060, Jetson Orin NX, 8-core x86 CPU}.
* Live loop: the HUD/state reports FPS and capture→publish latency. Targets: **≥ 25 FPS at 1280×720** with e2e
  latency p95 ≤ 120 ms on an RTX 3060-class GPU (YOLO26s, face async). On CPU-only hosts, use YOLO26n with adaptive
  skipping.

## 6. Reliability (soak & fault injection)

| Test | Pass criterion |
|---|---|
| 72 h soak on RTSP | RSS growth < 50 MB, FPS drift < 5%, no thread deaths |
| Camera unplug / RTSP drop | NO SIGNAL shown within read_timeout_s, auto-reconnect with back-off, zero crashes |
| Corrupt frames / decoder errors | frame skipped and logged, loop continues |
| Telegram / SMTP outage | retries with back-off, queue bounded, perception unaffected |
| Disk full | write errors logged, perception continues |
| Bad config value / typo | refuses to start with a precise validation message |

## 7. Release gate

All unit tests green, plus targets in §1–§6 met on the site set, plus the fairness audit passed, plus a signed DPIA.
