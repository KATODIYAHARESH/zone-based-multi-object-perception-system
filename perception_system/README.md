# ZBMOPS — Zone-Based Multi-Object Perception System

Real-time perception for user-defined zones in a live camera or video feed. It detects, tracks and
classifies **humans** (persistent ID, facial expression, age group, and identity for consenting
enrolled people only), **animals**, **vehicles** (with plate *region*), **food and everyday objects**, and
flags anything it cannot name as **`unclassified`** instead of guessing. Alerts go to the overlay, a log
file, Telegram, email and the optional web console. Analytics are stored in SQLite with snapshots, heatmaps and
CSV/JSON export.

---

## 1. Defaults chosen (no questions asked)

| Area | Default | Why |
|---|---|---|
| Primary detector | **YOLO26s** (`yolo26s.pt`, Ultralytics, Jan 2026), conf 0.35, NMS IoU 0.5 | Latest weights; small model reaches ≥25 FPS at 720p on mid-range GPUs. CPU: `yolo26n.pt` |
| Open-vocabulary fallback | YOLO-World v2 (`yolov8s-worldv2.pt`) every 15 detection frames | Fastest open-vocab option; adds auto-rickshaw, goat, mango, etc. Grounding DINO / YOLOE selectable |
| Tracker | In-house **ByteTrack** + Kalman, class-aware, label voting | Deterministic and unit-tested; IDs issued on confirmation only |
| Zone anchor | Bottom-centre of the box (ground contact) | Where a person or vehicle actually *is* on the floor plane |
| Zone hysteresis | 3 frames to enter, 5 to exit | Removes boundary-jitter false events |
| Face | InsightFace `buffalo_l` (SCRFD/RetinaFace-family + ArcFace + age), HSEmotion `enet_b0_8_best_vgaf` | Single pass for detection, embeddings and age; ONNX on CPU/GPU |
| Emotions | happy, sad, angry, neutral, surprise, fear, disgust (*contempt* folded into *disgust*) | Matches the requested 7 classes |
| Recognition | cosine ≥ 0.45 **and** margin ≥ 0.05 over the next identity | A false identification is worse than "unknown" |
| Dashboard | **Streamlit** (dark mission-control theme) + MJPEG stream server | Fast to operate; MJPEG gives full-rate viewing for remote consoles |
| Storage | SQLite (WAL) + JPEG snapshots, 30-day retention | Zero-ops single node; Postgres via the web console |
| Palette | Amber `#FFB000`, green `#3DDC5F`, red `#FF3B30`, white `#F2F2F2`, warm greys. **No blue** (enforced by `tests/test_theme.py`; heatmaps use `COLORMAP_HOT`) | Requested |

## 2. Architecture

```
            config.yaml (validated, typo-proof)      zones.yaml (editor output)
                         │                                   │
┌────────────┐  latest  ┌▼───────────────────────────────────▼─────────────────────────────┐
│ capture/   │──frame──▶│ pipeline/engine.py — single writer of perception state,          │
│ USB / RTSP │ (thread, │ fault-isolated loop, adaptive frame skipping                     │
│ file, HTTP │ reconnect│                                                                   │
└────────────┘ backoff) │ detection/ ─▶ tracking/ ─▶ zones/ ─▶ tracking/dwell ─▶ alerts/   │
                        │ YOLO26        ByteTrack    point-in-   hysteresis       rules +  │
                        │ +YOLO-World   +Kalman      polygon on  ENTRY/EXIT       cooldown │
                        │ +motion→uncl. +label vote  bottom-ctr  dwell time       per key  │
                        │ +plate region                                                     │
                        └──┬─────────────┬──────────────┬──────────────┬──────────────┬────┘
                    async  │      queue  │       queue  │      publish │        timer │
                  ┌────────▼───┐ ┌───────▼─────┐ ┌──────▼─────┐ ┌──────▼──────┐ ┌─────▼──────┐
                  │ face/      │ │ alerts/     │ │ analytics/ │ │ ui/         │ │ privacy/   │
                  │ SCRFD →    │ │ notifiers:  │ │ SQLite WAL │ │ overlay HUD │ │ pixelate,  │
                  │ HSEmotion →│ │ overlay,log,│ │ snapshots, │ │ MJPEG+JSON  │ │ retention, │
                  │ age group →│ │ Telegram,   │ │ heatmap,   │ │ Streamlit   │ │ consent    │
                  │ ArcFace    │ │ email,      │ │ CSV / JSON │ │ dashboard   │ │ expiry     │
                  │ (consent)  │ │ webhook     │ │            │ │             │ │            │
                  └────────────┘ └──────┬──────┘ └────────────┘ └──────┬──────┘ └────────────┘
                                        │ events + heartbeat           │ /stream.mjpg
                                        ▼                              ▼
                         ┌──────────────────────────────────────────────────────┐
                         │ Optional web console (Next.js + PostgreSQL, ../src)  │
                         └──────────────────────────────────────────────────────┘
```

**Threads:** `capture` → `engine` → {`face-worker`, `alert-dispatch`, `event-store`, `stream-server`}.
Only the engine thread mutates tracker and zone state. All cross-thread traffic goes through bounded queues
(oldest dropped on overflow, counted and logged) or lock-protected snapshots. Network and disk I/O can never stall the
video loop.

## 3. File tree

```
perception_system/
├── main.py                    CLI: run | dashboard | zones | enroll | unenroll | identities | export | purge | benchmark
├── config.yaml                every tunable (pydantic-validated, unknown keys rejected)
├── zones.yaml                 zone polygons (normalised coordinates)
├── requirements.txt / requirements-optional.txt / Dockerfile / docker-compose.yml
├── .streamlit/config.toml     dark amber theme
├── core/        config.py · types.py · boxes.py · device.py · logging_setup.py
├── capture/     video_source.py        threaded latest-frame capture, reconnect with exponential back-off
├── detection/   yolo_detector.py · open_vocab.py · motion.py · plate.py · taxonomy.py · manager.py
├── tracking/    kalman.py · byte_tracker.py · dwell.py
├── face/        analyzer.py · emotion.py · age.py · gallery.py · enrollment.py · pipeline.py
├── zones/       geometry.py · zone.py · editor.py (OpenCV mouse-click editor)
├── alerts/      rules.py · engine.py · notifiers.py
├── analytics/   store.py · heatmap.py · exporter.py · metrics.py
├── privacy/     guard.py
├── ui/          theme.py · overlay.py · stream_server.py · dashboard.py (Streamlit)
├── pipeline/    engine.py
├── tools/       benchmark.py · evaluate.py
├── tests/       test_geometry · test_zones · test_dwell · test_alert_rules · test_tracker · test_theme
└── docs/        EVALUATION.md · ROADMAP.md
```

## 4. Setup

```bash
cd perception_system
python3.11 -m venv .venv && source .venv/bin/activate      # Python 3.10–3.12
pip install --upgrade pip
# GPU (CUDA 12.x): install the matching torch wheel FIRST, e.g.
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
# CUDA for the face models:  pip uninstall -y onnxruntime && pip install onnxruntime-gpu
# optional back-ends:        pip install -r requirements-optional.txt
python -m pytest            # or: python -m unittest discover -s tests -t .
```

Model weights (YOLO26, YOLO-World, InsightFace `buffalo_l`, HSEmotion) download automatically on first use.
`insightface` compiles a small C++ extension, so install `build-essential` / MSVC Build Tools first.

## 5. Run

```bash
python main.py zones --source 0                  # 1. draw zones on the live feed, press 's' to save
python main.py run --source 0                    # 2. OpenCV window (q quit, h heatmap, b blur, p pause, s snapshot)
python main.py run --source rtsp://user:pass@cam/stream --headless   # server mode
python main.py run --source demo.mp4 --no-face --device cpu          # CPU smoke test
python main.py dashboard                         # Streamlit mission control → http://localhost:8501
python main.py benchmark --source demo.mp4 --frames 300
python main.py export --table alerts --format csv --since-hours 24
python main.py purge                             # apply the retention policy now
```

Headless and dashboard modes also serve `http://<host>:8765/stream.mjpg`, `/snapshot.jpg`, `/api/state` and
`/healthz`.

**Zone editor controls:** left-click adds a vertex. Right-click or Enter closes the polygon, then you type a name.
`u` undo · Tab select · `m` allowed/restricted · `1–6` toggle categories (human, animal, vehicle, food, object,
unclassified) · `c` colour · `d` delete · `s` save · `q` quit.

**Zone semantics:** `allowed` treats categories as a whitelist; `restricted` treats them as a blacklist (an empty
blacklist means nothing may enter). Membership is tested on the **bottom-centre** of each bounding box, with
enter/exit hysteresis.

## 6. Alerts

Rules live in `config.yaml → alerts.rules`. They are validated at start-up and hot-reloadable from the dashboard.

| type | example | fires when |
|---|---|---|
| `unknown_person` | unknown person in Zone A | human without an enrolled identity, optional `min_dwell_s` |
| `category_in_zone` | vehicle in pedestrian zone / animal entered farm zone | category (and optional `labels`) present, or on `trigger: entry` |
| `crowd` | crowd > N | count of a category in a zone `> threshold` (falls back to the zone's `max_occupancy`) |
| `expression` | angry expression detected | emotion ≥ `min_confidence` held ≥ `min_duration_s` |
| `dwell` | loitering | dwell > `max_dwell_s` (falls back to the zone's `dwell_limit_s`) |
| `unclassified_dwell` | unattended object | unclassified object remains > `max_dwell_s` |
| `zone_violation` | policy breach | zone mode (allowed/restricted) violated |

Cooldowns apply per *(rule, zone, track)*. Channels are overlay, `data/logs/alerts.jsonl`, Telegram
(`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`), email (`SMTP_HOST`, `SMTP_USER`, `SMTP_PASSWORD`) and webhook
(`INGEST_TOKEN`). Each channel has a minimum severity, a token-bucket rate limit and retries with back-off.
Secrets are **only** read from environment variables.

## 7. Face pipeline & consent

Faces are analysed only inside the head region of tracked persons, asynchronously, every `face.every_n_frames`.
Expression scores are EMA-smoothed per track, and age is reported as a coarse group.

```bash
python main.py enroll --name "Asha Rao" --images ./enroll/asha --consent-ref FORM-2026-014 \
       --purpose "staff access" --expires-days 365 --i-confirm-consent
python main.py identities          # consent records only, no biometric data
python main.py unenroll --name "Asha Rao"   # deletes embeddings immediately (audited)
```

Enrollment requires a consent reference, a purpose and the explicit `--i-confirm-consent` flag. It rejects images
without exactly one face and drops outlier samples. Only **ArcFace embeddings** are stored; images are discarded in
memory. Everyone not enrolled is shown as **`Person #ID`**.

## 8. Privacy & ethics (enforced in code)

* `privacy.monitoring_indicator` is typed `Literal[True]`: the blinking **MONITORING ACTIVE** indicator is drawn on
  every frame, stream and snapshot, and cannot be disabled.
* `privacy.store_face_images` is typed `Literal[False]`: face crops can never be persisted.
* `blur_faces` pixelates the faces of non-enrolled people (12× pixelation is irreversible). When no face has been
  measured yet, a conservative head region is masked instead (fails closed). Alert snapshots are pixelated by
  default (`blur_snapshots`).
* Retention: events, alerts, snapshots, rotated logs and expired consent records older than `retention_days` are
  purged at start-up and every `purge_interval_s`. `python main.py purge` runs it on demand.
* The gallery has an append-only audit table, and consent expires (365 days by default).
* Deployments involving biometrics need a DPIA and legal review (GDPR Art. 9, India DPDP Act 2023, BIPA).
  The **EU AI Act** prohibits emotion recognition in workplaces and education (outside medical or safety use) and
  restricts real-time remote biometric identification in public spaces. Disable `face.emotion` or
  `face.recognition` where required.

## 9. Performance (target ≥ 25 FPS at 720p)

* Threaded latest-frame capture means no queueing latency. The HUD shows FPS and end-to-end latency
  (capture → publish).
* CUDA FP16 is used automatically, with CPU or MPS fallback.
* `detect_every_n_frames` plus adaptive skipping: when the amortised cost exceeds the budget, the skip rises (up to
  `max_skip`). Kalman coasting keeps boxes smooth on skipped frames.
* The face stage runs off-thread (stale jobs dropped). Open-vocab runs sparsely, JPEG encoding is lazy and SQLite
  writes are batched.
* For more headroom, export to TensorRT (`yolo export model=yolo26s.pt format=engine half=True`) and set
  `detection.model: yolo26s.engine`.

## 10. Docker

```bash
docker build -t zbmops .                                    # CPU
docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime -t zbmops:gpu .
docker run --gpus all -p 8765:8765 -v $PWD/data:/app/data --device /dev/video0 zbmops:gpu
docker compose up engine        # or: docker compose --profile dashboard up dashboard
```

## 11. Web console integration (optional)

The repository root contains a Next.js + PostgreSQL **Mission Control Web Console**. Point the engine at it:

```yaml
alerts:
  channels:
    webhook: {enabled: true, url: "http://localhost:3000/api/events", heartbeat_url: "http://localhost:3000/api/heartbeat"}
```

Zones drawn in the console can be exported with `GET /api/zones?format=yaml > zones.yaml`.

**MY CAMERA (`/camera`)** analyses your own webcam entirely in the browser: MediaPipe EfficientDet-Lite0 for objects, face-api (TF.js) for expression, age group and head orientation, a ByteTrack-lite tracker, and the same zone and alert runtime as the simulator. All models are self-hosted in `public/`, so no frame leaves the device. Optional self-enrollment stores a single 128-d embedding in `localStorage` (30-day expiry, "Forget me" deletes it), and event logging to the console is opt-in and metadata-only. The console embeds the
engine's MJPEG feed and stores events in Postgres for multi-user analytics.

## 12. Evaluation, limitations, roadmap

* [docs/EVALUATION.md](docs/EVALUATION.md): mAP, precision/recall, tracking (MOTA/IDF1/HOTA), zone-event F1,
  expression accuracy and fairness, FPS benchmarks, soak and fault-injection tests, acceptance criteria.
* [docs/ROADMAP.md](docs/ROADMAP.md): known limitations; roadmap for thermal cameras, drone feeds and Jetson edge
  deployment.
