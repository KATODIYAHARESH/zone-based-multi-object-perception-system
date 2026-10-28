# Known Limitations & Roadmap

## Known limitations

1. **Expression ≠ emotion.** Classifiers read facial muscle configurations, not internal states. Accuracy drops
   with pose > 30°, masks, faces < 48 px and across cultures. Use it only as an aggregate or assistive signal,
   never for decisions about individuals. Emotion recognition in workplaces and education is prohibited under
   the EU AI Act (Art. 5).
2. **Age** is approximate (typical MAE 5–8 years) and demographically biased; buckets are coarse on purpose.
3. **Recognition** needs near-frontal faces ≥ ~60 px. Twins, ageing and heavy occlusion degrade it. Consent
   expires after 365 days by default, so re-enrol periodically.
4. **Monocular geometry.** The bottom-centre anchor assumes the ground contact point is visible; occluded feet
   shift membership. Mount cameras ≥ 3 m high with 30–60° tilt. There is no metric depth or real-world speed
   without calibration.
5. **Crowds and crossings.** ByteTrack has no appearance model, so IDs can swap when people cross closely; dense
   crowds (> 3 people/m²) degrade recall.
6. **Open vocabulary** is 2–5× slower than YOLO26s, runs sparsely, has lower precision and is sensitive to prompt
   wording.
7. **Motion-based `unclassified`** needs a static camera; shake, sudden lighting changes, rain and vegetation
   cause spurious blobs (mitigated by the zone-only, persistence and coverage filters).
8. **Plate localisation heuristic** produces false positives on signage text. Use a trained plate model for
   production. OCR is intentionally not included.
9. **Night/IR:** COCO-trained weights are daylight-RGB biased.
10. **Scale:** one camera per process, single-node SQLite, Streamlit refresh ≈ 10 Hz (use `/stream.mjpg` for
    full rate).

## Roadmap

### Phase 1: hardening (0–3 months)
* BoT-SORT / DeepSORT with OSNet re-ID embeddings for crossing-robust IDs; multi-camera re-ID.
* Per-zone homography calibration → metres, speed (m/s), density (people/m²).
* Confidence calibration (temperature scaling) per class on site data.
* Encrypted embeddings at rest (SQLCipher / Fernet with a KMS key) and signed, hash-chained audit logs.
* Prometheus `/metrics`, Grafana dashboards, ONVIF discovery, PTZ preset awareness.

### Phase 2: new sensors
* **Thermal camera:** radiometric LWIR (FLIR Boson/Lepton). YOLO26 fine-tuned on FLIR ADAS / LLVIP, RGB-T late
  fusion (box-level WBF) or early fusion (4-channel input), and a thermal-only night mode. Face recognition is
  disabled on thermal by design, which is a privacy gain.
* **Drone feed:** RTSP/RTMP ingest (DJI, MAVLink telemetry), digital stabilisation, SAHI tiled inference for small
  objects at altitude (VisDrone fine-tune), and geo-referenced zones (GeoJSON) projected into the image via GPS,
  altitude and gimbal pose, recomputed every frame.

### Phase 3: edge deployment on NVIDIA Jetson (Orin Nano / NX / AGX)
* Export YOLO26 to **TensorRT** (`format=engine half=True`, or INT8 with a calibration set); run the InsightFace
  and HSEmotion ONNX models through the TensorRT execution provider.
* Zero-copy capture with GStreamer (`nvarguscamerasrc`, `nvv4l2decoder`) or a DeepStream pipeline; JetPack 6
  container base (`nvcr.io/nvidia/l4t-jetpack`).
* `nvpmodel` power profiles, a thermal-throttle watchdog that raises frame skip, and store-and-forward event
  queues while offline.
* Fleet management: OTA updates, remote config with signed bundles, health telemetry to the web console.

### Phase 4: intelligence
* Active learning: `unclassified` flags and low-consensus tracks go to a labelling queue, then fine-tuning.
* Trajectory anomaly detection (wrong-way, sudden running, loitering patterns).
* Natural-language rule authoring compiled to validated rule specs.
