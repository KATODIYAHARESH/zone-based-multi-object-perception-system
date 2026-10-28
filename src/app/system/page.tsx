import { promises as fs } from "fs";
import path from "path";
import CodeBrowser from "@/components/CodeBrowser";
import { Panel } from "@/components/ui";
import { SOURCE_ROOT, buildTree, countTree } from "@/lib/source";

export const dynamic = "force-dynamic";
export const metadata = { title: "System // ZBMOPS" };

const ARCHITECTURE = String.raw`
            config.yaml (validated, typo-proof)        zones.yaml (editor / console export)
                         |                                          |
+--------------+ latest  v                                          v
|  capture/    |-frame-> +------------------------------------------------------------------+
|  USB / RTSP  | thread  | pipeline/engine.py  -- single writer of perception state,        |
|  file / HTTP | backoff | fault-isolated loop, adaptive frame skipping, FPS + latency HUD  |
+--------------+         |                                                                  |
                         |  detection/  ->  tracking/  ->  zones/  ->  dwell  ->  alerts/    |
                         |  YOLO26          ByteTrack      point-in-   ENTRY /    rules +    |
                         |  +YOLO-World     +Kalman        polygon on  EXIT       cooldowns  |
                         |  +motion=uncl.   +label vote    bottom-ctr  hysteresis per key    |
                         |  +plate region                                                    |
                         +---+------------+-------------+--------------+-------------+-------+
                             | async      | queue       | queue        | publish     | timer
                     +-------v----+ +-----v-------+ +---v--------+ +---v---------+ +-v----------+
                     | face/      | | alerts/     | | analytics/ | | ui/         | | privacy/   |
                     | SCRFD ->   | | overlay,    | | SQLite WAL | | overlay HUD | | pixelate,  |
                     | HSEmotion->| | JSONL log,  | | snapshots, | | MJPEG+JSON  | | retention, |
                     | age group->| | Telegram,   | | heatmaps,  | | Streamlit   | | consent    |
                     | ArcFace    | | email,      | | CSV / JSON | | dashboard   | | expiry     |
                     | (consent)  | | webhook --+ | |            | |      |      | |            |
                     +------------+ +-----------|-+ +------------+ +------|------+ +------------+
                                                |  events + heartbeat    |  /stream.mjpg
                                                v                        v
                              +----------------------------------------------------------+
                              |  THIS WEB CONSOLE  (Next.js + PostgreSQL via Drizzle)    |
                              |  zones CRUD . live counts . event log . analytics . export|
                              +----------------------------------------------------------+`;

const STEPS: [string, string][] = [
  ["Create env", "cd perception_system && python3.11 -m venv .venv && source .venv/bin/activate"],
  ["GPU (optional)", "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"],
  ["Install", "pip install -r requirements.txt   # + onnxruntime-gpu on CUDA hosts"],
  ["Verify", "python -m pytest   # zone geometry, hysteresis, alert rules, tracker, palette"],
  ["Draw zones", "python main.py zones --source 0   # click vertices, Enter, name, 's' to save"],
  ["Run (window)", "python main.py run --source 0"],
  ["Run (server)", "python main.py run --source rtsp://cam/stream --headless   # :8765/stream.mjpg"],
  ["Dashboard", "python main.py dashboard   # Streamlit mission control on :8501"],
  ["Enroll (consent)", 'python main.py enroll --name "Asha" --images ./asha --consent-ref FORM-014 --i-confirm-consent'],
  ["Benchmark", "python main.py benchmark --source bench_720p.mp4 --frames 300"],
  ["Analyse yourself", "Open MY CAMERA (/camera) in this console - on-device webcam perception, nothing uploaded"],
  ["Console link", "config.yaml -> alerts.channels.webhook.enabled: true  (events + heartbeat to this console)"],
  ["Docker", "docker build -t zbmops . && docker run --gpus all -p 8765:8765 -v $PWD/data:/app/data zbmops"],
];

const EVALUATION: [string, string, string][] = [
  ["Detection", "COCO val2017, IDD, Open Images subset, ≥2k-frame site set", "mAP@.5, mAP@.5:.95, P/R at operating threshold, open-set 'unclassified' rate"],
  ["Tracking", "MOT17 / MOT20 + site sequences", "MOTA, IDF1, HOTA, ID switches / 100 tracks, dwell MAE"],
  ["Zones & alerts", "Annotated ENTRY/EXIT events (±1 s)", "Event P/R/F1, false alarms / camera-hour, time-to-alert p95"],
  ["Expression", "RAF-DB, AffectNet-7, consented site set", "Accuracy, macro-F1, confusion matrix, fairness gap ≤ 5 pp"],
  ["Age / recognition", "UTKFace, Adience, LFW / IJB-C", "Bucket accuracy, TAR@FAR=1e-4, FAR at threshold ≤ 0.1%"],
  ["Performance", "720p benchmark clip, 5 runs × 300 frames", "Throughput FPS (target ≥ 25), stage p50/p95, e2e latency p95"],
  ["Reliability", "72 h soak + fault injection", "Reconnect, no thread deaths, RSS growth < 50 MB"],
];

const LIMITS = [
  "Facial expression ≠ emotion; accuracy drops with pose, masks and small faces. EU AI Act bans emotion recognition in workplaces and education.",
  "Age is approximate (MAE 5-8 yrs) and demographically biased; recognition needs near-frontal faces ≥ 60 px.",
  "Bottom-centre anchor assumes visible ground contact; occluded feet shift zone membership.",
  "ByteTrack has no appearance model, so IDs can swap at close crossings; dense crowds degrade recall.",
  "Open-vocabulary pass is slower and less precise; motion-based 'unclassified' needs a static camera.",
  "Plate heuristic has false positives on signage (no OCR by design); COCO weights are daylight-biased.",
  "MY CAMERA runs lighter in-browser models (EfficientDet-Lite0, face-api) than the engine (YOLO26, HSEmotion, ArcFace): lower accuracy, same zone logic.",
];

const ROADMAP: [string, string][] = [
  ["Hardening", "BoT-SORT / DeepSORT + OSNet re-ID, homography calibration (m, m/s, people/m²), encrypted embeddings, Prometheus metrics"],
  ["Thermal", "Radiometric LWIR, YOLO26 fine-tuned on FLIR ADAS / LLVIP, RGB-T late fusion, thermal-only night mode"],
  ["Drone", "RTSP/RTMP + MAVLink telemetry, stabilisation, SAHI tiling, geo-referenced GeoJSON zones"],
  ["Jetson edge", "TensorRT FP16/INT8, GStreamer nvv4l2decoder zero-copy, DeepStream, nvpmodel watchdog, OTA + store-and-forward"],
];

async function countTests(): Promise<number> {
  try {
    const dir = path.join(SOURCE_ROOT, "tests");
    const files = (await fs.readdir(dir)).filter((f) => f.startsWith("test_") && f.endsWith(".py"));
    let n = 0;
    for (const f of files) n += ((await fs.readFile(path.join(dir, f), "utf8")).match(/def test_/g) ?? []).length;
    return n;
  } catch {
    return 0;
  }
}

export default async function SystemPage() {
  const tree = await buildTree();
  const stats = countTree(tree);
  const tests = await countTests();
  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-lg font-bold tracking-[0.2em]">SYSTEM // PERCEPTION ENGINE DELIVERABLE</h1>
        <p className="text-[11px] tracking-[0.1em] text-hud-grey">
          Python package <span className="text-hud-amber">perception_system/</span>: {stats.pyFiles} modules · {stats.pyLines.toLocaleString()} lines of
          Python · {tests} unit tests · {stats.files} files total
        </p>
      </div>

      <Panel title="ARCHITECTURE" bodyClassName="overflow-x-auto p-3">
        <pre className="text-[10.5px] leading-[1.35] text-hud-amber">{ARCHITECTURE}</pre>
      </Panel>

      <div className="grid gap-4 xl:grid-cols-2">
        <Panel title="STEP-BY-STEP RUN INSTRUCTIONS" bodyClassName="p-0">
          <ol className="divide-y divide-hud-border text-[11px]">
            {STEPS.map(([title, cmd], i) => (
              <li key={title} className="flex gap-3 px-3 py-2">
                <span className="w-5 shrink-0 text-right text-hud-amber">{i + 1}.</span>
                <div className="min-w-0">
                  <div className="text-hud-white">{title}</div>
                  <code className="block overflow-x-auto whitespace-pre text-hud-grey">{cmd}</code>
                </div>
              </li>
            ))}
          </ol>
        </Panel>
        <Panel title="EVALUATION PLAN" bodyClassName="p-0">
          <table className="w-full text-[11px]">
            <thead>
              <tr className="text-hud-grey">
                <th className="px-3 py-2 text-left font-normal">AREA</th>
                <th className="px-3 py-2 text-left font-normal">DATA</th>
                <th className="px-3 py-2 text-left font-normal">METRICS</th>
              </tr>
            </thead>
            <tbody>
              {EVALUATION.map(([area, dataset, metrics]) => (
                <tr key={area} className="border-t border-hud-border align-top">
                  <td className="px-3 py-1.5 text-hud-amber">{area}</td>
                  <td className="px-3 py-1.5 text-hud-grey">{dataset}</td>
                  <td className="px-3 py-1.5 text-hud-white">{metrics}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="border-t border-hud-border px-3 py-2 text-[10px] text-hud-muted">Full protocol and acceptance gates: docs/EVALUATION.md · harness: tools/evaluate.py, tools/benchmark.py</p>
        </Panel>
      </div>

      <Panel title="SOURCE BROWSER" bodyClassName="p-0">
        <CodeBrowser tree={tree} initialPath="README.md" />
      </Panel>

      <div className="grid gap-4 xl:grid-cols-2">
        <Panel title="KNOWN LIMITATIONS">
          <ul className="list-inside list-disc space-y-1.5 text-[11px] leading-relaxed text-hud-grey">
            {LIMITS.map((l) => (
              <li key={l}>{l}</li>
            ))}
          </ul>
        </Panel>
        <Panel title="ROADMAP">
          <ul className="space-y-2 text-[11px]">
            {ROADMAP.map(([phase, text]) => (
              <li key={phase} className="flex gap-3">
                <span className="w-24 shrink-0 text-hud-green">{phase.toUpperCase()}</span>
                <span className="text-hud-grey">{text}</span>
              </li>
            ))}
          </ul>
        </Panel>
      </div>
    </div>
  );
}
