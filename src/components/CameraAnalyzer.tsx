"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Button, Panel, Segmented, Stat } from "@/components/ui";
import {
  ENROLL_EXPIRY_DAYS,
  MATCH_THRESHOLD,
  buildEnrollment,
  clearEnrollment,
  describeCameraError,
  euclidean,
  inIframe,
  listCameras,
  loadEnrollment,
  openCamera,
  saveEnrollment,
  stopStream,
  type CameraErrorKind,
  type SelfEnrollment,
} from "@/lib/camera";
import { drawCameraFrame, headRegion, type DrawFace, type DrawTrack } from "@/lib/cameraRender";
import { categoryOf, displayLabel } from "@/lib/taxonomy";
import { ByteTrackLite, DEFAULT_TRACKER_OPTIONS, intersectionOverArea, type TrackedObject, type TrackerDetection } from "@/lib/tracker";
import {
  CATEGORIES,
  CATEGORY_COLOR,
  DEFAULT_SETTINGS,
  EMOTIONS,
  EMOTION_COLOR,
  PALETTE,
  SEVERITY_COLOR,
  isCategory,
  type Emotion,
  type IngestEvent,
  type SettingsData,
  type Severity,
  type ZoneDTO,
} from "@/lib/types";
import { ageGroup, analyzeFaces, headLabel, headYaw, loadVisionStack, personFromFace, switchObjectDelegate, topEmotion, type FaceObservation, type VisionStack } from "@/lib/vision";
import { ZoneRuntime, type AnchorMode, type Box, type RuntimeAlert, type RuntimeTrack } from "@/lib/zoneRuntime";

type Phase = "idle" | "starting" | "running" | "error";

interface FaceState {
  box: Box;
  /** face box relative to the person box at measurement time (privacy + smooth overlay) */
  rel: Box;
  score: number;
  probs: Record<Emotion, number>;
  emotion: Emotion;
  emotionConf: number;
  emotionSince: number;
  age: number | null;
  ageGroup: string | null;
  identity: string | null;
  identityDist: number | null;
  landmarks: [number, number][];
  yaw: number | null;
  updatedAt: number;
}

interface LogItem {
  id: number;
  ts: number;
  kind: "entry" | "exit" | "alert";
  severity: Severity | null;
  text: string;
}

interface Subject {
  name: string;
  trackId: number | null;
  enrolled: boolean;
  score: number;
  trackedS: number;
  zones: string[];
  dwellS: number;
  emotion: Emotion | null;
  emotionConf: number;
  heldS: number;
  probs: Record<Emotion, number> | null;
  age: number | null;
  ageGroup: string | null;
  head: string;
  faceVisible: boolean;
}

interface UiSnapshot {
  fps: number;
  latencyMs: number;
  faceMs: number;
  detectMs: number;
  skip: number;
  trackCount: number;
  faces: number;
  scene: { id: number; name: string; category: string; score: number }[];
  categoryCounts: Record<string, number>;
  subject: Subject | null;
  log: LogItem[];
  timeline: (Emotion | null)[];
  frames: number;
  durationS: number;
  alerts: number;
  posted: number;
}

const FACE_INTERVAL_MS = 90;
const DESCRIPTOR_INTERVAL_MS = 1000;
const FACE_TTL_MS = 1500;
const TIMELINE_LEN = 120;
const ENROLL_SAMPLES = 5;
const ANCHORS = ["auto", "bottom_center", "center"] as const;

const ema = (prev: number, v: number, a: number) => (prev === 0 ? v : prev + a * (v - prev));
const area = (b: Box) => Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
const hhmmss = (ts: number) => new Date(ts).toISOString().slice(11, 19);
const round1 = (v: number) => Math.round(v * 10) / 10;
const toRel = (f: Box, b: Box): Box => {
  const w = Math.max(1, b[2] - b[0]);
  const h = Math.max(1, b[3] - b[1]);
  return [(f[0] - b[0]) / w, (f[1] - b[1]) / h, (f[2] - b[0]) / w, (f[3] - b[1]) / h];
};
const project = (rel: Box, b: Box): Box => {
  const w = b[2] - b[0];
  const h = b[3] - b[1];
  return [b[0] + rel[0] * w, b[1] + rel[1] * h, b[0] + rel[2] * w, b[1] + rel[3] * h];
};

function fmtDuration(totalS: number): string {
  const s = Math.max(0, Math.floor(totalS));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="flex cursor-pointer items-center gap-1.5">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      {label}
    </label>
  );
}

function Row({ k, v, color }: { k: string; v: string; color?: string }) {
  return (
    <div className="flex justify-between gap-3 border-t border-hud-border pt-1.5">
      <span className="shrink-0 text-hud-grey">{k}</span>
      <span className="text-right" style={{ color: color ?? PALETTE.white }}>
        {v}
      </span>
    </div>
  );
}

export default function CameraAnalyzer() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const stackRef = useRef<VisionStack | null>(null);
  const trackerRef = useRef<ByteTrackLite>(new ByteTrackLite({ ...DEFAULT_TRACKER_OPTIONS }));
  const runtimeRef = useRef<ZoneRuntime | null>(null);
  const facesRef = useRef<Map<number, FaceState>>(new Map());
  const looseRef = useRef<FaceObservation[]>([]);
  const lastFacesRef = useRef<{ at: number; faces: FaceObservation[] }>({ at: 0, faces: [] });
  const tracksRef = useRef<TrackedObject[]>([]);
  const alertsRef = useRef<RuntimeAlert[]>([]);
  const logRef = useRef<LogItem[]>([]);
  const queueRef = useRef<IngestEvent[]>([]);
  const timelineRef = useRef<(Emotion | null)[]>([]);
  const zonesRef = useRef<ZoneDTO[]>([]);
  const settingsRef = useRef<SettingsData>(DEFAULT_SETTINGS);
  const enrollRef = useRef<SelfEnrollment | null>(null);
  const collectRef = useRef<{ name: string; samples: Float32Array[]; startedAt: number; lastAt: number } | null>(null);
  const rafRef = useRef(0);
  const frameTimesRef = useRef<number[]>([]);
  const loopRef = useRef({ detections: 0, delegateSwitch: false, frame: 0, lastTs: 0, skip: 1, faceBusy: false, lastFaceAt: 0, lastDescAt: 0, fps: 0, latency: 0, faceMs: 0, detectMs: 0, lastFrameAt: 0, objectErrors: 0, loopErrors: 0 });
  const sessionRef = useRef({ startedAt: 0, frames: 0, alerts: 0, posted: 0, seq: 0 });
  const optsRef = useRef({ mirror: true, blur: false, landmarks: true, paused: false, anchor: "auto" as AnchorMode, log: false });

  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<{ kind: CameraErrorKind | "models"; message: string } | null>(null);
  const [progress, setProgress] = useState<string[]>([]);
  const [stackInfo, setStackInfo] = useState<{ objects: string; faces: string; errors: string[] } | null>(null);
  const [zones, setZones] = useState<ZoneDTO[]>([]);
  const [mirror, setMirror] = useState(true);
  const [blur, setBlur] = useState(false);
  const [landmarks, setLandmarks] = useState(true);
  const [paused, setPaused] = useState(false);
  const [anchor, setAnchor] = useState<AnchorMode>("auto");
  const [logEvents, setLogEvents] = useState(false);
  const [cameras, setCameras] = useState<MediaDeviceInfo[]>([]);
  const [deviceId, setDeviceId] = useState("");
  const [ui, setUi] = useState<UiSnapshot | null>(null);
  const [enrollment, setEnrollment] = useState<SelfEnrollment | null>(null);
  const [enrollName, setEnrollName] = useState("");
  const [enrollConsent, setEnrollConsent] = useState(false);
  const [enrolling, setEnrolling] = useState(false);
  const [enrollMsg, setEnrollMsg] = useState<{ text: string; tone: "ok" | "err" | "busy" } | null>(null);
  const [embedded, setEmbedded] = useState(false);
  const [runtimeError, setRuntimeError] = useState<string | null>(null);

  const faceTtl = () => Math.max(FACE_TTL_MS, loopRef.current.faceMs * 2.5 + 250);

  useEffect(() => {
    optsRef.current = { mirror, blur, landmarks, paused, anchor, log: logEvents };
  }, [mirror, blur, landmarks, paused, anchor, logEvents]);

  useEffect(() => {
    const e = loadEnrollment();
    enrollRef.current = e;
    setEnrollment(e);
    setEmbedded(inIframe());
    (async () => {
      try {
        const [z, s] = await Promise.all([
          fetch("/api/zones", { cache: "no-store" }).then((r) => r.json()),
          fetch("/api/settings", { cache: "no-store" }).then((r) => r.json()),
        ]);
        const list = (z.zones ?? []) as ZoneDTO[];
        zonesRef.current = list;
        setZones(list);
        if (s.settings) {
          settingsRef.current = s.settings as SettingsData;
          setBlur(Boolean((s.settings as SettingsData).blurFaces));
        }
      } catch {
        /* console defaults are used */
      }
    })();
    return () => {
      cancelAnimationFrame(rafRef.current);
      stopStream(streamRef.current);
      streamRef.current = null;
    };
  }, []);

  /* ------------------------------------------------------------ face pipeline */
  const updateFaceState = useCallback((id: number, f: FaceObservation, now: number, trackBox: Box) => {
    const prev = facesRef.current.get(id);
    const a = 0.6; // EMA weight of history (= face.emotion.smoothing in config.yaml)
    const probs = {} as Record<Emotion, number>;
    for (const e of EMOTIONS) probs[e] = prev ? a * prev.probs[e] + (1 - a) * f.probs[e] : f.probs[e];
    const emotion = topEmotion(probs);
    const age = prev?.age != null ? 0.8 * prev.age + 0.2 * f.age : f.age;
    let identity = prev?.identity ?? null;
    let identityDist = prev?.identityDist ?? null;
    const enr = enrollRef.current;
    if (!enr) {
      identity = null;
      identityDist = null;
    } else if (f.descriptor) {
      identityDist = euclidean(f.descriptor, enr.descriptor);
      identity = identityDist < MATCH_THRESHOLD ? enr.name : null;
    }
    facesRef.current.set(id, {
      box: f.box,
      rel: toRel(f.box, trackBox),
      score: f.score,
      probs,
      emotion,
      emotionConf: probs[emotion],
      emotionSince: prev && prev.emotion === emotion ? prev.emotionSince : now,
      age,
      ageGroup: ageGroup(age),
      identity,
      identityDist,
      landmarks: f.landmarks,
      yaw: headYaw(f.landmarks),
      updatedAt: now,
    });
  }, []);

  const collectSample = useCallback((obs: FaceObservation[], now: number) => {
    const c = collectRef.current;
    if (!c) return;
    if (now - c.startedAt > 20000) {
      collectRef.current = null;
      setEnrolling(false);
      setEnrollMsg({ text: "Enrollment timed out - make sure exactly one face is clearly visible and well lit.", tone: "err" });
      return;
    }
    if (obs.length !== 1) {
      setEnrollMsg({ text: `Exactly one face must be visible (found ${obs.length}).`, tone: "busy" });
      return;
    }
    const f = obs[0];
    if (!f.descriptor || f.score < 0.6 || f.box[2] - f.box[0] < 70) {
      setEnrollMsg({ text: "Move closer and face the camera directly.", tone: "busy" });
      return;
    }
    if (now - c.lastAt < 350) return;
    c.samples.push(f.descriptor);
    c.lastAt = now;
    if (c.samples.length < ENROLL_SAMPLES) {
      setEnrollMsg({ text: `Capturing ${c.samples.length}/${ENROLL_SAMPLES} - turn your head slightly between samples...`, tone: "busy" });
      return;
    }
    collectRef.current = null;
    setEnrolling(false);
    const result = buildEnrollment(c.name, c.samples);
    if (!result.enrollment) {
      setEnrollMsg({ text: result.error ?? "Enrollment failed.", tone: "err" });
      return;
    }
    saveEnrollment(result.enrollment);
    enrollRef.current = result.enrollment;
    setEnrollment(result.enrollment);
    setEnrollMsg({ text: `Enrolled "${result.enrollment.name}" from ${result.kept} samples - stored only in this browser.`, tone: "ok" });
  }, []);

  const facePass = useCallback(
    async (stack: VisionStack, video: HTMLVideoElement, started: number) => {
      const ls = loopRef.current;
      const faceapi = stack.faceapi;
      if (!faceapi) return;
      const wantDesc = collectRef.current !== null || (enrollRef.current !== null && started - ls.lastDescAt >= DESCRIPTOR_INTERVAL_MS);
      if (wantDesc) ls.lastDescAt = started;
      let obs: FaceObservation[];
      try {
        const inputSize = ls.faceMs > 1200 ? 224 : ls.faceMs > 400 ? 320 : 416; // adaptive cost
        obs = await analyzeFaces(faceapi, video, wantDesc, inputSize);
      } catch (err) {
        console.warn("[camera] face pass failed", err);
        return;
      }
      const done = performance.now();
      ls.faceMs = ema(ls.faceMs, done - started, 0.2);
      lastFacesRef.current = { at: done, faces: obs };
      if (collectRef.current) collectSample(obs, done);
      // Associate faces with person tracks: face centre inside the box, prefer high-in-box faces.
      const humans = tracksRef.current.filter((t) => t.category === "human");
      const used = new Set<number>();
      const loose: FaceObservation[] = [];
      for (const f of obs) {
        const fx = (f.box[0] + f.box[2]) / 2;
        const fy = (f.box[1] + f.box[3]) / 2;
        let bestId = -1;
        let bestBox: Box | null = null;
        let bestScore = 0;
        for (const tr of humans) {
          if (used.has(tr.id)) continue;
          const [x1, y1, x2, y2] = tr.bbox;
          if (fx < x1 || fx > x2 || fy < y1 || fy > y2) continue;
          const topness = 1 - Math.min(1, Math.max(0, (fy - y1) / Math.max(1, y2 - y1)));
          const score = 0.7 * intersectionOverArea(f.box, tr.bbox) + 0.3 * topness;
          if (score > bestScore) {
            bestScore = score;
            bestId = tr.id;
            bestBox = tr.bbox;
          }
        }
        if (bestId > 0 && bestBox) {
          used.add(bestId);
          updateFaceState(bestId, f, done, bestBox);
        } else loose.push(f);
      }
      looseRef.current = loose;
    },
    [collectSample, updateFaceState],
  );

  /* ------------------------------------------------------------ detection */
  const detectObjects = useCallback((stack: VisionStack, video: HTMLVideoElement, now: number, W: number, H: number): TrackerDetection[] => {
    const ls = loopRef.current;
    const out: TrackerDetection[] = [];
    const detector = stack.objects;
    if (detector) {
      try {
        const ts = Math.max(now, ls.lastTs + 1); // MediaPipe VIDEO mode needs strictly increasing timestamps
        ls.lastTs = ts;
        const res = detector.detectForVideo(video, ts);
        for (const d of res.detections) {
          const c = d.categories[0];
          const bb = d.boundingBox;
          if (!c || !bb || !c.categoryName || c.categoryName === "???") continue;
          const x1 = Math.max(0, bb.originX);
          const y1 = Math.max(0, bb.originY);
          const x2 = Math.min(W, bb.originX + bb.width);
          const y2 = Math.min(H, bb.originY + bb.height);
          if (x2 - x1 < 4 || y2 - y1 < 4) continue;
          out.push({ bbox: [x1, y1, x2, y2], score: c.score, label: displayLabel(c.categoryName), category: categoryOf(c.categoryName) });
        }
        ls.objectErrors = 0;
      } catch (err) {
        ls.objectErrors += 1;
        if (ls.objectErrors === 1) console.warn("[camera] object detection failed", err);
        if (ls.objectErrors > 30) {
          stack.objects = null;
          setRuntimeError("The object detector failed repeatedly and was disabled - continuing with face-based person tracking.");
        }
      }
    } else {
      // Degraded mode: derive person detections from the latest face pass.
      const lf = lastFacesRef.current;
      if (now - lf.at < 600) for (const f of lf.faces) out.push({ bbox: personFromFace(f.box, W, H), score: Math.max(0.5, f.score), label: "person", category: "human" });
    }
    return out;
  }, []);

  const handleEvents = useCallback((events: IngestEvent[], alerts: RuntimeAlert[]) => {
    const s = sessionRef.current;
    for (const ev of events) {
      s.seq += 1;
      const who = ev.category === "human" ? `Person #${ev.trackId}` : `${ev.label ?? "?"} #${ev.trackId}`;
      logRef.current.unshift(
        ev.type === "alert"
          ? { id: s.seq, ts: Date.now(), kind: "alert", severity: ev.severity ?? "info", text: ev.message ?? "" }
          : {
              id: s.seq,
              ts: Date.now(),
              kind: ev.type,
              severity: null,
              text: `${ev.zoneName ?? ev.zoneId ?? "-"} · ${who}${ev.type === "exit" && ev.dwellS != null ? ` · ${ev.dwellS.toFixed(0)}s` : ""}`,
            },
      );
    }
    if (logRef.current.length > 80) logRef.current.length = 80;
    if (alerts.length) {
      s.alerts += alerts.length;
      alertsRef.current = [...alerts.slice().reverse(), ...alertsRef.current].slice(0, 6);
    }
    if (optsRef.current.log) queueRef.current.push(...events);
  }, []);

  /* ------------------------------------------------------------ frame loop */
  const loop = useCallback(() => {
    rafRef.current = requestAnimationFrame(loop);
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const stack = stackRef.current;
    const runtime = runtimeRef.current;
    if (!video || !canvas || !stack || !runtime || video.readyState < 2 || !video.videoWidth) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const ls = loopRef.current;
    const o = optsRef.current;
    try {
      const W = video.videoWidth;
      const H = video.videoHeight;
      if (canvas.width !== W || canvas.height !== H) {
        canvas.width = W;
        canvas.height = H;
      }
      runtime.width = W;
      runtime.height = H;
      runtime.anchorMode = o.anchor;
      runtime.zones = zonesRef.current;
      runtime.settings = settingsRef.current;
      const now = performance.now();
      let tracks = tracksRef.current;
      if (!o.paused) {
        ls.frame += 1;
        sessionRef.current.frames += 1;
        const tracker = trackerRef.current;
        tracker.opts.highThresh = settingsRef.current.detectionConfidence;
        if (ls.frame % ls.skip === 0) {
          const td = performance.now();
          const dets = detectObjects(stack, video, now, W, H);
          ls.detectMs = ema(ls.detectMs, performance.now() - td, 0.15);
          ls.detections += 1;
          tracks = tracker.update(dets, now / 1000);
        } else tracks = tracker.coast();
        tracksRef.current = tracks;
        if (ls.frame % 30 === 0) {
          // Adaptive frame skipping (same idea as performance.adaptive_skip in the engine).
          if (ls.detectMs > 40 && ls.skip < 3) ls.skip += 1;
          else if (ls.skip > 1 && ls.detectMs < 18) ls.skip -= 1;
        }
        if (stack.objects && stack.objectsDelegate === "GPU" && !ls.delegateSwitch && ls.detections >= 15 && ls.detectMs > 150) {
          // Software WebGL / blocklisted GPU: the WASM (XNNPACK) path is far faster - swap delegates live.
          ls.delegateSwitch = true;
          void switchObjectDelegate(stack, "CPU")
            .then(() => {
              ls.detectMs = 0;
              ls.skip = 1;
              setStackInfo((si) => (si ? { ...si, objects: "EfficientDet-Lite0 · MediaPipe CPU (auto-fallback: GPU path too slow)" } : si));
            })
            .catch((err) => console.warn("[camera] CPU delegate fallback failed", err));
        }
        if (stack.faceapi && !ls.faceBusy && now - ls.lastFaceAt >= Math.max(FACE_INTERVAL_MS, ls.faceMs * 0.5)) {
          ls.faceBusy = true;
          ls.lastFaceAt = now;
          void facePass(stack, video, now).finally(() => {
            ls.faceBusy = false;
          });
        }
        const alive = tracker.aliveIds();
        for (const id of Array.from(facesRef.current.keys())) if (!alive.has(id)) facesRef.current.delete(id);
        const ttl = faceTtl();
        const rtracks: RuntimeTrack[] = tracks.map((tr) => {
          const f = facesRef.current.get(tr.id);
          const fresh = f && now - f.updatedAt < ttl ? f : null;
          return {
            id: tr.id,
            category: tr.category,
            label: tr.label,
            bbox: tr.bbox,
            emotion: fresh?.emotion ?? null,
            emotionConf: fresh?.emotionConf ?? 0,
            emotionSince: fresh ? fresh.emotionSince / 1000 : null,
            ageGroup: fresh?.ageGroup ?? null,
            identity: f?.identity ?? null,
          };
        });
        const res = runtime.update(now / 1000, rtracks, alive);
        if (res.events.length || res.alerts.length) handleEvents(res.events, res.alerts);
      }
      const ttl = faceTtl();
      const lf = lastFacesRef.current;
      const blurBoxes: Box[] = [];
      if (o.blur) {
        // Fail-closed privacy: known faces follow their person box between face passes;
        // people without a measured face get a generous head mask; raw detections are masked too.
        for (const tr of tracks) {
          if (tr.category !== "human") continue;
          const f = facesRef.current.get(tr.id);
          blurBoxes.push(f ? project(f.rel, tr.bbox) : headRegion(tr.bbox, H));
        }
        if (now - lf.at < ttl * 2) for (const f of lf.faces) blurBoxes.push(f.box);
      }
      const drawTracks: DrawTrack[] = tracks.map((tr) => {
        const f = facesRef.current.get(tr.id);
        const fresh = f && now - f.updatedAt < ttl ? f : null;
        const fbox = fresh ? project(fresh.rel, tr.bbox) : null;
        const dx = fresh && fbox ? fbox[0] - fresh.box[0] : 0;
        const dy = fresh && fbox ? fbox[1] - fresh.box[1] : 0;
        return {
          id: tr.id,
          bbox: tr.bbox,
          score: tr.score,
          label: tr.label,
          category: tr.category,
          coasting: tr.coasting,
          face:
            fresh && fbox
              ? {
                  box: fbox,
                  landmarks: dx || dy ? fresh.landmarks.map(([x, y]) => [x + dx, y + dy] as [number, number]) : fresh.landmarks,
                  emotion: fresh.emotion,
                  emotionConf: fresh.emotionConf,
                  ageGroup: fresh.ageGroup,
                  identity: fresh.identity,
                }
              : null,
          identity: f?.identity ?? null,
          inZone: (runtime.memberships.get(tr.id)?.size ?? 0) > 0,
        };
      });
      const looseFaces: DrawFace[] =
        now - lf.at < ttl
          ? looseRef.current.map((f) => {
              const e = topEmotion(f.probs);
              return { box: f.box, landmarks: f.landmarks, emotion: e, emotionConf: f.probs[e], ageGroup: ageGroup(f.age), identity: null };
            })
          : [];
      drawCameraFrame(ctx, video, {
        zones: zonesRef.current,
        violated: runtime.violated,
        counts: runtime.counts(tracks),
        tracks: drawTracks,
        looseFaces,
        blurBoxes,
        alerts: alertsRef.current,
        hud: {
          fps: ls.fps,
          latencyMs: ls.latency,
          faceMs: ls.faceMs,
          objects: stack.objects ? `EFFDET-L0/${stack.objectsDelegate}` : "OFF",
          faces: stack.faceapi ? `FACEAPI/${(stack.faceBackend ?? "").toUpperCase()}` : "OFF",
        },
        mirror: o.mirror,
        blur: o.blur,
        landmarks: o.landmarks,
        anchorMode: o.anchor,
        paused: o.paused,
      });
      ls.latency = ema(ls.latency, performance.now() - now, 0.1);
      const ft = frameTimesRef.current;
      ft.push(now);
      while (ft.length > 2 && now - ft[0] > 2000) ft.shift();
      ls.fps = ft.length > 1 ? ((ft.length - 1) * 1000) / Math.max(1, ft[ft.length - 1] - ft[0]) : 0;
      ls.lastFrameAt = now;
    } catch (err) {
      ls.loopErrors += 1;
      if (ls.loopErrors % 100 === 1) console.error("[camera] frame failed", err);
    }
  }, [detectObjects, facePass, handleEvents]);

  /* ------------------------------------------------------------ lifecycle */
  const watchStream = useCallback((stream: MediaStream) => {
    stream.getVideoTracks()[0]?.addEventListener("ended", () => {
      if (streamRef.current !== stream) return;
      cancelAnimationFrame(rafRef.current);
      streamRef.current = null;
      setError({ kind: "busy", message: "The camera stopped (disconnected, or taken by another application)." });
      setPhase("error");
    });
  }, []);

  const start = useCallback(
    async (devId?: string) => {
      setError(null);
      setRuntimeError(null);
      setPhase("starting");
      setProgress(["Requesting camera permission..."]);
      let stream: MediaStream;
      try {
        stream = await openCamera(devId || undefined);
      } catch (err) {
        setError(describeCameraError(err));
        setPhase("error");
        return;
      }
      stopStream(streamRef.current);
      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) {
        stopStream(stream);
        return;
      }
      video.srcObject = stream;
      video.muted = true;
      video.playsInline = true;
      await video.play().catch(() => undefined);
      watchStream(stream);
      const track = stream.getVideoTracks()[0];
      setDeviceId(track?.getSettings().deviceId ?? devId ?? "");
      listCameras()
        .then(setCameras)
        .catch(() => undefined);
      setProgress((p) => [...p, `Camera online: ${track?.label || "default camera"} (${video.videoWidth}x${video.videoHeight}).`]);
      let stack: VisionStack;
      try {
        stack = await loadVisionStack((m) => setProgress((p) => [...p, m].slice(-10)), 0.2);
      } catch (err) {
        stopStream(stream);
        streamRef.current = null;
        setError({ kind: "models", message: `Vision models failed to load: ${err instanceof Error ? err.message : String(err)}` });
        setPhase("error");
        return;
      }
      if (streamRef.current !== stream) return; // stopped while loading
      stackRef.current = stack;
      setStackInfo({
        objects: stack.objects ? `EfficientDet-Lite0 · MediaPipe ${stack.objectsDelegate}` : "unavailable",
        faces: stack.faceapi ? `face-api · TF.js ${String(stack.faceBackend).toUpperCase()}` : "unavailable",
        errors: stack.errors,
      });
      trackerRef.current.reset();
      facesRef.current.clear();
      looseRef.current = [];
      lastFacesRef.current = { at: 0, faces: [] };
      tracksRef.current = [];
      alertsRef.current = [];
      logRef.current = [];
      queueRef.current = [];
      timelineRef.current = [];
      frameTimesRef.current = [];
      Object.assign(loopRef.current, { detections: 0, delegateSwitch: false, frame: 0, skip: 1, faceBusy: false, lastFaceAt: 0, lastDescAt: 0, fps: 0, latency: 0, faceMs: 0, detectMs: 0, lastFrameAt: 0, objectErrors: 0, loopErrors: 0 });
      sessionRef.current = { startedAt: Date.now(), frames: 0, alerts: 0, posted: 0, seq: 0 };
      runtimeRef.current = new ZoneRuntime(zonesRef.current, settingsRef.current, video.videoWidth || 1280, video.videoHeight || 720, optsRef.current.anchor, { camera: true });
      setPhase("running");
      cancelAnimationFrame(rafRef.current);
      rafRef.current = requestAnimationFrame(loop);
    },
    [loop, watchStream],
  );

  const stop = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    stopStream(streamRef.current);
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    collectRef.current = null;
    setEnrolling(false);
    setPaused(false);
    setPhase("idle");
  }, []);

  const switchCamera = useCallback(
    async (id: string) => {
      setDeviceId(id);
      try {
        const s = await openCamera(id);
        const old = streamRef.current;
        streamRef.current = s;
        stopStream(old);
        watchStream(s);
        const v = videoRef.current;
        if (v) {
          v.srcObject = s;
          await v.play().catch(() => undefined);
        }
        trackerRef.current.reset();
        facesRef.current.clear();
        tracksRef.current = [];
        setRuntimeError(null);
      } catch (err) {
        setRuntimeError(describeCameraError(err).message);
      }
    },
    [watchStream],
  );

  const snapshot = useCallback(() => {
    canvasRef.current?.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `zbmops-camera-${new Date().toISOString().replace(/[:.]/g, "-")}.png`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 2000);
    }, "image/png");
  }, []);

  /* ------------------------------------------------------------ UI snapshot (4 Hz) */
  const computeSubject = useCallback((now: number): Subject | null => {
    const runtime = runtimeRef.current;
    const t = now / 1000;
    const ttl = faceTtl();
    const candidates = tracksRef.current
      .filter((tr) => tr.category === "human")
      .map((tr) => {
        const f = facesRef.current.get(tr.id);
        const face = f && now - f.updatedAt < ttl ? f : null;
        return { tr, face, identity: f?.identity ?? null, rank: face ? 1e9 + area(face.box) : area(tr.bbox) };
      })
      .sort((a, b) => b.rank - a.rank);
    const top = candidates[0];
    if (top) {
      const slugs = Array.from(runtime?.memberships.get(top.tr.id) ?? []);
      const dwellS = slugs.reduce((m, slug) => Math.max(m, runtime ? runtime.dwell(top.tr.id, slug, t) : 0), 0);
      const f = top.face;
      return {
        name: top.identity ?? `Person #${top.tr.id}`,
        trackId: top.tr.id,
        enrolled: Boolean(top.identity),
        score: top.tr.score,
        trackedS: t - top.tr.firstSeen,
        zones: slugs.map((slug) => runtime?.zoneName(slug) ?? slug),
        dwellS,
        emotion: f?.emotion ?? null,
        emotionConf: f?.emotionConf ?? 0,
        heldS: f ? (now - f.emotionSince) / 1000 : 0,
        probs: f?.probs ?? null,
        age: f?.age ?? null,
        ageGroup: f?.ageGroup ?? null,
        head: headLabel(f?.yaw ?? null),
        faceVisible: Boolean(f),
      };
    }
    const lf = lastFacesRef.current;
    if (now - lf.at < ttl && lf.faces.length) {
      const f = lf.faces.reduce((a, b) => (area(b.box) > area(a.box) ? b : a));
      const e = topEmotion(f.probs);
      return {
        name: "Face (acquiring person track)",
        trackId: null,
        enrolled: false,
        score: f.score,
        trackedS: 0,
        zones: [],
        dwellS: 0,
        emotion: e,
        emotionConf: f.probs[e],
        heldS: 0,
        probs: f.probs,
        age: f.age,
        ageGroup: ageGroup(f.age),
        head: headLabel(headYaw(f.landmarks)),
        faceVisible: true,
      };
    }
    return null;
  }, []);

  useEffect(() => {
    if (phase !== "running") return;
    let tick = 0;
    const id = window.setInterval(() => {
      tick += 1;
      const now = performance.now();
      const ls = loopRef.current;
      const tracks = tracksRef.current;
      const subject = computeSubject(now);
      if (tick % 4 === 0 && !optsRef.current.paused) {
        timelineRef.current.push(subject?.emotion ?? null);
        if (timelineRef.current.length > TIMELINE_LEN) timelineRef.current.shift();
      }
      const categoryCounts: Record<string, number> = Object.fromEntries(CATEGORIES.map((c) => [c, 0]));
      for (const tr of tracks) categoryCounts[tr.category] += 1;
      const scene = tracks
        .filter((tr) => tr.category !== "human")
        .sort((a, b) => b.score - a.score)
        .slice(0, 8)
        .map((tr) => ({ id: tr.id, name: tr.category === "unclassified" ? `UNCLASSIFIED #${tr.id}` : `${tr.label} #${tr.id}`, category: tr.category, score: tr.score }));
      const s = sessionRef.current;
      setUi({
        fps: ls.fps,
        latencyMs: ls.latency,
        faceMs: ls.faceMs,
        detectMs: ls.detectMs,
        skip: ls.skip,
        trackCount: tracks.length,
        faces: lastFacesRef.current.faces.length,
        scene,
        categoryCounts,
        subject,
        log: logRef.current.slice(0, 40),
        timeline: [...timelineRef.current],
        frames: s.frames,
        durationS: (Date.now() - s.startedAt) / 1000,
        alerts: s.alerts,
        posted: s.posted,
      });
    }, 250);
    return () => window.clearInterval(id);
  }, [phase, computeSubject]);

  /* ------------------------------------------------------------ optional console logging */
  useEffect(() => {
    if (phase !== "running" || !logEvents) return;
    const flush = async () => {
      const batch = queueRef.current.splice(0, 400);
      if (!batch.length) return;
      try {
        const res = await fetch("/api/events", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ source: "camera", events: batch }) });
        if (res.ok) sessionRef.current.posted += ((await res.json()) as { accepted: number }).accepted;
      } catch {
        /* best effort - metadata only */
      }
    };
    const beat = async () => {
      const ls = loopRef.current;
      const tracks = tracksRef.current;
      const st = stackRef.current;
      const emotions: Record<string, number> = Object.fromEntries(EMOTIONS.map((e) => [e, 0]));
      const now = performance.now();
      for (const tr of tracks) {
        const f = facesRef.current.get(tr.id);
        if (tr.category === "human" && f && now - f.updatedAt < faceTtl()) emotions[f.emotion] += 1;
      }
      await fetch("/api/heartbeat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source: "camera",
          fps: round1(ls.fps),
          latencyMs: round1(ls.latency),
          device: `BROWSER WEBCAM · ${st?.objectsDelegate ?? "NO-OBJ"} · ${(st?.faceBackend ?? "no-face").toUpperCase()}`,
          frameSkip: ls.skip,
          tracks: tracks.length,
          counts: runtimeRef.current?.counts(tracks) ?? {},
          emotions,
        }),
      }).catch(() => undefined);
    };
    const i1 = window.setInterval(() => void flush(), 1500);
    const i2 = window.setInterval(() => void beat(), 3000);
    return () => {
      window.clearInterval(i1);
      window.clearInterval(i2);
      void flush();
    };
  }, [phase, logEvents]);

  /* ------------------------------------------------------------ enrollment */
  const startEnroll = () => {
    if (!stackRef.current?.faceapi) {
      setEnrollMsg({ text: "Face models are not available.", tone: "err" });
      return;
    }
    collectRef.current = { name: enrollName.trim(), samples: [], startedAt: performance.now(), lastAt: 0 };
    setEnrolling(true);
    setEnrollMsg({ text: "Look at the camera...", tone: "busy" });
  };

  const forget = () => {
    clearEnrollment();
    enrollRef.current = null;
    setEnrollment(null);
    for (const f of facesRef.current.values()) {
      f.identity = null;
      f.identityDist = null;
    }
    setEnrollMsg({ text: "Your embedding was deleted from this browser.", tone: "ok" });
  };

  /* ------------------------------------------------------------ render */
  const running = phase === "running";
  const subject = ui?.subject ?? null;
  const timeline = ui?.timeline ?? [];
  const padded: (Emotion | null)[] = [...Array.from({ length: Math.max(0, TIMELINE_LEN - timeline.length) }, () => null), ...timeline];
  const seen = timeline.filter((e): e is Emotion => e !== null);
  const share = EMOTIONS.map((e) => ({ e, pct: seen.length ? (100 * seen.filter((x) => x === e).length) / seen.length : 0 }));
  const fps = ui?.fps ?? 0;
  const fpsColor = fps >= 25 ? PALETTE.green : fps >= 15 ? PALETTE.amber : PALETTE.red;
  const badge =
    phase === "running" ? (
      <span className="flex items-center gap-1.5 text-[10px] font-semibold tracking-[0.14em] text-hud-white">
        <span className="blink h-2 w-2 rounded-full bg-hud-red" /> {paused ? "PAUSED" : "LIVE · ON-DEVICE"}
      </span>
    ) : phase === "starting" ? (
      <span className="text-[10px] tracking-[0.14em] text-hud-amber">INITIALISING...</span>
    ) : phase === "error" ? (
      <span className="text-[10px] tracking-[0.14em] text-hud-red">FAULT</span>
    ) : (
      <span className="text-[10px] tracking-[0.14em] text-hud-muted">STANDBY</span>
    );

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_400px]">
      <div className="min-w-0 space-y-4">
        <Panel title="MY CAMERA · ON-DEVICE PERCEPTION" right={badge} bodyClassName="p-0">
          <div className="scanlines relative aspect-video w-full overflow-hidden bg-black">
            <video ref={videoRef} className="absolute inset-0 h-full w-full object-contain" playsInline muted aria-hidden />
            <canvas ref={canvasRef} className="absolute inset-0 h-full w-full object-contain" aria-label="Analysed camera feed" />
            {!running ? (
              <div className="absolute inset-0 z-10 overflow-y-auto bg-hud-bg/95 p-5 sm:p-8">
                {phase === "error" && error ? (
                  <div className="mx-auto flex h-full max-w-xl flex-col items-center justify-center gap-3 text-center">
                    <div className="text-xl font-bold tracking-[0.3em] text-hud-red">
                      {error.kind === "denied" ? "CAMERA BLOCKED" : error.kind === "models" ? "MODEL LOAD FAILURE" : error.kind === "notfound" ? "NO CAMERA" : "CAMERA FAULT"}
                    </div>
                    <p className="text-[12px] leading-relaxed text-hud-white">{error.message}</p>
                    <div className="flex flex-wrap justify-center gap-2">
                      <Button variant="primary" onClick={() => void start(deviceId)}>
                        TRY AGAIN
                      </Button>
                      {embedded || error.kind === "denied" ? (
                        <a href="/camera" target="_blank" rel="noreferrer" className="rounded border border-hud-amber px-3 py-1.5 text-[11px] font-semibold tracking-[0.14em] text-hud-amber hover:bg-hud-amber/10">
                          OPEN IN NEW TAB ↗
                        </a>
                      ) : null}
                      <Button
                        onClick={() => {
                          setError(null);
                          setPhase("idle");
                        }}
                      >
                        BACK
                      </Button>
                    </div>
                  </div>
                ) : phase === "starting" ? (
                  <div className="mx-auto flex h-full max-w-xl flex-col justify-center gap-2">
                    <div className="text-sm font-bold tracking-[0.24em] text-hud-amber">INITIALISING PERCEPTION STACK</div>
                    <ul className="space-y-1 text-[11px] text-hud-grey">
                      {progress.map((m, i) => (
                        <li key={`${i}-${m}`} className={i === progress.length - 1 ? "text-hud-white" : ""}>
                          {i === progress.length - 1 ? "▸ " : "✓ "}
                          {m}
                        </li>
                      ))}
                    </ul>
                    <p className="text-[10px] text-hud-muted">First start downloads ~30 MB of self-hosted models; later starts are cached by the browser.</p>
                    <div>
                      <Button onClick={stop}>CANCEL</Button>
                    </div>
                  </div>
                ) : (
                  <div className="mx-auto max-w-2xl space-y-4">
                    <div className="flex items-center gap-3">
                      <span className="blink h-3 w-3 rounded-full bg-hud-red shadow-[0_0_8px_#FF3B30]" />
                      <h2 className="text-sm font-bold tracking-[0.22em] text-hud-amber">ANALYSE ME · ON-DEVICE CAMERA PERCEPTION</h2>
                    </div>
                    <ul className="space-y-2 text-[12px] leading-relaxed text-hud-white">
                      <li>
                        ▸ <b className="text-hud-amber">Runs 100% in this browser</b>: objects (EfficientDet-Lite0, COCO-80), facial expression (7 classes), approximate age
                        group and head orientation. No video, images or face data are uploaded.
                      </li>
                      <li>
                        ▸ You appear as <b>Person #ID</b>. Recognition by name happens only if you enroll yourself (optional; the embedding stays in this browser).
                      </li>
                      <li>▸ This console&apos;s zones and alert rules are applied live: step into a zone, hold up a phone, pull a face.</li>
                      <li className="text-hud-grey">▸ Expression is not emotion. Results are approximate and for demonstration only; never use them to make decisions about people.</li>
                    </ul>
                    <label className="flex items-start gap-2 text-[11px] text-hud-grey">
                      <input type="checkbox" className="mt-0.5" checked={logEvents} onChange={(e) => setLogEvents(e.target.checked)} />
                      Also log zone events and alerts to the console database (metadata only: no images, no face data, no names). Off by default.
                    </label>
                    {ui ? (
                      <p className="text-[11px] text-hud-grey">
                        LAST SESSION: {fmtDuration(ui.durationS)} · {ui.frames.toLocaleString()} frames · {ui.alerts} alert(s)
                      </p>
                    ) : null}
                    <div className="flex flex-wrap items-center gap-3">
                      <Button variant="primary" onClick={() => void start(deviceId)}>
                        ● START CAMERA &amp; ANALYSE ME
                      </Button>
                      <span className="text-[10px] text-hud-muted">Your browser will ask for camera permission. You can stop at any time.</span>
                    </div>
                    {embedded ? (
                      <p className="text-[10px] text-hud-muted">
                        Viewing inside an embedded preview? If no permission prompt appears,{" "}
                        <a href="/camera" target="_blank" rel="noreferrer" className="text-hud-amber underline">
                          open this page in a new tab
                        </a>
                        .
                      </p>
                    ) : null}
                  </div>
                )}
              </div>
            ) : null}
          </div>
          {running ? (
            <>
              <div className="grid grid-cols-2 gap-2 border-t border-hud-border p-3 sm:grid-cols-3 lg:grid-cols-6">
                <Stat label="FPS" value={fps.toFixed(1)} color={fpsColor} />
                <Stat label="FRAME" value={`${(ui?.latencyMs ?? 0).toFixed(1)} ms`} />
                <Stat label="DETECT" value={`${(ui?.detectMs ?? 0).toFixed(0)} ms · 1/${ui?.skip ?? 1}`} />
                <Stat label="FACE PASS" value={`${(ui?.faceMs ?? 0).toFixed(0)} ms`} />
                <Stat label="TRACKS" value={ui?.trackCount ?? 0} />
                <Stat label="FACES" value={ui?.faces ?? 0} color={ui?.faces ? PALETTE.green : undefined} />
              </div>
              <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border-t border-hud-border px-3 py-2.5 text-[11px] text-hud-grey">
                <Button variant="danger" onClick={stop}>
                  ■ STOP CAMERA
                </Button>
                <Button variant={paused ? "primary" : "default"} onClick={() => setPaused((p) => !p)}>
                  {paused ? "RESUME" : "PAUSE"}
                </Button>
                <Button onClick={snapshot}>SNAPSHOT ↓</Button>
                <Toggle label="MIRROR" checked={mirror} onChange={setMirror} />
                <Toggle label="PIXELATE FACES" checked={blur} onChange={setBlur} />
                <Toggle label="LANDMARKS" checked={landmarks} onChange={setLandmarks} />
                <Toggle
                  label="LOG TO CONSOLE"
                  checked={logEvents}
                  onChange={(v) => {
                    setLogEvents(v);
                    if (!v) queueRef.current = [];
                  }}
                />
                <span className="flex items-center gap-2">
                  ANCHOR
                  <Segmented<AnchorMode> value={anchor} options={ANCHORS} labels={{ auto: "Auto", bottom_center: "Bottom", center: "Centre" }} onChange={setAnchor} />
                </span>
                {cameras.length > 1 ? (
                  <select className="hud-input w-auto max-w-[220px]" value={deviceId} onChange={(e) => void switchCamera(e.target.value)} aria-label="Camera">
                    {cameras.map((c, i) => (
                      <option key={c.deviceId || i} value={c.deviceId}>
                        {c.label || `Camera ${i + 1}`}
                      </option>
                    ))}
                  </select>
                ) : null}
              </div>
            </>
          ) : null}
        </Panel>
        {runtimeError ? <p className="text-xs text-hud-red">{runtimeError}</p> : null}
        {stackInfo?.errors.length ? <p className="text-[11px] text-hud-amber">Degraded mode: {stackInfo.errors.join(" · ")}</p> : null}
        <Panel title="TRY THIS">
          <ol className="grid gap-2 text-[11px] leading-relaxed text-hud-grey sm:grid-cols-2">
            <li>
              <span className="text-hud-amber">1.</span> Hold an <b className="text-hud-red">angry</b> expression for ~2 s while inside a zone: expression alert.
            </li>
            <li>
              <span className="text-hud-amber">2.</span> Move until the dot on your box sits inside <b className="text-hud-white">ZONE A</b> for 2 s: &quot;unknown person&quot; alert. Enroll
              yourself and it stops.
            </li>
            <li>
              <span className="text-hud-amber">3.</span> Hold up a <b className="text-hud-white">phone, cup or bottle</b> inside Zone A (humans only): zone-policy violation.
            </li>
            <li>
              <span className="text-hud-amber">4.</span> Stay in Zone A for 2 minutes: loitering alert. Toggle <b className="text-hud-white">PIXELATE FACES</b> to see privacy masking.
            </li>
          </ol>
          <p className="mt-2 text-[10px] leading-relaxed text-hud-muted">
            Zones come from the ZONES page (you can draw them over a captured camera frame). Anchor AUTO uses the bottom-centre point and falls back to the box centre when
            your feet are out of frame. Mirroring is display-only; analysis always uses true camera geometry.
          </p>
        </Panel>
      </div>

      <aside className="space-y-4">
        <Panel title="SUBJECT ANALYSIS" right={<span className="text-[10px] text-hud-muted">{subject?.faceVisible ? "FACE LOCK" : running ? "SEARCHING" : "OFFLINE"}</span>}>
          {subject ? (
            <div className="space-y-1.5 text-[11px]">
              <div className="flex items-baseline justify-between gap-2 pb-1">
                <span className="truncate text-base font-bold" style={{ color: subject.enrolled ? PALETTE.green : PALETTE.white }}>
                  {subject.name}
                </span>
                <span className={`shrink-0 text-[10px] ${subject.enrolled ? "text-hud-green" : "text-hud-muted"}`}>{subject.enrolled ? "✓ ENROLLED MATCH" : "NOT IDENTIFIED"}</span>
              </div>
              <Row k="DETECTION" v={`${(subject.score * 100).toFixed(0)}%`} />
              <Row k="TRACKED FOR" v={subject.trackId === null ? "-" : fmtDuration(subject.trackedS)} />
              <Row
                k="EXPRESSION"
                v={subject.emotion ? `${subject.emotion.toUpperCase()} ${(subject.emotionConf * 100).toFixed(0)}% · held ${subject.heldS.toFixed(1)} s` : "-"}
                color={subject.emotion ? EMOTION_COLOR[subject.emotion] : undefined}
              />
              <Row k="AGE GROUP" v={subject.ageGroup ? `${subject.ageGroup} (est. ${Math.round(subject.age ?? 0)}, approx.)` : "-"} />
              <Row k="HEAD" v={subject.head} />
              <Row
                k="ZONES"
                v={subject.zones.length ? `${subject.zones.join(", ")} · ${fmtDuration(subject.dwellS)}` : "outside all zones"}
                color={subject.zones.length ? PALETTE.amber : undefined}
              />
            </div>
          ) : (
            <p className="text-[11px] text-hud-muted">{running ? "No person in view - step in front of the camera." : "Start the camera to analyse yourself."}</p>
          )}
        </Panel>

        <Panel title="EXPRESSION METER · LIVE">
          <div className="space-y-1.5">
            {EMOTIONS.map((e) => {
              const v = subject?.probs?.[e] ?? 0;
              return (
                <div key={e}>
                  <div className="flex justify-between text-[10px] tracking-[0.14em]">
                    <span style={{ color: EMOTION_COLOR[e] }}>{e.toUpperCase()}</span>
                    <span className="tabular-nums text-hud-grey">{(v * 100).toFixed(0)}%</span>
                  </div>
                  <div className="h-1.5 overflow-hidden rounded bg-hud-panel2">
                    <div className="h-full transition-all duration-300" style={{ width: `${v * 100}%`, background: EMOTION_COLOR[e] }} />
                  </div>
                </div>
              );
            })}
          </div>
          <div className="mt-3 text-[10px] tracking-[0.14em] text-hud-grey">DOMINANT EXPRESSION · LAST {TIMELINE_LEN} s</div>
          <div className="mt-1 flex h-4 gap-px overflow-hidden rounded">
            {padded.map((e, i) => (
              <div key={i} className="flex-1" style={{ background: e ? EMOTION_COLOR[e] : "#1d1e19" }} title={e ?? "no face"} />
            ))}
          </div>
          <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[10px]">
            {share
              .filter((x) => x.pct > 0)
              .map(({ e, pct }) => (
                <span key={e} style={{ color: EMOTION_COLOR[e] }}>
                  {e} {pct.toFixed(0)}%
                </span>
              ))}
          </div>
          <p className="mt-2 text-[10px] leading-relaxed text-hud-muted">Facial expression is not emotion: approximate and for demonstration only.</p>
        </Panel>

        <Panel title="SCENE · TRACKED OBJECTS">
          <div className="mb-2 flex flex-wrap gap-1.5">
            {CATEGORIES.map((c) => (
              <span key={c} className="rounded border px-1.5 py-0.5 text-[10px]" style={{ borderColor: CATEGORY_COLOR[c], color: CATEGORY_COLOR[c] }}>
                {c.slice(0, 3).toUpperCase()} {ui?.categoryCounts[c] ?? 0}
              </span>
            ))}
          </div>
          <ul className="space-y-1 text-[11px]">
            {(ui?.scene ?? []).map((o) => (
              <li key={o.id} className="flex justify-between">
                <span style={{ color: isCategory(o.category) ? CATEGORY_COLOR[o.category] : PALETTE.white }}>{o.name}</span>
                <span className="tabular-nums text-hud-grey">{o.score.toFixed(2)}</span>
              </li>
            ))}
            {running && !ui?.scene.length ? <li className="text-hud-muted">Nothing besides people - hold up a phone, cup or book.</li> : null}
          </ul>
        </Panel>

        <Panel title="RECOGNITION · OPT-IN · LOCAL ONLY">
          {enrollment ? (
            <div className="space-y-2 text-[11px]">
              <div>
                Enrolled: <span className="font-semibold text-hud-green">{enrollment.name}</span>
              </div>
              <div className="text-hud-muted">
                {enrollment.samples} samples · 128-number embedding · expires {new Date(enrollment.expiresAt).toISOString().slice(0, 10)}
              </div>
              <div>
                Status:{" "}
                {subject?.enrolled ? <span className="text-hud-green">MATCHED IN VIEW</span> : <span className="text-hud-grey">{running ? "not matched" : "camera off"}</span>}
              </div>
              <Button variant="danger" onClick={forget}>
                FORGET ME
              </Button>
            </div>
          ) : (
            <div className="space-y-2">
              <input className="hud-input" placeholder="Your name" maxLength={40} value={enrollName} onChange={(e) => setEnrollName(e.target.value)} />
              <label className="flex items-start gap-2 text-[10px] leading-relaxed text-hud-grey">
                <input type="checkbox" className="mt-0.5" checked={enrollConsent} onChange={(e) => setEnrollConsent(e.target.checked)} />I consent to storing a face embedding of
                myself (numbers only, never an image) in this browser for {ENROLL_EXPIRY_DAYS} days. I can delete it at any time.
              </label>
              <Button variant="primary" disabled={!running || !enrollName.trim() || !enrollConsent || enrolling || !stackRef.current?.faceapi} onClick={startEnroll}>
                {enrolling ? "CAPTURING..." : "ENROLL ME"}
              </Button>
            </div>
          )}
          {enrollMsg ? (
            <p className={`mt-2 text-[11px] ${enrollMsg.tone === "ok" ? "text-hud-green" : enrollMsg.tone === "err" ? "text-hud-red" : "text-hud-amber"}`}>{enrollMsg.text}</p>
          ) : null}
          <p className="mt-2 text-[10px] text-hud-muted">Nothing is uploaded. Anyone else in view stays &quot;Person #ID&quot;. Names never appear in logged events.</p>
        </Panel>

        <Panel title="SESSION LOG" right={<span className="text-[10px] text-hud-muted">{logEvents ? `LOGGING · ${ui?.posted ?? 0} SENT` : "LOCAL ONLY"}</span>} bodyClassName="p-0">
          <ul className="max-h-[260px] divide-y divide-hud-border overflow-y-auto text-[11px]">
            {(ui?.log ?? []).length === 0 ? <li className="p-3 text-hud-muted">Zone entries, exits and alerts appear here.</li> : null}
            {(ui?.log ?? []).map((l) => {
              const color = l.kind === "alert" ? SEVERITY_COLOR[l.severity ?? "info"] : PALETTE.grey;
              return (
                <li key={l.id} className="flex gap-2 px-3 py-1.5">
                  <span className="shrink-0 tabular-nums text-hud-muted">{hhmmss(l.ts)}</span>
                  <span className="shrink-0 font-semibold" style={{ color }}>
                    {l.kind === "alert" ? (l.severity ?? "info").toUpperCase().slice(0, 4) : l.kind.toUpperCase()}
                  </span>
                  <span className="min-w-0 flex-1 truncate" style={{ color: l.kind === "alert" ? color : PALETTE.white }} title={l.text}>
                    {l.text}
                  </span>
                </li>
              );
            })}
          </ul>
        </Panel>

        <Panel title="SESSION · MODEL STACK">
          <div className="space-y-1.5 text-[11px]">
            <Row k="DURATION" v={ui ? fmtDuration(ui.durationS) : "-"} />
            <Row k="FRAMES ANALYSED" v={ui ? ui.frames.toLocaleString() : "-"} />
            <Row k="ALERTS" v={String(ui?.alerts ?? 0)} color={ui?.alerts ? PALETTE.amber : undefined} />
            <Row k="OBJECTS" v={stackInfo?.objects ?? "not loaded"} />
            <Row k="FACES" v={stackInfo?.faces ?? "not loaded"} />
            <Row k="TRACKER" v="ByteTrack-lite · class-aware" />
            <Row k="ZONES" v={`${zones.filter((z) => z.enabled).length} active · shared runtime`} />
          </div>
        </Panel>
      </aside>
    </div>
  );
}
