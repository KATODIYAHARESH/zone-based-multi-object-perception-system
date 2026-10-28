/**
 * In-browser vision stack (client-only, lazy-loaded):
 *   - MediaPipe Tasks ObjectDetector (EfficientDet-Lite0, COCO-80) - GPU delegate, CPU fallback
 *   - face-api (TF.js): TinyFaceDetector, 68 landmarks, 7-class expression, age, 128-d descriptor
 * All model files are self-hosted under /public, so analysis works without any CDN
 * and no frame ever leaves the browser.
 */
import type { ObjectDetector } from "@mediapipe/tasks-vision";
import type { Box } from "./zoneRuntime";
import { EMOTIONS, type Emotion } from "./types";

type FaceApiModule = typeof import("@vladmandic/face-api/dist/face-api.esm.js");
type MediaPipeModule = typeof import("@mediapipe/tasks-vision");
type Fileset = Awaited<ReturnType<MediaPipeModule["FilesetResolver"]["forVisionTasks"]>>;

export const ASSETS = {
  wasm: "/mediapipe/wasm",
  objectModel: "/models/mediapipe/efficientdet_lite0.tflite",
  objectModelCdn: "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite",
  faceModels: "/models/face-api",
} as const;

export interface VisionStack {
  objects: ObjectDetector | null;
  objectsDelegate: "GPU" | "CPU" | null;
  fileset: Fileset | null;
  objectModelPath: string | null;
  scoreThreshold: number;
  faceapi: FaceApiModule | null;
  faceBackend: string | null;
  errors: string[];
}

export interface FaceObservation {
  box: Box;
  score: number;
  probs: Record<Emotion, number>;
  age: number;
  landmarks: [number, number][];
  descriptor: Float32Array | null;
}

type TfLike = { setBackend(name: string): Promise<boolean>; ready(): Promise<void>; getBackend(): string };

const errText = (e: unknown) => (e instanceof Error ? e.message : String(e));

async function loadObjects(progress: (m: string) => void, scoreThreshold: number) {
  progress("Loading MediaPipe vision runtime (WASM)...");
  const { FilesetResolver, ObjectDetector } = await import("@mediapipe/tasks-vision");
  const fileset = await FilesetResolver.forVisionTasks(`${window.location.origin}${ASSETS.wasm}`);
  let lastErr: unknown = null;
  for (const modelAssetPath of [ASSETS.objectModel, ASSETS.objectModelCdn]) {
    for (const delegate of ["GPU", "CPU"] as const) {
      try {
        progress(`Creating object detector: EfficientDet-Lite0 (${delegate} delegate)...`);
        const detector = await ObjectDetector.createFromOptions(fileset, {
          baseOptions: { modelAssetPath, delegate },
          runningMode: "VIDEO",
          scoreThreshold,
          maxResults: 25,
        });
        return { detector, delegate, fileset, modelAssetPath };
      } catch (err) {
        lastErr = err;
      }
    }
  }
  throw new Error(errText(lastErr));
}

async function loadFaces(progress: (m: string) => void) {
  progress("Loading TF.js face runtime...");
  const faceapi = await import("@vladmandic/face-api/dist/face-api.esm.js");
  const tf = faceapi.tf as unknown as TfLike;
  const webgl = await tf.setBackend("webgl").catch(() => false);
  if (!webgl) await tf.setBackend("cpu");
  await tf.ready();
  progress("Loading face nets: detector, 68 landmarks, expression, age, recognition...");
  await Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri(ASSETS.faceModels),
    faceapi.nets.faceLandmark68Net.loadFromUri(ASSETS.faceModels),
    faceapi.nets.faceExpressionNet.loadFromUri(ASSETS.faceModels),
    faceapi.nets.ageGenderNet.loadFromUri(ASSETS.faceModels),
    faceapi.nets.faceRecognitionNet.loadFromUri(ASSETS.faceModels),
  ]);
  progress("Warming up face networks (shader compilation)...");
  try {
    const c = document.createElement("canvas");
    c.width = 160;
    c.height = 160;
    const g = c.getContext("2d");
    if (g) {
      g.fillStyle = "#808080";
      g.fillRect(0, 0, 160, 160);
    }
    const nets = faceapi.nets as unknown as Record<string, Record<string, (input: HTMLCanvasElement) => Promise<unknown>>>;
    await faceapi.detectAllFaces(c, new faceapi.TinyFaceDetectorOptions({ inputSize: 416, scoreThreshold: 0.5 }));
    await nets.faceLandmark68Net.detectLandmarks?.(c);
    await nets.faceExpressionNet.predictExpressions?.(c);
    await nets.ageGenderNet.predictAgeAndGender?.(c);
    await nets.faceRecognitionNet.computeFaceDescriptor?.(c);
  } catch {
    /* warm-up is an optimisation only */
  }
  return { faceapi, backend: tf.getBackend() };
}

let stackPromise: Promise<VisionStack> | null = null;

/** Load (once per page) both model families in parallel; degrades gracefully if one fails. */
export function loadVisionStack(progress: (m: string) => void, scoreThreshold = 0.2): Promise<VisionStack> {
  if (!stackPromise) {
    stackPromise = (async () => {
      const [o, f] = await Promise.allSettled([loadObjects(progress, scoreThreshold), loadFaces(progress)]);
      const stack: VisionStack = { objects: null, objectsDelegate: null, fileset: null, objectModelPath: null, scoreThreshold, faceapi: null, faceBackend: null, errors: [] };
      if (o.status === "fulfilled") {
        stack.objects = o.value.detector;
        stack.objectsDelegate = o.value.delegate;
        stack.fileset = o.value.fileset;
        stack.objectModelPath = o.value.modelAssetPath;
      } else stack.errors.push(`Object detector unavailable: ${errText(o.reason)}`);
      if (f.status === "fulfilled") {
        stack.faceapi = f.value.faceapi;
        stack.faceBackend = f.value.backend;
      } else stack.errors.push(`Face models unavailable: ${errText(f.reason)}`);
      if (!stack.objects && !stack.faceapi) throw new Error(stack.errors.join(" | "));
      return stack;
    })().catch((err) => {
      stackPromise = null;
      throw err;
    });
  }
  return stackPromise;
}

/**
 * Re-create the object detector on another delegate and swap it in atomically.
 * Used when the GPU path turns out to be slower than WASM/XNNPACK (software WebGL,
 * blocklisted GPUs, remote desktops).
 */
export async function switchObjectDelegate(stack: VisionStack, delegate: "GPU" | "CPU"): Promise<void> {
  if (!stack.fileset || !stack.objectModelPath) throw new Error("object detector not initialised");
  const { ObjectDetector } = await import("@mediapipe/tasks-vision");
  const next = await ObjectDetector.createFromOptions(stack.fileset, {
    baseOptions: { modelAssetPath: stack.objectModelPath, delegate },
    runningMode: "VIDEO",
    scoreThreshold: stack.scoreThreshold,
    maxResults: 25,
  });
  const old = stack.objects;
  stack.objects = next;
  stack.objectsDelegate = delegate;
  try {
    old?.close();
  } catch {
    /* already closed */
  }
}

/** One face pass: detection -> landmarks -> expression -> age (-> descriptor). */
export async function analyzeFaces(faceapi: FaceApiModule, video: HTMLVideoElement, withDescriptors: boolean, inputSize = 416): Promise<FaceObservation[]> {
  const options = new faceapi.TinyFaceDetectorOptions({ inputSize, scoreThreshold: 0.5 });
  const task = faceapi.detectAllFaces(video, options).withFaceLandmarks().withFaceExpressions().withAgeAndGender();
  const out: FaceObservation[] = [];
  if (withDescriptors) {
    for (const r of await task.withFaceDescriptors()) out.push(toObservation(r, r.descriptor));
  } else {
    for (const r of await task) out.push(toObservation(r, null));
  }
  return out;
}

interface RawFace {
  detection: { box: { x: number; y: number; width: number; height: number }; score: number };
  landmarks: { positions: { x: number; y: number }[] };
  expressions: { neutral: number; happy: number; sad: number; angry: number; fearful: number; disgusted: number; surprised: number };
  age: number;
}

function toObservation(r: RawFace, descriptor: Float32Array | null): FaceObservation {
  const b = r.detection.box;
  const e = r.expressions;
  // Gender output is discarded on purpose (data minimisation) - it is never read.
  return {
    box: [b.x, b.y, b.x + b.width, b.y + b.height],
    score: r.detection.score,
    probs: { happy: e.happy, sad: e.sad, angry: e.angry, neutral: e.neutral, surprise: e.surprised, fear: e.fearful, disgust: e.disgusted },
    age: r.age,
    landmarks: r.landmarks.positions.map((p) => [p.x, p.y] as [number, number]),
    descriptor,
  };
}

export function topEmotion(probs: Record<Emotion, number>): Emotion {
  return EMOTIONS.reduce((best, e) => (probs[e] > probs[best] ? e : best), EMOTIONS[0]);
}

export const AGE_GROUPS: [number, number, string][] = [
  [0, 12, "child"],
  [13, 19, "teen"],
  [20, 35, "20-35"],
  [36, 55, "36-55"],
  [56, 120, "56+"],
];

export function ageGroup(age: number | null): string | null {
  if (age == null || !Number.isFinite(age) || age < 0) return null;
  const a = Math.round(age);
  for (const [lo, hi, name] of AGE_GROUPS) if (a >= lo && a <= hi) return name;
  return null;
}

/** Yaw proxy from 68 landmarks: nose-tip offset from the eye midpoint, in inter-ocular units. */
export function headYaw(lm: [number, number][]): number | null {
  if (lm.length < 68) return null;
  const mean = (from: number, to: number) => {
    let x = 0;
    let y = 0;
    for (let i = from; i <= to; i++) {
      x += lm[i][0];
      y += lm[i][1];
    }
    const n = to - from + 1;
    return [x / n, y / n];
  };
  const right = mean(36, 41);
  const left = mean(42, 47);
  const d = Math.hypot(left[0] - right[0], left[1] - right[1]);
  if (d < 1) return null;
  return (lm[30][0] - (right[0] + left[0]) / 2) / d;
}

/** Subject's perspective, true (unmirrored) camera geometry. */
export function headLabel(yaw: number | null): string {
  if (yaw == null) return "-";
  if (Math.abs(yaw) < 0.18) return "FACING CAMERA";
  return yaw > 0 ? "TURNED LEFT" : "TURNED RIGHT";
}

/** Degraded mode (no object detector): approximate a person box from a face box. */
export function personFromFace(b: Box, w: number, h: number): Box {
  const fw = b[2] - b[0];
  const fh = b[3] - b[1];
  return [Math.max(0, b[0] - fw * 0.9), Math.max(0, b[1] - fh * 0.35), Math.min(w, b[2] + fw * 0.9), Math.min(h, b[3] + fh * 2.6)];
}
