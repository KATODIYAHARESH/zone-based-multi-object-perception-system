/** Browser camera helpers + consent-gated, local-only self-enrollment storage. */

export type CameraErrorKind = "insecure" | "unsupported" | "denied" | "notfound" | "busy" | "unknown";

export function inIframe(): boolean {
  try {
    return window.self !== window.top;
  } catch {
    return true;
  }
}

function named(name: string, message: string): Error {
  const err = new Error(message);
  err.name = name;
  return err;
}

export async function openCamera(deviceId?: string): Promise<MediaStream> {
  if (typeof window === "undefined" || !window.isSecureContext) throw named("InsecureContextError", "insecure context");
  if (!navigator.mediaDevices?.getUserMedia) throw named("NotSupportedError", "getUserMedia unavailable");
  const video: MediaTrackConstraints = { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } };
  if (deviceId) video.deviceId = { exact: deviceId };
  else video.facingMode = "user";
  return navigator.mediaDevices.getUserMedia({ video, audio: false });
}

export async function listCameras(): Promise<MediaDeviceInfo[]> {
  if (!navigator.mediaDevices?.enumerateDevices) return [];
  return (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
}

export function stopStream(stream: MediaStream | null | undefined): void {
  stream?.getTracks().forEach((t) => t.stop());
}

export function describeCameraError(err: unknown): { kind: CameraErrorKind; message: string } {
  const name = err && typeof err === "object" && "name" in err ? String((err as { name: unknown }).name) : "";
  switch (name) {
    case "InsecureContextError":
      return { kind: "insecure", message: "Camera access requires a secure context (HTTPS or localhost)." };
    case "NotSupportedError":
      return { kind: "unsupported", message: "This browser does not support camera access (getUserMedia)." };
    case "NotAllowedError":
    case "SecurityError":
    case "PermissionDeniedError":
      return {
        kind: "denied",
        message: inIframe()
          ? "Camera access was blocked. This console is embedded in a frame that may not be allowed to use the camera - open it in its own tab and try again."
          : "Camera permission was denied. Allow camera access for this site (camera icon in the address bar), then try again.",
      };
    case "NotFoundError":
    case "DevicesNotFoundError":
    case "OverconstrainedError":
      return { kind: "notfound", message: "No camera was found. Connect a webcam and try again." };
    case "NotReadableError":
    case "TrackStartError":
    case "AbortError":
      return { kind: "busy", message: "The camera is in use by another application or could not be started." };
    default:
      return { kind: "unknown", message: `The camera could not be started${err instanceof Error && err.message ? `: ${err.message}` : "."}` };
  }
}

/* --------------------------------------------------- local self-enrollment */
export interface SelfEnrollment {
  name: string;
  descriptor: number[];
  samples: number;
  createdAt: number;
  expiresAt: number;
}

const STORAGE_KEY = "zbmops.selfEnrollment.v1";
export const ENROLL_EXPIRY_DAYS = 30;
/** face-api 128-d descriptors: Euclidean distance; 0.6 is the library default, 0.5 is deliberately stricter. */
export const MATCH_THRESHOLD = 0.5;

export function euclidean(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return Math.sqrt(s);
}

export function loadEnrollment(): SelfEnrollment | null {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const e = JSON.parse(raw) as SelfEnrollment;
    if (!e || typeof e.name !== "string" || !Array.isArray(e.descriptor) || e.descriptor.length !== 128 || Date.now() > e.expiresAt) {
      window.localStorage.removeItem(STORAGE_KEY);
      return null;
    }
    return e;
  } catch {
    return null;
  }
}

export function saveEnrollment(e: SelfEnrollment): void {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(e));
}

export function clearEnrollment(): void {
  window.localStorage.removeItem(STORAGE_KEY);
}

/** Average consistent samples (outliers dropped); requires >= 3 agreeing samples. */
export function buildEnrollment(name: string, samples: ArrayLike<number>[]): { enrollment: SelfEnrollment | null; kept: number; error?: string } {
  const mean = (list: ArrayLike<number>[]) => {
    const m = new Array<number>(128).fill(0);
    for (const s of list) for (let i = 0; i < 128; i++) m[i] += s[i] / list.length;
    return m;
  };
  if (samples.length < 3) return { enrollment: null, kept: 0, error: "Not enough face samples." };
  const centre = mean(samples);
  const kept = samples.filter((s) => euclidean(s, centre) < 0.45);
  if (kept.length < 3) return { enrollment: null, kept: kept.length, error: "Samples were inconsistent (different people or poor lighting) - enrollment aborted." };
  const now = Date.now();
  return {
    enrollment: { name: name.trim().slice(0, 40), descriptor: mean(kept).map((v) => Math.round(v * 1e6) / 1e6), samples: kept.length, createdAt: now, expiresAt: now + ENROLL_EXPIRY_DAYS * 86_400_000 },
    kept: kept.length,
  };
}
