/** Canvas renderer for the live camera analyser (amber / green / red / white - no blue). */
import { anchorPoint, type AnchorMode, type Box, type RuntimeAlert } from "./zoneRuntime";
import { CATEGORY_COLOR, EMOTION_COLOR, PALETTE, SEVERITY_COLOR, ZONE_COLOR_HEX, hexToRgba, type Category, type Emotion, type ZoneDTO } from "./types";

export interface DrawFace {
  box: Box;
  landmarks: [number, number][];
  emotion: Emotion | null;
  emotionConf: number;
  ageGroup: string | null;
  identity: string | null;
}

export interface DrawTrack {
  id: number;
  bbox: Box;
  score: number;
  label: string;
  category: Category;
  coasting: boolean;
  face: DrawFace | null;
  /** sticky local identity (self-enrollment); survives brief face loss */
  identity: string | null;
  inZone: boolean;
}

export interface CameraDrawState {
  zones: ZoneDTO[];
  violated: Set<string>;
  counts: Record<string, Record<string, number>>;
  tracks: DrawTrack[];
  looseFaces: DrawFace[];
  /** regions to pixelate (already privacy-expanded by the caller) */
  blurBoxes: Box[];
  alerts: RuntimeAlert[];
  hud: { fps: number; latencyMs: number; faceMs: number; objects: string; faces: string };
  mirror: boolean;
  blur: boolean;
  landmarks: boolean;
  anchorMode: AnchorMode;
  paused: boolean;
}

let scratch: HTMLCanvasElement | null = null;

/** Irreversible pixelation (~10 blocks across the face), mirroring privacy/guard.py. */
function pixelate(ctx: CanvasRenderingContext2D, b: Box): void {
  const W = ctx.canvas.width;
  const H = ctx.canvas.height;
  const pw = (b[2] - b[0]) * 0.15;
  const ph = (b[3] - b[1]) * 0.15;
  const x = Math.max(0, Math.floor(b[0] - pw));
  const y = Math.max(0, Math.floor(b[1] - ph));
  const w = Math.min(W, Math.ceil(b[2] + pw)) - x;
  const h = Math.min(H, Math.ceil(b[3] + ph)) - y;
  if (w < 2 || h < 2) return;
  scratch = scratch ?? document.createElement("canvas");
  const block = Math.max(8, w / 10);
  const sw = Math.max(1, Math.round(w / block));
  const sh = Math.max(1, Math.round(h / block));
  scratch.width = sw;
  scratch.height = sh;
  const s = scratch.getContext("2d");
  if (!s) return;
  s.drawImage(ctx.canvas, x, y, w, h, 0, 0, sw, sh);
  ctx.save();
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(scratch, 0, 0, sw, sh, x, y, w, h);
  ctx.restore();
}

/**
 * Fail-closed head estimate for people whose face has not been measured yet.
 * Close-ups (box touching the top edge or taller than 60% of the frame) get a much
 * larger mask, because the face then fills the upper half of the person box.
 */
export function headRegion(b: Box, frameH: number): Box {
  const w = b[2] - b[0];
  const h = b[3] - b[1];
  const closeUp = b[1] <= frameH * 0.01 || h >= frameH * 0.6;
  return [b[0] + 0.12 * w, b[1], b[2] - 0.12 * w, b[1] + (closeUp ? 0.48 : 0.26) * h];
}

function brackets(ctx: CanvasRenderingContext2D, b: Box, color: string, width: number): void {
  const [x1, y1, x2, y2] = b;
  const lx = Math.max(6, (x2 - x1) * 0.2);
  const ly = Math.max(6, (y2 - y1) * 0.2);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  for (const [cx, cy, dx, dy] of [
    [x1, y1, 1, 1],
    [x2, y1, -1, 1],
    [x1, y2, 1, -1],
    [x2, y2, -1, -1],
  ] as const) {
    ctx.moveTo(cx + dx * lx, cy);
    ctx.lineTo(cx, cy);
    ctx.lineTo(cx, cy + dy * ly);
  }
  ctx.stroke();
  ctx.lineWidth = 1;
  ctx.strokeStyle = hexToRgba(color, 0.3);
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
}

function trackName(t: DrawTrack): string {
  if (t.category === "human") return t.identity ?? t.face?.identity ?? `Person #${t.id}`;
  if (t.category === "unclassified") return `UNCLASSIFIED #${t.id}`;
  return `${t.label} #${t.id}`;
}

export function drawCameraFrame(ctx: CanvasRenderingContext2D, video: HTMLVideoElement, s: CameraDrawState): void {
  const W = ctx.canvas.width;
  const H = ctx.canvas.height;
  const k = Math.max(0.6, Math.min(1.6, W / 1280));
  const font = (px: number, bold = false) => `${bold ? "bold " : ""}${Math.round(px * k)}px ui-monospace, Menlo, Consolas, monospace`;
  const mx = (x: number) => (s.mirror ? W - x : x);
  const mbox = (b: Box): Box => (s.mirror ? [W - b[2], b[1], W - b[0], b[3]] : b);
  const tag = (text: string, x: number, y: number, color: string, px = 12) => {
    ctx.font = font(px);
    const w = ctx.measureText(text).width;
    const h = px * k + 6;
    const xx = Math.min(Math.max(3, x), Math.max(3, W - w - 6));
    ctx.fillStyle = "rgba(10, 11, 10, 0.82)";
    ctx.fillRect(xx - 3, y - h + 4, w + 6, h);
    ctx.fillStyle = color;
    ctx.fillText(text, xx, y);
  };

  // 1. frame (mirroring is display-only; analysis always uses true camera geometry)
  ctx.save();
  if (s.mirror) {
    ctx.translate(W, 0);
    ctx.scale(-1, 1);
  }
  ctx.drawImage(video, 0, 0, W, H);
  ctx.restore();

  // 2. privacy pixelation before any overlay
  if (s.blur) for (const b of s.blurBoxes) pixelate(ctx, mbox(b));

  // 3. zones
  for (const z of s.zones) {
    if (!z.enabled || z.polygon.length < 3) continue;
    const violated = s.violated.has(z.slug);
    const color = violated ? PALETTE.red : ZONE_COLOR_HEX[z.color];
    ctx.beginPath();
    z.polygon.forEach(([nx, ny], i) => (i === 0 ? ctx.moveTo(mx(nx * W), ny * H) : ctx.lineTo(mx(nx * W), ny * H)));
    ctx.closePath();
    ctx.fillStyle = hexToRgba(color, violated ? 0.16 : 0.07);
    ctx.fill();
    ctx.lineWidth = 2 * k;
    ctx.strokeStyle = color;
    ctx.stroke();
    const top = z.polygon.reduce((b, p) => (p[1] < b[1] ? p : b), z.polygon[0]);
    const n = Object.values(s.counts[z.slug] ?? {}).reduce((a, v) => a + v, 0);
    tag(`${z.name.toUpperCase()} [${z.mode.toUpperCase()}] ${n}`, mx(top[0] * W) + 4, Math.max(18 * k, top[1] * H - 8), color);
  }

  // 4. tracks
  for (const t of s.tracks) {
    const color = CATEGORY_COLOR[t.category];
    const b = mbox(t.bbox);
    brackets(ctx, b, color, (t.coasting ? 1 : 2) * k);
    const lines = [`${trackName(t)}  ${t.score.toFixed(2)}`];
    if (t.face?.emotion) lines.push(`${t.face.emotion} ${Math.round(t.face.emotionConf * 100)}%${t.face.ageGroup ? ` | ${t.face.ageGroup}` : ""}`);
    const step = 18 * k;
    if (b[1] - lines.length * step - 4 > 0) {
      let ly = b[1] - 6 * k;
      for (const line of lines.slice().reverse()) {
        tag(line, b[0] + 2, ly, color, 11);
        ly -= step;
      }
    } else {
      let ly = Math.max(b[1] + step, 96 * k); // stay clear of the HUD band
      for (const line of lines) {
        tag(line, b[0] + 4, ly, color, 11);
        ly += step;
      }
    }
    const [ax, ay] = anchorPoint(t.bbox, W, H, s.anchorMode);
    ctx.fillStyle = t.inZone ? PALETTE.amber : PALETTE.grey;
    ctx.beginPath();
    ctx.arc(mx(ax * W), Math.min(H - 5 * k, ay * H), 5 * k, 0, Math.PI * 2);
    ctx.fill();
  }

  // 5. faces: box, landmarks, expression bar
  const drawFace = (f: DrawFace, withLabel: boolean) => {
    const b = mbox(f.box);
    ctx.lineWidth = 1.5 * k;
    ctx.strokeStyle = hexToRgba(PALETTE.white, 0.85);
    ctx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
    if (s.landmarks && !s.blur) {
      ctx.fillStyle = PALETTE.amber;
      const r = 1.3 * k;
      for (const [lx, ly] of f.landmarks) ctx.fillRect(mx(lx) - r, ly - r, 2 * r, 2 * r);
    }
    if (f.emotion) {
      const bw = b[2] - b[0];
      ctx.fillStyle = "rgba(10, 11, 10, 0.8)";
      ctx.fillRect(b[0], b[3] + 3 * k, bw, 5 * k);
      ctx.fillStyle = EMOTION_COLOR[f.emotion];
      ctx.fillRect(b[0], b[3] + 3 * k, bw * Math.max(0, Math.min(1, f.emotionConf)), 5 * k);
      if (withLabel) tag(`${f.emotion} ${Math.round(f.emotionConf * 100)}%${f.ageGroup ? ` | ${f.ageGroup}` : ""}`, b[0], b[3] + 22 * k, PALETTE.white, 11);
    }
  };
  for (const t of s.tracks) if (t.face) drawFace(t.face, false);
  for (const f of s.looseFaces) drawFace(f, true);

  // 6. HUD
  const fpsColor = s.hud.fps >= 25 ? PALETTE.green : s.hud.fps >= 15 ? PALETTE.amber : PALETTE.red;
  tag("ZBMOPS // MY CAMERA · ON-DEVICE ANALYSIS", 12, 26 * k, PALETTE.amber, 13);
  tag(`FPS ${s.hud.fps.toFixed(1)}`, 12, 50 * k, fpsColor);
  tag(`FRAME ${s.hud.latencyMs.toFixed(1)} ms · FACE ${s.hud.faceMs.toFixed(0)} ms · TRACKS ${s.tracks.length}`, 12 + 105 * k, 50 * k, PALETTE.white);
  tag(`OBJ ${s.hud.objects} · FACE ${s.hud.faces}${s.mirror ? " · MIRRORED VIEW" : ""}`, 12, 72 * k, PALETTE.grey, 11);

  // 7. mandatory monitoring indicator (always drawn - cannot be disabled)
  ctx.font = font(13, true);
  const text = "CAMERA LIVE · MONITORING ACTIVE";
  const tw = ctx.measureText(text).width;
  const bh = 26 * k;
  const bx = W - tw - 46 * k;
  const by = 8 * k;
  ctx.fillStyle = "rgba(10, 11, 10, 0.9)";
  ctx.fillRect(bx, by, tw + 38 * k, bh);
  ctx.strokeStyle = PALETTE.red;
  ctx.lineWidth = 1;
  ctx.strokeRect(bx, by, tw + 38 * k, bh);
  ctx.fillStyle = Math.floor(Date.now() / 500) % 2 === 0 ? PALETTE.red : PALETTE.redDim;
  ctx.beginPath();
  ctx.arc(bx + 14 * k, by + bh / 2, 5.5 * k, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = PALETTE.white;
  ctx.fillText(text, bx + 26 * k, by + bh / 2 + 4.5 * k);

  // 8. alert ticker
  let ay = H - 14 * k;
  for (const a of s.alerts.slice(0, 4)) {
    tag(`${new Date(a.ts).toISOString().slice(11, 19)} ${a.severity.toUpperCase().padEnd(8)} ${a.message.slice(0, 84)}`, 12, ay, SEVERITY_COLOR[a.severity], 11);
    ay -= 20 * k;
  }
  if (s.paused) {
    ctx.font = font(20, true);
    const pw = ctx.measureText("ANALYSIS PAUSED").width;
    tag("ANALYSIS PAUSED", (W - pw) / 2, H / 2, PALETTE.amber, 20);
  }
}
