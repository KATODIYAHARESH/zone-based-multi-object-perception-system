import { HORIZON, SIM_H, SIM_W, displayName, type Agent, type SimWorld } from "./sim";
import { CATEGORY_COLOR, PALETTE, SEVERITY_COLOR, ZONE_COLOR_HEX, hexToRgba, type ZoneDTO } from "./types";

const FONT = "12px ui-monospace, Menlo, Consolas, monospace";
const FONT_SM = "10px ui-monospace, Menlo, Consolas, monospace";

function label(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, color: string, font = FONT) {
  ctx.font = font;
  const w = ctx.measureText(text).width;
  ctx.fillStyle = "rgba(10, 11, 10, 0.85)";
  ctx.fillRect(x - 3, y - 12, w + 6, 16);
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
}

function background(ctx: CanvasRenderingContext2D) {
  const horizon = SIM_H * HORIZON;
  ctx.fillStyle = "#070806";
  ctx.fillRect(0, 0, SIM_W, horizon);
  ctx.fillStyle = "#11120f";
  for (let i = 0; i < 18; i++) {
    const w = 40 + ((i * 53) % 70);
    const h = 30 + ((i * 37) % 90);
    ctx.fillRect(i * 74 - 10, horizon - h, w, h);
  }
  const ground = ctx.createLinearGradient(0, horizon, 0, SIM_H);
  ground.addColorStop(0, "#0c0d0b");
  ground.addColorStop(1, "#16170f");
  ctx.fillStyle = ground;
  ctx.fillRect(0, horizon, SIM_W, SIM_H - horizon);
  ctx.strokeStyle = "rgba(255, 176, 0, 0.07)";
  ctx.lineWidth = 1;
  const vx = SIM_W / 2;
  for (let x = -SIM_W; x <= SIM_W * 2; x += 110) {
    ctx.beginPath();
    ctx.moveTo(vx, horizon);
    ctx.lineTo(x, SIM_H);
    ctx.stroke();
  }
  for (let i = 1; i < 12; i++) {
    const y = horizon + (SIM_H - horizon) * Math.pow(i / 12, 1.8);
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(SIM_W, y);
    ctx.stroke();
  }
  ctx.strokeStyle = "rgba(255, 176, 0, 0.25)";
  ctx.beginPath();
  ctx.moveTo(0, horizon);
  ctx.lineTo(SIM_W, horizon);
  ctx.stroke();
}

function zonesLayer(ctx: CanvasRenderingContext2D, world: SimWorld, counts: Record<string, Record<string, number>>) {
  const pulse = 0.5 + 0.5 * Math.sin(Date.now() / 180);
  for (const z of world.zones) {
    if (!z.enabled || z.polygon.length < 3) continue;
    const violated = world.violated.has(z.slug);
    const color = violated ? PALETTE.red : ZONE_COLOR_HEX[z.color];
    ctx.beginPath();
    z.polygon.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x * SIM_W, y * SIM_H) : ctx.lineTo(x * SIM_W, y * SIM_H)));
    ctx.closePath();
    ctx.fillStyle = hexToRgba(color, violated ? 0.1 + 0.08 * pulse : 0.07);
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = color;
    ctx.stroke();
    const top = z.polygon.reduce((best, p) => (p[1] < best[1] ? p : best), z.polygon[0]);
    const n = Object.values(counts[z.slug] ?? {}).reduce((s, v) => s + v, 0);
    label(ctx, `${z.name.toUpperCase()} [${z.mode.toUpperCase()}] ${n}`, top[0] * SIM_W + 4, Math.max(14, top[1] * SIM_H - 8), color);
  }
}

function silhouette(ctx: CanvasRenderingContext2D, a: Agent, box: [number, number, number, number]) {
  const [x1, y1, x2, y2] = box;
  const w = x2 - x1;
  const h = y2 - y1;
  const cx = (x1 + x2) / 2;
  ctx.fillStyle = "#2a2923";
  ctx.strokeStyle = "#3a382f";
  switch (a.spec.category) {
    case "human": {
      const r = Math.min(w * 0.32, h * 0.11);
      ctx.beginPath();
      ctx.arc(cx, y1 + r + 1, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillRect(cx - w * 0.3, y1 + r * 2 + 2, w * 0.6, h * 0.42);
      ctx.fillRect(cx - w * 0.26, y1 + r * 2 + h * 0.42, w * 0.2, h - r * 2 - h * 0.42 - 2);
      ctx.fillRect(cx + w * 0.06, y1 + r * 2 + h * 0.42, w * 0.2, h - r * 2 - h * 0.42 - 2);
      break;
    }
    case "vehicle": {
      ctx.fillRect(x1 + w * 0.04, y1 + h * 0.3, w * 0.92, h * 0.5);
      ctx.fillRect(x1 + w * 0.2, y1 + h * 0.05, w * 0.6, h * 0.3);
      ctx.fillStyle = "#151612";
      for (const fx of [0.22, 0.78]) {
        ctx.beginPath();
        ctx.arc(x1 + w * fx, y2 - h * 0.14, h * 0.14, 0, Math.PI * 2);
        ctx.fill();
      }
      break;
    }
    case "animal": {
      ctx.beginPath();
      ctx.ellipse(cx, y1 + h * 0.45, w * 0.38, h * 0.25, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(x1 + w * 0.86, y1 + h * 0.28, h * 0.16, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillRect(x1 + w * 0.22, y1 + h * 0.6, w * 0.06, h * 0.4);
      ctx.fillRect(x1 + w * 0.68, y1 + h * 0.6, w * 0.06, h * 0.4);
      break;
    }
    case "food": {
      ctx.fillStyle = "#3b4a26";
      ctx.beginPath();
      ctx.ellipse(cx, y1 + h / 2, w / 2, h / 2, 0, 0, Math.PI * 2);
      ctx.fill();
      break;
    }
    case "object":
      ctx.fillRect(x1 + w * 0.1, y1 + h * 0.12, w * 0.8, h * 0.88);
      break;
    default: {
      ctx.beginPath();
      ctx.moveTo(x1 + w * 0.1, y2);
      ctx.lineTo(x1 + w * 0.05, y1 + h * 0.4);
      ctx.lineTo(x1 + w * 0.45, y1);
      ctx.lineTo(x2 - w * 0.1, y1 + h * 0.3);
      ctx.lineTo(x2, y2);
      ctx.closePath();
      ctx.fill();
    }
  }
}

function brackets(ctx: CanvasRenderingContext2D, box: [number, number, number, number], color: string) {
  const [x1, y1, x2, y2] = box;
  const lx = Math.max(6, (x2 - x1) * 0.22);
  const ly = Math.max(6, (y2 - y1) * 0.22);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
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
  ctx.strokeStyle = hexToRgba(color, 0.35);
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
}

export function drawScene(
  ctx: CanvasRenderingContext2D,
  world: SimWorld,
  opts: { blurFaces: boolean; fps: number; latencyMs: number; counts: Record<string, Record<string, number>> },
) {
  background(ctx);
  zonesLayer(ctx, world, opts.counts);
  const agents = [...world.agents].sort((a, b) => a.y - b.y);
  for (const a of agents) {
    const box = world.bbox(a);
    const [x1, y1, x2, y2] = box;
    const color = CATEGORY_COLOR[a.spec.category];
    silhouette(ctx, a, box);
    if (opts.blurFaces && a.spec.category === "human") {
      const w = x2 - x1;
      const s = Math.max(3, w / 4);
      for (let i = 0; i < 4; i++)
        for (let j = 0; j < 3; j++) {
          ctx.fillStyle = (i + j + a.id) % 2 ? "#4a483f" : "#5c594e";
          ctx.fillRect(x1 + w * 0.15 + i * s * 0.75, y1 + j * s * 0.75, s * 0.75, s * 0.75);
        }
    }
    brackets(ctx, box, color);
    const inZone = (world.memberships.get(a.id)?.size ?? 0) > 0;
    ctx.fillStyle = inZone ? PALETTE.amber : PALETTE.grey;
    ctx.beginPath();
    ctx.arc((x1 + x2) / 2, y2, 3.5, 0, Math.PI * 2);
    ctx.fill();
    if (a.spec.category === "vehicle" && y2 - y1 > 60) {
      const pw = (x2 - x1) * 0.26;
      ctx.setLineDash([4, 3]);
      ctx.strokeStyle = PALETTE.amber;
      ctx.strokeRect((x1 + x2) / 2 - pw / 2, y2 - (y2 - y1) * 0.3, pw, (y2 - y1) * 0.1);
      ctx.setLineDash([]);
    }
    const lines = [`${displayName(a)}  ${a.conf.toFixed(2)}`];
    if (a.spec.category === "human") lines.push(`${a.emotion} ${Math.round(a.emotionConf * 100)}% | ${a.ageGroup}`);
    let ly = y1 - 6;
    for (const text of lines.reverse()) {
      label(ctx, text, x1 + 2, Math.max(14, ly), color, FONT_SM);
      ly -= 16;
    }
  }
  // HUD
  const fpsColor = opts.fps >= 25 ? PALETTE.green : opts.fps >= 15 ? PALETTE.amber : PALETTE.red;
  label(ctx, "ZBMOPS-SIM // SYNTHETIC FEED 1280x720", 12, 22, PALETTE.amber);
  label(ctx, `FPS ${opts.fps.toFixed(1)}`, 12, 44, fpsColor);
  label(ctx, `LAT ${opts.latencyMs.toFixed(1)} ms   TRACKS ${world.agents.length}`, 100, 44, PALETTE.white);
  ctx.font = "bold 13px ui-monospace, Menlo, monospace";
  const text = "MONITORING ACTIVE";
  const tw = ctx.measureText(text).width;
  const bx = SIM_W - tw - 46;
  ctx.fillStyle = "rgba(10, 11, 10, 0.9)";
  ctx.fillRect(bx, 8, tw + 38, 26);
  ctx.strokeStyle = PALETTE.red;
  ctx.strokeRect(bx, 8, tw + 38, 26);
  ctx.fillStyle = Math.floor(Date.now() / 500) % 2 === 0 ? PALETTE.red : PALETTE.redDim;
  ctx.beginPath();
  ctx.arc(bx + 14, 21, 5.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = PALETTE.white;
  ctx.fillText(text, bx + 26, 26);
  let ay = SIM_H - 14;
  for (const a of world.recentAlerts.slice(0, 4)) {
    const time = new Date(a.ts).toISOString().slice(11, 19);
    label(ctx, `${time} ${a.severity.toUpperCase().padEnd(8)} ${a.message.slice(0, 86)}`, 12, ay, SEVERITY_COLOR[a.severity], FONT_SM);
    ay -= 18;
  }
  ctx.font = FONT_SM;
  const wm = "SIMULATED DATA - PIPELINE VERIFICATION ONLY";
  ctx.fillStyle = "rgba(255, 176, 0, 0.45)";
  ctx.fillText(wm, SIM_W - ctx.measureText(wm).width - 12, SIM_H - 12);
}

export type { ZoneDTO };
