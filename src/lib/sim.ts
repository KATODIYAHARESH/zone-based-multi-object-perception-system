/**
 * Browser-side pipeline simulator.
 *
 * Generates SYNTHETIC tracks (clearly labelled as such). Zone semantics and the
 * default alert rules live in the shared ZoneRuntime (also used by the live
 * "My Camera" analyser), so simulator and camera can never drift apart.
 * Real perception happens in perception_system/ and on the /camera page.
 */
import { pointInPolygon } from "./geometry";
import { ZoneRuntime, type RuntimeTrack } from "./zoneRuntime";
import {
  CATEGORIES,
  EMOTIONS,
  type Category,
  type Emotion,
  type IngestEvent,
  type Point,
  type SettingsData,
  type Severity,
  type ZoneDTO,
} from "./types";

export const SIM_W = 1280;
export const SIM_H = 720;
export const HORIZON = 0.4;
const MAX_AGENTS = 26;

export interface Spec {
  label: string;
  category: Category;
  w: number;
  h: number;
  speed: number;
  weight: number;
  static?: boolean;
}

const SPECS: Spec[] = [
  { label: "person", category: "human", w: 44, h: 118, speed: 52, weight: 40 },
  { label: "car", category: "vehicle", w: 190, h: 104, speed: 140, weight: 7 },
  { label: "bike", category: "vehicle", w: 84, h: 92, speed: 120, weight: 5 },
  { label: "cycle", category: "vehicle", w: 72, h: 86, speed: 75, weight: 4 },
  { label: "auto-rickshaw", category: "vehicle", w: 120, h: 112, speed: 95, weight: 3 },
  { label: "bus", category: "vehicle", w: 300, h: 165, speed: 100, weight: 1.2 },
  { label: "truck", category: "vehicle", w: 260, h: 150, speed: 100, weight: 1.2 },
  { label: "dog", category: "animal", w: 70, h: 50, speed: 70, weight: 6 },
  { label: "cow", category: "animal", w: 150, h: 100, speed: 24, weight: 4 },
  { label: "cat", category: "animal", w: 48, h: 36, speed: 48, weight: 2.5 },
  { label: "goat", category: "animal", w: 72, h: 60, speed: 30, weight: 2 },
  { label: "horse", category: "animal", w: 150, h: 130, speed: 60, weight: 1.2 },
  { label: "bird", category: "animal", w: 30, h: 22, speed: 90, weight: 2.5 },
  { label: "apple", category: "food", w: 24, h: 24, speed: 0, weight: 1.5, static: true },
  { label: "banana", category: "food", w: 34, h: 18, speed: 0, weight: 1.2, static: true },
  { label: "mango", category: "food", w: 26, h: 24, speed: 0, weight: 1, static: true },
  { label: "backpack", category: "object", w: 40, h: 50, speed: 0, weight: 1.8, static: true },
  { label: "suitcase", category: "object", w: 46, h: 62, speed: 0, weight: 1.2, static: true },
  { label: "unclassified", category: "unclassified", w: 56, h: 44, speed: 0, weight: 1.6, static: true },
];

const AGE_GROUPS = ["child", "teen", "20-35", "36-55", "56+"];
const AGE_WEIGHTS = [0.08, 0.1, 0.42, 0.28, 0.12];
const EMOTION_WEIGHTS = [0.3, 0.08, 0.1, 0.36, 0.08, 0.04, 0.04]; // matches EMOTIONS order

export interface Agent {
  id: number;
  spec: Spec;
  x: number;
  y: number;
  tx: number;
  ty: number;
  born: number;
  ttl: number;
  idleUntil: number;
  conf: number;
  leaving: boolean;
  dead: boolean;
  phase: number;
  emotion: Emotion;
  emotionConf: number;
  emotionSince: number;
  nextEmotionAt: number;
  ageGroup: string;
}

export interface SimAlert {
  ts: number;
  severity: Severity;
  message: string;
}

export interface SimStats {
  fps: number;
  latencyMs: number;
  tracks: number;
  counts: Record<string, Record<string, number>>;
  emotions: Record<string, number>;
  violated: string[];
}

const rand = (a: number, b: number) => a + Math.random() * (b - a);

function pick<T>(items: readonly T[], weights: readonly number[]): T {
  const total = weights.reduce((s, w) => s + w, 0);
  let r = Math.random() * total;
  for (let i = 0; i < items.length; i++) {
    r -= weights[i];
    if (r <= 0) return items[i];
  }
  return items[items.length - 1];
}

export function displayName(a: Agent): string {
  if (a.spec.category === "human") return `Person #${a.id}`;
  if (a.spec.category === "unclassified") return `UNCLASSIFIED #${a.id}`;
  return `${a.spec.label} #${a.id}`;
}

export class SimWorld {
  agents: Agent[] = [];
  recentAlerts: SimAlert[] = [];
  t = 0;
  private nextId = 1;
  private spawnAcc = 0;
  private runtime: ZoneRuntime;

  constructor(
    public zones: ZoneDTO[],
    public settings: SettingsData,
  ) {
    this.runtime = new ZoneRuntime(zones, settings, SIM_W, SIM_H, "bottom_center", { simulated: true });
    for (let i = 0; i < 8; i++) this.spawn(true);
  }

  get memberships(): Map<number, Set<string>> {
    return this.runtime.memberships;
  }

  get violated(): Set<string> {
    return this.runtime.violated;
  }

  scale(y: number): number {
    return 0.4 + 0.8 * Math.max(0, (y / SIM_H - HORIZON) / (1 - HORIZON));
  }

  bbox(a: Agent): [number, number, number, number] {
    const s = this.scale(a.y);
    const w = a.spec.w * s;
    const h = a.spec.h * s;
    const jx = Math.sin(a.phase * 7.1) * 1.2;
    const jy = Math.cos(a.phase * 5.3) * 1.2;
    return [a.x - w / 2 + jx, a.y - h + jy, a.x + w / 2 + jx, a.y + jy];
  }

  private activeZones(): ZoneDTO[] {
    return this.zones.filter((z) => z.enabled && z.polygon.length >= 3);
  }

  private pointInZone(z: ZoneDTO): Point | null {
    const xs = z.polygon.map((p) => p[0]);
    const ys = z.polygon.map((p) => p[1]);
    for (let i = 0; i < 30; i++) {
      const p: Point = [rand(Math.min(...xs), Math.max(...xs)), rand(Math.max(HORIZON + 0.06, Math.min(...ys)), Math.max(...ys))];
      if (pointInPolygon(p, z.polygon)) return [p[0] * SIM_W, p[1] * SIM_H];
    }
    return null;
  }

  private randomGround(): Point {
    const zones = this.activeZones();
    if (zones.length && Math.random() < 0.65) {
      const p = this.pointInZone(zones[Math.floor(Math.random() * zones.length)]);
      if (p) return p;
    }
    return [rand(0.06, 0.94) * SIM_W, rand(HORIZON + 0.08, 0.97) * SIM_H];
  }

  spawn(initial = false): void {
    const spec = pick(SPECS, SPECS.map((s) => s.weight));
    let x: number;
    let y: number;
    let tx: number;
    let ty: number;
    if (spec.static) {
      [x, y] = this.randomGround();
      [tx, ty] = [x, y];
    } else if (initial) {
      [x, y] = this.randomGround();
      [tx, ty] = this.randomGround();
    } else {
      const fromLeft = Math.random() < 0.5;
      x = fromLeft ? -60 : SIM_W + 60;
      y = rand(HORIZON + 0.1, 0.97) * SIM_H;
      [tx, ty] = this.randomGround();
    }
    this.agents.push({
      id: this.nextId++,
      spec,
      x,
      y,
      tx,
      ty,
      born: this.t,
      ttl: spec.static ? rand(30, 80) : rand(15, 40),
      idleUntil: 0,
      conf: rand(0.55, 0.97),
      leaving: false,
      dead: false,
      phase: Math.random() * 10,
      emotion: "neutral",
      emotionConf: rand(0.5, 0.8),
      emotionSince: this.t,
      nextEmotionAt: this.t + rand(1, 5),
      ageGroup: pick(AGE_GROUPS, AGE_WEIGHTS),
    });
  }

  /** Advance the world; returns events to ingest (entry / exit / alert). */
  step(dt: number, spawnRate: number): IngestEvent[] {
    this.t += dt;
    const out: IngestEvent[] = [];
    this.spawnAcc += dt * spawnRate;
    while (this.spawnAcc >= 1) {
      this.spawnAcc -= 1;
      if (this.agents.length < MAX_AGENTS) this.spawn();
    }
    for (const a of this.agents) {
      a.phase += dt;
      if (a.spec.static) {
        if (this.t - a.born > a.ttl) a.dead = true;
      } else if (this.t >= a.idleUntil) {
        const dx = a.tx - a.x;
        const dy = a.ty - a.y;
        const dist = Math.hypot(dx, dy);
        if (dist < 6) {
          if (a.leaving) a.dead = true;
          else if (this.t - a.born > a.ttl) {
            a.leaving = true;
            a.tx = a.x < SIM_W / 2 ? -120 : SIM_W + 120;
            a.ty = a.y;
          } else {
            if (a.spec.category !== "vehicle" && Math.random() < 0.45) a.idleUntil = this.t + rand(2, 7);
            [a.tx, a.ty] = this.randomGround();
          }
        } else {
          const v = a.spec.speed * this.scale(a.y) * dt;
          a.x += (dx / dist) * Math.min(v, dist);
          a.y += (dy / dist) * Math.min(v, dist);
        }
      }
      if (a.spec.category === "human" && this.t >= a.nextEmotionAt) {
        const next = pick(EMOTIONS, EMOTION_WEIGHTS);
        if (next !== a.emotion) {
          a.emotion = next;
          a.emotionSince = this.t;
        }
        a.emotionConf = rand(0.5, 0.95);
        a.nextEmotionAt = this.t + rand(2, 7);
      }
    }
    this.agents = this.agents.filter((a) => !a.dead);
    this.runtime.zones = this.zones;
    this.runtime.settings = this.settings;
    const res = this.runtime.update(this.t, this.agents.map((a) => this.toTrack(a)));
    for (const al of res.alerts) this.recentAlerts.unshift({ ts: al.ts, severity: al.severity, message: al.message });
    if (this.recentAlerts.length > 30) this.recentAlerts = this.recentAlerts.slice(0, 30);
    return res.events;
  }

  private toTrack(a: Agent): RuntimeTrack {
    const human = a.spec.category === "human";
    return {
      id: a.id,
      category: a.spec.category,
      label: a.spec.label,
      bbox: this.bbox(a),
      emotion: human ? a.emotion : null,
      emotionConf: a.emotionConf,
      emotionSince: a.emotionSince,
      ageGroup: human ? a.ageGroup : null,
    };
  }

  stats(fps: number, latencyMs: number): SimStats {
    const counts: Record<string, Record<string, number>> = {};
    for (const z of this.activeZones()) counts[z.slug] = Object.fromEntries(CATEGORIES.map((c) => [c, 0]));
    const emotions: Record<string, number> = Object.fromEntries(EMOTIONS.map((e) => [e, 0]));
    for (const a of this.agents) {
      for (const slug of this.memberships.get(a.id) ?? []) if (counts[slug]) counts[slug][a.spec.category] += 1;
      if (a.spec.category === "human") emotions[a.emotion] += 1;
    }
    return { fps, latencyMs, tracks: this.agents.length, counts, emotions, violated: Array.from(this.violated) };
  }
}
