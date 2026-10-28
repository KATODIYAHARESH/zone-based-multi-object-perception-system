import { eq, lt, sql } from "drizzle-orm";
import { db } from "@/db";
import { events, settings, zones } from "@/db/schema";
import { slugify, validatePolygon } from "@/lib/geometry";
import {
  CATEGORIES,
  DEFAULT_SETTINGS,
  EVENT_TYPES,
  SEVERITIES,
  ZONE_COLORS,
  ZONE_MODES,
  type Category,
  type Point,
  type SettingsData,
  type ZoneColor,
  type ZoneDTO,
  type ZoneMode,
} from "@/lib/types";

/* ------------------------------------------------------------------ schema */
// Idempotent DDL mirroring src/db/schema.ts so a fresh database self-heals even
// before `drizzle-kit push` has run. Identical to drizzle's generated SQL.
const DDL = [
  `CREATE TABLE IF NOT EXISTS "zones" ("id" serial PRIMARY KEY NOT NULL, "slug" text NOT NULL, "name" text NOT NULL, "mode" text DEFAULT 'restricted' NOT NULL, "categories" jsonb DEFAULT '[]'::jsonb NOT NULL, "polygon" jsonb NOT NULL, "color" text DEFAULT 'amber' NOT NULL, "max_occupancy" integer, "dwell_limit_s" integer, "enabled" boolean DEFAULT true NOT NULL, "created_at" timestamp with time zone DEFAULT now() NOT NULL, "updated_at" timestamp with time zone DEFAULT now() NOT NULL, CONSTRAINT "zones_slug_unique" UNIQUE("slug"))`,
  `CREATE TABLE IF NOT EXISTS "events" ("id" serial PRIMARY KEY NOT NULL, "ts" timestamp with time zone DEFAULT now() NOT NULL, "source" text DEFAULT 'engine' NOT NULL, "type" text NOT NULL, "severity" text, "rule_id" text, "zone_id" text, "zone_name" text, "track_id" integer, "category" text, "label" text, "emotion" text, "age_group" text, "identity" text, "message" text, "dwell_s" real, "x" real, "y" real, "meta" jsonb)`,
  `CREATE INDEX IF NOT EXISTS "events_ts_idx" ON "events" USING btree ("ts")`,
  `CREATE INDEX IF NOT EXISTS "events_type_idx" ON "events" USING btree ("type")`,
  `CREATE TABLE IF NOT EXISTS "engine_status" ("source" text PRIMARY KEY NOT NULL, "ts" timestamp with time zone DEFAULT now() NOT NULL, "fps" real, "latency_ms" real, "device" text, "frame_skip" integer, "tracks" integer, "counts" jsonb, "emotions" jsonb)`,
  `CREATE TABLE IF NOT EXISTS "settings" ("id" integer PRIMARY KEY NOT NULL, "data" jsonb NOT NULL, "updated_at" timestamp with time zone DEFAULT now() NOT NULL)`,
];

const DEFAULT_ZONES: (typeof zones.$inferInsert)[] = [
  { slug: "zone_a", name: "Zone A", mode: "allowed", categories: ["human"], polygon: [[0.04, 0.52], [0.36, 0.52], [0.4, 0.97], [0.02, 0.97]], color: "amber", maxOccupancy: 8, dwellLimitS: 120 },
  { slug: "walkway", name: "Pedestrian Walkway", mode: "restricted", categories: ["vehicle"], polygon: [[0.4, 0.45], [0.58, 0.45], [0.66, 0.98], [0.44, 0.98]], color: "amber" },
  { slug: "farm", name: "Farm Zone", mode: "allowed", categories: ["human", "animal", "vehicle"], polygon: [[0.66, 0.5], [0.97, 0.5], [0.98, 0.97], [0.7, 0.97]], color: "green" },
];

let ready: Promise<void> | null = null;

export function ensureSchema(): Promise<void> {
  if (!ready) {
    ready = (async () => {
      for (const stmt of DDL) await db.execute(sql.raw(stmt));
      // Seed exactly once: the settings row doubles as the "initialised" marker,
      // so deleting every zone later does not resurrect the defaults.
      const inserted = await db
        .insert(settings)
        .values({ id: 1, data: { ...DEFAULT_SETTINGS } })
        .onConflictDoNothing()
        .returning({ id: settings.id });
      if (inserted.length > 0) await db.insert(zones).values(DEFAULT_ZONES).onConflictDoNothing();
    })().catch((err) => {
      ready = null;
      throw err;
    });
  }
  return ready;
}

export function serverError(err: unknown): Response {
  console.error("[api]", err);
  return Response.json({ error: "Internal server error" }, { status: 500 });
}

/* ---------------------------------------------------------------- settings */
export async function getSettings(): Promise<SettingsData> {
  await ensureSchema();
  const rows = await db.select().from(settings).where(eq(settings.id, 1)).limit(1);
  return { ...DEFAULT_SETTINGS, ...((rows[0]?.data ?? {}) as Partial<SettingsData>) };
}

type Result<T> = { ok: true; value: T } | { ok: false; error: string };

const NUMERIC_RULES: Partial<Record<keyof SettingsData, [number, number]>> = {
  retentionDays: [1, 3650],
  crowdThreshold: [0, 500],
  angryMinConfidence: [0, 1],
  angryMinDurationS: [0, 60],
  defaultDwellLimitS: [1, 86400],
  alertCooldownS: [0, 3600],
  detectionConfidence: [0.01, 1],
  nmsIou: [0.05, 0.95],
};

export function validateSettings(input: unknown, base: SettingsData): Result<SettingsData> {
  if (!input || typeof input !== "object") return { ok: false, error: "body must be a JSON object" };
  const next: SettingsData = { ...base };
  const src = input as Record<string, unknown>;
  for (const [key, value] of Object.entries(src)) {
    if (!(key in DEFAULT_SETTINGS)) return { ok: false, error: `unknown setting '${key}'` };
    const k = key as keyof SettingsData;
    const expected = typeof DEFAULT_SETTINGS[k];
    if (expected === "number") {
      const n = Number(value);
      const range = NUMERIC_RULES[k];
      if (!Number.isFinite(n) || (range && (n < range[0] || n > range[1])))
        return { ok: false, error: `${key} must be a number${range ? ` in [${range[0]}, ${range[1]}]` : ""}` };
      (next as unknown as Record<string, unknown>)[k] = n;
    } else if (expected === "boolean") {
      if (typeof value !== "boolean") return { ok: false, error: `${key} must be true or false` };
      (next as unknown as Record<string, unknown>)[k] = value;
    } else {
      if (typeof value !== "string" || value.length > 300) return { ok: false, error: `${key} must be a string (max 300 chars)` };
      if (k.endsWith("Url") && value && !/^https?:\/\//.test(value)) return { ok: false, error: `${key} must start with http:// or https://` };
      (next as unknown as Record<string, unknown>)[k] = value.trim();
    }
  }
  return { ok: true, value: next };
}

/* ------------------------------------------------------------------- zones */
export interface ZoneInput {
  name: string;
  mode: ZoneMode;
  categories: Category[];
  polygon: Point[];
  color: ZoneColor;
  maxOccupancy: number | null;
  dwellLimitS: number | null;
  enabled: boolean;
}

export function parseZoneInput(body: unknown, partial: boolean): Result<Partial<ZoneInput>> {
  if (!body || typeof body !== "object") return { ok: false, error: "body must be a JSON object" };
  const b = body as Record<string, unknown>;
  const out: Partial<ZoneInput> = {};
  try {
    if ("name" in b || !partial) {
      const name = typeof b.name === "string" ? b.name.trim() : "";
      if (!name || name.length > 64) throw new Error("name is required (max 64 characters)");
      out.name = name;
    }
    if ("mode" in b || !partial) {
      const mode = b.mode ?? "restricted";
      if (!(ZONE_MODES as readonly unknown[]).includes(mode)) throw new Error("mode must be 'allowed' or 'restricted'");
      out.mode = mode as ZoneMode;
    }
    if ("categories" in b || !partial) {
      const cats = b.categories ?? [];
      if (!Array.isArray(cats) || cats.some((c) => !(CATEGORIES as readonly unknown[]).includes(c)))
        throw new Error(`categories must be a subset of: ${CATEGORIES.join(", ")}`);
      out.categories = Array.from(new Set(cats as Category[])).sort();
    }
    if ("polygon" in b || !partial) out.polygon = validatePolygon(b.polygon);
    if ("color" in b || !partial) {
      const color = b.color ?? "amber";
      if (!(ZONE_COLORS as readonly unknown[]).includes(color)) throw new Error("color must be amber, green, red or white");
      out.color = color as ZoneColor;
    }
    for (const key of ["maxOccupancy", "dwellLimitS"] as const) {
      if (key in b || !partial) {
        const v = b[key];
        if (v === null || v === undefined || v === "" || v === 0) out[key] = null;
        else {
          const n = Number(v);
          if (!Number.isInteger(n) || n < 0 || n > 1_000_000) throw new Error(`${key} must be a non-negative integer or null`);
          out[key] = n;
        }
      }
    }
    if ("enabled" in b || !partial) out.enabled = b.enabled === undefined ? true : Boolean(b.enabled);
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : "invalid zone" };
  }
  return { ok: true, value: out };
}

export async function uniqueSlug(name: string): Promise<string> {
  const base = slugify(name);
  const taken = new Set((await db.select({ slug: zones.slug }).from(zones)).map((r) => r.slug));
  let slug = base;
  for (let i = 2; taken.has(slug); i++) slug = `${base.slice(0, 60)}_${i}`;
  return slug;
}

export function zoneRowToDTO(row: typeof zones.$inferSelect): ZoneDTO {
  return {
    id: row.id,
    slug: row.slug,
    name: row.name,
    mode: row.mode as ZoneMode,
    categories: (row.categories ?? []) as Category[],
    polygon: row.polygon as Point[],
    color: row.color as ZoneColor,
    maxOccupancy: row.maxOccupancy,
    dwellLimitS: row.dwellLimitS,
    enabled: row.enabled,
  };
}

/** zones.yaml schema used by perception_system/zones/zone.py */
export function zonesToEngine(list: ZoneDTO[]) {
  return list.map((z) => ({
    id: z.slug,
    name: z.name,
    mode: z.mode,
    categories: z.categories,
    polygon: z.polygon,
    max_occupancy: z.maxOccupancy,
    dwell_limit_s: z.dwellLimitS,
    color: z.color,
    enabled: z.enabled,
  }));
}

export function zonesToYaml(list: ZoneDTO[]): string {
  const lines = [
    "# Exported from the ZBMOPS web console - drop into perception_system/zones.yaml",
    `# generated ${new Date().toISOString()}`,
    "zones:",
  ];
  for (const z of list) {
    lines.push(`- id: ${z.slug}`, `  name: ${JSON.stringify(z.name)}`, `  mode: ${z.mode}`, `  categories: [${z.categories.join(", ")}]`);
    lines.push(`  polygon: [${z.polygon.map(([x, y]) => `[${x.toFixed(4)}, ${y.toFixed(4)}]`).join(", ")}]`);
    if (z.maxOccupancy != null) lines.push(`  max_occupancy: ${z.maxOccupancy}`);
    if (z.dwellLimitS != null) lines.push(`  dwell_limit_s: ${z.dwellLimitS}`);
    lines.push(`  color: ${z.color}`, `  enabled: ${z.enabled}`);
  }
  return lines.join("\n") + "\n";
}

/* ------------------------------------------------------------------ ingest */
export function checkIngestAuth(req: Request, source: string, allowSimulator: boolean): string | null {
  const token = process.env.INGEST_TOKEN;
  if (!token) return null;
  if ((source === "simulator" || source === "camera") && allowSimulator) return null; // browser-originated, metadata only
  return req.headers.get("authorization") === `Bearer ${token}` ? null : "missing or invalid ingest token";
}

const str = (v: unknown, max: number): string | null => (typeof v === "string" && v.trim() ? v.trim().slice(0, max) : null);
const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const unit = (v: unknown): number | null => {
  const n = num(v);
  return n === null ? null : Math.min(1, Math.max(0, n));
};

export function parseIngestEvent(raw: unknown, source: string): typeof events.$inferInsert | null {
  if (!raw || typeof raw !== "object") return null;
  const e = raw as Record<string, unknown>;
  if (!(EVENT_TYPES as readonly unknown[]).includes(e.type)) return null;
  let ts = typeof e.ts === "string" ? new Date(e.ts) : new Date();
  if (Number.isNaN(ts.getTime()) || ts.getTime() > Date.now() + 60_000) ts = new Date();
  const severity = (SEVERITIES as readonly unknown[]).includes(e.severity) ? (e.severity as string) : null;
  const trackId = num(e.trackId);
  return {
    ts,
    source,
    type: e.type as string,
    severity: e.type === "alert" ? (severity ?? "info") : severity,
    ruleId: str(e.ruleId, 64),
    zoneId: str(e.zoneId, 64),
    zoneName: str(e.zoneName, 64),
    trackId: trackId === null ? null : Math.trunc(trackId),
    category: str(e.category, 32),
    label: str(e.label, 64),
    emotion: str(e.emotion, 16),
    ageGroup: str(e.ageGroup, 16),
    identity: str(e.identity, 64),
    message: str(e.message, 500),
    dwellS: num(e.dwellS),
    x: unit(e.x),
    y: unit(e.y),
    meta: e.meta && typeof e.meta === "object" && !Array.isArray(e.meta) ? (e.meta as Record<string, unknown>) : null,
  };
}

/* --------------------------------------------------------------- retention */
let lastPurge = 0;

export async function purgeExpired(force = false): Promise<{ deleted: number; cutoff: string } | null> {
  const now = Date.now();
  if (!force && now - lastPurge < 10 * 60 * 1000) return null;
  lastPurge = now;
  const s = await getSettings();
  const cutoff = new Date(now - s.retentionDays * 86_400_000);
  const removed = await db.delete(events).where(lt(events.ts, cutoff)).returning({ id: events.id });
  return { deleted: removed.length, cutoff: cutoff.toISOString() };
}
