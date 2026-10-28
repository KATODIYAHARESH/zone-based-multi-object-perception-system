import { boolean, index, integer, jsonb, pgTable, real, serial, text, timestamp } from "drizzle-orm/pg-core";

/** Polygon zones (normalised 0..1 coordinates) shared with the Python engine (zones.yaml). */
export const zones = pgTable("zones", {
  id: serial("id").primaryKey(),
  slug: text("slug").notNull().unique(),
  name: text("name").notNull(),
  mode: text("mode").notNull().default("restricted"),
  categories: jsonb("categories").$type<string[]>().notNull().default([]),
  polygon: jsonb("polygon").$type<[number, number][]>().notNull(),
  color: text("color").notNull().default("amber"),
  maxOccupancy: integer("max_occupancy"),
  dwellLimitS: integer("dwell_limit_s"),
  enabled: boolean("enabled").notNull().default(true),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  updatedAt: timestamp("updated_at", { withTimezone: true }).notNull().defaultNow(),
});

/** Zone entry/exit events and alerts ingested from the engine (or the browser simulator). */
export const events = pgTable(
  "events",
  {
    id: serial("id").primaryKey(),
    ts: timestamp("ts", { withTimezone: true }).notNull().defaultNow(),
    source: text("source").notNull().default("engine"),
    type: text("type").notNull(),
    severity: text("severity"),
    ruleId: text("rule_id"),
    zoneId: text("zone_id"),
    zoneName: text("zone_name"),
    trackId: integer("track_id"),
    category: text("category"),
    label: text("label"),
    emotion: text("emotion"),
    ageGroup: text("age_group"),
    identity: text("identity"),
    message: text("message"),
    dwellS: real("dwell_s"),
    x: real("x"),
    y: real("y"),
    meta: jsonb("meta").$type<Record<string, unknown>>(),
  },
  (t) => [index("events_ts_idx").on(t.ts), index("events_type_idx").on(t.type)],
);

/** Latest heartbeat per engine instance (FPS, latency, live counts). */
export const engineStatus = pgTable("engine_status", {
  source: text("source").primaryKey(),
  ts: timestamp("ts", { withTimezone: true }).notNull().defaultNow(),
  fps: real("fps"),
  latencyMs: real("latency_ms"),
  device: text("device"),
  frameSkip: integer("frame_skip"),
  tracks: integer("tracks"),
  counts: jsonb("counts").$type<Record<string, Record<string, number>>>(),
  emotions: jsonb("emotions").$type<Record<string, number>>(),
});

/** Singleton settings row (id = 1). */
export const settings = pgTable("settings", {
  id: integer("id").primaryKey(),
  data: jsonb("data").$type<Record<string, unknown>>().notNull(),
  updatedAt: timestamp("updated_at", { withTimezone: true }).notNull().defaultNow(),
});
