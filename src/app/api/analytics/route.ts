import { sql } from "drizzle-orm";
import { db } from "@/db";
import { ensureSchema, serverError } from "@/lib/server";

export const dynamic = "force-dynamic";

type Row = Record<string, unknown>;
const n = (v: unknown) => (v === null || v === undefined ? 0 : Number(v));

function bucketFor(hours: number): number {
  if (hours <= 1) return 60;
  if (hours <= 6) return 300;
  if (hours <= 24) return 1200;
  if (hours <= 72) return 3600;
  return 7200;
}

export async function GET(req: Request) {
  try {
    await ensureSchema();
    const hours = Math.min(720, Math.max(1, Number(new URL(req.url).searchParams.get("hours")) || 24));
    const bucket = bucketFor(hours);
    const since = new Date(Date.now() - hours * 3_600_000).toISOString();
    const q = async (query: ReturnType<typeof sql>) => (await db.execute(query)).rows as Row[];

    const [totals, timeline, alertTimeline, zoneRows, categories, emotions, heat, rules, unique] = await Promise.all([
      q(sql`SELECT type, coalesce(severity, '') AS severity, count(*)::int AS n FROM events WHERE ts >= ${since}::timestamptz GROUP BY type, severity`),
      q(sql`SELECT (floor(extract(epoch FROM ts) / ${bucket}::int) * ${bucket}::int)::bigint AS b, category, count(*)::int AS n FROM events WHERE ts >= ${since}::timestamptz AND type = 'entry' GROUP BY b, category ORDER BY b`),
      q(sql`SELECT (floor(extract(epoch FROM ts) / ${bucket}::int) * ${bucket}::int)::bigint AS b, severity, count(*)::int AS n FROM events WHERE ts >= ${since}::timestamptz AND type = 'alert' GROUP BY b, severity ORDER BY b`),
      q(sql`SELECT coalesce(zone_name, zone_id, 'unzoned') AS zone, count(*) FILTER (WHERE type = 'entry')::int AS entries, count(*) FILTER (WHERE type = 'exit')::int AS exits, count(*) FILTER (WHERE type = 'alert')::int AS alerts, avg(dwell_s) FILTER (WHERE type = 'exit') AS avg_dwell FROM events WHERE ts >= ${since}::timestamptz GROUP BY 1 ORDER BY entries DESC LIMIT 20`),
      q(sql`SELECT category, count(*)::int AS n FROM events WHERE ts >= ${since}::timestamptz AND type = 'entry' AND category IS NOT NULL GROUP BY category ORDER BY n DESC`),
      q(sql`SELECT emotion, count(*)::int AS n FROM events WHERE ts >= ${since}::timestamptz AND emotion IS NOT NULL GROUP BY emotion ORDER BY n DESC`),
      q(sql`SELECT least(31, greatest(0, floor(x * 32)))::int AS gx, least(17, greatest(0, floor(y * 18)))::int AS gy, count(*)::int AS n FROM events WHERE ts >= ${since}::timestamptz AND x IS NOT NULL AND y IS NOT NULL GROUP BY gx, gy`),
      q(sql`SELECT rule_id, max(severity) AS severity, count(*)::int AS n FROM events WHERE ts >= ${since}::timestamptz AND type = 'alert' AND rule_id IS NOT NULL GROUP BY rule_id ORDER BY n DESC LIMIT 8`),
      q(sql`SELECT count(DISTINCT (source, track_id))::int AS n FROM events WHERE ts >= ${since}::timestamptz AND track_id IS NOT NULL`),
    ]);

    const sum = (pred: (r: Row) => boolean) => totals.filter(pred).reduce((acc, r) => acc + n(r.n), 0);
    return Response.json({
      hours,
      bucketSeconds: bucket,
      since,
      kpis: {
        entries: sum((r) => r.type === "entry"),
        exits: sum((r) => r.type === "exit"),
        alerts: sum((r) => r.type === "alert"),
        critical: sum((r) => r.type === "alert" && r.severity === "critical"),
        warning: sum((r) => r.type === "alert" && r.severity === "warning"),
        info: sum((r) => r.type === "alert" && r.severity === "info"),
        uniqueTracks: n(unique[0]?.n),
      },
      timeline: timeline.map((r) => ({ t: n(r.b) * 1000, category: String(r.category ?? "unclassified"), n: n(r.n) })),
      alertTimeline: alertTimeline.map((r) => ({ t: n(r.b) * 1000, severity: String(r.severity ?? "info"), n: n(r.n) })),
      zones: zoneRows.map((r) => ({
        zone: String(r.zone),
        entries: n(r.entries),
        exits: n(r.exits),
        alerts: n(r.alerts),
        avgDwell: r.avg_dwell === null ? null : Number(r.avg_dwell),
      })),
      categories: categories.map((r) => ({ category: String(r.category), n: n(r.n) })),
      emotions: emotions.map((r) => ({ emotion: String(r.emotion), n: n(r.n) })),
      heatmap: heat.map((r) => ({ gx: n(r.gx), gy: n(r.gy), n: n(r.n) })),
      rules: rules.map((r) => ({ ruleId: String(r.rule_id), severity: String(r.severity ?? "info"), n: n(r.n) })),
    });
  } catch (err) {
    return serverError(err);
  }
}
