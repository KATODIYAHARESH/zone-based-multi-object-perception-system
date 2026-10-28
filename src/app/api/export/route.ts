import { and, asc, eq, gte, type SQL } from "drizzle-orm";
import { db } from "@/db";
import { events } from "@/db/schema";
import { ensureSchema, serverError } from "@/lib/server";

export const dynamic = "force-dynamic";

const COLUMNS = ["id", "ts", "source", "type", "severity", "ruleId", "zoneId", "zoneName", "trackId", "category", "label", "emotion", "ageGroup", "identity", "message", "dwellS", "x", "y"] as const;

function csvCell(v: unknown): string {
  if (v === null || v === undefined) return "";
  const s = v instanceof Date ? v.toISOString() : String(v);
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

export async function GET(req: Request) {
  try {
    await ensureSchema();
    const q = new URL(req.url).searchParams;
    const format = q.get("format") === "json" ? "json" : "csv";
    const hours = Math.min(24 * 365, Math.max(1, Number(q.get("hours")) || 24));
    const where: SQL[] = [gte(events.ts, new Date(Date.now() - hours * 3_600_000))];
    const type = q.get("type");
    if (type) where.push(eq(events.type, type));
    const rows = await db.select().from(events).where(and(...where)).orderBy(asc(events.ts)).limit(50_000);
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    if (format === "json") {
      return new Response(JSON.stringify(rows.map((r) => ({ ...r, ts: r.ts.toISOString() })), null, 2), {
        headers: { "Content-Type": "application/json", "Content-Disposition": `attachment; filename="zbmops-events-${stamp}.json"` },
      });
    }
    const lines = [COLUMNS.join(","), ...rows.map((r) => COLUMNS.map((c) => csvCell(r[c])).join(","))];
    return new Response(lines.join("\n") + "\n", {
      headers: { "Content-Type": "text/csv; charset=utf-8", "Content-Disposition": `attachment; filename="zbmops-events-${stamp}.csv"` },
    });
  } catch (err) {
    return serverError(err);
  }
}
