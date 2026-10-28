import { and, desc, eq, gte, type SQL } from "drizzle-orm";
import { db } from "@/db";
import { events } from "@/db/schema";
import { checkIngestAuth, ensureSchema, getSettings, parseIngestEvent, purgeExpired, serverError } from "@/lib/server";

export const dynamic = "force-dynamic";

const MAX_BATCH = 500;

export async function GET(req: Request) {
  try {
    await ensureSchema();
    const q = new URL(req.url).searchParams;
    const limit = Math.min(500, Math.max(1, Number(q.get("limit")) || 50));
    const where: SQL[] = [];
    const type = q.get("type");
    if (type) where.push(eq(events.type, type));
    const source = q.get("source");
    if (source) where.push(eq(events.source, source));
    const zone = q.get("zone");
    if (zone) where.push(eq(events.zoneId, zone));
    const since = q.get("since");
    if (since && !Number.isNaN(new Date(since).getTime())) where.push(gte(events.ts, new Date(since)));
    const rows = await db
      .select()
      .from(events)
      .where(where.length ? and(...where) : undefined)
      .orderBy(desc(events.ts), desc(events.id))
      .limit(limit);
    return Response.json({ events: rows.map((r) => ({ ...r, ts: r.ts.toISOString() })) });
  } catch (err) {
    return serverError(err);
  }
}

/** Ingest endpoint for the Python engine webhook and the browser simulator. */
export async function POST(req: Request) {
  try {
    const body = (await req.json().catch(() => null)) as Record<string, unknown> | null;
    if (!body || typeof body !== "object") return Response.json({ error: "body must be JSON" }, { status: 400 });
    const source = typeof body.source === "string" && /^[a-z0-9_-]{1,32}$/i.test(body.source) ? body.source : "engine";
    const settings = await getSettings();
    const authError = checkIngestAuth(req, source, settings.allowSimulatorIngest);
    if (authError) return Response.json({ error: authError }, { status: 401 });
    const list = Array.isArray(body.events) ? body.events : [body];
    if (list.length > MAX_BATCH) return Response.json({ error: `max ${MAX_BATCH} events per request` }, { status: 413 });
    const rows = list.map((e) => parseIngestEvent(e, source)).filter((r): r is NonNullable<typeof r> => r !== null);
    if (rows.length) await db.insert(events).values(rows);
    const purge = await purgeExpired();
    return Response.json({ accepted: rows.length, rejected: list.length - rows.length, purged: purge?.deleted ?? 0 });
  } catch (err) {
    return serverError(err);
  }
}
