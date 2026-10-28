import { desc } from "drizzle-orm";
import { db } from "@/db";
import { engineStatus } from "@/db/schema";
import { checkIngestAuth, ensureSchema, getSettings, serverError } from "@/lib/server";

export const dynamic = "force-dynamic";

const ONLINE_WINDOW_MS = 10_000;
const finite = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : null);
const record = <T,>(v: unknown): T | null => (v && typeof v === "object" && !Array.isArray(v) ? (v as T) : null);

export async function GET() {
  try {
    await ensureSchema();
    const rows = await db.select().from(engineStatus).orderBy(desc(engineStatus.ts));
    const now = Date.now();
    return Response.json({
      engines: rows.map((r) => ({ ...r, ts: r.ts.toISOString(), online: now - r.ts.getTime() < ONLINE_WINDOW_MS })),
    });
  } catch (err) {
    return serverError(err);
  }
}

export async function POST(req: Request) {
  try {
    const b = (await req.json().catch(() => null)) as Record<string, unknown> | null;
    if (!b) return Response.json({ error: "body must be JSON" }, { status: 400 });
    const source = typeof b.source === "string" && /^[a-z0-9_-]{1,32}$/i.test(b.source) ? b.source : "engine";
    const authError = checkIngestAuth(req, source, (await getSettings()).allowSimulatorIngest);
    if (authError) return Response.json({ error: authError }, { status: 401 });
    const values = {
      source,
      ts: new Date(),
      fps: finite(b.fps),
      latencyMs: finite(b.latencyMs),
      device: typeof b.device === "string" ? b.device.slice(0, 64) : null,
      frameSkip: finite(b.frameSkip) === null ? null : Math.trunc(b.frameSkip as number),
      tracks: finite(b.tracks) === null ? null : Math.trunc(b.tracks as number),
      counts: record<Record<string, Record<string, number>>>(b.counts),
      emotions: record<Record<string, number>>(b.emotions),
    };
    await db.insert(engineStatus).values(values).onConflictDoUpdate({ target: engineStatus.source, set: values });
    return Response.json({ ok: true });
  } catch (err) {
    return serverError(err);
  }
}
