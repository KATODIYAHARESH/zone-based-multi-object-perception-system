import { eq } from "drizzle-orm";
import { db } from "@/db";
import { settings } from "@/db/schema";
import { ensureSchema, getSettings, serverError, validateSettings } from "@/lib/server";

export const dynamic = "force-dynamic";

export async function GET(req: Request) {
  try {
    const s = await getSettings();
    if (new URL(req.url).searchParams.get("format") === "engine") {
      // Overlay in perception_system/config.yaml terms (merge manually or via automation).
      return Response.json({
        detection: { confidence: s.detectionConfidence, iou: s.nmsIou },
        privacy: { blur_faces: s.blurFaces, retention_days: s.retentionDays },
        alerts: { default_cooldown_s: s.alertCooldownS },
        rules_hint: {
          crowd_limit: { threshold: s.crowdThreshold },
          angry_expression: { min_confidence: s.angryMinConfidence, min_duration_s: s.angryMinDurationS },
        },
      });
    }
    return Response.json({ settings: s, ingestTokenConfigured: Boolean(process.env.INGEST_TOKEN) });
  } catch (err) {
    return serverError(err);
  }
}

export async function PUT(req: Request) {
  try {
    await ensureSchema();
    const parsed = validateSettings(await req.json().catch(() => null), await getSettings());
    if (!parsed.ok) return Response.json({ error: parsed.error }, { status: 400 });
    await db.update(settings).set({ data: { ...parsed.value }, updatedAt: new Date() }).where(eq(settings.id, 1));
    return Response.json({ settings: parsed.value });
  } catch (err) {
    return serverError(err);
  }
}
