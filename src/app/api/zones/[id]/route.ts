import { eq } from "drizzle-orm";
import { db } from "@/db";
import { zones } from "@/db/schema";
import { ensureSchema, parseZoneInput, serverError, zoneRowToDTO } from "@/lib/server";

export const dynamic = "force-dynamic";

function parseId(raw: string): number | null {
  const id = Number(raw);
  return Number.isInteger(id) && id > 0 ? id : null;
}

export async function PUT(req: Request, { params }: { params: Promise<{ id: string }> }) {
  try {
    await ensureSchema();
    const id = parseId((await params).id);
    if (id === null) return Response.json({ error: "invalid zone id" }, { status: 400 });
    const parsed = parseZoneInput(await req.json().catch(() => null), true);
    if (!parsed.ok) return Response.json({ error: parsed.error }, { status: 400 });
    const [row] = await db
      .update(zones)
      .set({ ...parsed.value, updatedAt: new Date() })
      .where(eq(zones.id, id))
      .returning();
    if (!row) return Response.json({ error: "zone not found" }, { status: 404 });
    return Response.json({ zone: zoneRowToDTO(row) });
  } catch (err) {
    return serverError(err);
  }
}

export async function DELETE(_req: Request, { params }: { params: Promise<{ id: string }> }) {
  try {
    await ensureSchema();
    const id = parseId((await params).id);
    if (id === null) return Response.json({ error: "invalid zone id" }, { status: 400 });
    const removed = await db.delete(zones).where(eq(zones.id, id)).returning({ id: zones.id });
    if (removed.length === 0) return Response.json({ error: "zone not found" }, { status: 404 });
    return Response.json({ ok: true });
  } catch (err) {
    return serverError(err);
  }
}
