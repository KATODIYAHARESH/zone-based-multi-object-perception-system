import { asc } from "drizzle-orm";
import { db } from "@/db";
import { zones } from "@/db/schema";
import {
  ensureSchema,
  parseZoneInput,
  serverError,
  uniqueSlug,
  zoneRowToDTO,
  zonesToEngine,
  zonesToYaml,
  type ZoneInput,
} from "@/lib/server";

export const dynamic = "force-dynamic";

export async function GET(req: Request) {
  try {
    await ensureSchema();
    const list = (await db.select().from(zones).orderBy(asc(zones.id))).map(zoneRowToDTO);
    const format = new URL(req.url).searchParams.get("format");
    if (format === "engine") return Response.json({ zones: zonesToEngine(list) });
    if (format === "yaml")
      return new Response(zonesToYaml(list), {
        headers: { "Content-Type": "text/yaml; charset=utf-8", "Content-Disposition": 'attachment; filename="zones.yaml"' },
      });
    return Response.json({ zones: list });
  } catch (err) {
    return serverError(err);
  }
}

export async function POST(req: Request) {
  try {
    await ensureSchema();
    const parsed = parseZoneInput(await req.json().catch(() => null), false);
    if (!parsed.ok) return Response.json({ error: parsed.error }, { status: 400 });
    const value = parsed.value as ZoneInput;
    const [row] = await db
      .insert(zones)
      .values({ ...value, slug: await uniqueSlug(value.name) })
      .returning();
    return Response.json({ zone: zoneRowToDTO(row) }, { status: 201 });
  } catch (err) {
    return serverError(err);
  }
}
