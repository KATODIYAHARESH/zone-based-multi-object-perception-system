import { purgeExpired, serverError } from "@/lib/server";

export const dynamic = "force-dynamic";

/** Apply the data-retention policy immediately (also runs automatically on ingest). */
export async function POST() {
  try {
    return Response.json(await purgeExpired(true));
  } catch (err) {
    return serverError(err);
  }
}
