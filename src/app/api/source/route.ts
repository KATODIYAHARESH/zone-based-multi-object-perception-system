import { buildTree, readSourceFile } from "@/lib/source";

export const dynamic = "force-dynamic";

export async function GET(req: Request) {
  const rel = new URL(req.url).searchParams.get("path");
  if (!rel) return Response.json({ tree: await buildTree() });
  const content = await readSourceFile(rel);
  if (content === null) return Response.json({ error: "file not found or not viewable" }, { status: 404 });
  return Response.json({ path: rel, content });
}
