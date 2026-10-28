import { promises as fs } from "fs";
import path from "path";

export const SOURCE_ROOT = path.join(process.cwd(), "perception_system");
const SKIP_DIRS = new Set(["__pycache__", "data", ".pytest_cache", ".venv", "node_modules"]);
const TEXT_EXT = new Set([".py", ".yaml", ".yml", ".md", ".txt", ".toml", ".ini", ".cfg", ".json"]);
const TEXT_NAMES = new Set(["Dockerfile", ".gitignore", ".dockerignore"]);

export interface TreeNode {
  name: string;
  path: string;
  type: "dir" | "file";
  size?: number;
  lines?: number;
  children?: TreeNode[];
}

export function isTextFile(name: string): boolean {
  return TEXT_NAMES.has(name) || TEXT_EXT.has(path.extname(name));
}

export async function buildTree(dir = SOURCE_ROOT, rel = ""): Promise<TreeNode[]> {
  let entries;
  try {
    entries = await fs.readdir(dir, { withFileTypes: true });
  } catch {
    return [];
  }
  const nodes: TreeNode[] = [];
  for (const entry of entries.sort((a, b) => Number(b.isDirectory()) - Number(a.isDirectory()) || a.name.localeCompare(b.name))) {
    const childRel = rel ? `${rel}/${entry.name}` : entry.name;
    if (entry.isDirectory()) {
      if (SKIP_DIRS.has(entry.name)) continue;
      nodes.push({ name: entry.name, path: childRel, type: "dir", children: await buildTree(path.join(dir, entry.name), childRel) });
    } else if (isTextFile(entry.name)) {
      const full = path.join(dir, entry.name);
      const content = await fs.readFile(full, "utf8").catch(() => "");
      nodes.push({ name: entry.name, path: childRel, type: "file", size: Buffer.byteLength(content), lines: content.split("\n").length });
    }
  }
  return nodes;
}

/** Read a file inside SOURCE_ROOT only (path-traversal safe, text files only, size-capped). */
export async function readSourceFile(relPath: string): Promise<string | null> {
  const resolved = path.resolve(SOURCE_ROOT, relPath);
  if (!resolved.startsWith(SOURCE_ROOT + path.sep)) return null;
  if (!isTextFile(path.basename(resolved))) return null;
  if (resolved.split(path.sep).some((part) => SKIP_DIRS.has(part))) return null;
  try {
    const stat = await fs.stat(resolved);
    if (!stat.isFile() || stat.size > 512 * 1024) return null;
    return await fs.readFile(resolved, "utf8");
  } catch {
    return null;
  }
}

export function countTree(nodes: TreeNode[]): { files: number; lines: number; pyFiles: number; pyLines: number } {
  let files = 0, lines = 0, pyFiles = 0, pyLines = 0;
  const walk = (list: TreeNode[]) => {
    for (const n of list) {
      if (n.type === "dir") walk(n.children ?? []);
      else {
        files++;
        lines += n.lines ?? 0;
        if (n.name.endsWith(".py")) {
          pyFiles++;
          pyLines += n.lines ?? 0;
        }
      }
    }
  };
  walk(nodes);
  return { files, lines, pyFiles, pyLines };
}
