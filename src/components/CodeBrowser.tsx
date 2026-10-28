"use client";

import { useEffect, useState } from "react";
import type { TreeNode } from "@/lib/source";

function TreeView({ nodes, depth, selected, onSelect }: { nodes: TreeNode[]; depth: number; selected: string; onSelect: (p: string) => void }) {
  const [open, setOpen] = useState<Record<string, boolean>>({});
  return (
    <ul>
      {nodes.map((n) =>
        n.type === "dir" ? (
          <li key={n.path}>
            <button
              type="button"
              onClick={() => setOpen((o) => ({ ...o, [n.path]: !(o[n.path] ?? depth === 0) }))}
              className="flex w-full items-center gap-1 py-0.5 text-left text-hud-amber hover:text-hud-white"
              style={{ paddingLeft: depth * 12 }}
            >
              <span className="w-3">{(open[n.path] ?? depth === 0) ? "▾" : "▸"}</span>
              {n.name}/
            </button>
            {(open[n.path] ?? depth === 0) && n.children ? <TreeView nodes={n.children} depth={depth + 1} selected={selected} onSelect={onSelect} /> : null}
          </li>
        ) : (
          <li key={n.path}>
            <button
              type="button"
              onClick={() => onSelect(n.path)}
              className={`flex w-full justify-between gap-2 py-0.5 pr-2 text-left hover:text-hud-white ${selected === n.path ? "bg-hud-amber/10 text-hud-amber" : "text-hud-grey"}`}
              style={{ paddingLeft: depth * 12 + 14 }}
            >
              <span className="truncate">{n.name}</span>
              <span className="shrink-0 text-hud-muted">{n.lines}</span>
            </button>
          </li>
        ),
      )}
    </ul>
  );
}

export default function CodeBrowser({ tree, initialPath }: { tree: TreeNode[]; initialPath: string }) {
  const [path, setPath] = useState(initialPath);
  const [content, setContent] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    fetch(`/api/source?path=${encodeURIComponent(path)}`)
      .then(async (r) => {
        const d = await r.json();
        if (!alive) return;
        if (!r.ok) throw new Error(d.error ?? "unavailable");
        setContent(d.content as string);
        setError(null);
      })
      .catch((e: unknown) => alive && setError(e instanceof Error ? e.message : "unavailable"));
    return () => {
      alive = false;
    };
  }, [path]);

  const lines = content.split("\n");
  return (
    <div className="grid min-h-[560px] gap-0 md:grid-cols-[260px_minmax(0,1fr)]">
      <div className="max-h-[640px] overflow-y-auto border-b border-hud-border p-2 text-[11px] md:border-b-0 md:border-r">
        <TreeView nodes={tree} depth={0} selected={path} onSelect={setPath} />
      </div>
      <div className="min-w-0">
        <div className="flex items-center justify-between border-b border-hud-border px-3 py-1.5 text-[11px]">
          <span className="text-hud-amber">perception_system/{path}</span>
          <span className="text-hud-muted">{lines.length} lines</span>
        </div>
        {error ? (
          <p className="p-3 text-xs text-hud-red">{error}</p>
        ) : (
          <pre className="code-view max-h-[600px] overflow-auto p-0 text-[11.5px] leading-[1.55]">
            <code>
              {lines.map((line, i) => (
                <div key={i} className="flex hover:bg-hud-panel2">
                  <span className="w-12 shrink-0 select-none pr-3 text-right text-hud-muted">{i + 1}</span>
                  <span className={line.trimStart().startsWith("#") ? "whitespace-pre text-hud-grey" : "whitespace-pre text-hud-white"}>{line || " "}</span>
                </div>
              ))}
            </code>
          </pre>
        )}
      </div>
    </div>
  );
}
