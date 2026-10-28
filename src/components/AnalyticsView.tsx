"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Panel, Segmented, Stat } from "@/components/ui";
import { CATEGORIES, CATEGORY_COLOR, EMOTION_COLOR, PALETTE, SEVERITY_COLOR, isCategory, type Emotion, type Severity } from "@/lib/types";

interface Analytics {
  hours: number;
  bucketSeconds: number;
  since: string;
  kpis: { entries: number; exits: number; alerts: number; critical: number; warning: number; info: number; uniqueTracks: number };
  timeline: { t: number; category: string; n: number }[];
  alertTimeline: { t: number; severity: string; n: number }[];
  zones: { zone: string; entries: number; exits: number; alerts: number; avgDwell: number | null }[];
  categories: { category: string; n: number }[];
  emotions: { emotion: string; n: number }[];
  heatmap: { gx: number; gy: number; n: number }[];
  rules: { ruleId: string; severity: string; n: number }[];
}

const RANGES = ["1", "6", "24", "168"] as const;
type Range = (typeof RANGES)[number];

/** HOT-style ramp (black -> red -> amber -> white), mirroring cv2.COLORMAP_HOT. No blue. */
function heatColor(t: number): string {
  const c = Math.max(0, Math.min(1, t));
  const lerp = (a: number[], b: number[], f: number) => a.map((v, i) => Math.round(v + (b[i] - v) * f));
  const stops = [[18, 16, 12], [255, 59, 48], [255, 176, 0], [242, 242, 242]];
  const seg = Math.min(2, Math.floor(c * 3));
  const [r, g, b] = lerp(stops[seg], stops[seg + 1], c * 3 - seg);
  return `rgb(${r}, ${g}, ${b})`;
}

function HBar({ label, value, max, color }: { label: string; value: number; max: number; color: string }) {
  return (
    <div>
      <div className="flex justify-between text-[10px] tracking-[0.12em]">
        <span style={{ color }}>{label.toUpperCase()}</span>
        <span className="tabular-nums text-hud-grey">{value}</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded bg-hud-panel2">
        <div className="h-full" style={{ width: `${max ? (100 * value) / max : 0}%`, background: color }} />
      </div>
    </div>
  );
}

export default function AnalyticsView() {
  const [range, setRange] = useState<Range>("24");
  const [data, setData] = useState<Analytics | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const res = await fetch(`/api/analytics?hours=${range}`, { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setData((await res.json()) as Analytics);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "failed to load analytics");
    }
  }, [range]);

  useEffect(() => {
    void load();
    const id = setInterval(load, 15000);
    return () => clearInterval(id);
  }, [load]);

  const chart = useMemo(() => {
    if (!data) return null;
    const step = data.bucketSeconds * 1000;
    const start = Math.floor(new Date(data.since).getTime() / step) * step;
    const n = Math.max(1, Math.ceil((Date.now() - start) / step));
    const buckets = Array.from({ length: n }, (_, i) => ({ t: start + i * step, per: {} as Record<string, number>, alerts: {} as Record<string, number> }));
    const idx = (t: number) => Math.min(n - 1, Math.max(0, Math.round((t - start) / step)));
    for (const r of data.timeline) buckets[idx(r.t)].per[r.category] = (buckets[idx(r.t)].per[r.category] ?? 0) + r.n;
    for (const r of data.alertTimeline) buckets[idx(r.t)].alerts[r.severity] = (buckets[idx(r.t)].alerts[r.severity] ?? 0) + r.n;
    const max = Math.max(1, ...buckets.map((b) => Object.values(b.per).reduce((s, v) => s + v, 0)));
    return { buckets, max };
  }, [data]);

  const heatMax = Math.max(1, ...(data?.heatmap.map((h) => h.n) ?? [1]));
  const catMax = Math.max(1, ...(data?.categories.map((c) => c.n) ?? [1]));
  const emoMax = Math.max(1, ...(data?.emotions.map((e) => e.n) ?? [1]));
  const zoneMax = Math.max(1, ...(data?.zones.map((z) => z.entries) ?? [1]));
  const fmt = (t: number) => new Date(t).toISOString().slice(range === "168" ? 5 : 11, range === "168" ? 16 : 16).replace("T", " ");

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <Segmented<Range> value={range} options={RANGES} labels={{ "1": "1 h", "6": "6 h", "24": "24 h", "168": "7 d" }} onChange={setRange} />
        <span className="text-[10px] tracking-[0.14em] text-hud-muted">AUTO-REFRESH 15 s · UTC</span>
        <div className="ml-auto flex gap-2 text-[11px]">
          <a className="rounded border border-hud-border px-3 py-1.5 text-hud-amber hover:border-hud-amber" href={`/api/export?format=csv&hours=${range}`}>
            ↓ CSV
          </a>
          <a className="rounded border border-hud-border px-3 py-1.5 text-hud-amber hover:border-hud-amber" href={`/api/export?format=json&hours=${range}`}>
            ↓ JSON
          </a>
        </div>
      </div>
      {error ? <p className="text-xs text-hud-red">Analytics unavailable: {error}</p> : null}

      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
        <Stat label="ENTRIES" value={data?.kpis.entries ?? "-"} />
        <Stat label="EXITS" value={data?.kpis.exits ?? "-"} />
        <Stat label="ALERTS" value={data?.kpis.alerts ?? "-"} color={PALETTE.amber} />
        <Stat label="CRITICAL" value={data?.kpis.critical ?? "-"} color={PALETTE.red} />
        <Stat label="WARNING" value={data?.kpis.warning ?? "-"} color={PALETTE.amber} />
        <Stat label="UNIQUE TRACKS" value={data?.kpis.uniqueTracks ?? "-"} color={PALETTE.green} />
      </div>

      <Panel title="ZONE ENTRIES TIMELINE (STACKED BY CATEGORY) · ALERT TICKS">
        {chart ? (
          <div>
            <svg viewBox="0 0 1000 240" className="h-60 w-full" preserveAspectRatio="none" role="img" aria-label="Entries timeline">
              {[0.25, 0.5, 0.75, 1].map((f) => (
                <line key={f} x1={0} x2={1000} y1={220 - f * 190} y2={220 - f * 190} stroke="rgba(255,176,0,0.08)" />
              ))}
              {chart.buckets.map((b, i) => {
                const bw = 1000 / chart.buckets.length;
                let y = 220;
                const rects = CATEGORIES.map((c) => {
                  const v = b.per[c] ?? 0;
                  if (!v) return null;
                  const h = (v / chart.max) * 190;
                  y -= h;
                  return <rect key={c} x={i * bw + bw * 0.12} y={y} width={bw * 0.76} height={h} fill={CATEGORY_COLOR[c]} opacity={0.9} />;
                });
                const sev = (["critical", "warning", "info"] as Severity[]).find((s) => b.alerts[s]);
                return (
                  <g key={b.t}>
                    {rects}
                    {sev ? <rect x={i * bw + bw * 0.3} y={4} width={bw * 0.4} height={12} fill={SEVERITY_COLOR[sev]} /> : null}
                  </g>
                );
              })}
              <line x1={0} x2={1000} y1={220} y2={220} stroke="#2A2922" />
            </svg>
            <div className="flex justify-between text-[10px] text-hud-muted">
              <span>{fmt(chart.buckets[0].t)}</span>
              <span>peak {chart.max} / bucket ({data!.bucketSeconds / 60} min)</span>
              <span>{fmt(chart.buckets[chart.buckets.length - 1].t)}</span>
            </div>
            <div className="mt-2 flex flex-wrap gap-3 text-[10px]">
              {CATEGORIES.map((c) => (
                <span key={c} className="flex items-center gap-1" style={{ color: CATEGORY_COLOR[c] }}>
                  <span className="h-2 w-2" style={{ background: CATEGORY_COLOR[c] }} />
                  {c.toUpperCase()}
                </span>
              ))}
            </div>
          </div>
        ) : (
          <p className="text-xs text-hud-muted">Loading...</p>
        )}
      </Panel>

      <div className="grid gap-4 lg:grid-cols-3">
        <Panel title="ZONE TRAFFIC" className="lg:col-span-2" bodyClassName="p-0">
          <table className="w-full text-[11px]">
            <thead>
              <tr className="text-hud-grey">
                <th className="px-3 py-2 text-left font-normal">ZONE</th>
                <th className="px-3 py-2 text-left font-normal">ENTRIES</th>
                <th className="px-3 py-2 text-right font-normal">EXITS</th>
                <th className="px-3 py-2 text-right font-normal">ALERTS</th>
                <th className="px-3 py-2 text-right font-normal">AVG DWELL</th>
              </tr>
            </thead>
            <tbody>
              {(data?.zones ?? []).map((z) => (
                <tr key={z.zone} className="border-t border-hud-border">
                  <td className="px-3 py-1.5 text-hud-white">{z.zone}</td>
                  <td className="px-3 py-1.5">
                    <div className="flex items-center gap-2">
                      <div className="h-1.5 w-32 overflow-hidden rounded bg-hud-panel2">
                        <div className="h-full bg-hud-amber" style={{ width: `${(100 * z.entries) / zoneMax}%` }} />
                      </div>
                      <span className="tabular-nums">{z.entries}</span>
                    </div>
                  </td>
                  <td className="px-3 py-1.5 text-right tabular-nums">{z.exits}</td>
                  <td className="px-3 py-1.5 text-right tabular-nums" style={{ color: z.alerts ? PALETTE.amber : PALETTE.grey }}>
                    {z.alerts}
                  </td>
                  <td className="px-3 py-1.5 text-right tabular-nums">{z.avgDwell === null ? "-" : `${z.avgDwell.toFixed(1)} s`}</td>
                </tr>
              ))}
              {data && data.zones.length === 0 ? (
                <tr>
                  <td colSpan={5} className="px-3 py-4 text-center text-hud-muted">
                    No events in this window - run the simulator on the LIVE page or connect the engine.
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </Panel>
        <Panel title="TOP ALERT RULES">
          <ul className="space-y-1.5 text-[11px]">
            {(data?.rules ?? []).map((r) => (
              <li key={r.ruleId} className="flex justify-between">
                <span style={{ color: SEVERITY_COLOR[(r.severity as Severity) ?? "info"] ?? PALETTE.white }}>{r.ruleId}</span>
                <span className="tabular-nums text-hud-grey">{r.n}</span>
              </li>
            ))}
            {data && data.rules.length === 0 ? <li className="text-hud-muted">No alerts.</li> : null}
          </ul>
        </Panel>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <Panel title="CATEGORY DISTRIBUTION (ENTRIES)">
          <div className="space-y-1.5">
            {(data?.categories ?? []).map((c) => (
              <HBar key={c.category} label={c.category} value={c.n} max={catMax} color={isCategory(c.category) ? CATEGORY_COLOR[c.category] : PALETTE.white} />
            ))}
            {data && data.categories.length === 0 ? <p className="text-[11px] text-hud-muted">No entries.</p> : null}
          </div>
        </Panel>
        <Panel title="EXPRESSIONS (AT ZONE EVENTS)">
          <div className="space-y-1.5">
            {(data?.emotions ?? []).map((e) => (
              <HBar key={e.emotion} label={e.emotion} value={e.n} max={emoMax} color={EMOTION_COLOR[e.emotion as Emotion] ?? PALETTE.white} />
            ))}
            {data && data.emotions.length === 0 ? <p className="text-[11px] text-hud-muted">No expression samples.</p> : null}
          </div>
        </Panel>
        <Panel title="EVENT HEATMAP (GROUND CONTACT POINTS)">
          <svg viewBox="0 0 32 18" className="aspect-video w-full rounded bg-hud-bg" role="img" aria-label="Event heatmap">
            {(data?.heatmap ?? []).map((h) => (
              <rect key={`${h.gx}-${h.gy}`} x={h.gx} y={h.gy} width={1} height={1} fill={heatColor(Math.sqrt(h.n / heatMax))} />
            ))}
          </svg>
          <p className="mt-1 text-[10px] text-hud-muted">32×18 grid of normalised positions · HOT colour ramp (no blue)</p>
        </Panel>
      </div>
    </div>
  );
}
