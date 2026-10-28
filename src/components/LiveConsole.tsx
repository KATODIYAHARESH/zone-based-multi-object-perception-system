"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Simulator, { type SimSnapshot } from "@/components/Simulator";
import { Button, Panel, Segmented, Stat } from "@/components/ui";
import {
  CATEGORIES,
  CATEGORY_COLOR,
  DEFAULT_SETTINGS,
  EMOTIONS,
  EMOTION_COLOR,
  PALETTE,
  SEVERITY_COLOR,
  type EngineStatusDTO,
  type EventDTO,
  type SettingsData,
  type ZoneDTO,
} from "@/lib/types";

type Mode = "simulator" | "engine";

function fmtTime(iso: string) {
  return new Date(iso).toISOString().slice(11, 19);
}

export default function LiveConsole() {
  const [mode, setMode] = useState<Mode>("simulator");
  const [zones, setZones] = useState<ZoneDTO[]>([]);
  const [settings, setSettings] = useState<SettingsData>(DEFAULT_SETTINGS);
  const [events, setEvents] = useState<EventDTO[]>([]);
  const [engines, setEngines] = useState<EngineStatusDTO[]>([]);
  const [sim, setSim] = useState<SimSnapshot | null>(null);
  const [running, setRunning] = useState(true);
  const [spawnRate, setSpawnRate] = useState(0.9);
  const [blur, setBlur] = useState(false);
  const [streamError, setStreamError] = useState(false);
  const [streamKey, setStreamKey] = useState(0);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const [z, s] = await Promise.all([fetch("/api/zones").then((r) => r.json()), fetch("/api/settings").then((r) => r.json())]);
        setZones(z.zones ?? []);
        if (s.settings) {
          setSettings(s.settings);
          setBlur(Boolean(s.settings.blurFaces));
        }
      } catch {
        setLoadError("Could not load zones/settings from the console database.");
      }
    })();
  }, []);

  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const [ev, hb] = await Promise.all([
          fetch("/api/events?limit=40", { cache: "no-store" }).then((r) => r.json()),
          fetch("/api/heartbeat", { cache: "no-store" }).then((r) => r.json()),
        ]);
        if (!alive) return;
        setEvents(ev.events ?? []);
        setEngines(hb.engines ?? []);
      } catch {
        /* transient */
      }
    };
    void poll();
    const id = setInterval(poll, 2000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  const onStats = useCallback((s: SimSnapshot) => setSim(s), []);
  const engine = engines.find((e) => e.source === "engine");
  const zoneName = useMemo(() => Object.fromEntries(zones.map((z) => [z.slug, z.name])), [zones]);

  const counts = mode === "simulator" ? (sim?.counts ?? {}) : (engine?.counts ?? {});
  const emotions = mode === "simulator" ? (sim?.emotions ?? {}) : (engine?.emotions ?? {});
  const emotionTotal = Object.values(emotions).reduce((a, b) => a + b, 0) || 1;
  const fps = mode === "simulator" ? (sim?.fps ?? 0) : (engine?.online ? (engine.fps ?? 0) : 0);
  const fpsColor = fps >= 25 ? PALETTE.green : fps >= 15 ? PALETTE.amber : PALETTE.red;

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_380px]">
      <div className="min-w-0 space-y-4">
        <Panel
          title="LIVE FEED"
          right={
            <Segmented<Mode>
              value={mode}
              options={["simulator", "engine"] as const}
              labels={{ simulator: "Simulator", engine: "Engine stream" }}
              onChange={(m) => {
                setMode(m);
                setStreamError(false);
              }}
            />
          }
          bodyClassName="p-0"
        >
          <div className="scanlines relative aspect-video w-full overflow-hidden bg-black">
            {mode === "simulator" ? (
              <Simulator zones={zones} settings={settings} running={running} spawnRate={spawnRate} blurFaces={blur} onStats={onStats} />
            ) : streamError ? (
              <div className="flex h-full flex-col items-center justify-center gap-3 p-6 text-center">
                <div className="text-2xl font-bold tracking-[0.3em] text-hud-amber">NO SIGNAL</div>
                <p className="max-w-lg text-xs leading-relaxed text-hud-grey">
                  The perception engine stream at <span className="text-hud-white">{settings.engineStreamUrl}</span> is not
                  reachable from this browser. Start it with{" "}
                  <code className="text-hud-amber">python main.py run --headless</code> inside{" "}
                  <code className="text-hud-amber">perception_system/</code>, or change the URL under Settings.
                </p>
                <Button
                  onClick={() => {
                    setStreamError(false);
                    setStreamKey((k) => k + 1);
                  }}
                >
                  RETRY
                </Button>
              </div>
            ) : (
              // MJPEG streams require a plain <img>; next/image cannot proxy multipart streams.
              // eslint-disable-next-line @next/next/no-img-element
              <img
                key={streamKey}
                src={settings.engineStreamUrl}
                alt="Perception engine live stream"
                className="h-full w-full object-contain"
                onError={() => setStreamError(true)}
              />
            )}
          </div>
          <div className="grid grid-cols-2 gap-2 border-t border-hud-border p-3 sm:grid-cols-3 lg:grid-cols-6">
            <Stat label="FPS" value={fps.toFixed(1)} color={fpsColor} />
            <Stat
              label="LATENCY"
              value={`${(mode === "simulator" ? (sim?.latencyMs ?? 0) : (engine?.latencyMs ?? 0)).toFixed(1)} ms`}
            />
            <Stat label="TRACKS" value={mode === "simulator" ? (sim?.tracks ?? 0) : (engine?.tracks ?? 0)} />
            <Stat label="FRAME SKIP" value={mode === "simulator" ? 1 : (engine?.frameSkip ?? "-")} />
            <Stat label="DEVICE" value={mode === "simulator" ? "BROWSER" : (engine?.device ?? "-")} />
            <Stat
              label="LINK"
              value={mode === "simulator" ? "SIM" : engine?.online ? "ONLINE" : "OFFLINE"}
              color={mode === "simulator" ? PALETTE.amber : engine?.online ? PALETTE.green : PALETTE.red}
            />
          </div>
          {mode === "simulator" ? (
            <div className="flex flex-wrap items-center gap-4 border-t border-hud-border px-3 py-2.5 text-[11px] text-hud-grey">
              <Button onClick={() => setRunning((r) => !r)} variant={running ? "default" : "primary"}>
                {running ? "PAUSE" : "RESUME"}
              </Button>
              <label className="flex items-center gap-2">
                SPAWN RATE
                <input type="range" min={0.2} max={2.5} step={0.1} value={spawnRate} onChange={(e) => setSpawnRate(Number(e.target.value))} />
                <span className="w-10 text-hud-white">{spawnRate.toFixed(1)}/s</span>
              </label>
              <label className="flex cursor-pointer items-center gap-2">
                <input type="checkbox" checked={blur} onChange={(e) => setBlur(e.target.checked)} />
                PIXELATE FACES
              </label>
              <span className="ml-auto">
                SYNTHETIC EVENTS INGESTED: <span className="text-hud-amber">{sim?.posted ?? 0}</span>
              </span>
            </div>
          ) : null}
        </Panel>
        {loadError ? <p className="text-xs text-hud-red">{loadError}</p> : null}
      </div>

      <aside className="space-y-4">
        <Panel title="LIVE ZONE COUNTS" right={<span className="text-[10px] text-hud-muted">{mode === "simulator" ? "SIM" : "ENGINE"}</span>}>
          <table className="w-full text-[11px]">
            <thead>
              <tr className="text-hud-grey">
                <th className="py-1 text-left font-normal">ZONE</th>
                {CATEGORIES.map((c) => (
                  <th key={c} className="py-1 text-right font-normal" style={{ color: CATEGORY_COLOR[c] }} title={c}>
                    {c.slice(0, 3).toUpperCase()}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.keys(counts).length === 0 ? (
                <tr>
                  <td colSpan={7} className="py-3 text-center text-hud-muted">
                    {mode === "engine" ? "No engine heartbeat yet" : "Waiting for data"}
                  </td>
                </tr>
              ) : (
                Object.entries(counts).map(([slug, per]) => {
                  const violated = mode === "simulator" && sim?.violated.includes(slug);
                  return (
                    <tr key={slug} className="border-t border-hud-border">
                      <td className="py-1.5" style={{ color: violated ? PALETTE.red : PALETTE.white }}>
                        {violated ? "▲ " : ""}
                        {zoneName[slug] ?? slug}
                      </td>
                      {CATEGORIES.map((c) => (
                        <td key={c} className="py-1.5 text-right tabular-nums" style={{ color: per[c] ? CATEGORY_COLOR[c] : PALETTE.grey }}>
                          {per[c] ?? 0}
                        </td>
                      ))}
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </Panel>

        <Panel title="EXPRESSION METER" right={<span className="text-[10px] text-hud-muted">VISIBLE FACES</span>}>
          <div className="space-y-1.5">
            {EMOTIONS.map((e) => {
              const v = emotions[e] ?? 0;
              return (
                <div key={e}>
                  <div className="flex justify-between text-[10px] tracking-[0.14em]">
                    <span style={{ color: EMOTION_COLOR[e] }}>{e.toUpperCase()}</span>
                    <span className="tabular-nums text-hud-grey">{v}</span>
                  </div>
                  <div className="h-1.5 overflow-hidden rounded bg-hud-panel2">
                    <div className="h-full transition-all duration-500" style={{ width: `${(100 * v) / emotionTotal}%`, background: EMOTION_COLOR[e] }} />
                  </div>
                </div>
              );
            })}
          </div>
          <p className="mt-2 text-[10px] leading-relaxed text-hud-muted">
            Facial expression is not emotion. Use this as an aggregate signal only, never for decisions about individuals.
          </p>
        </Panel>

        <Panel title="EVENT LOG" right={<span className="text-[10px] text-hud-muted">DATABASE · 2 s POLL</span>} bodyClassName="p-0">
          <ul className="max-h-[420px] divide-y divide-hud-border overflow-y-auto text-[11px]">
            {events.length === 0 ? <li className="p-3 text-hud-muted">No events yet.</li> : null}
            {events.map((e) => {
              const color = e.type === "alert" ? SEVERITY_COLOR[e.severity ?? "info"] : PALETTE.grey;
              return (
                <li key={e.id} className="flex gap-2 px-3 py-1.5">
                  <span className="shrink-0 tabular-nums text-hud-muted">{fmtTime(e.ts)}</span>
                  <span className="shrink-0 font-semibold" style={{ color }}>
                    {e.type === "alert" ? (e.severity ?? "info").toUpperCase().slice(0, 4) : e.type.toUpperCase()}
                  </span>
                  <span className="min-w-0 flex-1 truncate" style={{ color: e.type === "alert" ? color : PALETTE.white }} title={e.message ?? ""}>
                    {e.type === "alert"
                      ? e.message
                      : `${e.zoneName ?? e.zoneId ?? "-"} · ${e.category === "human" ? `Person #${e.trackId}` : `${e.label ?? "?"} #${e.trackId}`}${
                          e.type === "exit" && e.dwellS != null ? ` · ${e.dwellS.toFixed(0)}s` : ""
                        }`}
                  </span>
                  <span className="shrink-0 text-[9px] text-hud-muted">{e.source === "simulator" ? "SIM" : e.source === "camera" ? "CAM" : "ENG"}</span>
                </li>
              );
            })}
          </ul>
        </Panel>
      </aside>
    </div>
  );
}
