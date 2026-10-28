"use client";

import { useEffect, useState } from "react";
import { Button, Field, Panel } from "@/components/ui";
import { DEFAULT_SETTINGS, type SettingsData } from "@/lib/types";

type NumKey = { [K in keyof SettingsData]: SettingsData[K] extends number ? K : never }[keyof SettingsData];

const NUMBER_FIELDS: { key: NumKey; label: string; step: number; hint: string }[] = [
  { key: "retentionDays", label: "DATA RETENTION (DAYS)", step: 1, hint: "Events older than this are purged automatically." },
  { key: "crowdThreshold", label: "CROWD THRESHOLD (PERSONS)", step: 1, hint: "Used when a zone has no max occupancy." },
  { key: "angryMinConfidence", label: "ANGRY MIN CONFIDENCE", step: 0.05, hint: "0..1 - expression rule sensitivity." },
  { key: "angryMinDurationS", label: "ANGRY MIN DURATION (S)", step: 0.5, hint: "Expression must persist this long." },
  { key: "defaultDwellLimitS", label: "DEFAULT DWELL LIMIT (S)", step: 1, hint: "Loitering threshold when the zone has none." },
  { key: "alertCooldownS", label: "ALERT COOLDOWN (S)", step: 1, hint: "Per rule / zone / track de-duplication." },
  { key: "detectionConfidence", label: "DETECTION CONFIDENCE", step: 0.01, hint: "Engine overlay: detection.confidence." },
  { key: "nmsIou", label: "NMS IOU", step: 0.05, hint: "Engine overlay: detection.iou." },
];

export default function SettingsForm() {
  const [form, setForm] = useState<SettingsData>(DEFAULT_SETTINGS);
  const [tokenConfigured, setTokenConfigured] = useState(false);
  const [status, setStatus] = useState<{ text: string; tone: "ok" | "err" } | null>(null);
  const [purge, setPurge] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    fetch("/api/settings", { cache: "no-store" })
      .then((r) => r.json())
      .then((d) => {
        if (d.settings) setForm(d.settings);
        setTokenConfigured(Boolean(d.ingestTokenConfigured));
      })
      .catch(() => setStatus({ text: "Could not load settings.", tone: "err" }));
  }, []);

  const save = async () => {
    setBusy(true);
    try {
      const res = await fetch("/api/settings", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(form) });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? "save failed");
      setForm(data.settings);
      setStatus({ text: "Settings saved. The simulator and console use them immediately; merge the engine overlay into config.yaml.", tone: "ok" });
    } catch (err) {
      setStatus({ text: err instanceof Error ? err.message : "save failed", tone: "err" });
    } finally {
      setBusy(false);
    }
  };

  const purgeNow = async () => {
    const res = await fetch("/api/retention", { method: "POST" });
    const data = await res.json();
    setPurge(res.ok ? `Purged ${data.deleted} event(s) older than ${data.cutoff}.` : "Purge failed.");
  };

  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_380px]">
      <div className="space-y-4">
        <Panel title="ENGINE LINK">
          <div className="grid gap-3 md:grid-cols-3">
            <Field label="SITE NAME">
              <input className="hud-input" value={form.siteName} onChange={(e) => setForm({ ...form, siteName: e.target.value })} />
            </Field>
            <Field label="MJPEG STREAM URL" hint="Engine ui.stream_server - /stream.mjpg">
              <input className="hud-input" value={form.engineStreamUrl} onChange={(e) => setForm({ ...form, engineStreamUrl: e.target.value })} />
            </Field>
            <Field label="STATE URL" hint="Engine /api/state (JSON telemetry)">
              <input className="hud-input" value={form.engineStateUrl} onChange={(e) => setForm({ ...form, engineStateUrl: e.target.value })} />
            </Field>
          </div>
        </Panel>
        <Panel title="THRESHOLDS & POLICY">
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
            {NUMBER_FIELDS.map((f) => (
              <Field key={f.key} label={f.label} hint={f.hint}>
                <input className="hud-input" type="number" step={f.step} value={form[f.key]} onChange={(e) => setForm({ ...form, [f.key]: Number(e.target.value) })} />
              </Field>
            ))}
          </div>
          <div className="mt-4 flex flex-wrap gap-6 text-[11px] text-hud-grey">
            <label className="flex cursor-pointer items-center gap-2">
              <input type="checkbox" checked={form.blurFaces} onChange={(e) => setForm({ ...form, blurFaces: e.target.checked })} />
              PIXELATE FACES OF NON-ENROLLED PEOPLE (DEFAULT FOR SIMULATOR VIEW)
            </label>
            <label className="flex cursor-pointer items-center gap-2">
              <input type="checkbox" checked={form.allowSimulatorIngest} onChange={(e) => setForm({ ...form, allowSimulatorIngest: e.target.checked })} />
              ACCEPT BROWSER EVENTS (SIMULATOR / MY CAMERA) WITHOUT INGEST TOKEN
            </label>
          </div>
          <div className="mt-4 flex items-center gap-3">
            <Button variant="primary" onClick={save} disabled={busy}>
              SAVE SETTINGS
            </Button>
            {status ? <span className={`text-[11px] ${status.tone === "ok" ? "text-hud-green" : "text-hud-red"}`}>{status.text}</span> : null}
          </div>
        </Panel>
      </div>
      <aside className="space-y-4">
        <Panel title="DATA RETENTION">
          <p className="text-[11px] leading-relaxed text-hud-grey">
            Retention is <span className="text-hud-white">{form.retentionDays} days</span>. Expired events are deleted automatically on ingest
            (at most every 10 min); the engine applies the same policy to SQLite, snapshots and logs.
          </p>
          <div className="mt-3 flex items-center gap-2">
            <Button variant="danger" onClick={purgeNow}>
              PURGE EXPIRED NOW
            </Button>
          </div>
          {purge ? <p className="mt-2 text-[11px] text-hud-green">{purge}</p> : null}
        </Panel>
        <Panel title="INGEST SECURITY">
          <p className="text-[11px] leading-relaxed text-hud-grey">
            INGEST_TOKEN:{" "}
            {tokenConfigured ? <span className="text-hud-green">CONFIGURED</span> : <span className="text-hud-amber">NOT SET (open ingest)</span>}
            . Set the <code className="text-hud-white">INGEST_TOKEN</code> environment variable on both the console and the engine to require{" "}
            <code className="text-hud-white">Authorization: Bearer …</code> on POST /api/events and /api/heartbeat.
          </p>
        </Panel>
        <Panel title="ENGINE CONFIG OVERLAY">
          <p className="text-[11px] leading-relaxed text-hud-grey">These values map onto perception_system/config.yaml keys:</p>
          <a href="/api/settings?format=engine" target="_blank" rel="noreferrer" className="mt-2 block rounded border border-hud-border px-3 py-2 text-[11px] text-hud-amber hover:border-hud-amber">
            GET /api/settings?format=engine
          </a>
        </Panel>
        <Panel title="PRIVACY GUARANTEES">
          <ul className="list-inside list-disc space-y-1 text-[11px] text-hud-grey">
            <li>Recognition only for enrolled people with documented consent.</li>
            <li>Everyone else is shown as &quot;Person #ID&quot;.</li>
            <li>Embeddings only - face images are never stored.</li>
            <li>Monitoring indicator cannot be disabled.</li>
          </ul>
        </Panel>
      </aside>
    </div>
  );
}
