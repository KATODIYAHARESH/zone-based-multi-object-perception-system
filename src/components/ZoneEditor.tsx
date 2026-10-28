"use client";

import { useCallback, useEffect, useRef, useState, type MouseEvent } from "react";
import { Button, Field, Panel, Segmented } from "@/components/ui";
import { describeCameraError, openCamera, stopStream } from "@/lib/camera";
import { pointInPolygon } from "@/lib/geometry";
import {
  CATEGORIES,
  CATEGORY_COLOR,
  DEFAULT_SETTINGS,
  PALETTE,
  ZONE_COLORS,
  ZONE_COLOR_HEX,
  ZONE_MODES,
  hexToRgba,
  type Category,
  type Point,
  type SettingsData,
  type ZoneColor,
  type ZoneDTO,
  type ZoneMode,
} from "@/lib/types";

const W = 1280;
const H = 720;
type Background = "grid" | "engine" | "upload" | "camera";

interface Form {
  name: string;
  mode: ZoneMode;
  categories: Category[];
  color: ZoneColor;
  maxOccupancy: string;
  dwellLimitS: string;
  enabled: boolean;
}

const EMPTY_FORM: Form = { name: "", mode: "restricted", categories: ["vehicle"], color: "amber", maxOccupancy: "", dwellLimitS: "", enabled: true };

export default function ZoneEditor() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [zones, setZones] = useState<ZoneDTO[]>([]);
  const [settings, setSettings] = useState<SettingsData>(DEFAULT_SETTINGS);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [draft, setDraft] = useState<Point[]>([]);
  const [closed, setClosed] = useState(false);
  const [hover, setHover] = useState<Point | null>(null);
  const [form, setForm] = useState<Form>(EMPTY_FORM);
  const [bgMode, setBgMode] = useState<Background>("grid");
  const [bgImage, setBgImage] = useState<HTMLImageElement | null>(null);
  const [status, setStatus] = useState<{ text: string; tone: "ok" | "err" } | null>(null);
  const [busy, setBusy] = useState(false);
  const [captureKey, setCaptureKey] = useState(0);

  const load = useCallback(async () => {
    const res = await fetch("/api/zones", { cache: "no-store" });
    const data = (await res.json()) as { zones: ZoneDTO[] };
    setZones(data.zones ?? []);
  }, []);

  useEffect(() => {
    void load();
    fetch("/api/settings")
      .then((r) => r.json())
      .then((d) => d.settings && setSettings(d.settings))
      .catch(() => undefined);
  }, [load]);

  useEffect(() => {
    if (bgMode !== "engine") return;
    const img = new Image();
    img.onload = () => setBgImage(img);
    img.onerror = () => {
      setBgImage(null);
      setStatus({ text: "Engine snapshot unavailable - is the engine running with the stream server enabled?", tone: "err" });
    };
    img.src = settings.engineStreamUrl.replace(/stream\.mjpg.*$/, "snapshot.jpg") + `?t=${Date.now()}`;
  }, [bgMode, settings.engineStreamUrl]);

  useEffect(() => {
    if (bgMode !== "camera") return;
    let cancelled = false;
    let stream: MediaStream | null = null;
    (async () => {
      try {
        setStatus({ text: "Capturing a still frame from your camera...", tone: "ok" });
        stream = await openCamera();
        const video = document.createElement("video");
        video.muted = true;
        video.playsInline = true;
        video.srcObject = stream;
        await video.play();
        await new Promise((r) => setTimeout(r, 500)); // let exposure settle
        const c = document.createElement("canvas");
        c.width = video.videoWidth || W;
        c.height = video.videoHeight || H;
        c.getContext("2d")?.drawImage(video, 0, 0, c.width, c.height);
        stopStream(stream);
        stream = null;
        const img = new Image();
        img.onload = () => {
          if (cancelled) return;
          setBgImage(img);
          setStatus({ text: "Camera frame captured locally (never uploaded). True camera orientation - not mirrored.", tone: "ok" });
        };
        img.src = c.toDataURL("image/jpeg", 0.9);
      } catch (err) {
        if (!cancelled) setStatus({ text: describeCameraError(err).message, tone: "err" });
      } finally {
        stopStream(stream);
      }
    })();
    return () => {
      cancelled = true;
      stopStream(stream);
    };
  }, [bgMode, captureKey]);

  const draw = useCallback(() => {
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "#0b0c0a";
    ctx.fillRect(0, 0, W, H);
    if (bgImage && bgMode !== "grid") ctx.drawImage(bgImage, 0, 0, W, H);
    else {
      ctx.strokeStyle = "rgba(255, 176, 0, 0.08)";
      for (let x = 0; x <= W; x += 64) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, H);
        ctx.stroke();
      }
      for (let y = 0; y <= H; y += 64) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(W, y);
        ctx.stroke();
      }
      ctx.fillStyle = "rgba(242, 242, 242, 0.25)";
      ctx.font = "12px ui-monospace, Menlo, monospace";
      ctx.fillText("REFERENCE GRID 1280x720 (normalised 0..1) - load an engine snapshot or a local image for real geometry", 16, H - 16);
    }
    for (const z of zones) {
      if (z.id === selectedId && draft.length) continue;
      const color = ZONE_COLOR_HEX[z.color];
      ctx.beginPath();
      z.polygon.forEach(([x, y], i) => (i ? ctx.lineTo(x * W, y * H) : ctx.moveTo(x * W, y * H)));
      ctx.closePath();
      ctx.fillStyle = hexToRgba(color, z.enabled ? 0.1 : 0.03);
      ctx.fill();
      ctx.setLineDash(z.enabled ? [] : [6, 6]);
      ctx.lineWidth = z.id === selectedId ? 3 : 2;
      ctx.strokeStyle = color;
      ctx.stroke();
      ctx.setLineDash([]);
      const top = z.polygon.reduce((b, p) => (p[1] < b[1] ? p : b), z.polygon[0]);
      const tag = `${z.name.toUpperCase()} [${z.mode.toUpperCase()}]${z.enabled ? "" : " OFF"}`;
      ctx.font = "12px ui-monospace, Menlo, monospace";
      const tw = ctx.measureText(tag).width;
      ctx.fillStyle = "rgba(10, 11, 10, 0.85)";
      ctx.fillRect(top[0] * W, Math.max(2, top[1] * H - 22), tw + 8, 17);
      ctx.fillStyle = color;
      ctx.fillText(tag, top[0] * W + 4, Math.max(15, top[1] * H - 9));
    }
    if (draft.length) {
      const color = ZONE_COLOR_HEX[form.color];
      ctx.beginPath();
      draft.forEach(([x, y], i) => (i ? ctx.lineTo(x * W, y * H) : ctx.moveTo(x * W, y * H)));
      if (closed) {
        ctx.closePath();
        ctx.fillStyle = hexToRgba(color, 0.14);
        ctx.fill();
      } else if (hover) ctx.lineTo(hover[0] * W, hover[1] * H);
      ctx.lineWidth = 2;
      ctx.strokeStyle = color;
      ctx.stroke();
      draft.forEach(([x, y], i) => {
        ctx.beginPath();
        ctx.arc(x * W, y * H, i === 0 ? 7 : 5, 0, Math.PI * 2);
        ctx.fillStyle = i === 0 ? PALETTE.white : color;
        ctx.fill();
      });
    }
    if (hover && !draft.length) {
      const inside = zones.filter((z) => z.enabled && pointInPolygon(hover, z.polygon)).map((z) => z.name);
      ctx.fillStyle = "rgba(10, 11, 10, 0.85)";
      ctx.fillRect(hover[0] * W + 10, hover[1] * H + 8, 280, 20);
      ctx.fillStyle = inside.length ? PALETTE.amber : PALETTE.grey;
      ctx.font = "11px ui-monospace, Menlo, monospace";
      ctx.fillText(`(${hover[0].toFixed(3)}, ${hover[1].toFixed(3)}) ${inside.length ? "IN " + inside.join(", ") : "outside zones"}`.slice(0, 44), hover[0] * W + 14, hover[1] * H + 22);
    }
  }, [zones, selectedId, draft, closed, hover, form.color, bgImage, bgMode]);

  useEffect(draw, [draw]);

  const toNorm = (e: MouseEvent<HTMLCanvasElement>): Point => {
    const rect = e.currentTarget.getBoundingClientRect();
    return [Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width)), Math.min(1, Math.max(0, (e.clientY - rect.top) / rect.height))];
  };

  const onClick = (e: MouseEvent<HTMLCanvasElement>) => {
    if (closed) return;
    const p = toNorm(e);
    if (draft.length >= 3) {
      const [fx, fy] = draft[0];
      if (Math.hypot((fx - p[0]) * W, (fy - p[1]) * H) < 14) {
        setClosed(true);
        return;
      }
    }
    setDraft((d) => [...d, [Math.round(p[0] * 10000) / 10000, Math.round(p[1] * 10000) / 10000]]);
  };

  const startNew = () => {
    setSelectedId(null);
    setDraft([]);
    setClosed(false);
    setForm({ ...EMPTY_FORM, name: `Zone ${zones.length + 1}` });
    setStatus(null);
  };

  const select = (z: ZoneDTO) => {
    setSelectedId(z.id);
    setDraft(z.polygon);
    setClosed(true);
    setForm({
      name: z.name,
      mode: z.mode,
      categories: z.categories,
      color: z.color,
      maxOccupancy: z.maxOccupancy?.toString() ?? "",
      dwellLimitS: z.dwellLimitS?.toString() ?? "",
      enabled: z.enabled,
    });
    setStatus(null);
  };

  const save = async () => {
    if (draft.length < 3) return setStatus({ text: "Draw at least 3 vertices on the canvas.", tone: "err" });
    setBusy(true);
    const body = {
      name: form.name || `Zone ${zones.length + 1}`,
      mode: form.mode,
      categories: form.categories,
      color: form.color,
      polygon: draft,
      maxOccupancy: form.maxOccupancy ? Number(form.maxOccupancy) : null,
      dwellLimitS: form.dwellLimitS ? Number(form.dwellLimitS) : null,
      enabled: form.enabled,
    };
    try {
      const res = await fetch(selectedId ? `/api/zones/${selectedId}` : "/api/zones", {
        method: selectedId ? "PUT" : "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? "save failed");
      await load();
      select(data.zone as ZoneDTO);
      setStatus({ text: `Zone '${data.zone.name}' saved (id ${data.zone.slug}).`, tone: "ok" });
    } catch (err) {
      setStatus({ text: err instanceof Error ? err.message : "save failed", tone: "err" });
    } finally {
      setBusy(false);
    }
  };

  const remove = async () => {
    if (!selectedId || !window.confirm("Delete this zone? Engines syncing from the console will drop it.")) return;
    setBusy(true);
    const res = await fetch(`/api/zones/${selectedId}`, { method: "DELETE" });
    setBusy(false);
    if (res.ok) {
      await load();
      startNew();
      setStatus({ text: "Zone deleted.", tone: "ok" });
    } else setStatus({ text: "Delete failed.", tone: "err" });
  };

  const toggleCategory = (c: Category) =>
    setForm((f) => ({ ...f, categories: f.categories.includes(c) ? f.categories.filter((x) => x !== c) : [...f.categories, c] }));

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_380px]">
      <div className="min-w-0 space-y-4">
        <Panel
          title="ZONE CANVAS"
          right={<Segmented<Background> value={bgMode} options={["grid", "engine", "upload", "camera"] as const} labels={{ grid: "Grid", engine: "Engine snapshot", upload: "Local image", camera: "My camera" }} onChange={setBgMode} />}
          bodyClassName="p-0"
        >
          <canvas
            ref={canvasRef}
            width={W}
            height={H}
            className="block aspect-video w-full cursor-crosshair"
            onClick={onClick}
            onMouseMove={(e) => setHover(toNorm(e))}
            onMouseLeave={() => setHover(null)}
            onContextMenu={(e) => {
              e.preventDefault();
              if (draft.length >= 3) setClosed(true);
            }}
          />
          <div className="flex flex-wrap items-center gap-2 border-t border-hud-border px-3 py-2.5 text-[11px] text-hud-grey">
            <Button onClick={() => setDraft((d) => d.slice(0, -1))} disabled={!draft.length || closed}>
              UNDO VERTEX
            </Button>
            <Button onClick={() => (draft.length >= 3 ? setClosed(true) : undefined)} disabled={draft.length < 3 || closed}>
              CLOSE POLYGON
            </Button>
            <Button
              onClick={() => {
                setDraft([]);
                setClosed(false);
              }}
              disabled={!draft.length}
            >
              REDRAW
            </Button>
            {bgMode === "camera" ? <Button onClick={() => setCaptureKey((k) => k + 1)}>RECAPTURE</Button> : null}
            {bgMode === "upload" ? (
              <label className="cursor-pointer rounded border border-hud-border px-3 py-1.5 hover:border-hud-amber">
                CHOOSE IMAGE
                <input
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (!file) return;
                    const img = new Image();
                    img.onload = () => setBgImage(img);
                    img.src = URL.createObjectURL(file);
                  }}
                />
              </label>
            ) : null}
            <span className="ml-auto">
              {draft.length} VERTICES · {closed ? <span className="text-hud-green">CLOSED</span> : "click to add · click first vertex or right-click to close"}
            </span>
          </div>
        </Panel>
        <p className="text-[11px] leading-relaxed text-hud-muted">
          Membership is evaluated on the <span className="text-hud-amber">bottom-centre</span> of each bounding box (the ground
          contact point) with 3-frame enter / 5-frame exit hysteresis, identical to the Python engine. Local images and camera
          captures stay in your browser and are never uploaded.
        </p>
      </div>

      <aside className="space-y-4">
        <Panel title={selectedId ? "EDIT ZONE" : "NEW ZONE"} right={<Button onClick={startNew}>+ NEW</Button>}>
          <div className="space-y-3">
            <Field label="NAME">
              <input className="hud-input" value={form.name} maxLength={64} placeholder="e.g. Loading Bay" onChange={(e) => setForm({ ...form, name: e.target.value })} />
            </Field>
            <Field label="MODE" hint={form.mode === "allowed" ? "Whitelist: anything not selected is a violation." : "Blacklist: selected categories are violations (none selected = nothing may enter)."}>
              <Segmented<ZoneMode> value={form.mode} options={ZONE_MODES} onChange={(mode) => setForm({ ...form, mode })} />
            </Field>
            <Field label="CATEGORIES">
              <div className="flex flex-wrap gap-1.5">
                {CATEGORIES.map((c) => {
                  const on = form.categories.includes(c);
                  return (
                    <button
                      key={c}
                      type="button"
                      onClick={() => toggleCategory(c)}
                      className="rounded border px-2 py-1 text-[10px] tracking-[0.12em]"
                      style={{ borderColor: on ? CATEGORY_COLOR[c] : PALETTE.border, color: on ? CATEGORY_COLOR[c] : PALETTE.grey, background: on ? hexToRgba(CATEGORY_COLOR[c], 0.1) : "transparent" }}
                    >
                      {c.toUpperCase()}
                    </button>
                  );
                })}
              </div>
            </Field>
            <Field label="COLOUR">
              <div className="flex gap-2">
                {ZONE_COLORS.map((c) => (
                  <button
                    key={c}
                    type="button"
                    aria-label={c}
                    onClick={() => setForm({ ...form, color: c })}
                    className="h-7 w-7 rounded-full border-2"
                    style={{ background: ZONE_COLOR_HEX[c], borderColor: form.color === c ? PALETTE.white : "transparent", boxShadow: form.color === c ? `0 0 0 2px ${PALETTE.bg}` : undefined }}
                  />
                ))}
              </div>
            </Field>
            <div className="grid grid-cols-2 gap-3">
              <Field label="MAX OCCUPANCY" hint="blank = global crowd threshold">
                <input className="hud-input" type="number" min={0} value={form.maxOccupancy} onChange={(e) => setForm({ ...form, maxOccupancy: e.target.value })} />
              </Field>
              <Field label="DWELL LIMIT (S)" hint="blank = default">
                <input className="hud-input" type="number" min={0} value={form.dwellLimitS} onChange={(e) => setForm({ ...form, dwellLimitS: e.target.value })} />
              </Field>
            </div>
            <label className="flex cursor-pointer items-center gap-2 text-[11px] text-hud-grey">
              <input type="checkbox" checked={form.enabled} onChange={(e) => setForm({ ...form, enabled: e.target.checked })} />
              ZONE ENABLED
            </label>
            <div className="flex gap-2 pt-1">
              <Button variant="primary" onClick={save} disabled={busy || draft.length < 3}>
                {selectedId ? "SAVE CHANGES" : "CREATE ZONE"}
              </Button>
              {selectedId ? (
                <Button variant="danger" onClick={remove} disabled={busy}>
                  DELETE
                </Button>
              ) : null}
            </div>
            {status ? <p className={`text-[11px] ${status.tone === "ok" ? "text-hud-green" : "text-hud-red"}`}>{status.text}</p> : null}
          </div>
        </Panel>

        <Panel title={`ZONES (${zones.length})`} bodyClassName="p-0">
          <ul className="divide-y divide-hud-border text-[11px]">
            {zones.length === 0 ? <li className="p-3 text-hud-muted">No zones yet - draw one on the canvas.</li> : null}
            {zones.map((z) => (
              <li key={z.id}>
                <button type="button" onClick={() => select(z)} className={`flex w-full items-center gap-2 px-3 py-2 text-left hover:bg-hud-panel2 ${z.id === selectedId ? "bg-hud-panel2" : ""}`}>
                  <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: ZONE_COLOR_HEX[z.color] }} />
                  <span className="min-w-0 flex-1 truncate text-hud-white">{z.name}</span>
                  <span className="text-hud-grey">{z.mode}</span>
                  <span className="truncate text-hud-muted">{z.categories.join(",") || "-"}</span>
                  {!z.enabled ? <span className="text-hud-red">OFF</span> : null}
                </button>
              </li>
            ))}
          </ul>
        </Panel>

        <Panel title="ENGINE SYNC">
          <div className="space-y-2 text-[11px] text-hud-grey">
            <a href="/api/zones?format=yaml" className="block rounded border border-hud-border px-3 py-2 text-hud-amber hover:border-hud-amber">
              ↓ DOWNLOAD zones.yaml
            </a>
            <a href="/api/zones?format=engine" target="_blank" rel="noreferrer" className="block rounded border border-hud-border px-3 py-2 hover:border-hud-amber">
              GET /api/zones?format=engine (JSON)
            </a>
            <p className="leading-relaxed text-hud-muted">
              Save as <code className="text-hud-white">perception_system/zones.yaml</code> and restart the engine, or use the engine&apos;s own
              OpenCV editor (<code className="text-hud-white">python main.py zones</code>).
            </p>
          </div>
        </Panel>
      </aside>
    </div>
  );
}
