"use client";

import { useEffect, useRef } from "react";
import { SIM_H, SIM_W, SimWorld, type SimAlert, type SimStats } from "@/lib/sim";
import { drawScene } from "@/lib/simRender";
import type { IngestEvent, SettingsData, ZoneDTO } from "@/lib/types";

export interface SimSnapshot extends SimStats {
  recentAlerts: SimAlert[];
  posted: number;
}

interface Props {
  zones: ZoneDTO[];
  settings: SettingsData;
  running: boolean;
  spawnRate: number;
  blurFaces: boolean;
  onStats: (s: SimSnapshot) => void;
}

export default function Simulator({ zones, settings, running, spawnRate, blurFaces, onStats }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const worldRef = useRef<SimWorld | null>(null);
  const queueRef = useRef<IngestEvent[]>([]);
  const postedRef = useRef(0);
  const metricsRef = useRef({ fps: 60, latency: 0 });
  const propsRef = useRef({ running, spawnRate, blurFaces, onStats });

  useEffect(() => {
    propsRef.current = { running, spawnRate, blurFaces, onStats };
  }, [running, spawnRate, blurFaces, onStats]);

  useEffect(() => {
    if (!worldRef.current) worldRef.current = new SimWorld(zones, settings);
    else {
      worldRef.current.zones = zones;
      worldRef.current.settings = settings;
    }
  }, [zones, settings]);

  useEffect(() => {
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx) return;
    let raf = 0;
    let last = performance.now();
    let statsAt = 0;
    const loop = (now: number) => {
      raf = requestAnimationFrame(loop);
      const world = worldRef.current;
      if (!world) return;
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      const t0 = performance.now();
      const p = propsRef.current;
      if (p.running) {
        const evs = world.step(dt, p.spawnRate);
        if (evs.length) queueRef.current.push(...evs);
      }
      const m = metricsRef.current;
      const stats = world.stats(m.fps, m.latency);
      drawScene(ctx, world, { blurFaces: p.blurFaces, fps: m.fps, latencyMs: m.latency, counts: stats.counts });
      m.latency = m.latency * 0.9 + (performance.now() - t0) * 0.1;
      if (dt > 0) m.fps = m.fps * 0.95 + (1 / dt) * 0.05;
      if (now - statsAt > 500) {
        statsAt = now;
        p.onStats({ ...stats, recentAlerts: world.recentAlerts.slice(0, 8), posted: postedRef.current });
      }
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, []);

  useEffect(() => {
    const flush = async () => {
      const batch = queueRef.current.splice(0, 400);
      if (!batch.length) return;
      try {
        const res = await fetch("/api/events", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ source: "simulator", events: batch }),
        });
        if (res.ok) postedRef.current += ((await res.json()) as { accepted: number }).accepted;
      } catch {
        /* synthetic data: a dropped batch is acceptable */
      }
    };
    const heartbeat = async () => {
      const world = worldRef.current;
      if (!world || !propsRef.current.running) return;
      const s = world.stats(metricsRef.current.fps, metricsRef.current.latency);
      await fetch("/api/heartbeat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source: "simulator",
          fps: Math.round(s.fps * 10) / 10,
          latencyMs: Math.round(s.latencyMs * 10) / 10,
          device: "BROWSER SIMULATOR",
          frameSkip: 1,
          tracks: s.tracks,
          counts: s.counts,
          emotions: s.emotions,
        }),
      }).catch(() => undefined);
    };
    const i1 = setInterval(flush, 1500);
    const i2 = setInterval(heartbeat, 3000);
    return () => {
      clearInterval(i1);
      clearInterval(i2);
      void flush();
    };
  }, []);

  return <canvas ref={canvasRef} width={SIM_W} height={SIM_H} className="block h-full w-full" aria-label="Simulated perception feed" />;
}
