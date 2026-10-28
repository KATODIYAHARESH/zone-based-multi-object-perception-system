"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

const NAV = [
  { href: "/", label: "LIVE" },
  { href: "/camera", label: "MY CAMERA" },
  { href: "/zones", label: "ZONES" },
  { href: "/analytics", label: "ANALYTICS" },
  { href: "/settings", label: "SETTINGS" },
  { href: "/system", label: "SYSTEM" },
];

interface EngineLite {
  source: string;
  online: boolean;
  fps: number | null;
}

export default function TopBar() {
  const pathname = usePathname();
  const [clock, setClock] = useState<string | null>(null);
  const [engines, setEngines] = useState<EngineLite[]>([]);

  useEffect(() => {
    const tick = () => setClock(new Date().toISOString().slice(11, 19));
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const res = await fetch("/api/heartbeat", { cache: "no-store" });
        if (res.ok && alive) setEngines(((await res.json()) as { engines: EngineLite[] }).engines);
      } catch {
        /* console stays usable offline */
      }
    };
    void poll();
    const id = setInterval(poll, 5000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  const engine = engines.find((e) => e.source === "engine");
  const sim = engines.find((e) => e.source === "simulator");
  const cam = engines.find((e) => e.source === "camera");

  return (
    <header className="sticky top-0 z-30 border-b border-hud-border bg-hud-bg/95 backdrop-blur">
      <div className="mx-auto flex w-full max-w-[1600px] flex-wrap items-center gap-x-6 gap-y-2 px-4 py-2.5">
        <Link href="/" className="flex items-center gap-2.5">
          <svg width="26" height="26" viewBox="0 0 26 26" aria-hidden>
            <rect x="3" y="3" width="20" height="20" transform="rotate(45 13 13)" fill="none" stroke="#FFB000" strokeWidth="2" />
            <circle cx="13" cy="13" r="3.2" fill="#FFB000" />
          </svg>
          <div className="leading-tight">
            <div className="text-[13px] font-bold tracking-[0.24em] text-hud-amber">ZBMOPS</div>
            <div className="text-[9px] tracking-[0.2em] text-hud-grey">MISSION CONTROL CONSOLE</div>
          </div>
        </Link>
        <nav className="flex flex-wrap gap-1">
          {NAV.map((item) => {
            const active = item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`rounded px-2.5 py-1 text-[11px] tracking-[0.16em] transition-colors ${
                  active ? "bg-hud-amber/15 text-hud-amber ring-1 ring-hud-amber/50" : "text-hud-grey hover:text-hud-white"
                }`}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>
        <div className="ml-auto flex flex-wrap items-center gap-3 text-[11px] tracking-[0.12em]">
          <span className="flex items-center gap-1.5 text-hud-grey">
            <span className={`h-2 w-2 rounded-full ${engine?.online ? "bg-hud-green" : "bg-hud-muted"}`} />
            ENGINE {engine?.online ? `ONLINE ${engine.fps?.toFixed(0) ?? "-"} FPS` : "OFFLINE"}
          </span>
          {cam?.online ? (
            <span className="flex items-center gap-1.5 text-hud-grey">
              <span className="h-2 w-2 rounded-full bg-hud-red" /> CAM
            </span>
          ) : null}
          {sim?.online ? (
            <span className="flex items-center gap-1.5 text-hud-grey">
              <span className="h-2 w-2 rounded-full bg-hud-amber" /> SIM
            </span>
          ) : null}
          <span className="tabular-nums text-hud-white">{clock ?? "--:--:--"} UTC</span>
          <span className="flex items-center gap-2 rounded border border-hud-red px-2 py-1 font-semibold text-hud-white">
            <span className="blink h-2.5 w-2.5 rounded-full bg-hud-red shadow-[0_0_8px_#FF3B30]" />
            MONITORING ACTIVE
          </span>
        </div>
      </div>
    </header>
  );
}
