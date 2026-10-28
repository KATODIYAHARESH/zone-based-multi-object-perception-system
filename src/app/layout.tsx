import type { Metadata } from "next";
import type { ReactNode } from "react";
import TopBar from "@/components/TopBar";
import "./globals.css";

export const metadata: Metadata = {
  title: "ZBMOPS // Mission Control",
  description: "Zone-Based Multi-Object Perception System - web console for zones, live telemetry, alerts and analytics.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="hud-grid-bg min-h-screen bg-hud-bg text-hud-white antialiased">
        <TopBar />
        <main className="mx-auto w-full max-w-[1600px] px-4 pb-12 pt-4">{children}</main>
        <footer className="border-t border-hud-border px-4 py-3 text-center text-[10px] tracking-[0.14em] text-hud-muted">
          VIDEO ANALYTICS IN USE · FACE RECOGNITION ONLY FOR ENROLLED, CONSENTING PEOPLE · EVERYONE ELSE IS SHOWN AS
          &quot;PERSON #ID&quot; · DATA AUTO-DELETED AFTER THE CONFIGURED RETENTION PERIOD
        </footer>
      </body>
    </html>
  );
}
