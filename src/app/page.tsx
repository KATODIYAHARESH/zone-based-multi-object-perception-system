import LiveConsole from "@/components/LiveConsole";

export default function HomePage() {
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <h1 className="text-lg font-bold tracking-[0.2em] text-hud-white">LIVE OPERATIONS</h1>
          <p className="text-[11px] tracking-[0.1em] text-hud-grey">
            Zone occupancy, labelled tracks, expression meter and alerts. The simulator runs the engine&apos;s zone logic
            on synthetic tracks; switch to <span className="text-hud-amber">Engine stream</span> for the real Python
            perception feed.
          </p>
        </div>
      </div>
      <LiveConsole />
    </div>
  );
}
