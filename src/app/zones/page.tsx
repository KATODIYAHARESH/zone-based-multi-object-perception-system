import ZoneEditor from "@/components/ZoneEditor";

export const metadata = { title: "Zones // ZBMOPS" };

export default function ZonesPage() {
  return (
    <div className="space-y-3">
      <div>
        <h1 className="text-lg font-bold tracking-[0.2em]">ZONE EDITOR</h1>
        <p className="text-[11px] tracking-[0.1em] text-hud-grey">
          Draw polygon zones, set allowed / restricted policies per category, occupancy and dwell limits, then sync them to the engine.
        </p>
      </div>
      <ZoneEditor />
    </div>
  );
}
