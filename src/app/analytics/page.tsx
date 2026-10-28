import AnalyticsView from "@/components/AnalyticsView";

export const metadata = { title: "Analytics // ZBMOPS" };

export default function AnalyticsPage() {
  return (
    <div className="space-y-3">
      <div>
        <h1 className="text-lg font-bold tracking-[0.2em]">ANALYTICS</h1>
        <p className="text-[11px] tracking-[0.1em] text-hud-grey">Per-zone traffic, timelines, alert statistics and heatmaps from PostgreSQL. Export CSV / JSON for offline analysis.</p>
      </div>
      <AnalyticsView />
    </div>
  );
}
