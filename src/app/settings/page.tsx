import SettingsForm from "@/components/SettingsForm";

export const metadata = { title: "Settings // ZBMOPS" };

export default function SettingsPage() {
  return (
    <div className="space-y-3">
      <div>
        <h1 className="text-lg font-bold tracking-[0.2em]">SETTINGS</h1>
        <p className="text-[11px] tracking-[0.1em] text-hud-grey">Engine link, alert thresholds, data retention and ingest security.</p>
      </div>
      <SettingsForm />
    </div>
  );
}
