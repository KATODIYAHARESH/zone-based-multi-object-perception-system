import CameraAnalyzer from "@/components/CameraAnalyzer";

export const metadata = { title: "My Camera // ZBMOPS" };

export default function CameraPage() {
  return (
    <div className="space-y-3">
      <div>
        <h1 className="text-lg font-bold tracking-[0.2em]">ANALYSE ME · MY CAMERA</h1>
        <p className="text-[11px] tracking-[0.1em] text-hud-grey">
          Real-time, on-device perception of your own webcam: person and object detection, persistent tracking, facial expression, age group, head
          orientation, zones and alerts. Nothing leaves your browser unless you opt in to event logging.
        </p>
      </div>
      <CameraAnalyzer />
    </div>
  );
}
