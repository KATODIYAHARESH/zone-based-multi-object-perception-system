export const CATEGORIES = ["human", "animal", "vehicle", "food", "object", "unclassified"] as const;
export type Category = (typeof CATEGORIES)[number];
export const EMOTIONS = ["happy", "sad", "angry", "neutral", "surprise", "fear", "disgust"] as const;
export type Emotion = (typeof EMOTIONS)[number];
export const ZONE_COLORS = ["amber", "green", "red", "white"] as const;
export type ZoneColor = (typeof ZONE_COLORS)[number];
export const ZONE_MODES = ["allowed", "restricted"] as const;
export type ZoneMode = (typeof ZONE_MODES)[number];
export const SEVERITIES = ["info", "warning", "critical"] as const;
export type Severity = (typeof SEVERITIES)[number];
export const EVENT_TYPES = ["entry", "exit", "alert"] as const;
export type EventType = (typeof EVENT_TYPES)[number];
export type Point = [number, number];

export interface ZoneDTO {
  id: number;
  slug: string;
  name: string;
  mode: ZoneMode;
  categories: Category[];
  polygon: Point[];
  color: ZoneColor;
  maxOccupancy: number | null;
  dwellLimitS: number | null;
  enabled: boolean;
}

export interface EventDTO {
  id: number;
  ts: string;
  source: string;
  type: EventType;
  severity: Severity | null;
  ruleId: string | null;
  zoneId: string | null;
  zoneName: string | null;
  trackId: number | null;
  category: string | null;
  label: string | null;
  emotion: string | null;
  ageGroup: string | null;
  identity: string | null;
  message: string | null;
  dwellS: number | null;
  x: number | null;
  y: number | null;
  meta: Record<string, unknown> | null;
}

/** Shape accepted by POST /api/events (engine webhook + simulator). */
export interface IngestEvent {
  type: EventType;
  ts?: string;
  severity?: Severity;
  ruleId?: string;
  zoneId?: string;
  zoneName?: string;
  trackId?: number;
  category?: string;
  label?: string;
  emotion?: string;
  ageGroup?: string;
  identity?: string;
  message?: string;
  dwellS?: number;
  x?: number;
  y?: number;
  meta?: Record<string, unknown>;
}

export interface SettingsData {
  siteName: string;
  engineStreamUrl: string;
  engineStateUrl: string;
  retentionDays: number;
  blurFaces: boolean;
  crowdThreshold: number;
  angryMinConfidence: number;
  angryMinDurationS: number;
  defaultDwellLimitS: number;
  alertCooldownS: number;
  detectionConfidence: number;
  nmsIou: number;
  allowSimulatorIngest: boolean;
}

export const DEFAULT_SETTINGS: SettingsData = {
  siteName: "default-site",
  engineStreamUrl: "http://localhost:8765/stream.mjpg",
  engineStateUrl: "http://localhost:8765/api/state",
  retentionDays: 30,
  blurFaces: false,
  crowdThreshold: 6,
  angryMinConfidence: 0.6,
  angryMinDurationS: 1.5,
  defaultDwellLimitS: 45,
  alertCooldownS: 20,
  detectionConfidence: 0.35,
  nmsIou: 0.5,
  allowSimulatorIngest: true,
};

export interface EngineStatusDTO {
  source: string;
  ts: string;
  fps: number | null;
  latencyMs: number | null;
  device: string | null;
  frameSkip: number | null;
  tracks: number | null;
  counts: Record<string, Record<string, number>> | null;
  emotions: Record<string, number> | null;
  online: boolean;
}

/** House palette: amber / green / red / white accents only - no blue anywhere. */
export const PALETTE = {
  amber: "#FFB000",
  amberDim: "#8A5E00",
  green: "#3DDC5F",
  greenSoft: "#9EE37D",
  red: "#FF3B30",
  redDim: "#6E1E1A",
  white: "#F2F2F2",
  grey: "#9A968A",
  bg: "#0A0B0A",
  panel: "#121411",
  border: "#2A2922",
} as const;

export const CATEGORY_COLOR: Record<Category, string> = {
  human: PALETTE.white,
  animal: PALETTE.green,
  vehicle: PALETTE.amber,
  food: PALETTE.greenSoft,
  object: PALETTE.grey,
  unclassified: PALETTE.red,
};
export const SEVERITY_COLOR: Record<Severity, string> = { info: PALETTE.white, warning: PALETTE.amber, critical: PALETTE.red };
export const ZONE_COLOR_HEX: Record<ZoneColor, string> = {
  amber: PALETTE.amber,
  green: PALETTE.green,
  red: PALETTE.red,
  white: PALETTE.white,
};
export const EMOTION_COLOR: Record<Emotion, string> = {
  happy: PALETTE.green,
  neutral: PALETTE.white,
  surprise: PALETTE.amber,
  sad: PALETTE.grey,
  fear: PALETTE.amberDim,
  angry: PALETTE.red,
  disgust: "#B8322A",
};

export function hexToRgba(hex: string, alpha: number): string {
  const h = hex.replace("#", "");
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

export function isCategory(v: unknown): v is Category {
  return typeof v === "string" && (CATEGORIES as readonly string[]).includes(v);
}
