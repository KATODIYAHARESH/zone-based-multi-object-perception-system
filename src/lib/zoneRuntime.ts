/**
 * Shared zone runtime: the browser twin of perception_system/tracking/dwell.py
 * plus the default alert rules from perception_system/config.yaml.
 *
 * Used by BOTH the synthetic simulator and the live "My Camera" analyser, so the
 * zone semantics can never drift between them:
 *   - anchor point: bottom-centre of the box (ground contact). "auto" falls back to
 *     the box centre when the box touches the top edge or ends within 6% of the
 *     bottom edge: the ground-contact point is then not observable (desk webcams,
 *     close-ups, lower body hidden behind a desk).
 *   - hysteresis: 3 consecutive frames inside -> ENTRY, 5 outside -> EXIT.
 *   - temporarily lost tracks keep their presence; dead tracks emit EXIT.
 *   - per-key cooldowns: one alert per (rule, zone, track) per window.
 * A local identity suppresses "unknown person" alerts but is never written into
 * events or alert messages, so names never leave the browser.
 */
import { pointInPolygon } from "./geometry";
import { CATEGORIES, type Category, type IngestEvent, type Point, type SettingsData, type Severity, type ZoneDTO } from "./types";

export const ENTER_FRAMES = 3;
export const EXIT_FRAMES = 5;
const UNATTENDED_AFTER_S = 20;

export type Box = [number, number, number, number];
export type AnchorMode = "bottom_center" | "center" | "auto";

export interface RuntimeTrack {
  id: number;
  category: Category;
  label: string;
  bbox: Box;
  emotion?: string | null;
  emotionConf?: number;
  /** seconds, same clock as the `t` passed to update() */
  emotionSince?: number | null;
  ageGroup?: string | null;
  /** local-only identity (self-enrollment); never emitted */
  identity?: string | null;
}

export interface RuntimeAlert {
  ts: number;
  severity: Severity;
  ruleId: string;
  message: string;
}

export interface RuntimeResult {
  events: IngestEvent[];
  alerts: RuntimeAlert[];
}

interface Presence {
  inside: boolean;
  inStreak: number;
  outStreak: number;
  firstIn: number;
  enteredAt: number | null;
  lastIn: number;
}

export function anchorPoint(b: Box, w: number, h: number, mode: AnchorMode): Point {
  const [x1, y1, x2, y2] = b;
  const clipped = y2 >= h * 0.94 || y1 <= h * 0.01;
  const useCenter = mode === "center" || (mode === "auto" && clipped);
  return [(x1 + x2) / 2 / w, (useCenter ? (y1 + y2) / 2 : y2) / h];
}

export function isViolation(z: ZoneDTO, category: Category): boolean {
  return z.mode === "allowed" ? !z.categories.includes(category) : z.categories.length === 0 || z.categories.includes(category);
}

const round2 = (v: number) => Math.round(v * 100) / 100;

export class ZoneRuntime {
  memberships = new Map<number, Set<string>>();
  violated = new Set<string>();
  private presence = new Map<string, Presence>();
  private lastFired = new Map<string, number>();
  private info = new Map<number, RuntimeTrack>();
  private updates = 0;

  constructor(
    public zones: ZoneDTO[],
    public settings: SettingsData,
    public width: number,
    public height: number,
    public anchorMode: AnchorMode = "bottom_center",
    private meta: Record<string, unknown> = {},
  ) {}

  private active(): ZoneDTO[] {
    return this.zones.filter((z) => z.enabled && z.polygon.length >= 3);
  }

  zoneName(slug: string): string {
    return this.zones.find((z) => z.slug === slug)?.name ?? slug;
  }

  dwell(trackId: number, slug: string, t: number): number {
    const st = this.presence.get(`${trackId}:${slug}`);
    return st && st.inside && st.enteredAt !== null ? Math.max(0, t - st.enteredAt) : 0;
  }

  /**
   * Advance one frame.
   * @param t        monotonic time in seconds
   * @param tracks   tracks visible this frame
   * @param aliveIds tracks that still exist (visible or temporarily lost); defaults to the visible set
   */
  update(t: number, tracks: RuntimeTrack[], aliveIds?: Iterable<number>): RuntimeResult {
    const events: IngestEvent[] = [];
    const alerts: RuntimeAlert[] = [];
    const zones = this.active();
    const slugs = new Set(zones.map((z) => z.slug));
    const alive = new Set<number>(aliveIds ?? []);
    for (const tr of tracks) {
      alive.add(tr.id);
      this.info.set(tr.id, tr);
    }
    const memberships = new Map<number, Set<string>>();
    for (const tr of tracks) {
      const anchor = anchorPoint(tr.bbox, this.width, this.height, this.anchorMode);
      const confirmed = new Set<string>();
      for (const z of zones) {
        const key = `${tr.id}:${z.slug}`;
        const inside = pointInPolygon(anchor, z.polygon);
        let st = this.presence.get(key);
        if (!st) {
          if (!inside) continue;
          st = { inside: false, inStreak: 0, outStreak: 0, firstIn: t, enteredAt: null, lastIn: t };
          this.presence.set(key, st);
        }
        if (inside) {
          st.inStreak += 1;
          st.outStreak = 0;
          st.lastIn = t;
          if (st.inStreak === 1) st.firstIn = t;
          if (!st.inside && st.inStreak >= ENTER_FRAMES) {
            st.inside = true;
            st.enteredAt = st.firstIn;
            events.push(this.zoneEvent("entry", tr, z, 0, anchor));
            this.onEntry(t, tr, z, events, alerts);
          }
        } else if (st.inside) {
          st.outStreak += 1;
          st.inStreak = 0;
          if (st.outStreak >= EXIT_FRAMES) {
            events.push(this.zoneEvent("exit", tr, z, st.lastIn - (st.enteredAt ?? st.lastIn), anchor));
            this.presence.delete(key);
          }
        } else {
          this.presence.delete(key); // aborted entry (boundary jitter)
        }
        if (this.presence.get(key)?.inside) confirmed.add(z.slug);
      }
      memberships.set(tr.id, confirmed);
    }
    for (const [key, st] of Array.from(this.presence.entries())) {
      const sep = key.indexOf(":");
      const id = Number(key.slice(0, sep));
      const slug = key.slice(sep + 1);
      if (alive.has(id) && slugs.has(slug)) continue;
      this.presence.delete(key);
      const tr = this.info.get(id);
      const zone = this.zones.find((z) => z.slug === slug);
      if (st.inside && tr && zone)
        events.push(this.zoneEvent("exit", tr, zone, st.lastIn - (st.enteredAt ?? st.lastIn), null, alive.has(id) ? "zone_removed" : "track_lost"));
    }
    for (const id of Array.from(this.info.keys())) if (!alive.has(id)) this.info.delete(id);
    this.memberships = memberships;
    this.evaluate(t, zones, tracks, events, alerts);
    if (++this.updates % 600 === 0) {
      for (const [key, ts] of Array.from(this.lastFired.entries())) if (t - ts > 900) this.lastFired.delete(key);
    }
    return { events, alerts };
  }

  counts(tracks: Pick<RuntimeTrack, "id" | "category">[]): Record<string, Record<string, number>> {
    const out: Record<string, Record<string, number>> = {};
    for (const z of this.active()) out[z.slug] = Object.fromEntries(CATEGORIES.map((c) => [c, 0]));
    for (const tr of tracks) for (const slug of this.memberships.get(tr.id) ?? []) if (out[slug]) out[slug][tr.category] += 1;
    return out;
  }

  private zoneEvent(type: "entry" | "exit", tr: RuntimeTrack, z: ZoneDTO, dwell: number, anchor: Point | null, reason?: string): IngestEvent {
    const human = tr.category === "human";
    return {
      type,
      ts: new Date().toISOString(),
      zoneId: z.slug,
      zoneName: z.name,
      trackId: tr.id,
      category: tr.category,
      label: tr.label,
      emotion: human && tr.emotion ? tr.emotion : undefined,
      ageGroup: human && tr.ageGroup ? tr.ageGroup : undefined,
      dwellS: round2(dwell),
      x: anchor ? anchor[0] : undefined,
      y: anchor ? anchor[1] : undefined,
      meta: reason ? { ...this.meta, reason } : { ...this.meta },
    };
  }

  private alert(ruleId: string, severity: Severity, message: string, z: ZoneDTO, tr?: RuntimeTrack, extra: Record<string, unknown> = {}): IngestEvent {
    const anchor = tr ? anchorPoint(tr.bbox, this.width, this.height, this.anchorMode) : null;
    return {
      type: "alert",
      ts: new Date().toISOString(),
      severity,
      ruleId,
      zoneId: z.slug,
      zoneName: z.name,
      trackId: tr?.id,
      category: tr?.category,
      label: tr?.label,
      emotion: tr && tr.category === "human" && tr.emotion ? tr.emotion : undefined,
      message,
      x: anchor ? anchor[0] : undefined,
      y: anchor ? anchor[1] : undefined,
      meta: { ...this.meta, ...extra },
    };
  }

  private fire(t: number, key: string, cooldown: number, ev: IngestEvent, events: IngestEvent[], alerts: RuntimeAlert[]): void {
    const last = this.lastFired.get(key);
    if (last !== undefined && t - last < cooldown) return;
    this.lastFired.set(key, t);
    events.push(ev);
    alerts.push({ ts: Date.now(), severity: ev.severity ?? "info", ruleId: ev.ruleId ?? "rule", message: ev.message ?? "" });
  }

  private onEntry(t: number, tr: RuntimeTrack, z: ZoneDTO, events: IngestEvent[], alerts: RuntimeAlert[]): void {
    if (tr.category === "animal" && z.mode === "allowed" && z.categories.includes("animal"))
      this.fire(t, `animal:${z.slug}:${tr.id}`, this.settings.alertCooldownS, this.alert("animal_entered", "warning", `Animal entered ${z.name}: ${tr.label} #${tr.id}`, z, tr), events, alerts);
  }

  private evaluate(t: number, zones: ZoneDTO[], tracks: RuntimeTrack[], events: IngestEvent[], alerts: RuntimeAlert[]): void {
    const s = this.settings;
    const cd = s.alertCooldownS;
    this.violated = new Set<string>();
    for (const z of zones) {
      const inZone = tracks.filter((tr) => this.memberships.get(tr.id)?.has(z.slug));
      for (const tr of inZone) {
        const cat = tr.category;
        if (isViolation(z, cat)) {
          this.violated.add(z.slug);
          const severity: Severity = z.mode === "restricted" && cat === "vehicle" ? "critical" : "warning";
          this.fire(t, `policy:${z.slug}:${tr.id}`, Math.max(cd, 30), this.alert("zone_policy", severity, `Zone policy violation: ${cat} '${tr.label}' #${tr.id} in ${z.mode} zone ${z.name}`, z, tr), events, alerts);
        }
        const d = this.dwell(tr.id, z.slug, t);
        if (cat === "human" && !tr.identity && z.slug === "zone_a" && d >= 2)
          this.fire(t, `unknown:${z.slug}:${tr.id}`, cd * 3, this.alert("unknown_person_zone_a", "warning", `Unknown person (Person #${tr.id}) in ${z.name}`, z, tr), events, alerts);
        if (cat === "human" && tr.emotion === "angry" && (tr.emotionConf ?? 0) >= s.angryMinConfidence && tr.emotionSince != null && t - tr.emotionSince >= s.angryMinDurationS)
          this.fire(t, `angry:${z.slug}:${tr.id}`, cd, this.alert("angry_expression", "info", `Angry expression detected: Person #${tr.id} in ${z.name}`, z, tr, { confidence: round2(tr.emotionConf ?? 0) }), events, alerts);
        const limit = z.dwellLimitS ?? s.defaultDwellLimitS;
        if (cat === "human" && d > limit)
          this.fire(t, `dwell:${z.slug}:${tr.id}`, cd * 3, this.alert("loitering", "warning", `Loitering: Person #${tr.id} in ${z.name} for ${Math.round(d)}s (limit ${limit}s)`, z, tr), events, alerts);
        if ((cat === "unclassified" || cat === "object") && d > UNATTENDED_AFTER_S)
          this.fire(t, `unattended:${z.slug}:${tr.id}`, 120, this.alert("unattended_object", "warning", `Unattended/unclassified object in ${z.name} for ${Math.round(d)}s`, z, tr), events, alerts);
      }
      const crowdLimit = z.maxOccupancy ?? s.crowdThreshold;
      const humans = inZone.filter((tr) => tr.category === "human").length;
      if (crowdLimit > 0 && humans > crowdLimit)
        this.fire(t, `crowd:${z.slug}`, cd, this.alert("crowd_limit", "warning", `Crowd in ${z.name}: ${humans} > ${crowdLimit}`, z, undefined, { count: humans, limit: crowdLimit }), events, alerts);
    }
  }
}
