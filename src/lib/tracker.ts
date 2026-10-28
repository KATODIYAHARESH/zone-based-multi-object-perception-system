/**
 * ByteTrack-lite: browser port of perception_system/tracking/byte_tracker.py.
 * Two-stage association (low-score detections only continue existing tracks),
 * class-aware gating, constant-velocity prediction, confidence-weighted label
 * voting with a consensus threshold ("unclassified" instead of guessing) and
 * contiguous public IDs issued only once a track is confirmed.
 */
import type { Category } from "./types";
import type { Box } from "./zoneRuntime";

export interface TrackerDetection {
  bbox: Box;
  score: number;
  label: string;
  category: Category;
}

export interface TrackedObject {
  id: number;
  bbox: Box;
  score: number;
  label: string;
  category: Category;
  hits: number;
  coasting: boolean;
  firstSeen: number;
  lastSeen: number;
}

export interface TrackerOptions {
  highThresh: number;
  lowThresh: number;
  newTrackThresh: number;
  matchIou: number;
  lowMatchIou: number;
  minHits: number;
  maxLost: number;
  showLostFrames: number;
  classAware: boolean;
  labelConsensus: number;
  voteDecay: number;
  smoothing: number;
}

export const DEFAULT_TRACKER_OPTIONS: TrackerOptions = {
  highThresh: 0.35,
  lowThresh: 0.2,
  newTrackThresh: 0.4,
  matchIou: 0.2,
  lowMatchIou: 0.35,
  minHits: 3,
  maxLost: 45,
  showLostFrames: 6,
  classAware: true,
  labelConsensus: 0.6,
  voteDecay: 0.9,
  smoothing: 0.6,
};

type State = "tentative" | "confirmed" | "lost" | "removed";

interface Vote {
  label: string;
  category: Category;
  w: number;
}

interface Internal {
  id: number;
  box: Box;
  cx: number;
  cy: number;
  vx: number;
  vy: number;
  since: number;
  score: number;
  state: State;
  hits: number;
  lost: number;
  category: Category;
  votes: Map<string, Vote>;
  firstSeen: number;
  lastSeen: number;
}

export function iou(a: Box, b: Box): number {
  const iw = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
  const ih = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  if (iw <= 0 || ih <= 0) return 0;
  const inter = iw * ih;
  const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return union > 0 ? inter / union : 0;
}

/** Fraction of box `a` covered by box `b`. */
export function intersectionOverArea(a: Box, b: Box): number {
  const iw = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
  const ih = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  const area = (a[2] - a[0]) * (a[3] - a[1]);
  return iw > 0 && ih > 0 && area > 0 ? (iw * ih) / area : 0;
}

export class ByteTrackLite {
  private tracks: Internal[] = [];
  private nextId = 1;

  constructor(public opts: TrackerOptions = { ...DEFAULT_TRACKER_OPTIONS }) {}

  reset(): void {
    this.tracks = [];
    this.nextId = 1;
  }

  private predict(t: Internal): void {
    t.box = [t.box[0] + t.vx, t.box[1] + t.vy, t.box[2] + t.vx, t.box[3] + t.vy];
    t.since += 1;
    if (t.state === "lost") {
      t.vx *= 0.8;
      t.vy *= 0.8;
    }
  }

  private associate(tracks: Internal[], dets: TrackerDetection[], minIou: number) {
    const pairs: [number, number, number][] = [];
    for (let i = 0; i < tracks.length; i++) {
      for (let j = 0; j < dets.length; j++) {
        if (this.opts.classAware && tracks[i].category !== dets[j].category) continue;
        const v = iou(tracks[i].box, dets[j].bbox);
        if (v >= minIou) pairs.push([v, i, j]);
      }
    }
    pairs.sort((a, b) => b[0] - a[0]);
    const freeT = new Set(tracks.map((_, i) => i));
    const freeD = new Set(dets.map((_, j) => j));
    const matches: [number, number][] = [];
    for (const [, i, j] of pairs) {
      if (freeT.has(i) && freeD.has(j)) {
        freeT.delete(i);
        freeD.delete(j);
        matches.push([i, j]);
      }
    }
    return { matches, unmatchedTracks: Array.from(freeT), unmatchedDets: Array.from(freeD) };
  }

  private vote(t: Internal, d: TrackerDetection): void {
    for (const v of t.votes.values()) v.w *= this.opts.voteDecay;
    const key = `${d.category}|${d.label}`;
    const cur = t.votes.get(key);
    if (cur) cur.w += d.score;
    else t.votes.set(key, { label: d.label, category: d.category, w: d.score });
  }

  private consensus(t: Internal): { label: string; category: Category } {
    let best: Vote | null = null;
    let total = 0;
    for (const v of t.votes.values()) {
      total += v.w;
      if (!best || v.w > best.w) best = v;
    }
    if (!best || best.category === "unclassified" || total <= 0 || best.w / total < this.opts.labelConsensus)
      return { label: "unclassified", category: "unclassified" };
    return { label: best.label, category: best.category };
  }

  private correct(t: Internal, d: TrackerDetection, now: number, vote: boolean): void {
    const a = this.opts.smoothing;
    const b = t.box;
    t.box = [b[0] + a * (d.bbox[0] - b[0]), b[1] + a * (d.bbox[1] - b[1]), b[2] + a * (d.bbox[2] - b[2]), b[3] + a * (d.bbox[3] - b[3])];
    const cx = (t.box[0] + t.box[2]) / 2;
    const cy = (t.box[1] + t.box[3]) / 2;
    const steps = Math.max(1, t.since);
    const maxV = Math.max(4, (t.box[2] - t.box[0]) * 0.5);
    t.vx = Math.max(-maxV, Math.min(maxV, 0.5 * t.vx + 0.5 * ((cx - t.cx) / steps)));
    t.vy = Math.max(-maxV, Math.min(maxV, 0.5 * t.vy + 0.5 * ((cy - t.cy) / steps)));
    t.cx = cx;
    t.cy = cy;
    t.since = 0;
    t.score = d.score;
    t.hits += 1;
    t.lost = 0;
    t.lastSeen = now;
    if (vote) this.vote(t, d);
    if (t.state === "lost") t.state = "confirmed";
    else if (t.state === "tentative" && t.hits >= this.opts.minHits) t.state = "confirmed";
  }

  private missed(t: Internal): void {
    t.lost += 1;
    if (t.state === "confirmed") t.state = "lost";
    else if (t.state === "tentative") t.state = "removed";
  }

  private birth(d: TrackerDetection, now: number): void {
    const t: Internal = {
      id: -1,
      box: [d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]],
      cx: (d.bbox[0] + d.bbox[2]) / 2,
      cy: (d.bbox[1] + d.bbox[3]) / 2,
      vx: 0,
      vy: 0,
      since: 0,
      score: d.score,
      state: this.opts.minHits <= 1 ? "confirmed" : "tentative",
      hits: 1,
      lost: 0,
      category: d.category,
      votes: new Map(),
      firstSeen: now,
      lastSeen: now,
    };
    this.vote(t, d);
    this.tracks.push(t);
  }

  /** Associate one frame of detections; returns visible confirmed tracks. */
  update(dets: TrackerDetection[], now: number): TrackedObject[] {
    const o = this.opts;
    for (const t of this.tracks) this.predict(t);
    const high = dets.filter((d) => d.score >= o.highThresh);
    const low = dets.filter((d) => d.score >= o.lowThresh && d.score < o.highThresh);
    const pool = this.tracks.filter((t) => t.state === "confirmed" || t.state === "lost");
    const tentative = this.tracks.filter((t) => t.state === "tentative");
    // Stage 1: confirmed + lost tracks <-> high-score detections.
    const s1 = this.associate(pool, high, o.matchIou);
    for (const [i, j] of s1.matches) this.correct(pool[i], high[j], now, true);
    // Stage 2: still-unmatched confirmed tracks <-> low-score detections (no label vote).
    const left = s1.unmatchedTracks.map((i) => pool[i]);
    const leftConfirmed = left.filter((t) => t.state === "confirmed");
    const s2 = this.associate(leftConfirmed, low, o.lowMatchIou);
    const rescued = new Set<Internal>();
    for (const [i, j] of s2.matches) {
      this.correct(leftConfirmed[i], low[j], now, false);
      rescued.add(leftConfirmed[i]);
    }
    for (const t of left) if (!rescued.has(t)) this.missed(t);
    // Stage 3: tentative tracks <-> remaining high-score detections, then births.
    const rest = s1.unmatchedDets.map((j) => high[j]);
    const s3 = this.associate(tentative, rest, 0.3);
    for (const [i, j] of s3.matches) this.correct(tentative[i], rest[j], now, true);
    for (const i of s3.unmatchedTracks) this.missed(tentative[i]);
    for (const j of s3.unmatchedDets) if (rest[j].score >= o.newTrackThresh) this.birth(rest[j], now);
    for (const t of this.tracks) {
      if (t.state === "lost" && t.lost > o.maxLost) t.state = "removed";
      if (t.state === "confirmed" && t.id < 0) t.id = this.nextId++;
    }
    this.tracks = this.tracks.filter((t) => t.state !== "removed");
    return this.visible(false);
  }

  /** Advance motion on a frame where detection was skipped (no miss penalty). */
  coast(): TrackedObject[] {
    for (const t of this.tracks) this.predict(t);
    return this.visible(true);
  }

  aliveIds(): Set<number> {
    return new Set(this.tracks.filter((t) => t.id > 0 && (t.state === "confirmed" || t.state === "lost")).map((t) => t.id));
  }

  private visible(coasting: boolean): TrackedObject[] {
    const out: TrackedObject[] = [];
    for (const t of this.tracks) {
      if (t.id < 0) continue;
      if (t.state === "confirmed" || (t.state === "lost" && t.lost <= this.opts.showLostFrames)) {
        const { label, category } = this.consensus(t);
        out.push({
          id: t.id,
          bbox: [t.box[0], t.box[1], t.box[2], t.box[3]],
          score: t.score,
          label,
          category,
          hits: t.hits,
          coasting: coasting || t.state === "lost",
          firstSeen: t.firstSeen,
          lastSeen: t.lastSeen,
        });
      }
    }
    return out;
  }
}
