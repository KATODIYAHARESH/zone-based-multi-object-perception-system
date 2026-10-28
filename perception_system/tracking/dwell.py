"""Per-zone presence with hysteresis, dwell time and ENTRY / EXIT events (pure Python).

Boundary jitter is the main source of false zone events: a person standing on a
zone edge flickers in/out every frame. We therefore require ``enter_frames``
consecutive inside observations to emit ENTRY and ``exit_frames`` consecutive
outside observations to emit EXIT. Temporarily lost tracks (occlusion) keep
their presence; tracks that die emit EXIT with reason ``track_lost``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple

from core.types import Category, EventType, Point, ZoneEvent


@dataclass
class _Presence:
    inside: bool = False
    in_streak: int = 0
    out_streak: int = 0
    first_in_ts: Optional[float] = None
    entered_at: Optional[float] = None
    last_inside_ts: Optional[float] = None


@dataclass(frozen=True)
class TrackInfo:
    category: Category
    label: str
    anchor_norm: Optional[Point] = None


class DwellTracker:
    def __init__(self, enter_frames: int = 3, exit_frames: int = 5) -> None:
        if enter_frames < 1 or exit_frames < 1:
            raise ValueError("enter_frames and exit_frames must be >= 1")
        self.enter_frames = enter_frames
        self.exit_frames = exit_frames
        self._state: Dict[Tuple[int, str], _Presence] = {}
        self._info: Dict[int, TrackInfo] = {}
        self._zone_names: Dict[str, str] = {}

    def update(self, timestamp: float, memberships: Mapping[int, Set[str]], infos: Mapping[int, TrackInfo],
               zone_names: Mapping[str, str], alive_ids: Optional[Iterable[int]] = None) -> List[ZoneEvent]:
        """Advance one frame.

        Args:
            memberships: raw zones containing each *visible* track this frame.
            infos: metadata of visible tracks.
            zone_names: all active zones (id -> name).
            alive_ids: tracks that still exist (visible or temporarily lost);
                absent tracks are gone and emit EXIT. Defaults to the visible set.
        """
        events: List[ZoneEvent] = []
        alive = set(alive_ids) if alive_ids is not None else set()
        alive |= set(memberships)
        self._info.update(infos)
        self._zone_names.update(zone_names)
        for tid, inside_set in memberships.items():
            for zid, zname in zone_names.items():
                key = (tid, zid)
                st = self._state.get(key)
                inside_now = zid in inside_set
                if st is None:
                    if not inside_now:
                        continue
                    st = self._state[key] = _Presence()
                if inside_now:
                    st.in_streak += 1
                    st.out_streak = 0
                    st.last_inside_ts = timestamp
                    if st.in_streak == 1:
                        st.first_in_ts = timestamp
                    if not st.inside and st.in_streak >= self.enter_frames:
                        st.inside = True
                        st.entered_at = st.first_in_ts
                        events.append(self._event(EventType.ENTRY, tid, zid, zname, timestamp, 0.0, "enter"))
                elif st.inside:
                    st.out_streak += 1
                    st.in_streak = 0
                    if st.out_streak >= self.exit_frames:
                        events.append(self._event(EventType.EXIT, tid, zid, zname, timestamp, self._dwell(st, timestamp), "exit"))
                        del self._state[key]
                else:
                    del self._state[key]  # aborted entry (boundary jitter)
        for key in list(self._state):
            tid, zid = key
            if tid in alive and zid in zone_names:
                continue
            st = self._state.pop(key)
            if st.inside:
                reason = "track_lost" if tid not in alive else "zone_removed"
                events.append(self._event(EventType.EXIT, tid, zid, self._zone_names.get(zid, zid), timestamp,
                                          self._dwell(st, timestamp), reason))
        for tid in [t for t in self._info if t not in alive]:
            del self._info[tid]
        return events

    @staticmethod
    def _dwell(st: _Presence, now: float) -> float:
        # explicit None checks: a timestamp of 0.0 is valid (falsy-zero bug caught by tests)
        last = st.last_inside_ts if st.last_inside_ts is not None else now
        entered = st.entered_at if st.entered_at is not None else now
        return max(0.0, last - entered)

    def _event(self, etype: EventType, tid: int, zid: str, zname: str, ts: float, dwell: float, reason: str) -> ZoneEvent:
        info = self._info.get(tid) or TrackInfo(Category.UNCLASSIFIED, "unclassified")
        return ZoneEvent(type=etype, zone_id=zid, zone_name=zname, track_id=tid, category=info.category,
                         label=info.label, timestamp=ts, dwell_s=dwell, anchor_norm=info.anchor_norm, reason=reason)

    def dwell_seconds(self, track_id: int, zone_id: str, now: float) -> float:
        st = self._state.get((track_id, zone_id))
        if st is None or not st.inside or st.entered_at is None:
            return 0.0
        return max(0.0, now - st.entered_at)

    def zones_of(self, track_id: int) -> Set[str]:
        """Hysteresis-confirmed zones of a track (stable membership for rules and counts)."""
        return {zid for (tid, zid), st in self._state.items() if tid == track_id and st.inside}

    def occupants(self, zone_id: str) -> Set[int]:
        return {tid for (tid, zid), st in self._state.items() if zid == zone_id and st.inside}
