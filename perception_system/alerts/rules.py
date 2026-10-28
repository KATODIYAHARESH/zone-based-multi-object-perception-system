"""Declarative alert rules (pure Python - unit-tested without OpenCV / NumPy).

Each rule receives a :class:`RuleContext` snapshot of the current frame and
returns zero or more :class:`AlertCandidate`. De-duplication and cooldowns
live centrally in :class:`alerts.engine.AlertEngine`, so rules stay stateless.
Temporal conditions (dwell, expression duration) are derived from data the
context already carries.

Rule spec (config.yaml -> alerts.rules)::

    {id: vehicle_in_walkway, type: category_in_zone, zones: [walkway],
     categories: [vehicle], severity: critical, cooldown_s: 30,
     message: "{Category} in {zone}: {label} #{track_id}"}
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Set, Tuple, Type

from core.types import EMOTIONS, Category, EventType, Severity, ZoneEvent


class ZoneLike(Protocol):
    id: str
    name: str
    max_occupancy: Optional[int]
    dwell_limit_s: Optional[float]

    def is_violation(self, category: Any) -> bool: ...


def _value(x: Any) -> str:
    """Enum-safe string conversion (avoids 'Category.HUMAN' on Python >= 3.12)."""
    return str(getattr(x, "value", x))


@dataclass(frozen=True)
class TrackView:
    """The minimal, immutable view of a track that rules may inspect."""

    track_id: int
    category: str
    label: str
    confidence: float = 1.0
    identity: Optional[str] = None      # only ever set for enrolled + consenting people
    emotion: Optional[str] = None
    emotion_conf: float = 0.0
    emotion_since: Optional[float] = None

    @classmethod
    def from_track(cls, t: Any) -> "TrackView":
        face = getattr(t, "face", None)
        return cls(track_id=int(t.track_id), category=_value(t.category), label=str(t.label),
                   confidence=float(getattr(t, "confidence", 1.0)), identity=getattr(face, "identity", None),
                   emotion=getattr(face, "emotion", None), emotion_conf=float(getattr(face, "emotion_conf", 0.0) or 0.0),
                   emotion_since=getattr(face, "emotion_since", None))


@dataclass
class RuleContext:
    timestamp: float
    tracks: Sequence[TrackView]
    memberships: Mapping[int, Set[str]]
    zones: Mapping[str, ZoneLike]
    events: Sequence[ZoneEvent] = ()
    dwell: Callable[[int, str], float] = lambda _tid, _zid: 0.0

    def tracks_in(self, zone_id: str) -> List[TrackView]:
        return [t for t in self.tracks if zone_id in self.memberships.get(t.track_id, ())]


@dataclass(frozen=True)
class AlertCandidate:
    rule_id: str
    rule_type: str
    severity: Severity
    message: str
    zone_id: Optional[str] = None
    track_id: Optional[int] = None
    category: Optional[str] = None
    data: Mapping[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> Tuple[str, Optional[str], Optional[int]]:
        """Cooldown key: one alert per (rule, zone, track) per cooldown window."""
        return (self.rule_id, self.zone_id, self.track_id)


def _categories(values: Iterable[Any]) -> Set[str]:
    return {Category.parse(v).value for v in values}


class Rule(ABC):
    type_name: ClassVar[str] = ""
    required: ClassVar[Tuple[str, ...]] = ()
    params: ClassVar[Dict[str, Any]] = {}

    def __init__(self, id: str, zones: Sequence[str] = ("*",), severity: Any = "warning",
                 cooldown_s: Optional[float] = None, enabled: bool = True, message: Optional[str] = None,
                 **params: Any) -> None:
        if not id or not isinstance(id, str):
            raise ValueError("rule id must be a non-empty string")
        self.id = id
        self.zones = tuple(zones) if zones else ("*",)
        self.severity = Severity.parse(severity)
        self.cooldown_s = None if cooldown_s is None else float(cooldown_s)
        if self.cooldown_s is not None and self.cooldown_s < 0:
            raise ValueError(f"rule {id!r}: cooldown_s must be >= 0")
        self.enabled = bool(enabled)
        self.message = message
        missing = [p for p in self.required if p not in params]
        if missing:
            raise ValueError(f"rule {id!r} ({self.type_name}) missing parameter(s): {missing}")
        unknown = set(params) - set(self.params) - set(self.required)
        if unknown:
            raise ValueError(f"rule {id!r} ({self.type_name}) has unknown parameter(s): {sorted(unknown)}")
        self.configure(**{**self.params, **params})

    def configure(self, **params: Any) -> None:
        """Validate and store rule-specific parameters."""

    def zone_ids(self, ctx: RuleContext) -> List[str]:
        if "*" in self.zones:
            return list(ctx.zones)
        return [z for z in self.zones if z in ctx.zones]

    def candidate(self, ctx: RuleContext, default_msg: str, zone_id: Optional[str] = None,
                  track: Optional[TrackView] = None, **data: Any) -> AlertCandidate:
        zone = ctx.zones.get(zone_id) if zone_id else None
        category = track.category if track else data.get("category")
        fields = {"zone": zone.name if zone else (zone_id or "-"), "track_id": track.track_id if track else "",
                  "label": track.label if track else "", "category": category or "",
                  "Category": str(category or "").capitalize(), **data}
        try:
            msg = (self.message or default_msg).format(**fields)
        except (KeyError, IndexError, ValueError):
            msg = default_msg.format(**fields)
        return AlertCandidate(self.id, self.type_name, self.severity, msg, zone_id,
                              track.track_id if track else None, category, dict(data))

    @abstractmethod
    def evaluate(self, ctx: RuleContext) -> List[AlertCandidate]:
        """Return the alert candidates for this frame."""


class UnknownPersonRule(Rule):
    """A person without an enrolled identity is in the zone."""

    type_name = "unknown_person"
    params = {"min_dwell_s": 0.0}

    def configure(self, min_dwell_s: float) -> None:
        self.min_dwell_s = float(min_dwell_s)

    def evaluate(self, ctx: RuleContext) -> List[AlertCandidate]:
        return [self.candidate(ctx, "Unknown person (Person #{track_id}) in {zone}", zid, t)
                for zid in self.zone_ids(ctx) for t in ctx.tracks_in(zid)
                if t.category == "human" and not t.identity and ctx.dwell(t.track_id, zid) >= self.min_dwell_s]


class CategoryInZoneRule(Rule):
    """Objects of given categories (optionally labels) present in / entering a zone."""

    type_name = "category_in_zone"
    required = ("categories",)
    params = {"labels": None, "trigger": "presence"}

    def configure(self, categories: Iterable[Any], labels: Optional[Iterable[str]], trigger: str) -> None:
        self.categories = _categories(categories)
        if not self.categories:
            raise ValueError(f"rule {self.id!r}: categories must not be empty")
        self.labels = {str(x).lower() for x in labels} if labels else None
        if trigger not in ("presence", "entry"):
            raise ValueError(f"rule {self.id!r}: trigger must be 'presence' or 'entry'")
        self.trigger = trigger

    def _match(self, category: str, label: str) -> bool:
        return category in self.categories and (self.labels is None or label.lower() in self.labels)

    def evaluate(self, ctx: RuleContext) -> List[AlertCandidate]:
        zids = set(self.zone_ids(ctx))
        if self.trigger == "entry":
            out = []
            for ev in ctx.events:
                cat = _value(ev.category)
                if _value(ev.type) == EventType.ENTRY.value and ev.zone_id in zids and self._match(cat, ev.label):
                    out.append(self.candidate(ctx, "{Category} entered {zone}: {label} #{track_id}", ev.zone_id,
                                              TrackView(ev.track_id, cat, ev.label)))
            return out
        return [self.candidate(ctx, "{Category} in {zone}: {label} #{track_id}", zid, t)
                for zid in zids for t in ctx.tracks_in(zid) if self._match(t.category, t.label)]


class CrowdRule(Rule):
    """More than N objects of a category in a zone (falls back to zone.max_occupancy)."""

    type_name = "crowd"
    params = {"threshold": None, "category": "human"}

    def configure(self, threshold: Optional[int], category: Optional[str]) -> None:
        self.threshold = None if threshold is None else int(threshold)
        if self.threshold is not None and self.threshold < 0:
            raise ValueError(f"rule {self.id!r}: threshold must be >= 0")
        self.category = None if category in (None, "*", "any") else Category.parse(category).value

    def evaluate(self, ctx: RuleContext) -> List[AlertCandidate]:
        out = []
        for zid in self.zone_ids(ctx):
            limit = self.threshold if self.threshold is not None else ctx.zones[zid].max_occupancy
            if limit is None:
                continue
            n = sum(1 for t in ctx.tracks_in(zid) if self.category is None or t.category == self.category)
            if n > limit:
                out.append(self.candidate(ctx, "Crowd in {zone}: {count} > {limit}", zid, None, count=n, limit=limit,
                                          category=self.category or "any"))
        return out


class ExpressionRule(Rule):
    """A facial expression held with sufficient confidence for a minimum duration."""

    type_name = "expression"
    params = {"emotions": ["angry"], "min_confidence": 0.6, "min_duration_s": 0.0}

    def configure(self, emotions: Iterable[str], min_confidence: float, min_duration_s: float) -> None:
        self.emotions = {str(e).lower() for e in emotions}
        bad = self.emotions - set(EMOTIONS)
        if bad or not self.emotions:
            raise ValueError(f"rule {self.id!r}: invalid emotion(s) {sorted(bad)}; valid: {list(EMOTIONS)}")
        self.min_confidence = float(min_confidence)
        self.min_duration_s = float(min_duration_s)

    def evaluate(self, ctx: RuleContext) -> List[AlertCandidate]:
        out = []
        for zid in self.zone_ids(ctx):
            for t in ctx.tracks_in(zid):
                if t.category != "human" or t.emotion not in self.emotions or t.emotion_conf < self.min_confidence:
                    continue
                held = ctx.timestamp - t.emotion_since if t.emotion_since is not None else 0.0
                if held < self.min_duration_s:
                    continue
                out.append(self.candidate(ctx, "{Emotion} expression detected: {who} in {zone}", zid, t,
                                          emotion=t.emotion, Emotion=str(t.emotion).capitalize(),
                                          who=t.identity or f"Person #{t.track_id}",
                                          confidence=round(t.emotion_conf, 2), held_s=round(held, 1)))
        return out


class DwellRule(Rule):
    """Loitering: dwell time above a limit (falls back to zone.dwell_limit_s)."""

    type_name = "dwell"
    params = {"max_dwell_s": None, "categories": ["human"]}

    def configure(self, max_dwell_s: Optional[float], categories: Iterable[Any]) -> None:
        self.max_dwell_s = None if max_dwell_s is None else float(max_dwell_s)
        if self.max_dwell_s is not None and self.max_dwell_s <= 0:
            raise ValueError(f"rule {self.id!r}: max_dwell_s must be > 0")
        self.categories = _categories(categories)

    def evaluate(self, ctx: RuleContext) -> List[AlertCandidate]:
        out = []
        for zid in self.zone_ids(ctx):
            limit = self.max_dwell_s if self.max_dwell_s is not None else ctx.zones[zid].dwell_limit_s
            if limit is None:
                continue
            for t in ctx.tracks_in(zid):
                d = ctx.dwell(t.track_id, zid)
                if t.category in self.categories and d > limit:
                    out.append(self.candidate(ctx, "Loitering: {label} #{track_id} in {zone} for {dwell}s (limit {limit}s)",
                                              zid, t, dwell=int(d), limit=int(limit)))
        return out


class UnclassifiedDwellRule(DwellRule):
    """Unclassified / unattended object remaining in a zone."""

    type_name = "unclassified_dwell"
    params = {"max_dwell_s": 60.0, "categories": ["unclassified"]}

    def evaluate(self, ctx: RuleContext) -> List[AlertCandidate]:
        return [AlertCandidate(c.rule_id, c.rule_type, c.severity,
                               c.message if self.message else f"Unattended/unclassified object in "
                               f"{ctx.zones[c.zone_id].name if c.zone_id in ctx.zones else c.zone_id} for {c.data.get('dwell')}s",
                               c.zone_id, c.track_id, c.category, c.data)
                for c in super().evaluate(ctx)]


class ZoneViolationRule(Rule):
    """Zone policy breach derived from the zone's allowed / restricted mode."""

    type_name = "zone_violation"
    params = {"ignore_categories": []}

    def configure(self, ignore_categories: Iterable[Any]) -> None:
        self.ignore = _categories(ignore_categories)

    def evaluate(self, ctx: RuleContext) -> List[AlertCandidate]:
        out = []
        for zid in self.zone_ids(ctx):
            zone = ctx.zones[zid]
            mode = _value(getattr(zone, "mode", ""))
            for t in ctx.tracks_in(zid):
                if t.category not in self.ignore and zone.is_violation(t.category):
                    out.append(self.candidate(ctx, "Zone policy violation: {category} '{label}' #{track_id} in {mode} zone {zone}",
                                              zid, t, mode=mode))
        return out


RULE_TYPES: Dict[str, Type[Rule]] = {cls.type_name: cls for cls in (
    UnknownPersonRule, CategoryInZoneRule, CrowdRule, ExpressionRule, DwellRule, UnclassifiedDwellRule, ZoneViolationRule)}


def build_rule(spec: Mapping[str, Any]) -> Rule:
    """Instantiate one rule from its config mapping (raises ``ValueError`` on any problem)."""
    if not isinstance(spec, Mapping):
        raise ValueError("rule spec must be a mapping")
    data = dict(spec)
    rtype = data.pop("type", None)
    if rtype not in RULE_TYPES:
        raise ValueError(f"unknown rule type {rtype!r}; valid: {sorted(RULE_TYPES)}")
    if "id" not in data:
        raise ValueError(f"rule of type {rtype!r} requires an 'id'")
    zones = data.pop("zones", ["*"])
    if isinstance(zones, str):
        zones = [zones]
    return RULE_TYPES[rtype](zones=zones, **data)


def build_rules(specs: Optional[Iterable[Mapping[str, Any]]]) -> List[Rule]:
    rules: List[Rule] = []
    seen: Set[str] = set()
    for spec in specs or []:
        rule = build_rule(spec)
        if rule.id in seen:
            raise ValueError(f"duplicate rule id {rule.id!r}")
        seen.add(rule.id)
        rules.append(rule)
    return rules
