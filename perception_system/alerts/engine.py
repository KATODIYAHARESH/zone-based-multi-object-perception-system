"""Alert engine: evaluates rules with fault isolation and per-key cooldowns."""
from __future__ import annotations

import logging
import threading
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from alerts.rules import RuleContext, Rule
from core.types import Alert

log = logging.getLogger(__name__)


class AlertEngine:
    """Turns rule candidates into :class:`Alert` objects.

    * A failing rule is logged and skipped for that frame - it never takes the
      engine (or the other rules) down.
    * Cooldown is applied per ``(rule_id, zone_id, track_id)``, so two
      different intruders both raise alerts while one loiterer raises one.
    """

    def __init__(self, rules: Sequence[Rule], default_cooldown_s: float = 30.0) -> None:
        self._rules: List[Rule] = list(rules)
        self.default_cooldown_s = float(default_cooldown_s)
        self._last_fired: Dict[Tuple[str, Optional[str], Optional[int]], float] = {}
        self._lock = threading.Lock()
        self._evals = 0
        self.stats: Counter = Counter()

    @property
    def rules(self) -> List[Rule]:
        with self._lock:
            return list(self._rules)

    def set_rules(self, rules: Sequence[Rule]) -> None:
        with self._lock:
            self._rules = list(rules)
        log.info("Alert rules reloaded (%d rules)", len(rules))

    def evaluate(self, ctx: RuleContext) -> List[Alert]:
        fired: List[Alert] = []
        for rule in self.rules:
            if not rule.enabled:
                continue
            try:
                candidates = rule.evaluate(ctx)
            except Exception:
                log.exception("Rule %s raised - skipped for this frame", rule.id)
                self.stats["rule_errors"] += 1
                continue
            cooldown = rule.cooldown_s if rule.cooldown_s is not None else self.default_cooldown_s
            for c in candidates:
                last = self._last_fired.get(c.key)
                if last is not None and ctx.timestamp - last < cooldown:
                    self.stats["suppressed"] += 1
                    continue
                self._last_fired[c.key] = ctx.timestamp
                zone = ctx.zones.get(c.zone_id) if c.zone_id else None
                fired.append(Alert(rule_id=c.rule_id, rule_type=c.rule_type, severity=c.severity, message=c.message,
                                   timestamp=ctx.timestamp, zone_id=c.zone_id, zone_name=zone.name if zone else None,
                                   track_id=c.track_id, category=c.category, data=dict(c.data)))
                self.stats[c.severity.value] += 1
        self._evals += 1
        if self._evals % 500 == 0:
            self._prune(ctx.timestamp)
        return fired

    def _prune(self, now: float) -> None:
        horizon = max([self.default_cooldown_s] + [r.cooldown_s or 0.0 for r in self.rules]) * 2 + 60.0
        for key in [k for k, ts in self._last_fired.items() if now - ts > horizon]:
            del self._last_fired[key]
