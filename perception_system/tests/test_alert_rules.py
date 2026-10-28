"""Alert rules, rule validation and the alert engine."""
import unittest

from alerts.engine import AlertEngine
from alerts.rules import Rule, RuleContext, TrackView, build_rule, build_rules
from core.types import Category, EventType, Severity, ZoneEvent
from zones.zone import Zone

SQ = [(0, 0), (1, 0), (1, 1), (0, 1)]
ZONES = {
    "zone_a": Zone(id="zone_a", name="Zone A", polygon=SQ, mode="allowed", categories=["human"],
                   max_occupancy=2, dwell_limit_s=10),
    "walkway": Zone(id="walkway", name="Pedestrian Walkway", polygon=SQ, mode="restricted", categories=["vehicle"]),
    "farm": Zone(id="farm", name="Farm Zone", polygon=SQ, mode="allowed", categories=["human", "animal", "vehicle"]),
}


def ctx(tracks, members, t=100.0, events=(), dwell=None):
    return RuleContext(timestamp=t, tracks=tracks, memberships=members, zones=ZONES, events=list(events),
                       dwell=dwell or (lambda tid, zid: 0.0))


def person(tid, identity=None, emotion=None, conf=0.0, since=None):
    return TrackView(tid, "human", "person", 0.9, identity, emotion, conf, since)


class RuleTests(unittest.TestCase):
    def test_unknown_person_ignores_enrolled(self):
        rule = build_rule({"id": "u", "type": "unknown_person", "zones": ["zone_a"]})
        out = rule.evaluate(ctx([person(1), person(2, identity="Asha")], {1: {"zone_a"}, 2: {"zone_a"}}))
        self.assertEqual([c.track_id for c in out], [1])
        self.assertEqual(out[0].message, "Unknown person (Person #1) in Zone A")

    def test_unknown_person_min_dwell(self):
        rule = build_rule({"id": "u", "type": "unknown_person", "zones": ["zone_a"], "min_dwell_s": 2})
        members = {1: {"zone_a"}}
        self.assertEqual(rule.evaluate(ctx([person(1)], members, dwell=lambda *_: 1.0)), [])
        self.assertEqual(len(rule.evaluate(ctx([person(1)], members, dwell=lambda *_: 2.5))), 1)

    def test_vehicle_in_pedestrian_zone(self):
        rule = build_rule({"id": "v", "type": "category_in_zone", "zones": ["walkway"], "categories": ["vehicle"],
                           "severity": "critical"})
        car, walker = TrackView(7, "vehicle", "car"), person(8)
        out = rule.evaluate(ctx([car, walker], {7: {"walkway"}, 8: {"walkway"}}))
        self.assertEqual(len(out), 1)
        self.assertIs(out[0].severity, Severity.CRITICAL)
        self.assertEqual(out[0].message, "Vehicle in Pedestrian Walkway: car #7")

    def test_label_filter(self):
        rule = build_rule({"id": "v", "type": "category_in_zone", "categories": ["vehicle"], "labels": ["truck"]})
        out = rule.evaluate(ctx([TrackView(1, "vehicle", "car"), TrackView(2, "vehicle", "truck")],
                                {1: {"walkway"}, 2: {"walkway"}}))
        self.assertEqual([c.track_id for c in out], [2])

    def test_animal_entered_farm_fires_on_entry_only(self):
        rule = build_rule({"id": "f", "type": "category_in_zone", "zones": ["farm"], "categories": ["animal"],
                           "trigger": "entry"})
        cow = TrackView(5, "animal", "cow")
        self.assertEqual(rule.evaluate(ctx([cow], {5: {"farm"}})), [])
        entry = ZoneEvent(EventType.ENTRY, "farm", "Farm Zone", 5, Category.ANIMAL, "cow", 100.0)
        out = rule.evaluate(ctx([cow], {5: {"farm"}}, events=[entry]))
        self.assertEqual(out[0].message, "Animal entered Farm Zone: cow #5")
        exit_ev = ZoneEvent(EventType.EXIT, "farm", "Farm Zone", 5, Category.ANIMAL, "cow", 100.0)
        self.assertEqual(rule.evaluate(ctx([cow], {}, events=[exit_ev])), [])

    def test_crowd_is_strictly_greater_than_threshold(self):
        rule = build_rule({"id": "c", "type": "crowd", "zones": ["farm"], "threshold": 2})
        two = [person(1), person(2)]
        self.assertEqual(rule.evaluate(ctx(two, {1: {"farm"}, 2: {"farm"}})), [])
        three = two + [person(3)]
        out = rule.evaluate(ctx(three, {i: {"farm"} for i in (1, 2, 3)}))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].data["count"], 3)
        self.assertIsNone(out[0].track_id)

    def test_crowd_falls_back_to_zone_max_occupancy(self):
        rule = build_rule({"id": "c", "type": "crowd"})
        tracks = [person(i) for i in (1, 2, 3)]
        out = rule.evaluate(ctx(tracks, {i: {"zone_a"} for i in (1, 2, 3)}))
        self.assertEqual([c.zone_id for c in out], ["zone_a"])

    def test_expression_requires_confidence_and_duration(self):
        rule = build_rule({"id": "e", "type": "expression", "emotions": ["angry"], "min_confidence": 0.6,
                           "min_duration_s": 1.5})
        m = {1: {"zone_a"}}
        self.assertEqual(rule.evaluate(ctx([person(1, emotion="angry", conf=0.7, since=99.0)], m)), [])
        self.assertEqual(rule.evaluate(ctx([person(1, emotion="angry", conf=0.5, since=90.0)], m)), [])
        self.assertEqual(rule.evaluate(ctx([person(1, emotion="happy", conf=0.9, since=90.0)], m)), [])
        out = rule.evaluate(ctx([person(1, emotion="angry", conf=0.7, since=98.0)], m))
        self.assertEqual(out[0].message, "Angry expression detected: Person #1 in Zone A")

    def test_dwell_uses_zone_limit_by_default(self):
        rule = build_rule({"id": "d", "type": "dwell", "zones": ["zone_a"]})
        m = {1: {"zone_a"}}
        self.assertEqual(rule.evaluate(ctx([person(1)], m, dwell=lambda *_: 8.0)), [])
        out = rule.evaluate(ctx([person(1)], m, dwell=lambda *_: 12.0))
        self.assertIn("Loitering", out[0].message)

    def test_zone_violation_respects_mode(self):
        rule = build_rule({"id": "z", "type": "zone_violation"})
        tracks = [TrackView(1, "vehicle", "car"), person(2), TrackView(3, "animal", "dog")]
        out = rule.evaluate(ctx(tracks, {1: {"walkway"}, 2: {"walkway", "zone_a"}, 3: {"zone_a"}}))
        self.assertEqual(sorted((c.zone_id, c.track_id) for c in out), [("walkway", 1), ("zone_a", 3)])

    def test_unclassified_dwell(self):
        rule = build_rule({"id": "o", "type": "unclassified_dwell", "zones": ["farm"], "max_dwell_s": 60})
        thing = TrackView(9, "unclassified", "unclassified")
        self.assertEqual(rule.evaluate(ctx([thing], {9: {"farm"}}, dwell=lambda *_: 30.0)), [])
        out = rule.evaluate(ctx([thing], {9: {"farm"}}, dwell=lambda *_: 70.0))
        self.assertTrue(out[0].message.startswith("Unattended/unclassified object in Farm Zone"))

    def test_custom_message_and_bad_template_fallback(self):
        rule = build_rule({"id": "v", "type": "category_in_zone", "categories": ["vehicle"],
                           "message": "ALERT {label} in {zone}"})
        out = rule.evaluate(ctx([TrackView(1, "vehicle", "car")], {1: {"walkway"}}))
        self.assertEqual(out[0].message, "ALERT car in Pedestrian Walkway")
        bad = build_rule({"id": "w", "type": "category_in_zone", "categories": ["vehicle"], "message": "{nope}"})
        self.assertIn("Vehicle in", bad.evaluate(ctx([TrackView(1, "vehicle", "car")], {1: {"walkway"}}))[0].message)

    def test_unknown_zone_ids_are_ignored(self):
        rule = build_rule({"id": "v", "type": "category_in_zone", "zones": ["ghost"], "categories": ["vehicle"]})
        self.assertEqual(rule.evaluate(ctx([TrackView(1, "vehicle", "car")], {1: {"walkway"}})), [])


class RuleValidationTests(unittest.TestCase):
    def test_invalid_specs(self):
        bad = [{"id": "x", "type": "nope"}, {"type": "crowd"}, {"id": "x", "type": "category_in_zone"},
               {"id": "x", "type": "crowd", "bogus": 1}, {"id": "x", "type": "crowd", "severity": "extreme"},
               {"id": "x", "type": "expression", "emotions": ["bored"]}, {"id": "x", "type": "crowd", "cooldown_s": -1},
               {"id": "x", "type": "category_in_zone", "categories": ["spaceship"]},
               {"id": "x", "type": "category_in_zone", "categories": ["animal"], "trigger": "always"},
               {"id": "x", "type": "dwell", "max_dwell_s": 0}, "not-a-mapping"]
        for spec in bad:
            with self.assertRaises(ValueError, msg=str(spec)):
                build_rule(spec)

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            build_rules([{"id": "x", "type": "crowd"}, {"id": "x", "type": "crowd"}])

    def test_string_zone_and_defaults(self):
        rule = build_rule({"id": "x", "type": "crowd", "zones": "farm"})
        self.assertEqual(rule.zones, ("farm",))
        self.assertIs(rule.severity, Severity.WARNING)


class AlertEngineTests(unittest.TestCase):
    def setUp(self):
        self.rules = build_rules([{"id": "v", "type": "category_in_zone", "zones": ["walkway"],
                                   "categories": ["vehicle"], "cooldown_s": 10}])

    def test_cooldown_per_key(self):
        engine = AlertEngine(self.rules, default_cooldown_s=30)
        car, m = TrackView(7, "vehicle", "car"), {7: {"walkway"}}
        self.assertEqual(len(engine.evaluate(ctx([car], m, t=0.0))), 1)
        self.assertEqual(len(engine.evaluate(ctx([car], m, t=5.0))), 0)
        self.assertEqual(len(engine.evaluate(ctx([car], m, t=10.5))), 1)
        self.assertEqual(engine.stats["suppressed"], 1)

    def test_distinct_tracks_not_deduplicated(self):
        engine = AlertEngine(self.rules)
        tracks = [TrackView(1, "vehicle", "car"), TrackView(2, "vehicle", "bus")]
        alerts = engine.evaluate(ctx(tracks, {1: {"walkway"}, 2: {"walkway"}}, t=0.0))
        self.assertEqual(sorted(a.track_id for a in alerts), [1, 2])
        self.assertEqual(alerts[0].zone_name, "Pedestrian Walkway")

    def test_failing_rule_is_isolated(self):
        class Boom(Rule):
            type_name = "boom"

            def evaluate(self, ctx):
                raise RuntimeError("boom")

        engine = AlertEngine([Boom("boom")] + self.rules)
        alerts = engine.evaluate(ctx([TrackView(1, "vehicle", "car")], {1: {"walkway"}}, t=0.0))
        self.assertEqual(len(alerts), 1)
        self.assertEqual(engine.stats["rule_errors"], 1)

    def test_disabled_rule_and_hot_reload(self):
        engine = AlertEngine(build_rules([{"id": "v", "type": "category_in_zone", "categories": ["vehicle"],
                                           "enabled": False}]))
        m = {1: {"walkway"}}
        self.assertEqual(engine.evaluate(ctx([TrackView(1, "vehicle", "car")], m, t=0.0)), [])
        engine.set_rules(self.rules)
        self.assertEqual(len(engine.evaluate(ctx([TrackView(1, "vehicle", "car")], m, t=1.0))), 1)


if __name__ == "__main__":
    unittest.main()


class EvaluationHelperTests(unittest.TestCase):
    """The evaluation maths must itself be verified (tools/evaluate.py)."""

    def test_event_matching(self):
        from tools.evaluate import match_events

        gt = [{"t": 1.0, "zone_id": "a", "type": "entry", "category": "human"},
              {"t": 5.0, "zone_id": "a", "type": "exit", "category": "human"}]
        pred = [{"t": 1.4, "zone_id": "a", "type": "entry", "category": "human"},
                {"t": 9.0, "zone_id": "a", "type": "exit", "category": "human"}]
        r = match_events(gt, pred, tolerance_s=1.0)
        self.assertEqual((r["tp"], r["fp"], r["fn"]), (1, 1, 1))
        self.assertAlmostEqual(r["mean_abs_timing_error_s"], 0.4)

    def test_classification_report(self):
        from tools.evaluate import classification_report

        r = classification_report({"happy": {"happy": 8, "sad": 2}, "sad": {"happy": 1, "sad": 9}})
        self.assertAlmostEqual(r["accuracy"], 0.85)
        self.assertAlmostEqual(r["per_class"]["happy"]["recall"], 0.8)
