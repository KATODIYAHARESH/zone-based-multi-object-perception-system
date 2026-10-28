"""Hysteresis, dwell time and entry / exit events."""
import unittest

from core.types import Category, EventType
from tracking.dwell import DwellTracker, TrackInfo


class DwellTrackerTests(unittest.TestCase):
    def setUp(self):
        self.d = DwellTracker(enter_frames=3, exit_frames=2)
        self.names = {"a": "Zone A"}
        self.info = {1: TrackInfo(Category.HUMAN, "person", (0.5, 0.5))}

    def step(self, t, inside, alive=None):
        return self.d.update(t, {1: {"a"} if inside else set()}, self.info, self.names, alive)

    def enter(self):
        for i in range(3):
            self.step(i * 0.1, True)

    def test_entry_requires_consecutive_frames(self):
        self.assertEqual(self.step(0.0, True), [])
        self.assertEqual(self.step(0.1, True), [])
        events = self.step(0.2, True)
        self.assertEqual([e.type for e in events], [EventType.ENTRY])
        self.assertEqual(events[0].zone_name, "Zone A")
        self.assertEqual(events[0].category, Category.HUMAN)
        self.assertAlmostEqual(self.d.dwell_seconds(1, "a", 10.0), 10.0)  # measured from first inside frame

    def test_boundary_jitter_produces_no_events(self):
        for i in range(20):
            self.assertEqual(self.step(i * 0.1, i % 2 == 0), [])
        self.assertEqual(self.d.occupants("a"), set())

    def test_exit_reports_dwell(self):
        self.enter()
        self.step(5.0, True)
        self.assertEqual(self.step(5.1, False), [])
        events = self.step(5.2, False)
        self.assertEqual([e.type for e in events], [EventType.EXIT])
        self.assertAlmostEqual(events[0].dwell_s, 5.0)

    def test_single_outside_frame_does_not_exit(self):
        self.enter()
        self.step(0.3, False)
        self.assertEqual(self.step(0.4, True), [])
        self.assertEqual(self.d.occupants("a"), {1})

    def test_track_loss_emits_exit(self):
        self.enter()
        events = self.d.update(1.0, {}, {}, self.names, alive_ids=set())
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].reason, "track_lost")

    def test_temporarily_lost_track_keeps_presence(self):
        self.enter()
        self.assertEqual(self.d.update(1.0, {}, {}, self.names, alive_ids={1}), [])
        self.assertEqual(self.d.occupants("a"), {1})
        self.assertAlmostEqual(self.d.dwell_seconds(1, "a", 2.0), 2.0)

    def test_zone_removed_emits_exit(self):
        self.enter()
        events = self.d.update(1.0, {1: set()}, self.info, {}, alive_ids={1})
        self.assertEqual(events[0].reason, "zone_removed")
        self.assertEqual(events[0].zone_name, "Zone A")

    def test_zones_of_reflects_hysteresis(self):
        self.step(0.0, True)
        self.assertEqual(self.d.zones_of(1), set())
        self.step(0.1, True)
        self.step(0.2, True)
        self.assertEqual(self.d.zones_of(1), {"a"})

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            DwellTracker(0, 1)


if __name__ == "__main__":
    unittest.main()
