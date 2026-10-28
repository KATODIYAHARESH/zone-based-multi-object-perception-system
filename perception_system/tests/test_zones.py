"""Zone model semantics and bottom-centre zone assignment."""
import unittest
from dataclasses import dataclass

from core.types import Category
from zones.zone import Zone, ZoneManager, ZoneMode, slugify

TRI = [(0, 0), (1, 0), (1, 1)]


@dataclass
class FakeTrack:
    track_id: int
    bbox: tuple


class ZoneModelTests(unittest.TestCase):
    def test_allowed_mode_is_whitelist(self):
        z = Zone(id="farm", name="Farm", polygon=TRI, mode="allowed", categories=["human", "animal"])
        self.assertIs(z.mode, ZoneMode.ALLOWED)
        self.assertFalse(z.is_violation("human"))
        self.assertFalse(z.is_violation(Category.ANIMAL))
        self.assertTrue(z.is_violation("vehicle"))
        self.assertTrue(z.is_violation("unclassified"))

    def test_restricted_mode_is_blacklist(self):
        z = Zone(id="walkway", name="Walkway", polygon=TRI, mode="restricted", categories=["vehicle"])
        self.assertTrue(z.is_violation("vehicle"))
        self.assertFalse(z.is_violation("human"))

    def test_empty_category_lists(self):
        self.assertTrue(Zone(id="r", name="R", polygon=TRI, mode="restricted").is_violation("human"))
        self.assertTrue(Zone(id="a", name="A", polygon=TRI, mode="allowed").is_violation("human"))

    def test_invalid_inputs_rejected(self):
        bad = [dict(id="Zone A"), dict(mode="forbidden"), dict(categories=["spaceship"]),
               dict(polygon=[(0, 0), (0.5, 0.5), (1, 1)]), dict(color="blue"), dict(max_occupancy=-1),
               dict(dwell_limit_s=0), dict(name="  "), dict(polygon=[(0, 0), (2, 0), (1, 1)])]
        for override in bad:
            kwargs = dict(id="z", name="Z", polygon=TRI)
            kwargs.update(override)
            with self.assertRaises(ValueError, msg=str(override)):
                Zone(**kwargs)

    def test_dict_roundtrip(self):
        z = Zone(id="zone_a", name="Zone A", polygon=[(0.1, 0.5), (0.4, 0.5), (0.4, 0.9)], mode="allowed",
                 categories=["human"], max_occupancy=8, dwell_limit_s=120, color="green")
        self.assertEqual(Zone.from_dict(z.to_dict()), z)
        with self.assertRaises(ValueError):
            Zone.from_dict({**z.to_dict(), "colour": "amber"})

    def test_slugify(self):
        self.assertEqual(slugify("Pedestrian Walkway #2"), "pedestrian_walkway_2")
        self.assertEqual(slugify("!!!"), "zone")


class ZoneManagerTests(unittest.TestCase):
    def setUp(self):
        self.mgr = ZoneManager([
            Zone(id="left", name="Left", polygon=[(0, 0), (0.5, 0), (0.5, 1), (0, 1)]),
            Zone(id="right", name="Right", polygon=[(0.5, 0), (1, 0), (1, 1), (0.5, 1)], mode="allowed",
                 categories=["human"]),
            Zone(id="lower", name="Lower", polygon=[(0, 0.5), (1, 0.5), (1, 1), (0, 1)]),
        ])

    def test_assignment_uses_bottom_center(self):
        # box centre (150, 350) is in the upper half; feet (150, 600) are in the lower half
        members = self.mgr.assign([FakeTrack(1, (100, 100, 200, 600))], 1000, 1000)
        self.assertEqual(members[1], {"left", "lower"})

    def test_head_inside_feet_outside_is_not_inside(self):
        mgr = ZoneManager([Zone(id="top", name="Top", polygon=[(0, 0), (1, 0), (1, 0.5), (0, 0.5)])])
        track = FakeTrack(2, (100, 50, 200, 800))
        self.assertEqual(mgr.assign([track], 1000, 1000)[2], set())
        center_mgr = ZoneManager(mgr.list(), anchor="center")
        self.assertEqual(center_mgr.assign([track], 1000, 1000)[2], {"top"})

    def test_shared_edge_counts_in_both_zones(self):
        members = self.mgr.assign([FakeTrack(3, (450, 100, 550, 300))], 1000, 1000)
        self.assertEqual(members[3], {"left", "right"})

    def test_disabled_zone_ignored(self):
        z = self.mgr.get("lower")
        self.mgr.add(Zone(**{**z.to_dict(), "polygon": z.polygon, "enabled": False}), replace=True)
        self.assertEqual(self.mgr.assign([FakeTrack(1, (100, 100, 200, 600))], 1000, 1000)[1], {"left"})
        self.assertNotIn("lower", self.mgr.names())

    def test_registry_operations(self):
        with self.assertRaises(KeyError):
            self.mgr.add(Zone(id="left", name="Dup", polygon=TRI))
        self.assertEqual(self.mgr.unique_id("Left"), "left_2")
        self.assertTrue(self.mgr.remove("left"))
        self.assertFalse(self.mgr.remove("left"))
        self.assertIsNone(self.mgr.get("left"))

    def test_invalid_frame_and_anchor(self):
        with self.assertRaises(ValueError):
            self.mgr.assign([], 0, 720)
        with self.assertRaises(ValueError):
            ZoneManager([], anchor="feet")

    def test_from_dicts(self):
        mgr = ZoneManager.from_dicts([{"name": "Gate 1", "polygon": [[0, 0], [1, 0], [1, 1]]}])
        self.assertEqual(mgr.list()[0].id, "gate_1")


if __name__ == "__main__":
    unittest.main()
