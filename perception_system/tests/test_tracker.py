"""ByteTrack behaviour (requires NumPy; skipped otherwise)."""
import unittest
from types import SimpleNamespace

from core.types import Category, Detection

try:
    import numpy  # noqa: F401
    HAVE_NUMPY = True
except ImportError:  # pragma: no cover
    HAVE_NUMPY = False

if HAVE_NUMPY:
    from tracking.byte_tracker import ByteTracker

CFG = SimpleNamespace(high_thresh=None, low_thresh=None, new_track_thresh=0.45, match_thresh=0.8,
                      second_match_thresh=0.5, track_buffer=30, min_hits=3, class_aware=True,
                      label_consensus=0.6, vote_decay=0.9)


def det(x, y, label="person", cat=Category.HUMAN, conf=0.9, w=50, h=120):
    return Detection((x, y, x + w, y + h), conf, label, cat)


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed")
class ByteTrackerTests(unittest.TestCase):
    def tracker(self):
        return ByteTracker(CFG, high_thresh=0.35, low_thresh=0.15)

    def test_id_persists_for_moving_object(self):
        trk, ids = self.tracker(), set()
        for i in range(20):
            ids |= {t.track_id for t in trk.update([det(100 + 5 * i, 200)], i / 30)}
        self.assertEqual(ids, {1})

    def test_confirmation_requires_min_hits(self):
        trk = self.tracker()
        self.assertEqual(trk.update([det(100, 100)], 0.0), [])
        self.assertEqual(trk.update([det(102, 100)], 0.03), [])
        self.assertEqual(len(trk.update([det(104, 100)], 0.06)), 1)

    def test_occlusion_recovery_keeps_id(self):
        trk, ids = self.tracker(), set()
        for i in range(30):
            dets = [] if 10 <= i < 20 else [det(100 + 5 * i, 200)]
            ids |= {t.track_id for t in trk.update(dets, i / 30)}
        self.assertEqual(ids, {1})

    def test_low_confidence_never_spawns_tracks(self):
        trk = self.tracker()
        for i in range(10):
            self.assertEqual(trk.update([det(100, 100, conf=0.2)], i / 30), [])

    def test_class_aware_gating(self):
        trk, ids = self.tracker(), set()
        for i in range(6):
            out = trk.update([det(100, 100), det(105, 110, "dog", Category.ANIMAL, w=50, h=110)], i / 30)
            ids |= {(t.track_id, t.category) for t in out}
        self.assertEqual({c for _, c in ids}, {Category.HUMAN, Category.ANIMAL})
        self.assertEqual(len({i for i, _ in ids}), 2)

    def test_label_disagreement_becomes_unclassified(self):
        trk, out = self.tracker(), []
        for i in range(12):
            label = "car" if i % 2 == 0 else "truck"
            out = trk.update([det(100, 100, label, Category.VEHICLE, w=200, h=100)], i / 30)
        self.assertEqual(out[0].category, Category.UNCLASSIFIED)
        trk2 = self.tracker()
        for i in range(12):
            out = trk2.update([det(100, 100, "car", Category.VEHICLE, w=200, h=100)], i / 30)
        self.assertEqual((out[0].label, out[0].category), ("car", Category.VEHICLE))

    def test_coast_does_not_penalise(self):
        trk = self.tracker()
        for i in range(5):
            trk.update([det(100 + 5 * i, 200)], i / 30)
        for i in range(60):
            self.assertEqual(len(trk.coast(1.0 + i / 30)), 1)


if __name__ == "__main__":
    unittest.main()
