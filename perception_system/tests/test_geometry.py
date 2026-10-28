"""Point-in-polygon and polygon utilities."""
import math
import unittest

from zones.geometry import (bbox_anchor, denormalize_polygon, normalize_polygon, point_in_polygon,
                            point_on_segment, polygon_area, polygon_centroid, validate_polygon)

SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]
# U-shape: base y in [0,3], arms x in [0,3] and [7,10]; the notch (3<x<7, y>3) is OUTSIDE.
CONCAVE_U = [(0, 0), (10, 0), (10, 10), (7, 10), (7, 3), (3, 3), (3, 10), (0, 10)]
DIAMOND = [(5, 0), (10, 5), (5, 10), (0, 5)]


class PointInPolygonTests(unittest.TestCase):
    def test_inside_and_outside_square(self):
        self.assertTrue(point_in_polygon((5, 5), SQUARE))
        self.assertFalse(point_in_polygon((15, 5), SQUARE))
        self.assertFalse(point_in_polygon((-0.01, 5), SQUARE))
        self.assertFalse(point_in_polygon((5, 10.01), SQUARE))

    def test_boundary_and_vertices_follow_flag(self):
        for p in [(0, 5), (10, 5), (5, 0), (5, 10), (0, 0), (10, 10)]:
            self.assertTrue(point_in_polygon(p, SQUARE), p)
            self.assertFalse(point_in_polygon(p, SQUARE, include_boundary=False), p)

    def test_concave_notch_is_outside(self):
        self.assertFalse(point_in_polygon((5, 6), CONCAVE_U))
        self.assertTrue(point_in_polygon((1.5, 8), CONCAVE_U))
        self.assertTrue(point_in_polygon((8.5, 8), CONCAVE_U))
        self.assertTrue(point_in_polygon((5, 1.5), CONCAVE_U))
        self.assertTrue(point_in_polygon((5, 3), CONCAVE_U))  # on the notch floor (boundary)

    def test_ray_through_vertex_counted_once(self):
        self.assertTrue(point_in_polygon((2, 5), DIAMOND))
        self.assertFalse(point_in_polygon((-1, 5), DIAMOND))
        self.assertFalse(point_in_polygon((11, 5), DIAMOND))

    def test_vertex_order_does_not_matter(self):
        rev = list(reversed(CONCAVE_U))
        for p in [(5, 6), (1.5, 8), (5, 1.5), (12, 1)]:
            self.assertEqual(point_in_polygon(p, CONCAVE_U), point_in_polygon(p, rev), p)

    def test_high_resolution_circle(self):
        circle = [(100 * math.cos(math.radians(a)), 100 * math.sin(math.radians(a))) for a in range(360)]
        self.assertTrue(point_in_polygon((99 * math.cos(1.0), 99 * math.sin(1.0)), circle))
        self.assertFalse(point_in_polygon((101 * math.cos(1.0), 101 * math.sin(1.0)), circle))

    def test_too_few_vertices_raises(self):
        with self.assertRaises(ValueError):
            point_in_polygon((0, 0), [(0, 0), (1, 1)])


class PolygonUtilityTests(unittest.TestCase):
    def test_validate_rejects_degenerate(self):
        with self.assertRaises(ValueError):
            validate_polygon([(0, 0), (1, 1), (2, 2)])          # collinear, zero area
        with self.assertRaises(ValueError):
            validate_polygon([(0, 0), (0, 0), (1, 1)])          # duplicates collapse to 2 vertices
        with self.assertRaises(ValueError):
            validate_polygon([(0, 0), (1, float("nan")), (1, 1)])
        with self.assertRaises(ValueError):
            validate_polygon([(0, 0), ("a",), (1, 1)])

    def test_validate_strips_closing_vertex(self):
        self.assertEqual(len(validate_polygon(SQUARE + [SQUARE[0]])), 4)

    def test_normalized_bounds(self):
        with self.assertRaises(ValueError):
            validate_polygon([(0, 0), (1.5, 0), (1, 1)], normalized=True)
        self.assertEqual(len(validate_polygon([(0, 0), (1, 0), (1, 1)], normalized=True)), 3)

    def test_area_and_centroid(self):
        self.assertAlmostEqual(abs(polygon_area(SQUARE)), 100.0)
        cx, cy = polygon_centroid(SQUARE)
        self.assertAlmostEqual(cx, 5.0)
        self.assertAlmostEqual(cy, 5.0)

    def test_point_on_segment(self):
        self.assertTrue(point_on_segment((5, 0), (0, 0), (10, 0)))
        self.assertFalse(point_on_segment((5, 0.1), (0, 0), (10, 0)))
        self.assertFalse(point_on_segment((11, 0), (0, 0), (10, 0)))

    def test_anchor_modes(self):
        self.assertEqual(bbox_anchor((10, 20, 30, 100)), (20.0, 100))
        self.assertEqual(bbox_anchor((10, 20, 30, 100), "center"), (20.0, 60.0))
        with self.assertRaises(ValueError):
            bbox_anchor((0, 0, 1, 1), "top")

    def test_normalize_roundtrip(self):
        poly = [(128, 72), (1152, 72), (640, 648)]
        back = denormalize_polygon(normalize_polygon(poly, 1280, 720), 1280, 720)
        for (ax, ay), (bx, by) in zip(poly, back):
            self.assertAlmostEqual(ax, bx)
            self.assertAlmostEqual(ay, by)
        with self.assertRaises(ValueError):
            normalize_polygon(poly, 0, 720)


if __name__ == "__main__":
    unittest.main()
