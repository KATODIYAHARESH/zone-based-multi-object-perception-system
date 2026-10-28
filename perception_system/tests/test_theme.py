"""House palette rule: amber / green / red / white accents only - no blue."""
import unittest

from ui.theme import CATEGORY_HEX, EMOTION_HEX, HEX, ZONE_COLORS, bgr


class PaletteTests(unittest.TestCase):
    def test_no_colour_is_blue_dominant(self):
        for name, value in {**HEX, **CATEGORY_HEX, **EMOTION_HEX}.items():
            b, g, r = bgr(value)
            self.assertLessEqual(b, max(g, r), f"{name}={value} has a dominant blue channel")

    def test_bgr_conversion(self):
        self.assertEqual(bgr("#FFB000"), (0, 176, 255))

    def test_zone_palette_excludes_blue(self):
        self.assertEqual(set(ZONE_COLORS), {"amber", "green", "red", "white"})


if __name__ == "__main__":
    unittest.main()
