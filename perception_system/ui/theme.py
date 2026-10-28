"""Mission-control palette: amber / green / red / white accents ONLY (no blue).

Colours are declared once as RGB hex and converted to OpenCV's BGR order.
``tests/test_theme.py`` asserts that no palette entry has a dominant blue channel.
"""
from __future__ import annotations

from typing import Dict, Tuple

from core.types import Category, Severity

BGR = Tuple[int, int, int]

HEX: Dict[str, str] = {
    "amber": "#FFB000",
    "amber_dim": "#8A5E00",
    "green": "#3DDC5F",
    "green_soft": "#9EE37D",
    "red": "#FF3B30",
    "red_dim": "#6E1E1A",
    "white": "#F2F2F2",
    "grey": "#9A968A",       # warm grey - no cool tint
    "panel": "#121411",
    "bg": "#0A0B0A",
}


def bgr(hex_color: str) -> BGR:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


AMBER, AMBER_DIM, GREEN, GREEN_SOFT, RED, RED_DIM, WHITE, GREY, PANEL, BG = (
    bgr(HEX[k]) for k in ("amber", "amber_dim", "green", "green_soft", "red", "red_dim", "white", "grey", "panel", "bg"))

CATEGORY_COLORS: Dict[Category, BGR] = {
    Category.HUMAN: WHITE, Category.ANIMAL: GREEN, Category.VEHICLE: AMBER,
    Category.FOOD: GREEN_SOFT, Category.OBJECT: GREY, Category.UNCLASSIFIED: RED,
}
SEVERITY_COLORS: Dict[Severity, BGR] = {Severity.INFO: WHITE, Severity.WARNING: AMBER, Severity.CRITICAL: RED}
ZONE_COLORS: Dict[str, BGR] = {"amber": AMBER, "green": GREEN, "red": RED, "white": WHITE}
EMOTION_HEX: Dict[str, str] = {
    "happy": HEX["green"], "neutral": HEX["white"], "surprise": HEX["amber"], "sad": HEX["grey"],
    "fear": HEX["amber_dim"], "angry": HEX["red"], "disgust": HEX["red_dim"],
}
EMOTION_COLORS: Dict[str, BGR] = {k: bgr(v) for k, v in EMOTION_HEX.items()}
CATEGORY_HEX: Dict[str, str] = {
    "human": HEX["white"], "animal": HEX["green"], "vehicle": HEX["amber"],
    "food": HEX["green_soft"], "object": HEX["grey"], "unclassified": HEX["red"],
}
