"""Label -> category taxonomy (COCO-80 + open-vocabulary extensions).

The taxonomy is the single source of truth for "what kind of thing is this
label". A label the taxonomy does not know maps to ``unclassified`` - the system
never invents a category for an unknown label.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional

from core.types import Category

_H, _A, _V, _F, _O = Category.HUMAN, Category.ANIMAL, Category.VEHICLE, Category.FOOD, Category.OBJECT

COCO_CATEGORY: Dict[str, Category] = {
    "person": _H,
    **{k: _V for k in ("bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat")},
    **{k: _A for k in ("bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe")},
    **{k: _F for k in ("banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
                       "donut", "cake")},
    **{k: _O for k in (
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush")},
}

# Display names tuned for the deployment region ("bike" = motorcycle, "cycle" = bicycle).
DISPLAY_ALIASES: Dict[str, str] = {
    "motorcycle": "bike",
    "bicycle": "cycle",
    "auto rickshaw": "auto-rickshaw",
    "tuk tuk": "auto-rickshaw",
    "e-rickshaw": "auto-rickshaw",
    "tv": "television",
    "cell phone": "phone",
}

DEFAULT_OPEN_VOCAB: Dict[str, List[str]] = {
    "vehicle": ["auto rickshaw", "tuk tuk", "e-rickshaw", "tractor", "van", "ambulance", "scooter", "cart"],
    "animal": ["goat", "buffalo", "camel", "monkey", "pig", "donkey", "deer", "chicken", "duck", "rabbit"],
    "food": ["mango", "watermelon", "grapes", "pineapple", "tomato", "lemon", "papaya", "coconut", "bread", "egg"],
    "object": ["helmet", "gas cylinder", "drone", "traffic cone", "cardboard box"],
}


def normalize_label(label: str) -> str:
    return " ".join(str(label).strip().lower().replace("_", " ").split())


class Taxonomy:
    """Maps detector labels to :class:`Category` and region-friendly display names."""

    def __init__(self, extensions: Optional[Mapping[str, Iterable[str]]] = None) -> None:
        self._map: Dict[str, Category] = dict(COCO_CATEGORY)
        self._extensions: Dict[str, List[str]] = {}
        for cat_name, labels in (extensions or DEFAULT_OPEN_VOCAB).items():
            cat = Category.parse(cat_name)
            norm = [normalize_label(lbl) for lbl in labels if str(lbl).strip()]
            self._extensions[cat.value] = norm
            for lbl in norm:
                self._map.setdefault(lbl, cat)  # never override COCO semantics
        for raw, shown in DISPLAY_ALIASES.items():
            if raw in self._map:
                self._map.setdefault(shown, self._map[raw])

    def category_of(self, label: str) -> Category:
        return self._map.get(normalize_label(label), Category.UNCLASSIFIED)

    def display(self, label: str) -> str:
        n = normalize_label(label)
        return DISPLAY_ALIASES.get(n, n)

    @property
    def open_vocab_prompts(self) -> List[str]:
        return [lbl for labels in self._extensions.values() for lbl in labels]

    def __contains__(self, label: str) -> bool:
        return normalize_label(label) in self._map
