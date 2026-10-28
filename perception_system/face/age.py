"""Age -> coarse age-group bucketing (approximate by design: +/- 5-8 years typical error)."""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple


class AgeGrouper:
    def __init__(self, groups: Sequence[Tuple[int, int, str]]) -> None:
        ordered: List[Tuple[int, int, str]] = sorted((int(lo), int(hi), str(name)) for lo, hi, name in groups)
        if not ordered:
            raise ValueError("at least one age group is required")
        for lo, hi, name in ordered:
            if lo > hi:
                raise ValueError(f"age group {name!r}: lower bound exceeds upper bound")
        for (_, hi_a, a), (lo_b, _, b) in zip(ordered, ordered[1:]):
            if lo_b <= hi_a:
                raise ValueError(f"age groups {a!r} and {b!r} overlap")
        self.groups = ordered

    def group(self, age: Optional[float]) -> Optional[str]:
        if age is None or not math.isfinite(age) or age < 0:
            return None
        a = int(round(age))
        for lo, hi, name in self.groups:
            if lo <= a <= hi:
                return name
        return None
