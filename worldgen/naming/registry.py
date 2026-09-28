"""Keeping a map's names apart.

Two towns both called Ashford happen in England, and a reader of a real atlas copes. A
reader of a wargame map does not: an order sent to the wrong Ashford is the whole of the
problem. So exact repeats are refused outright, and near ones — Tharnos and Tharnas —
are refused up to a configurable edit distance, because on a map they read as the same
word misprinted.
"""

from dataclasses import dataclass, field


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance: single-letter insertions, deletions and substitutions."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


@dataclass
class NameRegistry:
    """Every name given so far, and whether a new one stands far enough from all of them."""

    min_distance: int = 2
    _names: list[str] = field(default_factory=list)

    def _key(self, name: str) -> str:
        return name.casefold().replace("-", "").replace(" ", "")

    def taken(self, name: str) -> bool:
        return self._key(name) in {self._key(n) for n in self._names}

    def accepts(self, name: str) -> bool:
        key = self._key(name)
        # A distance only means something against names of about the same length: one
        # letter off a four-letter name is a misprint, one letter off a twelve-letter name
        # is not. The bar is capped at a quarter of the shorter word to say so.
        for other in self._names:
            okey = self._key(other)
            if okey == key:
                return False
            bar = min(self.min_distance, max(1, min(len(key), len(okey)) // 4))
            if abs(len(okey) - len(key)) < bar and edit_distance(key, okey) < bar:
                return False
        return True

    def add(self, name: str) -> None:
        self._names.append(name)

    def __contains__(self, name: str) -> bool:
        return self.taken(name)


__all__ = ["NameRegistry", "edit_distance"]
