"""What a naming culture has to be able to do.

A protocol rather than a base class, so a hand-written culture pack — a table of real
Old English or Norse elements — and a generated language can stand in for each other
without sharing any machinery. Neither needs to know how the other spells.
"""

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np


@dataclass(frozen=True)
class Qualifier:
    """What a place name's head is qualified by.

    ``kind`` is ``"meaning"`` when *value* is a key from `site.SPECIFICS` — the culture
    translates it — and ``"proper"`` when *value* is a name already spelled, a river's or a
    founder's, which the culture uses as it stands. That distinction is what lets a town
    carry a river name from an older language: the substrate spelled it, and the newcomers
    only borrowed it.

    ``relation`` is ``"of"`` for the ordinary compound (the oxen's ford, Tarin's farm) and
    ``"on"`` for a place named by the river it stands on (Stratford-on-Avon, Avonmouth).
    """

    kind: Literal["meaning", "proper"]
    value: str
    relation: Literal["of", "on"] = "of"


class Culture(Protocol):
    """A people's way of naming places."""

    name: str

    def place_name(
        self,
        generic: str,
        qualifier: Qualifier | None,
        rng: np.random.Generator,
        rank: int = 1,
    ) -> str:
        """A place name with *generic* as its head, qualified or bare.

        *rank* is how high the place stands — 0 a city, 1 a town, 2 a village. Most
        cultures name every rank alike and ignore it; one that does not (a hive, whose
        queen's seat is named with ceremony and whose outposts are not) reads it.
        """
        ...

    def proper_name(self, key: str, syllables: int, rank: int = 1) -> str:
        """A name that means nothing any more — a river's, a founder's.

        The same *key* gives the same name every time in one culture. *rank* is as for
        `place_name`; for a river, 0 is a great river and 2 a lesser one.
        """
        ...


__all__ = ["Culture", "Qualifier"]
