"""Culture packs: naming cultures written by hand from real place-name elements.

A generated language is endless and alien. A pack is the other thing a map sometimes
wants: names a reader half-recognises, because they are built from the elements real
places are built from — English -ford and -ton, Norse -by and -thwaite, Welsh aber- and
pont-, French -ville and plessis, Slavic brod and hradište, Arabic jisr and kafr.

Each pack is data: for every meaning in `site.GENERICS` and `site.SPECIFICS`, the words
that culture used for it, and a handful of templates for how those words go together.
`PackCulture` turns that data into the `Culture` protocol the stage speaks, so a pack and
a generated language are interchangeable region by region.

Templates are `str.format` strings over lower-case parts, title-cased at the end:

    {h}   the head, a word from `heads`
    {q}   a qualifier from `quals`, used as it stands
    {a}   the same qualifier agreeing with the head: a trailing "~" in the qualifier is
          replaced by the ending for the head's gender (Slavic adjectives)
    {p}   a founder's name
    {r}   a river's name

A qualifier's "~" is simply dropped where it is used as {q}, which is how a Slavic stem
makes a one-word name — dubov~ gives Dubovec as well as Dubová Hora.
"""

import re
import zlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np

from ..culture import Qualifier


@dataclass(frozen=True)
class Pack:
    key: str
    # What the people are called, and so what their region is called on the map.
    name: str
    heads: Mapping[str, tuple[str, ...]]
    quals: Mapping[str, tuple[str, ...]]
    # How a head and a qualifier combine; one is drawn per name.
    compound: tuple[str, ...]
    # How a head and a founder's name combine.
    founder: tuple[str, ...]
    # How a head and the river it stands on combine.
    river_on: tuple[str, ...]
    # Personal names are built as first + second, as most early naming systems built them.
    person_first: tuple[str, ...]
    river_roots: tuple[str, ...]
    person_second: tuple[str, ...] = ("",)
    # How a river root becomes a river's name; {root} is the root.
    river_patterns: tuple[str, ...] = ("{root}",)
    bare: tuple[str, ...] = ("{h}",)
    # Words left in lower case inside a name: of, on, the.
    particles: frozenset[str] = frozenset()
    # Grammatical gender of each head word, and the adjective ending each gender takes.
    genders: Mapping[str, str] = field(default_factory=dict)
    endings: Mapping[str, str] = field(default_factory=dict)
    # Whether a part after a hyphen is capitalised: Gué-du-Chêne, but Khazad-dûm.
    hyphen_caps: bool = True
    # Spelling fixes applied to a finished name, as (regex, replacement) pairs — Quenya
    # writes a final ë but drops the diaeresis once the word is inside a compound.
    fixes: tuple[tuple[str, str], ...] = ()
    # Templates that differ by rank, keyed "compound", "founder", "river_on" or "bare",
    # each a tuple of three template tuples for rank 0 (city), 1 (town) and 2 (village).
    # A kind not listed uses the rank-blind field of the same name.
    ranked: Mapping[str, tuple[tuple[str, ...], ...]] = field(default_factory=dict)
    # A pack that makes its own proper names — (key, syllables, rank) -> name — in place
    # of drawing them from `person_first` and `river_roots`.
    proper: Callable[[str, int, int], str] | None = None
    # How a founder is glossed in the etymology, over {p} the founder and {h} the head.
    founder_gloss: str = "{p}'s {h}"


_SEPARATORS = re.compile(r"([ -])")
_ELIDED = re.compile(r"^([ld])'(.)(.*)$")


def _pick(options: tuple[str, ...], rng: np.random.Generator) -> str:
    return options[int(rng.integers(len(options)))]


class PackCulture:
    """A `Culture` that names from a `Pack`."""

    def __init__(self, pack: Pack):
        self.pack = pack
        self.name = pack.name
        self.founder_gloss = pack.founder_gloss

    def _templates(self, kind: str, rank: int) -> tuple[str, ...]:
        by_rank = self.pack.ranked.get(kind)
        if by_rank is not None:
            return by_rank[min(max(rank, 0), len(by_rank) - 1)]
        return getattr(self.pack, kind)

    def _title(self, text: str) -> str:
        parts = _SEPARATORS.split(text)
        out = []
        for i, part in enumerate(parts):
            # A particle, or anything after a hyphen in a pack that does not capitalise
            # there, is left as it stands.
            keep = part in self.pack.particles or (
                parts[i - 1] == "-" and not self.pack.hyphen_caps
            )
            if i and keep:
                out.append(part)
            elif m := _ELIDED.match(part):
                out.append(m.group(1) + "'" + m.group(2).upper() + m.group(3))
            else:
                out.append(part[:1].upper() + part[1:])
        name = "".join(out)
        for pattern, replacement in self.pack.fixes:
            name = re.sub(pattern, replacement, name)
        # Three of a letter running together is always a join, never a word.
        return re.sub(r"(.)\1\1", r"\1\1", name)

    def place_name(
        self,
        generic: str,
        qualifier: Qualifier | None,
        rng: np.random.Generator,
        rank: int = 1,
    ) -> str:
        pack = self.pack
        head = _pick(pack.heads[generic], rng)
        parts = {"h": head.lower()}
        if qualifier is None:
            template = _pick(self._templates("bare", rank), rng)
        elif qualifier.kind == "meaning":
            word = _pick(pack.quals[qualifier.value], rng).lower()
            ending = pack.endings.get(pack.genders.get(head, ""), "")
            parts["q"] = word.replace("~", "")
            parts["a"] = word.replace("~", ending)
            template = _pick(self._templates("compound", rank), rng)
        elif qualifier.relation == "on":
            parts["r"] = qualifier.value.lower()
            template = _pick(self._templates("river_on", rank), rng)
        else:
            parts["p"] = qualifier.value.lower()
            template = _pick(self._templates("founder", rank), rng)
        return self._title(template.format(**parts))

    def _persons(self) -> list[str]:
        return [
            f + s
            for f in self.pack.person_first
            for s in self.pack.person_second
            if f.lower() != s.lower()
        ]

    def _rivers(self) -> list[str]:
        return [
            pattern.format(root=root)
            for pattern in self.pack.river_patterns
            for root in self.pack.river_roots
        ]

    def proper_name(self, key: str, syllables: int, rank: int = 1) -> str:
        """A founder's or a river's name, stable for a key.

        *syllables* means nothing to a pack — its names are whole words — so it is used to
        step to the next name instead, which is what a caller raising it is asking for:
        something different. Past the end of the list, two names are joined, so a caller
        that keeps asking always eventually gets one it has not seen.
        """
        kind = key.split(":", 1)[0]
        if kind == "people":
            return self.name
        if self.pack.proper is not None:
            return self._title(self.pack.proper(key, syllables, rank))
        names = self._rivers() if kind == "river" else self._persons()
        h = zlib.crc32(key.encode())
        n = len(names)
        first = names[(h + syllables) % n]
        if syllables < n:
            return self._title(first)
        return self._title(first + names[(h // n + syllables // n) % n].lower())


__all__ = ["Pack", "PackCulture"]
