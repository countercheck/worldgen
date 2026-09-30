"""The one processor every culture pack runs through.

A pack is data (see `schema`); this turns it into the `Culture` protocol the stage speaks,
so a pack and a generated language are interchangeable region by region, and any pack a
user writes in YAML is run exactly as the built-in ones are.

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

import numpy as np

from ..culture import Qualifier
from .schema import Pack

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
            kind = "adjective" if "~" in word and pack.adjective else "compound"
            template = _pick(self._templates(kind, rank), rng)
        elif qualifier.relation == "on":
            parts["r"] = qualifier.value.lower()
            template = _pick(self._templates("river_on", rank), rng)
        else:
            parts["p"] = qualifier.value.lower()
            template = _pick(self._templates("founder", rank), rng)
        return self._title(template.format(**parts))

    def _persons(self) -> list[str]:
        proper = self.pack.proper
        return [
            f + s
            for f in proper.person_first
            for s in proper.person_second
            if f.lower() != s.lower()
        ]

    def _rivers(self) -> list[str]:
        proper = self.pack.proper
        return [
            pattern.format(root=root)
            for pattern in proper.river_patterns
            for root in proper.river_roots
        ]

    def _syllables(self, key: str, syllables: int, rank: int) -> str:
        """A word from the registers the rank allows, stable for its key.

        A random draw picks the register only where the rank allows more than one, so a
        pack with one register per rank spends no draw on it.
        """
        proper = self.pack.proper
        rng = np.random.default_rng([zlib.crc32(key.encode()), max(1, syllables), rank])
        choices = proper.by_rank[min(max(rank, 0), len(proper.by_rank) - 1)]
        register = choices[0] if len(choices) == 1 else choices[int(rng.random() * len(choices))]
        onsets, vowels, codas = proper.registers[register]
        return "".join(
            onsets[int(rng.integers(len(onsets)))]
            + vowels[int(rng.integers(len(vowels)))]
            + codas[int(rng.integers(len(codas)))]
            for _ in range(max(1, syllables))
        )

    def proper_name(self, key: str, syllables: int, rank: int = 1) -> str:
        """A founder's or a river's name, stable for a key.

        For a pack of lists, *syllables* means nothing — its names are whole words — so
        it is used to step to the next name instead, which is what a caller raising it is
        asking for: something different. Past the end of the list, two names are joined,
        so a caller that keeps asking always eventually gets one it has not seen.
        """
        kind = key.split(":", 1)[0]
        if kind == "people":
            return self.name
        if self.pack.proper.style == "syllables":
            return self._title(self._syllables(key, syllables, rank))
        names = self._rivers() if kind == "river" else self._persons()
        h = zlib.crc32(key.encode())
        n = len(names)
        first = names[(h + syllables) % n]
        if syllables < n:
            return self._title(first)
        return self._title(first + names[(h // n + syllables // n) % n].lower())


__all__ = ["PackCulture"]
