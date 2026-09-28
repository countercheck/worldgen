"""What a culture pack is, and the checks that turn a bad pack file into a clear message.

A pack is data — the words a people named places with, and how they put them together —
and it arrives as a YAML mapping, usually written by hand and sometimes by someone who
has never seen this code. So every rule the processor relies on is checked here, once,
at load, with the file and the field in the message: "welsh.yaml: heads is missing
'ford'" rather than a KeyError three stages into a run.

Pure: this module parses a mapping it is handed. Reading the file is
`export.culture_packs`'s business.

The format is documented for pack authors in `docs/CULTURE_PACKS.md`.
"""

import hashlib
import json
import re
import string
from collections.abc import Mapping
from dataclasses import dataclass, field

from ..site import GENERICS, SPECIFICS


class CulturePackError(ValueError):
    """A culture pack that cannot be used, and why."""


@dataclass(frozen=True)
class ProperNames:
    """How a pack makes the names that mean nothing: a founder's, a river's.

    ``lists`` builds a founder's name as first + second from two lists, and a river's by
    putting a root into a pattern. ``syllables`` builds either from registers of onsets,
    vowels and codas, the register chosen by rank — a queen's hive buzzes, an outpost
    clicks.
    """

    style: str
    person_first: tuple[str, ...] = ()
    person_second: tuple[str, ...] = ("",)
    river_roots: tuple[str, ...] = ()
    river_patterns: tuple[str, ...] = ("{root}",)
    # register name -> (onsets, vowels, codas)
    registers: Mapping[str, tuple[tuple[str, ...], ...]] = field(default_factory=dict)
    # For ranks 0, 1 and 2, the registers a name may be drawn in.
    by_rank: tuple[tuple[str, ...], ...] = ()


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
    proper: ProperNames
    description: str = ""
    bare: tuple[str, ...] = ("{h}",)
    # Templates for a qualifier that is an agreeing adjective (its word carries "~"), in
    # place of `compound`. Where the two differ — Fuente Blanca and Villablanca, but
    # Fuente de los Robles — a noun phrase must not be run into the head.
    adjective: tuple[str, ...] = ()
    # Template kinds that differ by rank: three tuples, for rank 0 (city), 1 and 2.
    ranked: Mapping[str, tuple[tuple[str, ...], ...]] = field(default_factory=dict)
    # Words left in lower case inside a name: of, on, the.
    particles: frozenset[str] = frozenset()
    # Grammatical gender of each head word, and the adjective ending each gender takes.
    genders: Mapping[str, str] = field(default_factory=dict)
    endings: Mapping[str, str] = field(default_factory=dict)
    # Whether a part after a hyphen is capitalised: Gué-du-Chêne, but Khazad-dûm.
    hyphen_caps: bool = True
    # (regex, replacement) pairs applied to a finished name.
    fixes: tuple[tuple[str, str], ...] = ()
    # How a founder is glossed in the etymology, over {p} the founder and {h} the head.
    founder_gloss: str = "{p}'s {h}"
    # Meaning -> a note on the word: that it is invented, or means something else.
    notes: Mapping[str, str] = field(default_factory=dict)


_TOP = {
    "key",
    "name",
    "description",
    "heads",
    "quals",
    "templates",
    "ranked",
    "agreement",
    "proper_names",
    "spelling",
    "founder_gloss",
    "notes",
}
_REQUIRED = ("key", "name", "heads", "quals", "templates", "proper_names")
# The placeholders each kind of template may use.
_PLACEHOLDERS = {
    "compound": {"h", "q", "a"},
    "adjective": {"h", "q", "a"},
    "founder": {"h", "p"},
    "river_on": {"h", "r"},
    "bare": {"h"},
}
_KEY = re.compile(r"^[a-z][a-z0-9_]*$")


def pack_hash(data: Mapping) -> str:
    """A short fingerprint of a pack's content.

    Taken over the parsed mapping, not the file text, so re-wrapping a list or editing a
    comment leaves it alone and changing a word does not.
    """
    canonical = json.dumps(data, sort_keys=True, ensure_ascii=False, default=list)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


class _Reader:
    """Field access that knows which file and field it is reading, for its errors."""

    def __init__(self, source: str):
        self.source = source

    def fail(self, where: str, message: str):
        raise CulturePackError(f"{self.source}: {where} {message}")

    def mapping(self, value, where: str) -> Mapping:
        if not isinstance(value, Mapping):
            self.fail(where, f"must be a mapping, got {type(value).__name__}")
        return value

    def text(self, value, where: str, allow_empty: bool = False) -> str:
        if isinstance(value, bool):
            # YAML 1.1 reads a bare on, off, yes or no as a boolean — the commonest way a
            # hand-written pack goes wrong, since "on" is a particle half the packs need.
            self.fail(
                where,
                f"is {str(value).lower()}, not a word: YAML reads a bare on, off, yes or "
                "no as true or false — put the word in quotes",
            )
        if not isinstance(value, str):
            self.fail(where, f"must be text, got {type(value).__name__}")
        if not value and not allow_empty:
            self.fail(where, "must not be empty")
        return value

    def words(self, value, where: str, allow_empty_words: bool = False) -> tuple[str, ...]:
        if not isinstance(value, list) or not value:
            self.fail(where, "must be a non-empty list")
        return tuple(
            self.text(v, f"{where}[{i}]", allow_empty=allow_empty_words)
            for i, v in enumerate(value)
        )

    def templates(self, value, where: str, kind: str) -> tuple[str, ...]:
        out = self.words(value, where)
        allowed = _PLACEHOLDERS[kind]
        for i, template in enumerate(out):
            used = self.placeholders(template, f"{where}[{i}]")
            if used - allowed:
                self.fail(
                    f"{where}[{i}]",
                    f"uses {', '.join('{' + u + '}' for u in sorted(used - allowed))}; "
                    f"a {kind} template may use only "
                    f"{', '.join('{' + a + '}' for a in sorted(allowed))}",
                )
        return out

    def placeholders(self, template: str, where: str) -> set[str]:
        try:
            return {f for _, f, _, _ in string.Formatter().parse(template) if f is not None}
        except ValueError as exc:
            self.fail(where, f"is not a valid template ({exc})")

    def meanings(self, value, where: str, expected) -> dict[str, tuple[str, ...]]:
        value = self.mapping(value, where)
        missing = [k for k in expected if k not in value]
        unknown = sorted(set(value) - set(expected))
        if missing or unknown:
            parts = []
            if missing:
                parts.append("is missing " + ", ".join(repr(k) for k in missing))
            if unknown:
                parts.append("has unknown " + ", ".join(repr(k) for k in unknown))
            self.fail(where, " and ".join(parts))
        return {k: self.words(value[k], f"{where}.{k}") for k in expected}


def parse_pack(data, source: str) -> Pack:
    """Check a loaded pack mapping and build the `Pack` it describes.

    *source* names where it came from — a file name — and heads every error message.
    """
    r = _Reader(source)
    data = r.mapping(data, "the file")
    unknown = sorted(set(data) - _TOP)
    if unknown:
        r.fail("the file", "has unknown key " + ", ".join(repr(k) for k in unknown))
    for name in _REQUIRED:
        if name not in data:
            r.fail("the file", f"is missing {name!r}")

    key = r.text(data["key"], "key")
    if not _KEY.match(key):
        r.fail("key", f"{key!r} must be lower-case letters, digits and underscores")

    heads = r.meanings(data["heads"], "heads", GENERICS)
    quals = r.meanings(data["quals"], "quals", SPECIFICS)

    templates = r.mapping(data["templates"], "templates")
    extra = sorted(set(templates) - set(_PLACEHOLDERS))
    if extra:
        r.fail("templates", "has unknown kind " + ", ".join(repr(k) for k in extra))
    for need in ("compound", "founder", "river_on"):
        if need not in templates:
            r.fail("templates", f"is missing {need!r}")
    kinds = {
        kind: r.templates(templates[kind], f"templates.{kind}", kind)
        for kind in _PLACEHOLDERS
        if kind in templates
    }

    ranked = {}
    for kind, ranks in r.mapping(data.get("ranked", {}), "ranked").items():
        if kind not in _PLACEHOLDERS:
            r.fail(f"ranked.{kind}", "is not a template kind")
        if not isinstance(ranks, list) or len(ranks) != 3:
            r.fail(f"ranked.{kind}", "must list exactly three template lists, for ranks 0, 1, 2")
        ranked[kind] = tuple(
            r.templates(t, f"ranked.{kind}[{i}]", kind) for i, t in enumerate(ranks)
        )

    agreement = r.mapping(data.get("agreement", {}), "agreement")
    genders = {
        k: r.text(v, f"agreement.genders.{k}")
        for k, v in r.mapping(agreement.get("genders", {}), "agreement.genders").items()
    }
    endings = {
        k: r.text(v, f"agreement.endings.{k}", allow_empty=True)
        for k, v in r.mapping(agreement.get("endings", {}), "agreement.endings").items()
    }
    agreeing = any("~" in w for words in quals.values() for w in words)
    if agreeing and not endings:
        r.fail("quals", "use '~' for an agreeing ending, but agreement.endings is not given")
    if agreeing:
        for word in sorted({w for words in heads.values() for w in words}):
            if word not in genders:
                r.fail("agreement.genders", f"has no gender for head word {word!r}")
    for word, gender in genders.items():
        if gender not in endings:
            r.fail(f"agreement.genders.{word}", f"is {gender!r}, which has no ending")

    proper = _proper(r, data["proper_names"])

    spelling = r.mapping(data.get("spelling", {}), "spelling")
    # An empty list is fine here: plenty of peoples write no little words into names.
    raw_particles = spelling.get("particles") or []
    particles = frozenset(r.words(raw_particles, "spelling.particles") if raw_particles else ())
    hyphen_caps = spelling.get("hyphen_caps", True)
    if not isinstance(hyphen_caps, bool):
        r.fail("spelling.hyphen_caps", "must be true or false")
    fixes = []
    for i, pair in enumerate(spelling.get("fixes", []) or []):
        if not isinstance(pair, list) or len(pair) != 2:
            r.fail(f"spelling.fixes[{i}]", "must be a [pattern, replacement] pair")
        pattern = r.text(pair[0], f"spelling.fixes[{i}][0]")
        replacement = r.text(pair[1], f"spelling.fixes[{i}][1]", allow_empty=True)
        try:
            re.compile(pattern)
        except re.error as exc:
            r.fail(f"spelling.fixes[{i}]", f"pattern {pattern!r} does not compile ({exc})")
        fixes.append((pattern, replacement))

    founder_gloss = r.text(data.get("founder_gloss", "{p}'s {h}"), "founder_gloss")
    if r.placeholders(founder_gloss, "founder_gloss") - {"p", "h"}:
        r.fail("founder_gloss", "may use only {p} and {h}")

    known = set(GENERICS) | set(SPECIFICS)
    notes = {}
    for k, v in r.mapping(data.get("notes", {}), "notes").items():
        if k not in known:
            r.fail(f"notes.{k}", "is not a meaning in heads or quals")
        notes[k] = r.text(v, f"notes.{k}")

    return Pack(
        key=key,
        name=r.text(data["name"], "name"),
        description=r.text(data.get("description", ""), "description", allow_empty=True),
        heads=heads,
        quals=quals,
        compound=kinds["compound"],
        founder=kinds["founder"],
        river_on=kinds["river_on"],
        bare=kinds.get("bare", ("{h}",)),
        adjective=kinds.get("adjective", ()),
        ranked=ranked,
        proper=proper,
        particles=particles,
        genders=genders,
        endings=endings,
        hyphen_caps=hyphen_caps,
        fixes=tuple(fixes),
        founder_gloss=founder_gloss,
        notes=notes,
    )


def _proper(r: _Reader, value) -> ProperNames:
    value = r.mapping(value, "proper_names")
    style = r.text(value.get("style", ""), "proper_names.style", allow_empty=True)
    if style == "lists":
        persons = r.mapping(value.get("persons", {}), "proper_names.persons")
        rivers = r.mapping(value.get("rivers", {}), "proper_names.rivers")
        patterns = r.words(rivers.get("patterns", ["{root}"]), "proper_names.rivers.patterns")
        for i, pattern in enumerate(patterns):
            if r.placeholders(pattern, f"proper_names.rivers.patterns[{i}]") != {"root"}:
                r.fail(f"proper_names.rivers.patterns[{i}]", "must use {root} and nothing else")
        return ProperNames(
            style=style,
            person_first=r.words(persons.get("first"), "proper_names.persons.first"),
            person_second=r.words(
                persons.get("second", [""]), "proper_names.persons.second", allow_empty_words=True
            ),
            river_roots=r.words(rivers.get("roots"), "proper_names.rivers.roots"),
            river_patterns=patterns,
        )
    if style == "syllables":
        registers = {}
        for name, reg in r.mapping(value.get("registers", {}), "proper_names.registers").items():
            where = f"proper_names.registers.{name}"
            reg = r.mapping(reg, where)
            registers[name] = tuple(
                r.words(reg.get(part), f"{where}.{part}", allow_empty_words=True)
                for part in ("onsets", "vowels", "codas")
            )
        if not registers:
            r.fail("proper_names.registers", "must name at least one register")
        by_rank = value.get("by_rank")
        if not isinstance(by_rank, list) or len(by_rank) != 3:
            r.fail("proper_names.by_rank", "must list exactly three entries, for ranks 0, 1, 2")
        ranks = []
        for i, entry in enumerate(by_rank):
            names = r.words(entry, f"proper_names.by_rank[{i}]")
            for n in names:
                if n not in registers:
                    r.fail(f"proper_names.by_rank[{i}]", f"names unknown register {n!r}")
            ranks.append(names)
        return ProperNames(style=style, registers=registers, by_rank=tuple(ranks))
    r.fail("proper_names.style", f"is {style!r}; use 'lists' or 'syllables'")


__all__ = ["CulturePackError", "Pack", "ProperNames", "pack_hash", "parse_pack"]
