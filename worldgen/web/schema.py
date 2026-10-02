"""The config form, read from the annotated `default_config.yaml`.

The form is not written by hand. `default_config.yaml` already documents every
`WorldConfig` field under a section heading, and `tests/test_config.py` holds it to that,
so reading the sections and help text from it means a field added to the dataclass turns
up in the form, explained, the moment it is documented — and the test makes sure it is.

Defaults come from `WorldConfig()` itself, not from the YAML text: the dataclass is what a
run actually uses.
"""

import re
import types
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, get_args, get_origin

from ..core.config import (
    CLIMATE_CONTEXTS,
    ELEVATION_PROFILE_CHOICES,
    HEIGHTMAP_MODES,
    MODELS,
    WorldConfig,
)
from ..core.hex_grid import GRID_LAYOUTS

_TEMPLATE = Path(__file__).resolve().parent.parent / "default_config.yaml"
_PACKS = Path(__file__).resolve().parent.parent / "naming" / "packs"

# Fields the browser may not set. Each names a path on the machine running the server;
# accepting one from a form would let whoever can reach the page make it read any file.
HIDDEN = frozenset({"heightmap_path", "naming_pack_dirs"})

EDGES = ("north", "south", "east", "west")

_DIVIDER = re.compile(r"^#\s*-{5,}\s*$")
_KEY = re.compile(r"^([a-z_][a-z0-9_]*):(.*)$")
_DEFAULT_NOTE = re.compile(r"\s*Default:.*$")


def _choices() -> dict[str, tuple[str, ...]]:
    packs = tuple(sorted(p.stem for p in _PACKS.glob("*.yaml")))
    return {
        "grid_layout": GRID_LAYOUTS,
        "model": MODELS,
        "heightmap_mode": HEIGHTMAP_MODES,
        "elevation_profile": ELEVATION_PROFILE_CHOICES,
        "regional_climate": tuple(CLIMATE_CONTEXTS),
        "chokepoint_min_road_tier": ("primary", "secondary", "track"),
        "naming_substrate_pack": ("", *packs),
        "naming_packs": packs,
        "continent_falloff_edges": EDGES,
        "river_inflow_edges": EDGES,
    }


def _kind(annotation: Any) -> tuple[str, bool]:
    """The form control a field wants, and whether it may be left empty (None)."""
    nullable = False
    if isinstance(annotation, types.UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        nullable = len(args) < len(get_args(annotation))
        annotation = args[0]
    if get_origin(annotation) is tuple:
        args = get_args(annotation)
        if len(args) == 2 and args[1] is Ellipsis:
            return "list", nullable
        return "pair", nullable
    return {bool: "bool", int: "int", float: "float", str: "str"}[annotation], nullable


def _strip_comment(line: str) -> str:
    return line.lstrip("#").strip()


def _parse_template(text: str) -> list[tuple[str, list[tuple[str, str]]]]:
    """Sections of (name, help) pairs, in file order, up to the `export:` block."""
    sections: list[tuple[str, list[tuple[str, str]]]] = [("General", [])]
    lines = text.splitlines()
    pending: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("export:"):
            break
        # A heading is a comment line fenced by two dividers.
        if _DIVIDER.match(line) and i + 2 < len(lines) and _DIVIDER.match(lines[i + 2]):
            sections.append((_strip_comment(lines[i + 1]), []))
            pending = []
            i += 3
            continue
        if not line.strip():
            pending = []
        elif line.startswith("#"):
            if not _DIVIDER.match(line):
                pending.append(_strip_comment(line))
        elif match := _KEY.match(line):
            name, rest = match.groups()
            inline = rest.split("#", 1)[1].strip() if "#" in rest else ""
            help_text = "\n".join([*pending, _DEFAULT_NOTE.sub("", inline)]).strip()
            sections[-1][1].append((name, help_text))
            pending = []
        i += 1
    return [(title, items) for title, items in sections if items]


def config_schema() -> list[dict]:
    """The form: sections of fields, each with its control, default and help text."""
    declared = {f.name: f for f in fields(WorldConfig)}
    defaults = asdict(WorldConfig())
    choices = _choices()
    out = []
    for title, items in _parse_template(_TEMPLATE.read_text(encoding="utf-8")):
        section = []
        for name, help_text in items:
            if name not in declared or name in HIDDEN:
                continue
            kind, nullable = _kind(declared[name].type)
            default = defaults[name]
            entry = {
                "name": name,
                "kind": kind,
                "nullable": nullable,
                "default": list(default) if isinstance(default, tuple) else default,
                "help": help_text,
            }
            if name in choices:
                entry["choices"] = list(choices[name])
            section.append(entry)
        if section:
            out.append({"title": title, "fields": section})
    return out


def coerce(overrides: Any) -> dict:
    """Check a browser's overrides against the dataclass and return them typed.

    `WorldConfig` validates its enumerated fields but not the types of its numbers, and a
    string where a float belongs would not fail until some stage deep in the run did
    arithmetic on it. The form is the place to catch that, with the field's name attached.
    """
    if not isinstance(overrides, dict):
        raise ValueError("config must be an object of field: value")
    declared = {f.name: f for f in fields(WorldConfig)}
    choices = _choices()
    out: dict[str, Any] = {}
    for name, value in overrides.items():
        if name in HIDDEN:
            raise ValueError(f"{name!r} cannot be set from the web interface")
        if name not in declared:
            # Left for WorldConfig to reject, with its did-you-mean suggestion.
            out[name] = value
            continue
        kind, nullable = _kind(declared[name].type)
        if value is None:
            if not nullable:
                raise ValueError(f"{name} needs a value")
            out[name] = None
            continue
        out[name] = _coerce_one(name, kind, value, choices.get(name))
    return out


def _coerce_one(name: str, kind: str, value: Any, allowed: tuple[str, ...] | None) -> Any:
    def number(v: Any) -> float:
        if isinstance(v, bool) or not isinstance(v, int | float):
            raise ValueError(f"{name} must be a number, got {v!r}")
        return float(v)

    if kind == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be true or false, got {value!r}")
        return value
    if kind == "int":
        v = number(value)
        if not v.is_integer():
            raise ValueError(f"{name} must be a whole number, got {value!r}")
        return int(v)
    if kind == "float":
        return number(value)
    if kind == "pair":
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"{name} must be a pair of numbers")
        return [number(v) for v in value]
    if kind == "list":
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ValueError(f"{name} must be a list of names")
        if allowed is not None and (bad := [v for v in value if v not in allowed]):
            raise ValueError(f"{name}: unknown {', '.join(map(repr, bad))}")
        return value
    if not isinstance(value, str):
        raise ValueError(f"{name} must be text, got {value!r}")
    return value
