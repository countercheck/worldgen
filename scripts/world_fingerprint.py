"""Print a hash of every field of a handful of finished worlds, to prove a refactor changed nothing.

The suite asserts that one seed builds one world *within* an implementation; it cannot see
a rewrite that builds a different world just as deterministically. Run this before a change
that should be behaviour-preserving, run it after, and compare:

    python3 scripts/world_fingerprint.py > before.txt
    # ... change ...
    python3 scripts/world_fingerprint.py | diff before.txt -

It walks the whole `WorldState` rather than `to_dict()`, because the JSON leaves out fields
(`journeys`) that later stages read. Floats are hashed by `repr`, which round-trips exactly,
and sets are sorted, because a set of strings iterates in a per-process order.

Hashes are only comparable on one machine: NumPy's AVX-512 paths round differently, so the
same seed builds a different world on a different CPU (see the note in `ci.yml`).
"""

import dataclasses
import hashlib
import sys
import time
from enum import Enum
from pathlib import Path

# Repository root first, so a worktree hashes its own `worldgen/`, not the editable install's.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from tests.worlds import build_pipeline  # noqa: E402

_CHOKE = {
    "model": "organic",
    "grid_layout": "axial",
    "regional_climate": "temperate",
    "continent_falloff_edges": ("south",),
    "chokepoint_min_road_tier": "track",
}

# Chosen to cover both grid layouts, both models, two climates and every routing stage.
WORLDS = {
    "choke-112-s1": dict(seed=1, width=112, height=112, **_CHOKE),
    "choke-112-s8": dict(seed=8, width=112, height=112, **_CHOKE),
    "organic-96-s42": dict(seed=42, width=96, height=96, model="organic"),
    "classic-64-s42": dict(seed=42, width=64, height=64),
    "offset-32-s7": dict(seed=7, width=32, height=32),
    "med-64-s3": dict(seed=3, width=64, height=64, regional_climate="mediterranean"),
}


def _canon(value):
    """A string that is equal for equal values, whatever order containers were filled in."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = (f.name for f in dataclasses.fields(value))
        return f"{type(value).__name__}({','.join(f'{n}={_canon(getattr(value, n))}' for n in fields)})"
    if isinstance(value, dict):
        items = sorted((_canon(k), _canon(v)) for k, v in value.items())
        return "{" + ",".join(f"{k}:{v}" for k, v in items) + "}"
    if isinstance(value, (set, frozenset)):
        return "{" + ",".join(sorted(_canon(v) for v in value)) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_canon(v) for v in value) + "]"
    if isinstance(value, np.ndarray):
        return f"nd{value.dtype}{value.shape}:{hashlib.sha256(value.tobytes()).hexdigest()}"
    if isinstance(value, Enum):
        return f"{type(value).__name__}.{value.name}"
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    return repr(value)


def main() -> None:
    names = sys.argv[1:] or list(WORLDS)
    for name in names:
        started = time.perf_counter()
        state = build_pipeline(**WORLDS[name]).run()
        digest = hashlib.sha256(_canon(state).encode()).hexdigest()[:16]
        print(f"{name:16} {digest}", flush=True)
        print(f"  ({time.perf_counter() - started:.1f}s)", file=sys.stderr)


if __name__ == "__main__":
    main()
