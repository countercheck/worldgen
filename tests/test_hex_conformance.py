"""The committed hex conformance fixture must match what this Python actually computes.

The fixture is the contract between `worldgen/core/hex_grid.py` and its TypeScript port
at `campaign/shared/src/hex.ts`. The TS suite asserts against the committed file; this
test asserts the committed file still describes the Python.

Both halves are needed. Without this one, a change to the hex model here would leave a
stale fixture that the TS still passes against, and the two implementations would part
company without any test noticing — which is the exact failure the fixture exists to
prevent.

A failure here means either a deliberate change to the hex model (regenerate with
`python3 scripts/gen_hex_fixture.py`, and expect the TS suite to fail until the port is
updated to match) or an accidental one (fix the code, not the fixture).
"""

import json
import sys
from pathlib import Path

import pytest

# The generator is a script rather than a package, because it is a build step and not
# part of the library. Importing it here keeps the fixture's definition in one place.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from gen_hex_fixture import FIXTURE, build  # noqa: E402


@pytest.fixture(scope="module")
def committed():
    assert FIXTURE.is_file(), (
        f"{FIXTURE} is missing — run `python3 scripts/gen_hex_fixture.py` and commit it"
    )
    return json.loads(FIXTURE.read_text())


def test_the_fixture_matches_this_python(committed):
    """The whole document, so a drift anywhere in the hex model is caught."""
    fresh = build()
    assert committed == fresh, (
        "The committed hex fixture no longer matches this Python. If the hex model "
        "changed on purpose, run `python3 scripts/gen_hex_fixture.py`, commit the "
        "result, and update campaign/shared/src/hex.ts to match."
    )


def test_the_fixture_covers_the_rounding_trap(committed):
    """Half-to-even is the one place a naive port silently disagrees.

    Python's `round()` rounds half to even; JavaScript's `Math.round` rounds half up.
    The difference only appears on an exact .5, which is what a point sitting on a hex
    boundary produces — so a port using `Math.round` puts boundary clicks in the wrong
    hex and nothing else looks wrong. The fixture is only a guard if it contains those
    cases, so assert they are present rather than trusting the generator.
    """
    halves = [c for c in committed["roundAxial"] if abs(c["q"]) == 0.5 or abs(c["r"]) == 0.5]
    assert len(halves) >= 4, "the fixture must exercise exact .5 rounding"


def test_the_pathfinding_case_actually_detours(committed):
    """A path that runs straight to the goal would not exercise the frontier at all.

    The obstacle is there so the search has to choose between equal-cost routes, which
    is what the heap's tie-break decides — and the tie-break is the part of A* two
    implementations most easily disagree about.
    """
    astar_case = committed["astar"]
    path = astar_case["path"]
    assert path is not None, "the fixture's A* case must find a path"

    start, goal = astar_case["start"], astar_case["goal"]
    assert path[0] == start and path[-1] == goal

    straight = (
        abs(goal[0] - start[0])
        + abs(goal[1] - start[1])
        + abs(goal[0] + goal[1] - start[0] - start[1])
    ) // 2
    assert len(path) > straight + 1, "the path should be forced around the obstacle"

    blocked = {(astar_case["blockedColumn"], r) for r in astar_case["blockedRows"]}
    assert not blocked & {tuple(c) for c in path}, "the path runs through the obstacle"
