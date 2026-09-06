"""The generated TypeScript artefacts must still match the Python they came from.

Two files under `campaign/` are generated from this package and committed:

- `campaign/shared/src/palette.ts`, from `worldgen/render/debug_viewer.py`
- `campaign/shared/test/fixtures/world-32x32.json`, from the generator itself

Committing generated output is what lets the browser build run without Python. The risk
is the obvious one: the source moves, nobody re-runs the generator, and the committed copy
quietly becomes a lie. A new biome colour would not break anything loudly — hexes would
just draw in the fallback shade on the web map while the SVG exports showed the real one.

So the check is here, where the source lives, and it fails at the moment the Python
changes rather than whenever somebody next happens to look at the map.

The world fixture is checked only for its shape, not byte for byte. It is a full pipeline
run and pinning its exact contents would turn every deliberate change to worldgen into a
failure in the campaign tests, which is noise rather than signal — `tests/worlds.py` and
the stage suites are where generator behaviour is pinned. What matters here is that the
fixture still exercises the features the TypeScript tests rely on it having.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PALETTE = ROOT / "campaign" / "shared" / "src" / "palette.ts"
WORLD = ROOT / "campaign" / "shared" / "test" / "fixtures" / "world-32x32.json"

# The generators are scripts rather than a package, because they are a build step and not
# part of the library. Importing them keeps each artefact's definition in one place.
sys.path.insert(0, str(ROOT / "scripts"))

from export_palette import render as render_palette  # noqa: E402


def test_the_committed_palette_matches_this_python():
    assert PALETTE.is_file(), f"{PALETTE} is missing — run scripts/export_palette.py"
    assert PALETTE.read_text() == render_palette(), (
        "campaign/shared/src/palette.ts is stale. The renderer's colours changed and the "
        "web map would still be drawing the old ones. Run "
        "`python3 scripts/export_palette.py` and commit the result."
    )


@pytest.fixture(scope="module")
def world():
    assert WORLD.is_file(), f"{WORLD} is missing — run scripts/make_world_fixture.py"
    return json.loads(WORLD.read_text())


def test_the_world_fixture_is_a_schema_the_typescript_reads(world):
    """The TS parser pins an explicit supported set and refuses anything else."""
    from worldgen.core.world_state import SCHEMA_VERSION

    assert world["version"] == SCHEMA_VERSION, (
        f"the fixture is schema {world['version']} but the generator now writes "
        f"{SCHEMA_VERSION}. Run `python3 scripts/make_world_fixture.py`, and widen "
        f"SUPPORTED_SCHEMA_VERSIONS in campaign/shared/src/world.ts to match."
    )


def test_the_world_fixture_still_exercises_what_the_typescript_tests_need(world):
    """Each of these is something a TS test asserts on. Losing one makes it vacuous.

    A fixture that quietly stops containing roads does not fail the TypeScript suite —
    the loops asserting over road edges simply run zero times and pass. That is the
    failure this test exists to make loud.
    """
    tiers = {e["tier"] for e in world["road_edges"]}
    assert tiers == {"primary", "secondary", "track"}, (
        f"the fixture must carry all three road tiers for the movement grade rules; got {tiers}"
    )

    terrain = {h["terrain_class"] for h in world["hexes"]}
    assert {"land", "coast", "open_water"} <= terrain, f"missing terrain classes: got {terrain}"

    tags = {t for h in world["hexes"] for t in h["tags"]}
    assert "ford" in tags, (
        "the fixture must carry ford tags — they only exist in the organic model, and the "
        "river-crossing rules key on them"
    )
    assert "river" in tags, "the fixture must contain a river"

    assert world["settlements"], "the fixture must contain settlements"
    assert world["sea_edges"], "the fixture must contain sea edges"

    deltas = [e["delta_elevation_m"] for e in world["road_edges"]]
    assert max(deltas) > 0 and min(deltas) < 0, (
        "roads must both climb and descend, or the on-road grade term is untested"
    )
