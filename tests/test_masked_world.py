"""A commander's map must still be a world this package can read.

The campaign layer exports a faction's partial map as a `world.json` in the generator's
own schema, so that a referee can open it in the existing tooling:

    worldgen render --input fog.json --attribute biome --output fog.svg

That claim is made in TypeScript, and TypeScript cannot check it. `WorldState.from_dict`
is the only thing that can say whether a document loads, and the renderers are the only
things that can say whether it draws — so the masked file is produced by the real masking
code, over in Node, and put through the real loader here.

This is the test that would catch the mistake the masking design exists to avoid: omitting
unseen hexes rather than blanking them. `from_dict` requires six fields on every hex, and
every renderer sizes its canvas from the hexes present, so an omitted hex either fails to
load or silently rescales the map. Both would sail past the TypeScript suite.

The Node build is a prerequisite. Where it is unavailable the test skips loudly rather
than passing quietly, because a silently skipped cross-language test is worse than none.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from worldgen.core.world_state import SCHEMA_VERSION, WorldState
from worldgen.export import json_export
from worldgen.render import debug_viewer

ROOT = Path(__file__).resolve().parent.parent
BRIDGE = ROOT / "scripts" / "emit_masked_world.mjs"
CAMPAIGN = ROOT / "campaign"
FOG_TAG = "fog"
REMEMBERED_TAG = "remembered"


@pytest.fixture(scope="module")
def masked_json():
    """Run the real TypeScript masking and hand back what it produced."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; cannot exercise the TypeScript masking")
    if not (CAMPAIGN / "node_modules").is_dir():
        pytest.skip("campaign/node_modules is missing; run `npm --prefix campaign install`")

    build = subprocess.run(
        ["npm", "--prefix", str(CAMPAIGN), "run", "build"],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, f"the TypeScript build failed:\n{build.stdout}\n{build.stderr}"

    result = subprocess.run([node, str(BRIDGE), "6"], capture_output=True, text=True)
    assert result.returncode == 0, f"masking failed:\n{result.stderr}"
    return json.loads(result.stdout)


@pytest.fixture(scope="module")
def masked(masked_json, tmp_path_factory):
    path = tmp_path_factory.mktemp("fog") / "fog.json"
    path.write_text(json.dumps(masked_json))
    return json_export.load(path)


def test_a_masked_world_loads(masked):
    """The whole point. If `from_dict` refuses it, a commander has no map."""
    assert isinstance(masked, WorldState)
    assert masked.hexes


def test_it_is_the_current_schema(masked_json):
    assert masked_json["version"] == SCHEMA_VERSION


def test_no_hex_is_missing(masked, masked_json):
    """Blanked, not omitted — the reason the masking works the way it does.

    Omitting unseen hexes would fail `from_dict` outright, and even if it did not, every
    renderer takes min/max over the hexes present to size its canvas. Two commanders'
    maps would then come out different sizes and could not be laid over one another.
    """
    assert len(masked.hexes) == masked.width * masked.height


def test_the_bounds_match_the_unmasked_world(masked):
    """What the canvas would be sized from."""
    qs = [c[0] for c in masked.hexes]
    rs = [c[1] for c in masked.hexes]
    assert min(qs) == 0 and max(qs) == masked.width - 1
    assert min(rs) == 0 and max(rs) == masked.height - 1


def test_unseen_ground_is_tagged_and_blank(masked):
    fogged = [h for h in masked.hexes.values() if FOG_TAG in h.tags]
    assert fogged, "the fixture should leave most of the map unseen"

    for hx in fogged:
        assert hx.elevation == 0.0
        assert hx.biome is None
        assert hx.land_cover is None
        assert hx.settlement is None
        assert not hx.road_connections


def test_seen_ground_survives_intact(masked):
    known = [h for h in masked.hexes.values() if FOG_TAG not in h.tags]
    assert known, "the fixture should leave some of the map seen"
    # Real terrain, not defaults: at least one known hex has a biome and a land cover.
    assert any(h.biome is not None for h in known)
    assert any(h.land_cover is not None for h in known)


def test_memory_reads_differently_from_observation(masked):
    """`remembered` is what lets a map distinguish what is known from what is watched."""
    remembered = [h for h in masked.hexes.values() if REMEMBERED_TAG in h.tags]
    assert remembered
    for hx in remembered:
        assert FOG_TAG not in hx.tags


def test_nothing_stands_on_ground_the_faction_has_not_seen(masked):
    """The leak this guards is subtle: a road running on into the dark draws the enemy's
    lines of communication onto a map they were never entitled to."""
    known = {c for c, h in masked.hexes.items() if FOG_TAG not in h.tags}

    for s in masked.settlements:
        assert s.coord in known, f"settlement {s.name} on unseen ground"

    for a, b in masked.road_edges:
        assert a in known and b in known, f"road edge {a}->{b} touches unseen ground"

    for a, b in masked.sea_edges:
        assert a in known and b in known

    for f in masked.ferries:
        assert f.a in known and f.b in known

    for hx in masked.hexes.values():
        for c in hx.road_connections:
            assert c in known, f"{hx.coord} has a road running to unseen {c}"

    for river in masked.rivers:
        for c in river.hexes:
            assert c in known, f"river hex {c} is on unseen ground"


def test_each_river_is_still_a_connected_polyline(masked):
    """Runs are split rather than filtered, or a renderer draws a straight line between
    two banks either side of unseen country."""
    from worldgen.core.hex_grid import distance

    for river in masked.rivers:
        assert len(river.hexes) >= 2
        for a, b in zip(river.hexes, river.hexes[1:], strict=False):
            assert distance(a, b) == 1, f"river jumps from {a} to {b}"


def test_the_file_says_what_the_fog_tag_means(masked_json):
    """A consumer that ignores tags reads flat land where there is sea, so the document
    states the contract rather than relying on the reader knowing it."""
    fog = masked_json["metadata"]["fog"]
    assert fog["faction"] == "red"
    assert fog["fog_tag"] == FOG_TAG
    assert fog["remembered_tag"] == REMEMBERED_TAG
    assert fog["known"] < fog["total"]
    assert "do not read them as terrain" in fog["note"]


def test_the_generating_config_still_travels(masked_json):
    """The river rules need the thresholds the world was generated with."""
    config = masked_json["metadata"]["config"]
    assert config["navigable_min_discharge"] > 0


def test_a_masked_world_renders(masked, tmp_path):
    """The end of the chain: a referee can look at a commander's map."""
    for attribute in ("biome", "terrain_class", "elevation", "settlements", "roads"):
        out = tmp_path / f"{attribute}.svg"
        debug_viewer.render(masked, attribute, str(out))
        assert out.is_file()
        assert out.read_text().startswith("<?xml") or out.read_text().lstrip().startswith("<svg")


def test_it_round_trips_through_json_again(masked, tmp_path):
    """A commander's map is a world like any other, including being savable."""
    path = tmp_path / "again.json"
    json_export.save(masked, path)
    again = json_export.load(path)
    assert len(again.hexes) == len(masked.hexes)
    assert {c for c, h in again.hexes.items() if FOG_TAG in h.tags} == {
        c for c, h in masked.hexes.items() if FOG_TAG in h.tags
    }
