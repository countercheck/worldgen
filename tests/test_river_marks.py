"""Where a river rises, where it ends, and where it runs white: tagged by the stages, marked
on every map."""

import re
from pathlib import Path

import pytest

from tests.worlds import build_world
from worldgen.core.hex import TerrainClass
from worldgen.core.hex_grid import corner_hexes, side_hexes
from worldgen.export import legend
from worldgen.export.svg_export import _river_mark
from worldgen.render import glyphs

_WATER = (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER)
_TS = Path(__file__).resolve().parent.parent / "campaign" / "client" / "src" / "map" / "glyphs.ts"


@pytest.fixture(scope="module")
def world():
    return build_world(seed=42, width=64, height=64, until="CataractStage")


def _tagged(ws, tag):
    return {c for c, tags in ws.river_corners.items() if tag in tags}


def _at_water(ws, corner):
    return any(h in ws.hexes and ws.hexes[h].terrain_class in _WATER for h in corner_hexes(corner))


def test_every_source_is_the_first_corner_of_a_drawn_river(world):
    firsts = {r.corners[0] for r in world.rivers}
    sources = _tagged(world, "river_source")
    assert sources
    assert sources <= firsts
    assert not sources & _tagged(world, "river_source_offmap")


def test_a_river_reaching_the_sea_or_a_lake_is_marked_where_it_ends(world):
    ends = _tagged(world, "river_end")
    for river in world.rivers:
        if _at_water(world, river.corners[-1]):
            assert river.corners[-1] in ends


def test_a_tributary_has_no_end_of_its_own(world):
    """Its last corner is its trunk's, and the trunk flows on from there."""
    flows_on = {c for r in world.rivers for c in r.corners[:-1]}
    joined = [
        r.corners[-1]
        for r in world.rivers
        if r.corners[-1] in flows_on and not _at_water(world, r.corners[-1])
    ]
    assert joined
    assert not set(joined) & _tagged(world, "river_end")


def test_rapids_are_off_a_cataract_and_obey_their_settings():
    ws = build_world(seed=42, width=64, height=64, until="CataractStage", rapids_min_drop_m=1.0)
    rapids = [h for h in ws.hexes.values() if "rapids" in h.tags]
    assert rapids
    assert not any("cataract" in h.tags for h in rapids)
    assert all(h.catchment_km2 >= 200.0 for h in rapids)
    off = build_world(seed=42, width=64, height=64, until="CataractStage", rapids_min_drop_m=0.0)
    assert not any("rapids" in h.tags for h in off.hexes.values())


def test_the_legend_and_the_svg_draw_the_marks(world):
    from worldgen.core.hex_grid import axial_to_pixel

    marks = legend.river_marks(world, axial_to_pixel, 12.0)
    assert {k for _, k, _ in marks} >= {"source", "end"}
    labels = {r.label for r in legend.rows(world, "terrain", {"rivers"})}
    assert {"River source", "River end"} <= labels
    for kind in ("source", "end", "rapids"):
        assert _river_mark(kind, 10, 10, 30.0).startswith(("<g>", "<polygon"))


def test_an_end_arrow_points_downstream():
    east = _river_mark("end", 0, 0, 0.0)
    xs = [float(x) for x in re.findall(r"(-?\d+\.\d+),", east)]
    assert max(xs) > abs(min(xs)), "the arrow's tip is not on the downstream side"


def _ts_numbers(name: str) -> list[float]:
    block = re.search(rf"export const {name}[^=]*= (.*?);\n", _TS.read_text(), re.S)
    assert block, f"{name} is missing from {_TS.name}"
    return [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", block.group(1))]


def test_the_campaign_client_draws_the_same_marks():
    assert _ts_numbers("RIVER_SOURCE_RING") == [glyphs.RIVER_SOURCE_RING]
    assert _ts_numbers("RIVER_SOURCE_DOT") == [glyphs.RIVER_SOURCE_DOT]
    assert _ts_numbers("RIVER_END_ARROW") == [v for p in glyphs.RIVER_END_ARROW for v in p]
    bars = [v for bar in glyphs.RAPIDS_BARS for p in bar for v in p]
    assert _ts_numbers("RAPIDS_BARS") == bars
    text = _TS.read_text()
    assert f"RAPIDS_INK = '{glyphs.RAPIDS_INK}'" in text
    assert f"RAPIDS_CASING = '{glyphs.RAPIDS_CASING}'" in text


def test_no_river_is_drawn_across_water():
    """A river ends at a lake as at the sea; what leaves the lake is a river of its own.

    On hexsides that is a rule about sides: every one a river runs along has land on both
    hands, so no course crosses water or runs along a shore.
    """
    for seed in (42, 7, 3):
        ws = build_world(seed=seed, width=64, height=64, until="HydrologyStage")
        for river in ws.rivers:
            for side in river.sides():
                wet = [h for h in side_hexes(side) if ws.hexes[h].terrain_class in _WATER]
                assert not wet, f"seed {seed}: a river runs along water at {side}"
