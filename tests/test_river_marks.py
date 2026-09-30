"""Where a river rises, where it ends, and where it runs white: tagged by the stages, marked
on every map."""

import re
from pathlib import Path

import pytest

from tests.worlds import build_world
from worldgen.core.hex import TerrainClass
from worldgen.export import legend
from worldgen.export.svg_export import _river_mark
from worldgen.render import glyphs

_WATER = (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER)
_TS = Path(__file__).resolve().parent.parent / "campaign" / "client" / "src" / "map" / "glyphs.ts"


@pytest.fixture(scope="module")
def world():
    return build_world(seed=42, width=64, height=64, until="CataractStage")


def _land(ws, path):
    return [c for c in path if c in ws.hexes and ws.hexes[c].terrain_class not in _WATER]


def test_every_source_is_the_first_hex_of_a_drawn_river(world):
    firsts = {r.hexes[0] for r in world.rivers if len(_land(world, r.hexes)) >= 2}
    sources = {c for c, h in world.hexes.items() if "river_source" in h.tags}
    assert sources
    assert sources <= firsts
    for c in sources:
        assert "river_source_offmap" not in world.hexes[c].tags


def test_a_river_reaching_the_sea_or_a_lake_is_marked_where_it_ends(world):
    for river in world.rivers:
        land = _land(world, river.hexes)
        last = river.hexes[-1]
        if len(land) >= 2 and world.hexes[last].terrain_class in _WATER:
            assert "river_end" in world.hexes[land[-1]].tags


def test_a_tributary_has_no_end_of_its_own(world):
    """Its last hex is its trunk's, and the trunk flows on from there."""
    flows_on = {c for r in world.rivers for c in r.hexes[:-1]}
    joined = [
        r.hexes[-1]
        for r in world.rivers
        if world.hexes[r.hexes[-1]].terrain_class not in _WATER and r.hexes[-1] in flows_on
    ]
    assert joined
    assert not any("river_end" in world.hexes[c].tags for c in joined)


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
    """A river ends at a lake as at the sea; what leaves the lake is a river of its own."""
    for seed in (42, 7, 3):
        ws = build_world(seed=seed, width=64, height=64, until="HydrologyStage")
        for river in ws.rivers:
            inner = [c for c in river.hexes[1:-1] if ws.hexes[c].terrain_class in _WATER]
            assert not inner, f"seed {seed}: a river crosses {len(inner)} water hexes"


def test_split_at_water_ends_at_the_shore_and_starts_again_beyond():
    from worldgen.core.world_state import River
    from worldgen.stages.hydrology import _split_at_water

    lake = {(2, 0), (3, 0), (4, 0)}
    path = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0)]
    pieces = _split_at_water([River(hexes=path, flow_volume=1.0)], lake, {}, 1.0)
    assert [p.hexes for p in pieces] == [[(0, 0), (1, 0), (2, 0)], [(4, 0), (5, 0), (6, 0)]]
