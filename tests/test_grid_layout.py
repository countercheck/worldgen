"""The offset grid layout: a rectangular map with ragged north and south edges.

Both layouts key hexes by axial coordinates and differ only in *which* hexes exist, so
these tests check the two things that follow from that — the shape the grid draws as,
and that everything downstream of the shape (adjacency, drainage, the pipeline) is
indifferent to which one is in use.
"""

import math

import pytest

from tests.worlds import build_pipeline, build_world
from worldgen.core.config import WorldConfig
from worldgen.core.hex import TerrainClass
from worldgen.core.hex_grid import (
    axial_to_offset,
    axial_to_pixel,
    distance,
    neighbors,
    offset_to_axial,
)
from worldgen.core.world_state import WorldState

_SIZE = 1.0
_ROW_SPACING = math.sqrt(3) * _SIZE
_COL_SPACING = 1.5 * _SIZE


# --- the coordinate transform ------------------------------------------------


def test_offset_and_axial_round_trip():
    for col in range(-4, 12):
        for row in range(-4, 12):
            assert axial_to_offset(offset_to_axial(col, row)) == (col, row)


def test_even_columns_sit_on_the_row_line_and_odd_ones_half_below():
    """Odd-q: the half-hex stagger is what makes the north and south edges ragged."""
    for col in range(6):
        _, y = axial_to_pixel(offset_to_axial(col, 3), _SIZE)
        expected = _ROW_SPACING * (3 + (0.5 if col % 2 else 0.0))
        assert y == pytest.approx(expected)


def test_column_and_row_neighbours_are_hex_neighbours():
    """Offset column/row is a relabelling, not a different grid: the six hexes around
    (col, row) are still the six axial neighbours."""
    coord = offset_to_axial(5, 5)
    assert len(set(neighbors(coord))) == 6
    for nbr in neighbors(coord):
        col, row = axial_to_offset(nbr)
        assert abs(col - 5) <= 1 and abs(row - 5) <= 1
        assert distance(coord, nbr) == 1


# --- the grid a world is built from ------------------------------------------


def test_offset_world_has_one_hex_per_column_and_row():
    ws = WorldState.empty(1, 9, 7, layout="offset")
    assert len(ws.hexes) == 9 * 7
    assert {ws.grid_index(c) for c in ws.hexes} == {
        (col, row) for col in range(9) for row in range(7)
    }


def test_coord_at_and_grid_index_invert_each_other():
    for layout in ("axial", "offset"):
        ws = WorldState.empty(1, 6, 5, layout=layout)
        for coord in ws.hexes:
            assert ws.coord_at(*ws.grid_index(coord)) == coord


def test_offset_grid_draws_a_rectangle_with_ragged_north_and_south():
    ws = WorldState.empty(1, 16, 12, layout="offset")
    by_col: dict[int, list[float]] = {}
    for coord in ws.hexes:
        col, _ = ws.grid_index(coord)
        by_col.setdefault(col, []).append(axial_to_pixel(coord, _SIZE)[1])

    # Every column spans the same height and starts at one of two staggered offsets:
    # straight east and west edges, ragged north and south ones.  Rounded, because the
    # stagger is a sum of irrationals and columns reach it by different routes.
    spans = {(round(min(ys), 9), round(max(ys), 9)) for ys in by_col.values()}
    assert len(spans) == 2
    (lo_a, hi_a), (lo_b, hi_b) = sorted(spans)
    assert hi_a - lo_a == pytest.approx(hi_b - lo_b)
    assert lo_b - lo_a == pytest.approx(_ROW_SPACING / 2)


def test_axial_grid_still_draws_a_parallelogram():
    """The default is unchanged: the shear puts each column half a row below the last,
    so the north edge runs away diagonally rather than staggering by a half hex."""
    ws = WorldState.empty(1, 16, 12, layout="axial")
    tops = {}
    for coord in ws.hexes:
        col, row = ws.grid_index(coord)
        if row == 0:
            tops[col] = axial_to_pixel(coord, _SIZE)[1]
    assert tops[1] - tops[0] == pytest.approx(_ROW_SPACING / 2)
    assert tops[8] - tops[0] == pytest.approx(4 * _ROW_SPACING)


def test_offset_map_is_square_when_height_is_scaled_by_the_row_spacing():
    """A square map needs height ~= width * 1.5/sqrt(3); the docs quote 0.87."""
    width = 128
    height = round(width * _COL_SPACING / _ROW_SPACING)
    ws = WorldState.empty(1, width, height, layout="offset")
    pixels = [axial_to_pixel(c, _SIZE) for c in ws.hexes]
    span_x = max(p[0] for p in pixels) - min(p[0] for p in pixels)
    span_y = max(p[1] for p in pixels) - min(p[1] for p in pixels)
    assert span_y / span_x == pytest.approx(1.0, abs=0.02)


def test_on_border_is_the_outermost_ring_in_both_layouts():
    for layout in ("axial", "offset"):
        ws = WorldState.empty(1, 8, 6, layout=layout)
        border = {c for c in ws.hexes if ws.on_border(c)}
        interior = set(ws.hexes) - border
        # An interior hex has all six neighbours on the map; a border hex does not.
        assert interior
        assert all(all(n in ws.hexes for n in neighbors(c)) for c in interior)
        assert all(any(n not in ws.hexes for n in neighbors(c)) for c in border)


# --- config and serialisation ------------------------------------------------


def test_config_rejects_an_unknown_layout():
    with pytest.raises(ValueError, match="grid_layout"):
        WorldConfig(grid_layout="hexagonal")


def test_layout_round_trips_through_the_schema():
    ws = WorldState.empty(3, 5, 4, layout="offset")
    restored = WorldState.from_dict(ws.to_dict())
    assert restored.layout == "offset"
    assert set(restored.hexes) == set(ws.hexes)


def test_a_file_without_a_layout_loads_as_axial():
    """Worlds saved before offset grids existed predate the field."""
    data = WorldState.empty(3, 5, 4).to_dict()
    del data["layout"]
    assert WorldState.from_dict(data).layout == "axial"


def test_a_file_with_an_unknown_layout_is_rejected():
    data = WorldState.empty(3, 5, 4).to_dict()
    data["layout"] = "hexagonal"
    with pytest.raises(ValueError, match="layout"):
        WorldState.from_dict(data)


# --- the full pipeline on an offset grid -------------------------------------


@pytest.fixture(scope="module")
def offset_world():
    return build_world(
        seed=42,
        width=56,
        height=48,
        grid_layout="offset",
        target_city_count=3,
        target_town_count=8,
        road_travellers_city=60,
        road_travellers_town=10,
        road_travellers_village=2,
    )


def test_pipeline_fills_every_column_and_row(offset_world):
    assert offset_world.layout == "offset"
    assert len(offset_world.hexes) == 56 * 48


def test_offset_world_has_sea_land_rivers_and_settlements(offset_world):
    assert offset_world.all_ocean()
    assert offset_world.all_land()
    assert offset_world.rivers
    assert offset_world.settlements
    assert offset_world.roads


def test_offset_rivers_run_downhill_to_water_or_the_border(offset_world):
    water = {
        c
        for c, h in offset_world.hexes.items()
        if h.terrain_class in (TerrainClass.OCEAN, TerrainClass.LAKE)
    }
    river_hexes = {c for r in offset_world.rivers for c in r.hexes}
    for river in offset_world.rivers:
        for a, b in zip(river.hexes, river.hexes[1:], strict=False):
            assert b in neighbors(a), f"river path breaks at {a} -> {b}"
        mouth = river.hexes[-1]
        reaches = (
            mouth in water
            or any(n in water for n in neighbors(mouth))
            or offset_world.on_border(mouth)
            or any(n in river_hexes and n not in river.hexes for n in neighbors(mouth))
        )
        assert reaches, f"river mouth {mouth} drains nowhere"


def test_offset_roads_are_connected_paths(offset_world):
    for road in offset_world.roads:
        assert len(road.path) >= 2
        for a, b in zip(road.path, road.path[1:], strict=False):
            assert b in neighbors(a), f"road path breaks at {a} -> {b}"


def test_offset_worlds_are_reproducible_from_the_seed():
    def build():
        return build_pipeline(
            seed=7,
            width=24,
            height=21,
            grid_layout="offset",
            erosion_iterations=200,
            until="HydrologyStage",
        ).run()

    a, b = build(), build()
    assert {c: h.elevation for c, h in a.hexes.items()} == {
        c: h.elevation for c, h in b.hexes.items()
    }
    assert [r.hexes for r in a.rivers] == [r.hexes for r in b.rivers]
