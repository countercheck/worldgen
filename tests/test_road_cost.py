"""Unit tests for the road cost model in worldgen.stages.road_cost.

These tests construct tiny synthetic hex grids and exercise the cost helpers
directly, without running the full pipeline. They verify both the arithmetic
of individual cost components and the A* behaviour they produce.
"""

import pytest

from worldgen.core.config import WorldConfig
from worldgen.core.hex import Hex, TerrainClass
from worldgen.core.hex_grid import astar
from worldgen.stages.road_cost import (
    river_crossing_edge_cost,
    river_discount,
    road_edge_cost,
    terrain_base_cost,
    water_edge_cost,
)


def _flat(coord):
    return Hex(coord=coord, elevation=0.5, terrain_class=TerrainClass.FLAT)


def _ocean(coord):
    return Hex(coord=coord, elevation=0.0, terrain_class=TerrainClass.OCEAN)


def _lake(coord):
    return Hex(coord=coord, elevation=0.0, terrain_class=TerrainClass.LAKE)


def _mountain(coord):
    return Hex(coord=coord, elevation=0.9, terrain_class=TerrainClass.MOUNTAIN)


def _hill(coord):
    return Hex(coord=coord, elevation=0.7, terrain_class=TerrainClass.HILL)


def _river_flat(coord, flow=1.0):
    h = _flat(coord)
    h.river_flow = flow
    return h


# ---------- terrain_base_cost ----------------------------------------------


def test_terrain_base_cost_water_is_finite():
    cfg = WorldConfig()
    assert terrain_base_cost(_ocean((0, 0)), cfg) == cfg.road_water_cost
    assert terrain_base_cost(_lake((0, 0)), cfg) == cfg.road_water_cost
    assert cfg.road_water_cost > 0
    assert cfg.road_water_cost < cfg.road_flat_cost  # water is cheaper per-hex than land


def test_terrain_base_cost_land_classes():
    cfg = WorldConfig()
    assert terrain_base_cost(_flat((0, 0)), cfg) == cfg.road_flat_cost
    assert terrain_base_cost(_hill((0, 0)), cfg) == cfg.road_hill_cost
    assert terrain_base_cost(_mountain((0, 0)), cfg) == cfg.road_mountain_cost


# ---------- river_discount -------------------------------------------------


def test_river_discount_zero_when_not_river():
    cfg = WorldConfig()
    assert river_discount(_flat((0, 0)), cfg) == 0.0


def test_river_discount_scales_with_flow():
    cfg = WorldConfig()
    small = river_discount(_river_flat((0, 0), flow=0.3), cfg)
    big = river_discount(_river_flat((1, 0), flow=1.0), cfg)
    assert big > small


def test_river_discount_min_flow_floor():
    cfg = WorldConfig(road_river_discount_min_flow=0.4)
    # Tiny flow but discount should use the min_flow floor (0.4)
    d = river_discount(_river_flat((0, 0), flow=0.05), cfg)
    assert d == pytest.approx(cfg.road_river_discount * 0.4)


# ---------- water_edge_cost ------------------------------------------------


def test_water_edge_cost_zero_when_same_class():
    cfg = WorldConfig()
    assert water_edge_cost(_flat((0, 0)), _flat((1, 0)), cfg) == 0.0
    assert water_edge_cost(_ocean((0, 0)), _ocean((1, 0)), cfg) == 0.0


def test_water_edge_cost_embark_on_land_to_water():
    cfg = WorldConfig()
    cost = water_edge_cost(_flat((0, 0)), _ocean((1, 0)), cfg)
    assert cost == cfg.road_embark_cost


def test_water_edge_cost_disembark_on_water_to_land():
    cfg = WorldConfig()
    cost = water_edge_cost(_ocean((0, 0)), _flat((1, 0)), cfg)
    assert cost == cfg.road_disembark_cost


def test_water_edge_cost_lake_treated_as_water():
    cfg = WorldConfig()
    assert water_edge_cost(_flat((0, 0)), _lake((1, 0)), cfg) == cfg.road_embark_cost
    assert water_edge_cost(_lake((0, 0)), _flat((1, 0)), cfg) == cfg.road_disembark_cost


# ---------- river_crossing_edge_cost ---------------------------------------


def test_river_crossing_zero_when_no_transition():
    cfg = WorldConfig()
    # Two land hexes, no rivers
    assert river_crossing_edge_cost(_flat((0, 0)), _flat((1, 0)), cfg) == 0.0
    # Two river hexes — travelling along, not across
    assert river_crossing_edge_cost(_river_flat((0, 0)), _river_flat((1, 0)), cfg) == 0.0


def test_river_crossing_scales_monotonically_with_flow():
    cfg = WorldConfig()
    small = river_crossing_edge_cost(_flat((0, 0)), _river_flat((1, 0), flow=0.1), cfg)
    big = river_crossing_edge_cost(_flat((0, 0)), _river_flat((1, 0), flow=1.0), cfg)
    assert big > small
    # base + 0.1 * flow_factor vs base + 1.0 * flow_factor
    assert big - small == pytest.approx(0.9 * cfg.road_river_crossing_flow)


def test_river_crossing_uses_max_of_two_flows():
    cfg = WorldConfig()
    # land → river: max is the river hex's flow
    a = river_crossing_edge_cost(_flat((0, 0)), _river_flat((1, 0), flow=0.7), cfg)
    # river → land: same edge, reversed; should be identical
    b = river_crossing_edge_cost(_river_flat((1, 0), flow=0.7), _flat((0, 0)), cfg)
    assert a == b
    assert a == pytest.approx(cfg.road_river_crossing_base + 0.7 * cfg.road_river_crossing_flow)


# ---------- road_edge_cost (composition) -----------------------------------


def test_road_edge_cost_symmetric():
    cfg = WorldConfig()
    a = _flat((0, 0))
    b = _river_flat((1, 0), flow=0.6)
    assert road_edge_cost(a, b, cfg) == road_edge_cost(b, a, cfg)


def test_road_edge_cost_zero_for_identical_flat_hexes():
    cfg = WorldConfig()
    a = _flat((0, 0))
    b = _flat((1, 0))
    assert road_edge_cost(a, b, cfg) == 0.0


def test_road_edge_cost_combines_water_and_river():
    """An edge that both crosses a shoreline AND a river edge accumulates both costs."""
    cfg = WorldConfig()
    # Match elevations to neutralise slope_edge_cost; isolate water + river contributions.
    flat = Hex(coord=(0, 0), elevation=0.5, terrain_class=TerrainClass.FLAT)
    river_at_shore = Hex(coord=(1, 0), elevation=0.5, terrain_class=TerrainClass.OCEAN)
    river_at_shore.river_flow = 0.5  # river mouth
    cost = road_edge_cost(flat, river_at_shore, cfg)
    expected = (
        cfg.road_embark_cost + cfg.road_river_crossing_base + 0.5 * cfg.road_river_crossing_flow
    )
    assert cost == pytest.approx(expected)


# ---------- A* integration on synthetic grids ------------------------------


def _build_grid(width, height, hex_factory):
    """Build a small rectangular hex grid with all coords and a custom factory."""
    return {(q, r): hex_factory(q, r) for q in range(width) for r in range(height)}


def test_astar_takes_water_shortcut_across_strait():
    """Two land masses separated by a 6-hex water strait. Going around takes 30+
    hexes of land detour; cutting through water costs ~16 (embark+disembark) + 6×0.05.
    The water route should win."""
    cfg = WorldConfig()

    def factory(q, r):
        # Strait is the band 6 <= q < 12, full height
        if 6 <= q < 12:
            return _ocean((q, r))
        return _flat((q, r))

    hexes = _build_grid(40, 3, factory)

    def node_cost(hx):
        return terrain_base_cost(hx, cfg)

    def edge_cost(a, b):
        return road_edge_cost(a, b, cfg)

    path = astar(hexes, (0, 1), (20, 1), node_cost, edge_cost)
    assert path is not None
    has_water = any(hexes[c].terrain_class == TerrainClass.OCEAN for c in path)
    assert has_water, "A* should cross the strait rather than take an impossible detour"


def test_astar_avoids_water_when_short_land_detour_available():
    """A 2-hex water hop is more expensive than a 4-hex land detour (embark+disembark = 16 ≫ 4×1)."""
    cfg = WorldConfig()

    # Land everywhere except a 1-hex pond at (2, 1)
    def factory(q, r):
        if (q, r) == (2, 1):
            return _ocean((q, r))
        return _flat((q, r))

    hexes = _build_grid(8, 3, factory)

    def node_cost(hx):
        return terrain_base_cost(hx, cfg)

    def edge_cost(a, b):
        return road_edge_cost(a, b, cfg)

    path = astar(hexes, (0, 1), (4, 1), node_cost, edge_cost)
    assert path is not None
    has_water = any(hexes[c].terrain_class == TerrainClass.OCEAN for c in path)
    assert not has_water, f"Short land detour should beat a 1-hex water hop, got {path}"


def test_astar_prefers_low_flow_river_for_crossing():
    """A single river barrier spans the full grid at row r=2, but the left half
    (q < 3) is a high-flow trunk and the right half (q >= 3) is a low-flow stream.
    A path from (0, 0) to (0, 4) must cross r=2 somewhere; A* should detour right
    to use the cheaper stream crossing rather than the direct but costly trunk crossing."""
    cfg = WorldConfig()

    def factory(q, r):
        if r == 2:
            flow = 1.0 if q < 3 else 0.1
            return _river_flat((q, r), flow=flow)
        return _flat((q, r))

    hexes = _build_grid(7, 5, factory)

    def node_cost(hx):
        return terrain_base_cost(hx, cfg)

    def edge_cost(a, b):
        return road_edge_cost(a, b, cfg)

    # Path from (0, 0) to (0, 4) must cross r=2; the crossing column is the choice.
    # Direct crossing at q=0 (trunk, flow=1.0): 2 × (4 + 12×1.0) = 32 in edge cost
    #   plus 4 nodes × 1.0 = 36 total.
    # Detour to q=3 (stream, flow=0.1): 2 × (4 + 12×0.1) = 10.4, plus 10 nodes = 20.4.
    path = astar(hexes, (0, 0), (0, 4), node_cost, edge_cost)
    assert path is not None

    # Find the column(s) where the path crosses the river row.
    crossing_cols = [c[0] for c in path if c[1] == 2]
    assert crossing_cols, "Path must cross river row r=2"
    assert all(q >= 3 for q in crossing_cols), (
        f"A* should detour to the low-flow stream half (q>=3), "
        f"but crossed at q={crossing_cols}"
    )


def test_astar_follows_river_when_along_path_available():
    """A river hex chain offers a discount for travel along it. A* between two
    points on the same river should choose the river path over a parallel land path."""
    cfg = WorldConfig()

    # River runs along r=1, full width
    def factory(q, r):
        if r == 1:
            return _river_flat((q, r), flow=0.8)
        return _flat((q, r))

    hexes = _build_grid(8, 3, factory)

    def node_cost(hx):
        return max(0.1, terrain_base_cost(hx, cfg) - river_discount(hx, cfg))

    def edge_cost(a, b):
        return road_edge_cost(a, b, cfg)

    # Source and destination are both on the river — A* should hug r=1
    path = astar(hexes, (0, 1), (7, 1), node_cost, edge_cost)
    assert path is not None
    # Every hex on the path should be on the river row
    on_river = sum(1 for c in path if c[1] == 1)
    assert on_river == len(path), (
        f"Expected all hexes on river row, got {on_river}/{len(path)}: {path}"
    )
