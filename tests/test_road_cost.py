"""Unit tests for the road cost model in worldgen.stages.road_cost.

These tests construct tiny synthetic hex grids and exercise the cost helpers
directly, without running the full pipeline. They verify both the arithmetic
of individual cost components and the A* behaviour they produce.
"""

import pytest

from tests.worlds import lay_river
from worldgen.core.config import WorldConfig
from worldgen.core.hex import Hex, TerrainClass
from worldgen.core.hex_grid import astar, side_hexes
from worldgen.core.world_state import RoadTier, WorldState, road_edge_key
from worldgen.stages.road_cost import (
    fill_tier_gaps,
    make_road_edge_cost,
    river_crossing_edge_cost,
    river_crossings,
    road_edge_cost,
    slope_edge_cost,
    tag_river_crossings,
    terrain_base_cost,
    tier_near,
    water_edge_cost,
)


def _flat(coord):
    return Hex(coord=coord, elevation=0.5, terrain_class=TerrainClass.LAND)


def _ocean(coord):
    return Hex(coord=coord, elevation=0.0, terrain_class=TerrainClass.OPEN_WATER)


def _lake(coord):
    return Hex(coord=coord, elevation=0.0, terrain_class=TerrainClass.INLAND_WATER)


def _between(a, b, flow):
    """A crossings map with one river running between hexes *a* and *b*."""
    return {frozenset((a, b)): flow}


# ---------- terrain_base_cost ----------------------------------------------


def test_terrain_base_cost_water_is_finite():
    cfg = WorldConfig()
    assert terrain_base_cost(_ocean((0, 0)), cfg) == cfg.road_water_cost
    assert terrain_base_cost(_lake((0, 0)), cfg) == cfg.road_water_cost
    assert cfg.road_water_cost > 0
    assert cfg.road_water_cost < cfg.road_flat_cost  # water is cheaper per-hex than land


def test_terrain_base_cost_is_flat_across_land():
    """Steepness is charged per edge, by grade, and must not be charged again per hex.

    The node cost used to add a hill and a mountain rate on top of `slope_edge_cost`,
    which prices the elevation a road actually climbs. That was the same terrain paid for
    twice — and paid off a threshold, so a level floodplain under a bluff was billed as
    mountain while a long gentle haul up to the same height was billed as flat.
    """
    cfg = WorldConfig()
    steep = Hex(coord=(0, 0), elevation=900.0, slope=200.0)
    gentle = Hex(coord=(1, 0), elevation=900.0, slope=1.0)
    assert terrain_base_cost(steep, cfg) == cfg.road_flat_cost
    assert terrain_base_cost(gentle, cfg) == cfg.road_flat_cost


def test_grade_is_what_makes_a_climb_expensive():
    # The term that does carry steepness: a road up a wall costs more than one along a
    # shelf, and it reads the elevations rather than a label either hex carries.
    cfg = WorldConfig()
    low = Hex(coord=(0, 0), elevation=100.0)
    high = Hex(coord=(1, 0), elevation=900.0)
    level = Hex(coord=(1, 0), elevation=110.0)
    assert slope_edge_cost(low, high, cfg) > slope_edge_cost(low, level, cfg)


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


def test_river_crossing_zero_where_no_river_runs_between():
    cfg = WorldConfig()
    assert river_crossing_edge_cost(_flat((0, 0)), _flat((1, 0)), cfg) == 0.0
    crossings = _between((0, 0), (1, 0), 0.5)
    # A river elsewhere, even beside one of the hexes, is not crossed by this step.
    assert river_crossing_edge_cost(_flat((0, 0)), _flat((0, 1)), cfg, crossings) == 0.0


def test_river_crossing_scales_monotonically_with_flow():
    cfg = WorldConfig()
    a, b = _flat((0, 0)), _flat((1, 0))
    small = river_crossing_edge_cost(a, b, cfg, _between((0, 0), (1, 0), 0.1))
    big = river_crossing_edge_cost(a, b, cfg, _between((0, 0), (1, 0), 1.0))
    assert big - small == pytest.approx(0.9 * cfg.road_river_crossing_flow)


def test_river_crossing_is_charged_once_whichever_way():
    cfg = WorldConfig()
    crossings = _between((0, 0), (1, 0), 0.7)
    there = river_crossing_edge_cost(_flat((0, 0)), _flat((1, 0)), cfg, crossings)
    back = river_crossing_edge_cost(_flat((1, 0)), _flat((0, 0)), cfg, crossings)
    assert there == back
    assert there == pytest.approx(cfg.road_river_crossing_base + 0.7 * cfg.road_river_crossing_flow)


def test_river_crossings_reads_the_world_river_sides():
    ws = WorldState.empty(seed=1, width=4, height=4)
    river = lay_river(ws, [(1, 1), (2, 1), (2, 2)], flow_volume=0.4)
    crossings = river_crossings(ws.river_sides)
    assert len(crossings) == len(river.sides())
    assert all(v == pytest.approx(0.4) for v in crossings.values())


# ---------- road_edge_cost (composition) -----------------------------------


def test_road_edge_cost_symmetric():
    cfg = WorldConfig()
    a, b = _flat((0, 0)), _flat((1, 0))
    b.elevation = 30.0
    crossings = _between((0, 0), (1, 0), 0.6)
    assert road_edge_cost(a, b, cfg, crossings=crossings) == road_edge_cost(
        b, a, cfg, crossings=crossings
    )


def test_road_edge_cost_zero_for_identical_flat_hexes():
    cfg = WorldConfig()
    a = _flat((0, 0))
    b = _flat((1, 0))
    assert road_edge_cost(a, b, cfg) == 0.0


def test_road_edge_cost_adds_the_crossing_to_the_climb():
    cfg = WorldConfig()
    a, b = _flat((0, 0)), _flat((1, 0))
    b.elevation = 30.0
    crossings = _between((0, 0), (1, 0), 0.5)
    climb = road_edge_cost(a, b, cfg)
    assert road_edge_cost(a, b, cfg, crossings=crossings) == pytest.approx(
        climb + cfg.road_river_crossing_base + 0.5 * cfg.road_river_crossing_flow
    )


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
    has_water = any(hexes[c].terrain_class == TerrainClass.OPEN_WATER for c in path)
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
    has_water = any(hexes[c].terrain_class == TerrainClass.OPEN_WATER for c in path)
    assert not has_water, f"Short land detour should beat a 1-hex water hop, got {path}"


def _river_between_rows(width, flow_of):
    """A river along every side between rows r=1 and r=2, its flow set by column."""
    crossings = {}
    for q in range(width):
        for below in ((q, 2), (q - 1, 2)):
            if 0 <= below[0] < width:
                crossings[frozenset(((q, 1), below))] = flow_of(q)
    return crossings


def _crossed(path, crossings):
    return [
        crossings[frozenset(e)]
        for e in zip(path, path[1:], strict=False)
        if frozenset(e) in crossings
    ]


def test_astar_prefers_low_flow_river_for_crossing():
    """A river runs the width of the grid between rows 1 and 2: a high-flow trunk on the
    left (q < 3) and a low-flow stream on the right.  A path from (0, 0) to (0, 4) must
    cross it somewhere, and should detour right to the cheap stream crossing."""
    cfg = WorldConfig()
    hexes = _build_grid(7, 5, lambda q, r: _flat((q, r)))
    crossings = _river_between_rows(7, lambda q: 1.0 if q < 3 else 0.1)
    path = astar(
        hexes,
        (0, 0),
        (0, 4),
        lambda hx: terrain_base_cost(hx, cfg),
        make_road_edge_cost(cfg, crossings),
    )
    assert path is not None
    assert _crossed(path, crossings) == [pytest.approx(0.1)]


def test_a_road_beside_a_river_pays_nothing_for_it():
    """A river along a hexside has no channel to run down: a road on the bank is on land."""
    cfg = WorldConfig()
    hexes = _build_grid(7, 5, lambda q, r: _flat((q, r)))
    crossings = _river_between_rows(7, lambda q: 1.0)
    path = astar(
        hexes,
        (0, 1),
        (6, 1),
        lambda hx: terrain_base_cost(hx, cfg),
        make_road_edge_cost(cfg, crossings),
    )
    assert all(c[1] == 1 for c in path), f"the road left the bank: {path}"
    assert not _crossed(path, crossings)


def test_tag_river_crossings_tags_the_side_by_tier():
    ws = WorldState.empty(seed=1, width=6, height=6)
    river = lay_river(ws, [(1, 2), (2, 2), (3, 2), (4, 2)])
    sides = river.sides()
    primary, track = (road_edge_key(*side_hexes(s)) for s in sides[:2])
    tag_river_crossings({primary: RoadTier.PRIMARY, track: RoadTier.TRACK}, ws)
    assert "bridge" in ws.river_sides[sides[0]].tags
    assert "ford" in ws.river_sides[sides[1]].tags
    assert not ws.river_sides[sides[2]].tags


def test_tag_river_crossings_never_demotes_a_bridge():
    ws = WorldState.empty(seed=1, width=6, height=6)
    river = lay_river(ws, [(1, 2), (2, 2), (3, 2)])
    side = river.sides()[0]
    ws.river_sides[side].tags.add("bridge")
    tag_river_crossings({road_edge_key(*side_hexes(side)): RoadTier.TRACK}, ws)
    assert ws.river_sides[side].tags == {"bridge"}


# -- tier gaps ---------------------------------------------------------------

P, S, T = RoadTier.PRIMARY, RoadTier.SECONDARY, RoadTier.TRACK


def _line(tiers):
    """A straight road along r = 0 whose i-th edge has tiers[i]."""
    return {road_edge_key((i, 0), (i + 1, 0)): t for i, t in enumerate(tiers)}


def test_fill_tier_gaps_promotes_a_dip_between_two_primary_ends():
    edges = _line([P, P, S, T, P, P])
    assert fill_tier_gaps(edges, 6) == 2
    assert set(edges.values()) == {P}


def test_fill_tier_gaps_leaves_a_dip_longer_than_the_limit():
    edges = _line([P, S, S, S, P])
    assert fill_tier_gaps(edges, 2) == 0
    assert list(edges.values()) == [P, S, S, S, P]


def test_fill_tier_gaps_zero_is_off():
    edges = _line([P, S, P])
    assert fill_tier_gaps(edges, 0) == 0
    assert edges[road_edge_key((1, 0), (2, 0))] is S


def test_fill_tier_gaps_fills_secondary_dips_too():
    edges = _line([S, T, S])
    fill_tier_gaps(edges, 6)
    assert set(edges.values()) == {S}


def test_fill_tier_gaps_keeps_a_connector_between_two_trunks():
    """A secondary leaving the middle of one primary for the middle of another is a
    junction road, not a break in either trunk."""
    edges = _line([P, P])  # (0,0)-(2,0), passing through (1,0)
    edges.update({road_edge_key((0, 3), (1, 3)): P, road_edge_key((1, 3), (2, 3)): P})
    edges.update({road_edge_key((1, 0), (1, 1)): S, road_edge_key((1, 1), (1, 2)): S})
    edges[road_edge_key((1, 2), (1, 3))] = S
    assert fill_tier_gaps(edges, 6) == 0


def test_fill_tier_gaps_ignores_a_lane_back_onto_the_same_road():
    """Two ends of one primary network are already joined; a lane between them is a
    shortcut the traffic declined."""
    edges = _line([P, P, P, P])
    edges.update({road_edge_key((0, 0), (0, 1)): T, road_edge_key((4, 0), (4, 1)): T})
    for q in range(4):
        edges[road_edge_key((q, 1), (q + 1, 1))] = T
    assert fill_tier_gaps(edges, 6) == 0


def test_fill_tier_gaps_closes_a_trunk_stopping_short_of_another():
    """One primary ends a hex from the middle of a second: the T is closed."""
    edges = _line([P, P])  # (0,0)-(2,0), passing through (1,0)
    edges.update({road_edge_key((1, 3), (1, 4)): P, road_edge_key((1, 4), (1, 5)): P})
    edges.update({road_edge_key((1, 0), (1, 1)): S, road_edge_key((1, 1), (1, 2)): S})
    edges[road_edge_key((1, 2), (1, 3))] = S
    assert fill_tier_gaps(edges, 6) == 3


def test_tier_near_reads_the_best_road_within_reach():
    edges = _line([T, T, P, S])
    assert tier_near(edges, (0, 0), 1) is T
    assert tier_near(edges, (0, 0), 3) is P
    assert tier_near(edges, (4, 0), 1) is S
    assert tier_near(edges, (9, 9), 5) is T
