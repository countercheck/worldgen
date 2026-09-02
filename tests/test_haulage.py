"""The transport economics, on synthetic grids.

Everything here is a pure function over a hand-built map, so a failure names the rule that
broke rather than the seed that happened to expose it.
"""

import pytest

from worldgen.core.config import WorldConfig
from worldgen.core.hex import Hex, LandCover, TerrainClass
from worldgen.stages.haulage import (
    allocate_catchments,
    fishery_rim,
    gather,
    haulage_range,
    navigable,
    settleable,
    usable_fraction,
)


def _hex(coord, terrain=TerrainClass.FLAT, **kw):
    return Hex(coord=coord, terrain_class=terrain, **kw)


def _strip(length, terrain=TerrainClass.FLAT):
    """A one-row corridor, so travel cost equals distance on flat ground."""
    return {(q, 0): _hex((q, 0), terrain) for q in range(length)}


def _river(hx, flow):
    hx.river_flow = flow
    hx.tags.add("river")
    return hx


# --- usable_fraction ---------------------------------------------------------


def test_full_value_at_the_door():
    assert usable_fraction(0.0, 10.0) == 1.0


def test_exactly_zero_at_the_limit():
    """The substance of the model: the range is where the team has eaten the cargo.

    A soft decay leaving a trace of value at any distance would let a catchment reach
    everywhere, and the ranges would stop bounding anything.
    """
    assert usable_fraction(10.0, 10.0) == 0.0
    assert usable_fraction(10.001, 10.0) == 0.0
    assert usable_fraction(1e6, 10.0) == 0.0


def test_monotone_decreasing_between():
    values = [usable_fraction(c / 10, 10.0) for c in range(0, 101)]
    assert values == sorted(values, reverse=True)
    assert all(0.0 <= v <= 1.0 for v in values)


def test_halfway_is_half_value():
    assert usable_fraction(5.0, 10.0) == pytest.approx(0.5)


def test_zero_range_carries_nothing():
    assert usable_fraction(0.0, 0.0) == 0.0


# --- navigability and range --------------------------------------------------


def test_open_water_is_navigable():
    cfg = WorldConfig()
    for terrain in (TerrainClass.OCEAN, TerrainClass.LAKE):
        assert navigable(_hex((0, 0), terrain), cfg)


def test_a_big_river_floats_a_boat_and_a_headwater_does_not():
    cfg = WorldConfig()
    big = _river(_hex((0, 0)), cfg.navigable_river_flow + 0.1)
    trickle = _river(_hex((1, 0)), cfg.navigable_river_flow - 0.1)
    assert navigable(big, cfg)
    assert not navigable(trickle, cfg)


def test_flow_without_the_river_tag_is_not_a_channel():
    """`river_flow_continuous` puts flow on every draining hex; only the tag means channel."""
    cfg = WorldConfig()
    hx = _hex((0, 0))
    hx.river_flow = 1.0
    assert not navigable(hx, cfg)


def test_water_multiplies_reach():
    """The model's central claim, as arithmetic.

    Nothing gates a city on water; water simply extends what can feed it, and the size
    gap between a river city and an inland one follows from this one multiplier.
    """
    cfg = WorldConfig()
    inland = haulage_range(_hex((0, 0)), cfg)
    port = haulage_range(_hex((1, 0), TerrainClass.OCEAN), cfg)
    assert inland == cfg.haulage_range_land
    assert port == pytest.approx(inland * cfg.haulage_range_water_mult)
    assert port > inland


# --- catchment allocation ----------------------------------------------------


def test_a_seat_owns_itself_at_zero_cost():
    hexes = _strip(5)
    owner, cost = allocate_catchments(hexes, [(2, 0)], 3.0, WorldConfig())
    assert owner[(2, 0)] == (2, 0)
    assert cost[(2, 0)] == 0.0


def test_budget_bounds_the_catchment():
    """On flat ground with road_flat_cost 1.0, cost is distance, so the edge is countable."""
    cfg = WorldConfig()
    hexes = _strip(20)
    owner, cost = allocate_catchments(hexes, [(0, 0)], 4.0, cfg)
    assert max(cost.values()) < 4.0
    assert (3, 0) in owner
    assert (5, 0) not in owner


def test_catchments_are_disjoint_and_go_to_the_nearer_seat():
    hexes = _strip(11)
    owner, _ = allocate_catchments(hexes, [(0, 0), (10, 0)], 20.0, WorldConfig())
    assert owner[(1, 0)] == (0, 0)
    assert owner[(9, 0)] == (10, 0)
    # One owner per hex is structural: `owner` is a dict, so this asserts coverage.
    assert set(owner) == set(hexes)


def test_a_ridge_splits_two_catchments():
    """Why catchments are costed rather than drawn as discs.

    A ridge is made of *elevation*, not of terrain class: what stops a catchment is the
    climb, so this raises real ground between the two seats rather than labelling a hex
    a mountain.  Each seat then keeps its own valley instead of the boundary falling at
    the midpoint.
    """
    cfg = WorldConfig()
    hexes = _strip(9)
    # 500 m of climb over one hex, against travel_ascent_per_hex of 125 m: four units of
    # budget to go up, which is most of a six-unit day.
    hexes[(4, 0)].elevation = 500.0 / cfg.road_elev_range_m

    owner, _ = allocate_catchments(hexes, [(0, 0), (8, 0)], 6.0, cfg)
    assert owner[(3, 0)] == (0, 0), "west valley should be wholly western"
    assert owner[(5, 0)] == (8, 0), "east valley should be wholly eastern"
    assert (4, 0) not in owner, "the ridge is dearer than the budget and stays unclaimed"

    # The counterfactual: level the ridge and the catchments meet across it instead, so
    # the split above is the relief talking and not the budget.
    hexes[(4, 0)].elevation = 0.0
    flat_owner, _ = allocate_catchments(hexes, [(0, 0), (8, 0)], 6.0, cfg)
    assert (4, 0) in flat_owner


def test_descending_a_ridge_is_free():
    """Naismith counts climb only.

    A catchment that paid for going downhill would refuse to follow a valley, which is
    the one direction it ought to run.
    """
    cfg = WorldConfig()
    hexes = _strip(6)
    for q in range(6):
        hexes[(q, 0)].elevation = (5 - q) * 400.0 / cfg.road_elev_range_m

    owner, cost = allocate_catchments(hexes, [(0, 0)], 6.0, cfg)
    assert set(owner) == set(hexes), "walking downhill should cost no more than the distance"
    assert cost[(5, 0)] == pytest.approx(5 * cfg.road_flat_cost)


def test_a_high_plateau_is_walkable():
    """Relief enters as ascent, not as altitude: level ground is level however high."""
    cfg = WorldConfig()
    hexes = _strip(6)
    for q in range(6):
        hexes[(q, 0)].elevation = 0.9

    owner, _ = allocate_catchments(hexes, [(0, 0)], 6.0, cfg)
    assert set(owner) == set(hexes)


def test_water_is_not_traversable():
    """Otherwise one coastal seat claims a whole sea, and every catchment beyond it."""
    cfg = WorldConfig()
    hexes = _strip(9)
    hexes[(4, 0)] = _hex((4, 0), TerrainClass.OCEAN)
    owner, _ = allocate_catchments(hexes, [(0, 0)], 50.0, cfg)
    assert (3, 0) in owner
    assert (5, 0) not in owner, "catchment walked across open water"
    assert (8, 0) not in owner


def test_no_seats_yields_nothing():
    assert allocate_catchments(_strip(5), [], 10.0, WorldConfig()) == ({}, {})


def test_allocation_is_deterministic_regardless_of_seat_order():
    """Ties break on (cost, coord, owner), never on the order seats were passed."""
    cfg = WorldConfig()
    hexes = _strip(11)
    a, _ = allocate_catchments(hexes, [(0, 0), (10, 0)], 20.0, cfg)
    b, _ = allocate_catchments(hexes, [(10, 0), (0, 0)], 20.0, cfg)
    assert a == b


# --- the fishery rim ---------------------------------------------------------


def test_a_coastal_seat_gets_the_water_it_touches():
    cfg = WorldConfig()
    hexes = _strip(4)
    hexes[(3, 0)] = _hex((3, 0), TerrainClass.OCEAN)
    owner, cost = allocate_catchments(hexes, [(0, 0)], 10.0, cfg)
    assert (3, 0) not in owner

    owner, cost = fishery_rim(hexes, owner, cost)
    assert owner[(3, 0)] == (0, 0), "coastal seat did not gain its fishery"
    assert cost[(3, 0)] == cost[(2, 0)], "rim should cost what the land that fishes it costs"


def test_the_rim_does_not_walk_out_to_sea():
    """One hex of water per claimed land hex — granted, not traversed."""
    cfg = WorldConfig()
    hexes = _strip(2)
    for q in range(2, 8):
        hexes[(q, 0)] = _hex((q, 0), TerrainClass.OCEAN)
    owner, cost = allocate_catchments(hexes, [(0, 0)], 10.0, cfg)
    owner, _ = fishery_rim(hexes, owner, cost)
    assert (2, 0) in owner
    assert (3, 0) not in owner, "fishery rim spread across open water"


def test_fishery_rim_does_not_mutate_its_input():
    cfg = WorldConfig()
    hexes = _strip(3)
    hexes[(2, 0)] = _hex((2, 0), TerrainClass.OCEAN)
    owner, cost = allocate_catchments(hexes, [(0, 0)], 10.0, cfg)
    before = dict(owner)
    fishery_rim(hexes, owner, cost)
    assert owner == before


# --- gather ------------------------------------------------------------------


def test_gather_weights_by_distance():
    hexes = _strip(5)
    owner, cost = allocate_catchments(hexes, [(0, 0)], 10.0, WorldConfig())
    values = dict.fromkeys(hexes, 1.0)

    near = gather(values, {(1, 0): (0, 0)}, {(1, 0): 1.0}, 10.0)
    far = gather(values, {(1, 0): (0, 0)}, {(1, 0): 9.0}, 10.0)
    assert near[(0, 0)] > far[(0, 0)]


def test_gather_is_monotone_in_value():
    """More food in the catchment can only mean a larger draw."""
    owner = {(0, 0): (0, 0), (1, 0): (0, 0)}
    cost = {(0, 0): 0.0, (1, 0): 1.0}
    lean = gather({(0, 0): 1.0, (1, 0): 1.0}, owner, cost, 10.0)
    rich = gather({(0, 0): 1.0, (1, 0): 2.0}, owner, cost, 10.0)
    assert rich[(0, 0)] > lean[(0, 0)]


def test_gather_ignores_what_lies_beyond_the_range():
    owner = {(0, 0): (0, 0), (1, 0): (0, 0)}
    cost = {(0, 0): 0.0, (1, 0): 20.0}
    totals = gather({(0, 0): 1.0, (1, 0): 99.0}, owner, cost, 10.0)
    assert totals[(0, 0)] == pytest.approx(1.0)


def test_a_seat_with_nothing_to_draw_is_absent_rather_than_zero():
    totals = gather({(0, 0): 0.0}, {(0, 0): (0, 0)}, {(0, 0): 0.0}, 10.0)
    assert totals == {}


# --- settleable --------------------------------------------------------------


def test_settleable_excludes_water_mountain_and_bog():
    from worldgen.core.hex import Biome

    hexes = {
        (0, 0): _hex((0, 0)),
        (1, 0): _hex((1, 0), TerrainClass.OCEAN),
        (2, 0): _hex((2, 0), TerrainClass.LAKE),
        (3, 0): _hex((3, 0), TerrainClass.MOUNTAIN),
        (4, 0): _hex((4, 0), biome=Biome.WETLAND),
        (5, 0): _hex((5, 0), TerrainClass.HILL, land_cover=LandCover.OPEN),
    }
    assert settleable(hexes, WorldConfig()) == {(0, 0), (5, 0)}
