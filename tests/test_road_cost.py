"""Unit tests for the road cost model in worldgen.stages.road_cost.

These tests construct tiny synthetic hex grids and exercise the cost helpers
directly, without running the full pipeline. They verify both the arithmetic
of individual cost components and the A* behaviour they produce.
"""

import pytest

from worldgen.core.config import WorldConfig
from worldgen.core.errors import RoutingError
from worldgen.core.hex import Hex, TerrainClass
from worldgen.core.hex_grid import astar, distance, neighbors
from worldgen.core.world_state import River
from worldgen.stages.road_cost import (
    bank_discount,
    ferry_link,
    make_road_edge_cost,
    reachable_under_constraint,
    river_crossing_edge_cost,
    river_edges,
    river_hex_cost,
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
    # HydrologyStage sets both on every channel hex, and the road costs identify a river
    # by the tag — flow alone is written on all draining land when river_flow_continuous
    # is on, so it cannot be the identity. A fixture setting only the flow would be
    # describing a hex that production never produces.
    h.tags.add("river")
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


# ---------- bank_discount --------------------------------------------------


def _bank_grid(flow=0.8):
    """A river hex at (1, 0) with its neighbours as plain land."""
    hexes = {(1, 0): _river_flat((1, 0), flow=flow)}
    for n in neighbors((1, 0)):
        hexes[n] = _flat(n)
    hexes[(9, 9)] = _flat((9, 9))  # far from any river
    return hexes


def test_bank_discount_zero_away_from_any_river():
    cfg = WorldConfig()
    hexes = _bank_grid()
    assert bank_discount(hexes[(9, 9)], hexes, cfg) == 0.0


def test_bank_discount_zero_on_the_river_itself():
    """The pull belongs on the bank; a river hex is a crossing, not a route."""
    cfg = WorldConfig()
    hexes = _bank_grid()
    assert bank_discount(hexes[(1, 0)], hexes, cfg) == 0.0


def test_bank_discount_applies_beside_the_river():
    cfg = WorldConfig()
    hexes = _bank_grid()
    for n in neighbors((1, 0)):
        assert bank_discount(hexes[n], hexes, cfg) > 0.0


def test_bank_discount_scales_with_adjacent_flow():
    cfg = WorldConfig()
    small = _bank_grid(flow=0.3)
    big = _bank_grid(flow=1.0)
    bank = next(iter(neighbors((1, 0))))
    assert bank_discount(big[bank], big, cfg) > bank_discount(small[bank], small, cfg)


def test_bank_discount_min_flow_floor():
    cfg = WorldConfig(road_bank_discount_min_flow=0.4)
    hexes = _bank_grid(flow=0.05)
    bank = next(iter(neighbors((1, 0))))
    assert bank_discount(hexes[bank], hexes, cfg) == pytest.approx(cfg.road_bank_discount * 0.4)


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
    # The river is the last *land* hex of its course, with the sea beyond it: hydrology
    # never puts flow or a river tag on a water hex, so a river mouth is this shape, not
    # an ocean hex carrying flow.  Stepping from it to the sea both embarks and leaves
    # the channel, which is the combination under test.
    river_mouth = _river_flat((0, 0), flow=0.5)
    river_mouth.elevation = 0.5
    sea = Hex(coord=(1, 0), elevation=0.5, terrain_class=TerrainClass.OCEAN)
    cost = road_edge_cost(river_mouth, sea, cfg)
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
        f"A* should detour to the low-flow stream half (q>=3), but crossed at q={crossing_cols}"
    )


def _valley_grid(cfg, flow=0.8):
    """An 8x3 grid with a river running the length of row r=1, plus its edge set."""

    def factory(q, r):
        if r == 1:
            return _river_flat((q, r), flow=flow)
        return _flat((q, r))

    hexes = _build_grid(8, 3, factory)
    river = River(hexes=[(q, 1) for q in range(8)], flow_volume=flow)
    return hexes, river_edges([river])


def test_astar_follows_the_bank_not_the_channel():
    """The valley still pulls routes in, but along the bank rather than down the river.

    A road drawn on the channel hides which side of the river it is on, so the discount
    lives on the bank and the river's own hexsides are excluded outright.
    """
    cfg = WorldConfig()
    hexes, blocked = _valley_grid(cfg)

    def node_cost(hx):
        return max(0.1, terrain_base_cost(hx, cfg) - bank_discount(hx, hexes, cfg))

    edge_cost = make_road_edge_cost(cfg, blocked)

    path = astar(hexes, (0, 0), (7, 0), node_cost, edge_cost)
    assert path is not None
    # No leg of the route may run along the river's own hexsides.
    used = {frozenset((a, b)) for a, b in zip(path, path[1:], strict=False)}
    assert not (used & blocked), "route travelled down the river channel"
    # And it should stay in the valley rather than wandering off the far row.
    assert all(c[1] in (0, 1) for c in path), f"route left the valley: {path}"


def test_astar_may_still_cross_the_river():
    """Crossing is untouched — only travelling *along* the channel is forbidden."""
    cfg = WorldConfig()
    hexes, blocked = _valley_grid(cfg)

    def node_cost(hx):
        return max(0.1, terrain_base_cost(hx, cfg) - bank_discount(hx, hexes, cfg))

    path = astar(hexes, (0, 0), (0, 2), node_cost, make_road_edge_cost(cfg, blocked))
    assert path is not None
    assert any(c[1] == 1 for c in path), "a crossing must be able to enter the river row"
    used = {frozenset((a, b)) for a, b in zip(path, path[1:], strict=False)}
    assert not (used & blocked)


def test_channel_hexside_between_two_river_hexes_is_never_exempt():
    """A town on the water may be reached, not used as a licence to carry on down it.

    Both ends here are river hexes, so exempting the edge would let a road step out of
    the town and keep going along the channel one hex at a time — exactly what the
    exclusion exists to stop.
    """
    cfg = WorldConfig()
    hexes, blocked = _valley_grid(cfg)
    assert frozenset(((3, 1), (4, 1))) in blocked  # really is a channel hexside

    plain = make_road_edge_cost(cfg, blocked)
    assert plain(hexes[(3, 1)], hexes[(4, 1)]) == float("inf")

    exempt = make_road_edge_cost(cfg, blocked, exempt_coords={(3, 1)})
    assert exempt(hexes[(3, 1)], hexes[(4, 1)]) == float("inf")


def test_settlement_exemption_opens_a_channel_hexside_onto_dry_land():
    """Where a river's drawn path runs onto a dry hex, a town there is still reachable."""
    cfg = WorldConfig()
    hexes = {
        (0, 0): _river_flat((0, 0), flow=0.8),
        (1, 0): _flat((1, 0)),  # on the river's drawn path, but carries no flow
    }
    blocked = river_edges([River(hexes=[(0, 0), (1, 0)], flow_volume=0.8)])
    assert frozenset(((0, 0), (1, 0))) in blocked

    plain = make_road_edge_cost(cfg, blocked)
    assert plain(hexes[(0, 0)], hexes[(1, 0)]) == float("inf")

    exempt = make_road_edge_cost(cfg, blocked, exempt_coords={(0, 0)})
    assert exempt(hexes[(0, 0)], hexes[(1, 0)]) < float("inf")


# ---------- river_hex_cost -------------------------------------------------


def test_river_hex_cost_only_on_river_hexes():
    cfg = WorldConfig()
    assert river_hex_cost(_flat((0, 0)), cfg) == 0.0
    assert river_hex_cost(_river_flat((0, 0), flow=0.5), cfg) == cfg.road_river_hex_cost


def test_river_hex_cost_leaves_a_crossing_affordable():
    """Priced to stop channel travel without stopping a crossing outright."""
    cfg = WorldConfig()
    # One river hex crossed once, versus the same hex travelled along for five steps.
    crossing = cfg.road_river_hex_cost
    channel_run = 5 * cfg.road_river_hex_cost
    detour_budget = 5 * cfg.road_flat_cost
    assert crossing < 2 * (cfg.road_river_crossing_base + cfg.road_river_crossing_flow)
    assert channel_run > detour_budget


# ---------- ferries ---------------------------------------------------------


def _cut_grid():
    """A one-hex-wide corridor whose middle two hexes are a river running lengthwise.

    Crossing a river is always legal, so a river only truly severs the map where the
    channel *is* the corridor — there is no bank to walk along. Getting from q=0 to q=5
    means using the drawn hexside between (2, 0) and (3, 0), which roads may not.
    """

    def factory(q, r):
        return _river_flat((q, r), flow=0.9) if q in (2, 3) else _flat((q, r))

    hexes = _build_grid(6, 1, factory)
    river = River(hexes=[(2, 0), (3, 0)], flow_volume=0.9)
    return hexes, river


def test_reachable_under_constraint_stops_at_the_channel():
    hexes, river = _cut_grid()
    seen = reachable_under_constraint(hexes, (0, 0), river_edges([river]), frozenset())
    assert seen == {(0, 0), (1, 0), (2, 0)}, "walk should stop at the channel hexside"


def test_reachable_under_constraint_matches_the_tightened_exemption():
    """A town on the channel does not reopen it — the walk still stops at the hexside.

    The component walk and the edge cost have to agree, or the ferry fallback would be
    reasoning about a different map than the router.
    """
    hexes, river = _cut_grid()
    blocked = river_edges([river])
    assert (3, 0) not in reachable_under_constraint(hexes, (0, 0), blocked, frozenset())
    assert (3, 0) not in reachable_under_constraint(hexes, (0, 0), blocked, {(2, 0)})


def test_ferry_link_picks_the_shortest_hop():
    cfg = WorldConfig()
    hexes, river = _cut_grid()
    blocked = river_edges([river])
    near = reachable_under_constraint(hexes, (0, 0), blocked, frozenset())
    far = set(hexes) - near
    assert far, "the corridor must actually be severed for this test to mean anything"

    ferry, paths = ferry_link(
        hexes,
        (0, 0),
        "City Testburg",
        far,
        cfg,
        blocked,
        frozenset(),
        lambda hx: terrain_base_cost(hx, cfg),
        make_road_edge_cost(cfg, blocked),
    )
    assert distance(ferry.a, ferry.b) <= cfg.road_ferry_max_hop
    assert ferry.a in near and ferry.b in far
    for p in paths:
        assert p[0] == (0, 0) and p[-1] == ferry.a


def test_ferry_lands_on_dry_land_off_the_channel():
    """A ferry is drawn as two anchorages, so neither may sit in the channel.

    Both components hold river hexes here — (2, 0) on the near side, (3, 0) on the far —
    and they are the closest pair of all, so an unfiltered shortest-hop search picks a
    crossing whose anchors both land mid-river.
    """
    cfg = WorldConfig()
    hexes, river = _cut_grid()
    blocked = river_edges([river])
    near = reachable_under_constraint(hexes, (0, 0), blocked, frozenset())
    far = set(hexes) - near
    assert (2, 0) in near and (3, 0) in far, "the tempting mid-channel pair must be on offer"

    ferry, _ = ferry_link(
        hexes,
        (0, 0),
        "City Testburg",
        far,
        cfg,
        blocked,
        frozenset(),
        lambda hx: terrain_base_cost(hx, cfg),
        make_road_edge_cost(cfg, blocked),
    )
    for landing in (ferry.a, ferry.b):
        hx = hexes[landing]
        assert hx.terrain_class not in (TerrainClass.OCEAN, TerrainClass.LAKE)
        assert hx.river_flow <= 0, f"anchorage at {landing} sits in the channel"


def test_ferry_link_raises_when_every_landing_is_wet():
    """No shore on one side is a routing failure, not a ferry moored in open water."""
    cfg = WorldConfig()
    hexes, river = _cut_grid()
    blocked = river_edges([river])
    for coord in ((4, 0), (5, 0)):
        hexes[coord].terrain_class = TerrainClass.OCEAN
    near = reachable_under_constraint(hexes, (0, 0), blocked, frozenset())

    with pytest.raises(RoutingError, match="no dry land off the channel"):
        ferry_link(
            hexes,
            (0, 0),
            "City Testburg",
            set(hexes) - near,
            cfg,
            blocked,
            frozenset(),
            lambda hx: terrain_base_cost(hx, cfg),
            make_road_edge_cost(cfg, blocked),
        )


def test_ferry_link_raises_when_no_plausible_hop_exists():
    """Beyond road_ferry_max_hop a ferry is not a plausible reading of the map."""
    cfg = WorldConfig(road_ferry_max_hop=1)
    hexes, river = _cut_grid()
    blocked = river_edges([river])
    near = reachable_under_constraint(hexes, (0, 0), blocked, frozenset())

    # The nearest dry landings are (1, 0) and (4, 0), three hexes apart.
    with pytest.raises(RoutingError, match="no plausible ferry"):
        ferry_link(
            hexes,
            (0, 0),
            "City Testburg",
            set(hexes) - near,
            cfg,
            blocked,
            frozenset(),
            lambda hx: terrain_base_cost(hx, cfg),
            make_road_edge_cost(cfg, blocked),
        )


def test_ferry_link_raises_when_nothing_lies_outside_the_component():
    cfg = WorldConfig()
    hexes, river = _cut_grid()
    blocked = river_edges([river])
    near = reachable_under_constraint(hexes, (0, 0), blocked, frozenset())

    with pytest.raises(RoutingError, match="no hex outside"):
        ferry_link(
            hexes,
            (0, 0),
            "City Testburg",
            set(near),
            cfg,
            blocked,
            frozenset(),
            lambda hx: terrain_base_cost(hx, cfg),
            make_road_edge_cost(cfg, blocked),
        )
