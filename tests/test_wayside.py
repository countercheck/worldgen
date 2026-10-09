"""Tolls on the road traffic, and crossroads towns (tech-debt #142).

`ChokepointStage` charges every journey `InterurbanRoadStage` routed at the bridges, passes
and desert stops it crosses, and founds crossroads towns where busy roads meet. Structural
claims: a toll is charged only to journeys that cross its site, by someone other than the
journey's ends, out of those ends rather than out of nothing; a crossroads stands only
where three busy roads meet, clear of every other settlement; and the same seed gives the
same towns.
"""

import numpy as np
import pytest

from tests.worlds import build_pipeline, build_world
from worldgen.core.config import WorldConfig
from worldgen.core.hex import Biome, Hex, SettlementRole, SettlementTier, TerrainClass
from worldgen.core.hex_grid import distance, hex_range
from worldgen.core.world_state import WorldState
from worldgen.stages.chokepoints import ChokepointStage
from worldgen.stages.riverside import Rivers, river_index
from worldgen.stages.wayside import BRIDGE, DESERT, PASS, crossroads, toll_sites

_CFG = dict(width=96, height=96, model="organic", regional_climate="temperate")
_KW = dict(seed=42, **_CFG)
_WAYSIDE = (
    SettlementRole.BRIDGE,
    SettlementRole.PASS,
    SettlementRole.CARAVANSARY,
    SettlementRole.CROSSROADS,
)


def _line(n=12, biome=None):
    """A road along r = 0, every hex joined to the next."""
    hexes = {}
    for q in range(n):
        for r in range(3):
            hexes[(q, r)] = Hex(coord=(q, r), terrain_class=TerrainClass.LAND, biome=biome)
    for q in range(n - 1):
        hexes[(q, 0)].road_connections.add((q + 1, 0))
        hexes[(q + 1, 0)].road_connections.add((q, 0))
    state = WorldState.empty(1, n, 3, WorldConfig().grid_layout)
    state.hexes = hexes
    return state


# --- where a journey pays ------------------------------------------------------


def test_a_bridge_is_charged_only_to_the_journeys_that_cross_it():
    state = _line()
    path = tuple((q, 0) for q in range(12))
    state.journeys = {((0, 0), (11, 0)): (5.0, path), ((0, 0), (3, 0)): (2.0, path[:4])}
    rivers = Rivers(bridged=frozenset({frozenset({(6, 0), (7, 0)})}))
    sites = toll_sites(state, WorldConfig(), rivers, lambda c: False)
    assert set(sites) == {((6, 0), BRIDGE)} or set(sites) == {((7, 0), BRIDGE)}
    (pairs,) = sites.values()
    assert pairs == {((0, 0), (11, 0)): 5.0}


def test_a_pass_off_the_road_charges_nothing():
    state = _line()
    path = tuple((q, 0) for q in range(12))
    state.journeys = {((0, 0), (11, 0)): (5.0, path)}
    sites = toll_sites(state, WorldConfig(), Rivers(), lambda c: c in {(5, 0), (5, 1)})
    assert set(sites) == {((5, 0), PASS)}


def test_a_desert_crossing_waters_at_the_oasis_by_the_road():
    cfg = WorldConfig(market_day_radius=10.0, toll_radius=2)
    state = _line(14, biome=Biome.DESERT)
    state.hexes[(9, 2)].tags.add("oasis")
    path = tuple((q, 0) for q in range(14))
    state.journeys = {((0, 0), (13, 0)): (5.0, path)}
    sites = toll_sites(state, cfg, Rivers(), lambda c: False)
    ((site, kind),) = sites
    assert kind == DESERT and distance(site, (9, 2)) == 2


def test_a_desert_stretch_shorter_than_a_day_needs_no_stop():
    cfg = WorldConfig(market_day_radius=10.0)
    state = _line(8, biome=Biome.DESERT)
    path = tuple((q, 0) for q in range(8))
    state.journeys = {((0, 0), (7, 0)): (5.0, path)}
    assert toll_sites(state, cfg, Rivers(), lambda c: False) == {}


def test_a_crossroads_needs_three_busy_roads():
    from worldgen.core.world_state import RoadEdge, RoadTier, road_edge_key

    cfg = WorldConfig(road_min_traffic=3, toll_per_journey=0.03)
    state = _line()
    centre = (5, 1)
    for n in [(5, 0), (4, 2), (6, 1)]:
        state.road_edges[road_edge_key(centre, n)] = RoadEdge(RoadTier.TRACK, 0.0, 10.0)
    state.journeys = {((5, 0), (6, 1)): (10.0, ((5, 0), centre, (6, 1)))}
    assert [c for _, c, _ in crossroads(state, cfg)] == [centre]
    # Make one of the three quiet: a junction of two busy roads is a bend.
    state.road_edges[road_edge_key(centre, (6, 1))] = RoadEdge(RoadTier.TRACK, 0.0, 1.0)
    assert crossroads(state, cfg) == []


# --- a whole world -------------------------------------------------------------


@pytest.fixture(scope="module")
def routed():
    return build_world(until="InterurbanRoadStage", **_KW)


@pytest.fixture(scope="module")
def tolled():
    return build_world(until="ChokepointStage", **_KW)


def test_the_road_tolls_move_people_and_make_none(routed):
    """Run the stage with the tolls on and off over the same roads: the villages are the
    same, so the totals must be too, but for rounding."""
    import copy

    def total(toll):
        state = copy.deepcopy(routed)
        cfg = WorldConfig(**_CFG, toll_per_journey=toll)
        state = ChokepointStage(cfg, np.random.default_rng(0)).run(state)
        return sum(s.population for s in state.settlements), len(state.settlements)

    on, n_on = total(0.03)
    off, _ = total(0.0)
    assert abs(on - off) <= n_on


def test_a_road_toll_is_collected_only_from_journeys_crossing_its_site(tolled):
    cfg = WorldConfig(**_CFG)
    rows = tolled.metadata.get("road_tolls", [])
    assert rows, "the fixture world should collect a road toll"
    sites = toll_sites(
        tolled, cfg, river_index(tolled, cfg), lambda c: PASS in tolled.hexes[c].tags
    )
    for q, r, kind, cq, cr, food, _people in rows:
        pairs = sites[((q, r), kind)]
        own = sum(n for pair, n in pairs.items() if (cq, cr) not in pair)
        assert distance((q, r), (cq, cr)) <= cfg.toll_radius
        assert food == pytest.approx(own * cfg.toll_per_journey, abs=1e-2)


def test_crossroads_stand_where_three_busy_roads_meet_clear_of_every_town(tolled):
    cfg = WorldConfig(**_CFG)
    found = [s for s in tolled.settlements if s.role is SettlementRole.CROSSROADS]
    for s in found:
        busy = sum(
            1
            for key, e in tolled.road_edges.items()
            if s.coord in key and e.traffic >= cfg.road_min_traffic
        )
        assert busy >= 3, f"crossroads at {s.coord} has {busy} busy roads"
        near = set(hex_range(s.coord, 2 * cfg.toll_radius))
        others = [o.coord for o in tolled.settlements if o is not s and o.coord in near]
        assert others == [], f"crossroads at {s.coord} crowds {others}"


def test_wayside_towns_stand_on_the_road(tolled):
    for s in tolled.settlements:
        if s.role in _WAYSIDE and s.tier is not SettlementTier.VILLAGE:
            assert tolled.hexes[s.coord].road_connections, f"{s.role.value} at {s.coord}"


def test_same_seed_same_wayside_towns():
    def towns():
        st = build_pipeline(until="ChokepointStage", **_KW).run()
        return sorted((s.coord, s.role.value, s.tier.value, s.population) for s in st.settlements)

    assert towns() == towns()


def test_toll_per_journey_at_zero_founds_no_wayside_town():
    st = build_world(until="ChokepointStage", **{**_KW, "toll_per_journey": 0.0})
    assert not any(
        s.role in (SettlementRole.PASS, SettlementRole.CROSSROADS) for s in st.settlements
    )
    assert "road_tolls" not in st.metadata
