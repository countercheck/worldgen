"""Tolls at chokepoints (tech-debt #142): bridges and portages, and the towns they found.

Structural claims only. How many toll towns a seed grows is the seed's business; what is
tested is that a toll is paid only where a cargo passes, by somebody other than the two
ends, out of the cargo rather than on top of it; that a toll nobody collected founds a
settlement only when it pays for one; and that tolls at 0 leave the world as it was.
"""

import numpy as np
import pytest

from tests.worlds import build_pipeline, build_world
from worldgen.core.config import WorldConfig
from worldgen.core.hex import (
    Hex,
    Settlement,
    SettlementRole,
    SettlementTier,
    SoilQuality,
    TerrainClass,
)
from worldgen.core.hex_grid import distance, side_hexes
from worldgen.core.world_state import WorldState
from worldgen.export import json_export
from worldgen.stages.cities import BRIDGE, PORTAGE, QUAY, CityPromotionStage
from worldgen.stages.haulage import make_bulk_cost
from worldgen.stages.resources import ResourceStage
from worldgen.stages.riverside import Rivers, river_index

# A 96x96 temperate world whose cataracts and bridges carry trade: existing towns collect
# tolls on it, and a portage town is founded on one nobody held.
_DEFAULTS = {"regional_climate": "temperate"}
_KW = dict(seed=11, width=96, height=96, model="organic", **_DEFAULTS)
_OFF = dict(toll_bridge_share=0.0, toll_portage_share=0.0)


def _grid(n=8):
    return {
        (q, r): Hex(coord=(q, r), terrain_class=TerrainClass.LAND)
        for q in range(n)
        for r in range(n)
    }


def _town(coord, pop=1000, tier=SettlementTier.TOWN, role=SettlementRole.MARKET):
    return Settlement(coord=coord, tier=tier, role=role, population=pop, name=f"t{coord}")


def _eastward(r, n=8):
    return {(q, r): (q + 1, r) for q in range(n - 1)}


# --- where a cargo pays -------------------------------------------------------


def test_a_bridge_is_charged_only_to_cargo_that_crosses_it():
    """A bridge on row 0 tolls the flow along row 0 and not the flow along row 6."""
    cfg = WorldConfig(toll_bridge_share=0.1)
    hexes = _grid()
    rivers = Rivers(bridged=frozenset({frozenset({(3, 0), (4, 0)})}))
    over = CityPromotionStage._charges((0, 0), (7, 0), _eastward(0), hexes, cfg, rivers)
    past = CityPromotionStage._charges((0, 6), (7, 6), _eastward(6), hexes, cfg, rivers)
    assert [(c.kind, c.share) for c in over] == [(BRIDGE, 0.1)]
    assert over[0].site in {(3, 0), (4, 0)}
    assert past == []


def test_a_toll_town_collects_only_from_the_flows_that_cross_its_site():
    cfg = WorldConfig(toll_bridge_share=0.1, toll_radius=1)
    hexes = _grid()
    rivers = Rivers(bridged=frozenset({frozenset({(3, 0), (4, 0)})}))
    stage = CityPromotionStage(cfg, np.random.default_rng(0))
    holder = (3, 1)  # a hex from the bridge, four rows from the other flow
    seats = {c: _town(c) for c in [(0, 0), (7, 0), (0, 6), (7, 6), holder]}

    over = CityPromotionStage._charges((0, 0), (7, 0), _eastward(0), hexes, cfg, rivers)
    paid = stage._pay(over, (0, 0), (7, 0), seats, 500.0)
    assert [(h, cut) for _, h, cut in paid] == [(holder, pytest.approx(50.0))]

    past = CityPromotionStage._charges((0, 6), (7, 6), _eastward(6), hexes, cfg, rivers)
    assert stage._pay(past, (0, 6), (7, 6), seats, 500.0) == []


def test_neither_end_of_a_flow_pays_itself_a_toll():
    """A market tolling its own grain over its own bridge would only keep what it had."""
    cfg = WorldConfig(toll_bridge_share=0.1, toll_radius=2)
    hexes = _grid()
    rivers = Rivers(bridged=frozenset({frozenset({(1, 0), (2, 0)})}))
    stage = CityPromotionStage(cfg, np.random.default_rng(0))
    seats = {c: _town(c) for c in [(0, 0), (7, 0)]}
    charges = CityPromotionStage._charges((0, 0), (7, 0), _eastward(0), hexes, cfg, rivers)
    assert charges and stage._pay(charges, (0, 0), (7, 0), seats, 500.0) == []


def test_a_boat_passing_along_the_river_pays_no_bridge():
    """Two banks of one navigable reach are one voyage, whatever is built over it."""
    cfg = WorldConfig(toll_bridge_share=0.1)
    hexes = _grid()
    pair = frozenset({(3, 0), (4, 0)})
    rivers = Rivers(
        reaches={(3, 0): frozenset({0}), (4, 0): frozenset({0})}, bridged=frozenset({pair})
    )
    charges = CityPromotionStage._charges((0, 0), (7, 0), _eastward(0), hexes, cfg, rivers)
    assert BRIDGE not in {c.kind for c in charges}


def test_a_portage_pays_once_in_place_of_its_two_landings():
    """Down a river, round a cataract, and on: one toll at the falls, not three charges."""
    cfg = WorldConfig(toll_portage_share=0.4, transship_share=0.2)
    hexes = _grid()
    # A reach either side of the falls at (3, 0)-(4, 0); the cargo boards at (1, 0).
    rivers = Rivers(
        reaches={(1, 0): frozenset({0}), (2, 0): frozenset({0}), (5, 0): frozenset({1})},
        portage=frozenset({(3, 0), (4, 0)}),
    )
    single = CityPromotionStage._charges((0, 0), (7, 0), _eastward(0), hexes, cfg, rivers)
    assert [c.kind for c in single].count(PORTAGE) == 1
    assert next(c for c in single if c.kind == PORTAGE).share == 0.4

    off = WorldConfig(toll_portage_share=0.0, transship_share=0.2)
    landings = CityPromotionStage._charges((0, 0), (7, 0), _eastward(0), hexes, off, rivers)
    assert {c.kind for c in landings} == {QUAY}
    # The two landings are back, and are exactly the quays on the route.
    quays = CityPromotionStage._break_points((0, 0), (7, 0), _eastward(0), hexes, off, rivers)
    assert [c.site for c in landings] == quays
    assert len(single) == len(quays) - 2 + 1


def test_a_tolled_bridge_costs_the_haul_the_toll_is_worth():
    """A toll of share s weighs what s of `haulage_range_land` of haul does."""
    hexes = _grid()
    rivers = Rivers(bridged=frozenset({frozenset({(3, 0), (4, 0)})}))
    free = make_bulk_cost(hexes, WorldConfig(toll_bridge_share=0.0), rivers)[1]
    cfg = WorldConfig(toll_bridge_share=0.1)
    tolled = make_bulk_cost(hexes, cfg, rivers)[1]
    a, b = hexes[(3, 0)], hexes[(4, 0)]
    assert tolled(a, b) - free(a, b) == pytest.approx(0.1 * cfg.haulage_range_land)
    assert tolled(hexes[(1, 0)], hexes[(2, 0)]) == free(hexes[(1, 0)], hexes[(2, 0)])


# --- founding on a toll nobody collected ---------------------------------------


def _toll_state(soil, kind=BRIDGE, people=100.0):
    state = WorldState.empty(1, 12, 12, WorldConfig().grid_layout)
    state.hexes = {
        (q, r): Hex(coord=(q, r), terrain_class=TerrainClass.LAND, soil=SoilQuality.ARABLE)
        for q in range(12)
        for r in range(12)
    }
    state.hexes[(1, 1)].soil = soil
    city = _town((8, 8), 20_000, SettlementTier.CITY)
    state.settlements = [city]
    state.metadata["unhandled_tolls"] = [[1, 1, 8, 8, 2.0, people, kind]]
    return state, city


@pytest.mark.parametrize("people, founded", [(79.0, False), (81.0, True)])
def test_a_caravansary_is_founded_only_where_the_toll_pays_for_it(people, founded):
    """On ground that feeds nobody only the toll can found a place, and only above the floor."""
    cfg = WorldConfig(toll_min_draw=1.0, people_per_food=80.0, port_min_population=250)
    state, city = _toll_state(SoilQuality.UNUSABLE, people=people)
    ResourceStage(cfg, np.random.default_rng(0))._tolls(state, lambda c: True)
    new = [s for s in state.settlements if s is not city]
    if not founded:
        assert new == [] and city.population == 20_000
        return
    (caravansary,) = new
    assert caravansary.role is SettlementRole.CARAVANSARY
    assert caravansary.tier is SettlementTier.VILLAGE  # below port_min_population
    assert caravansary.population + city.population == 20_000
    assert caravansary.population >= cfg.toll_min_draw * cfg.people_per_food


@pytest.mark.parametrize(
    "soil, kind, role",
    [
        (SoilQuality.ARABLE, BRIDGE, SettlementRole.BRIDGE),
        (SoilQuality.UNUSABLE, PORTAGE, SettlementRole.PORTAGE),
        (SoilQuality.ARABLE, PORTAGE, SettlementRole.PORTAGE),
    ],
)
def test_a_toll_town_is_named_for_its_toll(soil, kind, role):
    """A bridge town on good ground is a bridge town; a portage town is one on any ground."""
    cfg = WorldConfig(toll_min_draw=1.0, port_min_population=250)
    state, city = _toll_state(soil, kind=kind, people=300.0)
    ResourceStage(cfg, np.random.default_rng(0))._tolls(state, lambda c: True)
    (town,) = [s for s in state.settlements if s is not city]
    assert town.role is role and town.tier is SettlementTier.TOWN


def test_a_settlement_founded_beside_a_toll_collects_it_instead():
    """A port founded a hex from the bridge takes its toll rather than a second town."""
    cfg = WorldConfig(toll_min_draw=1.0, toll_radius=2)
    state, city = _toll_state(SoilQuality.ARABLE, people=300.0)
    port = _town((2, 1), 300, role=SettlementRole.PORT)
    state.settlements.append(port)
    ResourceStage(cfg, np.random.default_rng(0))._tolls(state, lambda c: True)
    assert len(state.settlements) == 2
    assert port.population == 600 and city.population == 20_000 - 300


# --- whole worlds ----------------------------------------------------------------


@pytest.fixture(scope="module")
def tolled():
    return build_world(until="ResourceStage", **_KW)


def test_tolls_move_people_and_make_none(tolled):
    """Conserved: the same world with tolls off has the same people, differently placed."""
    off = build_world(until="ResourceStage", **_KW, **_OFF)
    total = sum(s.population for s in tolled.settlements)
    assert abs(total - sum(s.population for s in off.settlements)) <= len(tolled.settlements)
    before = build_world(until="LandUseStage", **_KW)
    assert abs(total - sum(s.population for s in before.settlements)) <= len(tolled.settlements)


def test_every_toll_is_collected_at_a_chokepoint_within_reach(tolled):
    cfg = WorldConfig(**_DEFAULTS)
    rivers = river_index(tolled, cfg)
    bridge_ends = {h for pair in rivers.bridged for h in pair}
    rows = tolled.metadata.get("tolls", [])
    assert rows, "the fixture world should collect some toll"
    for q, r, kind, cq, cr, food, people in rows:
        assert distance((q, r), (cq, cr)) <= cfg.toll_radius
        assert people > 0.0 and food > 0.0
        if kind == BRIDGE:
            assert (q, r) in bridge_ends
        else:
            assert kind == PORTAGE and (q, r) in rivers.portage


def test_tolls_at_zero_are_the_world_without_tolls(monkeypatch):
    """With both shares at 0 a cargo pays exactly the quays it always did, and nothing else.

    Built twice: once as it ships with tolls at 0, once with the toll code bypassed — the
    charges replaced by the bare quays and toll towns never founded. The two worlds must
    match in every field but the settings that name the tolls.
    """

    def world():
        return build_pipeline(**_KW, **_OFF).run().to_dict()

    shipped = world()

    def quays_only(cls, source, seat, toward, hexes, cfg, rivers):
        from worldgen.stages.cities import Charge

        return [
            Charge(q, QUAY, cfg.transship_share, cfg.transship_radius, True)
            for q in cls._break_points(source, seat, toward, hexes, cfg, rivers)
        ]

    monkeypatch.setattr(CityPromotionStage, "_charges", classmethod(quays_only))
    monkeypatch.setattr(ResourceStage, "_tolls", lambda self, state, reachable: None)
    bare = world()
    for key in shipped:
        if key == "metadata":
            continue
        assert shipped[key] == bare[key], key
    assert shipped["metadata"] == bare["metadata"]
    assert "tolls" not in shipped["metadata"]


def test_the_new_roles_survive_a_json_round_trip(tmp_path):
    ws = WorldState.empty(1, 4, 4, WorldConfig().grid_layout)
    ws.hexes = {
        (q, r): Hex(coord=(q, r), terrain_class=TerrainClass.LAND)
        for q in range(4)
        for r in range(4)
    }
    roles = [SettlementRole.BRIDGE, SettlementRole.PORTAGE, SettlementRole.CARAVANSARY]
    ws.settlements = [
        _town((i, 0), 100 + i, SettlementTier.VILLAGE, role) for i, role in enumerate(roles)
    ]
    path = tmp_path / "w.json"
    json_export.save(ws, path)
    back = json_export.load(path)
    assert [s.role for s in back.settlements] == roles
    assert [(s.coord, s.population) for s in back.settlements] == [
        (s.coord, s.population) for s in ws.settlements
    ]


@pytest.mark.parametrize("name", ["toll_bridge_share", "toll_portage_share"])
def test_toll_shares_are_bounded(name):
    with pytest.raises(ValueError, match=name):
        WorldConfig(**{name: 0.6})


@pytest.mark.parametrize("name", ["toll_radius", "toll_min_draw"])
def test_toll_settings_are_not_negative(name):
    with pytest.raises(ValueError, match=name):
        WorldConfig(**{name: -1})


def test_bridged_sides_are_the_ones_crossing_stage_bridged(tolled):
    rivers = river_index(tolled, WorldConfig(**_DEFAULTS))
    bridged = {
        frozenset(side_hexes(side))
        for side, rs in tolled.river_sides.items()
        if "bridge" in rs.tags and "ford" not in rs.tags
    }
    assert rivers.bridged == frozenset(bridged)
