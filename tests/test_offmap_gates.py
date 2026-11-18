"""Gates on the map's edges, where off-map trade comes in (tech-debt #149).

The claims: a gate is a border hex of its kind on its own edge; every edge that trades
and has land gets a land gate, and every one with sea a sea gate; a longer stretch of
border gets more of them; the first one on an edge is the cheapest way in; crowding
pushes gates apart; and the choice reads the terrain alone, so it is the same every time.
"""

import dataclasses
from itertools import combinations

import pytest

from tests.worlds import build_pipeline, build_world
from worldgen.core.config import WorldConfig
from worldgen.core.hex import TerrainClass
from worldgen.core.hex_grid import distance
from worldgen.core.world_state import WorldState
from worldgen.stages.haulage import travel_field
from worldgen.stages.offmap_gates import LAND, SEA, _land_scores, choose_gates
from worldgen.stages.riverside import river_index

# A temperate coast with the sea to the south, so the map has land edges and a sea edge.
_WORLD = {
    "seed": 42,
    "width": 96,
    "height": 96,
    "model": "organic",
    "regional_climate": "temperate",
    "continent_falloff_edges": ("south",),
}
_WATER = (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER)


@pytest.fixture(scope="module")
def world():
    return build_world(until="OffmapGateStage", **_WORLD)


def _cfg(state, **changes):
    return dataclasses.replace(WorldConfig(**state.metadata["config"]), **changes)


def _border(state, edge, kind):
    hexes = state.hexes
    return [
        c
        for c in hexes
        if edge in state.edges_of(c)
        and (
            hexes[c].terrain_class not in _WATER
            if kind == LAND
            else hexes[c].terrain_class is TerrainClass.OPEN_WATER
        )
    ]


def test_a_gate_is_a_border_hex_of_its_kind_on_its_own_edge(world):
    assert world.gates, "a 96x96 map traded with nothing beyond it"
    for g in world.gates:
        assert g.edge in world.edges_of(g.coord), g
        tc = world.hexes[g.coord].terrain_class
        if g.kind == LAND:
            assert tc not in _WATER, g
        else:
            assert g.kind == SEA and tc is TerrainClass.OPEN_WATER, g
    coords = [g.coord for g in world.gates]
    assert len(coords) == len(set(coords)), "two gates on one hex"


def test_every_trading_edge_gets_a_gate_of_each_kind_it_has_ground_for(world):
    cfg = _cfg(world)
    for edge in cfg.offmap_trade_edges:
        for kind, spacing in (
            (LAND, cfg.offmap_land_gate_spacing),
            (SEA, cfg.offmap_sea_gate_spacing),
        ):
            cells = _border(world, edge, kind)
            got = [g for g in world.gates if g.edge == edge and g.kind == kind]
            if not cells:
                assert not got, (edge, kind)
                continue
            assert 1 <= len(got) <= max(1, round(len(cells) / spacing)), (edge, kind, len(got))
    assert any(g.kind == SEA for g in world.gates), "the southern sea has no sea gate"
    assert any(g.kind == LAND for g in world.gates), "no road leaves the map"


def test_more_border_per_gate_means_fewer_gates(world):
    dense = choose_gates(world, _cfg(world, offmap_land_gate_spacing=8, offmap_sea_gate_spacing=8))
    sparse = choose_gates(
        world, _cfg(world, offmap_land_gate_spacing=80, offmap_sea_gate_spacing=80)
    )
    assert len(dense) > len(sparse)


def test_the_first_land_gate_on_an_edge_is_the_cheapest_way_in(world):
    cfg = _cfg(world, offmap_land_gate_spacing=10_000, offmap_gate_crowding_penalty=0.0)
    field = travel_field(world.hexes, cfg, river_index(world, cfg))
    for g in choose_gates(world, cfg):
        if g.kind != LAND:
            continue
        scores = _land_scores(world, cfg, field, g.edge, _border(world, g.edge, LAND))
        assert scores[g.coord] == min(scores.values()), g


def test_crowding_pushes_gates_on_an_edge_apart(world):
    def closest(gates):
        gaps = [
            distance(a.coord, b.coord)
            for a, b in combinations(gates, 2)
            if a.edge == b.edge and a.kind == b.kind
        ]
        return min(gaps)

    # Spaced wider than they crowd, as the defaults are: with the radius beyond the
    # spacing every candidate is crowded and the penalty ranks nothing.
    knobs = {
        "offmap_land_gate_spacing": 24,
        "offmap_sea_gate_spacing": 24,
        "offmap_gate_crowding_radius": 12,
    }
    loose = choose_gates(world, _cfg(world, offmap_gate_crowding_penalty=0.0, **knobs))
    tight = choose_gates(world, _cfg(world, offmap_gate_crowding_penalty=1.0, **knobs))
    assert closest(loose) == 1, "nothing to push apart: every cheap way in is alone"
    assert closest(tight) > closest(loose)


def test_an_edge_that_does_not_trade_has_no_gates(world):
    gates = choose_gates(world, _cfg(world, offmap_trade_edges=("north", "south")))
    assert gates
    assert {g.edge for g in gates} <= {"north", "south"}


def test_no_trade_means_no_gates(world):
    assert choose_gates(world, _cfg(world, offmap_trade_share=0.0)) == []


def test_the_gates_are_the_same_every_time():
    a = build_pipeline(until="OffmapGateStage", **{**_WORLD, "width": 48, "height": 48}).run()
    b = build_pipeline(until="OffmapGateStage", **{**_WORLD, "width": 48, "height": 48}).run()
    assert a.gates == b.gates and a.gates


def test_the_gates_survive_the_round_trip(world):
    again = WorldState.from_dict(world.to_dict())
    assert again.gates == world.gates
