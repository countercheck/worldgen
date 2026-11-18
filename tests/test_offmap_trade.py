"""Trade from beyond the map, through the gates on its edges (tech-debt #149).

The claims: every flow starts at a gate and follows a connected path to where it goes;
imports go to towns and cities, and transit to a gate on another edge, never from one sea
gate to another; a gate sends what it was given, no more; the people the trade brings are
counted, and none when it is off; and the roads carry it out to the land gates.
"""

import copy
import dataclasses

import numpy as np
import pytest

from tests.worlds import build_world
from worldgen.core.config import WorldConfig
from worldgen.core.hex import SettlementTier
from worldgen.core.hex_grid import neighbors
from worldgen.stages.cities import CityPromotionStage
from worldgen.stages.offmap_gates import LAND, SEA
from worldgen.stages.offmap_trade import IMPORT, TRANSIT, offmap_trade_flows
from worldgen.stages.riverside import river_index

_WORLD = {
    "seed": 42,
    "width": 96,
    "height": 96,
    "model": "organic",
    "regional_climate": "temperate",
    "continent_falloff_edges": ("south",),
}


@pytest.fixture(scope="module")
def promoted():
    return build_world(until="CityPromotionStage", **_WORLD)


@pytest.fixture(scope="module")
def routed():
    return build_world(until="InterurbanRoadStage", **_WORLD)


def _cfg(state, **changes):
    return dataclasses.replace(WorldConfig(**state.metadata["config"]), **changes)


@pytest.fixture(scope="module")
def flows(promoted):
    cfg = _cfg(promoted)
    return offmap_trade_flows(promoted, cfg, river_index(promoted, cfg))


def test_every_flow_runs_from_its_gate_along_a_connected_path(promoted, flows):
    gates = {g.coord for g in promoted.gates}
    assert flows
    for f in flows:
        assert f.origin in gates and f.people > 0.0
        assert f.path[0] == f.origin and f.path[-1] == f.dest
        for a, b in zip(f.path, f.path[1:], strict=False):
            assert b in neighbors(a), f"{f.origin}->{f.dest} jumps from {a} to {b}"


def test_imports_go_to_towns_and_transit_to_another_edge(promoted, flows):
    towns = {
        s.coord
        for s in promoted.settlements
        if s.tier in (SettlementTier.TOWN, SettlementTier.CITY)
    }
    gate = {g.coord: g for g in promoted.gates}
    kinds = {f.kind for f in flows}
    assert kinds == {IMPORT, TRANSIT}, kinds
    for f in flows:
        if f.kind == IMPORT:
            assert f.dest in towns
        else:
            a, b = gate[f.origin], gate[f.dest]
            assert a.edge != b.edge
            assert not (a.kind == b.kind == SEA), "a ship sailing past never touches the map"


def test_a_gate_sends_no_more_than_its_share(promoted, flows):
    cfg = _cfg(promoted)
    # The towns as `flows` found them, which is after the trade's own gains.
    before = sum(
        s.population
        for s in promoted.settlements
        if s.tier in (SettlementTier.TOWN, SettlementTier.CITY)
    )
    per_gate = cfg.offmap_trade_share * before / len(promoted.gates)
    sent: dict[tuple[int, int], float] = {}
    for f in flows:
        sent[f.origin] = sent.get(f.origin, 0.0) + f.people
    for origin, people in sent.items():
        assert people <= per_gate * 1.0001, origin


def test_the_trade_brings_people_and_counts_them(promoted):
    imported = promoted.metadata.get("offmap_imports", 0.0)
    assert imported > 0.0
    assert any(row[5] == "offmap" for row in promoted.metadata["freight"])


def test_no_trade_means_no_flows_and_no_people():
    off = build_world(until="CityPromotionStage", **_WORLD, offmap_trade_share=0.0)
    assert off.gates == []
    assert "offmap_imports" not in off.metadata
    assert "offmap_trade" not in off.metadata
    assert not any(row[5] == "offmap" for row in off.metadata.get("freight", []))


def test_more_trade_brings_more_people(promoted):
    """The stage run again over the same towns at twice the share brings more."""
    before = build_world(until="LandUseStage", **_WORLD)

    def brought(share):
        state = copy.deepcopy(before)
        cfg = _cfg(before, offmap_trade_share=share)
        state.gates = promoted.gates
        state = CityPromotionStage(cfg, np.random.default_rng(0)).run(state)
        return state.metadata.get("offmap_imports", 0.0)

    assert brought(0.1) > brought(0.05) > 0.0


def test_the_roads_carry_it_out_to_the_land_gates(routed):
    land = [g for g in routed.gates if g.kind == LAND]
    assert land
    reached = [g for g in land if routed.hexes[g.coord].road_connections]
    assert reached, "no road runs out to any land gate"
    assert any(a in {g.coord for g in routed.gates} for a, _ in routed.journeys)


def test_a_town_pays_its_own_tolls_on_its_trade_with_a_gate():
    from worldgen.stages.chokepoints import _payers

    gate, town, other = (0, 5), (10, 5), (20, 5)
    offmap = {gate, other}
    assert _payers(gate, town, offmap) == (town, town)
    assert _payers(town, gate, offmap) == (town, town)
    assert _payers(gate, other, offmap) == (gate, other)


def test_road_tolls_bring_no_more_than_the_through_trade_had_left():
    tolled = build_world(until="ChokepointStage", **_WORLD)
    routed = build_world(until="InterurbanRoadStage", **_WORLD)
    left = sum(row[6] for row in routed.metadata["offmap_trade"])
    brought = tolled.metadata["offmap_imports"] - routed.metadata["offmap_imports"]
    assert 0.0 <= brought <= left + 1.0
