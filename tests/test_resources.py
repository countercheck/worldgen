"""Resource settlements: ports at unattended quays, mines on the ore, camps in the woods.

Each claim here is structural — where a place may stand, and that founding it moved people
rather than made them — because how many of each a seed produces is the seed's business.
"""

import numpy as np
import pytest

from tests.worlds import build_world
from worldgen.core.config import WorldConfig
from worldgen.core.hex import (
    Hex,
    LandUse,
    Settlement,
    SettlementRole,
    SettlementTier,
    TerrainClass,
)
from worldgen.core.world_state import WorldState
from worldgen.stages.haulage import floatable
from worldgen.stages.resources import ResourceStage

# The same 96x96 world the city tests build, so the "before" half is shared with them.
_DEFAULTS = {
    "regional_climate": "temperate",
    "continent_falloff_edges": ("south",),
    "city_min_draw": 22.0,
}
_KW = dict(seed=42, width=96, height=96, model="organic", **_DEFAULTS)


def _resourced(**over):
    return build_world(until="ResourceStage", **{**_KW, **over})


def _founded(state, role):
    return [s for s in state.settlements if s.role is role]


def test_founding_moves_people_and_makes_none():
    before = build_world(until="CityPromotionStage", **_KW)
    after = _resourced()
    assert sum(s.population for s in after.settlements) == sum(
        s.population for s in before.settlements
    )


def test_the_world_gets_some_of_each():
    """Not a count to hit, only that none of the three is dead code on a temperate map."""
    state = _resourced()
    assert _founded(state, SettlementRole.MINING)
    assert _founded(state, SettlementRole.LUMBER)
    assert any("_port_" in s.name for s in state.settlements)


def test_a_mining_village_stands_on_the_ore():
    cfg = WorldConfig(**_DEFAULTS)
    state = _resourced()
    for s in _founded(state, SettlementRole.MINING):
        assert state.hexes[s.coord].relief >= cfg.ore_min_relief_m
        assert s.tier is SettlementTier.VILLAGE


def test_a_lumber_camp_stands_in_the_woods_on_water_that_floats_timber():
    cfg = WorldConfig(**_DEFAULTS)
    state = _resourced()
    for s in _founded(state, SettlementRole.LUMBER):
        hx = state.hexes[s.coord]
        assert hx.land_use is LandUse.WOOD
        assert floatable(hx, cfg) or any(
            floatable(state.hexes[n], cfg) for n in _neighbours(s.coord) if n in state.hexes
        )


def test_ports_stand_on_land():
    state = _resourced()
    for s in state.settlements:
        if "_port_" in s.name:
            assert state.hexes[s.coord].terrain_class not in (
                TerrainClass.OPEN_WATER,
                TerrainClass.INLAND_WATER,
            )
            assert s.role is SettlementRole.PORT


def test_every_knob_at_zero_founds_nothing():
    off = _resourced(port_min_population=0, ore_deposits_per_1000_km2=0.0, lumber_min_score=0.0)
    before = build_world(until="CityPromotionStage", **_KW)
    assert len(off.settlements) == len(before.settlements)


def test_the_road_network_reaches_the_mines_and_the_camps():
    state = build_world(**_KW)
    stranded = {tuple(u["coord"]) for u in state.metadata.get("unreachable_settlements", [])}
    for role in (SettlementRole.MINING, SettlementRole.LUMBER):
        for s in _founded(state, role):
            assert state.hexes[s.coord].road_connections or s.coord in stranded, s.name


# --- the arithmetic, in isolation ---------------------------------------------


def _neighbours(coord):
    from worldgen.core.hex_grid import neighbors

    return neighbors(coord)


def _flat_state(n=12):
    cfg = WorldConfig()
    state = WorldState.empty(1, n, n, cfg.grid_layout)
    state.hexes = {
        (q, r): Hex(coord=(q, r), terrain_class=TerrainClass.LAND)
        for q in range(n)
        for r in range(n)
    }
    return state


def _town(coord, pop, tier=SettlementTier.TOWN):
    return Settlement(
        coord=coord, tier=tier, role=SettlementRole.MARKET, population=pop, name=f"t{coord}"
    )


def test_a_draw_never_takes_more_than_its_share_and_balances():
    cfg = WorldConfig(resource_draw_share=0.25)
    state = _flat_state()
    a, b = _town((0, 0), 1000), _town((2, 0), 400)
    state.settlements = [a, b]
    stage = ResourceStage(cfg, np.random.default_rng(0))

    moved, sent = stage._draw(state, (1, 0), need=10_000)
    assert moved == sum(n for _, n in sent)
    assert a.population + b.population + moved == 1400
    # Both senders stand a hex away on level ground, so each keeps at least three quarters.
    assert a.population >= 750 and b.population >= 300


def test_a_port_that_falls_short_takes_nobody():
    cfg = WorldConfig(port_min_population=500, transship_radius=1)
    state = _flat_state()
    city = _town((5, 5), 20_000, SettlementTier.CITY)
    state.settlements = [city]
    state.metadata["unhandled_quays"] = [[1, 1, 5, 5, 3.0, 499.0]]
    ResourceStage(cfg, np.random.default_rng(0))._ports(state, lambda c: True)
    assert city.population == 20_000
    assert len(state.settlements) == 1


def test_a_port_is_paid_what_the_city_kept_for_it():
    cfg = WorldConfig(port_min_population=100, transship_radius=1)
    state = _flat_state()
    city = _town((5, 5), 20_000, SettlementTier.CITY)
    state.settlements = [city]
    # Two quays side by side are one waterfront, pooled onto the busier.
    state.metadata["unhandled_quays"] = [[1, 1, 5, 5, 3.0, 120.0], [2, 1, 5, 5, 1.0, 80.0]]
    ResourceStage(cfg, np.random.default_rng(0))._ports(state, lambda c: True)
    port = next(s for s in state.settlements if s is not city)
    assert port.coord == (1, 1) and port.role is SettlementRole.PORT
    assert port.population == 200
    assert city.population == 20_000 - 200


@pytest.mark.parametrize("name", ["resource_draw_share"])
def test_resource_draw_share_is_a_fraction(name):
    with pytest.raises(ValueError, match=name):
        WorldConfig(**{name: 1.5})


@pytest.mark.parametrize(("mult", "worked"), [(0.5, False), (1.5, True)])
def test_ore_is_worked_only_where_it_can_get_out(mult, worked):
    """A deposit thirty hexes of level ground from the only outlet: beyond grain's reach at
    half range, within it at one and a half. The deposit is there either way."""
    cfg = WorldConfig(
        ore_haul_range_mult=mult,
        ore_min_relief_m=100.0,
        ore_deposits_per_1000_km2=1e6,
        settlement_min_reachable=1,
        resource_min_population=1,
    )
    state = WorldState.empty(1, 32, 3, cfg.grid_layout)
    state.hexes = {
        (q, r): Hex(coord=(q, r), terrain_class=TerrainClass.LAND, relief=500.0 if q == 30 else 0)
        for q in range(32)
        for r in range(3)
    }
    city = _town((0, 1), 20_000, SettlementTier.CITY)
    state.settlements = [city]
    ResourceStage(cfg, np.random.default_rng(0))._mines(state, lambda c: True)
    assert bool(_founded(state, SettlementRole.MINING)) is worked


def test_ore_is_shipped_to_the_cities_as_freight():
    state = _resourced()
    if not _founded(state, SettlementRole.MINING):
        pytest.skip("no mine on this map")
    kinds = {f[5] for f in state.metadata.get("freight", [])}
    assert "ore" in kinds
