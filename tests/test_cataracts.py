"""Cataracts: big rivers falling too fast for boats — barriers, portages and mill sites."""

import numpy as np

from tests.worlds import build_world
from worldgen.core.config import WorldConfig
from worldgen.core.hex import Hex, TerrainClass
from worldgen.stages.cataracts import CataractStage
from worldgen.stages.crossings import channel_drop_m
from worldgen.stages.habitability import site_bonus
from worldgen.stages.haulage import bulk_routes, carries_a_barge, navigable

# The 96x96 temperate world the city tests use: the 64x64 default has no river big and
# steep enough to make a cataract, and a test asserting over no cataracts asserts nothing.
_KW = dict(
    seed=42,
    width=96,
    height=96,
    model="organic",
    regional_climate="temperate",
    continent_falloff_edges=("south",),
    city_min_draw=22.0,
)


def _world(**over):
    return build_world(until="CataractStage", **{**_KW, **over})


def test_a_cataract_is_a_barge_river_falling_fast():
    state = _world()
    cfg = WorldConfig(**state.metadata["config"])
    for hx in state.hexes.values():
        if "cataract" in hx.tags:
            assert "river" in hx.tags and carries_a_barge(hx, cfg)
            assert channel_drop_m(hx, state.hexes, cfg) >= cfg.cataract_min_drop_m


def test_no_boat_passes_a_cataract():
    state = _world()
    cfg = WorldConfig(**state.metadata["config"])
    falls = [hx for hx in state.hexes.values() if "cataract" in hx.tags]
    assert falls, "the fixture map has no cataract, so the tests above assert over nothing"
    assert all(not navigable(hx, cfg) for hx in falls)


def test_zero_turns_cataracts_off():
    state = _world(cataract_min_drop_m=0.0)
    assert not any("cataract" in hx.tags for hx in state.hexes.values())


def _river(drops):
    """A barge-sized river along r = 0 falling *drops[i]* metres into hex i + 1."""
    hexes = {}
    height = sum(drops) + 10.0
    for q in range(len(drops) + 1):
        hx = Hex(coord=(q, 0), terrain_class=TerrainClass.LAND, elevation=height)
        hx.tags.add("river")
        hx.catchment_km2 = 1e6
        hexes[(q, 0)] = hx
        if q < len(drops):
            height -= drops[q]
    return hexes


def test_the_stage_marks_only_the_steep_reach():
    from worldgen.core.world_state import WorldState

    cfg = WorldConfig(cataract_min_drop_m=20.0)
    state = WorldState.empty(1, 8, 1, cfg.grid_layout)
    state.hexes = _river([1.0, 1.0, 40.0, 1.0, 1.0])
    CataractStage(cfg, np.random.default_rng(0)).run(state)
    assert [q for (q, _), hx in state.hexes.items() if "cataract" in hx.tags] == [2]


def test_a_cataract_forces_a_portage():
    """Hauling past the falls costs a landing and a loading more than hauling past a slack
    reach of the same river, which is what makes the portage a quay. River landings, since
    both sides of the falls are river."""
    cfg = WorldConfig()
    slack, falls = _river([1.0] * 6), _river([1.0] * 6)
    falls[(3, 0)].tags.add("cataract")
    cost_slack, _ = bulk_routes(slack, [(6, 0)], cfg)
    cost_falls, _ = bulk_routes(falls, [(6, 0)], cfg)
    assert cost_falls[(0, 0)] >= cost_slack[(0, 0)] + 2 * cfg.haulage_river_transship_cost


def test_a_site_beside_the_falls_has_water_power():
    cfg = WorldConfig()
    hexes = _river([1.0] * 4)
    before = site_bonus((1, 0), hexes[(1, 0)], hexes, cfg)
    hexes[(2, 0)].tags.add("cataract")
    after = site_bonus((1, 0), hexes[(1, 0)], hexes, cfg)
    assert after == before + cfg.habitability_mill_bonus


def test_a_river_landing_is_cheaper_than_a_harbour():
    """A barge ties up at a bank; a sea-going ship wants a harbour."""
    from worldgen.stages.haulage import make_bulk_cost

    cfg = WorldConfig()
    land = Hex(coord=(0, 0), terrain_class=TerrainClass.LAND)
    river = _river([1.0])[(1, 0)]
    sea = Hex(coord=(1, 0), terrain_class=TerrainClass.OPEN_WATER)
    _, edge = make_bulk_cost({}, cfg)
    assert edge(land, river) == cfg.haulage_river_transship_cost
    assert edge(land, sea) == cfg.haulage_transship_cost
    assert cfg.haulage_river_transship_cost < cfg.haulage_transship_cost
