"""Cataracts: big rivers falling too fast for boats — barriers, portages and mill sites."""

import numpy as np

from tests.worlds import build_world
from worldgen.core.config import WorldConfig
from worldgen.core.hex import Hex, TerrainClass
from worldgen.stages.cataracts import CataractStage
from worldgen.stages.habitability import site_bonus
from worldgen.stages.haulage import bulk_routes, catchment_carries_a_barge, navigable
from worldgen.stages.riverside import side_gradients

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
    gradient = side_gradients(state)
    falls = [s for s, rs in state.river_sides.items() if "cataract" in rs.tags]
    assert falls
    for side in falls:
        assert catchment_carries_a_barge(state.river_sides[side].catchment_km2, cfg)
        assert gradient[side] >= cfg.cataract_min_drop_m


def test_no_boat_passes_a_cataract():
    state = _world()
    cfg = WorldConfig(**state.metadata["config"])
    from worldgen.core.hex_grid import side_hexes
    from worldgen.stages.riverside import river_index

    rivers = river_index(state, cfg)
    falls = [s for s, rs in state.river_sides.items() if "cataract" in rs.tags]
    assert falls, "the fixture map has no cataract, so the tests above assert over nothing"
    assert all(not navigable(state.hexes[h], cfg, rivers) for s in falls for h in side_hexes(s))


def test_zero_turns_cataracts_off():
    state = _world(cataract_min_drop_m=0.0)
    assert not any("cataract" in hx.tags for hx in state.hexes.values())


def _river(drops):
    """A barge-sized river along r = 0 falling *drops[i]* metres into hex i + 1."""
    hexes = {}
    height = sum(drops) + 10.0
    for q in range(len(drops) + 1):
        hx = Hex(coord=(q, 0), terrain_class=TerrainClass.LAND, elevation=height)
        hx.catchment_km2 = 1e6
        hexes[(q, 0)] = hx
        if q < len(drops):
            height -= drops[q]
    return hexes


def test_the_stage_marks_only_the_steep_reach():
    """A fall of 40 m over one side marks it and the sides either side — the gradient is
    read over three sides — and nothing further off."""
    from tests.worlds import lay_river
    from worldgen.core.world_state import WorldState

    cfg = WorldConfig(cataract_min_drop_m=20.0)
    state = WorldState.empty(1, 10, 3, cfg.grid_layout)
    river = lay_river(state, [(q, 1) for q in range(9)])
    sides = river.sides()
    steep = len(sides) // 2
    for i, side in enumerate(sides):
        state.river_sides[side].catchment_km2 = 1e6
        state.river_sides[side].drop_m = 40.0 if i == steep else 1.0
    CataractStage(cfg, np.random.default_rng(0)).run(state)
    marked = [i for i, s in enumerate(sides) if "cataract" in state.river_sides[s].tags]
    assert marked == [steep - 1, steep, steep + 1]


def test_a_cataract_forces_a_portage():
    """Hauling past the falls costs a landing and a loading more than hauling past a slack
    reach of the same river, which is what makes the portage a quay. River landings, since
    both sides of the falls are river."""
    from tests.worlds import river_to_sea
    from worldgen.stages.riverside import river_index

    cfg = WorldConfig()
    slack, _ = river_to_sea()
    falls, _ = river_to_sea(cataract_at=2)
    seat, start = (3, 1), (1, 1)
    cost_slack, _ = bulk_routes(slack.hexes, [seat], cfg, rivers=river_index(slack, cfg))
    cost_falls, _ = bulk_routes(falls.hexes, [seat], cfg, rivers=river_index(falls, cfg))
    assert cost_falls[start] >= cost_slack[start] + 2 * cfg.haulage_river_transship_cost


def test_a_site_beside_the_falls_has_water_power():
    from tests.worlds import river_to_sea
    from worldgen.core.hex_grid import side_hexes
    from worldgen.stages.riverside import river_index

    cfg = WorldConfig()
    slack, river = river_to_sea()
    falls, _ = river_to_sea(cataract_at=2)
    bank = side_hexes(river.sides()[2])[0]
    before = site_bonus(bank, slack.hexes[bank], slack.hexes, cfg, river_index(slack, cfg))
    after = site_bonus(bank, falls.hexes[bank], falls.hexes, cfg, river_index(falls, cfg))
    # The falls stop barges at the bank, but it is still a step from the navigable reach
    # either side, so it keeps its harbour and gains the mill.
    assert after == before + cfg.habitability_mill_bonus


def test_a_river_landing_is_cheaper_than_a_harbour():
    """A barge ties up at a bank; a sea-going ship wants a harbour."""
    from tests.worlds import river_to_sea
    from worldgen.stages.haulage import make_bulk_cost
    from worldgen.stages.riverside import river_index

    cfg = WorldConfig()
    ws, _ = river_to_sea()
    _, edge = make_bulk_cost(ws.hexes, cfg, river_index(ws, cfg))
    field, bank = ws.hexes[(1, 2)], ws.hexes[(1, 1)]
    shore, sea = ws.hexes[(5, 0)], ws.hexes[(5, 1)]
    assert edge(field, bank) == cfg.haulage_river_transship_cost
    assert edge(shore, sea) == cfg.haulage_transship_cost
    assert cfg.haulage_river_transship_cost < cfg.haulage_transship_cost
