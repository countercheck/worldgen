"""The ground a settlement stands on, fields that look cleared, and ports that stay towns."""

import pytest

from tests.worlds import build_world
from worldgen.core.config import WorldConfig
from worldgen.core.hex import LandCover, LandUse, SettlementRole, SettlementTier, TerrainClass
from worldgen.core.hex_grid import distance, hex_range
from worldgen.stages.land_use import CLEARED_COVER, PLOUGHABLE, rent

_WATER = (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER)
_FOREST = {LandCover.DENSE_FOREST, LandCover.WOODLAND, LandCover.SCRUB}


@pytest.fixture(scope="module", params=["organic", "classic"])
def world(request):
    return build_world(seed=42, width=64, height=64, model=request.param)


def test_no_settlement_but_a_lumber_camp_stands_in_forest(world):
    for s in world.settlements:
        cover = world.hexes[s.coord].land_cover
        if s.role is SettlementRole.LUMBER:
            continue
        assert cover not in _FOREST, f"{s.tier.value} at {s.coord} stands in {cover.value}"


def test_a_city_clears_the_ring_round_it(world):
    cfg = WorldConfig()
    cities = [s for s in world.settlements if s.tier is SettlementTier.CITY]
    assert cities
    for s in cities:
        for c in hex_range(s.coord, cfg.settled_clear_radius_city):
            hx = world.hexes.get(c)
            if hx is not None and hx.terrain_class not in _WATER:
                assert hx.land_cover not in _FOREST


def test_a_lumber_camp_keeps_its_trees():
    ws = build_world(seed=42, width=64, height=64, model="organic")
    camps = [s for s in ws.settlements if s.role is SettlementRole.LUMBER]
    if not camps:
        pytest.skip("no lumber camp on this map")
    assert any(ws.hexes[s.coord].land_cover in _FOREST for s in camps)


# -- clearing ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def organic():
    return build_world(seed=42, width=64, height=64, model="organic", until="LandUseStage")


def test_ploughland_and_pasture_are_open_ground(organic):
    for hx in organic.hexes.values():
        if hx.land_use in (LandUse.ARABLE, LandUse.PASTURE):
            assert hx.land_cover not in CLEARED_COVER


def test_wood_in_reach_is_the_worst_ground_and_pasture_between(organic):
    cfg = WorldConfig(**{k: v for k, v in organic.metadata["config"].items()})
    best: dict = {}
    for hx in organic.hexes.values():
        if hx.territory is not None:
            best[hx.territory] = max(best.get(hx.territory, 0.0), rent(hx, cfg))
    for hx in organic.hexes.values():
        if hx.territory is None or hx.soil not in PLOUGHABLE:
            continue
        share = rent(hx, cfg) / best[hx.territory] if best[hx.territory] else 0.0
        if hx.land_use is LandUse.WOOD:
            assert share < cfg.pasture_margin
        elif hx.land_use is LandUse.PASTURE:
            assert cfg.pasture_margin <= share < cfg.clearing_margin


def test_a_pasture_margin_at_the_clearing_margin_grazes_nothing_extra():
    none = build_world(
        seed=42, width=64, height=64, model="organic", until="LandUseStage", pasture_margin=0.35
    )
    grazed = build_world(seed=42, width=64, height=64, model="organic", until="LandUseStage")
    count = lambda ws, use: sum(h.land_use is use for h in ws.hexes.values())  # noqa: E731
    assert count(grazed, LandUse.PASTURE) > count(none, LandUse.PASTURE)
    assert count(grazed, LandUse.WOOD) < count(none, LandUse.WOOD)


# -- ports ------------------------------------------------------------------------------


def test_a_port_made_a_city_by_trade_has_farmland_and_room():
    """Only a port founded by `ResourceStage` is tested; placeholders survive until naming."""
    cfg = WorldConfig()
    for seed in (42, 7):
        ws = build_world(seed=seed, width=96, height=96, model="organic", until="ResourceStage")
        cities = [s for s in ws.settlements if s.tier is SettlementTier.CITY]
        for s in cities:
            if "_port_" not in s.name:
                continue
            others = [o for o in cities if o is not s]
            assert all(distance(o.coord, s.coord) >= cfg.port_city_min_separation for o in others)
            ploughed = sum(
                1
                for c in hex_range(s.coord, int(cfg.market_day_radius))
                if c in ws.hexes and ws.hexes[c].land_use is LandUse.ARABLE
            )
            assert ploughed >= cfg.port_city_min_farmland


@pytest.mark.parametrize(
    "kwargs",
    [{"pasture_margin": 1.5}, {"port_city_min_separation": -1}, {"settled_clear_radius_city": -1}],
)
def test_bad_settings_are_refused(kwargs):
    with pytest.raises(ValueError):
        WorldConfig(**kwargs)
