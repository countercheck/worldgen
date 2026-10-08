"""Long-haul trade down the rivers to the coast (tech-debt #125).

The claims: the trade goes downstream, along the water, and ends at the coast; it changes
nothing when it is off; more of it never means less passing the portages; and it wears no
road, because it went by boat.
"""

import copy
import dataclasses

import numpy as np
import pytest

from tests.worlds import build_pipeline, build_world
from worldgen.core.config import WorldConfig
from worldgen.core.hex_grid import neighbors
from worldgen.stages.cities import CityPromotionStage
from worldgen.stages.interurban_roads import InterurbanRoadStage
from worldgen.stages.river_trade import coastal, downstream_rank
from worldgen.stages.riverside import river_index

# The 96x96 temperate coast `test_cities` uses. Seed 42 shares its memoised world; seed 11
# is the one whose rivers fall over cataracts on their way to a coastal town. Seed 42's
# great river leaves the map, so none of its river trade passes a portage.
_DEFAULTS = {
    "regional_climate": "temperate",
    "continent_falloff_edges": ("south",),
    "city_min_draw": 22.0,
}
_SIZE = {"width": 96, "height": 96, "model": "organic"}


@pytest.fixture(scope="module")
def before():
    """The world as `CityPromotionStage` leaves it with the river trade off."""
    return build_pipeline(
        until="CityPromotionStage", seed=11, **_SIZE, **_DEFAULTS, river_trade_share=0.0
    ).run()


def _traded(state, share):
    """*state* with the river trade run over it at *share*, and the config it ran with."""
    cfg = dataclasses.replace(WorldConfig(**state.metadata["config"]), river_trade_share=share)
    out = copy.deepcopy(state)
    CityPromotionStage(cfg, np.random.default_rng(0))._river_trade(out, cfg)
    return out, cfg


def _through_portages(state, cfg) -> float:
    rivers = river_index(state, cfg)
    return sum(
        people
        for *_, people, path in state.metadata.get("river_trade", [])
        if any(tuple(h) in rivers.portage for h in path)
    )


def test_share_zero_changes_nothing(before):
    """Off is off: the world is identical to one built without the mechanism at all."""
    off = before
    real = CityPromotionStage._river_trade
    CityPromotionStage._river_trade = lambda self, state, cfg: None
    try:
        without = build_pipeline(
            until="CityPromotionStage", seed=11, **_SIZE, **_DEFAULTS, river_trade_share=0.0
        ).run()
    finally:
        CityPromotionStage._river_trade = real

    assert "river_trade" not in off.metadata
    assert off.to_dict() == without.to_dict()


def test_river_trade_runs_downstream_along_the_water_to_the_coast(before):
    state, cfg = _traded(before, 0.2)
    flows = state.metadata.get("river_trade", [])
    assert flows, "no inland river town traded down to the coast — the test did not bite"
    rivers = river_index(state, cfg)
    rank = downstream_rank(state, rivers)
    hexes = state.hexes
    seats = {s.coord for s in state.settlements}

    for oq, or_, dq, dr, people, path in flows:
        path = [tuple(h) for h in path]
        origin, dest = (oq, or_), (dq, dr)
        assert people > 0.0
        assert path[0] == origin and path[-1] == dest
        assert origin in seats and dest in seats
        assert coastal(dest, hexes), f"{origin} traded to {dest}, which is not on the coast"
        assert not coastal(origin, hexes), f"coastal {origin} shipped river trade"
        assert any(n in rivers.reaches for n in (origin, *neighbors(origin)))
        for a, b in zip(path, path[1:], strict=False):
            assert b in neighbors(a), f"{a} -> {b} is not a step"
        # Between loading at the origin and landing at the coast, the cargo is on the
        # river, on a lake or the sea, or walking round a cataract — and never climbing.
        afloat = path[1:-1] if path[0] not in rank else path[:-1]
        for a, b in zip(afloat, afloat[1:], strict=False):
            assert a in rank and b in rank, f"{origin}'s cargo went ashore at {a} -> {b}"
            assert rank[b] >= rank[a], f"{origin}'s cargo went upstream at {a} -> {b}"
            if rivers.afloat(hexes[a]) and rivers.afloat(hexes[b]):
                assert rivers.joined(hexes[a], hexes[b]), f"{a} -> {b} is not one water"

    river_freight = sorted(f[:5] for f in state.metadata["freight"] if f[5] == "river")
    assert river_freight == sorted(f[:5] for f in flows)


def test_more_river_trade_never_means_less_through_the_portages(before):
    through = [_through_portages(*_traded(before, share)) for share in (0.0, 0.1, 0.2, 0.4)]
    assert through[0] == 0.0
    assert through[2] > 0.0, "no river trade passed a portage on this world"
    assert through == sorted(through), through


def test_river_trade_wears_no_road():
    """It went by boat. Dropping every river flow leaves the road network as it was."""
    state = build_world(until="ResourceStage", seed=42, **_SIZE, **_DEFAULTS)
    assert any(f[5] == "river" for f in state.metadata.get("freight", []))
    cfg = WorldConfig(**state.metadata["config"])

    def roads(freight):
        world = copy.deepcopy(state)
        world.metadata["freight"] = freight
        return InterurbanRoadStage(cfg, np.random.default_rng(7)).run(world).to_dict()

    with_river = roads(state.metadata["freight"])
    without = roads([f for f in state.metadata["freight"] if f[5] != "river"])
    without["metadata"]["freight"] = with_river["metadata"]["freight"]
    assert with_river == without


@pytest.mark.parametrize("name", ["river_trade_share", "river_trade_range_mult"])
def test_river_trade_settings_cannot_be_negative(name):
    with pytest.raises(ValueError, match=name):
        WorldConfig(**{name: -0.1})
