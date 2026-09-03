"""City promotion: the claim that a port is fed from further away than a market.

The whole tier rests on one asymmetry — bulk goods go by water at a fifteenth the cost —
so these test that the asymmetry is what decides, rather than that some number came out.
"""

import pytest

from tests.worlds import build_pipeline, build_world
from worldgen.core.config import WorldConfig
from worldgen.core.hex import SettlementTier
from worldgen.core.hex_grid import neighbors
from worldgen.stages.haulage import navigable

# 96x96 with a threshold to match, rather than the 128x128 and 40.0 that ship.
# `city_min_draw` is deliberately not scale-free — a city's reach runs along water and a
# bigger map has more coastline inside it, so the best draw goes 39.2 at 96x96 against 97.7
# at 128x128 (see the note on the setting). Testing at production's size costs three and a
# half minutes of pre-commit; 96x96 at 22.0 gives 4 cities against 36 towns, which is the
# same shape and exercises the same arithmetic.
_CITY_DEFAULTS = {
    "regional_climate": "temperate",
    "continent_falloff_edges": ("south",),
    "city_min_draw": 22.0,
}


def _world(**over):
    """Memoised, so the several tests wanting the same world pay for it once."""
    return build_world(seed=42, width=96, height=96, model="organic", **{**_CITY_DEFAULTS, **over})


def _split(state):
    cities = [s for s in state.settlements if s.tier is SettlementTier.CITY]
    towns = [s for s in state.settlements if s.tier is SettlementTier.TOWN]
    return cities, towns


def _on_water(state, cfg, coord):
    if navigable(state.hexes[coord], cfg):
        return True
    return any(navigable(state.hexes[n], cfg) for n in neighbors(coord) if n in state.hexes)


@pytest.fixture(scope="module")
def city_world():
    return _world()


def test_a_temperate_coast_grows_cities(city_world):
    cities, towns = _split(city_world)
    assert cities, "a well-watered temperate coast supported no city at all"
    assert towns, "every market was promoted — the threshold is doing nothing"


def test_every_city_stands_on_navigable_water(city_world):
    """The model's central claim, and the one that fails loudest if it breaks.

    A city is a place fed from beyond a day's reach, and before the railway the only way to
    move bulk that far was by water. If a city ever appears inland it means the reach is
    measuring something else — when this was first written it measured nothing but how
    central a market was on land, because the bulk Dijkstra had been given
    `make_travel_cost`, which makes water impassable on purpose.
    """
    cfg = WorldConfig(**city_world.metadata["config"])
    cities, _ = _split(city_world)
    inland = [c.coord for c in cities if not _on_water(city_world, cfg, c.coord)]
    assert not inland, (
        f"{len(inland)} cities stand away from navigable water, e.g. {inland[0]} — "
        "bulk haulage is not what promoted them"
    )


def test_a_city_is_far_larger_than_the_markets_around_it(city_world):
    """Fertility sets how many markets; water sets how big one place can get.

    Markets come out at a uniform size whatever the country is like, because each is bounded
    by a day's cart. A city is not a large market, it is a place fed from further, and the
    gap should be a different order of magnitude rather than a percentage.
    """
    cities, towns = _split(city_world)
    if not cities or not towns:
        return
    biggest_city = max(c.population for c in cities)
    median_town = sorted(t.population for t in towns)[len(towns) // 2]
    assert biggest_city > median_town * 5, (
        f"largest city {biggest_city} against a median town of {median_town} — "
        "promotion is not producing a hierarchy"
    )


def test_arid_land_grows_no_cities_even_with_a_coast():
    """A harbour with nothing to ship is not a city.

    The acceptance test for the whole redesign. `classic` would place its configured six
    cities here regardless; this model asks what can actually reach a place, and on thin
    land the answer is not enough — coastline or no coastline.
    """
    coastal = _world(regional_climate="arid")
    landlocked = _world(regional_climate="arid", continent_falloff=False)
    for label, state in (("arid coast", coastal), ("arid landlocked", landlocked)):
        cities, _ = _split(state)
        assert not cities, f"{label} grew {len(cities)} cities on land that cannot feed one"


def test_raising_the_threshold_promotes_fewer():
    low = _world(city_min_draw=14.0)
    high = _world(city_min_draw=60.0)
    assert len(_split(low)[0]) >= len(_split(high)[0])


def test_promotion_founds_nothing():
    """The stage changes tiers and sizes; it must not add or remove a settlement."""
    before = build_world(
        seed=42, width=96, height=96, model="organic", until="MarketStage", **_CITY_DEFAULTS
    )
    after = _world()
    assert len(after.settlements) == len(before.settlements)
    assert sorted(s.coord for s in after.settlements) == sorted(s.coord for s in before.settlements)


def test_a_city_takes_its_surplus_from_the_markets_it_reaches():
    """Conserved, not conjured. A city is the same countryside feeding a different place.

    Total population may move a little — a market's own draw is jittered — but a city that
    added tens of thousands of people without taking them from anywhere would mean the
    surplus was being counted twice.
    """
    before = build_world(
        seed=42, width=96, height=96, model="organic", until="MarketStage", **_CITY_DEFAULTS
    )
    after = _world()
    if not _split(after)[0]:
        return
    was = sum(s.population for s in before.settlements)
    now = sum(s.population for s in after.settlements)
    assert now == pytest.approx(was, rel=0.05), (
        f"population went from {was:,} to {now:,} — promotion is creating surplus, not moving it"
    )


def test_same_seed_same_cities():
    a = _world()
    b = build_pipeline(seed=42, width=96, height=96, model="organic", **_CITY_DEFAULTS).run()
    assert sorted((s.coord, s.tier.value, s.population) for s in a.settlements) == sorted(
        (s.coord, s.tier.value, s.population) for s in b.settlements
    )
