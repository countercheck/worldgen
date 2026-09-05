"""City promotion: the claim that a port is fed from further away than a market.

The whole tier rests on one asymmetry — bulk goods go by water at a fifteenth the cost —
so these test that the asymmetry is what decides, rather than that some number came out.
"""

import pytest

from tests.worlds import build_pipeline, build_world
from worldgen.core.config import WorldConfig
from worldgen.core.hex import SettlementTier, TerrainClass
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


def test_cities_sharing_ground_are_joined_by_road(city_world):
    """An army marches; it does not take ship between two cities on one landmass.

    A corollary of the network guarantee rather than a rule of its own — `_join_by_land`
    puts one road network on each landmass — but cities are the tier this matters most for,
    and a corollary nobody asserts is a corollary that can quietly stop holding. Land only:
    `sea_edges` are deliberately kept apart from `road_edges` so that "is there a land
    route" is a question the world can answer, and this asks it.
    """
    cities = [c.coord for c in _split(city_world)[0]]
    if len(cities) < 2:
        pytest.skip(f"{len(cities)} cities on this map; nothing to join")

    water = (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER)
    dry = {c for c, hx in city_world.hexes.items() if hx.terrain_class not in water}

    adj: dict = {}
    for a, b in city_world.road_edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    def reachable_from(start):
        seen, stack = {start}, [start]
        while stack:
            c = stack.pop()
            for n in adj.get(c, ()):
                if n not in seen:
                    seen.add(n)
                    stack.append(n)
        return seen

    def shares_ground(a, b):
        seen, stack = {a}, [a]
        while stack:
            c = stack.pop()
            if c == b:
                return True
            for n in neighbors(c):
                if n in dry and n not in seen:
                    seen.add(n)
                    stack.append(n)
        return False

    for i, a in enumerate(cities):
        by_road = reachable_from(a)
        for b in cities[i + 1 :]:
            if shares_ground(a, b):
                assert b in by_road, (
                    f"cities at {a} and {b} stand on the same landmass but no road joins "
                    "them — they can only reach each other by sea"
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


def _before_and_after():
    """The world either side of promotion, and nothing after it.

    Stopped at `CityPromotionStage` rather than run to the end, because `ChokepointStage`
    downstream founds a village tier of its own — real settlements that this stage did not
    make and must not be blamed for. The "before" world stops at `LandUseStage`, which is
    where markets are founded and sized.
    """
    kw = dict(seed=42, width=96, height=96, model="organic", **_CITY_DEFAULTS)
    return (
        build_world(until="LandUseStage", **kw),
        build_world(until="CityPromotionStage", **kw),
    )


def test_promotion_founds_nothing():
    """The stage changes tiers and sizes; it must not add or remove a settlement."""
    before, after = _before_and_after()
    assert len(after.settlements) == len(before.settlements)
    assert sorted(s.coord for s in after.settlements) == sorted(s.coord for s in before.settlements)


def test_a_city_takes_its_surplus_from_the_markets_it_reaches():
    """Conserved, not conjured. A city is the same countryside feeding a different place.

    Exactly conserved, to the rounding: promotion is a transfer applied as a delta on the
    populations the markets already have, so the books balance to within half a person
    per settlement touched. This used to allow 5% slack, which was wide enough to hide
    both a promoted seat keeping surplus another city had taken and the founding jitter
    being silently stripped from the whole tier.
    """
    before, after = _before_and_after()
    if not _split(after)[0]:
        return
    was = sum(s.population for s in before.settlements)
    now = sum(s.population for s in after.settlements)
    assert abs(now - was) <= len(after.settlements), (
        f"population went from {was:,} to {now:,} — promotion is creating surplus, not moving it"
    )


def test_promotion_leaves_an_untouched_market_byte_identical():
    """A market no city drew on keeps the exact population it was founded with.

    Founding sizes carry a deliberate jitter; recomputing every market from the
    un-jittered draw erased it map-wide, which made town sizes an invertible function of
    catchment again — the exact mechanical look the jitter exists to break.
    """
    before, after = _before_and_after()
    cities, _ = _split(after)
    if not cities:
        return
    pop_before = {s.coord: s.population for s in before.settlements}
    unchanged = [s for s in after.settlements if s.population == pop_before[s.coord]]
    assert unchanged, (
        "every single settlement's population changed across promotion — the stage is "
        "rewriting the whole tier rather than moving surplus between the places it touched"
    )


def test_same_seed_same_cities():
    a = _world()
    b = build_pipeline(seed=42, width=96, height=96, model="organic", **_CITY_DEFAULTS).run()
    assert sorted((s.coord, s.tier.value, s.population) for s in a.settlements) == sorted(
        (s.coord, s.tier.value, s.population) for s in b.settlements
    )


# --- the arithmetic, in isolation ---------------------------------------------


def test_bulk_reach_prices_the_haul_toward_the_seat():
    """The Dijkstra expands outward; the cargo travels inward, and only climb is charged.

    A seat in the valley is provisioned downhill — the wagons descend, so its reach runs
    far up the slope. A seat on the hill is provisioned uphill, and the same budget buys
    less. Getting the edge direction wrong is silent: nothing crashes, the two cases
    simply swap, and every market the country rises toward has its draw inflated.
    """
    from worldgen.core.hex import Hex
    from worldgen.stages.cities import CityPromotionStage

    cfg = WorldConfig()
    hexes = {(q, 0): Hex(coord=(q, 0), elevation=q * 60.0) for q in range(10)}

    to_valley = CityPromotionStage._bulk_reach(hexes, (0, 0), cfg)
    to_hilltop = CityPromotionStage._bulk_reach(hexes, (9, 0), cfg)

    assert to_hilltop[(0, 0)] > to_valley[(9, 0)], (
        "hauling 540 m uphill must cost more than hauling the same road down"
    )


def test_resize_moves_exactly_what_was_taken():
    """Conserved to the round, and untouched markets come out byte-identical.

    Two regressions live here. Recomputing every market from the un-jittered draw
    silently stripped the founding jitter off the whole tier; and a promoted seat that
    had itself been drawn on by an earlier city kept surplus already counted elsewhere,
    so promotion conjured people. The delta form fixes both at once.
    """
    import numpy as np

    from worldgen.core.hex import Settlement, SettlementRole
    from worldgen.core.world_state import WorldState
    from worldgen.stages.cities import CityPromotionStage

    cfg = WorldConfig()

    def market(coord, pop):
        return Settlement(
            coord=coord,
            tier=SettlementTier.TOWN,
            role=SettlementRole.MARKET,
            population=pop,
            name=f"market_{coord}",
        )

    a, b, c = market((1, 1), 1013), market((5, 5), 977), market((9, 9), 1200)
    state = WorldState.empty(1, 12, 12, cfg.grid_layout)

    # A is promoted on B's surplus; B is then promoted on C's — so B both sends and
    # receives, which is exactly the case the old recomputation double-counted.
    absorbed = {(1, 1): {(5, 5): 2.0}, (5, 5): {(9, 9): 1.0}}
    stage = CityPromotionStage(cfg, np.random.default_rng(0))
    stage._resize(state, [a, b, c], absorbed, [(1, 1), (5, 5)], cfg)

    ppf = cfg.people_per_food
    assert a.population == 1013 + round(2.0 * ppf)
    assert b.population == 977 + round((1.0 - 2.0) * ppf), "B must be charged what A took"
    assert c.population == 1200 - round(1.0 * ppf)
    assert a.population + b.population + c.population == 1013 + 977 + 1200, (
        "promotion moved people it did not have"
    )
    assert a.tier is SettlementTier.CITY and b.tier is SettlementTier.CITY
    assert c.tier is SettlementTier.TOWN
