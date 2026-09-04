"""Land use, clearing, and the people the land feeds.

Soil says what the ground could take. This is about what is taken from it, and the claim
the whole stage exists to make is that **the margin responds to scarcity**: a settlement on
poor ground is pushed further down the soil ladder than one with a floodplain in reach.
"""

import collections
import statistics

import pytest

from tests.worlds import build_pipeline, build_world
from worldgen.core.config import WorldConfig
from worldgen.core.hex import SOIL_RANK, LandUse, SoilQuality, TerrainClass
from worldgen.stages.habitability import actual_food, potential_food

_DEFAULTS = {"regional_climate": "temperate", "continent_falloff_edges": ("south",)}


def _world(**over):
    return build_world(seed=42, width=96, height=96, model="organic", **{**_DEFAULTS, **over})


@pytest.fixture(scope="module")
def used():
    return _world()


def _land(state):
    return [h for h in state.hexes.values() if h.terrain_class is not TerrainClass.OCEAN]


# --- the margin --------------------------------------------------------------


def test_the_margin_responds_to_scarcity(used):
    """The claim the rule exists to make, and the reason the bar is relative.

    Group markets by the best soil they hold, and ask how far down the ladder each one
    ploughs. A market with a floodplain has a high bar and leaves its hillsides alone; a
    market whose best ground is ordinary arable has a low bar and takes in marginal land.
    An absolute threshold could not produce this — under one it would simply clear less.
    """
    by_seat = collections.defaultdict(list)
    for hx in used.hexes.values():
        if hx.territory is not None and hx.soil is not SoilQuality.UNUSABLE:
            by_seat[hx.territory].append(hx)

    reach = collections.defaultdict(list)
    for held in by_seat.values():
        ploughed = [h for h in held if h.land_use is LandUse.ARABLE]
        if not ploughed:
            continue
        best = max(SOIL_RANK[h.soil] for h in held)
        reach[best].append(min(SOIL_RANK[h.soil] for h in ploughed))

    prime, arable = SOIL_RANK[SoilQuality.PRIME], SOIL_RANK[SoilQuality.ARABLE]
    assert reach[prime] and reach[arable], (
        f"need markets of both kinds to compare; got {dict((k, len(v)) for k, v in reach.items())}"
    )
    rich = statistics.mean(reach[prime])
    poor = statistics.mean(reach[arable])
    assert poor < rich, (
        f"markets with prime land in reach plough down to soil rank {rich:.2f} and markets "
        f"without down to {poor:.2f} — the poorer should reach further down, not less far"
    )


def test_nothing_is_cleared_outside_a_catchment(used):
    """Good ground nobody has reached is wildwood, not ploughland.

    This is what should leave trees standing between the markets rather than a continuous
    sheet of farmland, and it is the visible difference from a disc-based cultivation rule.
    """
    for hx in _land(used):
        if hx.territory is None:
            assert hx.land_use is not LandUse.ARABLE


def test_wood_survives_between_the_markets(used):
    """A map that clears wholesale has its margin set wrong, and so has one that clears
    nothing. Both ends are failures with the same cause, so both are tested here."""
    land = _land(used)
    share = collections.Counter(h.land_use for h in land)
    arable = share[LandUse.ARABLE] / len(land)
    wood = share[LandUse.WOOD] / len(land)
    assert 0.02 < arable < 0.4, f"{arable:.0%} of the map is under the plough"
    assert wood > 0.1, f"only {wood:.0%} of the map is still wooded"


def test_a_wider_margin_clears_less(used):
    """The knob does what it says, monotonically."""
    counts = [
        sum(1 for h in _land(_world(clearing_margin=m)) if h.land_use is LandUse.ARABLE)
        for m in (0.2, 0.45, 0.8)
    ]
    assert counts == sorted(counts, reverse=True), counts


def test_only_ploughable_soil_is_ploughed(used):
    """Grazing ground fails for the two reasons that make it grazing: too steep for the
    share, or too dry for the seed. Neither is fixed by wanting it more."""
    for hx in _land(used):
        if hx.land_use is LandUse.ARABLE:
            assert SOIL_RANK[hx.soil] >= SOIL_RANK[SoilQuality.MARGINAL]
        if hx.soil is SoilQuality.UNUSABLE:
            assert hx.land_use in (LandUse.WASTE, LandUse.WATER)


# --- what it yields ----------------------------------------------------------


def test_working_land_never_yields_more_than_the_ground_allows(used):
    cfg = WorldConfig(**used.metadata["config"])
    for hx in used.hexes.values():
        assert actual_food(hx, cfg) <= potential_food(hx, cfg) + 1e-9


def test_only_the_plough_gets_the_whole_of_it(used):
    """Equality with potential exactly where the ground is cleared, and nowhere else.

    This is the gap that makes clearing worth doing, so it is worth asserting that it is
    actually there rather than trusting the yield table.
    """
    cfg = WorldConfig(**used.metadata["config"])
    for hx in _land(used):
        if hx.soil is SoilQuality.UNUSABLE or potential_food(hx, cfg) == 0.0:
            continue
        full = actual_food(hx, cfg) == pytest.approx(potential_food(hx, cfg))
        assert full == (hx.land_use is LandUse.ARABLE), (
            f"{hx.coord} is {hx.land_use} and yields "
            f"{actual_food(hx, cfg):.2f} of {potential_food(hx, cfg):.2f}"
        )


# --- who lives there ---------------------------------------------------------


def test_the_countryside_holds_the_people_the_markets_do_not(used):
    """Rural population is not a second model, it is the same sum read the other way.

    A market draws `marketable_surplus_fraction` of what its catchment yields, so the rest
    feeds the people who grew it. Defining it that way is what makes the two figures
    reconcile by construction — so the ratio between them should fall out of the constant
    rather than needing its own calibration.
    """
    rural = sum(h.rural_population for h in used.hexes.values())
    town = sum(s.population for s in used.settlements)
    assert rural > town, "more people should live on the land than in the towns"
    assert rural / (rural + town) > 0.8, (
        f"only {100 * rural / (rural + town):.0f}% of the map's people are rural; "
        "a pre-industrial society is not that urban"
    )


def test_rural_density_is_pre_industrial(used):
    """England in 1300 carried about 35 people per km2, and a hex is 1 km2.

    A figure that came out at 90 would mean `people_per_food` had been calibrated on the
    settlements alone and left to say something absurd about the countryside — which is
    exactly what it did before this stage existed.
    """
    land = _land(used)
    total = sum(h.rural_population for h in land) + sum(s.population for s in used.settlements)
    density = total / len(land)
    assert 15.0 < density < 60.0, f"{density:.1f} people per km2"


def test_nobody_lives_on_ground_that_feeds_nobody(used):
    for hx in _land(used):
        if hx.land_use is LandUse.WASTE:
            assert hx.rural_population == 0.0


# --- structure ---------------------------------------------------------------


def test_cultivated_still_means_under_the_plough(used):
    """Kept as a derived boolean: the classic village stages and the JSON schema read it."""
    for hx in used.hexes.values():
        assert hx.cultivated == (hx.land_use is LandUse.ARABLE)


def test_every_hex_gets_a_use(used):
    assert all(h.land_use is not None for h in used.hexes.values())


def test_same_seed_same_land_use():
    a = _world()
    b = build_pipeline(seed=42, width=96, height=96, model="organic", **_DEFAULTS).run()
    assert {c: h.land_use for c, h in a.hexes.items()} == {
        c: h.land_use for c, h in b.hexes.items()
    }


def test_nobody_lives_on_the_water(used):
    """The fishermen live on the shore that works the water, not on the sea itself.

    The food model gives open water a non-zero yield so the fishery can feed a market;
    read naively as residents, it put a fifth of the map's people on the open sea, and
    every density figure quietly counted them.
    """
    water = (TerrainClass.OCEAN, TerrainClass.LAKE)
    afloat = [
        c for c, h in used.hexes.items() if h.terrain_class in water and h.rural_population > 0
    ]
    assert not afloat, f"{len(afloat)} water hexes house people, first at {afloat[:3]}"
    assert any(h.rural_population > 0 for h in used.hexes.values())
