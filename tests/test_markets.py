"""Market planting and catchment allocation.

The claims worth testing are structural rather than numeric: that counts follow the land
rather than a config target, that every market is a plausible site with a real catchment,
and that the same seed gives the same markets.
"""

import pytest

from tests.worlds import build_pipeline
from worldgen.core.config import WorldConfig
from worldgen.core.hex import Biome, SettlementTier, TerrainClass
from worldgen.core.hex_grid import distance
from worldgen.stages.markets import MarketStage, depletion_kernel

_WATER = (TerrainClass.OCEAN, TerrainClass.LAKE)


def _market_world(seed=42, width=64, height=64, **overrides):
    p = build_pipeline(
        seed=seed, width=width, height=height, until="HabitabilityStage", **overrides
    )
    p.add_stage(MarketStage)
    return p.run()


@pytest.fixture(scope="module")
def markets():
    return _market_world()


# --- the depletion kernel ----------------------------------------------------


def test_kernel_takes_everything_at_the_seat_and_less_further_out():
    kernel = depletion_kernel(10.0, 4.0)
    shares = [share for _, share in kernel]
    assert shares[0] == 1.0
    assert shares == sorted(shares, reverse=True)
    assert all(0.0 < s <= 1.0 for s in shares)


def test_kernel_covers_the_whole_radius():
    assert [d for d, _ in depletion_kernel(10.0, 4.0)] == list(range(11))


def test_slower_decay_leaves_less_behind_further_out():
    """The decay is what grades spacing with the land rather than fixing it at a radius."""
    quick = dict(depletion_kernel(10.0, 1.0))
    slow = dict(depletion_kernel(10.0, 8.0))
    assert slow[8] > quick[8]


# --- where markets land ------------------------------------------------------


def test_markets_are_planted(markets):
    assert markets.settlements, "no markets on a 64x64 temperate map"


def test_markets_avoid_unsettleable_ground(markets):
    """The same exclusions habitability scores to zero."""
    for s in markets.settlements:
        hx = markets.hexes[s.coord]
        assert hx.terrain_class not in _WATER, f"market in the water at {s.coord}"
        assert hx.terrain_class != TerrainClass.MOUNTAIN, f"market on a peak at {s.coord}"
        assert hx.biome is not Biome.WETLAND, f"market in a bog at {s.coord}"


def test_markets_respect_the_suppression_disc(markets):
    cfg = markets.metadata["config"]
    sep = cfg["market_min_separation"]
    coords = [s.coord for s in markets.settlements]
    for i, a in enumerate(coords):
        for b in coords[i + 1 :]:
            assert distance(a, b) > sep, f"markets {a} and {b} closer than {sep}"


def test_the_hex_points_back_at_its_market(markets):
    for s in markets.settlements:
        assert markets.hexes[s.coord].settlement is s


def test_every_market_is_a_town_for_now(markets):
    """Cities are a promotion, made in a later stage; nothing is born one."""
    assert {s.tier for s in markets.settlements} == {SettlementTier.TOWN}


# --- catchments --------------------------------------------------------------


def test_every_market_owns_at_least_its_own_hex(markets):
    for s in markets.settlements:
        hx = markets.hexes[s.coord]
        assert hx.territory == s.coord
        assert hx.territory_cost == 0.0


def test_territory_only_ever_names_a_real_market(markets):
    seats = {s.coord for s in markets.settlements}
    for coord, hx in markets.hexes.items():
        if hx.territory is not None:
            assert hx.territory in seats, f"{coord} claimed by a non-market {hx.territory}"


def test_catchments_are_bounded_by_the_day_radius(markets):
    budget = markets.metadata["config"]["market_day_radius"]
    for hx in markets.hexes.values():
        if hx.territory is not None and hx.terrain_class not in _WATER:
            assert hx.territory_cost < budget


def test_catchments_reach_beyond_the_seat(markets):
    """A catchment of one hex would mean the day radius is doing nothing."""
    claimed = [h for h in markets.hexes.values() if h.territory is not None]
    assert len(claimed) > 5 * len(markets.settlements)


def test_high_ground_is_left_unclaimed(markets):
    """Nobody's market catchment includes the peaks; that is what bounds them."""
    unclaimed_mountain = [
        h
        for h in markets.hexes.values()
        if h.terrain_class == TerrainClass.MOUNTAIN and h.territory is None
    ]
    assert unclaimed_mountain, "every mountain got claimed — the budget is not binding"


def test_a_coastal_market_gets_its_fishery(markets):
    """Water is never traversed, but a claimed shore hex donates the water it touches."""
    claimed_water = [
        h for h in markets.hexes.values() if h.terrain_class in _WATER and h.territory is not None
    ]
    assert claimed_water, "no market gained any water; the fishery rim did not run"


# --- population --------------------------------------------------------------


def test_population_is_positive(markets):
    for s in markets.settlements:
        assert s.population > 0


def test_bigger_catchments_make_bigger_markets(markets):
    """Population is what the catchment can send, not a draw from a range.

    Comparing the largest and smallest market: if size were random these would be
    uncorrelated with the ground each one holds.
    """
    by_pop = sorted(markets.settlements, key=lambda s: s.population)
    smallest, largest = by_pop[0], by_pop[-1]

    def owned(seat):
        return sum(1 for h in markets.hexes.values() if h.territory == seat)

    assert owned(largest.coord) > owned(smallest.coord)


# --- counts follow the land --------------------------------------------------


def test_market_count_tracks_the_surplus_on_offer():
    """The acceptance test for the whole model: counts follow the land.

    Five climates over one seed and one map, ordered by the food their land actually
    produces — the market count must rise with it.  The classic model would place exactly
    `target_town_count` on all five.

    Deliberately phrased as "monotone in food" rather than naming which climate is
    richest.  That ranking is a property of `food_value`, not of this stage, and the
    surface currently rates an arid map above a temperate one: temperate land sits at
    median moisture 0.68, past the wet edge of the agricultural tent at
    `biome_wet_moist`, and its wetness pushes cover into dense forest, which scores
    marginal.  Whether that is right is a question about the food surface; what this
    stage owes is that it follows whatever surface it is given.
    """
    from worldgen.stages.habitability import food_value

    rows = []
    for climate in ("boreal", "tropical", "temperate", "arid", "mediterranean"):
        state = _market_world(regional_climate=climate)
        cfg = state.metadata["config"]
        total = sum(
            food_value(h, WorldConfig(**cfg), cfg["biome_dry_moist"], cfg["biome_wet_moist"])
            for h in state.hexes.values()
            if h.terrain_class not in _WATER
        )
        rows.append((total, len(state.settlements), climate))

    rows.sort()
    counts = [n for _, n, _ in rows]
    assert counts == sorted(counts), (
        "market count does not rise with available surplus: "
        + ", ".join(f"{c} food={t:.0f} markets={n}" for t, n, c in rows)
    )
    assert counts[-1] > counts[0], "richest and poorest regions supported the same count"


def test_raising_the_floor_plants_fewer_markets():
    lenient = _market_world(market_viability_floor=3.0)
    strict = _market_world(market_viability_floor=9.0)
    assert len(strict.settlements) < len(lenient.settlements)


def test_no_target_count_is_consulted():
    """target_town_count belongs to the classic model and must not leak into this one."""
    few = _market_world(target_town_count=1)
    many = _market_world(target_town_count=999)
    assert len(few.settlements) == len(many.settlements)


# --- determinism -------------------------------------------------------------


def test_same_seed_same_markets():
    a = _market_world(seed=99)
    b = _market_world(seed=99)
    assert sorted(s.coord for s in a.settlements) == sorted(s.coord for s in b.settlements)
    assert sorted(s.population for s in a.settlements) == sorted(
        s.population for s in b.settlements
    )


def test_same_seed_same_catchments():
    """The partition is the most order-sensitive artefact here, so it is checked directly."""
    a = _market_world(seed=99)
    b = _market_world(seed=99)
    assert {c: h.territory for c, h in a.hexes.items()} == {
        c: h.territory for c, h in b.hexes.items()
    }
