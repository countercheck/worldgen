"""Market planting and catchment allocation.

The claims worth testing are structural rather than numeric: that counts follow the land
rather than a config target, that every market is a plausible site with a real catchment,
and that the same seed gives the same markets.
"""

import statistics

import pytest

from tests.worlds import build_pipeline
from worldgen.core.config import WorldConfig
from worldgen.core.hex import Biome, SettlementTier, TerrainClass
from worldgen.core.hex_grid import distance, grade_reachable_count, hex_range, ring
from worldgen.stages.habitability import potential_food, site_bonus
from worldgen.stages.haulage import (
    allocate_catchments,
    fishery_rim,
    settleable,
    travel_field,
    usable_fraction,
)
from worldgen.stages.land_use import LandUseStage
from worldgen.stages.markets import MarketStage, day_reach
from worldgen.stages.riverside import river_index
from worldgen.stages.road_cost import grade_is_under_cap

_WATER = (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER)


def _market_world(seed=42, width=64, height=64, **overrides):
    """Sited *and* founded: `MarketStage` plants and allocates, `LandUseStage` sizes.

    Both are needed for a world with settlements in it. A market is worth what its
    countryside sends, and until the countryside has been put to use that is not known —
    so population has one owner, one stage later than the siting.
    """
    p = build_pipeline(
        seed=seed, width=width, height=height, until="HabitabilityStage", **overrides
    )
    p.add_stage(MarketStage)
    p.add_stage(LandUseStage)
    return p.run()


@pytest.fixture(scope="module")
def markets():
    return _market_world()


# --- siting reads the ground the catchment walks ---------------------------


@pytest.fixture(scope="module")
def sited():
    """A small world stopped before markets, with what siting reads off it."""
    state = build_pipeline(seed=42, width=32, height=32, until="HabitabilityStage").run()
    cfg = WorldConfig(**state.metadata["config"])
    rivers = river_index(state, cfg)
    return state, cfg, rivers


def test_day_reach_is_the_catchment_a_lone_seat_would_get(sited):
    """Siting and the gather agree: a candidate is scored on exactly its catchment.

    The disc this replaced counted every hex within ten rings, whatever stood between, so
    a market was ranked on countryside its catchment then failed to deliver. The day-reach
    must be what `allocate_catchments` and `fishery_rim` hand a seat standing alone, at
    the weight the market is sized on.
    """
    state, cfg, rivers = sited
    hexes = state.hexes
    radius = cfg.market_day_radius
    field = travel_field(hexes, cfg, rivers)
    for seat in sorted(settleable(hexes, cfg))[::17]:
        owner, cost = allocate_catchments(hexes, [seat], radius, cfg, rivers)
        owner, cost = fishery_rim(hexes, owner, cost)
        expected = {
            c: usable_fraction(cost[c], radius)
            for c in owner
            if usable_fraction(cost[c], radius) > 0.0
        }
        got = dict(day_reach(seat, radius, field))
        assert got.keys() == expected.keys(), f"day-reach of {seat} is not its catchment"
        for c, w in got.items():
            assert w == pytest.approx(expected[c])


def test_a_ridge_beside_a_candidate_lowers_its_score(sited):
    """The bug in #117: the disc scored the far side of a ridge as if it could be walked."""
    import copy

    state, cfg, rivers = sited
    radius = cfg.market_day_radius
    surplus = {c: potential_food(h, cfg) for c, h in state.hexes.items()}

    def score(hexes, seat):
        field = travel_field(hexes, cfg, rivers)
        return sum(surplus[c] * w for c, w in day_reach(seat, radius, field))

    # The best-fed candidate, so there is countryside beyond the ridge to lose.
    seat = max(sorted(settleable(state.hexes, cfg)), key=lambda c: score(state.hexes, c))
    open_score = score(state.hexes, seat)

    walled = copy.deepcopy(state.hexes)
    for c in ring(seat, 2):
        if c in walled:
            walled[c].elevation += 5000.0
    walled_score = score(walled, seat)

    assert walled_score < open_score, (
        f"a ridge round {seat} left its score at {walled_score:.1f} against "
        f"{open_score:.1f} — siting is not reading travel cost"
    )


def test_lazy_greedy_sites_the_same_markets_as_brute_force(sited):
    """Lazy greedy over the disc bound is exact, not a heuristic.

    The heap is seeded with the plain ring disc as an upper bound so the day-reach Dijkstra
    only runs on candidates that pop. If that bound were ever below a true score, lazy
    greedy would plant somewhere other than the true best site. So rescore every candidate
    after every plant, the slow way, and the seats must come out the same.
    """
    state, cfg, rivers = sited
    hexes = state.hexes
    radius = cfg.market_day_radius
    floor = 6.0
    cfg = WorldConfig(**{**state.metadata["config"], "market_viability_floor": floor})
    surplus = {
        c: potential_food(h, cfg) * cfg.marketable_surplus_fraction for c, h in hexes.items()
    }
    lazy = MarketStage(cfg, None)._plant(hexes, surplus, cfg, rivers)
    assert len(lazy) >= 3, f"only {len(lazy)} markets; the comparison needs a subject"

    field = travel_field(hexes, cfg, rivers)
    candidates = [
        c
        for c in sorted(settleable(hexes, cfg))
        if grade_reachable_count(
            c, hexes, lambda a, b: grade_is_under_cap(a, b, cfg), cfg.settlement_min_reachable
        )
        >= cfg.settlement_min_reachable
    ]
    reach = {c: day_reach(c, radius, field) for c in candidates}
    bonus = {c: 1.0 + site_bonus(c, hexes[c], hexes, cfg, rivers) for c in candidates}
    remaining = dict(surplus)
    suppressed: set = set()
    brute = []
    while True:
        scored = [
            (sum(remaining[n] * w for n, w in reach[c]) * bonus[c], c)
            for c in candidates
            if c not in suppressed
        ]
        if not scored:
            break
        best, seat = max(scored, key=lambda sc: (sc[0], tuple(-x for x in sc[1])))
        if best < floor:
            break
        brute.append(seat)
        suppressed |= set(hex_range(seat, cfg.market_min_separation))
        for n, w in reach[seat]:
            remaining[n] *= 1.0 - w

    assert lazy == brute


# --- where markets land ------------------------------------------------------


def test_a_subarctic_region_is_settled_but_thinly():
    """Cold country supports people; it just does not support many of them.

    Boreal used to come out all but empty — five markets on fifteen thousand land hexes —
    and the cause was the biome rule rather than anything about haulage: two-fifths of the
    map was bare rock by an elevation test and another quarter was tundra because the same
    400 mm that separates desert from steppe was being asked where trees stop. Both are
    worth no food at all, so there was nothing to gather. Taiga is poor ground, not dead
    ground, and the difference is the whole tier.

    On a 128x128 mainland rather than the 64x64 island the rest of this module uses. Since
    #117 a market is sited on what a day's walk from it actually reaches, which reads
    *local* concentration where the old ring disc read a regional total — and taiga is
    fertility spread thin, so its best sites score near the viability floor wherever they
    are. How many of them clear it is then a question of how many sites there are to draw
    from: the island has too few for the claim to rest on, a bigger mainland has enough.
    Temperate is grown on the same ground, so "thinly" is still measured against it.
    """

    def world(climate):
        return _market_world(
            width=128,
            height=128,
            model="organic",
            continent_falloff_edges=("south",),
            regional_climate=climate,
        )

    boreal = len(world("boreal").settlements)
    temperate = len(world("temperate").settlements)
    assert boreal >= 3, f"a subarctic region supported {boreal} markets — it is not empty land"
    assert boreal < temperate, (
        f"boreal grew {boreal} markets against temperate's {temperate}; cold country should "
        "be the thinner of the two"
    )


def test_markets_are_planted(markets):
    assert markets.settlements, "no markets on a 64x64 temperate map"


def test_markets_avoid_unsettleable_ground(markets):
    """The same exclusions habitability scores to zero."""
    for s in markets.settlements:
        hx = markets.hexes[s.coord]
        assert hx.terrain_class not in _WATER, f"market in the water at {s.coord}"
        assert hx.slope < WorldConfig().terrain_steep_gradient_m, f"market on a peak at {s.coord}"
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
        if h.slope >= WorldConfig().terrain_steep_gradient_m and h.territory is None
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

    If size were random it would be uncorrelated with the ground each market holds, so the
    larger half of the markets must hold more ground than the smaller half.  Halves rather
    than the single largest against the single smallest: the extremes are one market each,
    and a fertile market holding little rich ground can sit beside a poor one holding much
    — the correlation across all of them (0.44 here) is the claim, not either outlier.
    """
    by_pop = sorted(markets.settlements, key=lambda s: s.population)
    half = len(by_pop) // 2

    def owned(seat):
        return sum(1 for h in markets.hexes.values() if h.territory == seat)

    smaller = sum(owned(s.coord) for s in by_pop[:half]) / half
    larger = sum(owned(s.coord) for s in by_pop[-half:]) / half
    assert larger > smaller, (
        f"the larger markets hold {larger:.1f} hexes, the smaller {smaller:.1f}"
    )


# --- counts follow the land --------------------------------------------------


def test_market_count_tracks_the_surplus_on_offer():
    """The acceptance test for the whole model: counts follow the land.

    Five climates over one seed and one map, ordered by the food their land actually
    produces — the market count must rise with it.  The classic model would place exactly
    `target_town_count` on all five.

    Phrased against measured food rather than against a list of climates in an assumed
    order.  The ranking is a property of `food_value`, not of this stage, and what this
    stage owes is that it follows whatever surface it is given — so the surface is
    measured and the counts checked against it.
    """
    from worldgen.stages.habitability import potential_food

    rows = []
    for climate in ("boreal", "tropical", "temperate", "arid", "mediterranean"):
        state = _market_world(regional_climate=climate)
        cfg = state.metadata["config"]
        total = sum(
            potential_food(h, WorldConfig(**cfg))
            for h in state.hexes.values()
            if h.terrain_class not in _WATER
        )
        rows.append((total, len(state.settlements), climate))

    rows.sort()
    summary = ", ".join(f"{c} food={t:.0f} markets={n}" for t, n, c in rows)

    # Compared only between regions whose food differs by more than a quarter. Two
    # climates within a few per cent of each other are a tie, and demanding the model
    # resolve a tie into a strict ordering tests the noise rather than the claim.
    for i, (food_lo, count_lo, name_lo) in enumerate(rows):
        for food_hi, count_hi, name_hi in rows[i + 1 :]:
            if food_hi > food_lo * 1.25:
                assert count_hi >= count_lo, (
                    f"{name_hi} has {food_hi / food_lo:.1f}x the food of {name_lo} but "
                    f"fewer markets — count is not following the land. {summary}"
                )

    assert rows[-1][1] > rows[0][1], f"richest and poorest supported the same count. {summary}"


def test_fertility_decides_how_many_markets_not_how_big():
    """The other half of the claim `test_market_count_tracks_the_surplus_on_offer` makes.

    A rich region does not grow bigger market towns than a poor one — it grows *more* of
    them, because a market is bounded by what a cart can reach in a day's return and
    surplus beyond that radius founds the next market rather than swelling this one.  So
    counts diverge across climates (the sibling test) while the typical market stays the
    same size, which is why the same absolute `market_viability_floor` calibrates on every
    map instead of needing a per-climate target.

    Climates supporting fewer than three markets are skipped: a median over two is an
    accident, not a distribution.
    """
    medians = {}
    for climate in ("boreal", "tropical", "temperate", "arid", "mediterranean"):
        pops = [s.population for s in _market_world(regional_climate=climate).settlements]
        if len(pops) >= 3:
            medians[climate] = statistics.median(pops)

    assert len(medians) >= 3, f"too few climates produced markets to compare: {medians}"
    summary = ", ".join(f"{c}={m:.0f}" for c, m in sorted(medians.items()))

    lo, hi = min(medians.values()), max(medians.values())
    assert hi <= lo * 2.5, (
        f"median market population varies {hi / lo:.1f}x across climates — fertility is "
        f"sizing markets rather than counting them. {summary}"
    )
    # And it lands where a market town historically did, on every climate rather than on
    # a calibration climate. `people_per_food` is what sets the level.
    assert all(500 <= m <= 4000 for m in medians.values()), (
        f"a median market fell outside the 500-4000 band of a real market town. {summary}"
    )


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
