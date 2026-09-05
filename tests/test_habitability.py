import pytest

from tests.worlds import build_pipeline, build_world
from worldgen.core.config import WorldConfig
from worldgen.core.hex import Biome, Hex, LandCover, LandUse, SoilQuality, TerrainClass
from worldgen.stages.habitability import actual_food, catchment_means, potential_food

TIERS = ("habitability_city", "habitability_town", "habitability_village")

# Scoring reads land_cover, so the run has to reach LandCoverStage — but nothing
# downstream of HabitabilityStage affects these scores, so it stops there.
_HAB_STOP = "HabitabilityStage"


def _build_pipeline(seed: int = 42, width: int = 48, height: int = 48):
    return build_pipeline(seed=seed, width=width, height=height, until=_HAB_STOP)


# --- what a hex is worth --------------------------------------------------------


def _hex(soil=None, cover=None, use=None, precip_mm=700):
    return Hex(coord=(0, 0), soil=soil, land_cover=cover, land_use=use, moisture=precip_mm)


def test_unusable_ground_feeds_nobody():
    assert potential_food(_hex(SoilQuality.UNUSABLE), WorldConfig()) == 0.0


def test_the_soil_ladder_is_ordered():
    """Each rung is worth strictly more than the one below.

    The ordering is what every other claim in the model rests on, and it is asserted here
    rather than assumed from the config defaults, which anybody may retune.
    """
    cfg = WorldConfig()
    ladder = [
        potential_food(_hex(s), cfg)
        for s in (
            SoilQuality.UNUSABLE,
            SoilQuality.GRAZING,
            SoilQuality.MARGINAL,
            SoilQuality.ARABLE,
            SoilQuality.PRIME,
        )
    ]
    assert ladder == sorted(ladder), ladder
    assert len(set(ladder)) == len(ladder), "two soil classes are worth the same"


def test_water_is_worth_something_because_a_coast_fishes():
    """Sea in a catchment is not waste ground, and it is valued as a fishery rather than
    as soil — the seabed has no soil class worth the name."""
    cfg = WorldConfig()
    assert potential_food(_hex(cover=LandCover.OPEN_WATER), cfg) == cfg.food_water_value
    assert cfg.food_water_value > 0


def test_wetland_ranks_below_open_water():
    """A marsh is neither good fishing nor good ploughing.

    Valued on cover rather than on soil for the same reason water is: a fen is not
    ploughland, whatever the ground under it would be once drained.
    """
    cfg = WorldConfig()
    marsh = potential_food(_hex(cover=LandCover.MARSH), cfg)
    water = potential_food(_hex(cover=LandCover.OPEN_WATER), cfg)
    assert 0 < marsh < water


def test_water_and_wetland_ignore_the_soil_under_them():
    """Fishing does not care what the seabed is like."""
    cfg = WorldConfig()
    for cover in (LandCover.OPEN_WATER, LandCover.BOG):
        poor = potential_food(_hex(soil=SoilQuality.UNUSABLE, cover=cover), cfg)
        rich = potential_food(_hex(soil=SoilQuality.PRIME, cover=cover), cfg)
        assert poor == rich


def test_soil_ignores_rainfall_a_second_time():
    """Rainfall enters once, in `SoilStage`, where it chooses the class.

    The tent function it replaced was multiplied onto a cover band, so moving the rainfall
    test into the class boundaries and leaving the multiplier in would have priced the same
    rain twice.
    """
    cfg = WorldConfig()
    parched = potential_food(_hex(SoilQuality.ARABLE, precip_mm=120), cfg)
    watered = potential_food(_hex(SoilQuality.ARABLE, precip_mm=700), cfg)
    assert parched == watered


# --- what it is worth as it is being used ---------------------------------------


def test_clearing_is_worth_something():
    """The gap that gives assarting economic weight.

    Wood standing on prime soil feeds far fewer people than the same soil under the plough,
    which is why a settlement grows by clearing its hinterland rather than merely by
    sitting in it.
    """
    cfg = WorldConfig()
    ploughed = actual_food(_hex(SoilQuality.PRIME, use=LandUse.ARABLE), cfg)
    grazed = actual_food(_hex(SoilQuality.PRIME, use=LandUse.PASTURE), cfg)
    wooded = actual_food(_hex(SoilQuality.PRIME, use=LandUse.WOOD), cfg)
    waste = actual_food(_hex(SoilQuality.PRIME, use=LandUse.WASTE), cfg)
    assert ploughed > grazed > wooded > waste == 0.0


def test_actual_never_exceeds_potential():
    cfg = WorldConfig()
    for soil in SoilQuality:
        for use in LandUse:
            hx = _hex(soil, use=use)
            assert actual_food(hx, cfg) <= potential_food(hx, cfg) + 1e-12


def test_ground_with_no_use_yet_reads_at_its_potential():
    """Every stage before `LandUseStage` sees the surface it is entitled to.

    Siting runs first and must score land for what it will be once worked, not for the
    wildwood standing on it — so an unassigned hex is not silently discounted.
    """
    cfg = WorldConfig()
    hx = _hex(SoilQuality.ARABLE)
    assert hx.land_use is None
    assert actual_food(hx, cfg) == potential_food(hx, cfg)


# --- the catchment -----------------------------------------------------------


def _strip(values):
    return {(q, 0): v for q, v in enumerate(values)}


def test_catchment_is_a_mean_not_a_sum():
    """Otherwise a wider radius always wins regardless of what is in it."""
    food = _strip([1.0] * 9)
    means = catchment_means(food, food, [1, 4])
    assert means[1][(4, 0)] == pytest.approx(1.0)
    assert means[4][(4, 0)] == pytest.approx(1.0)


def test_catchment_averages_what_is_in_range():
    food = {(0, 0): 1.0, (1, 0): 0.0}
    means = catchment_means(food, food, [1])
    assert means[1][(0, 0)] == pytest.approx(0.5)


def test_off_map_neighbours_are_excluded_not_counted_as_zero():
    """A hex on the map border must not be scored as though the edge were desert."""
    food = {(0, 0): 1.0}
    means = catchment_means(food, food, [3])
    assert means[3][(0, 0)] == pytest.approx(1.0)


def test_wider_catchment_reaches_further():
    # A lone fertile hex six away is invisible at radius 2 and visible at radius 8.
    food = {(q, 0): 0.0 for q in range(10)}
    food[(6, 0)] = 1.0
    means = catchment_means(food, food, [2, 8])
    assert means[2][(0, 0)] == 0.0
    assert means[8][(0, 0)] > 0.0


# --- the three scores --------------------------------------------------------


@pytest.mark.parametrize("field", TIERS)
def test_habitability_in_range(hab_state, field):
    for h in hab_state.hexes.values():
        assert 0.0 <= getattr(h, field) <= 1.0, f"{field} {getattr(h, field)} out of [0, 1]"


@pytest.mark.parametrize("field", TIERS)
def test_unsettleable_terrain_scores_zero(hab_state, field):
    """You cannot found a town on open water, a mountain face, or a bog."""
    for h in hab_state.hexes.values():
        if (
            h.terrain_class in (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER)
            or h.slope >= WorldConfig().terrain_steep_gradient_m
            or h.biome == Biome.WETLAND
        ):
            assert getattr(h, field) == 0.0


@pytest.mark.parametrize("field", TIERS)
def test_at_least_one_nonzero(hab_state, field):
    land = [h for h in hab_state.hexes.values() if h.terrain_class != TerrainClass.OPEN_WATER]
    assert any(getattr(h, field) > 0 for h in land), f"No hex scores on {field}"


@pytest.mark.parametrize("field", TIERS)
def test_river_hexes_score_higher(hab_state, field):
    from worldgen.core.hex_grid import neighbors

    river_scores = []
    plain_scores = []
    for coord, h in hab_state.hexes.items():
        if (
            h.terrain_class == TerrainClass.OPEN_WATER
            or h.slope >= WorldConfig().terrain_steep_gradient_m
        ):
            continue
        if h.biome == Biome.WETLAND:
            continue
        nbrs = [hab_state.hexes[n] for n in neighbors(coord) if n in hab_state.hexes]
        has_river = "river" in h.tags or any("river" in n.tags for n in nbrs)
        has_coast = any(n.terrain_class == TerrainClass.COAST for n in nbrs)
        if has_river:
            river_scores.append(getattr(h, field))
        elif not has_coast:
            plain_scores.append(getattr(h, field))

    if river_scores and plain_scores:
        assert sum(river_scores) / len(river_scores) > sum(plain_scores) / len(plain_scores), (
            f"River-adjacent hexes not more habitable than plain land on {field}"
        )


def test_the_three_scores_rank_sites_differently():
    """If every tier ranked sites identically the split would buy nothing.

    The best hex may well top all three — a confluence on good ground is a good site at
    any size.  What has to differ is the *order* further down, where a wide catchment
    and a narrow one disagree about which of two sites is better.
    """
    state = build_world(width=48, height=48, until=_HAB_STOP)

    def ranking(field):
        scored = [(getattr(h, field), c) for c, h in state.hexes.items() if getattr(h, field) > 0]
        return [c for _, c in sorted(scored, reverse=True)]

    city_rank, village_rank = ranking("habitability_city"), ranking("habitability_village")
    assert city_rank and village_rank
    assert city_rank != village_rank, "city and village catchments produced the same ordering"


def test_a_wider_catchment_is_smoother():
    """A radius-8 mean varies less between neighbours than a radius-2 one.

    Measured on the catchment means themselves, not on the habitability scores derived
    from them.  Each tier is normalised against its own best site, and the wide
    catchment has the narrower raw spread, so dividing by its smaller maximum inflates
    its neighbour deltas right back — comparing the normalised scores tests the
    normalisation, not the smoothing, and gives an answer that flips with the seed.
    """
    from worldgen.core.hex_grid import neighbors
    from worldgen.stages.habitability import catchment_means, potential_food

    cfg = WorldConfig(width=48, height=48)
    state = build_world(width=48, height=48, until=_HAB_STOP)
    hexes = state.hexes

    food = {coord: potential_food(hx, cfg) for coord, hx in hexes.items()}
    wide, narrow = cfg.cultivation_city_radius, cfg.cultivation_village_radius
    means = catchment_means(hexes.keys(), food, [wide, narrow])

    def mean_neighbour_delta(table):
        deltas = []
        for coord in hexes:
            own = table[coord]
            for n in neighbors(coord):
                if n in hexes:
                    deltas.append(abs(own - table[n]))
        return sum(deltas) / len(deltas)

    assert mean_neighbour_delta(means[wide]) < mean_neighbour_delta(means[narrow])


def test_reproducibility():
    s1 = _build_pipeline(seed=7).run()
    s2 = _build_pipeline(seed=7).run()
    for coord in s1.hexes:
        for field in TIERS:
            assert getattr(s1.hexes[coord], field) == getattr(s2.hexes[coord], field), (
                f"{field} differs at {coord} between identical seeds"
            )
