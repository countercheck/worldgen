import pytest

from tests.worlds import build_pipeline, build_world
from worldgen.core.config import WorldConfig
from worldgen.core.hex import Biome, Hex, LandCover, TerrainClass
from worldgen.stages.habitability import catchment_means, food_value, moisture_factor

TIERS = ("habitability_city", "habitability_town", "habitability_village")

# Scoring reads land_cover, so the run has to reach LandCoverStage — but nothing
# downstream of HabitabilityStage affects these scores, so it stops there.
_HAB_STOP = "HabitabilityStage"


def _build_pipeline(seed: int = 42, width: int = 48, height: int = 48):
    return build_pipeline(seed=seed, width=width, height=height, until=_HAB_STOP)


# --- the moisture curve ------------------------------------------------------


def test_moisture_factor_is_full_across_the_temperate_band():
    for mm in (400, 600, 800, 1000):
        assert moisture_factor(mm, 400, 1000, 3000) == pytest.approx(1.0)


def test_moisture_factor_falls_off_both_ends():
    """Not a ramp: a swamp must not outrank a meadow."""
    assert moisture_factor(0, 400, 1000, 3000) == 0.0
    assert moisture_factor(3000, 400, 1000, 3000) == 0.0, "drowned ground feeds nobody"
    assert 0.0 < moisture_factor(200, 400, 1000, 3000) < 1.0
    assert 0.0 < moisture_factor(2000, 400, 1000, 3000) < 1.0


def test_moisture_factor_is_monotonic_toward_the_band():
    dry_side = [moisture_factor(mm, 400, 1000, 3000) for mm in range(20, 400, 20)]
    wet_side = [moisture_factor(mm, 400, 1000, 3000) for mm in range(1020, 3000, 100)]
    assert dry_side == sorted(dry_side), "drier than the band should only improve"
    assert wet_side == sorted(wet_side, reverse=True), "wetter than the band should only worsen"


def test_moisture_factor_stays_in_unit_range():
    for m in range(0, 101):
        assert 0.0 <= moisture_factor(m * 40, 400, 1000, 3000) <= 1.0


# --- the food bands ----------------------------------------------------------


# Rainfall in millimetres a year. 700 sits inside the agricultural band, so a cover's own
# value is what these tests are reading.
_BAND = (400, 1000)


def _hex(cover, precip_mm=700):
    return Hex(coord=(0, 0), land_cover=cover, moisture=precip_mm)


def test_barren_covers_feed_nobody():
    cfg = WorldConfig()
    for cover in (
        LandCover.TUNDRA,
        LandCover.DESERT,
        LandCover.ALPINE,
        LandCover.BARE_ROCK,
    ):
        assert food_value(_hex(cover), cfg, *_BAND) == 0.0


def test_water_is_worth_something_because_a_coast_fishes():
    """The whole point of the band: sea in a catchment is not waste ground."""
    cfg = WorldConfig()
    assert food_value(_hex(LandCover.OPEN_WATER), cfg, *_BAND) == cfg.food_water_value
    assert cfg.food_water_value > 0


def test_wetland_ranks_below_open_water():
    """A marsh is neither good fishing nor good ploughing."""
    cfg = WorldConfig()
    marsh = food_value(_hex(LandCover.MARSH), cfg, *_BAND)
    water = food_value(_hex(LandCover.OPEN_WATER), cfg, *_BAND)
    assert 0 < marsh < water


def test_fertile_outranks_marginal():
    cfg = WorldConfig()
    fertile = food_value(_hex(LandCover.OPEN), cfg, *_BAND)
    marginal = food_value(_hex(LandCover.SCRUB), cfg, *_BAND)
    assert fertile > marginal > 0


def test_moisture_discriminates_within_a_band():
    """The detail land cover throws away: same cover, different ground."""
    cfg = WorldConfig()
    in_band = food_value(_hex(LandCover.OPEN, precip_mm=700), cfg, *_BAND)
    parched = food_value(_hex(LandCover.OPEN, precip_mm=120), cfg, *_BAND)
    drowned = food_value(_hex(LandCover.OPEN, precip_mm=2600), cfg, *_BAND)
    assert in_band > parched
    assert in_band > drowned


def test_water_and_wetland_ignore_moisture():
    """Fishing does not care about rainfall, and a bog is saturated by definition."""
    cfg = WorldConfig()
    for cover in (LandCover.OPEN_WATER, LandCover.BOG):
        dry = food_value(_hex(cover, precip_mm=80), cfg, *_BAND)
        wet = food_value(_hex(cover, precip_mm=2800), cfg, *_BAND)
        assert dry == wet


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
            h.terrain_class in (TerrainClass.OCEAN, TerrainClass.LAKE, TerrainClass.STEEP)
            or h.biome == Biome.WETLAND
        ):
            assert getattr(h, field) == 0.0


@pytest.mark.parametrize("field", TIERS)
def test_at_least_one_nonzero(hab_state, field):
    land = [h for h in hab_state.hexes.values() if h.terrain_class != TerrainClass.OCEAN]
    assert any(getattr(h, field) > 0 for h in land), f"No hex scores on {field}"


@pytest.mark.parametrize("field", TIERS)
def test_river_hexes_score_higher(hab_state, field):
    from worldgen.core.hex_grid import neighbors

    river_scores = []
    plain_scores = []
    for coord, h in hab_state.hexes.items():
        if h.terrain_class in (TerrainClass.OCEAN, TerrainClass.STEEP):
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
    from worldgen.stages.habitability import catchment_means, food_value

    cfg = WorldConfig(width=48, height=48)
    state = build_world(width=48, height=48, until=_HAB_STOP)
    hexes = state.hexes

    food = {
        coord: food_value(hx, cfg, cfg.biome_dry_precip_mm, cfg.biome_wet_precip_mm)
        for coord, hx in hexes.items()
    }
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
