import pytest

from worldgen.core.config import WorldConfig
from worldgen.core.hex import Biome, Hex, LandCover, TerrainClass
from worldgen.core.pipeline import GeneratorPipeline
from worldgen.stages.biomes import BiomeStage
from worldgen.stages.climate import ClimateStage
from worldgen.stages.elevation import ElevationStage
from worldgen.stages.erosion import ErosionStage
from worldgen.stages.habitability import (
    HabitabilityStage,
    catchment_means,
    food_value,
    moisture_factor,
)
from worldgen.stages.hydrology import HydrologyStage
from worldgen.stages.land_cover import LandCoverStage
from worldgen.stages.terrain_class import TerrainClassificationStage
from worldgen.stages.water_bodies import WaterBodiesStage

TIERS = ("habitability_city", "habitability_town", "habitability_village")


def _build_pipeline(seed: int = 42, width: int = 48, height: int = 48):
    cfg = WorldConfig(width=width, height=height, erosion_iterations=500)
    p = GeneratorPipeline(seed, cfg)
    p.add_stage(ElevationStage)
    p.add_stage(ErosionStage)
    p.add_stage(TerrainClassificationStage)
    p.add_stage(WaterBodiesStage)
    p.add_stage(HydrologyStage)
    p.add_stage(ClimateStage)
    p.add_stage(BiomeStage)
    # Scoring reads land_cover, so the cover stage is no longer optional here.
    p.add_stage(LandCoverStage)
    p.add_stage(HabitabilityStage)
    return p


@pytest.fixture(scope="module")
def hab_state():
    return _build_pipeline().run()


# --- the moisture curve ------------------------------------------------------


def test_moisture_factor_is_full_across_the_temperate_band():
    for m in (0.2, 0.3, 0.4, 0.5):
        assert moisture_factor(m, 0.2, 0.5) == pytest.approx(1.0)


def test_moisture_factor_falls_off_both_ends():
    """Not a ramp: a swamp must not outrank a meadow."""
    assert moisture_factor(0.0, 0.2, 0.5) == 0.0
    assert moisture_factor(1.0, 0.2, 0.5) == 0.0
    assert 0.0 < moisture_factor(0.1, 0.2, 0.5) < 1.0
    assert 0.0 < moisture_factor(0.75, 0.2, 0.5) < 1.0


def test_moisture_factor_is_monotonic_toward_the_band():
    dry_side = [moisture_factor(m / 100, 0.2, 0.5) for m in range(1, 20)]
    wet_side = [moisture_factor(0.5 + m / 100, 0.2, 0.5) for m in range(1, 50)]
    assert dry_side == sorted(dry_side), "drier than the band should only improve"
    assert wet_side == sorted(wet_side, reverse=True), "wetter than the band should only worsen"


def test_moisture_factor_stays_in_unit_range():
    for m in range(0, 101):
        assert 0.0 <= moisture_factor(m / 100, 0.2, 0.5) <= 1.0


# --- the food bands ----------------------------------------------------------


def _hex(cover, moisture=0.35):
    return Hex(coord=(0, 0), land_cover=cover, moisture=moisture)


def test_barren_covers_feed_nobody():
    cfg = WorldConfig()
    for cover in (
        LandCover.TUNDRA,
        LandCover.DESERT,
        LandCover.ALPINE,
        LandCover.BARE_ROCK,
    ):
        assert food_value(_hex(cover), cfg, 0.2, 0.5) == 0.0


def test_water_is_worth_something_because_a_coast_fishes():
    """The whole point of the band: sea in a catchment is not waste ground."""
    cfg = WorldConfig()
    assert food_value(_hex(LandCover.OPEN_WATER), cfg, 0.2, 0.5) == cfg.food_water_value
    assert cfg.food_water_value > 0


def test_wetland_ranks_below_open_water():
    """A marsh is neither good fishing nor good ploughing."""
    cfg = WorldConfig()
    marsh = food_value(_hex(LandCover.MARSH), cfg, 0.2, 0.5)
    water = food_value(_hex(LandCover.OPEN_WATER), cfg, 0.2, 0.5)
    assert 0 < marsh < water


def test_fertile_outranks_marginal():
    cfg = WorldConfig()
    fertile = food_value(_hex(LandCover.OPEN), cfg, 0.2, 0.5)
    marginal = food_value(_hex(LandCover.SCRUB), cfg, 0.2, 0.5)
    assert fertile > marginal > 0


def test_moisture_discriminates_within_a_band():
    """The detail land cover throws away: same cover, different ground."""
    cfg = WorldConfig()
    in_band = food_value(_hex(LandCover.OPEN, moisture=0.35), cfg, 0.2, 0.5)
    parched = food_value(_hex(LandCover.OPEN, moisture=0.05), cfg, 0.2, 0.5)
    drowned = food_value(_hex(LandCover.OPEN, moisture=0.98), cfg, 0.2, 0.5)
    assert in_band > parched
    assert in_band > drowned


def test_water_and_wetland_ignore_moisture():
    """Fishing does not care about rainfall, and a bog is saturated by definition."""
    cfg = WorldConfig()
    for cover in (LandCover.OPEN_WATER, LandCover.BOG):
        dry = food_value(_hex(cover, moisture=0.01), cfg, 0.2, 0.5)
        wet = food_value(_hex(cover, moisture=0.99), cfg, 0.2, 0.5)
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
            h.terrain_class in (TerrainClass.OCEAN, TerrainClass.LAKE, TerrainClass.MOUNTAIN)
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
        if h.terrain_class in (TerrainClass.OCEAN, TerrainClass.MOUNTAIN):
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
    state = _build_pipeline().run()

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

    cfg = WorldConfig(width=48, height=48, erosion_iterations=500)
    state = _build_pipeline().run()
    hexes = state.hexes

    food = {
        coord: food_value(hx, cfg, cfg.biome_dry_moist, cfg.biome_wet_moist)
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
