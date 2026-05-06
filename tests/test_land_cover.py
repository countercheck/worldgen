import pytest

from worldgen.core.config import WorldConfig
from worldgen.core.hex import Biome, LandCover, TerrainClass
from worldgen.core.pipeline import GeneratorPipeline
from worldgen.stages.biomes import BiomeStage
from worldgen.stages.climate import ClimateStage
from worldgen.stages.elevation import ElevationStage
from worldgen.stages.erosion import ErosionStage
from worldgen.stages.hydrology import HydrologyStage
from worldgen.stages.land_cover import LandCoverStage
from worldgen.stages.terrain_class import TerrainClassificationStage


def _build_pipeline(seed: int = 42, width: int = 48, height: int = 48):
    cfg = WorldConfig(width=width, height=height, erosion_iterations=500)
    p = GeneratorPipeline(seed, cfg)
    p.add_stage(ElevationStage)
    p.add_stage(ErosionStage)
    p.add_stage(TerrainClassificationStage)
    p.add_stage(HydrologyStage)
    p.add_stage(ClimateStage)
    p.add_stage(BiomeStage)
    p.add_stage(LandCoverStage)
    return p


@pytest.fixture(scope="module")
def lc_state():
    return _build_pipeline().run()


def test_all_hexes_have_land_cover(lc_state):
    for coord, h in lc_state.hexes.items():
        assert h.land_cover is not None, f"Hex {coord} has no land_cover"


def test_ocean_terrain_is_open_water(lc_state):
    for h in lc_state.hexes.values():
        if h.terrain_class == TerrainClass.OCEAN:
            assert h.land_cover == LandCover.OPEN_WATER, f"Ocean hex has land_cover {h.land_cover}"


def test_mountain_terrain_is_bare_rock(lc_state):
    for h in lc_state.hexes.values():
        if h.terrain_class == TerrainClass.MOUNTAIN:
            assert h.land_cover == LandCover.BARE_ROCK, (
                f"Mountain hex has land_cover {h.land_cover}"
            )


def test_wetland_biome_is_bog_or_marsh(lc_state):
    wetland_hexes = [h for h in lc_state.hexes.values() if h.biome == Biome.WETLAND]
    assert wetland_hexes, "No WETLAND biome hexes found — wetland assignment may be broken"
    for h in wetland_hexes:
        assert h.land_cover in (LandCover.BOG, LandCover.MARSH), (
            f"WETLAND hex has land_cover {h.land_cover}"
        )


def test_boreal_biome_is_dense_forest(lc_state):
    # MOUNTAIN terrain overrides biome for land cover; skip those hexes
    for h in lc_state.hexes.values():
        if h.biome == Biome.BOREAL and h.terrain_class != TerrainClass.MOUNTAIN:
            assert h.land_cover == LandCover.DENSE_FOREST, (
                f"BOREAL non-mountain hex has land_cover {h.land_cover}"
            )


def test_woodland_and_dense_forest_both_present(lc_state):
    covers = {h.land_cover for h in lc_state.hexes.values()}
    assert LandCover.DENSE_FOREST in covers and LandCover.WOODLAND in covers, (
        "Both DENSE_FOREST and WOODLAND must be present"
    )


def test_marsh_land_cover_present(lc_state):
    covers = {h.land_cover for h in lc_state.hexes.values()}
    assert LandCover.MARSH in covers, (
        "MARSH land cover never assigned — coastal wetland biome logic may be broken"
    )


def test_land_cover_values_are_valid(lc_state):
    valid = set(LandCover)
    for coord, h in lc_state.hexes.items():
        assert h.land_cover in valid, f"Invalid land_cover at {coord}: {h.land_cover}"


def test_reproducibility():
    s1 = _build_pipeline(seed=77).run()
    s2 = _build_pipeline(seed=77).run()
    for coord in s1.hexes:
        assert s1.hexes[coord].land_cover == s2.hexes[coord].land_cover, (
            f"land_cover differs at {coord} between identical seeds"
        )
