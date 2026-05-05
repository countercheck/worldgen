import pytest

from worldgen.core.config import WorldConfig
from worldgen.core.hex import Biome, TerrainClass
from worldgen.core.pipeline import GeneratorPipeline
from worldgen.stages.biomes import BiomeStage
from worldgen.stages.climate import ClimateStage
from worldgen.stages.elevation import ElevationStage
from worldgen.stages.erosion import ErosionStage
from worldgen.stages.hydrology import HydrologyStage
from worldgen.stages.terrain_class import TerrainClassificationStage


def _build_pipeline(seed: int = 42, width: int = 32, height: int = 32):
    cfg = WorldConfig(width=width, height=height, erosion_iterations=500)
    p = GeneratorPipeline(seed, cfg)
    p.add_stage(ElevationStage)
    p.add_stage(ErosionStage)
    p.add_stage(TerrainClassificationStage)
    p.add_stage(HydrologyStage)
    p.add_stage(ClimateStage)
    p.add_stage(BiomeStage)
    return p


@pytest.fixture(scope="module")
def biome_state():
    return _build_pipeline().run()


def test_all_hexes_have_biome(biome_state):
    for coord, h in biome_state.hexes.items():
        assert h.biome is not None, f"Hex {coord} has no biome assigned"


def test_ocean_hexes_have_ocean_biome(biome_state):
    for h in biome_state.hexes.values():
        if h.terrain_class == TerrainClass.OCEAN:
            assert h.biome == Biome.OCEAN, f"Ocean hex has biome {h.biome}"


def test_alpine_hexes_assigned(biome_state):
    alpine_elev = biome_state.metadata["config"]["biome_alpine_elev"]
    high_land = [
        h
        for h in biome_state.hexes.values()
        if h.elevation > alpine_elev and h.terrain_class != TerrainClass.OCEAN
    ]
    for h in high_land:
        assert h.biome == Biome.ALPINE, (
            f"High-elevation hex (elev={h.elevation:.2f}) has biome {h.biome}, expected ALPINE"
        )


def test_biome_distribution_sanity(biome_state):
    # A 32×32 map should produce at least a few distinct biome types
    biomes_present = {h.biome for h in biome_state.hexes.values()}
    assert len(biomes_present) >= 4, f"Too few distinct biomes: {biomes_present}"


def test_reproducibility():
    s1 = _build_pipeline(seed=7).run()
    s2 = _build_pipeline(seed=7).run()
    for coord in s1.hexes:
        assert s1.hexes[coord].biome == s2.hexes[coord].biome, (
            f"biome differs at {coord} between identical seeds"
        )
