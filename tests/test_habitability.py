import pytest

from worldgen.core.config import WorldConfig
from worldgen.core.hex import Biome, TerrainClass
from worldgen.core.pipeline import GeneratorPipeline
from worldgen.stages.biomes import BiomeStage
from worldgen.stages.climate import ClimateStage
from worldgen.stages.elevation import ElevationStage
from worldgen.stages.erosion import ErosionStage
from worldgen.stages.habitability import HabitabilityStage
from worldgen.stages.hydrology import HydrologyStage
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
    p.add_stage(HabitabilityStage)
    return p


@pytest.fixture(scope="module")
def hab_state():
    return _build_pipeline().run()


def test_habitability_in_range(hab_state):
    for h in hab_state.hexes.values():
        assert 0.0 <= h.habitability <= 1.0, f"habitability {h.habitability} out of [0, 1]"


def test_ocean_habitability_zero(hab_state):
    for h in hab_state.hexes.values():
        if h.terrain_class == TerrainClass.OCEAN:
            assert h.habitability == 0.0


def test_mountain_habitability_zero(hab_state):
    for h in hab_state.hexes.values():
        if h.terrain_class == TerrainClass.MOUNTAIN:
            assert h.habitability == 0.0


def test_wetland_habitability_zero(hab_state):
    for h in hab_state.hexes.values():
        if h.biome == Biome.WETLAND:
            assert h.habitability == 0.0


def test_river_hexes_higher_habitability(hab_state):
    from worldgen.core.hex import TerrainClass
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
            river_scores.append(h.habitability)
        elif not has_coast:
            plain_scores.append(h.habitability)

    if river_scores and plain_scores:
        assert sum(river_scores) / len(river_scores) > sum(plain_scores) / len(plain_scores), (
            "River-adjacent hexes not more habitable than plain land hexes"
        )


def test_at_least_one_nonzero(hab_state):
    land = [h for h in hab_state.hexes.values() if h.terrain_class != TerrainClass.OCEAN]
    assert any(h.habitability > 0 for h in land), "No habitable land hexes found"


def test_reproducibility():
    s1 = _build_pipeline(seed=7).run()
    s2 = _build_pipeline(seed=7).run()
    for coord in s1.hexes:
        assert s1.hexes[coord].habitability == s2.hexes[coord].habitability, (
            f"habitability differs at {coord} between identical seeds"
        )
