from collections import Counter

import pytest

from worldgen.core.config import CLIMATE_CONTEXTS, WorldConfig
from worldgen.core.hex import Biome, TerrainClass
from worldgen.core.pipeline import GeneratorPipeline
from worldgen.stages.biomes import BiomeStage
from worldgen.stages.climate import ClimateStage
from worldgen.stages.elevation import ElevationStage
from worldgen.stages.erosion import ErosionStage
from worldgen.stages.hydrology import HydrologyStage
from worldgen.stages.terrain_class import TerrainClassificationStage


def _build_pipeline(seed: int = 42, width: int = 32, height: int = 32):
    cfg = WorldConfig(width=width, height=height)
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
    alpine_elev = WorldConfig(**biome_state.metadata["config"]).treeline_m()
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


def test_every_climate_has_a_treeline_above_its_own_lowland():
    """No region may have its treeline at sea level.

    The alpine test runs before every temperature rule, so a treeline at 0 m makes the
    whole of a region bare rock — no forest, no tundra, nothing its palette says it should
    grow.  That is what happened when `biome_treeline_temp_c` was set to the same 1 C as
    the boreal region's own mean: the boreal map came out entirely ALPINE and supported
    five settlements on sixteen thousand hexes.

    A quarter of the map's vertical scale is the bar, not merely "above zero", because a
    treeline just above the shoreline is the same failure in slower motion.
    """
    for climate in CLIMATE_CONTEXTS:
        cfg = WorldConfig(regional_climate=climate)
        assert cfg.treeline_m() > 0.25 * cfg.max_elevation_m, (
            f"{climate} region has its treeline at {cfg.treeline_m():.0f} m, so most of it "
            f"is above the treeline before any terrain is generated"
        )


def test_a_boreal_region_grows_boreal_forest():
    """A region named for its forest must actually produce it.

    The palette test alone does not catch this: TUNDRA and BOREAL stayed in the boreal
    palette throughout, they were simply never reached.
    """
    state = _build_pipeline(width=48, height=48).run()
    cfg = WorldConfig(**state.metadata["config"])
    assert cfg.regional_climate == "temperate", "fixture climate changed; update this test"

    p = GeneratorPipeline(42, WorldConfig(width=48, height=48, regional_climate="boreal"))
    for stage in (
        ElevationStage,
        ErosionStage,
        TerrainClassificationStage,
        HydrologyStage,
        ClimateStage,
        BiomeStage,
    ):
        p.add_stage(stage)
    boreal = p.run()

    land = [h for h in boreal.hexes.values() if h.terrain_class != TerrainClass.OCEAN]
    wooded = sum(1 for h in land if h.biome in (Biome.BOREAL, Biome.TUNDRA))
    assert wooded / len(land) > 0.1, (
        f"Only {wooded / len(land):.1%} of a boreal region is taiga or tundra; "
        f"got {Counter(h.biome.name for h in land).most_common(4)}"
    )


def test_reproducibility():
    s1 = _build_pipeline(seed=7).run()
    s2 = _build_pipeline(seed=7).run()
    for coord in s1.hexes:
        assert s1.hexes[coord].biome == s2.hexes[coord].biome, (
            f"biome differs at {coord} between identical seeds"
        )
