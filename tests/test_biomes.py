from collections import Counter

import numpy as np
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
        if h.terrain_class == TerrainClass.OPEN_WATER:
            assert h.biome == Biome.OCEAN, f"Ocean hex has biome {h.biome}"


def _cold_country(climate: str = "boreal", max_elevation_m: float = 3500.0, size: int = 48):
    """A map with enough relief to reach past both cold lines.

    The shipped 1500 m never gets near the snowline, which is the point of where that line
    is set — but it means a fixture at the default can say nothing about ALPINE at all.
    The test this replaced looked like it checked the alpine rule and in fact asserted
    over an empty list, because a temperate treeline stands at 1846 m on a map 1500 m high.
    """
    cfg = WorldConfig(
        width=size, height=size, regional_climate=climate, max_elevation_m=max_elevation_m
    )
    p = GeneratorPipeline(42, cfg)
    for stage in (
        ElevationStage,
        ErosionStage,
        TerrainClassificationStage,
        HydrologyStage,
        ClimateStage,
        BiomeStage,
    ):
        p.add_stage(stage)
    state = p.run()
    land = [h for h in state.hexes.values() if h.terrain_class != TerrainClass.OPEN_WATER]
    return cfg, state, land


@pytest.fixture(scope="module")
def cold_country():
    return _cold_country()


def test_the_two_cold_lines_stack_in_temperature_order(cold_country):
    """Bare rock above the snowline, tundra between the lines, trees below the treeline.

    The single claim the whole cold band rests on, and it is an ordering rather than a
    count: whatever the terrain does, no ALPINE hex may be warmer than a TUNDRA one and no
    TUNDRA hex warmer than a hex with trees on it.
    """
    cfg, _, land = cold_country
    alpine = [h.temperature for h in land if h.biome is Biome.ALPINE]
    tundra = [h.temperature for h in land if h.biome is Biome.TUNDRA]
    wooded = [h.temperature for h in land if h.biome in (Biome.BOREAL, Biome.TEMPERATE_FOREST)]
    assert alpine and tundra and wooded, (
        f"fixture reached only {sorted({h.biome.name for h in land})} — it cannot test the "
        "ordering without all three"
    )
    assert max(alpine) < cfg.biome_snowline_temp_c <= min(tundra), (
        f"alpine runs to {max(alpine):.1f} C and tundra starts at {min(tundra):.1f} C, "
        f"either side of a snowline at {cfg.biome_snowline_temp_c} C"
    )
    assert max(tundra) < cfg.biome_treeline_temp_c <= min(wooded), (
        f"tundra runs to {max(tundra):.1f} C and forest starts at {min(wooded):.1f} C, "
        f"either side of a treeline at {cfg.biome_treeline_temp_c} C"
    )


def test_a_default_map_carries_no_permanent_snow():
    """The snowline is set out of reach of 1500 m of relief, deliberately.

    A temperate range that high has no glaciers, so neither should the map. This is what
    makes ALPINE mean something when it does appear, rather than being the label for every
    hill top — under the elevation-keyed rule it was 41% of a boreal map.
    """
    for climate in CLIMATE_CONTEXTS:
        _, _, land = _cold_country(climate, max_elevation_m=1500.0, size=32)
        bare = [h for h in land if h.biome is Biome.ALPINE]
        assert not bare, f"{climate} at 1500 m grew {len(bare)} hexes of bare peak"


def test_relief_alone_brings_the_snowline_into_reach():
    """And raising the ground is all it takes — no per-map alpine setting."""
    _, _, land = _cold_country("boreal", max_elevation_m=2400.0)
    assert any(h.biome is Biome.ALPINE for h in land), (
        "2400 m of subarctic relief reached no bare ground at all"
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

    land = [h for h in boreal.hexes.values() if h.terrain_class != TerrainClass.OPEN_WATER]
    wooded = sum(1 for h in land if h.biome in (Biome.BOREAL, Biome.TUNDRA))
    assert wooded / len(land) > 0.1, (
        f"Only {wooded / len(land):.1%} of a boreal region is taiga or tundra; "
        f"got {Counter(h.biome.name for h in land).most_common(4)}"
    )


def test_below_the_treeline_a_boreal_region_is_taiga():
    """Not merely present — the staple.

    The weaker test above counts taiga and tundra together, and passed throughout the
    period when a boreal map was two-fifths bare rock and the rest split between the two.
    What a subarctic region should look like is forest wherever trees can stand at all,
    and bare fell above that.
    """
    cfg, _, land = _cold_country("boreal", max_elevation_m=1500.0)
    below = [h for h in land if h.temperature >= cfg.biome_treeline_temp_c]
    taiga = [h for h in below if h.biome is Biome.BOREAL]
    assert len(taiga) / len(below) > 0.7, (
        f"only {len(taiga) / len(below):.0%} of the ground below the treeline is taiga; "
        f"got {Counter(h.biome.name for h in below).most_common(4)}"
    )


def test_rainfall_does_not_decide_where_trees_stop():
    """Cold stops trees in the subarctic, not drought — so the dry threshold cannot move
    the cold band at all.

    This is the defect the two-line rule fixes. TUNDRA used to be the dry arm of the cold
    band, split on the same 400 mm that separates desert from steppe, and the boreal
    region's own mean is 450 — so any rain shadow tipped land into a biome worth no food
    whatever. Siberian larch grows on 200-400 mm. Moving `biome_dry_precip_mm` across the
    region's whole rainfall range must now leave every cold hex exactly where it was.
    """
    # Its own world, not the shared fixture: the loop re-runs BiomeStage in place, and
    # the warm bands really do move with this threshold even though the cold one must not.
    _, state, _ = _cold_country("boreal", max_elevation_m=1500.0)
    baseline = None
    for dry in (200.0, 400.0, 700.0):
        cfg = WorldConfig(width=48, height=48, regional_climate="boreal", biome_dry_precip_mm=dry)
        BiomeStage(cfg, np.random.default_rng(0)).run(state)
        cold = {
            c: h.biome
            for c, h in state.hexes.items()
            if h.temperature < cfg.biome_cold_temp_c and h.terrain_class != TerrainClass.OPEN_WATER
        }
        assert cold, "the fixture has no cold ground to test"
        if baseline is None:
            baseline = cold
            continue
        moved = [c for c in cold if cold[c] is not baseline[c]]
        assert not moved, (
            f"{len(moved)} cold hexes changed biome when biome_dry_precip_mm moved to "
            f"{dry} mm — rainfall is still deciding the cold band"
        )


def test_reproducibility():
    s1 = _build_pipeline(seed=7).run()
    s2 = _build_pipeline(seed=7).run()
    for coord in s1.hexes:
        assert s1.hexes[coord].biome == s2.hexes[coord].biome, (
            f"biome differs at {coord} between identical seeds"
        )
