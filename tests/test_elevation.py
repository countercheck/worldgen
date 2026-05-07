import pytest

from worldgen.core.config import WorldConfig
from worldgen.core.hex import TerrainClass
from worldgen.core.hex_grid import neighbors
from worldgen.core.pipeline import GeneratorPipeline
from worldgen.stages.elevation import ElevationStage
from worldgen.stages.erosion import ErosionStage
from worldgen.stages.terrain_class import TerrainClassificationStage


@pytest.fixture(scope="module")
def phase1_state():
    cfg = WorldConfig(width=64, height=64, erosion_iterations=3000)
    pipeline = GeneratorPipeline(42, cfg)
    pipeline.add_stage(ElevationStage).add_stage(ErosionStage).add_stage(TerrainClassificationStage)
    return pipeline.run()


def test_elevations_normalized(phase1_state):
    for h in phase1_state.hexes.values():
        assert 0.0 <= h.elevation <= 1.0


def test_land_coverage(phase1_state):
    total = len(phase1_state.hexes)
    land = sum(1 for h in phase1_state.hexes.values() if h.terrain_class != TerrainClass.OCEAN)
    assert land / total >= 0.40, f"Only {land / total:.1%} land, expected >= 40%"


def test_coast_borders_ocean(phase1_state):
    for coord, h in phase1_state.hexes.items():
        if h.terrain_class != TerrainClass.COAST:
            continue
        neighbor_classes = [
            phase1_state.hexes[n].terrain_class for n in neighbors(coord) if n in phase1_state.hexes
        ]
        assert TerrainClass.OCEAN in neighbor_classes, f"COAST hex {coord} has no OCEAN neighbor"


def test_mountain_not_isolated(phase1_state):
    mountains = [
        coord for coord, h in phase1_state.hexes.items() if h.terrain_class == TerrainClass.MOUNTAIN
    ]
    if not mountains:
        pytest.skip("No mountain hexes generated")

    isolated = sum(
        1
        for coord in mountains
        if not any(
            phase1_state.hexes.get(n, None)
            and phase1_state.hexes[n].terrain_class == TerrainClass.MOUNTAIN
            for n in neighbors(coord)
            if n in phase1_state.hexes
        )
    )
    assert isolated / len(mountains) < 0.5, (
        f"{isolated / len(mountains):.1%} of mountains are isolated, expected < 50%"
    )


def test_elevation_gradient_tilts_north_high():
    """Negative south_bias should make northern rows higher on average."""
    cfg = WorldConfig(width=32, height=32, erosion_iterations=0, elevation_gradient=(0.0, -0.8))
    p = GeneratorPipeline(42, cfg)
    p.add_stage(ElevationStage)
    state = p.run()

    north_elev = [h.elevation for (_, r), h in state.hexes.items() if r < 8]
    south_elev = [h.elevation for (_, r), h in state.hexes.items() if r >= 24]
    assert sum(north_elev) / len(north_elev) > sum(south_elev) / len(south_elev), (
        "Negative south gradient did not raise northern elevations above southern"
    )


def test_elevation_gradient_default_no_bias():
    """Default gradient (0, 0) should not skew east vs. west elevation."""
    cfg = WorldConfig(width=32, height=32, erosion_iterations=0, elevation_gradient=(0.0, 0.0))
    p = GeneratorPipeline(42, cfg)
    p.add_stage(ElevationStage)
    state = p.run()

    west_elev = [h.elevation for (q, _), h in state.hexes.items() if q < 8]
    east_elev = [h.elevation for (q, _), h in state.hexes.items() if q >= 24]
    diff = abs(sum(east_elev) / len(east_elev) - sum(west_elev) / len(west_elev))
    assert diff < 0.2, f"Default gradient produced unexpected east-west bias: {diff:.3f}"


def test_reproducible():
    cfg = WorldConfig(width=32, height=32, erosion_iterations=500)

    def build():
        p = GeneratorPipeline(7, cfg)
        p.add_stage(ElevationStage).add_stage(ErosionStage).add_stage(TerrainClassificationStage)
        return p.run()

    s1, s2 = build(), build()
    for coord in s1.hexes:
        assert s1.hexes[coord].elevation == s2.hexes[coord].elevation
        assert s1.hexes[coord].terrain_class == s2.hexes[coord].terrain_class
