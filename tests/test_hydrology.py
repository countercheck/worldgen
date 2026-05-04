import pytest

from worldgen.core.config import WorldConfig
from worldgen.core.hex import TerrainClass
from worldgen.core.hex_grid import neighbors
from worldgen.core.pipeline import GeneratorPipeline
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
    return p


@pytest.fixture(scope="module")
def hydro_state():
    return _build_pipeline().run()


def test_river_flow_nonzero(hydro_state):
    river_hexes = [h for h in hydro_state.hexes.values() if h.river_flow > 0]
    assert len(river_hexes) > 0, "No river hexes found after hydrology stage"


def test_river_flow_normalized(hydro_state):
    for h in hydro_state.hexes.values():
        assert 0.0 <= h.river_flow <= 1.0, f"river_flow {h.river_flow} out of [0, 1]"


def test_river_paths_connected(hydro_state):
    for river in hydro_state.rivers:
        assert len(river.hexes) >= 2, "River has fewer than 2 hexes"
        for i in range(len(river.hexes) - 1):
            a, b = river.hexes[i], river.hexes[i + 1]
            assert b in neighbors(a), f"Non-adjacent hexes in river path: {a} -> {b}"


def test_rivers_reach_ocean(hydro_state):
    ocean_set = {
        coord for coord, h in hydro_state.hexes.items() if h.terrain_class == TerrainClass.OCEAN
    }
    w, h = hydro_state.width, hydro_state.height
    for river in hydro_state.rivers:
        mouth = river.hexes[-1]
        q, r = mouth
        on_border = q == 0 or q == w - 1 or r == 0 or r == h - 1
        reaches_ocean = any(n in ocean_set for n in neighbors(mouth)) or mouth in ocean_set
        assert reaches_ocean or on_border, (
            f"River mouth {mouth} does not reach ocean or grid border"
        )


def test_flow_accumulates_downstream(hydro_state):
    # Flow accumulation (river_flow) must be non-decreasing along a river path —
    # each step downstream collects more water. Checks the accumulation invariant
    # without depending on the filled vs. actual elevation distinction.
    for river in hydro_state.rivers:
        river_hexes = [
            hydro_state.hexes[c]
            for c in river.hexes
            if c in hydro_state.hexes and hydro_state.hexes[c].river_flow > 0
        ]
        for i in range(len(river_hexes) - 1):
            flow_a = river_hexes[i].river_flow
            flow_b = river_hexes[i + 1].river_flow
            assert flow_b >= flow_a - 1e-9, (
                f"River_flow decreases downstream: {flow_a:.4f} -> {flow_b:.4f}"
            )


def test_tags_assigned(hydro_state):
    all_tags: set[str] = set()
    for h in hydro_state.hexes.values():
        all_tags.update(h.tags)
    assert "headwater" in all_tags, "No headwater tags found"
    assert "river_mouth" in all_tags, "No river_mouth tags found"


def test_reproducibility():
    s1 = _build_pipeline(seed=7).run()
    s2 = _build_pipeline(seed=7).run()
    for coord in s1.hexes:
        assert s1.hexes[coord].river_flow == s2.hexes[coord].river_flow, (
            f"river_flow differs at {coord} between identical seeds"
        )
    assert len(s1.rivers) == len(s2.rivers)
