import pytest

from worldgen.core.config import WorldConfig
from worldgen.core.pipeline import GeneratorPipeline
from worldgen.stages.biomes import BiomeStage
from worldgen.stages.city_town import CityTownStage
from worldgen.stages.climate import ClimateStage
from worldgen.stages.cultivation import CultivationStage
from worldgen.stages.elevation import ElevationStage
from worldgen.stages.erosion import ErosionStage
from worldgen.stages.habitability import HabitabilityStage
from worldgen.stages.hydrology import HydrologyStage
from worldgen.stages.interurban_roads import InterurbanRoadStage
from worldgen.stages.land_cover import LandCoverStage
from worldgen.stages.terrain_class import TerrainClassificationStage


@pytest.fixture(scope="module")
def small_state():
    cfg = WorldConfig(
        width=32,
        height=32,
        erosion_iterations=200,
        target_city_count=2,
        target_town_count=4,
        road_travellers_city=50,
        road_travellers_town=10,
        road_travellers_village=2,
    )
    p = GeneratorPipeline(42, cfg)
    (
        p.add_stage(ElevationStage)
        .add_stage(ErosionStage)
        .add_stage(TerrainClassificationStage)
        .add_stage(HydrologyStage)
        .add_stage(ClimateStage)
        .add_stage(BiomeStage)
        .add_stage(LandCoverStage)
        .add_stage(HabitabilityStage)
        .add_stage(CityTownStage)
        .add_stage(InterurbanRoadStage)
    )
    return p.run()


@pytest.fixture(scope="module")
def land_cover_state():
    cfg = WorldConfig(
        width=32,
        height=32,
        erosion_iterations=200,
        target_city_count=2,
        target_town_count=4,
        road_travellers_city=50,
        road_travellers_town=10,
    )
    p = GeneratorPipeline(42, cfg)
    (
        p.add_stage(ElevationStage)
        .add_stage(ErosionStage)
        .add_stage(TerrainClassificationStage)
        .add_stage(HydrologyStage)
        .add_stage(ClimateStage)
        .add_stage(BiomeStage)
        .add_stage(LandCoverStage)
        .add_stage(HabitabilityStage)
        .add_stage(CityTownStage)
        .add_stage(InterurbanRoadStage)
        .add_stage(CultivationStage)
    )
    return p.run()


def test_render_roads_produces_file(small_state, tmp_path):
    from worldgen.render.debug_viewer import render

    out = tmp_path / "roads.svg"
    render(small_state, "roads", str(out))
    assert out.exists() and out.stat().st_size > 0
    assert out.read_text().startswith("<svg")


def test_render_unknown_attribute_raises(small_state, tmp_path):
    from worldgen.render.debug_viewer import render

    with pytest.raises(ValueError, match="Unknown attribute"):
        render(small_state, "nonexistent", str(tmp_path / "x.svg"))


def test_render_land_cover_produces_file(land_cover_state, tmp_path):
    from worldgen.render.debug_viewer import render

    out = tmp_path / "land_cover.svg"
    render(land_cover_state, "land_cover", str(out))
    assert out.exists() and out.stat().st_size > 0


def test_render_cultivation_produces_file(land_cover_state, tmp_path):
    from worldgen.render.debug_viewer import render

    out = tmp_path / "cultivation.svg"
    render(land_cover_state, "cultivation", str(out))
    assert out.exists() and out.stat().st_size > 0


def test_debug_viewer_paints_the_primary_road_over_the_branching_track():
    """Iterating RoadTier drew tracks last, so a branch painted over its own trunk."""
    import re

    from worldgen.core.world_state import Road, RoadTier, WorldState
    from worldgen.render.debug_viewer import render_svg

    ws = WorldState.empty(seed=1, width=5, height=3)
    ws.roads = [
        Road(path=[(0, 1), (1, 1), (2, 1), (3, 1)], tier=RoadTier.PRIMARY),
        Road(path=[(0, 1), (1, 1), (2, 1), (2, 2)], tier=RoadTier.TRACK),
    ]
    body = render_svg(ws, "roads").split('<g id="layer-roads">')[1].split("</g>")[0]

    assert body.index('stroke="#b8a070"') < body.index('stroke="#5c3d1e"')
    # The shared trunk is drawn once, by the primary road only.
    polylines = re.findall(r'<polyline points="([^"]+)"', body)
    assert len(polylines) == 2
    edges = [
        frozenset((a, b))
        for pts in (p.split() for p in polylines)
        for a, b in zip(pts, pts[1:], strict=False)
    ]
    assert len(edges) == len(set(edges)), "an edge was drawn twice"
