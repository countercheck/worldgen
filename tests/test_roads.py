import pytest

from worldgen.core.config import WorldConfig
from worldgen.core.hex import SettlementTier, TerrainClass
from worldgen.core.pipeline import GeneratorPipeline
from worldgen.core.world_state import RoadTier
from worldgen.stages.biomes import BiomeStage
from worldgen.stages.climate import ClimateStage
from worldgen.stages.elevation import ElevationStage
from worldgen.stages.erosion import ErosionStage
from worldgen.stages.habitability import HabitabilityStage
from worldgen.stages.hydrology import HydrologyStage
from worldgen.stages.roads import RoadStage
from worldgen.stages.settlements import SettlementStage
from worldgen.stages.terrain_class import TerrainClassificationStage


def _build_pipeline(seed: int = 42, width: int = 64, height: int = 64):
    cfg = WorldConfig(
        width=width,
        height=height,
        erosion_iterations=500,
        target_city_count=4,
        target_town_count=10,
        road_travellers_city=100,
        road_travellers_town=20,
        road_travellers_village=5,
    )
    p = GeneratorPipeline(seed, cfg)
    p.add_stage(ElevationStage)
    p.add_stage(ErosionStage)
    p.add_stage(TerrainClassificationStage)
    p.add_stage(HydrologyStage)
    p.add_stage(ClimateStage)
    p.add_stage(BiomeStage)
    p.add_stage(HabitabilityStage)
    p.add_stage(SettlementStage)
    p.add_stage(RoadStage)
    return p


@pytest.fixture(scope="module")
def road_state():
    return _build_pipeline().run()


def test_has_roads(road_state):
    assert len(road_state.roads) >= 1


def test_road_paths_min_length(road_state):
    for road in road_state.roads:
        assert len(road.path) >= 2, f"Road has path length {len(road.path)}"


def test_road_paths_connected(road_state):
    from worldgen.core.hex_grid import distance

    for road in road_state.roads:
        for a, b in zip(road.path, road.path[1:], strict=False):
            assert distance(a, b) == 1, f"Non-adjacent coords in road path: {a} -> {b}"


def test_no_ocean_in_roads(road_state):
    for road in road_state.roads:
        for c in road.path:
            hx = road_state.hexes[c]
            assert hx.terrain_class != TerrainClass.OCEAN, f"Road passes through ocean hex {c}"


def test_road_connections_symmetric(road_state):
    hexes = road_state.hexes
    for coord, hx in hexes.items():
        for neighbor in hx.road_connections:
            assert coord in hexes[neighbor].road_connections, (
                f"road_connections not symmetric: {coord} -> {neighbor} but not reverse"
            )


def test_river_hexes_in_roads_tagged(road_state):
    road_hexes = {c for road in road_state.roads for c in road.path}
    for c in road_hexes:
        hx = road_state.hexes[c]
        if hx.river_flow > 0:
            assert "ford" in hx.tags or "bridge" in hx.tags, (
                f"River hex {c} on road not tagged ford/bridge"
            )


def test_valid_road_tiers(road_state):
    for road in road_state.roads:
        assert isinstance(road.tier, RoadTier)


def test_cities_mutually_reachable(road_state):
    from collections import deque

    hexes = road_state.hexes
    cities = [s for s in road_state.settlements if s.tier == SettlementTier.CITY]
    if len(cities) <= 1:
        return

    # BFS over road_connections
    start = cities[0].coord
    visited = {start}
    queue = deque([start])
    while queue:
        c = queue.popleft()
        for n in hexes[c].road_connections:
            if n not in visited:
                visited.add(n)
                queue.append(n)

    for city in cities[1:]:
        assert city.coord in visited, (
            f"City {city.name} at {city.coord} not reachable via road network"
        )


def test_river_preference_in_roads(road_state):
    hexes = road_state.hexes
    road_hexes = {c for road in road_state.roads for c in road.path if c in hexes}
    all_land = {c for c, h in hexes.items() if h.terrain_class != TerrainClass.OCEAN}

    if not road_hexes or not all_land:
        return

    river_in_roads = sum(1 for c in road_hexes if hexes[c].river_flow > 0)
    river_in_map = sum(1 for c in all_land if hexes[c].river_flow > 0)

    road_river_rate = river_in_roads / len(road_hexes) if road_hexes else 0
    map_river_rate = river_in_map / len(all_land) if all_land else 0

    assert road_river_rate >= map_river_rate, (
        f"River preference not detected: road river rate {road_river_rate:.3f} < "
        f"map river rate {map_river_rate:.3f}"
    )


def test_reproducibility():
    s1 = _build_pipeline(seed=99).run()
    s2 = _build_pipeline(seed=99).run()
    tiers1 = sorted((r.tier.value, tuple(r.path)) for r in s1.roads)
    tiers2 = sorted((r.tier.value, tuple(r.path)) for r in s2.roads)
    assert tiers1 == tiers2, "Roads differ between identical seeds"
