import pytest

from worldgen.core.config import WorldConfig
from worldgen.core.hex import LandCover, SettlementTier
from worldgen.core.hex_grid import distance
from worldgen.core.pipeline import GeneratorPipeline
from worldgen.stages.biomes import BiomeStage
from worldgen.stages.city_town import CityTownStage
from worldgen.stages.climate import ClimateStage
from worldgen.stages.cultivation import CultivationStage, VillageCultivationStage
from worldgen.stages.elevation import ElevationStage
from worldgen.stages.erosion import ErosionStage
from worldgen.stages.habitability import HabitabilityStage
from worldgen.stages.hydrology import HydrologyStage
from worldgen.stages.interurban_roads import InterurbanRoadStage
from worldgen.stages.land_cover import LandCoverStage
from worldgen.stages.terrain_class import TerrainClassificationStage
from worldgen.stages.village_placement import VillagePlacementStage
from worldgen.stages.village_tracks import VillageTrackStage

_RESISTANT = {
    LandCover.BOG,
    LandCover.MARSH,
    LandCover.BARE_ROCK,
    LandCover.ALPINE,
    LandCover.TUNDRA,
    LandCover.DESERT,
    LandCover.OPEN_WATER,
}


def _build_pipeline(seed: int = 42, width: int = 64, height: int = 64):
    cfg = WorldConfig(
        width=width,
        height=height,
        erosion_iterations=500,
        target_city_count=3,
        target_town_count=8,
        road_travellers_city=100,
        road_travellers_town=20,
        cultivation_city_radius=6,
        cultivation_town_radius=3,
        cultivation_village_radius=2,
    )
    p = GeneratorPipeline(seed, cfg)
    p.add_stage(ElevationStage)
    p.add_stage(ErosionStage)
    p.add_stage(TerrainClassificationStage)
    p.add_stage(HydrologyStage)
    p.add_stage(ClimateStage)
    p.add_stage(BiomeStage)
    p.add_stage(LandCoverStage)
    p.add_stage(HabitabilityStage)
    p.add_stage(CityTownStage)
    p.add_stage(InterurbanRoadStage)
    p.add_stage(CultivationStage)
    p.add_stage(VillagePlacementStage)
    p.add_stage(VillageTrackStage)
    p.add_stage(VillageCultivationStage)
    return p


@pytest.fixture(scope="module")
def cult_state():
    return _build_pipeline().run()


def test_default_uncultivated(cult_state):
    """Most land hexes should be uncultivated (wilderness)."""
    land = [h for h in cult_state.hexes.values() if h.land_cover != LandCover.OPEN_WATER]
    cultivated_count = sum(1 for h in land if h.cultivated)
    # Wilderness should dominate — cultivated should be a minority
    assert cultivated_count < len(land), "All land is cultivated — expected wilderness"


def test_resistant_hexes_never_cultivated(cult_state):
    for coord, h in cult_state.hexes.items():
        if h.land_cover in _RESISTANT:
            assert not h.cultivated, (
                f"Resistant hex {coord} (land_cover={h.land_cover}) was marked cultivated"
            )


def test_cities_have_cultivation_nearby(cult_state):
    hexes = cult_state.hexes
    cities = [s for s in cult_state.settlements if s.tier == SettlementTier.CITY]
    for city in cities:
        nearby = [
            hexes[n]
            for n in hexes
            if distance(n, city.coord) <= 6 and hexes[n].land_cover not in _RESISTANT
        ]
        cultivated_nearby = [h for h in nearby if h.cultivated]
        assert cultivated_nearby, f"No cultivated hexes near city at {city.coord}"


def test_villages_on_frontier_or_road(cult_state):
    hexes = cult_state.hexes
    from worldgen.core.hex_grid import neighbors as nbrs

    villages = [s for s in cult_state.settlements if s.tier == SettlementTier.VILLAGE]
    for v in villages:
        hx = hexes[v.coord]
        # Village should be: cultivated with uncultivated neighbor (frontier), OR road-adjacent
        on_frontier = hx.cultivated and any(
            not hexes[n].cultivated
            for n in nbrs(v.coord)
            if n in hexes and hexes[n].land_cover != LandCover.OPEN_WATER
        )
        road_adjacent = bool(hx.road_connections) or any(
            hexes[n].road_connections for n in nbrs(v.coord) if n in hexes
        )
        assert on_frontier or road_adjacent, (
            f"Village at {v.coord} is neither on cultivation frontier nor road-adjacent"
        )


def test_village_separation(cult_state):
    villages = [s for s in cult_state.settlements if s.tier == SettlementTier.VILLAGE]
    for i, a in enumerate(villages):
        for b in villages[i + 1 :]:
            d = distance(a.coord, b.coord)
            assert d >= 3, f"Villages {a.name} and {b.name} too close: {d} < 3"


def test_villages_have_track_connection(cult_state):
    """Every village must be an endpoint of a TRACK road connecting it to the road network."""
    from worldgen.core.world_state import RoadTier

    track_endpoints: set = set()
    for road in cult_state.roads:
        if road.tier == RoadTier.TRACK and len(road.path) >= 2:
            track_endpoints.add(road.path[0])
            track_endpoints.add(road.path[-1])

    villages = [s for s in cult_state.settlements if s.tier == SettlementTier.VILLAGE]
    for v in villages:
        assert v.coord in track_endpoints, (
            f"Village at {v.coord} is not an endpoint of any TRACK road — "
            "VillageTrackStage may have failed to connect it"
        )


def test_has_all_tiers(cult_state):
    tiers = {s.tier for s in cult_state.settlements}
    assert SettlementTier.CITY in tiers
    assert SettlementTier.TOWN in tiers
    assert SettlementTier.VILLAGE in tiers


def test_reproducibility():
    s1 = _build_pipeline(seed=55).run()
    s2 = _build_pipeline(seed=55).run()
    cult1 = sorted(c for c, h in s1.hexes.items() if h.cultivated)
    cult2 = sorted(c for c, h in s2.hexes.items() if h.cultivated)
    assert cult1 == cult2, "Cultivated hexes differ between identical seeds"
