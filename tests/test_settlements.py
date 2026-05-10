import pytest

from worldgen.core.config import WorldConfig
from worldgen.core.hex import Biome, Hex, SettlementRole, SettlementTier, TerrainClass
from worldgen.core.pipeline import GeneratorPipeline
from worldgen.stages.biomes import BiomeStage
from worldgen.stages.city_town import _assign_role as assign_city_town_role
from worldgen.stages.climate import ClimateStage
from worldgen.stages.elevation import ElevationStage
from worldgen.stages.erosion import ErosionStage
from worldgen.stages.habitability import HabitabilityStage
from worldgen.stages.hydrology import HydrologyStage
from worldgen.stages.roads import RoadStage
from worldgen.stages.settlements import SettlementStage
from worldgen.stages.settlements import _assign_role as assign_settlement_role
from worldgen.stages.terrain_class import TerrainClassificationStage


def _build_pipeline(seed: int = 42, width: int = 64, height: int = 64):
    cfg = WorldConfig(
        width=width,
        height=height,
        erosion_iterations=500,
        target_city_count=4,
        target_town_count=12,
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
    return p


@pytest.fixture(scope="module")
def settle_state():
    return _build_pipeline().run()


def test_has_settlements(settle_state):
    assert len(settle_state.settlements) >= 1


def test_has_each_tier(settle_state):
    tiers = {s.tier for s in settle_state.settlements}
    assert SettlementTier.CITY in tiers, "No cities placed"
    assert SettlementTier.TOWN in tiers, "No towns placed"
    assert SettlementTier.VILLAGE in tiers, "No villages placed"


def test_settlements_on_land(settle_state):
    for s in settle_state.settlements:
        hx = settle_state.hexes[s.coord]
        assert hx.terrain_class != TerrainClass.OCEAN, f"Settlement {s.name} placed on ocean hex"


def test_city_separation(settle_state):
    from worldgen.core.hex_grid import distance

    cities = [s for s in settle_state.settlements if s.tier == SettlementTier.CITY]
    cfg_sep = settle_state.metadata["config"]["city_min_separation"]
    for i, a in enumerate(cities):
        for b in cities[i + 1 :]:
            d = distance(a.coord, b.coord)
            assert d >= cfg_sep, f"Cities {a.name} and {b.name} too close: {d} < {cfg_sep}"


def test_town_separation(settle_state):
    from worldgen.core.hex_grid import distance

    towns = [s for s in settle_state.settlements if s.tier == SettlementTier.TOWN]
    cfg_sep = settle_state.metadata["config"]["town_min_separation"]
    for i, a in enumerate(towns):
        for b in towns[i + 1 :]:
            d = distance(a.coord, b.coord)
            assert d >= cfg_sep, f"Towns {a.name} and {b.name} too close: {d} < {cfg_sep}"


def test_village_separation(settle_state):
    from worldgen.core.hex_grid import distance

    villages = [s for s in settle_state.settlements if s.tier == SettlementTier.VILLAGE]
    for i, a in enumerate(villages):
        for b in villages[i + 1 :]:
            d = distance(a.coord, b.coord)
            assert d >= 3, f"Villages {a.name} and {b.name} too close: {d} < 3"


def test_hex_settlement_backref(settle_state):
    for s in settle_state.settlements:
        hx = settle_state.hexes[s.coord]
        assert hx.settlement is s, f"hex at {s.coord} does not reference settlement {s.name}"


def test_valid_tiers_and_roles(settle_state):
    for s in settle_state.settlements:
        assert isinstance(s.tier, SettlementTier)
        assert isinstance(s.role, SettlementRole)


def test_positive_population(settle_state):
    for s in settle_state.settlements:
        assert s.population > 0


def test_reproducibility():
    s1 = _build_pipeline(seed=13).run()
    s2 = _build_pipeline(seed=13).run()
    names1 = sorted(s.name for s in s1.settlements)
    names2 = sorted(s.name for s in s2.settlements)
    assert names1 == names2, "Settlement names differ between identical seeds"
    coords1 = sorted(s.coord for s in s1.settlements)
    coords2 = sorted(s.coord for s in s2.settlements)
    assert coords1 == coords2, "Settlement coords differ between identical seeds"


@pytest.mark.parametrize(
    "assign_role",
    [assign_settlement_role, assign_city_town_role, RoadStage._assign_role_simple],
)
def test_port_role_requires_river_tag(assign_role):
    def call_assign_role():
        if assign_role is RoadStage._assign_role_simple:
            return assign_role(None, center.coord, center, hexes)
        return assign_role(center.coord, center, hexes)

    center = Hex(coord=(0, 0), biome=Biome.GRASSLAND)
    river_neighbor = Hex(coord=(1, 0), biome=Biome.GRASSLAND, river_flow=1.0)
    hexes = {center.coord: center, river_neighbor.coord: river_neighbor}

    assert call_assign_role() is not SettlementRole.PORT

    river_neighbor.tags.add("river")
    assert call_assign_role() is SettlementRole.PORT
