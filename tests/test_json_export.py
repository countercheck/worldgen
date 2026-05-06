import pytest

from worldgen.core.hex import (
    Biome,
    LandCover,
    Settlement,
    SettlementRole,
    SettlementTier,
    TerrainClass,
)
from worldgen.core.world_state import River, Road, RoadTier, WorldState
from worldgen.export import json_export


def _small_world() -> WorldState:
    ws = WorldState.empty(seed=99, width=4, height=4)
    h = ws.hexes[(0, 0)]
    h.elevation = 0.5
    h.moisture = 0.3
    h.temperature = 0.6
    h.biome = Biome.GRASSLAND
    h.terrain_class = TerrainClass.FLAT
    h.land_cover = LandCover.OPEN
    h.habitability = 0.7
    h.cultivated = True
    h.tags = {"test"}
    ws.settlements = [
        Settlement(
            coord=(1, 1),
            tier=SettlementTier.CITY,
            role=SettlementRole.MARKET,
            population=5000,
            name="Ironhaven",
        )
    ]
    ws.hexes[(1, 1)].settlement = ws.settlements[0]
    ws.hexes[(1, 1)].road_connections = {(2, 1)}
    ws.rivers = [River(hexes=[(0, 0), (1, 0), (2, 0)], flow_volume=1.5)]
    ws.roads = [Road(path=[(1, 1), (2, 1), (3, 1)], tier=RoadTier.PRIMARY)]
    return ws


def test_round_trip(tmp_path):
    ws = _small_world()
    path = tmp_path / "world.json"
    json_export.save(ws, path)
    ws2 = json_export.load(path)
    assert ws2.seed == ws.seed
    assert ws2.width == ws.width
    assert ws2.height == ws.height
    assert len(ws2.hexes) == len(ws.hexes)
    assert len(ws2.settlements) == len(ws.settlements)
    assert len(ws2.rivers) == len(ws.rivers)
    assert len(ws2.roads) == len(ws.roads)


def test_hex_fields_preserved(tmp_path):
    ws = _small_world()
    path = tmp_path / "world.json"
    json_export.save(ws, path)
    ws2 = json_export.load(path)
    h = ws2.hexes[(0, 0)]
    assert abs(h.elevation - 0.5) < 1e-9
    assert abs(h.moisture - 0.3) < 1e-9
    assert h.biome == Biome.GRASSLAND
    assert h.terrain_class == TerrainClass.FLAT
    assert h.land_cover == LandCover.OPEN
    assert h.cultivated is True
    assert "test" in h.tags
    s2 = ws2.settlements[0]
    assert s2.name == "Ironhaven"
    assert s2.tier == SettlementTier.CITY
    assert s2.role == SettlementRole.MARKET
    assert s2.population == 5000


def test_settlement_linked_on_hex(tmp_path):
    ws = _small_world()
    path = tmp_path / "world.json"
    json_export.save(ws, path)
    ws2 = json_export.load(path)
    assert ws2.hexes[(1, 1)].settlement is not None
    assert ws2.hexes[(1, 1)].settlement.name == "Ironhaven"


def test_road_connections_preserved(tmp_path):
    ws = _small_world()
    path = tmp_path / "world.json"
    json_export.save(ws, path)
    ws2 = json_export.load(path)
    assert (2, 1) in ws2.hexes[(1, 1)].road_connections


def test_river_preserved(tmp_path):
    ws = _small_world()
    path = tmp_path / "world.json"
    json_export.save(ws, path)
    ws2 = json_export.load(path)
    assert len(ws2.rivers) == 1
    assert ws2.rivers[0].flow_volume == pytest.approx(1.5)
    assert ws2.rivers[0].hexes[0] == (0, 0)


def test_road_tier_preserved(tmp_path):
    ws = _small_world()
    path = tmp_path / "world.json"
    json_export.save(ws, path)
    ws2 = json_export.load(path)
    assert ws2.roads[0].tier == RoadTier.PRIMARY


def test_empty_world(tmp_path):
    ws = WorldState(seed=1, width=2, height=2)
    path = tmp_path / "empty.json"
    json_export.save(ws, path)
    ws2 = json_export.load(path)
    assert ws2.seed == 1
    assert ws2.width == 2
    assert len(ws2.hexes) == 0
    assert len(ws2.settlements) == 0


def test_from_json_classmethod(tmp_path):
    ws = _small_world()
    path = tmp_path / "world.json"
    json_export.save(ws, path)
    ws2 = WorldState.from_json(str(path))
    assert ws2.seed == ws.seed
    assert len(ws2.hexes) == len(ws.hexes)


def test_none_biome_and_land_cover(tmp_path):
    ws = WorldState.empty(seed=7, width=2, height=2)
    # hexes default to biome=None, land_cover=None
    path = tmp_path / "world.json"
    json_export.save(ws, path)
    ws2 = json_export.load(path)
    h = ws2.hexes[(0, 0)]
    assert h.biome is None
    assert h.land_cover is None
