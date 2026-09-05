import pytest

from worldgen.core.hex import (
    Biome,
    LandCover,
    Settlement,
    SettlementRole,
    SettlementTier,
    TerrainClass,
)
from worldgen.core.world_state import Ferry, River, Road, RoadTier, WorldState
from worldgen.export import json_export


def _small_world() -> WorldState:
    ws = WorldState.empty(seed=99, width=4, height=4)
    h = ws.hexes[(0, 0)]
    h.elevation = 0.5
    h.moisture = 0.3
    h.temperature = 0.6
    h.biome = Biome.GRASSLAND
    h.terrain_class = TerrainClass.LAND
    h.land_cover = LandCover.OPEN
    # Distinct per tier, so a round trip that collapsed them would show up.
    h.habitability_city = 0.7
    h.habitability_town = 0.5
    h.habitability_village = 0.3
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
    assert h.terrain_class == TerrainClass.LAND
    assert h.land_cover == LandCover.OPEN
    assert abs(h.habitability_city - 0.7) < 1e-9
    assert abs(h.habitability_town - 0.5) < 1e-9
    assert abs(h.habitability_village - 0.3) < 1e-9
    assert h.cultivated is True
    assert "test" in h.tags
    s2 = ws2.settlements[0]
    assert s2.name == "Ironhaven"
    assert s2.tier == SettlementTier.CITY
    assert s2.role == SettlementRole.MARKET
    assert s2.population == 5000


def _downgrade_to_v1_0(path):
    """Rewrite a saved world as the pre-split 1.0 schema."""
    import json

    data = json.loads(path.read_text())
    data["version"] = "1.0"
    for hd in data["hexes"]:
        for key in ("habitability_city", "habitability_town", "habitability_village"):
            hd.pop(key, None)
        hd["habitability"] = 0.6
    path.write_text(json.dumps(data))


def test_pre_split_habitability_still_loads(tmp_path):
    """Worlds saved before habitability was split per tier must still open."""
    ws = _small_world()
    path = tmp_path / "world.json"
    json_export.save(ws, path)
    _downgrade_to_v1_0(path)

    h = json_export.load(path).hexes[(0, 0)]
    assert h.habitability_city == 0.6
    assert h.habitability_town == 0.6
    assert h.habitability_village == 0.6


def test_new_saves_carry_the_bumped_version(tmp_path):
    """The schema changed shape, so it must not keep claiming to be 1.0."""
    import json

    path = tmp_path / "world.json"
    json_export.save(_small_world(), path)
    assert json.loads(path.read_text())["version"] == "1.4"


def test_an_unknown_version_is_rejected_by_name(tmp_path):
    import json

    import pytest

    path = tmp_path / "world.json"
    json_export.save(_small_world(), path)
    data = json.loads(path.read_text())
    data["version"] = "2.0"
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="Supported: 1.0, 1.1"):
        json_export.load(path)


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


def test_ferries_round_trip(tmp_path):
    ws = WorldState.empty(seed=5, width=3, height=3)
    ws.ferries = [Ferry(a=(0, 0), b=(2, 2))]
    path = tmp_path / "ferries.json"
    json_export.save(ws, str(path))
    ws2 = json_export.load(str(path))
    assert len(ws2.ferries) == 1
    assert ws2.ferries[0].a == (0, 0)
    assert ws2.ferries[0].b == (2, 2)


def test_worlds_without_ferries_still_load(tmp_path):
    """Worlds saved before ferries existed have no such key and must still load."""
    import json

    ws = WorldState.empty(seed=5, width=3, height=3)
    path = tmp_path / "legacy.json"
    json_export.save(ws, str(path))
    data = json.loads(path.read_text())
    del data["ferries"]
    path.write_text(json.dumps(data))
    assert json_export.load(str(path)).ferries == []


def test_an_older_world_translates_its_terrain_names(tmp_path):
    """Files written before 1.4 name water for a chemistry the generator never tracked.

    "ocean" and "lake" meant, and only ever meant, whether the water reaches the map edge.
    The steepness bands alongside them were thresholds on a slope those files do not
    record — so the class translates and the slope comes back 0.0, rather than being
    invented from the band the hex once fell in.
    """
    import json as _json

    ws = _small_world()
    path = tmp_path / "legacy.json"
    json_export.save(ws, str(path))

    raw = _json.loads(path.read_text())
    raw["version"] = "1.2"
    for hd in raw["hexes"]:
        hd.pop("slope", None)
        hd.pop("relief", None)
    raw["hexes"][0]["terrain_class"] = "ocean"
    raw["hexes"][1]["terrain_class"] = "lake"
    raw["hexes"][2]["terrain_class"] = "mountain"
    raw["hexes"][3]["terrain_class"] = "hill"
    path.write_text(_json.dumps(raw))

    back = json_export.load(str(path))
    by_coord = [back.hexes[(hd["q"], hd["r"])] for hd in raw["hexes"][:4]]
    assert by_coord[0].terrain_class is TerrainClass.OPEN_WATER
    assert by_coord[1].terrain_class is TerrainClass.INLAND_WATER
    assert by_coord[2].terrain_class is TerrainClass.LAND
    assert by_coord[3].terrain_class is TerrainClass.LAND
    assert all(h.slope == 0.0 for h in by_coord)
