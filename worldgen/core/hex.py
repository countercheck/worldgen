from dataclasses import dataclass, field
from enum import Enum

HexCoord = tuple[int, int]


class TerrainClass(Enum):
    OCEAN = "ocean"
    COAST = "coast"
    FLAT = "flat"
    HILL = "hill"
    MOUNTAIN = "mountain"


class LandCover(Enum):
    OPEN_WATER = "open_water"
    BOG = "bog"
    MARSH = "marsh"
    DENSE_FOREST = "dense_forest"
    WOODLAND = "woodland"
    SCRUB = "scrub"
    OPEN = "open"
    TUNDRA = "tundra"
    DESERT = "desert"
    ALPINE = "alpine"
    BARE_ROCK = "bare_rock"


class Biome(Enum):
    TUNDRA = "tundra"
    BOREAL = "boreal"
    TEMPERATE_FOREST = "temperate_forest"
    GRASSLAND = "grassland"
    SHRUBLAND = "shrubland"
    DESERT = "desert"
    TROPICAL = "tropical"
    WETLAND = "wetland"
    OCEAN = "ocean"
    ALPINE = "alpine"


@dataclass
class Settlement:
    coord: HexCoord
    tier: "SettlementTier"
    role: "SettlementRole"
    population: int
    name: str


class SettlementTier(Enum):
    CITY = "city"
    TOWN = "town"
    VILLAGE = "village"


class SettlementRole(Enum):
    AGRICULTURAL = "agricultural"
    PORT = "port"
    MINING = "mining"
    FORTRESS = "fortress"
    MARKET = "market"


@dataclass
class Hex:
    coord: HexCoord
    elevation: float = 0.0
    moisture: float = 0.0
    temperature: float = 0.0
    biome: Biome | None = None
    river_flow: float = 0.0
    terrain_class: TerrainClass = TerrainClass.FLAT
    settlement: Settlement | None = None
    road_connections: set[HexCoord] = field(default_factory=set)
    tags: set[str] = field(default_factory=set)
    habitability: float = 0.0
    land_cover: LandCover | None = None
    cultivated: bool = False
