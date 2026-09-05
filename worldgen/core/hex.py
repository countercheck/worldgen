from dataclasses import dataclass, field
from enum import Enum

HexCoord = tuple[int, int]


class TerrainClass(Enum):
    OCEAN = "ocean"
    LAKE = "lake"
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
    # How steep the ground is: the mean elevation difference to the neighbours.  Kept as
    # the number rather than thresholded into hill and mountain, because steepness is a
    # continuum and the thresholds were load-bearing in six places — which is how a flat
    # floodplain beside a bluff came out classified as mountain, and scored as unfarmable
    # ground a road should climb around.
    slope: float = 0.0
    # How far this hex stands above the lowest ground touching it.  Steepness alone cannot
    # express a site "overlooking a plain": what makes that site good is the drop it
    # commands, not the gradient it sits on.
    relief: float = 0.0
    terrain_class: TerrainClass = TerrainClass.FLAT
    settlement: Settlement | None = None
    road_connections: set[HexCoord] = field(default_factory=set)
    tags: set[str] = field(default_factory=set)
    # Site quality at each tier's catchment radius — a city draws on a far wider
    # hinterland than a village, so the same hex scores differently for each.
    habitability_city: float = 0.0
    habitability_town: float = 0.0
    habitability_village: float = 0.0
    land_cover: LandCover | None = None
    cultivated: bool = False
