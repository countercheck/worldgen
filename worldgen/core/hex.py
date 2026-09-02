from dataclasses import dataclass, field
from enum import Enum

HexCoord = tuple[int, int]


class TerrainClass(Enum):
    """How the ground lies underfoot — a gradient, not a landform.

    These used to be FLAT / HILL / MOUNTAIN, which read as altitude and were partly
    computed that way: anything above four fifths of the elevation range was called a
    mountain however level it was, so nearly a third of "mountain" hexes on a 128x128 map
    were gentler than 75 m/km.  A high plateau is walkable, ploughable ground and should
    not be priced like a peak.

    So every name here describes slope, and the thresholds are in metres of rise per
    kilometre rather than a fraction of the map's elevation range.  Absolute figures mean
    the same thing whatever `road_elev_range_m` is set to; the old fractions silently
    became nonsense at any other span, calling a 20 m/km rise a mountain on a 500 m map.

        FLAT        under 3%    level going: plough it, cart across it
        ROLLING     3 to 10%    undulating; farmed, and a laden cart manages
        STEEP       10 to 25%   pack animals, terraces, no wheels
        ESCARPMENT  over 25%    a break of slope; on foot and with effort

    Altitude still matters, but it is asked about where it belongs: `biome_alpine_elev`
    decides where the treeline is, and roles read `elevation` directly.
    """

    OCEAN = "ocean"
    LAKE = "lake"
    COAST = "coast"
    FLAT = "flat"
    ROLLING = "rolling"
    STEEP = "steep"
    ESCARPMENT = "escarpment"


# Ground too steep to plough, settle, or take a cart across. Several stages ask this
# question and they must agree: before ESCARPMENT existed each of them tested MOUNTAIN
# alone, and adding a steeper band without this would have quietly let settlements onto
# the steepest ground on the map.
STEEP_LAND = frozenset({TerrainClass.STEEP, TerrainClass.ESCARPMENT})


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
    # How much water passes, twice over. `river_flow` is normalised against the largest
    # accumulation on the map, so it is a rank — right for drawing, where width by rank
    # reads correctly, but not comparable between maps. `catchment_km2` is the upstream
    # area draining through this hex, which is a physical quantity and does compare: a
    # stream draining 30 km2 is the same stream whatever else is on the map.
    river_flow: float = 0.0
    catchment_km2: float = 0.0
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
    # Which settlement works this hex, and what it costs that settlement to reach it.
    # Catchments are allocated by travel cost rather than drawn as discs, so a ridge
    # between two settlements becomes a genuine watershed and the shapes follow valleys.
    # `territory_cost` is what weights the hex's contribution: ground at the far edge of a
    # catchment feeds its owner less than ground at the gate.
    territory: HexCoord | None = None
    territory_cost: float = 0.0
