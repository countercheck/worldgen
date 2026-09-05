from dataclasses import dataclass, field
from enum import Enum

HexCoord = tuple[int, int]


class TerrainClass(Enum):
    """What kind of place a hex is, where the kinds are genuinely different kinds.

    Water is not steep land with the water turned up, and a shore is a fact about what a
    hex adjoins — those are categories.  Steepness is not: it is a continuum, and the
    FLAT/HILL/MOUNTAIN bands that used to live here were thresholds on `Hex.slope` that
    six stages read in place of the terrain.  A hex fell either side of a cutoff for
    reasons unrelated to the question being asked of it, which is how a level floodplain
    beside a bluff came out classified as mountain.  Ask `slope` and `relief` instead.

    The words survive where they belong: `terrain_label` derives them for maps and
    legends, which is a presentation concern, so nothing in the pipeline branches on them.
    """

    OCEAN = "ocean"
    LAKE = "lake"
    COAST = "coast"
    LAND = "land"


class TerrainLabel(Enum):
    """The vocabulary a map is drawn and read in — not one the generator reasons in.

    A GM wants to see mountains and hills, and a wargame's movement rules are written in
    those words, so the bands are worth keeping at the edge.  Deriving them here rather
    than storing them on the hex is what stops them becoming load-bearing again.
    """

    OCEAN = "ocean"
    LAKE = "lake"
    COAST = "coast"
    FLAT = "flat"
    HILL = "hill"
    MOUNTAIN = "mountain"


def terrain_label(hx, hill_gradient: float, mountain_gradient: float) -> TerrainLabel:
    """Band a hex's steepness into the word a reader expects to see on the map."""
    if hx.terrain_class is TerrainClass.OCEAN:
        return TerrainLabel.OCEAN
    if hx.terrain_class is TerrainClass.LAKE:
        return TerrainLabel.LAKE
    if hx.terrain_class is TerrainClass.COAST:
        return TerrainLabel.COAST
    if hx.slope > mountain_gradient:
        return TerrainLabel.MOUNTAIN
    if hx.slope >= hill_gradient:
        return TerrainLabel.HILL
    return TerrainLabel.FLAT


def terrain_labels(ws) -> dict:
    """Every hex's map label, using the thresholds the world was generated with."""
    cfg = ws.metadata.get("config", {})
    hill = cfg.get("terrain_hill_gradient", 0.02)
    mountain = cfg.get("terrain_mountain_gradient", 0.04)
    return {coord: terrain_label(h, hill, mountain) for coord, h in ws.hexes.items()}


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
    terrain_class: TerrainClass = TerrainClass.LAND
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
