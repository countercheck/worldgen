from dataclasses import dataclass, field
from enum import Enum

HexCoord = tuple[int, int]


class TerrainClass(Enum):
    """What kind of place a hex is, where the kinds are genuinely different kinds.

    Water is not steep land with the water turned up, and a shore is a fact about what a
    hex adjoins — those are categories.  So is whether a body of water drains off the map,
    which is the whole of what OPEN_WATER and INLAND_WATER distinguish.  Steepness is not: it is a continuum, and the
    FLAT/HILL/MOUNTAIN bands that used to live here were thresholds on `Hex.slope` that
    six stages read in place of the terrain.  A hex fell either side of a cutoff for
    reasons unrelated to the question being asked of it, which is how a level floodplain
    beside a bluff came out classified as mountain.  Ask `slope` and `relief` instead.

    The words survive where they belong: `terrain_label` derives them for maps and
    legends, which is a presentation concern, so nothing in the pipeline branches on them.
    """

    # Water that reaches the map edge, and so carries what enters it away.  Called
    # "ocean" once, which claimed a salinity nothing tracks; what the generator actually
    # knows is that this body drains off the map, and that is what every use of it means
    # — the outlet priority-flood seeds from, the terminal a river may end at, the thing a
    # basin is *not*.
    OPEN_WATER = "open_water"
    # Water with no way off the map.  A basin, in other words: it fills to its spillway
    # and overflows, or evaporates what reaches it and is closed.  Fresh or salt is not
    # the question — the Caspian is the second of these and salt, Baikal the second and
    # fresh.
    INLAND_WATER = "inland_water"
    COAST = "coast"
    LAND = "land"


class TerrainLabel(Enum):
    """The vocabulary a map is drawn and read in — not one the generator reasons in.

    A GM wants to see mountains and hills, and a wargame's movement rules are written in
    those words, so the bands are worth keeping at the edge.  Deriving them here rather
    than storing them on the hex is what stops them becoming load-bearing again.

    Ocean and lake stay in this vocabulary for the same reason: they are the words an
    atlas uses, and a reader wants them.  It is only the *generator* that has no business
    claiming a chemistry it does not model.
    """

    OCEAN = "ocean"
    LAKE = "lake"
    COAST = "coast"
    FLAT = "flat"
    HILL = "hill"
    MOUNTAIN = "mountain"


def terrain_label(hx, hill_gradient: float, mountain_gradient: float) -> TerrainLabel:
    """Band a hex's steepness into the word a reader expects to see on the map."""
    if hx.terrain_class is TerrainClass.OPEN_WATER:
        return TerrainLabel.OCEAN
    if hx.terrain_class is TerrainClass.INLAND_WATER:
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
    # Depth of loose river-laid sediment, 0 to 1 against the map's own richest ground.
    # Not a terrain type: a floodplain is ordinary land that happens to be floored with
    # silt, and it can be forest or grass or ploughed like any other.  What it is *not*
    # is a fact about slope — flat ground is not alluvial and alluvium is not always
    # flat — which is why this is measured from where the sediment went rather than
    # inferred from the shape of the ground it went to.
    alluvium: float = 0.0
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
