from dataclasses import dataclass, field
from enum import Enum

HexCoord = tuple[int, int]


class TerrainClass(Enum):
    """What kind of place a hex is, where the kinds are genuinely different kinds.

    Water is not steep land with the water turned up, and a shore is a fact about what a
    hex adjoins — those are categories.  So is whether a body of water drains off the map,
    which is the whole of what OPEN_WATER and INLAND_WATER distinguish.

    Steepness is not a category.  It is a continuum, and the FLAT / ROLLING / STEEP /
    ESCARPMENT bands that used to live here were thresholds on `Hex.slope` that six stages
    read in place of the terrain.  A hex fell either side of a cutoff for reasons unrelated
    to the question being asked of it, which is how a level floodplain beside a bluff came
    out classified as escarpment, and was then scored as unfarmable ground a road should
    climb around.  Ask `slope` and `relief` instead, in the units the question is really
    in: metres of rise per kilometre, and metres of command over the ground below.

    The words survive where they belong.  `terrain_label` derives them for maps and
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

    A GM wants to see the ground named, and a wargame's movement rules are written in
    these words, so the bands are worth keeping at the edge.  Deriving them here rather
    than storing them on the hex is what stops them becoming load-bearing again.

        FLAT        under 30 m/km   level going: plough it, cart across it
        ROLLING     30 to 100       undulating; farmed, and a laden cart manages
        STEEP       100 to 250      pack animals, terraces, no wheels
        ESCARPMENT  over 250        a break of slope; on foot and with effort

    Ocean and lake stay in this vocabulary for the same reason: they are the words an
    atlas uses, and a reader wants them.  It is only the *generator* that has no business
    claiming a chemistry it does not model.
    """

    OCEAN = "ocean"
    LAKE = "lake"
    COAST = "coast"
    FLAT = "flat"
    ROLLING = "rolling"
    STEEP = "steep"
    ESCARPMENT = "escarpment"


def is_steep(hx, steep_gradient_m: float) -> bool:
    """Ground too steep to plough, settle, or take a cart across.

    Several stages ask this question and they must agree, which is what the `STEEP_LAND`
    frozenset used to be for.  It is a threshold on a measured gradient now rather than a
    set of class names, so a stage that wants a different bar can ask for one — a road
    refuses a grade a village merely finds inconvenient — instead of every caller being
    stuck with wherever the enum's boundary happened to fall.
    """
    return hx.slope >= steep_gradient_m


def terrain_label(hx, rolling: float, steep: float, escarpment: float) -> TerrainLabel:
    """Band a hex's steepness into the word a reader expects to see on the map."""
    if hx.terrain_class is TerrainClass.OPEN_WATER:
        return TerrainLabel.OCEAN
    if hx.terrain_class is TerrainClass.INLAND_WATER:
        return TerrainLabel.LAKE
    if hx.terrain_class is TerrainClass.COAST:
        return TerrainLabel.COAST
    if hx.slope >= escarpment:
        return TerrainLabel.ESCARPMENT
    if hx.slope >= steep:
        return TerrainLabel.STEEP
    if hx.slope >= rolling:
        return TerrainLabel.ROLLING
    return TerrainLabel.FLAT


DEFAULT_TERRAIN_BANDS = (30.0, 100.0, 250.0)


def terrain_bands(ws) -> tuple[float, float, float]:
    """The gradient bands a world was generated with, for anything drawing it.

    Read back off the serialised config rather than off a live `WorldConfig`, so a world
    loaded from JSON is drawn in its own bands instead of in today's defaults.
    """
    cfg = ws.metadata.get("config", {})
    return (
        cfg.get("terrain_rolling_gradient_m", DEFAULT_TERRAIN_BANDS[0]),
        cfg.get("terrain_steep_gradient_m", DEFAULT_TERRAIN_BANDS[1]),
        cfg.get("terrain_escarpment_gradient_m", DEFAULT_TERRAIN_BANDS[2]),
    )


def terrain_labels(ws) -> dict:
    """Every hex's map label, using the thresholds the world was generated with.

    Read back off the serialised config rather than off a live `WorldConfig`, so a world
    loaded from JSON is drawn in the bands it was made with instead of in today's
    defaults.
    """
    bands = terrain_bands(ws)
    return {coord: terrain_label(h, *bands) for coord, h in ws.hexes.items()}


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


class SoilQuality(Enum):
    """What the ground could support, before anything is done with it.

    A ladder, and the order is load-bearing — `SOIL_RANK` compares them, and the rules in
    `SoilStage` take the worse of two arms. This is the *land*, not what is growing on it:
    the best soil in northern Europe carried wildwood until somebody cleared it, so a hex
    is not fertile because grass grows there.
    """

    UNUSABLE = "unusable"  # desert, bare rock, above the treeline, drowned
    GRAZING = "grazing"  # too steep to plough or too dry to crop; run stock on it
    MARGINAL = "marginal"  # ploughable and poor: leached, waterlogged, or podzol
    ARABLE = "arable"  # ordinary farmland
    PRIME = "prime"  # alluvium: the floodplain of a river too big to wade


SOIL_RANK = {
    SoilQuality.UNUSABLE: 0,
    SoilQuality.GRAZING: 1,
    SoilQuality.MARGINAL: 2,
    SoilQuality.ARABLE: 3,
    SoilQuality.PRIME: 4,
}


class LandUse(Enum):
    """What people actually do with a hex, given the settlements that exist.

    Distinct from `SoilQuality`, which is what the ground could take, and from `LandCover`,
    which is what grows there. Good soil nobody has reached stays WOOD.
    """

    WATER = "water"
    WASTE = "waste"  # nothing worth doing
    WOOD = "wood"  # workable ground still under trees; nobody has cleared it
    PASTURE = "pasture"
    ARABLE = "arable"  # cleared and under the plough


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
    # How steep the ground is: the mean elevation difference to the neighbours, in metres
    # per kilometre.  Kept as the number rather than thresholded into bands, because
    # steepness is a continuum and the thresholds were load-bearing in six places — which
    # is how a flat floodplain beside a bluff came out classified as escarpment, and was
    # scored as unfarmable ground a road should climb around.
    slope: float = 0.0
    # How far this hex stands above the lowest ground touching it, in metres.  Steepness
    # alone cannot express a site "overlooking a plain": what makes that site good is the
    # drop it commands, not the gradient it sits on.
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
    # Three orthogonal facts, and keeping them apart is the point. `soil` is what the
    # ground could take, `land_cover` what grows on it, `land_use` what is done with it.
    # `cultivated` is derived from `land_use` and kept because the classic village stages
    # and the JSON schema both read it.
    soil: SoilQuality | None = None
    land_use: LandUse | None = None
    cultivated: bool = False
    # People living on this hex and working it, as distinct from the population of any
    # settlement. Four fifths of what the land yields feeds the people who grew it —
    # `marketable_surplus_fraction` is the other fifth — so this is the same arithmetic
    # markets are sized by, read the other way round.
    rural_population: float = 0.0
    # Which settlement works this hex, and what it costs that settlement to reach it.
    # Catchments are allocated by travel cost rather than drawn as discs, so a ridge
    # between two settlements becomes a genuine watershed and the shapes follow valleys.
    # `territory_cost` is what weights the hex's contribution: ground at the far edge of a
    # catchment feeds its owner less than ground at the gate.
    territory: HexCoord | None = None
    territory_cost: float = 0.0
