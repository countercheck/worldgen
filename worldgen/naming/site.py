"""What a site means, in words no culture owns.

A place name has a head and usually a qualifier. The head — the *generic* — says what kind
of place it is: a ford, a harbour, a farm. The qualifier — the *specific* — says which one:
the oxen's ford, the black harbour, the farm in the marsh. Both are read off the ground
here, each with a weight for how much it would strike somebody arriving: a cataract or a
pass is the first thing anyone would say about a place, and grassland the last.

The keys are meanings, not English. `GLOSS` is the English only because the etymology
stored beside each name is written in it.
"""

from dataclasses import dataclass, field

from ..core.hex import (
    Biome,
    Hex,
    HexCoord,
    LandCover,
    Settlement,
    SettlementRole,
    SettlementTier,
    SoilQuality,
    TerrainClass,
)
from ..core.hex_grid import axial_to_pixel, neighbors

# The kind of place. Every key here is a head a name can be built on.
GENERICS: dict[str, str] = {
    "ford": "ford",
    "bridge": "bridge",
    "confluence": "watersmeet",
    "mouth": "river mouth",
    "falls": "falls",
    "pass": "pass",
    "harbour": "harbour",
    "shore": "shore",
    "market": "market",
    "mine": "mine",
    "clearing": "clearing",
    "spring": "spring",
    "hill": "hill",
    "steading": "farm",
    "hamlet": "hamlet",
    "enclosure": "enclosure",
    "cottages": "cottages",
    "field": "field",
    "town": "town",
    "wick": "trading place",
    "stow": "meeting place",
    "burgh": "stronghold",
    "hall": "hall",
    "wall": "walled town",
}

# Which one. Qualifiers a site can be named for: what grows round it, what grazes it,
# what colour its ground is.
SPECIFICS: dict[str, str] = {
    "marsh": "marsh",
    "fen": "fen",
    "deepwood": "deep wood",
    "wood": "wood",
    "thorn": "thorn",
    "meadow": "meadow",
    "cold": "cold",
    "sand": "sand",
    "stone": "stone",
    "salt": "salt",
    "rich": "rich",
    "high": "high",
    "oak": "oak",
    "ash": "ash",
    "elm": "elm",
    "birch": "birch",
    "pine": "pine",
    "palm": "palm",
    "broom": "broom",
    "reed": "reed",
    "ox": "oxen",
    "deer": "deer",
    "swine": "swine",
    "horse": "horse",
    "hare": "hare",
    "elk": "elk",
    "wolf": "wolf",
    "bear": "bear",
    "goat": "goat",
    "crane": "crane",
    "eagle": "eagle",
    "white": "white",
    "black": "black",
    "red": "red",
    "green": "green",
    "north": "north",
    "south": "south",
    "east": "east",
    "west": "west",
    "great": "great",
    "little": "little",
}

# Words a culture needs that are neither: the particles a phrase name is built with, and
# the word a people calls itself by, which is what its language is named.
GRAMMAR: dict[str, str] = {
    "of": "of",
    "on": "on",
    "the": "the",
    "river": "river",
    "people": "people",
}

GLOSS: dict[str, str] = {**GENERICS, **SPECIFICS, **GRAMMAR}

# How striking each head is. The site's strongest feature should usually win, so these
# run from a cataract (nobody arriving would call it anything else) down to "farm".
_GENERIC_WEIGHT = {
    "falls": 4.0,
    "pass": 4.0,
    "mine": 4.0,
    "mouth": 3.0,
    "confluence": 3.0,
    "harbour": 3.0,
    "clearing": 3.0,
    "ford": 2.5,
    "bridge": 2.5,
    "spring": 1.5,
    "shore": 1.5,
    "hill": 1.5,
    "market": 1.5,
}

# Hex tags that are themselves a head.
_TAG_GENERIC = {
    "cataract": "falls",
    "pass": "pass",
    "river_mouth": "mouth",
    "confluence": "confluence",
    "ford": "ford",
    "bridge": "bridge",
    "headwater": "spring",
}

_ROLE_GENERIC = {
    SettlementRole.PORT: "harbour",
    SettlementRole.MINING: "mine",
    SettlementRole.LUMBER: "clearing",
    SettlementRole.MARKET: "market",
}

# What any place of a tier can be called, whatever its site. Several heads per tier, as
# English has -ton, -ham, -by, -worth and -thorpe for a farm: with only one, three hundred
# villages share a head word, run out of qualifiers, and all end up named for founders.
_TIER_GENERIC = {
    SettlementTier.CITY: {"burgh": 1.0, "hall": 0.6, "wall": 0.6},
    SettlementTier.TOWN: {"town": 1.0, "wick": 0.8, "stow": 0.6},
    SettlementTier.VILLAGE: {
        "steading": 1.0,
        "hamlet": 1.0,
        "enclosure": 0.8,
        "cottages": 0.6,
        "field": 0.6,
    },
}
_TIER_WEIGHT = {SettlementTier.CITY: 1.5, SettlementTier.TOWN: 1.0, SettlementTier.VILLAGE: 1.0}
_TIER_SIZE = {SettlementTier.CITY: ("great", 0.8), SettlementTier.VILLAGE: ("little", 0.3)}

_COVER_SPECIFIC = {
    LandCover.MARSH: "marsh",
    LandCover.BOG: "fen",
    LandCover.DENSE_FOREST: "deepwood",
    LandCover.WOODLAND: "wood",
    LandCover.SCRUB: "thorn",
    LandCover.OPEN: "meadow",
    LandCover.TUNDRA: "cold",
    LandCover.DESERT: "sand",
    LandCover.ALPINE: "stone",
    LandCover.BARE_ROCK: "stone",
}

# The trees and beasts a biome would put in a name. Each is a candidate at a low weight,
# so over a map they spread rather than every forest town being "oak".
_BIOME_LIFE = {
    Biome.TEMPERATE_FOREST: ("oak", "ash", "elm", "ox", "deer", "swine"),
    Biome.BOREAL: ("pine", "birch", "elk", "wolf", "bear"),
    Biome.GRASSLAND: ("horse", "hare", "ox"),
    Biome.SHRUBLAND: ("broom", "goat", "hare"),
    Biome.DESERT: ("goat",),
    Biome.TROPICAL: ("palm",),
    Biome.WETLAND: ("reed", "crane"),
    Biome.TUNDRA: ("birch", "elk"),
    Biome.ALPINE: ("eagle", "goat"),
}

_COVER_COLOUR = {
    LandCover.BOG: "black",
    LandCover.DENSE_FOREST: "black",
    LandCover.ALPINE: "white",
    LandCover.TUNDRA: "white",
    LandCover.DESERT: "red",
    LandCover.BARE_ROCK: "red",
    LandCover.OPEN: "green",
    LandCover.MARSH: "green",
}

_WATER = (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER)


@dataclass
class Site:
    """What a place could be called for, weighted by how much each would strike a visitor."""

    coord: HexCoord
    generics: dict[str, float] = field(default_factory=dict)
    specifics: dict[str, float] = field(default_factory=dict)

    def add_generic(self, key: str, weight: float) -> None:
        self.generics[key] = max(self.generics.get(key, 0.0), weight)

    def add_specific(self, key: str, weight: float) -> None:
        self.specifics[key] = self.specifics.get(key, 0.0) + weight


def read_site(
    hexes: dict[HexCoord, Hex],
    settlement: Settlement,
    hill_relief_m: float,
    high_elevation_m: float,
    river_features: dict[HexCoord, frozenset[str]] | None = None,
) -> Site:
    """Read a settlement's hex and the ring around it into weighted meanings.

    The ring matters: a town beside a marsh is named for the marsh even though its own hex
    was drained and ploughed, and a harbour town's water is next door, not underfoot.

    *river_features* carries what the rivers beside each hex have there — a ford, a bridge,
    falls, a mouth, a confluence, a spring — since rivers run between hexes rather than on
    them.
    """
    coord = settlement.coord
    hx = hexes[coord]
    ring = [hexes[n] for n in neighbors(coord) if n in hexes]
    site = Site(coord)

    here = hx.tags | (river_features or {}).get(coord, frozenset())
    for tag, generic in _TAG_GENERIC.items():
        if tag in here:
            site.add_generic(generic, _GENERIC_WEIGHT[generic])
    if settlement.role in _ROLE_GENERIC:
        generic = _ROLE_GENERIC[settlement.role]
        site.add_generic(generic, _GENERIC_WEIGHT[generic])
    # The tier's heads share one slot's weight between them, split so their squares — which
    # is how the stage weighs a head — sum to the slot's: more synonyms must not make a
    # farm likelier than a ford.
    shares = _TIER_GENERIC[settlement.tier]
    total = sum(shares.values())
    for generic, share in shares.items():
        site.add_generic(generic, _TIER_WEIGHT[settlement.tier] * (share / total) ** 0.5)
    if settlement.tier in _TIER_SIZE:
        site.add_specific(*_TIER_SIZE[settlement.tier])
    if any(n.terrain_class in _WATER for n in ring) and "harbour" not in site.generics:
        site.add_generic("shore", _GENERIC_WEIGHT["shore"])
    if hill_relief_m > 0 and hx.relief >= hill_relief_m:
        # Twice the bar is a crag the town stands on, and nobody would call it anything else.
        bonus = 1.0 if hx.relief >= 2 * hill_relief_m else 0.0
        site.add_generic("hill", _GENERIC_WEIGHT["hill"] + bonus)

    # What covers the ground round about, by share of the seven hexes.
    around = [hx, *ring]
    land = [h for h in around if h.terrain_class not in _WATER]
    for h in land:
        if h.land_cover in _COVER_SPECIFIC:
            site.add_specific(_COVER_SPECIFIC[h.land_cover], 2.0 / len(around))
        if h.land_cover in _COVER_COLOUR:
            site.add_specific(_COVER_COLOUR[h.land_cover], 0.6 / len(around))
    if hx.biome in _BIOME_LIFE:
        for key in _BIOME_LIFE[hx.biome]:
            site.add_specific(key, 0.5)
    if hx.soil is SoilQuality.PRIME:
        site.add_specific("rich", 1.0)
    if high_elevation_m > 0 and hx.elevation >= high_elevation_m:
        site.add_specific("high", 1.2)
    if any("endorheic_shore" in h.tags or "endorheic" in h.tags for h in around):
        site.add_specific("salt", 2.0)
    return site


# Norton, Sutton, Easton, Weston: a place named for which side of a bigger one it lies.
_DIRECTION_WEIGHT = 0.7


def add_direction(site: Site, anchor: HexCoord) -> None:
    """Qualify *site* by the compass point it lies at from *anchor*, a bigger place nearby.

    North is up the map, as the renderers draw it.
    """
    ax, ay = axial_to_pixel(anchor, 1.0)
    x, y = axial_to_pixel(site.coord, 1.0)
    dx, dy = x - ax, y - ay
    if abs(dx) > abs(dy):
        direction = "east" if dx > 0 else "west"
    else:
        direction = "south" if dy > 0 else "north"
    site.add_specific(direction, _DIRECTION_WEIGHT)


__all__ = ["GENERICS", "GLOSS", "GRAMMAR", "SPECIFICS", "Site", "add_direction", "read_site"]
