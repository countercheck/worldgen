"""Habitability: how good a settlement site each hex is, at three catchment sizes.

A site is scored on the land it can actually feed itself from, not on the biome of the
single hex it stands on.  The catchment is the mean food value of every hex within reach,
so a town ringed by grassland beats one on an identical hex ringed by desert — a
distinction the old single-hex biome term could not make at all.

Reach depends on tier: a city draws on a far wider hinterland than a village, so the same
hex gets three scores, one per cultivation radius.  Each tier's placement stage then sorts
on its own.
"""

from ..core.hex import STEEP_LAND, Biome, LandCover, TerrainClass
from ..core.hex_grid import neighbors, ring
from ..core.pipeline import GeneratorStage
from ..core.world_state import WorldState

# Land cover bands.  Cover is a better key than biome: it already folds in terrain and
# moisture (the dense-forest/woodland split is a moisture threshold), and it is what
# CultivationStage tests against, so the two cannot disagree about what is farmable.
FERTILE = frozenset({LandCover.OPEN, LandCover.WOODLAND})
MARGINAL = frozenset({LandCover.SCRUB, LandCover.DENSE_FOREST})
WETLAND_COVER = frozenset({LandCover.BOG, LandCover.MARSH})
# TUNDRA, DESERT, ALPINE and BARE_ROCK feed nobody and are worth zero.


def moisture_factor(precip_mm: float, dry: float, wet: float, drowned: float = 3000.0) -> float:
    """Agricultural suitability of an annual rainfall, as a factor in [0, 1].

    Rainfall is not monotonic for farming — desert at one end, waterlogged at the other —
    so this is a tent, not a ramp: full value across the band `[dry, wet]`, falling to
    zero at nothing and again at *drowned*.  The band is the same pair of thresholds
    `BiomeStage` classifies on, so the two systems cannot drift.

    The wet arm used to fall away to zero at 1.0, which was the ceiling of the old
    normalised moisture scale rather than a fact about farming.  In millimetres there is
    no such ceiling, so the point where ground drowns has to be stated: `drowned` is where
    leaching and waterlogging have taken the yield to nothing, around 3000 mm a year.
    Leaving the old 1.0 in place would have zeroed the food value of every hex on the map,
    since all of them get more than a millimetre of rain.
    """
    if precip_mm <= 0.0 or precip_mm >= drowned:
        return 0.0
    if precip_mm < dry:
        return precip_mm / dry if dry > 0.0 else 1.0
    if precip_mm <= wet:
        return 1.0
    return (drowned - precip_mm) / (drowned - wet) if wet < drowned else 1.0


def food_value(hx, cfg, dry: float, wet: float) -> float:
    """How much food one hex contributes to a catchment.

    Water is not zero: a coastal site fishes.  Scoring the sea at nothing penalised
    coastal sites twice over — half their catchment counted as waste ground, and the flat
    coastal bonus existed largely to repair the damage.

    Wetland sits *below* open water, being neither good fishing nor good ploughing, which
    matches bog and marsh already resisting cultivation outright.
    """
    cover = hx.land_cover
    if cover is LandCover.OPEN_WATER:
        return cfg.food_water_value
    if cover in WETLAND_COVER:
        return cfg.food_wetland_value
    if cover in FERTILE:
        base = cfg.food_fertile_value
    elif cover in MARGINAL:
        base = cfg.food_marginal_value
    else:
        return 0.0
    return base * moisture_factor(hx.moisture, dry, wet, cfg.food_drowned_precip_mm)


def site_bonus(coord, hx, hexes, cfg) -> float:
    """What the hex itself is worth as a site, independent of the land around it.

    A river to carry goods and drive a mill, a coast to land a boat, a rise to see and be
    seen from, a confluence where two routes must meet.  These are facts about the point,
    not about its catchment, so every scorer that ranks sites should read the same
    function — a second copy would drift from this one the first time a bonus changed.
    """
    nbrs = [hexes[n] for n in neighbors(coord) if n in hexes]
    bonus = 0.0

    if "river" in hx.tags or any("river" in n.tags for n in nbrs):
        bonus += cfg.habitability_river_bonus

    if hx.terrain_class == TerrainClass.COAST or any(
        n.terrain_class == TerrainClass.COAST for n in nbrs
    ):
        bonus += cfg.habitability_coast_bonus

    if hx.terrain_class == TerrainClass.ROLLING and any(
        n.terrain_class == TerrainClass.FLAT for n in nbrs
    ):
        bonus += cfg.habitability_hill_bonus

    if "confluence" in hx.tags:
        bonus += cfg.habitability_confluence_bonus

    return bonus


def _ring_offsets(max_radius: int) -> list[list[tuple[int, int]]]:
    """Offsets from a hex, grouped by ring, out to *max_radius*.

    Identical for every hex, so it is built once and walked per hex.  Grouping by ring
    means one pass out to the largest radius yields every smaller radius on the way,
    rather than one sweep per radius.
    """
    return [ring((0, 0), d) for d in range(max_radius + 1)]


def catchment_means(coords, food, radii: list[int]) -> dict[int, dict]:
    """Mean food value within each radius in *radii*, for every coord in *coords*.

    Off-map neighbours are left out of the mean rather than counted as zero, so a hex on
    the map border is not scored as though the edge were desert.
    """
    wanted = sorted(set(radii))
    offsets = _ring_offsets(wanted[-1])
    out: dict[int, dict] = {r: {} for r in wanted}

    for coord in coords:
        q, r = coord
        total = 0.0
        count = 0
        target = 0
        for d, ring_offsets in enumerate(offsets):
            for dq, dr in ring_offsets:
                value = food.get((q + dq, r + dr))
                if value is not None:
                    total += value
                    count += 1
            while target < len(wanted) and wanted[target] == d:
                out[wanted[target]][coord] = total / count if count else 0.0
                target += 1
    return out


class HabitabilityStage(GeneratorStage):
    def run(self, state: WorldState) -> WorldState:
        hexes = state.hexes
        cfg = self.config
        dry, wet = cfg.biome_dry_precip_mm, cfg.biome_wet_precip_mm

        # One food value per hex, computed once — every catchment reads this table rather
        # than re-deriving the value for each of its ~217 members.
        food = {coord: food_value(hx, cfg, dry, wet) for coord, hx in hexes.items()}

        radii = {
            "city": cfg.cultivation_city_radius,
            "town": cfg.cultivation_town_radius,
            "village": cfg.cultivation_village_radius,
        }
        means = catchment_means(hexes.keys(), food, list(radii.values()))

        raw: dict[str, dict] = {tier: {} for tier in radii}
        for coord, hx in hexes.items():
            if (
                hx.terrain_class in (TerrainClass.OCEAN, TerrainClass.LAKE)
                or hx.terrain_class in STEEP_LAND
                or hx.biome == Biome.WETLAND
            ):
                for tier in radii:
                    raw[tier][coord] = 0.0
                continue

            # Site bonuses describe the hex itself and so are identical across tiers;
            # only the catchment term changes with reach.
            bonus = site_bonus(coord, hx, hexes, cfg)

            for tier, radius in radii.items():
                raw[tier][coord] = cfg.habitability_agri_weight * means[radius][coord] + bonus

        # Normalise each tier against its own best site: the scores are only ever
        # compared within a tier, and a shared divisor would let the widest catchment
        # squash the other two.
        for tier, scores in raw.items():
            top = max(scores.values(), default=0.0) or 1.0
            for coord, score in scores.items():
                setattr(hexes[coord], f"habitability_{tier}", score / top)

        return state
