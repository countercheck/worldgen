"""Habitability: how good a settlement site each hex is, at three catchment sizes.

A site is scored on the land it can actually feed itself from, not on the biome of the
single hex it stands on.  The catchment is the mean food value of every hex within reach,
so a town ringed by grassland beats one on an identical hex ringed by desert — a
distinction the old single-hex biome term could not make at all.

Reach depends on tier: a city draws on a far wider hinterland than a village, so the same
hex gets three scores, one per cultivation radius.  Each tier's placement stage then sorts
on its own.
"""

from ..core.hex import Biome, LandCover, LandUse, SoilQuality, TerrainClass, is_steep
from ..core.hex_grid import neighbors, ring
from ..core.pipeline import GeneratorStage
from ..core.world_state import WorldState

WETLAND_COVER = frozenset({LandCover.BOG, LandCover.MARSH})


def soil_value(soil: SoilQuality | None, cfg) -> float:
    """What a soil class could yield, per hex, if it were worked.

    This replaced a `moisture_factor` tent multiplied onto a cover band. Rainfall now
    enters exactly once, in `SoilStage`, where it *chooses* the class — the tent's own
    thresholds became the class boundaries, so the same three figures still draw the same
    lines. Applying it again here would price rainfall twice.
    """
    return {
        SoilQuality.PRIME: cfg.food_prime_value,
        SoilQuality.ARABLE: cfg.food_arable_value,
        SoilQuality.MARGINAL: cfg.food_marginal_value,
        SoilQuality.GRAZING: cfg.food_grazing_value,
    }.get(soil, 0.0)


def potential_food(hx, cfg) -> float:
    """What one hex could contribute to a catchment, if it were worked as well as it can be.

    Water and wetland are valued in their own right rather than by a soil class, because
    neither is ploughland: the sea is a fishery and a bog is a bog. Water is deliberately
    non-zero — scoring it at nothing penalised coastal sites twice over, once for the waste
    ground and once for the coast bonus that existed to repair the damage.

    This is what *siting* reads. A settler picks land for what it will be once worked, not
    for the wildwood standing on it today.

    Alluvium multiplies the result rather than adding to it, and so only rewards ground
    that could already feed somebody.  Silt is a soil, not a climate: it renews what
    cropping strips and is why the great river valleys carry the people they do, but it
    does not water a desert or hold a crop on bare rock.  Adding a flat bonus would have
    made a silted dune farmland, and the deltas are exactly where the alluvium is deepest.
    `soil_value` is already zero for UNUSABLE ground, so no multiplier can lift it.
    """
    cover = hx.land_cover
    if cover is LandCover.OPEN_WATER:
        return cfg.food_water_value
    if cover in WETLAND_COVER:
        return cfg.food_wetland_value
    return soil_value(hx.soil, cfg) * (1.0 + cfg.food_alluvium_bonus * hx.alluvium)


def actual_food(hx, cfg) -> float:
    """What the hex yields as it is actually being used.

    The gap between this and `potential_food` is what gives clearing economic weight: wood
    standing on prime soil feeds far fewer people than the same soil under the plough, so a
    settlement that assarts its hinterland genuinely grows on it. Ground with no land use
    yet assigned is read at its potential, so every stage before `LandUseStage` sees the
    surface it is entitled to.
    """
    base = potential_food(hx, cfg)
    if hx.land_use is None or hx.land_cover is LandCover.OPEN_WATER:
        return base
    return base * {
        LandUse.ARABLE: cfg.yield_arable,
        LandUse.PASTURE: cfg.yield_pasture,
        LandUse.WOOD: cfg.yield_wood,
    }.get(hx.land_use, 0.0)


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

    # Water that floats a barge, which is a different question from a pleasant shore: a
    # navigable river twenty miles inland is a port, and a rocky coast on a dead-end bay is
    # not. Imported here rather than at module scope because `haulage` reads this module's
    # `food_value`; the cycle is only a problem at import time.
    from .haulage import navigable

    if navigable(hx, cfg) or any(navigable(n, cfg) for n in nbrs):
        bonus += cfg.habitability_harbour_bonus

    if hx.terrain_class == TerrainClass.COAST or any(
        n.terrain_class == TerrainClass.COAST for n in nbrs
    ):
        bonus += cfg.habitability_coast_bonus

    # A site that overlooks the ground beside it, scaled by the drop it actually
    # commands.  This used to be paid flat to any ROLLING hex with a FLAT neighbour,
    # which asked two band questions and got the wrong answer to both: a knoll and a
    # bluff were worth the same, and a level floodplain beside a bluff collected the
    # bonus for standing under the very drop that commands it.  `relief` is the drop
    # itself, so the bonus is now proportional to what a wall on this hex would see.
    if hx.relief > 0.0:
        bonus += cfg.habitability_hill_bonus * min(1.0, hx.relief / cfg.habitability_hill_relief_m)

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

        # Potential, not actual: this stage runs before anything is cleared, and a site is
        # chosen for what its hinterland will yield once worked. One value per hex,
        # computed once — every catchment reads this table rather than re-deriving the
        # value for each of its ~217 members.
        food = {coord: potential_food(hx, cfg) for coord, hx in hexes.items()}

        radii = {
            "city": cfg.cultivation_city_radius,
            "town": cfg.cultivation_town_radius,
            "village": cfg.cultivation_village_radius,
        }
        means = catchment_means(hexes.keys(), food, list(radii.values()))

        raw: dict[str, dict] = {tier: {} for tier in radii}
        for coord, hx in hexes.items():
            if (
                hx.terrain_class in (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER)
                or is_steep(hx, cfg.terrain_steep_gradient_m)
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
