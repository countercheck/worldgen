"""What is actually done with the ground, and how many people that feeds.

`SoilStage` says what a hex could take. This says what is being taken from it, which is a
different question and has a different answer nearly everywhere: good soil nobody has
reached is still wildwood, and wildwood feeds far fewer people than the same soil under the
plough. That gap is what gives clearing economic weight — a settlement grows by assarting
its hinterland, not merely by sitting in it.

**Where clearing stops is set by scarcity, not by a fixed radius.** The old rule drew a
disc — eight hexes round a city, four round a town — so a city on thin ground cleared
exactly as far as one on a floodplain. What decides it here is rent:

    rent(hex)  =  potential_food * usable_fraction(territory_cost, market_day_radius)
    cleared    ⟺  rent >= clearing_margin * the best rent in that catchment

Transport cost stands in for the effort of working land that far out, and the bar is
**relative to the best land in reach**. A market with a floodplain has a high bar and
leaves its hillsides to sheep; a market on uniformly thin ground has a low bar and ploughs
the scrub. The worse the land, the more pressure to use bad land — the extensive margin set
against the best alternative available, which is what rent theory actually says. Von
Thünen's rings still fall out because rent falls with cost, but a ring's *width* now varies
with what its catchment holds instead of being the same everywhere.

It needs one knob and one pass. No per-soil clearing costs, and no fixed point to iterate:
rent depends on the catchment and the soil, both of which are settled before this runs.

**Sizing lives here too**, because a market is worth what its countryside actually sends
and that is not known until the countryside has been put to use. `MarketStage` plants and
allocates; this founds. One owner for population rather than a provisional figure written
twice.
"""

from ..core.hex import (
    LandCover,
    LandUse,
    Settlement,
    SettlementTier,
    SoilQuality,
    TerrainClass,
)
from ..core.pipeline import GeneratorStage
from ..core.world_state import WorldState
from .city_town import _assign_role
from .habitability import actual_food, potential_food
from .haulage import gather, usable_fraction

_WATER = (TerrainClass.OCEAN, TerrainClass.LAKE)

WATER = (TerrainClass.OCEAN, TerrainClass.LAKE)
# Soil you can get a plough into at all. GRAZING fails it for the two reasons that make
# ground grazing in the first place: too steep for the share, or too dry for the seed.
PLOUGHABLE = frozenset({SoilQuality.MARGINAL, SoilQuality.ARABLE, SoilQuality.PRIME})


def rent(hx, cfg) -> float:
    """What a hex is worth to the settlement that holds it.

    How good the ground is, discounted by what it costs to get at — the same
    `usable_fraction` falloff that decides what a market can haul, so distance means one
    thing throughout the model. Ground nobody holds has no rent: it is not that it is
    worthless, it is that there is nobody to work it.
    """
    if hx.territory is None:
        return 0.0
    return potential_food(hx, cfg) * usable_fraction(hx.territory_cost, cfg.market_day_radius)


def decide_land_use(hexes, cfg) -> None:
    """Assign `hex.land_use` over the whole map, and `cultivated` with it."""
    best: dict = {}
    rents: dict = {}
    for coord, hx in hexes.items():
        value = rent(hx, cfg)
        rents[coord] = value
        if hx.territory is not None and value > best.get(hx.territory, 0.0):
            best[hx.territory] = value

    for coord, hx in hexes.items():
        hx.land_use = _use_for(hx, rents[coord], best.get(hx.territory, 0.0), cfg)
        hx.cultivated = hx.land_use is LandUse.ARABLE


def _use_for(hx, hex_rent: float, best_rent: float, cfg) -> LandUse:
    if hx.terrain_class in WATER or hx.land_cover is LandCover.OPEN_WATER:
        return LandUse.WATER
    if hx.soil is SoilQuality.UNUSABLE:
        return LandUse.WASTE
    if hx.territory is None:
        # Beyond every catchment. Good ground out here is not poor, merely unreached —
        # this is the wildwood, and it is why the map should show trees standing between
        # the markets rather than a continuous sheet of ploughland.
        return LandUse.WOOD if hx.soil in PLOUGHABLE else LandUse.WASTE
    if hx.soil is SoilQuality.GRAZING:
        # Grazing needs no clearing, so the rent margin does not apply to it: if anybody is
        # near enough to hold the ground, they run stock on it.
        return LandUse.PASTURE
    if hex_rent >= cfg.clearing_margin * best_rent:
        return LandUse.ARABLE
    return LandUse.WOOD


def rural_population(hx, cfg) -> float:
    """People living on this hex and working it.

    Not a new model — the same arithmetic markets have always been sized by, read the other
    way round. A market draws `marketable_surplus_fraction` of what its catchment yields, so
    the other four fifths is what feeds the people who grew it. Defining it this way means
    the two figures reconcile by construction rather than by calibration.

    Zero on water, whatever the food model says a fishery yields: the fishermen live on
    the shore that works the water, not on the water itself. Without this a fifth of the
    map's people stood on the open sea, and every density figure quietly counted them.
    """
    if hx.terrain_class in _WATER:
        return 0.0
    return actual_food(hx, cfg) * (1.0 - cfg.marketable_surplus_fraction) * cfg.people_per_food


class LandUseStage(GeneratorStage):
    """Clears the countryside, founds the markets on it, and counts who lives there."""

    def run(self, state: WorldState) -> WorldState:
        hexes = state.hexes
        cfg = self.config

        decide_land_use(hexes, cfg)

        for hx in hexes.values():
            hx.rural_population = rural_population(hx, cfg)

        seats = [tuple(c) for c in state.metadata.get("market_seats", [])]
        if seats:
            owner = {c: hx.territory for c, hx in hexes.items() if hx.territory is not None}
            cost = {c: hexes[c].territory_cost for c in owner}
            surplus = {
                coord: actual_food(hx, cfg) * cfg.marketable_surplus_fraction
                for coord, hx in hexes.items()
            }
            draw = gather(surplus, owner, cost, cfg.market_day_radius)
            state.settlements.extend(self._found(seats, draw, hexes, cfg))

        return state

    def _found(self, seats, draw, hexes, cfg) -> list:
        """Turn planted seats into settlements sized by the surplus they gather.

        Population is what the catchment can actually send, not a random draw — so a market
        on a wide fertile plain outgrows one wedged in a valley, and the difference is
        visible on the map rather than an accident of the seed.
        """
        # One vectorised draw over coord-sorted seats: deterministic, and it keeps size
        # from being a perfectly invertible function of catchment, which reads as
        # mechanical when a player compares two towns.
        jitter = self.rng.uniform(0.9, 1.1, size=len(seats))

        out = []
        for i, coord in enumerate(sorted(seats)):
            hx = hexes[coord]
            population = max(1, round(draw.get(coord, 0.0) * cfg.people_per_food * jitter[i]))
            s = Settlement(
                coord=coord,
                tier=SettlementTier.TOWN,
                role=_assign_role(coord, hx, hexes),
                population=population,
                name=f"{hx.biome.name.lower()}_market_{i}",
            )
            hx.settlement = s
            out.append(s)
        return out


__all__ = ["LandUseStage", "decide_land_use", "rent", "rural_population"]
