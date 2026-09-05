"""What the ground could support, before anything is done with it.

The productive statement of the model used to be `food_value` keyed on `land_cover`, which
said a hex was fertile *because grass grew on it*. That is backwards, and it is why the map
could not tell a floodplain from a chalk down. Grass on temperate lowland is what you get
after clearing or on thin soil; the best ground in northern Europe carried wildwood until
somebody assarted it.

So soil is separated out and asked about directly, on three things that decide it:

**Slope.** Ground the plough cannot work. The bands are already drawn — a hex is 1 km
across, so `terrain_steep_gradient_m` is the gradient at which this map says "pack animals,
terraces, no wheels", and `terrain_escarpment_gradient_m` is a break of slope.

**Rainfall, asymmetrically.** Too dry and too wet are not the same failure. Under the
dry-farming limit nothing is grown at all; between that and the arable band you get steppe,
where grass grows and a crop will not, which is grazing. Above the arable band the ground is
leached and waterlogged — that is poor arable, not pasture, so the wet arm lands on MARGINAL
and calling a rainforest "grazing" was the tell that one symmetric rule would not do.

**Position in the drainage.** Alluvium is the best ground there is, and a river deposits it
where it can spread: gentle ground beside a channel with a real catchment behind it. This
skips the rainfall arm entirely, because the Nile does not need rain — in a desert the
floodplain is not merely the best land, it is the only land.

Cold caps the whole thing at MARGINAL. Podzol under taiga is poor ground however flat it is
and however much rain falls on it, which is why the boreal map grows no wheat.
"""

from ..core.hex import SOIL_RANK, Biome, SoilQuality, TerrainClass
from ..core.hex_grid import neighbors
from ..core.pipeline import GeneratorStage
from ..core.world_state import WorldState

WATER = (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER)
# Neither is ploughland, so neither takes a soil class: the sea is a fishery and a bog is a
# bog. `potential_food` values them in their own right, as it always has.
NOT_PLOUGHLAND = (Biome.OCEAN, Biome.WETLAND)
# Above the treeline nothing is grown at any rainfall or on any slope.
TOO_COLD_TO_GROW = (Biome.ALPINE, Biome.TUNDRA)


def _worse(a: SoilQuality, b: SoilQuality) -> SoilQuality:
    """The lower rung of the ladder. Land is as good as its worst binding constraint."""
    return a if SOIL_RANK[a] <= SOIL_RANK[b] else b


def slope_soil(gradient_m: float, cfg) -> SoilQuality:
    """What the lie of the ground alone allows."""
    if gradient_m >= cfg.terrain_escarpment_gradient_m:
        return SoilQuality.UNUSABLE
    if gradient_m >= cfg.terrain_steep_gradient_m:
        return SoilQuality.GRAZING
    return SoilQuality.ARABLE


def rainfall_soil(precip_mm: float, cfg) -> SoilQuality:
    """What the rainfall alone allows. Asymmetric: dry fails differently from wet."""
    if precip_mm < cfg.soil_dry_farming_min_precip_mm:
        return SoilQuality.UNUSABLE
    if precip_mm < cfg.biome_dry_precip_mm:
        return SoilQuality.GRAZING
    if precip_mm <= cfg.biome_wet_precip_mm:
        return SoilQuality.ARABLE
    if precip_mm < cfg.food_drowned_precip_mm:
        return SoilQuality.MARGINAL
    return SoilQuality.UNUSABLE


def is_alluvium(coord, hx, hexes, cfg) -> bool:
    """Gentle ground on or beside a river too big to wade.

    `ford_max_catchment_km2` is the threshold rather than one of its own, and it is exactly
    the right question asked from the other side: a stream draining a few tens of square
    kilometres is ankle deep and a step across, and a river you cannot wade is one that
    floods and lays down silt. `catchment_km2` is upstream drainage area, a physical
    quantity comparable between maps, so this means the same thing on any of them.
    """
    if hx.slope >= cfg.terrain_rolling_gradient_m:
        return False

    def big(other) -> bool:
        return "river" in other.tags and other.catchment_km2 >= cfg.ford_max_catchment_km2

    if big(hx):
        return True
    return any(big(hexes[n]) for n in neighbors(coord) if n in hexes)


class SoilStage(GeneratorStage):
    """Assigns `hex.soil` from slope, rainfall and position in the drainage."""

    def run(self, state: WorldState) -> WorldState:
        hexes = state.hexes
        cfg = self.config

        for coord, hx in hexes.items():
            if hx.terrain_class in WATER or hx.biome in NOT_PLOUGHLAND:
                hx.soil = SoilQuality.UNUSABLE
                continue
            if hx.biome in TOO_COLD_TO_GROW:
                hx.soil = SoilQuality.UNUSABLE
                continue

            if is_alluvium(coord, hx, hexes, cfg):
                soil = SoilQuality.PRIME
            else:
                soil = _worse(
                    slope_soil(hx.slope, cfg),
                    rainfall_soil(hx.moisture, cfg),
                )

            # The cold cap is applied last, so it binds alluvium too. A flood meadow on the
            # Lena is the best ground in the taiga and it still will not grow wheat: the
            # season is too short and the soil under it is podzol. Capping before this
            # branch let a boreal floodplain out at PRIME, which would have said a subarctic
            # river bottom is worth what Kent is.
            if hx.temperature < cfg.biome_cold_temp_c:
                soil = _worse(soil, SoilQuality.MARGINAL)
            hx.soil = soil

        return state


__all__ = ["SoilStage", "is_alluvium", "rainfall_soil", "slope_soil"]
