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
and calling a rainforest "grazing" was the tell that one symmetric rule would not do. And
each arm reads its own season: drought is a dry-season failure and leaching a wet-season
one, so the same annual rain farms worse the more it is bunched into one half of the year
(`rainfall_soil`, `WorldConfig.wet_season_share`).

**Position in the drainage.** Alluvium is the best ground there is, and a river deposits it
where it can spread: gentle ground beside a channel with a real catchment behind it. This
skips the rainfall arm entirely, because the Nile does not need rain — in a desert the
floodplain is not merely the best land, it is the only land.

Cold caps the whole thing at MARGINAL. Podzol under taiga is poor ground however flat it is
and however much rain falls on it, which is why the boreal map grows no wheat.
"""

from ..core.hex import SOIL_RANK, Biome, SoilQuality, TerrainClass
from ..core.pipeline import GeneratorStage
from ..core.world_state import WorldState
from .oases import place_oases
from .riverside import river_index

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


def rainfall_soil(
    wet_season_mm: float, dry_season_mm: float, cfg, groundwater_mm: float = 0.0
) -> SoilQuality:
    """What the rainfall alone allows. Asymmetric: dry fails differently from wet.

    And each failure has its season. A crop fails for want of water in the season it grows
    in, and ground is leached and waterlogged by the wet half of the year, so the dry arm
    reads the growing season and the wet arm the wet one, each against half of the annual
    bands. The growing season is the dry half in the Mediterranean, where the drought falls
    on the summer, and the wet half under a monsoon or in the taiga, where the dry half is
    a fallow winter (`crops_grow_in_wet_season`). Half,
    because the bands are a year's rain and a season is half a year: in a year without
    seasons both halves are half the year's rain and this reads the annual bands exactly,
    which is what keeps soil and BiomeStage on the one pair of figures.

    So the same annual rain farms worse the more it bunches. 480 mm falling evenly is
    arable; 480 mm with three quarters of it in winter leaves a half-year's steppe even
    after the ground has carried some of the winter over (`soil_water_carryover`), and it
    is grazing — the Mediterranean's summer drought, which is what made it a country of
    flocks and transhumance on rainfall that would plough in Kent.
    """
    # The ground carries some of the wet season's water into the dry one: what it held at
    # the end of the rains is what a crop draws on after them.
    # A crop that grows in the rains already has that water, so where the growing season
    # is the wet one the dry arm reads the wet season as it fell.
    carried = cfg.soil_water_carryover * max(0.0, wet_season_mm - dry_season_mm)
    growing = wet_season_mm if cfg.crops_grow_in_wet_season else dry_season_mm + carried
    # An oasis's groundwater (`oases`) comes up from below into the same growing season. It
    # waters the crop and leaches nothing, so it goes on the dry arm alone.
    dry = 2.0 * (growing + groundwater_mm)
    wet = 2.0 * (wet_season_mm - carried)
    if dry < cfg.soil_dry_farming_min_precip_mm:
        return SoilQuality.UNUSABLE
    if dry < cfg.biome_dry_precip_mm:
        return SoilQuality.GRAZING
    if wet <= cfg.biome_wet_precip_mm:
        return SoilQuality.ARABLE
    if wet < cfg.food_drowned_precip_mm:
        return SoilQuality.MARGINAL
    return SoilQuality.UNUSABLE


def is_alluvium(coord, hx, cfg, rivers) -> bool:
    """Gentle ground beside a river too big to wade.

    `ford_max_catchment_km2` is the threshold rather than one of its own, and it is exactly
    the right question asked from the other side: a stream draining a few tens of square
    kilometres is ankle deep and a step across, and a river you cannot wade is one that
    floods and lays down silt. `catchment_km2` is upstream drainage area, a physical
    quantity comparable between maps, so this means the same thing on any of them.

    A river runs along a hexside; its floodplain is the level ground of its valley, the banks
    and the hexes next to them (`Rivers.near`).
    """
    if hx.slope >= cfg.terrain_rolling_gradient_m:
        return False
    return rivers.near.get(coord, 0.0) >= cfg.ford_max_catchment_km2


class SoilStage(GeneratorStage):
    """Assigns `hex.soil` from slope, rainfall and position in the drainage."""

    def run(self, state: WorldState) -> WorldState:
        hexes = state.hexes
        cfg = self.config
        rivers = river_index(state, cfg)
        # Groundwater first, since the rainfall arm reads it. On this stage's own generator,
        # which no other stage draws from, so siting oases moves nothing else on the map.
        place_oases(state, cfg, self.rng)

        for coord, hx in hexes.items():
            if hx.terrain_class in WATER or hx.biome in NOT_PLOUGHLAND:
                hx.soil = SoilQuality.UNUSABLE
                continue
            if hx.biome in TOO_COLD_TO_GROW:
                hx.soil = SoilQuality.UNUSABLE
                continue

            if is_alluvium(coord, hx, cfg, rivers):
                soil = SoilQuality.PRIME
            else:
                soil = _worse(
                    slope_soil(hx.slope, cfg),
                    rainfall_soil(
                        hx.wet_season_precip_mm,
                        hx.dry_season_precip_mm,
                        cfg,
                        hx.groundwater_mm,
                    ),
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
