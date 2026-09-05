from ..core.hex import Biome, LandCover, TerrainClass
from ..core.pipeline import GeneratorStage
from ..core.world_state import WorldState


class LandCoverStage(GeneratorStage):
    def run(self, state: WorldState) -> WorldState:
        wet_moist = self.config.biome_wet_moist

        for h in state.hexes.values():
            h.land_cover = _derive(
                h,
                wet_moist,
                self.config.terrain_mountain_gradient,
                self.config.terrain_bare_elevation,
            )

        return state


def _derive(h, wet_moist: float, bare_slope: float, bare_elevation: float) -> LandCover:
    tc = h.terrain_class
    b = h.biome

    if tc in (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER):
        return LandCover.OPEN_WATER
    # Broken ground carries no soil — but a shore is a shore however steep it is, which
    # the terrain classes said by making COAST win over the steepness bands.  Reading the
    # slope directly loses that precedence unless it is stated, and a cliff-backed river
    # mouth came out bare rock instead of the marsh it is.
    if tc != TerrainClass.COAST and (h.slope > bare_slope or h.elevation > bare_elevation):
        return LandCover.BARE_ROCK
    if b == Biome.ALPINE:
        return LandCover.ALPINE
    if b == Biome.TUNDRA:
        return LandCover.TUNDRA
    if b == Biome.DESERT:
        return LandCover.DESERT
    if b == Biome.WETLAND:
        return LandCover.MARSH if tc == TerrainClass.COAST else LandCover.BOG
    if b == Biome.BOREAL:
        return LandCover.DENSE_FOREST
    # Split TEMPERATE_FOREST into dense (very wet) vs woodland (moderately wet).
    # All TEMPERATE_FOREST hexes have moisture >= wet_moist, so we need a higher
    # threshold here to ensure both cover types actually appear.
    dense_thresh = (wet_moist + 1.0) / 2.0
    if b == Biome.TEMPERATE_FOREST and h.moisture > dense_thresh:
        return LandCover.DENSE_FOREST
    if b in (Biome.TEMPERATE_FOREST, Biome.TROPICAL):
        return LandCover.WOODLAND
    if b == Biome.SHRUBLAND:
        return LandCover.SCRUB
    return LandCover.OPEN
