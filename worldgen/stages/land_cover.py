from ..core.hex import Biome, LandCover, TerrainClass
from ..core.pipeline import GeneratorStage
from ..core.world_state import WorldState


class LandCoverStage(GeneratorStage):
    def run(self, state: WorldState) -> WorldState:
        wet_moist = self.config.biome_wet_moist

        for h in state.hexes.values():
            h.land_cover = _derive(h, wet_moist)

        return state


def _derive(h, wet_moist: float) -> LandCover:
    tc = h.terrain_class
    b = h.biome

    if tc == TerrainClass.OCEAN:
        return LandCover.OPEN_WATER
    if tc == TerrainClass.MOUNTAIN:
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
    if b == Biome.TEMPERATE_FOREST and h.moisture > wet_moist:
        return LandCover.DENSE_FOREST
    if b in (Biome.TEMPERATE_FOREST, Biome.TROPICAL):
        return LandCover.WOODLAND
    if b == Biome.SHRUBLAND:
        return LandCover.SCRUB
    return LandCover.OPEN
