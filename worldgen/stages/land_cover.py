from ..core.hex import Biome, LandCover, TerrainClass
from ..core.pipeline import GeneratorStage
from ..core.world_state import WorldState


class LandCoverStage(GeneratorStage):
    def run(self, state: WorldState) -> WorldState:
        wet_moist = self.config.biome_wet_precip_mm

        for h in state.hexes.values():
            h.land_cover = _derive(h, wet_moist)

        return state


def _derive(h, wet_moist: float) -> LandCover:
    tc = h.terrain_class
    b = h.biome

    if tc in (TerrainClass.OCEAN, TerrainClass.LAKE):
        return LandCover.OPEN_WATER
    # Only a genuine break of slope is bare. STEEP ground — a tenth to a quarter — holds
    # soil perfectly well: it is where terraces, vineyards and hanging woods go. Stripping
    # it to rock put a third of a 128x128 map under bare stone. Above the treeline it
    # comes out ALPINE on the next line anyway, which is the honest reason high steep
    # ground looks barren.
    if tc == TerrainClass.ESCARPMENT:
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
    # Closed canopy wants markedly more rain than the wet band's floor. This used to sit
    # midway between that floor and the old normalised ceiling of 1.0; millimetres have no
    # ceiling, so it is expressed as a multiple of the band instead and means the same
    # thing on a dry map as on a wet one.
    dense_thresh = wet_moist * 1.5
    if b == Biome.TEMPERATE_FOREST and h.moisture > dense_thresh:
        return LandCover.DENSE_FOREST
    if b in (Biome.TEMPERATE_FOREST, Biome.TROPICAL):
        return LandCover.WOODLAND
    if b == Biome.SHRUBLAND:
        return LandCover.SCRUB
    return LandCover.OPEN
