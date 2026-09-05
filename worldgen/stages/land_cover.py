from ..core.config import CLIMATE_CONTEXTS
from ..core.hex import Biome, LandCover, SoilQuality, TerrainClass
from ..core.pipeline import GeneratorStage
from ..core.world_state import WorldState

# Soil good enough that, left alone, trees take it. This is the wildwood: the best ground
# in northern Europe carried oak and lime until somebody cleared it, and a temperate map
# showing open grass on its best soil has the causality backwards. What actually stands
# there is the region's own woodland, so a region whose palette has no forest — a desert,
# a steppe — is unaffected and keeps whatever its climate gives it.
WOODED_SOIL = frozenset({SoilQuality.PRIME, SoilQuality.ARABLE})


class LandCoverStage(GeneratorStage):
    def run(self, state: WorldState) -> WorldState:
        cfg = self.config
        wet_moist = cfg.biome_wet_precip_mm
        palette = CLIMATE_CONTEXTS[cfg.regional_climate].palette
        wildwood = next(
            (b for b in (Biome.TEMPERATE_FOREST, Biome.TROPICAL, Biome.BOREAL) if b in palette),
            None,
        )

        for h in state.hexes.values():
            h.land_cover = _derive(h, wet_moist, wildwood)

        return state


def _derive(h, wet_moist: float, wildwood: Biome | None = None) -> LandCover:
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
    # Good soil grows trees unless the region has none to grow. Tested after the barren
    # covers above, so alluvium under a glacier stays alpine, and after wetland, so a fen
    # is not reclassified as oakwood for being flat and beside a river.
    if wildwood is not None and h.soil in WOODED_SOIL and b is not wildwood:
        b = wildwood
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
