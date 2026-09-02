from ..core.config import CLIMATE_CONTEXTS
from ..core.hex import Biome, TerrainClass
from ..core.pipeline import GeneratorStage
from ..core.world_state import WorldState


class BiomeStage(GeneratorStage):
    def run(self, state: WorldState) -> WorldState:
        alpine_elev = self.config.biome_alpine_elev
        cold_temp = self.config.biome_cold_temp_c
        warm_temp = self.config.biome_warm_temp_c
        dry_moist = self.config.biome_dry_precip_mm
        wet_moist = self.config.biome_wet_precip_mm

        palette = CLIMATE_CONTEXTS[self.config.regional_climate].palette

        def pick(*candidates: Biome) -> Biome:
            """First candidate the region's climate can actually produce.

            The thresholds below still read temperature and moisture, so terrain drives
            the variation exactly as before — but the answer is drawn from the region's
            own palette.  Candidates are given warmest-or-wettest first and fall back
            towards the region's staple, so a hex that would have been tropical in a
            boreal region becomes the closest thing that region has rather than
            importing a biome from three climate zones away.
            """
            for biome in candidates:
                if biome in palette:
                    return biome
            return candidates[-1]

        for h in state.hexes.values():
            if h.terrain_class in (TerrainClass.OCEAN, TerrainClass.LAKE):
                h.biome = Biome.OCEAN
            elif h.elevation > alpine_elev:
                h.biome = Biome.ALPINE
            elif h.temperature < cold_temp:
                h.biome = (
                    pick(Biome.TUNDRA, Biome.SHRUBLAND, Biome.GRASSLAND)
                    if h.moisture < dry_moist
                    else pick(Biome.BOREAL, Biome.TEMPERATE_FOREST, Biome.GRASSLAND)
                )
            elif h.temperature >= warm_temp:
                if h.moisture < dry_moist:
                    h.biome = pick(Biome.DESERT, Biome.SHRUBLAND, Biome.GRASSLAND)
                elif h.moisture < wet_moist:
                    h.biome = pick(Biome.GRASSLAND, Biome.SHRUBLAND)
                else:
                    h.biome = pick(Biome.TROPICAL, Biome.TEMPERATE_FOREST, Biome.GRASSLAND)
            else:
                if h.moisture < dry_moist:
                    # Desert first, not shrubland: a dry region does not stop being a
                    # desert because it is cool.  The Gobi and the Great Basin are cold
                    # deserts.  Regions whose palette has no desert fall straight through
                    # to shrubland, so this changes nothing outside an arid region.
                    h.biome = pick(Biome.DESERT, Biome.SHRUBLAND, Biome.GRASSLAND)
                elif h.moisture < wet_moist:
                    h.biome = pick(Biome.GRASSLAND, Biome.SHRUBLAND)
                else:
                    h.biome = pick(Biome.TEMPERATE_FOREST, Biome.BOREAL, Biome.GRASSLAND)

        # Assign WETLAND to flat or coastal river hexes with very high moisture (below alpine
        # elevation).  FLAT → BOG, COAST → MARSH in LandCoverStage.
        for h in state.hexes.values():
            if (
                h.terrain_class in (TerrainClass.FLAT, TerrainClass.COAST)
                and h.moisture > wet_moist
                and "river" in h.tags
                and h.elevation <= alpine_elev
            ):
                h.biome = Biome.WETLAND

        # The shore of a closed basin is wetland too.  HydrologyStage tags these where a
        # lake receives rivers but has no outlet: the water that arrives leaves by
        # evaporation, spreading out into marsh and bog around the rim rather than
        # running off.  The moisture floor keeps arid basins as salt pans — a closed
        # basin in a desert is a playa, not a swamp — and the terrain and elevation
        # tests match the river-wetland rule above so the two look consistent.
        marsh_min_moisture = self.config.endorheic_marsh_min_precip_mm
        for h in state.hexes.values():
            if (
                "endorheic_shore" in h.tags
                and h.terrain_class in (TerrainClass.FLAT, TerrainClass.COAST)
                and h.moisture >= marsh_min_moisture
                and h.elevation <= alpine_elev
            ):
                h.biome = Biome.WETLAND

        return state
