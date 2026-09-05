from ..core.config import CLIMATE_CONTEXTS
from ..core.hex import Biome, TerrainClass
from ..core.pipeline import GeneratorStage
from ..core.world_state import WorldState


class BiomeStage(GeneratorStage):
    def run(self, state: WorldState) -> WorldState:
        treeline_temp = self.config.biome_treeline_temp_c
        snowline_temp = self.config.biome_snowline_temp_c
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
            # Two lines divide cold country, and both are temperatures: the treeline,
            # above which nothing grows tall, and the snowline, above which nothing grows
            # at all.  Between them is tundra — treeless but vegetated, which is what most
            # ground above the subarctic treeline actually is.  Neither goes through
            # `pick`: they are what happens when ground is too cold to grow anything,
            # which every region has somewhere above it, so `_ALWAYS` lists them both.
            elif h.temperature < snowline_temp:
                h.biome = Biome.ALPINE
            elif h.temperature < treeline_temp:
                h.biome = Biome.TUNDRA
            elif h.temperature < cold_temp:
                # Below the treeline and cold is taiga, whatever the rainfall.  This used
                # to split on `biome_dry_precip_mm` — the same 400 mm that separates
                # desert from steppe — which made rain the thing that stops trees in the
                # subarctic.  It is not; cold is, and the treeline above already says so.
                # Siberian larch grows on 200-400 mm, and the boreal region's own mean is
                # 450, so any rain shadow tipped land into tundra: 23% of a boreal map,
                # all of it at zero food value.
                h.biome = pick(Biome.BOREAL, Biome.TEMPERATE_FOREST, Biome.GRASSLAND)
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

        # Assign WETLAND to flat or coastal river hexes that cannot shed what falls on
        # them.  FLAT → BOG, COAST → MARSH in LandCoverStage.
        #
        # Tested on runoff rather than on rainfall.  Waterlogging is not a question of how
        # much rain arrives but of whether the ground can get rid of it: flat land beside
        # a river, where what the sky delivers exceeds what the air takes back, holds a
        # water table at the surface.  A rainfall test asked for more than the wet biome
        # band, which on a temperate map at 800 mm almost nothing reaches, so bogs
        # vanished entirely; and it would have called a cold region dry when cold country
        # is exactly where peat forms, because so little of its rain evaporates away.
        min_runoff = self.config.wetland_min_runoff_mm
        for h in state.hexes.values():
            if (
                h.terrain_class in (TerrainClass.FLAT, TerrainClass.COAST)
                and self.config.runoff_mm(h.moisture, h.temperature) > min_runoff
                and "river" in h.tags
                and h.temperature >= treeline_temp
            ):
                h.biome = Biome.WETLAND

        # The shore of a closed basin is wetland too.  HydrologyStage tags these where a
        # lake receives rivers but has no outlet: the water that arrives leaves by
        # evaporation, spreading out into marsh and bog around the rim rather than
        # running off.  The moisture floor keeps arid basins as salt pans — a closed
        # basin in a desert is a playa, not a swamp — and the terrain and treeline
        # tests match the river-wetland rule above so the two look consistent.
        marsh_min_moisture = self.config.endorheic_marsh_min_precip_mm
        for h in state.hexes.values():
            if (
                "endorheic_shore" in h.tags
                and h.terrain_class in (TerrainClass.FLAT, TerrainClass.COAST)
                and h.moisture >= marsh_min_moisture
                and h.temperature >= treeline_temp
            ):
                h.biome = Biome.WETLAND

        return state
