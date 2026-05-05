from ..core.hex import Biome, TerrainClass
from ..core.pipeline import GeneratorStage
from ..core.world_state import WorldState


class BiomeStage(GeneratorStage):
    def run(self, state: WorldState) -> WorldState:
        alpine_elev = self.config.biome_alpine_elev
        cold_temp = self.config.biome_cold_temp
        warm_temp = self.config.biome_warm_temp
        dry_moist = self.config.biome_dry_moist
        wet_moist = self.config.biome_wet_moist

        for h in state.hexes.values():
            if h.terrain_class == TerrainClass.OCEAN:
                h.biome = Biome.OCEAN
            elif h.elevation > alpine_elev:
                h.biome = Biome.ALPINE
            elif h.temperature < cold_temp:
                h.biome = Biome.TUNDRA if h.moisture < dry_moist else Biome.BOREAL
            elif h.temperature >= warm_temp:
                if h.moisture < dry_moist:
                    h.biome = Biome.DESERT
                elif h.moisture < wet_moist:
                    h.biome = Biome.GRASSLAND
                else:
                    h.biome = Biome.TROPICAL
            else:
                if h.moisture < dry_moist:
                    h.biome = Biome.SHRUBLAND
                elif h.moisture < wet_moist:
                    h.biome = Biome.GRASSLAND
                else:
                    h.biome = Biome.TEMPERATE_FOREST

        return state
