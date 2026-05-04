from ..core.hex import TerrainClass
from ..core.hex_grid import neighbors
from ..core.pipeline import GeneratorStage
from ..core.world_state import WorldState


class TerrainClassificationStage(GeneratorStage):
    def run(self, state: WorldState) -> WorldState:
        sea = self.config.sea_level
        coast_threshold = sea + 0.05

        # Pass 1: assign OCEAN
        for h in state.hexes.values():
            if h.elevation < sea:
                h.terrain_class = TerrainClass.OCEAN

        # Pass 2: classify land hexes
        for (q, r), h in state.hexes.items():
            if h.terrain_class == TerrainClass.OCEAN:
                continue

            elev = h.elevation
            nbrs = [state.hexes[n] for n in neighbors((q, r)) if n in state.hexes]

            # COAST: low-elevation land adjacent to ocean
            if elev < coast_threshold and any(n.terrain_class == TerrainClass.OCEAN for n in nbrs):
                h.terrain_class = TerrainClass.COAST
                continue

            neighbor_elevs = [n.elevation for n in nbrs]
            if neighbor_elevs:
                gradient = sum(abs(elev - ne) for ne in neighbor_elevs) / len(neighbor_elevs)
            else:
                gradient = 0.0

            if gradient > self.config.terrain_mountain_gradient or elev > 0.8:
                h.terrain_class = TerrainClass.MOUNTAIN
            elif gradient >= self.config.terrain_hill_gradient:
                h.terrain_class = TerrainClass.HILL
            else:
                h.terrain_class = TerrainClass.FLAT

        return state
