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

        # Pass 2: measure the ground, then classify what is genuinely categorical.
        #
        # Slope and relief are recorded per hex rather than folded into a class, because
        # steepness is a continuum: thresholding it here made six downstream stages read a
        # label instead of the terrain, and a hex is either side of a cutoff for reasons
        # that have nothing to do with what is being asked of it.
        for (q, r), h in state.hexes.items():
            elev = h.elevation
            nbrs = [state.hexes[n] for n in neighbors((q, r)) if n in state.hexes]
            neighbor_elevs = [n.elevation for n in nbrs]
            if neighbor_elevs:
                gradient = sum(abs(elev - ne) for ne in neighbor_elevs) / len(neighbor_elevs)
                h.relief = elev - min(neighbor_elevs)
            else:
                gradient = 0.0
                h.relief = 0.0
            h.slope = gradient

            if h.terrain_class == TerrainClass.OCEAN:
                continue

            # COAST: low-elevation land adjacent to ocean
            if elev < coast_threshold and any(n.terrain_class == TerrainClass.OCEAN for n in nbrs):
                h.terrain_class = TerrainClass.COAST
                continue

            h.terrain_class = TerrainClass.LAND

        return state
