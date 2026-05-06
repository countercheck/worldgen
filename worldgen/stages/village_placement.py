from ..core.hex import LandCover, Settlement, SettlementTier, TerrainClass
from ..core.hex_grid import distance, neighbors
from ..core.pipeline import GeneratorStage
from ..core.world_state import WorldState
from .city_town import _assign_role

_RESISTANT = {
    LandCover.BOG,
    LandCover.MARSH,
    LandCover.BARE_ROCK,
    LandCover.ALPINE,
    LandCover.TUNDRA,
    LandCover.DESERT,
    LandCover.OPEN_WATER,
}


class VillagePlacementStage(GeneratorStage):
    def run(self, state: WorldState) -> WorldState:
        hexes = state.hexes

        placed_coords = [s.coord for s in state.settlements]

        candidates = []
        weights = []
        for coord, hx in hexes.items():
            if hx.terrain_class == TerrainClass.OCEAN:
                continue
            if hx.settlement is not None:
                continue
            if hx.habitability <= 0:
                continue
            if hx.land_cover in _RESISTANT:
                continue

            on_frontier = hx.cultivated and any(
                not hexes[n].cultivated
                for n in neighbors(coord)
                if n in hexes and hexes[n].terrain_class != TerrainClass.OCEAN
            )
            road_adjacent = bool(hx.road_connections) or any(
                hexes[n].road_connections for n in neighbors(coord) if n in hexes
            )

            if not on_frontier and not road_adjacent:
                continue

            w = hx.habitability
            if on_frontier:
                w *= 2.0
            if road_adjacent:
                w *= 1.5

            candidates.append(coord)
            weights.append(w)

        if not candidates:
            return state

        total_w = sum(weights)
        probs = [w / total_w for w in weights]

        u = self.rng.random(len(candidates))
        order = sorted(range(len(candidates)), key=lambda i: -(u[i] ** (1.0 / max(probs[i], 1e-9))))

        village_idx = 0
        for i in order:
            coord = candidates[i]
            hx = hexes[coord]
            if hx.settlement is not None:
                continue
            if all(distance(coord, c) >= 3 for c in placed_coords):
                pop = int(self.rng.integers(100, 1_001))
                role = _assign_role(coord, hx, hexes)
                name = f"{hx.biome.name.lower()}_village_{village_idx}"
                s = Settlement(
                    coord=coord,
                    tier=SettlementTier.VILLAGE,
                    role=role,
                    population=pop,
                    name=name,
                )
                hx.settlement = s
                placed_coords.append(coord)
                state.settlements.append(s)
                village_idx += 1

        return state
