from ..core.hex import Biome, TerrainClass
from ..core.hex_grid import neighbors
from ..core.pipeline import GeneratorStage
from ..core.world_state import WorldState

_AGRI_SCORE: dict[Biome, float] = {
    Biome.GRASSLAND: 1.0,
    Biome.TROPICAL: 0.8,
    Biome.TEMPERATE_FOREST: 0.7,
    Biome.SHRUBLAND: 0.5,
    Biome.BOREAL: 0.3,
    Biome.DESERT: 0.0,
    Biome.TUNDRA: 0.0,
    Biome.ALPINE: 0.0,
    Biome.WETLAND: 0.0,
    Biome.OCEAN: 0.0,
}


class HabitabilityStage(GeneratorStage):
    def run(self, state: WorldState) -> WorldState:
        hexes = state.hexes
        raw: dict = {}

        for coord, hx in hexes.items():
            if (
                hx.terrain_class == TerrainClass.OCEAN
                or hx.terrain_class == TerrainClass.MOUNTAIN
                or hx.biome == Biome.WETLAND
            ):
                raw[coord] = 0.0
                continue

            score = 0.0
            nbrs = [hexes[n] for n in neighbors(coord) if n in hexes]

            # River adjacency: hex itself or any neighbor carries flow
            if hx.river_flow > 0 or any(n.river_flow > 0 for n in nbrs):
                score += 0.35

            # Agricultural potential
            score += 0.25 * _AGRI_SCORE.get(hx.biome, 0.0)

            # Defensibility: hill overlooking a plain
            if hx.terrain_class == TerrainClass.HILL and any(
                n.terrain_class == TerrainClass.FLAT for n in nbrs
            ):
                score += 0.15

            # Coastal access (including the hex itself)
            if hx.terrain_class == TerrainClass.COAST or any(
                n.terrain_class == TerrainClass.COAST for n in nbrs
            ):
                score += 0.15

            # Confluence bonus: river junction hexes → historically where cities form
            if "confluence" in hx.tags:
                score += 0.10

            raw[coord] = score

        max_score = max(raw.values()) if raw else 1.0
        if max_score == 0.0:
            max_score = 1.0

        for coord, score in raw.items():
            hexes[coord].habitability = score / max_score

        return state
