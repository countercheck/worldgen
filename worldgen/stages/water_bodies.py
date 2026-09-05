from collections import deque

from ..core.hex import TerrainClass
from ..core.hex_grid import neighbors
from ..core.pipeline import GeneratorStage
from ..core.world_state import WorldState


class WaterBodiesStage(GeneratorStage):
    """Sort water into what drains off the map and what does not.

    TerrainClassificationStage assigns OPEN_WATER to every hex below sea level.
    This stage flood-fills connected water components: any component that
    touches the map border keeps TerrainClass.OPEN_WATER; inland components are
    reclassified to TerrainClass.INLAND_WATER.

    A follow-up pass fixes COAST hexes that are now adjacent only to lakes
    (not open ocean) by re-evaluating their terrain class.
    """

    def run(self, state: WorldState) -> WorldState:
        hexes = state.hexes

        water: set = {c for c, hx in hexes.items() if hx.terrain_class == TerrainClass.OPEN_WATER}
        visited: set = set()

        for seed in water:
            if seed in visited:
                continue
            component = _bfs_component(seed, water)
            visited |= component
            touches_edge = any(state.on_border(c) for c in component)
            if not touches_edge:
                for c in component:
                    hexes[c].terrain_class = TerrainClass.INLAND_WATER

        _fix_coast_hexes(state)
        return state


def _bfs_component(seed, water: set) -> set:
    """Return all water hexes reachable from seed."""
    component: set = {seed}
    queue: deque = deque([seed])
    while queue:
        coord = queue.popleft()
        for nbr in neighbors(coord):
            if nbr in water and nbr not in component:
                component.add(nbr)
                queue.append(nbr)
    return component


def _fix_coast_hexes(state: WorldState) -> None:
    """Re-classify COAST hexes that border only lakes (not open ocean).

    TerrainClassificationStage runs before water body labelling, so it
    tagged lake-adjacent land as COAST.  Now that lakes are identified,
    we correct those hexes using the same gradient-based logic as the
    original terrain classification.
    """
    hexes = state.hexes
    cfg_dict = state.metadata.get("config", {})
    sea = cfg_dict.get("sea_level", 0.45)
    coast_threshold = sea + 0.05

    for coord, hx in hexes.items():
        if hx.terrain_class != TerrainClass.COAST:
            continue
        nbrs = [hexes[n] for n in neighbors(coord) if n in hexes]
        adjacent_to_ocean = any(n.terrain_class == TerrainClass.OPEN_WATER for n in nbrs)
        if adjacent_to_ocean:
            continue  # correctly COAST

        # Not adjacent to open ocean — reclassify
        elev = hx.elevation
        if elev < coast_threshold and any(
            n.terrain_class == TerrainClass.INLAND_WATER for n in nbrs
        ):
            # Low-elevation land beside a lake — leave as COAST (lake shore)
            # so downstream stages can treat it like coastal terrain if desired.
            continue

        # Not a shore after all, so it is simply land.  This used to re-derive a
        # steepness band here, duplicating the classification stage's arithmetic to pick
        # between three labels; with steepness carried on the hex as a number there is
        # one answer and no sum to repeat.
        hx.terrain_class = TerrainClass.LAND
