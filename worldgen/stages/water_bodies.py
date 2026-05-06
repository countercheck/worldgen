from collections import deque

from ..core.hex import TerrainClass
from ..core.hex_grid import neighbors
from ..core.pipeline import GeneratorStage
from ..core.world_state import WorldState


class WaterBodiesStage(GeneratorStage):
    """Classify water hexes as OCEAN (map-edge-connected) or LAKE (inland).

    TerrainClassificationStage assigns OCEAN to every hex below sea level.
    This stage flood-fills connected water components: any component that
    touches the map border keeps TerrainClass.OCEAN; inland components are
    reclassified to TerrainClass.LAKE.

    A follow-up pass fixes COAST hexes that are now adjacent only to lakes
    (not open ocean) by re-evaluating their terrain class.
    """

    def run(self, state: WorldState) -> WorldState:
        w, h = state.width, state.height
        hexes = state.hexes

        water: set = {c for c, hx in hexes.items() if hx.terrain_class == TerrainClass.OCEAN}
        visited: set = set()

        for seed in water:
            if seed in visited:
                continue
            component = _bfs_component(seed, water)
            visited |= component
            touches_edge = any(_on_border(c, w, h) for c in component)
            if not touches_edge:
                for c in component:
                    hexes[c].terrain_class = TerrainClass.LAKE

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


def _on_border(coord, w: int, h: int) -> bool:
    q, r = coord
    return q == 0 or q == w - 1 or r == 0 or r == h - 1


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
    mountain_gradient = cfg_dict.get("terrain_mountain_gradient", 0.04)
    hill_gradient = cfg_dict.get("terrain_hill_gradient", 0.02)

    for coord, hx in hexes.items():
        if hx.terrain_class != TerrainClass.COAST:
            continue
        nbrs = [hexes[n] for n in neighbors(coord) if n in hexes]
        adjacent_to_ocean = any(n.terrain_class == TerrainClass.OCEAN for n in nbrs)
        if adjacent_to_ocean:
            continue  # correctly COAST

        # Not adjacent to open ocean — reclassify
        elev = hx.elevation
        if elev < coast_threshold and any(n.terrain_class == TerrainClass.LAKE for n in nbrs):
            # Low-elevation land beside a lake — leave as COAST (lake shore)
            # so downstream stages can treat it like coastal terrain if desired.
            continue

        neighbor_elevs = [n.elevation for n in nbrs]
        gradient = (
            sum(abs(elev - ne) for ne in neighbor_elevs) / len(neighbor_elevs)
            if neighbor_elevs
            else 0.0
        )
        if gradient > mountain_gradient or elev > 0.8:
            hx.terrain_class = TerrainClass.MOUNTAIN
        elif gradient >= hill_gradient:
            hx.terrain_class = TerrainClass.HILL
        else:
            hx.terrain_class = TerrainClass.FLAT
