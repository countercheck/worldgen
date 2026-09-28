import heapq
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

    Then the closed hollows on land: a big, deep one fills to its rim and becomes a lake,
    a small or shallow one is tagged `hollow` for `BiomeStage` to waterlog.

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

        _fill_hollows(state, self.config)
        _fix_coast_hexes(state, self.config)
        return state


def _fill_hollows(state: WorldState, cfg) -> None:
    """Make lakes of the closed hollows big enough to hold one, and tag the rest.

    A hollow is land below the level water would have to rise to before it could run out:
    a Priority-Flood from the sea, the lakes and the map edge finds that level for every
    hex, and a hollow is a connected run of hexes lying under it. Water fills a hollow to
    its rim and spills over, so in a wet enough region a big one *is* a lake, standing at
    the rim with a river leaving it. Sub-sea-level basins were the only lakes before, and
    only erosion dug those; a hollow above sea level was filled flat for routing, and every
    river crossing it ran straight at the exit.
    """
    hexes = state.hexes
    water = {
        c
        for c, hx in hexes.items()
        if hx.terrain_class in (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER)
    }
    level = {c: hx.elevation for c, hx in hexes.items()}
    seen = set(water) | {c for c in hexes if c not in water and state.on_border(c)}
    heap = [(level[c], c) for c in seen]
    heapq.heapify(heap)
    while heap:
        e, coord = heapq.heappop(heap)
        for nbr in neighbors(coord):
            if nbr in hexes and nbr not in seen:
                seen.add(nbr)
                level[nbr] = max(level[nbr], e)
                heapq.heappush(heap, (level[nbr], nbr))

    under = {c for c in hexes if c not in water and level[c] > hexes[c].elevation}
    wet_enough = cfg.runoff_mm(cfg.mean_precip_mm) >= cfg.lake_min_runoff_mm
    # Sorted so the hollows are found in the same order every run.
    for start in sorted(under):
        if start not in under:
            continue
        hollow = _bfs_component(start, under)
        under -= hollow
        depth = max(level[c] - hexes[c].elevation for c in hollow)
        if wet_enough and len(hollow) >= cfg.lake_min_hexes and depth >= cfg.lake_min_depth_m:
            for c in hollow:
                hexes[c].terrain_class = TerrainClass.INLAND_WATER
        elif depth >= cfg.hollow_wetland_min_depth_m:
            for c in hollow:
                hexes[c].tags.add("hollow")

    # A lake whose floor is uneven near its rim comes out as open water laced with islands a
    # hex or two across. An island that small, standing a few metres out of a lake, is reed
    # bed and carr rather than dry ground: tag it to waterlog with the hollows.
    lake = {c for c, hx in hexes.items() if hx.terrain_class == TerrainClass.INLAND_WATER}
    dry = {c for c in hexes if c not in water and c not in lake}
    for start in sorted(dry):
        if start not in dry:
            continue
        island = _bfs_component(start, dry)
        dry -= island
        if len(island) >= cfg.lake_min_hexes or any(state.on_border(c) for c in island):
            continue
        rim = {n for c in island for n in neighbors(c) if n in hexes and n not in island}
        if rim and rim <= lake:
            for c in island:
                hexes[c].tags.add("hollow")


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


def _fix_coast_hexes(state: WorldState, cfg) -> None:
    """Re-classify COAST hexes that border only lakes (not open ocean).

    TerrainClassificationStage runs before water body labelling, so it
    tagged lake-adjacent land as COAST.  Now that lakes are identified,
    we correct those hexes using the same gradient-based logic as the
    original terrain classification.
    """
    hexes = state.hexes
    coast_threshold = cfg.coast_max_elevation_m

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
