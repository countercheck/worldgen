"""Which people lives where.

Language frontiers run along what is hard to cross — a mountain wall, a river too big to
wade, a strait — because that is where people stopped meeting each other often. So the map
is divided the way a flood would divide it from a handful of homelands: each spreads at a
cost, the cost is what the ground makes it, and a hex belongs to whichever homeland reaches
it cheapest. The frontiers fall on the ridges and the big rivers without either being
drawn.

Every hex is assigned, water included, so an island joins whoever holds the nearest
shore rather than being left out.
"""

import heapq

import numpy as np

from ..core.hex import Hex, HexCoord, TerrainClass
from ..core.hex_grid import distance, neighbors

_WATER = (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER)


def _homelands(hexes: dict[HexCoord, Hex], count: int, rng: np.random.Generator) -> list[HexCoord]:
    """Spread *count* homelands over the land, each as far from the others as it can be.

    Farthest-point sampling from one random start: a random scatter puts two homelands in
    one valley often enough to leave a people with a single village.
    """
    land = sorted(c for c, h in hexes.items() if h.terrain_class not in _WATER)
    if not land:
        return []
    chosen = [land[int(rng.integers(len(land)))]]
    nearest = {c: distance(c, chosen[0]) for c in land}
    while len(chosen) < min(count, len(land)):
        # Ties broken on the coordinate, so the draw alone decides the result.
        far = max(land, key=lambda c: (nearest[c], c))
        chosen.append(far)
        for c in land:
            nearest[c] = min(nearest[c], distance(c, far))
    return chosen


def _step_cost(
    a: Hex, b: Hex, climb_m: float, river_cost: float, great_river_km2: float, water_cost: float
) -> float:
    cost = 1.0
    if climb_m > 0:
        cost += abs(b.elevation - a.elevation) / climb_m
    if b.terrain_class in _WATER:
        cost += water_cost
    elif "river" in b.tags and b.catchment_km2 >= great_river_km2:
        cost += river_cost
    return cost


def culture_regions(
    hexes: dict[HexCoord, Hex],
    count: int,
    rng: np.random.Generator,
    climb_m: float,
    river_cost: float,
    great_river_km2: float,
    water_cost: float,
    homes: list[HexCoord] | None = None,
) -> dict[HexCoord, int]:
    """Map every hex to the index of the culture that holds it, 0 to *count* - 1.

    A multi-source cheapest-path flood, so each region is connected: a hex is only ever
    claimed through a neighbour its own culture already holds. *homes* places the
    homelands by hand; otherwise they are spread over the land from *rng*.
    """
    if homes is None:
        homes = _homelands(hexes, count, rng)
    owner: dict[HexCoord, int] = {}
    best: dict[HexCoord, float] = {}
    frontier: list[tuple[float, HexCoord, int]] = []
    for i, home in enumerate(homes):
        best[home] = 0.0
        heapq.heappush(frontier, (0.0, home, i))
    while frontier:
        cost, coord, culture = heapq.heappop(frontier)
        if coord in owner:
            continue
        owner[coord] = culture
        here = hexes[coord]
        for n in neighbors(coord):
            if n not in hexes or n in owner:
                continue
            step = _step_cost(here, hexes[n], climb_m, river_cost, great_river_km2, water_cost)
            if cost + step < best.get(n, float("inf")):
                best[n] = cost + step
                heapq.heappush(frontier, (cost + step, n, culture))
    return owner


__all__ = ["culture_regions"]
