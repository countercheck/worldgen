import heapq
import math
from collections import deque
from collections.abc import Callable

from .hex import Hex, HexCoord, TerrainClass


def neighbors(coord: HexCoord) -> list[HexCoord]:
    """Six neighbors in axial coordinates."""
    q, r = coord
    return [
        (q + 1, r),
        (q + 1, r - 1),
        (q, r - 1),
        (q - 1, r),
        (q - 1, r + 1),
        (q, r + 1),
    ]


def distance(a: HexCoord, b: HexCoord) -> int:
    """Manhattan distance in axial coordinates."""
    qa, ra = a
    qb, rb = b
    return (abs(qa - qb) + abs(ra - rb) + abs((qa + ra) - (qb + rb))) // 2


def ring(center: HexCoord, radius: int) -> list[HexCoord]:
    """All hexes at exactly radius distance from center."""
    if radius == 0:
        return [center]

    results = []
    qc, rc = center
    q, r = qc + radius, rc - radius

    for _ in range(6):
        for _ in range(radius):
            results.append((q, r))
            q, r = q - 1, r + 1
        q, r = q + 1, r

    return results


def hex_range(center: HexCoord, radius: int) -> list[HexCoord]:
    """All hexes within radius distance from center."""
    results = []
    for r in range(radius + 1):
        results.extend(ring(center, r))
    return results


def axial_to_pixel(coord: HexCoord, hex_size: float) -> tuple[float, float]:
    """Convert axial coordinates to pixel (flat-top layout)."""
    q, r = coord
    x = hex_size * (3.0 / 2 * q)
    y = hex_size * (math.sqrt(3) / 2 * q + math.sqrt(3) * r)
    return x, y


def pixel_to_axial(x: float, y: float, hex_size: float) -> HexCoord:
    """Convert pixel to axial coordinates (flat-top layout)."""
    q = (2.0 / 3 * x) / hex_size
    r = (-1.0 / 3 * x + math.sqrt(3) / 3 * y) / hex_size
    return round_axial((q, r))


def round_axial(coord: tuple[float, float]) -> HexCoord:
    """Round fractional axial coordinates to nearest hex."""
    q, r = coord
    s = -q - r
    rq, rr, rs = round(q), round(r), round(s)

    q_diff, r_diff, s_diff = abs(rq - q), abs(rr - r), abs(rs - s)

    if q_diff > r_diff and q_diff > s_diff:
        rq = -rr - rs
    elif r_diff > s_diff:
        rr = -rq - rs

    return int(rq), int(rr)


def astar(
    grid: dict[HexCoord, Hex],
    start: HexCoord,
    goal: HexCoord,
    cost_fn: Callable[[Hex], float],
    edge_cost_fn: Callable[[Hex, Hex], float] | None = None,
) -> list[HexCoord] | None:
    """A* pathfinding. cost_fn returns node entry cost (inf = impassable).
    edge_cost_fn(from_hex, to_hex) adds an optional per-edge cost (e.g. slope)."""
    if start not in grid or goal not in grid:
        return None

    open_set = [(0, start)]
    came_from = {start: None}
    g_score = {start: 0.0}

    visited = set()

    while open_set:
        _, current = heapq.heappop(open_set)

        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = came_from[node]
            return list(reversed(path))

        for neighbor in neighbors(current):
            if neighbor not in grid or neighbor in visited:
                continue

            cost = cost_fn(grid[neighbor])

            if edge_cost_fn is not None:
                cost += edge_cost_fn(grid[current], grid[neighbor])

            # Checked after the edge term so an impassable *edge* is skipped too. Adding
            # inf and pushing the node instead would let the search reach the goal with an
            # infinite score and return a path straight through the forbidden edge.
            if cost == float("inf"):
                continue

            tentative_g = g_score[current] + cost

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                h = distance(neighbor, goal)
                f = tentative_g + h
                heapq.heappush(open_set, (f, neighbor))

    return None


def _is_water(hexes: dict[HexCoord, Hex], coord: HexCoord) -> bool:
    """Water test used by the path helpers.

    A coord absent from the grid has no terrain to judge, so it counts as land — the
    same way `split_path_on_water` keeps it rather than cutting the path there.
    """
    hx = hexes.get(coord)
    return hx is not None and hx.terrain_class in (TerrainClass.OCEAN, TerrainClass.LAKE)


def water_transitions(path: list[HexCoord], hexes: dict[HexCoord, Hex]) -> list[HexCoord]:
    """Land hexes where *path* meets water — the points a route embarks and lands.

    Roads may cross water (see road_cost.py), but the water leg is not drawn, so those
    land legs otherwise appear to stop dead at a shore.  Renderers mark these hexes so a
    route that continues by boat reads as one.  Returns the *land* side of each
    transition, in path order, with repeats collapsed.
    """
    out: list[HexCoord] = []
    for prev, cur in zip(path, path[1:], strict=False):
        prev_water = _is_water(hexes, prev)
        if prev_water != _is_water(hexes, cur):
            land = cur if prev_water else prev
            if not out or out[-1] != land:
                out.append(land)
    return out


def dedupe_road_paths(roads, hexes: dict[HexCoord, Hex], rank) -> list[tuple]:
    """Split roads into land legs with every map edge drawn exactly once.

    Routes between different settlement pairs share trunk segments, so a naive render
    stacks many polylines on the same edge — mostly invisible, but where the tiers differ
    a thin dashed track paints over a primary road, and the redundancy bloats the output.

    Each edge is awarded to the highest-ranked road using it (`rank(road) -> int`, higher
    wins; ties go to the first road in list order).  Results come back in ascending rank
    so a renderer drawing them in order paints important roads last, on top.  Returns
    (road, polyline) pairs; the road carries whatever styling the caller needs.
    """
    best: dict[frozenset, int] = {}
    for road in roads:
        r = rank(road)
        for a, b in zip(road.path, road.path[1:], strict=False):
            edge = frozenset((a, b))
            if best.get(edge, -1) < r:
                best[edge] = r

    out: list[tuple] = []
    claimed: set[frozenset] = set()
    for road in sorted(roads, key=rank):  # stable, so ties keep list order
        r = rank(road)
        for leg in split_path_on_water(road.path, hexes):
            run: list[HexCoord] = []
            for a, b in zip(leg, leg[1:], strict=False):
                edge = frozenset((a, b))
                if best[edge] == r and edge not in claimed:
                    claimed.add(edge)
                    if not run:
                        run.append(a)
                    run.append(b)
                else:
                    if len(run) >= 2:
                        out.append((road, run))
                    run = []
            if len(run) >= 2:
                out.append((road, run))
    return out


def split_path_on_water(path: list[HexCoord], hexes: dict[HexCoord, Hex]) -> list[list[HexCoord]]:
    """Split a hex path into contiguous land-only runs, dropping OCEAN/LAKE hexes.

    Roads may path through water (see road_cost.py), but rendering a straight
    line across open water is misleading, so renderers use this to draw only
    the land legs of a route."""
    segments: list[list[HexCoord]] = []
    current: list[HexCoord] = []
    for coord in path:
        hx = hexes.get(coord)
        if hx is not None and hx.terrain_class in (TerrainClass.OCEAN, TerrainClass.LAKE):
            if len(current) >= 2:
                segments.append(current)
            current = []
        else:
            current.append(coord)
    if len(current) >= 2:
        segments.append(current)
    return segments


def grade_reachable_count(
    start: HexCoord,
    hexes: dict[HexCoord, Hex],
    grade_ok: Callable[[Hex, Hex], bool],
    max_count: int,
) -> int:
    """BFS from start over non-water hexes where grade_ok(from_hex, to_hex) is True.
    Returns the number of reachable hexes, stopping once max_count is reached.
    If start is missing or water, returns 0."""
    if start not in hexes:
        return 0
    if hexes[start].terrain_class in (TerrainClass.OCEAN, TerrainClass.LAKE):
        return 0

    visited: set[HexCoord] = {start}
    q: deque[HexCoord] = deque([start])
    count = 0
    while q and count < max_count:
        coord = q.popleft()
        count += 1
        for nb in neighbors(coord):
            if nb not in hexes or nb in visited:
                continue
            nb_hx = hexes[nb]
            if nb_hx.terrain_class in (TerrainClass.OCEAN, TerrainClass.LAKE):
                continue
            if grade_ok(hexes[coord], nb_hx):
                visited.add(nb)
                q.append(nb)
    return count
