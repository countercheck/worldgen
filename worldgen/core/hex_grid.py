import heapq
import math
from collections import deque
from collections.abc import Callable

from .hex import Hex, HexCoord, TerrainClass

# Grid layouts.  Both store hexes under axial coordinates — only the *set* of hexes a
# world is built from differs, so adjacency, distance and pathfinding are unaffected.
#
#   AXIAL   q in [0, width), r in [0, height).  A rhombus in axial space, which the
#           flat-top pixel transform shears into a parallelogram: the drawn map is a
#           leaning diamond with a straight edge on every side.
#   OFFSET  odd-q offset coordinates: column in [0, width), row in [0, height), stored
#           as the axial coord that column/row names.  The drawn map is a rectangle,
#           with the north and south edges ragged because odd columns sit half a hex
#           lower than even ones.
AXIAL = "axial"
OFFSET = "offset"
GRID_LAYOUTS = (AXIAL, OFFSET)


def offset_to_axial(col: int, row: int) -> HexCoord:
    """Odd-q offset column/row to axial (flat-top layout)."""
    return col, row - (col - (col & 1)) // 2


def axial_to_offset(coord: HexCoord) -> tuple[int, int]:
    """Axial to odd-q offset column/row (flat-top layout)."""
    q, r = coord
    return q, r + (q - (q & 1)) // 2


def grid_coord(layout: str, col: int, row: int) -> HexCoord:
    """The hex coordinate *layout* stores at grid column/row."""
    return offset_to_axial(col, row) if layout == OFFSET else (col, row)


def grid_index(layout: str, coord: HexCoord) -> tuple[int, int]:
    """The grid column/row *layout* stores *coord* at — the inverse of `grid_coord`."""
    return axial_to_offset(coord) if layout == OFFSET else coord


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


_DIRECTIONS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]


def ring(center: HexCoord, radius: int) -> list[HexCoord]:
    """All hexes at exactly radius distance from center.

    Walks the ring one corner at a time: start `radius` steps along one direction, then
    take `radius` steps along each of the six in turn, which closes the loop exactly.
    """
    if radius <= 0:
        return [center]

    q = center[0] + _DIRECTIONS[4][0] * radius
    r = center[1] + _DIRECTIONS[4][1] * radius

    results = []
    for dq, dr in _DIRECTIONS:
        for _ in range(radius):
            results.append((q, r))
            q, r = q + dq, r + dr

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


def astar_to_any(
    grid: dict[HexCoord, Hex],
    start: HexCoord,
    goals,
    cost_fn: Callable[[Hex], float],
    edge_cost_fn: Callable[[Hex, Hex], float] | None = None,
    aim: HexCoord | None = None,
    goal_cost: dict | None = None,
) -> list[HexCoord] | None:
    """`astar`, but it ends at the best hex in *goals* rather than at one named hex.

    What it is for: a traveller bound for a town does not need a road of his own the whole
    way there, he needs to reach the road that already goes there.  So the search runs
    against the set of hexes from which the destination is already reachable, and stops at
    whichever it touches first.

    *aim* steers the heuristic — the destination itself, so the search heads the right way
    rather than sprawling.  Without it this is a Dijkstra that happens to stop early.

    *goal_cost* is what remains to be travelled after reaching each goal, so that the search
    minimises the whole journey rather than the cost of reaching the network.  Those are not
    the same thing: without it a traveller joins the road at whatever hex is cheapest to
    reach and follows it however far round it goes.

    **Pass `goal_cost` only with `aim=None`.**  The two are in different units — `goal_cost`
    is real cost, while the heuristic counts hexes at 1.0 apiece, and road travel costs a
    fraction of that.  Mixing them makes the cutoff below fire on the first expansion, so
    the search returns the network route every time without ever looking at an alternative.
    With `aim=None` the frontier is ordered by true cost and the comparison is sound.
    """
    if start not in grid or not goals:
        return None
    if start in goals:
        return [start]

    open_set = [(0.0, start)]
    came_from: dict = {start: None}
    g_score = {start: 0.0}
    visited = set()
    best_total, best_node = float("inf"), None

    while open_set:
        f, current = heapq.heappop(open_set)
        # Nothing left on the frontier can better the journey already found. Sound only
        # because f and best_total are in the same units, which is what `aim` would break.
        if f > best_total:
            break
        if current in visited:
            continue
        visited.add(current)

        if current in goals:
            total = g_score[current] + (goal_cost.get(current, 0.0) if goal_cost else 0.0)
            if total < best_total:
                best_total, best_node = total, current
            # With no residual to weigh, the first goal reached is the cheapest reached.
            if goal_cost is None:
                break

        for neighbor in neighbors(current):
            if neighbor not in grid or neighbor in visited:
                continue
            cost = cost_fn(grid[neighbor])
            if edge_cost_fn is not None:
                cost += edge_cost_fn(grid[current], grid[neighbor])
            if cost == float("inf"):
                continue
            tentative_g = g_score[current] + cost
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                h = distance(neighbor, aim) if aim is not None else 0.0
                heapq.heappush(open_set, (tentative_g + h, neighbor))

    if best_node is None:
        return None
    path = []
    node = best_node
    while node is not None:
        path.append(node)
        node = came_from[node]
    return list(reversed(path))


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


def road_water_transitions(road_edges, hexes: dict[HexCoord, Hex]) -> set[HexCoord]:
    """Land hexes where the road network meets water — where a route takes to a boat.

    `road_polylines` drops edges with a foot in a lake or the sea, so a crossing leaves two
    drawn lines stopping dead at opposite shores.  Renderers mark these hexes so the two
    read as one route.  The graph answers this directly: a land hex holding a road edge
    into water is a landing.
    """
    out: set[HexCoord] = set()
    for a, b in road_edges:
        a_wet, b_wet = _is_water(hexes, a), _is_water(hexes, b)
        if a_wet != b_wet:
            out.add(b if a_wet else a)
    return out


def road_polylines(road_edges, hexes: dict[HexCoord, Hex]) -> list[tuple]:
    """The road graph as drawable runs: `(tier, polyline)` pairs, each edge appearing once.

    Replaces `dedupe_road_paths`, which existed only to undo the old representation.  When
    the model stored a whole journey per settlement pair, routes overlapped almost
    completely and every renderer had to award each shared edge to its highest-ranked user
    before drawing.  One tier per edge makes that step unnecessary: there is nothing left
    to deduplicate, and this only has to decide where one drawn line stops and the next
    begins.

    A run breaks at three things — a junction, a change of tier, and water.  Junctions,
    because a line through a fork would draw an edge that is not there; tier, because a
    polyline carries one style; water, because a road may path across a lake (see
    `road_cost.py`) but a straight line drawn over open water reads as a bridge.

    Results come back in ascending tier rank, so a renderer drawing them in order paints
    primary roads last and a track never overdraws a highway.
    """
    from .world_state import ROAD_TIER_RANK

    # Water is dropped rather than split around: an edge with a foot in a lake is part of
    # a crossing, and `water_transitions` is what marks where the line resumes.
    adjacency: dict[HexCoord, list[tuple[HexCoord, object]]] = {}
    edges: dict[frozenset, object] = {}
    for (a, b), tier in road_edges.items():
        if _is_water(hexes, a) or _is_water(hexes, b):
            continue
        edges[frozenset((a, b))] = tier
        adjacency.setdefault(a, []).append((b, tier))
        adjacency.setdefault(b, []).append((a, tier))

    def continues(coord: HexCoord, tier) -> list[HexCoord]:
        """Neighbours reachable from *coord* along an edge of the same tier."""
        return [n for n, t in adjacency.get(coord, ()) if t == tier]

    out: list[tuple] = []
    walked: set[frozenset] = set()

    def walk(start: HexCoord, first: HexCoord, tier) -> None:
        run = [start, first]
        walked.add(frozenset((start, first)))
        while True:
            here = run[-1]
            # Carry on only through a plain degree-2 node of this tier: anywhere else is a
            # junction, and the next run starts there rather than sweeping through it.
            onward = [n for n in continues(here, tier) if n != run[-2]]
            if len(onward) != 1 or len(continues(here, tier)) != 2:
                break
            edge = frozenset((here, onward[0]))
            if edge in walked:
                break
            walked.add(edge)
            run.append(onward[0])
        out.append((tier, run))

    # Open runs first, from every end and junction, so the interior of a corridor is only
    # reached along it rather than started in the middle.
    for coord in sorted(adjacency):
        for nbr, tier in sorted(adjacency[coord]):
            if len(continues(coord, tier)) == 2 or frozenset((coord, nbr)) in walked:
                continue
            walk(coord, nbr, tier)

    # Whatever is left is a closed loop of one tier, with no end to start from.
    for coord in sorted(adjacency):
        for nbr, tier in sorted(adjacency[coord]):
            if frozenset((coord, nbr)) not in walked:
                walk(coord, nbr, tier)

    out.sort(key=lambda pair: ROAD_TIER_RANK[pair[0]])
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
