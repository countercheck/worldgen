"""Drainage on the corner graph: how water runs along the sides between hexes.

Rivers run along hexsides, so the network water flows through is not the hexes but their
corners.  Every corner touches three hexes and three other corners, one side away each.
This module builds that graph over a world, fills its depressions, gives each corner
somewhere lower to go, and accumulates rain down it — the same four steps hydrology has
always taken on hexes, on a graph with half the connectivity and twice the nodes.

What the graph is
-----------------
*   A **corner** is a node if any of its three hexes is land.  Its height is the lowest
    of its land hexes — a corner is where valley sides meet, so it sits at the floor —
    lifted a little toward their mean, so a corner against a valley side stands above
    one in the middle of the floor and water gathers along the floor's middle.
*   Water may run between two corners only along a side with **land on both sides**.  A
    side with water on one hand is a shoreline and a side with the map edge on one hand is
    the border; a river running along either would be a line drawn beside the water or
    beside the frame, not a watercourse.
*   A corner touching the sea, a closed lake, or the map edge is a **terminal**: water
    reaching it leaves the network there.  The ones touching water are *wet*.
*   An **open lake** — one hydrology has given an outlet — is a single node of its own,
    joined to every corner of its shore and sitting at its water level.  Water reaching
    the shore runs into it, and it spills wherever the fill first reached it from below.
    Lake nodes are named ``(q, r, 2)`` after the lake's least hex, so they sort and hash
    with the corners (whose third component is 0 or 1) without ever colliding with one.

Pure functions over plain dicts: nothing here reads a `WorldState` beyond the hexes it is
given, writes anything, or draws a random number except through the generator passed in.
"""

import heapq
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np

from ..core.hex import HexCoord
from ..core.hex_grid import Corner, corner_hexes, hex_corner_keys, hex_side_keys, side_corners
from ..core.hex_grid import side_hexes as _side_hexes

Node = tuple[int, int, int]

LAKE = 2


def is_lake_node(node: Node) -> bool:
    return node[2] == LAKE


@dataclass
class CornerNetwork:
    """The corner graph of a world, ready to be drained."""

    elevation: dict[Node, float] = field(default_factory=dict)
    neighbors: dict[Node, list[Node]] = field(default_factory=dict)
    terminal: set[Node] = field(default_factory=set)
    # Terminals touching the sea or a closed lake, which water runs into rather than past.
    wet: set[Node] = field(default_factory=set)
    # Open lake hex -> its lake node, and back.
    lake_node_of: dict[HexCoord, Node] = field(default_factory=dict)
    lake_hexes: dict[Node, list[HexCoord]] = field(default_factory=dict)


def build_network(
    elevation: Mapping[HexCoord, float],
    land: set[HexCoord],
    ocean: set[HexCoord],
    closed_lakes: set[HexCoord],
    open_lakes: Iterable[Iterable[HexCoord]],
    floor_blend: float = 0.0,
) -> CornerNetwork:
    """The corner graph over the hexes of *elevation*, which are the map.

    *land*, *ocean* and *closed_lakes* partition the water and land hexes (anything in none
    of them is ignored); *open_lakes* lists the hexes of each lake that drains.  Taking
    heights rather than hexes lets erosion drain its own working surface the same way.

    A corner stands at the lowest of its land hexes plus *floor_blend* of the way to their
    mean (`corner_floor_blend`).  At 0, every corner of a valley-floor hex sits exactly at
    the floor, and the corners against the valley side are as low as the river's own.
    """
    hexes = elevation
    net = CornerNetwork()
    adjacency: dict[Node, set[Node]] = defaultdict(set)

    for coord in sorted(land):
        for corner in hex_corner_keys(coord):
            if corner in net.elevation:
                continue
            around = corner_hexes(corner)
            heights = [hexes[h] for h in around if h in land]
            low = min(heights)
            net.elevation[corner] = low + floor_blend * (sum(heights) / len(heights) - low)
            if any(h not in hexes for h in around):
                net.terminal.add(corner)
            if any(h in ocean or h in closed_lakes for h in around):
                net.terminal.add(corner)
                net.wet.add(corner)
        for side in hex_side_keys(coord):
            a, b = _side_hexes(side)
            if a in land and b in land:
                c1, c2 = side_corners(side)
                adjacency[c1].add(c2)
                adjacency[c2].add(c1)

    for comp in open_lakes:
        comp = sorted(comp)
        if not comp:
            continue
        node: Node = (comp[0][0], comp[0][1], LAKE)
        net.lake_hexes[node] = comp
        net.elevation[node] = min(hexes[h] for h in comp)
        for h in comp:
            net.lake_node_of[h] = node
            for corner in hex_corner_keys(h):
                if corner in net.elevation and corner not in net.terminal:
                    adjacency[corner].add(node)
                    adjacency[node].add(corner)

    net.neighbors = {n: sorted(adjacency.get(n, ())) for n in net.elevation}
    return net


@dataclass
class Drainage:
    """Where water goes on a `CornerNetwork`, and how much of it."""

    filled: dict[Node, float]
    # The order the fill reached each node in.  Strictly lower in this order is downhill,
    # which is what keeps the flow free of cycles on flats, where heights are equal.
    order: dict[Node, int]
    parent: dict[Node, Node]
    flow: dict[Node, Node | None]
    acc: dict[Node, float] = field(default_factory=dict)


def priority_flood(
    net: CornerNetwork,
) -> tuple[dict[Node, float], dict[Node, int], dict[Node, Node]]:
    """Fill every depression up to its spill height, outward from the terminals.

    Barnes' priority flood, with the heap broken by insertion order so that across a flat
    the fill spreads outward from where it drains, one step at a time.  The order nodes
    leave the heap is therefore a downhill order everywhere — on slopes because heights
    rise, on flats because distance from the drain does — and `parent` records, for every
    node, the neighbour the fill reached it from: a way down that always exists.
    """
    filled: dict[Node, float] = {}
    order: dict[Node, int] = {}
    parent: dict[Node, Node] = {}
    heap: list[tuple[float, int, Node]] = []
    counter = 0
    for node in sorted(net.terminal):
        filled[node] = net.elevation[node]
        heapq.heappush(heap, (filled[node], counter, node))
        counter += 1
    while heap:
        level, _, node = heapq.heappop(heap)
        if node in order:
            continue
        order[node] = len(order)
        for nbr in net.neighbors[node]:
            if nbr in filled:
                continue
            filled[nbr] = max(net.elevation[nbr], level)
            parent[nbr] = node
            heapq.heappush(heap, (filled[nbr], counter, nbr))
            counter += 1
    return filled, order, parent


def flow_direction(
    net: CornerNetwork,
    rng: np.random.Generator,
    wander_exponent: float,
    draws: Mapping[Node, float] | None = None,
) -> Drainage:
    """Fill the network, then send each node's water to one lower neighbour.

    A terminal goes nowhere.  An open lake spills where the fill reached it.  A corner
    beside the water runs into it, the steepest way if there is a choice.  Any other corner
    picks among its lower neighbours at random, weighted by ``(drop / largest drop) **
    wander_exponent`` — the same rule hydrology used on hexes, so a steep valley keeps to
    its floor and a flat one wanders.  "Lower" means earlier in the fill order, so a pick
    is always downhill and the network never loops.

    Draws are made in sorted node order, and only where there is a real choice.  With
    *draws*, each node uses its own fixed draw instead, so a surface drained again after
    it has changed keeps choosing the same way where the choice has not changed.
    """
    filled, order, parent = priority_flood(net)
    flow: dict[Node, Node | None] = {}
    for node in sorted(order):
        if node in net.terminal:
            flow[node] = None
            continue
        if is_lake_node(node):
            flow[node] = parent.get(node)
            continue
        here = order[node]
        options = [n for n in net.neighbors[node] if n in order and order[n] < here]
        if not options:
            flow[node] = None
            continue
        drops = [filled[node] - filled[n] for n in options]
        wet = [
            (d, n) for d, n in zip(drops, options, strict=True) if n in net.wet or is_lake_node(n)
        ]
        if wet:
            flow[node] = max(wet, key=lambda dn: (dn[0], dn[1]))[1]
        elif len(options) == 1:
            flow[node] = options[0]
        else:
            top = max(drops)
            rel = np.array(drops) / top if top > 0 else np.ones(len(drops))
            weights = rel**wander_exponent
            if weights.sum() <= 0:
                weights = np.ones(len(drops))
            draw = draws.get(node, 0.5) if draws is not None else rng.random()
            pick = draw * weights.sum()
            idx = min(int(np.searchsorted(np.cumsum(weights), pick)), len(options) - 1)
            flow[node] = options[idx]
    return Drainage(filled=filled, order=order, parent=parent, flow=flow)


def accumulate(flow: dict[Node, Node | None], sources: dict[Node, float]) -> dict[Node, float]:
    """Water passing each node: what starts there plus everything upstream of it."""
    nodes = set(flow) | {d for d in flow.values() if d is not None}
    indegree: dict[Node, int] = {n: 0 for n in nodes}
    for d in flow.values():
        if d is not None:
            indegree[d] += 1
    acc = {n: float(sources.get(n, 0.0)) for n in nodes}
    queue = deque(sorted(n for n in nodes if indegree[n] == 0))
    while queue:
        n = queue.popleft()
        d = flow.get(n)
        if d is None:
            continue
        acc[d] += acc[n]
        indegree[d] -= 1
        if indegree[d] == 0:
            queue.append(d)
    return acc


def drain_corner(coord: HexCoord, net: CornerNetwork, drainage: Drainage) -> Corner | None:
    """The corner a land hex's rain runs off to: its lowest, earliest in the fill."""
    corners = [c for c in hex_corner_keys(coord) if c in drainage.order]
    if not corners:
        return None
    return min(corners, key=lambda c: drainage.order[c])


@dataclass
class Streams:
    """The river network read off a drainage: courses, and the corners that matter."""

    paths: list[list[Corner]]
    channel: set[Corner]
    # Upstream channel corners of each channel corner.
    upstream: dict[Corner, list[Corner]]


def trace_streams(drainage: Drainage, threshold: float, forced: Iterable[Node] = ()) -> Streams:
    """Cut the channelled part of the network into courses.

    A corner is channel where at least *threshold* passes it.  Each course starts where a
    channel begins — a spring, a lake's outlet, an inflow — and runs downstream until it
    leaves the network or meets a bigger river.  At a confluence the larger branch carries
    on and the smaller ends on the junction corner, so every channel corner belongs to one
    course and a junction is the last corner of each tributary.  A course that reaches an
    open lake ends on the shore, and what leaves the lake is a course of its own.

    Everything downstream of a corner in *forced* is channel whatever it carries: a lake
    that overflows has a river leaving it, however little spills.  If nothing reaches the
    threshold, the map keeps its single largest drainage line, so the coast still has a
    river mouth.
    """
    acc = drainage.acc
    flow = drainage.flow
    corners = [n for n in acc if not is_lake_node(n)]
    if not corners:
        return Streams([], set(), {})
    # Only a corner that sends its water on can head a channel.  A terminal is channel
    # where a channel runs into it, and not otherwise: a stretch of coast gathering the
    # rain of the hexes beside it can pass the threshold without anything flowing there.
    flowing = [n for n in corners if flow.get(n) is not None]
    if not flowing:
        return Streams([], set(), {})
    bar = threshold
    if not any(acc[n] >= bar for n in flowing):
        # Nothing big enough: keep the largest drainage line there is.
        bar = max(acc[n] for n in flowing)
    channel = {n for n in flowing if acc[n] >= bar}
    for start in forced:
        node: Node | None = start
        while node is not None and not is_lake_node(node) and node not in channel:
            channel.add(node)
            node = flow.get(node)
    channel |= {flow[n] for n in channel if flow.get(n) is not None and not is_lake_node(flow[n])}

    upstream: dict[Corner, list[Corner]] = defaultdict(list)
    for n in sorted(channel):
        d = flow.get(n)
        if d in channel:
            upstream[d].append(n)
    main = {d: max(ups, key=lambda u: (acc[u], u)) for d, ups in upstream.items()}

    paths: list[list[Corner]] = []
    for start in sorted(n for n in channel if not upstream.get(n)):
        path = [start]
        cur = start
        while True:
            nxt = flow.get(cur)
            if nxt is None or nxt not in channel:
                break
            path.append(nxt)
            if main[nxt] != cur:
                break
            cur = nxt
        if len(path) >= 2:
            paths.append(path)
    return Streams(paths=paths, channel=channel, upstream=dict(upstream))


def inlet_corner(coord: HexCoord, net: CornerNetwork, drainage: Drainage) -> Corner | None:
    """Where a river arriving over the border at *coord* joins the corner network.

    The corner of that hex one side in from the map edge whose water then runs furthest,
    so the inflow heads inland rather than along the frame.
    """

    def course(c) -> int:
        n, seen = 0, set()
        while c is not None and c not in seen:
            seen.add(c)
            n += 1
            c = drainage.flow.get(c)
        return n

    corners = [
        c
        for c in hex_corner_keys(coord)
        if c in drainage.order and c not in net.terminal and drainage.flow.get(c) is not None
    ]
    beside_edge = [
        c for c in corners if any(n in net.terminal and n not in net.wet for n in net.neighbors[c])
    ]
    pool = beside_edge or corners
    if not pool:
        return None
    return max(pool, key=lambda c: (course(c), c))
