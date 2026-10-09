"""Shortest-path searches over a hex map, on dense arrays, compiled with numba where it is installed.

The stages that route — markets, cities, resources, roads — used to run their Dijkstras over
`dict[HexCoord, Hex]` with a Python closure called per step, and at 112x112 that was about
thirty of a thirty-six second build: five function calls and a `frozenset` per edge, two
hundred million calls a world. The searches themselves are a few dozen lines each; the cost
was all in the calling.

So the cost of every step is worked out once, as arrays — `node[i]` for entering hex *i*,
`edge[i, d]` for the step from *i* toward its neighbour in direction *d* — and the searches
read those. The rules stay in the stages that own them (`haulage.py`, `road_cost.py`); this
module knows nothing about terrain or rivers.

**Each kernel is a port of a search that was already there, and finds exactly what it
found.** The world is reproducible bit for bit from its seed (`docs/REFERENCE.md`), and a
path that changes on a tie changes the roads, so these keep every detail that decides one:

*   The heap pops in the order `heapq` popped `(cost, (q, r))`. Hexes are numbered across
    the (q, r) bounding box row by row, so comparing indices compares coordinates exactly
    as the tuples did; with no counter in the key, any heap pops the same sequence.
*   Neighbours are visited in `hex_grid.neighbors` order, and a relaxation needs a strict
    `<`, so the first parent found keeps a tie.
*   Costs are summed in the same order, as IEEE doubles, with no `fastmath` — numba would
    otherwise fuse a multiply and an add, and the last bit would move.

`scripts/world_fingerprint.py` is how that is checked: it hashes whole worlds, before and
after.

Without numba, `_jit` is the identity and these run as plain Python — the same answers,
slowly. `tests/test_routing.py` runs both against the searches they replaced.
"""

from dataclasses import dataclass

import numpy as np

from .hex import HexCoord

try:
    import numba as _numba

    def _jit(fn):
        return _numba.njit(cache=True)(fn)

except ImportError:  # numba optional — fall back to pure Python
    _numba = None  # type: ignore[assignment]

    def _jit(fn):  # type: ignore[misc]
        return fn


INF = np.inf

# `hex_grid.neighbors` order. The step back from direction d is direction (d + 3) % 6.
_DIRECTIONS = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))
_DIRECTION_OF = {delta: d for d, delta in enumerate(_DIRECTIONS)}


@dataclass(frozen=True)
class Grid:
    """A map's hexes, numbered for the searches.

    Index *i* is `(q - qmin) * rspan + (r - rmin)` over the bounding box of the map, so
    index order is coordinate order. Box cells with no hex are left as holes: `valid` is
    False there and no `nbr` entry points at them.
    """

    qmin: int
    rmin: int
    rspan: int
    valid: np.ndarray  # bool[N]
    nbr: np.ndarray  # int64[N, 6], -1 off the map
    index: dict[HexCoord, int]

    @classmethod
    def of(cls, hexes) -> "Grid":
        coords = np.array(list(hexes), dtype=np.int64).reshape(-1, 2)
        if not len(coords):
            return cls(0, 0, 1, np.zeros(0, bool), np.zeros((0, 6), np.int64), {})
        qmin, rmin = (int(v) for v in coords.min(axis=0))
        qspan, rspan = (int(v) + 1 for v in coords.max(axis=0) - coords.min(axis=0))
        flat = (coords[:, 0] - qmin) * rspan + (coords[:, 1] - rmin)
        valid = np.zeros(qspan * rspan, bool)
        valid[flat] = True
        nbr = np.full((qspan * rspan, 6), -1, np.int64)
        for d, (dq, dr) in enumerate(_DIRECTIONS):
            q, r = coords[:, 0] + dq - qmin, coords[:, 1] + dr - rmin
            inside = (q >= 0) & (q < qspan) & (r >= 0) & (r < rspan)
            target = np.where(inside, q * rspan + r, 0)
            ok = inside & valid[target]
            nbr[flat[ok], d] = target[ok]
        index = dict(zip(map(tuple, coords.tolist()), flat.tolist(), strict=True))
        return cls(qmin, rmin, rspan, valid, nbr, index)

    @property
    def size(self) -> int:
        return len(self.valid)

    def coord(self, i: int) -> HexCoord:
        q, r = divmod(int(i), self.rspan)
        return q + self.qmin, r + self.rmin

    def coords(self, idx: np.ndarray) -> list[HexCoord]:
        """`coord` of each of *idx*, at once."""
        q, r = np.divmod(np.asarray(idx, np.int64), self.rspan)
        return list(zip((q + self.qmin).tolist(), (r + self.rmin).tolist(), strict=True))

    def column(self, hexes, value, dtype=np.float64, fill=0) -> np.ndarray:
        """A per-hex array of `value(hx)`, *fill* in the holes."""
        out = np.full(self.size, fill, dtype)
        out[list(self.index.values())] = [value(hx) for hx in hexes.values()]
        return out

    def pairs(self):
        """`(i, d, j)` arrays for every step on the map: from *i*, direction *d*, into *j*."""
        i, d = np.nonzero(self.nbr >= 0)
        return i, d, self.nbr[i, d]

    def direction(self, a: HexCoord, b: HexCoord) -> int:
        """Which of *a*'s neighbours *b* is."""
        return _DIRECTION_OF[(b[0] - a[0], b[1] - a[1])]

    def mask(self, coords) -> np.ndarray:
        """True at each of *coords* that is on the map."""
        out = np.zeros(self.size, np.bool_)
        out[[self.index[c] for c in coords if c in self.index]] = True
        return out

    def edges(self) -> np.ndarray:
        """An `edge` array with every step unpriced: `inf`, to be filled in."""
        return np.full((self.size, 6), np.inf)

    def pair_steps(self, pairs):
        """`(i, d, j, item)` for each `(frozenset_pair, item)` whose hexes are both on the map.

        Side-keyed data comes keyed by the unordered pair of hexes either side; this
        yields it once for each direction across, which is how an `edge` array holds it.
        """
        for pair, item in pairs:
            a, b = tuple(pair)
            i, j = self.index.get(a), self.index.get(b)
            if i is None or j is None:
                continue
            d = self.direction(a, b)
            yield i, d, j, item
            yield j, (d + 3) % 6, i, item


@dataclass(frozen=True)
class CostField:
    """What it costs to enter each hex, and to take each step: the searches' whole input."""

    grid: Grid
    node: np.ndarray  # float64[N]
    edge: np.ndarray  # float64[N, 6]: the step from i toward nbr[i, d]


# --- the heap -----------------------------------------------------------------------------
# Three parallel arrays — cost, then two integer keys — ordered as the tuple
# `(cost, k1, k2)` would be. Capacity is fixed up front: no search here pushes more than
# once per successful relaxation, and a hex relaxes at most six times per pop.


@_jit
def _before(f1, a1, b1, f2, a2, b2):
    if f1 != f2:
        return f1 < f2
    if a1 != a2:
        return a1 < a2
    return b1 < b2


@_jit
def _push(hf, ha, hb, n, f, a, b):
    i = n
    while i > 0:
        p = (i - 1) >> 1
        if not _before(f, a, b, hf[p], ha[p], hb[p]):
            break
        hf[i], ha[i], hb[i] = hf[p], ha[p], hb[p]
        i = p
    hf[i], ha[i], hb[i] = f, a, b
    return n + 1


@_jit
def _pop(hf, ha, hb, n):
    """Remove the least entry. Returns it and the new size."""
    f, a, b = hf[0], ha[0], hb[0]
    n -= 1
    if n > 0:
        lf, la, lb = hf[n], ha[n], hb[n]
        i = 0
        while True:
            c = 2 * i + 1
            if c >= n:
                break
            if c + 1 < n and _before(hf[c + 1], ha[c + 1], hb[c + 1], hf[c], ha[c], hb[c]):
                c += 1
            if not _before(hf[c], ha[c], hb[c], lf, la, lb):
                break
            hf[i], ha[i], hb[i] = hf[c], ha[c], hb[c]
            i = c
        hf[i], ha[i], hb[i] = lf, la, lb
    return f, a, b, n


@_jit
def _heap(capacity):
    return (
        np.empty(capacity, np.float64),
        np.empty(capacity, np.int64),
        np.empty(capacity, np.int64),
    )


@_jit
def _worn(base, traffic, factor):
    """`road_cost.pheromone_discount`, as `max(0.0, ...)` computes it."""
    v = base - factor * traffic
    return v if v > 0.0 else 0.0


# --- the searches -------------------------------------------------------------------------


@_jit
def from_seats(nbr, node, edge, allowed, use_allowed, seeds, budget):
    """`haulage.bulk_routes`: cost to the nearest seed from everywhere within *budget*.

    Prices the step a cargo takes, *toward* the seed: `node[n] + edge[n, back]`. Returns
    `(cost, toward, order)`; *order* is the hexes in the order they were first reached,
    which is the order the dict it replaces was filled in.
    """
    size = nbr.shape[0]
    cost = np.full(size, np.inf)
    toward = np.full(size, -1, np.int64)
    order = np.empty(size, np.int64)
    reached = 0
    hf, ha, hb = _heap(6 * size + len(seeds) + 1)
    n = 0
    for s in seeds:
        cost[s] = 0.0
    for s in seeds:
        n = _push(hf, ha, hb, n, 0.0, s, 0)
    while n > 0:
        d, c, _, n = _pop(hf, ha, hb, n)
        if d > cost[c]:
            continue
        for k in range(6):
            m = nbr[c, k]
            if m < 0:
                continue
            back = (k + 3) % 6
            if use_allowed and not allowed[m, back]:
                continue
            step = node[m] + edge[m, back]
            if step == np.inf:
                continue
            nd = d + step
            if nd < budget and nd < cost[m]:
                if cost[m] == np.inf:
                    order[reached] = m
                    reached += 1
                cost[m] = nd
                toward[m] = c
                n = _push(hf, ha, hb, n, nd, m, 0)
    return cost, toward, order[:reached]


@_jit
def catchments(nbr, node, edge, seeds, budget):
    """`haulage.allocate_catchments`: each hex to the seed that reaches it cheapest.

    Ownership is taken at the pop, and ties go to the lower `(cost, hex, seed)`. Returns
    `(owner, cost, order)`, *order* being the pop order.
    """
    size = nbr.shape[0]
    owner = np.full(size, -1, np.int64)
    cost = np.full(size, np.inf)
    order = np.empty(size, np.int64)
    taken = 0
    hf, ha, hb = _heap(6 * size + len(seeds) + 1)
    n = 0
    for s in seeds:
        n = _push(hf, ha, hb, n, 0.0, s, s)
    while n > 0:
        d, c, seat, n = _pop(hf, ha, hb, n)
        if owner[c] >= 0:
            continue
        owner[c] = seat
        cost[c] = d
        order[taken] = c
        taken += 1
        for k in range(6):
            m = nbr[c, k]
            if m < 0 or owner[m] >= 0:
                continue
            step = node[m] + edge[c, k]
            if step == np.inf:
                continue
            nd = d + step
            if nd < budget:
                n = _push(hf, ha, hb, n, nd, m, seat)
    return owner, cost, order[:taken]


@_jit
def day_walk(nbr, node, edge, water, start, radius):
    """`markets.day_reach`'s search: one seat's catchment, and the water at its edge.

    Returns `(cost, done, rim)`: the cost to every hex settled, which were, and for each
    water hex beside them the cheapest cost of a settled neighbour (`inf` elsewhere).
    """
    size = nbr.shape[0]
    cost = np.full(size, np.inf)
    done = np.zeros(size, np.bool_)
    seen = np.empty(size, np.int64)
    settled = 0
    hf, ha, hb = _heap(6 * size + 2)
    cost[start] = 0.0
    n = _push(hf, ha, hb, 0, 0.0, start, 0)
    while n > 0:
        d, c, _, n = _pop(hf, ha, hb, n)
        if done[c]:
            continue
        done[c] = True
        seen[settled] = c
        settled += 1
        for k in range(6):
            m = nbr[c, k]
            if m < 0 or done[m]:
                continue
            step = node[m] + edge[c, k]
            if step == np.inf:
                continue
            nd = d + step
            if nd < radius and nd < cost[m]:
                cost[m] = nd
                n = _push(hf, ha, hb, n, nd, m, 0)
    rim = np.full(size, np.inf)
    for t in range(settled):
        c = seen[t]
        for k in range(6):
            m = nbr[c, k]
            if m < 0 or done[m] or not water[m]:
                continue
            if cost[c] < rim[m]:
                rim[m] = cost[c]
    return cost, done, rim


# How `to_any` weighs a goal it reaches: stop at the first, or weigh each by what remains
# to be travelled from it (`astar_to_any`'s `goal_cost`).
FIRST_GOAL, GOAL_COST = 0, 1


@_jit
def to_any(nbr, node, traffic, factor, edge, start, goals, goal_cost, mode):
    """`hex_grid.astar_to_any` with no `aim`: a Dijkstra to the best of *goals*.

    *traffic*, if it is non-empty, wears *node* down as `pheromone_discount` does. Returns
    `(came_from, best)`, *best* being -1 where no goal is reachable.
    """
    size = nbr.shape[0]
    worn = len(traffic) > 0
    g = np.full(size, np.inf)
    came = np.full(size, -1, np.int64)
    visited = np.zeros(size, np.bool_)
    hf, ha, hb = _heap(6 * size + 2)
    g[start] = 0.0
    n = _push(hf, ha, hb, 0, 0.0, start, 0)
    best_total, best = np.inf, -1
    while n > 0:
        f, c, _, n = _pop(hf, ha, hb, n)
        if f > best_total:
            break
        if visited[c]:
            continue
        visited[c] = True
        if goals[c]:
            total = g[c] + (goal_cost[c] if mode == GOAL_COST else 0.0)
            if total < best_total:
                best_total, best = total, c
            if mode == FIRST_GOAL:
                break
        for k in range(6):
            m = nbr[c, k]
            if m < 0 or visited[m]:
                continue
            cost = _worn(node[m], traffic[m], factor) if worn else node[m]
            cost += edge[c, k]
            if cost == np.inf:
                continue
            t = g[c] + cost
            if t < g[m]:
                came[m] = c
                g[m] = t
                n = _push(hf, ha, hb, n, t + 0.0, m, 0)
    return came, best


@_jit
def along(adj, nbr, node, traffic, factor, edge, dest):
    """`InterurbanRoadStage._road_home_tree`: the road network's way to *dest*, and its cost.

    *adj* holds, per hex, the directions of the roads leaving it, in the order the stage's
    adjacency sets iterate — that order breaks ties here, so it is passed in, not derived.
    Returns `(parent, home)`, -1 and `inf` off the tree.
    """
    size = nbr.shape[0]
    parent = np.full(size, -1, np.int64)
    home = np.full(size, np.inf)
    hf, ha, hb = _heap(6 * size + 2)
    home[dest] = 0.0
    n = _push(hf, ha, hb, 0, 0.0, dest, 0)
    while n > 0:
        cost, c, _, n = _pop(hf, ha, hb, n)
        if cost > home[c]:
            continue
        for j in range(6):
            k = adj[c, j]
            if k < 0:
                break
            m = nbr[c, k]
            step = _worn(node[m], traffic[m], factor) + edge[c, k]
            if step == np.inf:
                continue
            if cost + step < home[m]:
                home[m] = cost + step
                parent[m] = c
                n = _push(hf, ha, hb, n, cost + step, m, 0)
    return parent, home


def walk_back(grid: Grid, came: np.ndarray, end: int) -> list[HexCoord]:
    """The path a `came_from` array records into *end*, first hex first."""
    path = []
    while end >= 0:
        path.append(grid.coord(end))
        end = int(came[end])
    path.reverse()
    return path
