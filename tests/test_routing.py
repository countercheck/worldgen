"""The array searches in `core/routing.py` find exactly what the dict searches they replaced found.

Not "a shortest path": the same path, the same costs to the last bit, the same dict order.
The world is reproducible from its seed, and a search that breaks a tie the other way builds
different roads just as reproducibly — which no same-seed test can see. So each search the
port replaced is kept here verbatim as a reference, and the port is held to it on a real
world, `==` throughout.

Every comparison runs twice: compiled, and as plain Python (`py_func`), which is both the
no-numba fallback and the only way coverage can see inside a kernel.
"""

import heapq
import random
from collections import defaultdict

import numpy as np
import pytest

from worldgen.core import routing
from worldgen.core.config import WorldConfig
from worldgen.core.hex_grid import astar_to_any, neighbors
from worldgen.core.routing import Grid
from worldgen.stages import interurban_roads
from worldgen.stages.haulage import (
    allocate_catchments,
    bulk_field,
    bulk_routes,
    make_bulk_cost,
    make_travel_cost,
    travel_field,
)
from worldgen.stages.markets import day_reach
from worldgen.stages.riverside import WATER, river_index
from worldgen.stages.road_cost import (
    make_road_edge_cost,
    pheromone_discount,
    river_crossings,
    road_edge_costs,
    road_node_costs,
    settlement_rings,
    terrain_base_cost,
)

_KERNELS = ("from_seats", "catchments", "day_walk", "to_any", "along", "_push", "_pop")


@pytest.fixture(params=["compiled", "python"])
def kernels(request, monkeypatch):
    """Run the test against the compiled kernels, then against their Python source."""
    if request.param == "python":
        for name in _KERNELS:
            fn = getattr(routing, name)
            monkeypatch.setattr(routing, name, getattr(fn, "py_func", fn))
    return request.param


@pytest.fixture(scope="module")
def world(settle_state):
    cfg = WorldConfig(**settle_state.metadata["config"])
    return settle_state, cfg, river_index(settle_state, cfg)


def _steps(hexes):
    for coord, hx in hexes.items():
        for n in neighbors(coord):
            if n in hexes:
                yield hx, hexes[n]


def _seats(hexes, k, seed=0):
    land = sorted(c for c, hx in hexes.items() if hx.terrain_class not in WATER)
    return random.Random(seed).sample(land, k)


# --- the searches as they were ------------------------------------------------------------


def _ref_bulk_routes(hexes, seats, cfg, budget, rivers, step_allowed=None):
    node_cost, edge_cost = make_bulk_cost(hexes, cfg, rivers)
    cost = {seat: 0.0 for seat in seats}
    toward = {}
    heap = [(0.0, seat) for seat in sorted(seats)]
    heapq.heapify(heap)
    while heap:
        d, coord = heapq.heappop(heap)
        if d > cost.get(coord, float("inf")):
            continue
        hx = hexes[coord]
        for n in neighbors(coord):
            n_hx = hexes.get(n)
            if n_hx is None:
                continue
            if step_allowed is not None and not step_allowed(n_hx, hx):
                continue
            step = node_cost(n_hx) + edge_cost(n_hx, hx)
            if step == float("inf"):
                continue
            nd = d + step
            if nd < budget and nd < cost.get(n, float("inf")):
                cost[n] = nd
                toward[n] = coord
                heapq.heappush(heap, (nd, n))
    return cost, toward


def _ref_catchments(hexes, seats, budget, cfg, rivers):
    seats = sorted(seats)
    node_cost, edge_cost = make_travel_cost(hexes, cfg, rivers)
    owner, cost = {}, {}
    heap = [(0.0, seat, seat) for seat in seats if seat in hexes]
    heapq.heapify(heap)
    while heap:
        d, coord, seat = heapq.heappop(heap)
        if coord in owner:
            continue
        owner[coord] = seat
        cost[coord] = d
        hx = hexes[coord]
        for n in neighbors(coord):
            if n in owner:
                continue
            n_hx = hexes.get(n)
            if n_hx is None:
                continue
            step = node_cost(n_hx) + edge_cost(hx, n_hx)
            if step == float("inf"):
                continue
            nd = d + step
            if nd < budget:
                heapq.heappush(heap, (nd, n, seat))
    return owner, cost


def _ref_day_reach(coord, hexes, radius, node_cost, edge_cost):
    from worldgen.stages.haulage import usable_fraction

    cost = {coord: 0.0}
    heap = [(0.0, coord)]
    done = set()
    while heap:
        d, c = heapq.heappop(heap)
        if c in done:
            continue
        done.add(c)
        hx = hexes[c]
        for n in neighbors(c):
            if n in done:
                continue
            n_hx = hexes.get(n)
            if n_hx is None:
                continue
            step = node_cost(n_hx) + edge_cost(hx, n_hx)
            if step == float("inf"):
                continue
            nd = d + step
            if nd < radius and nd < cost.get(n, float("inf")):
                cost[n] = nd
                heapq.heappush(heap, (nd, n))
    rim = {}
    for c in done:
        for n in neighbors(c):
            if n in done:
                continue
            n_hx = hexes.get(n)
            if n_hx is None or n_hx.terrain_class not in WATER:
                continue
            if cost[c] < rim.get(n, float("inf")):
                rim[n] = cost[c]
    out = [(c, usable_fraction(cost[c], radius)) for c in sorted(done)]
    out += [(n, usable_fraction(rim[n], radius)) for n in sorted(rim)]
    return [(c, w) for c, w in out if w > 0.0]


def _ref_road_home_tree(hexes, dest, net_adj, node_cost, edge_cost):
    tree = {dest: None}
    home_cost = {dest: 0.0}
    if dest not in net_adj:
        return tree, home_cost
    queue = [(0.0, dest)]
    while queue:
        cost, c = heapq.heappop(queue)
        if cost > home_cost.get(c, float("inf")):
            continue
        for n in net_adj[c]:
            if n not in hexes:
                continue
            step = node_cost(hexes[n]) + edge_cost(hexes[c], hexes[n])
            if step == float("inf"):
                continue
            if cost + step < home_cost.get(n, float("inf")):
                home_cost[n] = cost + step
                tree[n] = c
                heapq.heappush(queue, (cost + step, n))
    return tree, home_cost


# --- the heap and the grid ----------------------------------------------------------------


def test_the_heap_pops_in_heapq_order(kernels):
    """Ties everywhere, on all three keys: the order must still be the tuples' order."""
    rnd = random.Random(1)
    entries = [
        (rnd.choice([0.0, 0.5, 1.0, np.inf]), rnd.randrange(5), rnd.randrange(3))
        for _ in range(400)
    ]
    hf, ha, hb = (
        np.empty(len(entries)),
        np.empty(len(entries), np.int64),
        np.empty(len(entries), np.int64),
    )
    n = 0
    ref = []
    for e in entries:
        n = routing._push(hf, ha, hb, n, *e)
        heapq.heappush(ref, e)
    out = []
    while n:
        f, a, b, n = routing._pop(hf, ha, hb, n)
        out.append((f, a, b))
    assert out == [heapq.heappop(ref) for _ in entries]


@pytest.mark.parametrize("layout", ["axial", "offset"])
def test_index_order_is_coordinate_order(layout):
    """The tie-break the heap relies on: comparing indices compares (q, r) tuples."""
    from worldgen.core.world_state import WorldState

    hexes = WorldState.empty(1, 9, 7, layout).hexes
    grid = Grid.of(hexes)
    by_index = sorted(hexes, key=grid.index.__getitem__)
    assert by_index == sorted(hexes)
    assert [grid.coord(grid.index[c]) for c in hexes] == list(hexes)
    for c in hexes:
        expected = [n for n in neighbors(c) if n in hexes]
        assert [grid.coord(j) for j in grid.nbr[grid.index[c]] if j >= 0] == expected


# --- the costs ----------------------------------------------------------------------------


def test_travel_and_bulk_arrays_are_the_closures(world):
    state, cfg, rivers = world
    hexes = state.hexes
    for field, (node_cost, edge_cost) in (
        (travel_field(hexes, cfg, rivers), make_travel_cost(hexes, cfg, rivers)),
        (bulk_field(hexes, cfg, rivers), make_bulk_cost(hexes, cfg, rivers)),
    ):
        grid = field.grid
        for a, b in _steps(hexes):
            i, j = grid.index[a.coord], grid.index[b.coord]
            assert field.node[j] == node_cost(b)
            assert field.edge[i, grid.direction(a.coord, b.coord)] == edge_cost(a, b), (
                a.coord,
                b.coord,
            )


def test_road_arrays_are_the_closures(world):
    state, cfg, _ = world
    hexes = state.hexes
    grid = Grid.of(hexes)
    crossings = river_crossings(state.river_sides)
    ring = settlement_rings({s.coord for s in state.settlements})
    assert crossings and ring, "the world needs crossings and towns for this to test anything"
    node = road_node_costs(hexes, grid, cfg)
    for args in ((crossings, ring), (crossings, None), (None, None)):
        edge = road_edge_costs(hexes, grid, cfg, *args)
        edge_cost = make_road_edge_cost(cfg, *args)
        for a, b in _steps(hexes):
            i, d = grid.index[a.coord], grid.direction(a.coord, b.coord)
            assert edge[i, d] == edge_cost(a, b), (a.coord, b.coord, args is None)
            assert node[i] == terrain_base_cost(a, cfg)


# --- the searches -------------------------------------------------------------------------


def test_bulk_routes_finds_what_it_found(world, kernels):
    state, cfg, rivers = world
    hexes = state.hexes
    budget = cfg.haulage_range_land
    for seats in ([s] for s in _seats(hexes, 6)):
        got = bulk_routes(hexes, seats, cfg, rivers=rivers)
        want = _ref_bulk_routes(hexes, seats, cfg, budget, rivers)
        assert [list(d.items()) for d in got] == [list(d.items()) for d in want]
    seats = _seats(hexes, 5, seed=2)
    got = bulk_routes(hexes, seats, cfg, budget=budget * 2, rivers=rivers)
    assert [list(d.items()) for d in got] == [
        list(d.items()) for d in _ref_bulk_routes(hexes, seats, cfg, budget * 2, rivers)
    ]


def test_bulk_routes_honours_step_allowed(world, kernels):
    """The river trade's downhill-only rule: refused steps are refused in the same places."""
    state, cfg, rivers = world
    hexes = state.hexes

    def downhill(a, b):
        return b.elevation <= a.elevation

    seats = _seats(hexes, 3, seed=3)
    got = bulk_routes(hexes, seats, cfg, rivers=rivers, step_allowed=downhill)
    want = _ref_bulk_routes(hexes, seats, cfg, cfg.haulage_range_land, rivers, downhill)
    assert [list(d.items()) for d in got] == [list(d.items()) for d in want]


def test_catchments_are_what_they_were(world, kernels):
    state, cfg, rivers = world
    hexes = state.hexes
    for k, budget in ((1, 6.0), (8, cfg.market_day_radius), (30, 25.0)):
        seats = _seats(hexes, k, seed=k)
        got = allocate_catchments(hexes, seats, budget, cfg, rivers)
        want = _ref_catchments(hexes, seats, budget, cfg, rivers)
        assert [list(d.items()) for d in got] == [list(d.items()) for d in want]


def test_day_reach_is_what_it_was(world, kernels):
    state, cfg, rivers = world
    hexes = state.hexes
    radius = cfg.market_day_radius
    field = travel_field(hexes, cfg, rivers)
    node_cost, edge_cost = make_travel_cost(hexes, cfg, rivers)
    for seat in _seats(hexes, 12, seed=4):
        assert day_reach(seat, radius, field) == _ref_day_reach(
            seat, hexes, radius, node_cost, edge_cost
        )


def _worn_network(state, cfg, seed=5):
    """The world's roads as the road stage's adjacency sets, and traffic laid over them.

    Traffic heavy enough that the pheromone floors hexes at zero: the plateaus where ties
    decide the route, which is what this needs to test.
    """
    rnd = random.Random(seed)
    net_adj = defaultdict(set)
    for a, b in sorted(state.road_edges):
        net_adj[a].add(b)
        net_adj[b].add(a)
    traffic = {c: rnd.choice([0.0, 0.0, 3.0, 12.0, 40.0]) for c in sorted(net_adj)}
    return net_adj, traffic


def test_the_road_searches_find_what_they_found(world, kernels):
    """`along` against `_road_home_tree`, and `to_any` against `astar_to_any`, on worn roads."""
    state, cfg, _ = world
    hexes = state.hexes
    crossings = river_crossings(state.river_sides)
    ring = settlement_rings({s.coord for s in state.settlements})
    net_adj, traffic = _worn_network(state, cfg)

    def node_cost(hx):
        return pheromone_discount(terrain_base_cost(hx, cfg), traffic.get(hx.coord, 0.0), cfg)

    edge_cost = make_road_edge_cost(cfg, crossings, ring)

    net = interurban_roads._Network.of(hexes, cfg, crossings, ring)
    for c, n in traffic.items():
        net.wear(c, n)
    for c, roads in net_adj.items():
        net.join(c, roads)
    grid = net.grid

    places = sorted(s.coord for s in state.settlements)
    for dest in places[:4]:
        tree, home = _ref_road_home_tree(hexes, dest, net_adj, node_cost, edge_cost)
        parent, cost = routing.along(
            net.adj, grid.nbr, net.node, net.traffic, net.factor, net.edge, grid.index[dest]
        )
        reached = np.flatnonzero(cost < np.inf)
        assert sorted(grid.coords(reached)) == sorted(home)
        for i in reached.tolist():
            c = grid.coord(i)
            assert cost[i] == home[c]
            assert (grid.coord(parent[i]) if parent[i] >= 0 else None) == tree[c]

        for origin in places[-6:]:
            want = astar_to_any(hexes, origin, set(tree), node_cost, edge_cost, goal_cost=home)
            got = interurban_roads._to_any(
                grid,
                net.node,
                net.edge,
                origin,
                cost < np.inf,
                traffic=net.traffic,
                factor=net.factor,
                goal_cost=cost,
            )
            assert got == want, (origin, dest)


def test_the_join_searches_find_what_they_found(world, kernels):
    """`to_any` with no residual stops at the first goal, as `astar_to_any` does."""
    state, cfg, _ = world
    hexes = state.hexes
    crossings = river_crossings(state.river_sides)
    grid = Grid.of(hexes)
    land = np.array([np.inf if hx.terrain_class in WATER else 0.0 for hx in hexes.values()])
    node = road_node_costs(hexes, grid, cfg)
    node[list(grid.index.values())] += land
    edge = road_edge_costs(hexes, grid, cfg, crossings)

    def land_cost(hx):
        return float("inf") if hx.terrain_class in WATER else terrain_base_cost(hx, cfg)

    edge_cost = make_road_edge_cost(cfg, crossings)
    places = sorted(s.coord for s in state.settlements)
    goals = set(places[::3])
    for start in _seats(hexes, 8, seed=6):
        want = astar_to_any(hexes, start, goals, land_cost, edge_cost)
        assert interurban_roads._to_any(grid, node, edge, start, grid.mask(goals)) == want
