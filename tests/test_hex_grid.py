from worldgen.core.hex import Hex
from worldgen.core.hex_grid import astar, distance, neighbors, ring


def test_neighbor_distance():
    origin = (0, 0)
    for n in neighbors(origin):
        assert distance(origin, n) == 1


def test_ring_size():
    assert len(ring((0, 0), 2)) == 12


def test_ring_radius_zero():
    assert ring((3, 4), 0) == [(3, 4)]


def test_astar_finds_path():
    grid = {(q, r): Hex(coord=(q, r)) for q in range(5) for r in range(5)}
    path = astar(grid, (0, 0), (4, 0), cost_fn=lambda h: 1.0)
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (4, 0)
    for a, b in zip(path, path[1:], strict=False):
        assert distance(a, b) == 1


def test_astar_blocked():
    # Build a 3-wide corridor, then wall off the middle column
    grid = {(q, r): Hex(coord=(q, r)) for q in range(3) for r in range(3)}

    def cost(h):
        return float("inf") if h.coord[0] == 1 else 1.0

    path = astar(grid, (0, 0), (2, 0), cost_fn=cost)
    assert path is None


def test_astar_start_equals_goal():
    grid = {(0, 0): Hex(coord=(0, 0))}
    path = astar(grid, (0, 0), (0, 0), cost_fn=lambda h: 1.0)
    assert path == [(0, 0)]
