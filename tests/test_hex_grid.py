from worldgen.core.hex import Hex, TerrainClass
from worldgen.core.hex_grid import astar, distance, grade_reachable_count, neighbors, ring


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


def test_grade_reachable_count_all_flat():
    # 5×5 flat grid — all 25 hexes reachable with any grade threshold
    grid = {(q, r): Hex(coord=(q, r)) for q in range(5) for r in range(5)}
    count = grade_reachable_count((0, 0), grid, lambda a, b: True, max_count=100)
    assert count == 25


def test_grade_reachable_count_blocked():
    # Two flat patches separated by a steep wall (col q=2)
    # grade_ok returns False for neighbors in the wall column
    grid = {(q, r): Hex(coord=(q, r)) for q in range(5) for r in range(3)}
    for r in range(3):
        grid[(2, r)].elevation = 1.0  # steep wall

    def grade_ok(a_hx, b_hx):
        return abs(b_hx.elevation - a_hx.elevation) < 0.5

    count = grade_reachable_count((0, 0), grid, grade_ok, max_count=100)
    # Only left patch (q=0,1 × r=0..2 = 6 hexes) should be reachable
    assert count == 6


def test_grade_reachable_count_early_stop():
    # Large grid; max_count stops the BFS early
    grid = {(q, r): Hex(coord=(q, r)) for q in range(20) for r in range(20)}
    count = grade_reachable_count((0, 0), grid, lambda a, b: True, max_count=10)
    assert count == 10


def test_grade_reachable_count_skips_water():
    # Ocean hexes should never be counted or crossed
    grid = {(q, r): Hex(coord=(q, r)) for q in range(3) for r in range(3)}
    for r in range(3):
        grid[(1, r)].terrain_class = TerrainClass.OCEAN

    count = grade_reachable_count((0, 0), grid, lambda a, b: True, max_count=100)
    # Only left column (q=0, r=0..2 = 3 hexes)
    assert count == 3
