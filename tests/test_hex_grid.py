from worldgen.core.hex import Hex, TerrainClass
from worldgen.core.hex_grid import (
    astar,
    distance,
    grade_reachable_count,
    neighbors,
    ring,
    split_path_on_water,
)


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


def test_grade_reachable_count_start_missing():
    grid = {(0, 0): Hex(coord=(0, 0))}
    count = grade_reachable_count((1, 1), grid, lambda a, b: True, max_count=100)
    assert count == 0


def test_grade_reachable_count_start_water():
    grid = {(0, 0): Hex(coord=(0, 0), terrain_class=TerrainClass.OCEAN)}
    count = grade_reachable_count((0, 0), grid, lambda a, b: True, max_count=100)
    assert count == 0


# --- split_path_on_water -----------------------------------------------------


def _water_grid(water: set, length: int = 6) -> dict:
    """A straight run of hexes (q, 0), with the given q values made into water."""
    return {
        (q, 0): Hex(
            coord=(q, 0),
            terrain_class=TerrainClass.OCEAN if q in water else TerrainClass.FLAT,
        )
        for q in range(length)
    }


def _path(length: int = 6) -> list:
    return [(q, 0) for q in range(length)]


def test_split_path_no_water_returns_whole_path():
    grid = _water_grid(set())
    assert split_path_on_water(_path(), grid) == [_path()]


def test_split_path_drops_the_water_hexes():
    """Water hexes are removed, not merely used as separators."""
    grid = _water_grid({2, 3})
    assert split_path_on_water(_path(), grid) == [[(0, 0), (1, 0)], [(4, 0), (5, 0)]]


def test_split_path_multiple_water_runs():
    grid = _water_grid({2, 5}, length=8)
    segments = split_path_on_water(_path(8), grid)
    assert segments == [[(0, 0), (1, 0)], [(3, 0), (4, 0)], [(6, 0), (7, 0)]]


def test_split_path_leading_and_trailing_water():
    grid = _water_grid({0, 1, 4, 5})
    assert split_path_on_water(_path(), grid) == [[(2, 0), (3, 0)]]


def test_split_path_drops_single_hex_runs():
    """A one-hex run cannot be drawn as a polyline, so it is discarded.

    Water at index 1 leaves (0,0) stranded alone; only the longer run survives.
    """
    grid = _water_grid({1})
    assert split_path_on_water(_path(), grid) == [[(2, 0), (3, 0), (4, 0), (5, 0)]]


def test_split_path_all_water_returns_nothing():
    grid = _water_grid({0, 1, 2, 3, 4, 5})
    assert split_path_on_water(_path(), grid) == []


def test_split_path_lake_counts_as_water():
    grid = _water_grid(set())
    grid[(2, 0)] = Hex(coord=(2, 0), terrain_class=TerrainClass.LAKE)
    assert split_path_on_water(_path(), grid) == [[(0, 0), (1, 0)], [(3, 0), (4, 0), (5, 0)]]


def test_split_path_keeps_coords_missing_from_the_grid():
    """An off-grid coord has no terrain to judge, so it is kept rather than cut on."""
    grid = _water_grid(set())
    del grid[(2, 0)]
    assert split_path_on_water(_path(), grid) == [_path()]


def test_split_path_empty_input():
    assert split_path_on_water([], _water_grid(set())) == []
