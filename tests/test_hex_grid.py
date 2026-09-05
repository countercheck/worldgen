from worldgen.core.hex import Hex, TerrainClass
from worldgen.core.hex_grid import (
    astar,
    astar_to_any,
    distance,
    grade_reachable_count,
    hex_range,
    neighbors,
    ring,
    road_polylines,
    road_water_transitions,
    split_path_on_water,
    water_transitions,
)
from worldgen.core.world_state import ROAD_TIER_RANK, RoadTier, road_edge_key


def test_neighbor_distance():
    origin = (0, 0)
    for n in neighbors(origin):
        assert distance(origin, n) == 1


def test_ring_size():
    assert len(ring((0, 0), 2)) == 12


def test_ring_radius_zero():
    assert ring((3, 4), 0) == [(3, 4)]


def test_ring_one_is_exactly_the_neighbours():
    assert sorted(ring((0, 0), 1)) == sorted(neighbors((0, 0)))


def test_ring_members_are_all_at_that_distance():
    """Counting the ring is not enough — a wrong walk returns 6r hexes in a smear."""
    for radius in range(1, 9):
        members = ring((2, -3), radius)
        assert len(members) == len(set(members)), f"ring {radius} repeats a hex"
        assert len(members) == 6 * radius
        for coord in members:
            assert distance((2, -3), coord) == radius, (
                f"ring {radius} returned {coord} at distance {distance((2, -3), coord)}"
            )


def test_hex_range_is_a_disc():
    """1 + 3r(r+1) is the hex-disc size; anything else means gaps or overspill."""
    for radius in range(0, 9):
        members = hex_range((-4, 1), radius)
        assert len(members) == len(set(members)), f"hex_range {radius} repeats a hex"
        assert len(members) == 1 + 3 * radius * (radius + 1)
        for coord in members:
            assert distance((-4, 1), coord) <= radius


def test_hex_range_is_symmetric():
    """A lopsided radius would smear cultivation and catchments off to one side."""
    members = set(hex_range((0, 0), 4))
    for q, r in members:
        assert (-q, -r) in members, f"({q}, {r}) has no opposite in the disc"


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
        grid[(1, r)].terrain_class = TerrainClass.OPEN_WATER

    count = grade_reachable_count((0, 0), grid, lambda a, b: True, max_count=100)
    # Only left column (q=0, r=0..2 = 3 hexes)
    assert count == 3


def test_grade_reachable_count_start_missing():
    grid = {(0, 0): Hex(coord=(0, 0))}
    count = grade_reachable_count((1, 1), grid, lambda a, b: True, max_count=100)
    assert count == 0


def test_grade_reachable_count_start_water():
    grid = {(0, 0): Hex(coord=(0, 0), terrain_class=TerrainClass.OPEN_WATER)}
    count = grade_reachable_count((0, 0), grid, lambda a, b: True, max_count=100)
    assert count == 0


# --- split_path_on_water -----------------------------------------------------


def _water_grid(water: set, length: int = 6) -> dict:
    """A straight run of hexes (q, 0), with the given q values made into water."""
    return {
        (q, 0): Hex(
            coord=(q, 0),
            terrain_class=TerrainClass.OPEN_WATER if q in water else TerrainClass.LAND,
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
    grid[(2, 0)] = Hex(coord=(2, 0), terrain_class=TerrainClass.INLAND_WATER)
    assert split_path_on_water(_path(), grid) == [[(0, 0), (1, 0)], [(3, 0), (4, 0), (5, 0)]]


def test_split_path_keeps_coords_missing_from_the_grid():
    """An off-grid coord has no terrain to judge, so it is kept rather than cut on."""
    grid = _water_grid(set())
    del grid[(2, 0)]
    assert split_path_on_water(_path(), grid) == [_path()]


def test_split_path_empty_input():
    assert split_path_on_water([], _water_grid(set())) == []


# --- water_transitions -------------------------------------------------------


def test_water_transitions_none_on_a_dry_path():
    assert water_transitions(_path(), _water_grid(set())) == []


def test_water_transitions_marks_both_shores():
    """A crossing yields the land hex before the water and the one after it."""
    grid = _water_grid({2, 3})
    assert water_transitions(_path(), grid) == [(1, 0), (4, 0)]


def test_water_transitions_path_starting_in_water():
    """Nothing to mark on the seaward side — only the landing point."""
    grid = _water_grid({0, 1})
    assert water_transitions(_path(), grid) == [(2, 0)]


def test_water_transitions_path_ending_in_water():
    grid = _water_grid({4, 5})
    assert water_transitions(_path(), grid) == [(3, 0)]


def test_water_transitions_multiple_crossings():
    grid = _water_grid({2, 5}, length=8)
    assert water_transitions(_path(8), grid) == [(1, 0), (3, 0), (4, 0), (6, 0)]


def test_water_transitions_lake_counts_as_water():
    grid = _water_grid(set())
    grid[(2, 0)] = Hex(coord=(2, 0), terrain_class=TerrainClass.INLAND_WATER)
    assert water_transitions(_path(), grid) == [(1, 0), (3, 0)]


def test_water_transitions_collapses_repeats():
    """A route that puts to sea and returns to the same hex marks it once."""
    grid = _water_grid({1})
    assert water_transitions([(0, 0), (1, 0), (0, 0)], grid) == [(0, 0)]


def test_water_transitions_keeps_coords_missing_from_the_grid():
    """An off-grid coord counts as land, matching split_path_on_water."""
    grid = _water_grid(set())
    del grid[(2, 0)]
    assert water_transitions(_path(), grid) == []


def test_water_transitions_empty_path():
    assert water_transitions([], _water_grid(set())) == []


def test_water_transitions_single_hex_path():
    assert water_transitions([(0, 0)], _water_grid(set())) == []


# --- road_polylines ----------------------------------------------------------


def _grid(*paths, water: frozenset = frozenset()) -> dict:
    """A grid covering every coord in *paths*, with the given coords made ocean."""
    return {
        c: Hex(coord=c, terrain_class=TerrainClass.OPEN_WATER if c in water else TerrainClass.LAND)
        for p in paths
        for c in p
    }


def _edges(path, tier) -> dict:
    """The edges of a hex path, all of one tier."""
    return {road_edge_key(a, b): tier for a, b in zip(path, path[1:], strict=False)}


def _drawn_edges(result) -> list:
    return [frozenset((a, b)) for _, leg in result for a, b in zip(leg, leg[1:], strict=False)]


def test_a_single_corridor_comes_back_as_one_run():
    edges = _edges(_path(), RoadTier.PRIMARY)
    assert road_polylines(edges, _grid(_path())) == [(RoadTier.PRIMARY, _path())]


def test_no_edges_draw_nothing():
    assert road_polylines({}, _water_grid(set())) == []


def test_every_edge_is_drawn_exactly_once():
    """The point of the graph: there is nothing left to deduplicate."""
    trunk = _path(6)
    branch = [(2, 0), (2, 1), (2, 2)]
    edges = _edges(trunk, RoadTier.PRIMARY) | _edges(branch, RoadTier.TRACK)
    drawn = _drawn_edges(road_polylines(edges, _grid(trunk, branch)))
    assert sorted(map(sorted, drawn)) == sorted(map(sorted, (set(k) for k in edges)))
    assert len(drawn) == len(set(drawn))


def test_a_run_breaks_at_a_junction():
    """A polyline through a fork would draw an edge that is not in the graph."""
    trunk = _path(5)
    branch = [(2, 0), (2, 1)]
    result = road_polylines(
        _edges(trunk, RoadTier.PRIMARY) | _edges(branch, RoadTier.PRIMARY),
        _grid(trunk, branch),
    )
    assert all(len(leg) >= 2 for _, leg in result)
    # The trunk is cut either side of the fork rather than swept through it.
    assert (2, 0) in {c for _, leg in result for c in (leg[0], leg[-1])}


def test_a_run_breaks_where_the_tier_changes():
    path = _path(5)
    edges = _edges(path[:3], RoadTier.PRIMARY) | _edges(path[2:], RoadTier.TRACK)
    result = road_polylines(edges, _grid(path))
    assert {tier for tier, _ in result} == {RoadTier.PRIMARY, RoadTier.TRACK}
    for _tier, leg in result:
        assert len(leg) == 3


def test_runs_come_back_with_the_most_important_last():
    """A renderer drawing in order must paint a primary road over a track."""
    trunk = _path(5)
    branch = [(2, 0), (2, 1)]
    result = road_polylines(
        _edges(trunk, RoadTier.PRIMARY) | _edges(branch, RoadTier.TRACK),
        _grid(trunk, branch),
    )
    ranks = [ROAD_TIER_RANK[tier] for tier, _ in result]
    assert ranks == sorted(ranks)


def test_edges_touching_water_are_not_drawn():
    path = _path(6)
    grid = _grid(path, water=frozenset({(2, 0), (3, 0)}))
    result = road_polylines(_edges(path, RoadTier.PRIMARY), grid)
    assert _drawn_edges(result) == [frozenset({(0, 0), (1, 0)}), frozenset({(4, 0), (5, 0)})]


def test_a_network_entirely_on_water_draws_nothing():
    path = _path(4)
    assert road_polylines(_edges(path, RoadTier.PRIMARY), _grid(path, water=frozenset(path))) == []


def test_a_closed_loop_is_still_drawn():
    """A ring has no end to start walking from, and must not be silently dropped."""
    loop = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    edges = _edges(loop, RoadTier.SECONDARY)
    result = road_polylines(edges, _grid(loop))
    assert len(_drawn_edges(result)) == len(edges)


def test_road_water_transitions_marks_the_land_side():
    path = _path(5)
    grid = _grid(path, water=frozenset({(2, 0)}))
    assert road_water_transitions(_edges(path, RoadTier.PRIMARY), grid) == {(1, 0), (3, 0)}


def test_road_water_transitions_none_on_a_dry_network():
    path = _path()
    assert road_water_transitions(_edges(path, RoadTier.PRIMARY), _grid(path)) == set()


# --- astar_to_any ------------------------------------------------------------


def test_astar_to_any_stops_at_the_nearest_goal():
    grid = {(q, r): Hex(coord=(q, r)) for q in range(8) for r in range(3)}
    path = astar_to_any(grid, (0, 0), {(6, 0), (2, 0)}, lambda h: 1.0, aim=(6, 0))
    assert path[0] == (0, 0)
    assert path[-1] == (2, 0), "walked past the nearer goal"


def test_astar_to_any_returns_a_single_hex_when_it_starts_on_one():
    grid = {(0, 0): Hex(coord=(0, 0))}
    assert astar_to_any(grid, (0, 0), {(0, 0)}, lambda h: 1.0) == [(0, 0)]


def test_astar_to_any_with_no_goals():
    grid = {(0, 0): Hex(coord=(0, 0))}
    assert astar_to_any(grid, (0, 0), set(), lambda h: 1.0) is None


def test_astar_to_any_returns_none_when_every_goal_is_walled_off():
    grid = {(q, r): Hex(coord=(q, r)) for q in range(3) for r in range(3)}
    cost = lambda h: float("inf") if h.coord[0] == 1 else 1.0  # noqa: E731
    assert astar_to_any(grid, (0, 0), {(2, 0)}, cost) is None


def test_astar_to_any_takes_the_cheapest_goal_not_the_closest():
    """Nearest in hexes is not nearest in cost, and cost is what decides."""
    grid = {(q, r): Hex(coord=(q, r)) for q in range(6) for r in range(3)}

    def cost(h):
        return 50.0 if h.coord == (1, 0) else 1.0

    # (2,0) is two hexes away but behind an expensive hex; (0,2) is further but cheap.
    path = astar_to_any(grid, (0, 0), {(2, 0), (0, 2)}, cost)
    assert path[-1] == (0, 2)
