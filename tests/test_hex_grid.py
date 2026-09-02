from worldgen.core.hex import Hex, TerrainClass
from worldgen.core.hex_grid import (
    astar,
    dedupe_road_paths,
    distance,
    grade_reachable_count,
    neighbors,
    ring,
    split_path_on_water,
    water_transitions,
)
from worldgen.core.world_state import ROAD_TIER_RANK, Road, RoadTier


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
    grid[(2, 0)] = Hex(coord=(2, 0), terrain_class=TerrainClass.LAKE)
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


# --- dedupe_road_paths -------------------------------------------------------


def _grid(*paths, water: frozenset = frozenset()) -> dict:
    """A grid covering every coord in *paths*, with the given coords made ocean."""
    return {
        c: Hex(coord=c, terrain_class=TerrainClass.OCEAN if c in water else TerrainClass.FLAT)
        for p in paths
        for c in p
    }


def _tier_rank(road) -> int:
    return ROAD_TIER_RANK[road.tier]


def _drawn_edges(result) -> list:
    """Every edge in a dedupe result, as unordered pairs, in draw order."""
    return [frozenset((a, b)) for _, leg in result for a, b in zip(leg, leg[1:], strict=False)]


def _legs_of(result, road) -> list:
    return [leg for r, leg in result if r is road]


def test_dedupe_single_road_keeps_its_whole_path():
    road = Road(path=_path(), tier=RoadTier.PRIMARY)
    grid = _grid(road.path)
    assert dedupe_road_paths([road], grid, _tier_rank) == [(road, _path())]


def test_dedupe_no_roads():
    assert dedupe_road_paths([], _water_grid(set()), _tier_rank) == []


def test_dedupe_draws_every_edge_exactly_once():
    """The whole point: two routes down a shared trunk must not stack polylines."""
    trunk = Road(path=_path(4), tier=RoadTier.PRIMARY)
    branch = Road(path=[(0, 0), (1, 0), (2, 0), (2, 1)], tier=RoadTier.TRACK)
    grid = _grid(trunk.path, branch.path)

    edges = _drawn_edges(dedupe_road_paths([trunk, branch], grid, _tier_rank))
    assert len(edges) == len(set(edges)), "an edge was drawn more than once"
    # Nothing is lost either — every edge either road travels is still drawn.
    expected = {
        frozenset((a, b))
        for road in (trunk, branch)
        for a, b in zip(road.path, road.path[1:], strict=False)
    }
    assert set(edges) == expected


def test_dedupe_awards_shared_edges_to_the_higher_tier():
    """The track keeps only the spur it does not share with the primary road."""
    trunk = Road(path=_path(4), tier=RoadTier.PRIMARY)
    branch = Road(path=[(0, 0), (1, 0), (2, 0), (2, 1)], tier=RoadTier.TRACK)
    grid = _grid(trunk.path, branch.path)

    result = dedupe_road_paths([trunk, branch], grid, _tier_rank)
    assert _legs_of(result, trunk) == [_path(4)]
    assert _legs_of(result, branch) == [[(2, 0), (2, 1)]]


def test_dedupe_returns_legs_in_ascending_rank():
    """Renderers paint in order, so the important roads must come last."""
    track = Road(path=[(0, 1), (1, 1)], tier=RoadTier.TRACK)
    secondary = Road(path=[(0, 2), (1, 2)], tier=RoadTier.SECONDARY)
    primary = Road(path=[(0, 0), (1, 0)], tier=RoadTier.PRIMARY)
    grid = _grid(track.path, secondary.path, primary.path)

    result = dedupe_road_paths([primary, track, secondary], grid, _tier_rank)
    assert [road.tier for road, _ in result] == [
        RoadTier.TRACK,
        RoadTier.SECONDARY,
        RoadTier.PRIMARY,
    ]


def test_dedupe_ties_go_to_the_first_road_in_list_order():
    first = Road(path=[(0, 0), (1, 0), (2, 0)], tier=RoadTier.TRACK)
    second = Road(path=[(0, 0), (1, 0), (1, 1)], tier=RoadTier.TRACK)
    grid = _grid(first.path, second.path)

    result = dedupe_road_paths([first, second], grid, _tier_rank)
    assert _legs_of(result, first) == [[(0, 0), (1, 0), (2, 0)]]
    assert _legs_of(result, second) == [[(1, 0), (1, 1)]]


def test_dedupe_tie_break_follows_list_order_when_reversed():
    """Same two roads, swapped — the shared trunk changes hands."""
    first = Road(path=[(0, 0), (1, 0), (2, 0)], tier=RoadTier.TRACK)
    second = Road(path=[(0, 0), (1, 0), (1, 1)], tier=RoadTier.TRACK)
    grid = _grid(first.path, second.path)

    result = dedupe_road_paths([second, first], grid, _tier_rank)
    assert _legs_of(result, second) == [[(0, 0), (1, 0), (1, 1)]]
    assert _legs_of(result, first) == [[(1, 0), (2, 0)]]


def test_dedupe_breaks_a_run_claimed_in_its_middle():
    """A higher road taking the middle edges leaves two separate runs, not one."""
    track = Road(path=_path(), tier=RoadTier.TRACK)
    primary = Road(path=[(2, 0), (3, 0)], tier=RoadTier.PRIMARY)
    grid = _grid(track.path, primary.path)

    result = dedupe_road_paths([track, primary], grid, _tier_rank)
    assert _legs_of(result, track) == [[(0, 0), (1, 0), (2, 0)], [(3, 0), (4, 0), (5, 0)]]
    assert _legs_of(result, primary) == [[(2, 0), (3, 0)]]


def test_dedupe_drops_a_leg_claimed_end_to_end():
    """A track duplicating a primary road contributes nothing to draw."""
    primary = Road(path=_path(4), tier=RoadTier.PRIMARY)
    track = Road(path=_path(4), tier=RoadTier.TRACK)
    grid = _grid(primary.path)

    result = dedupe_road_paths([primary, track], grid, _tier_rank)
    assert _legs_of(result, track) == []
    assert _legs_of(result, primary) == [_path(4)]


def test_dedupe_splits_on_water_and_never_bridges_it():
    """Legs stop at the shore; no polyline spans the water gap."""
    road = Road(path=_path(), tier=RoadTier.PRIMARY)
    grid = _grid(road.path, water=frozenset({(2, 0), (3, 0)}))

    result = dedupe_road_paths([road], grid, _tier_rank)
    assert _legs_of(result, road) == [[(0, 0), (1, 0)], [(4, 0), (5, 0)]]
    for _, leg in result:
        for a, b in zip(leg, leg[1:], strict=False):
            assert distance(a, b) == 1, "a leg jumped across the water gap"


def test_dedupe_never_draws_a_water_hex():
    track = Road(path=_path(), tier=RoadTier.TRACK)
    primary = Road(path=_path(4), tier=RoadTier.PRIMARY)
    water = frozenset({(2, 0)})
    grid = _grid(track.path, water=water)

    result = dedupe_road_paths([track, primary], grid, _tier_rank)
    drawn = {c for _, leg in result for c in leg}
    assert not (drawn & water)


def test_dedupe_water_split_still_dedupes_the_shared_trunk():
    """Deduping and water splitting compose: shared edges go once, on the land side."""
    primary = Road(path=_path(), tier=RoadTier.PRIMARY)
    track = Road(path=_path(), tier=RoadTier.TRACK)
    grid = _grid(primary.path, water=frozenset({(2, 0)}))

    result = dedupe_road_paths([primary, track], grid, _tier_rank)
    assert _legs_of(result, track) == []
    assert _legs_of(result, primary) == [[(0, 0), (1, 0)], [(3, 0), (4, 0), (5, 0)]]
    edges = _drawn_edges(result)
    assert len(edges) == len(set(edges))


def test_dedupe_road_entirely_on_water_draws_nothing():
    road = Road(path=_path(4), tier=RoadTier.PRIMARY)
    grid = _grid(road.path, water=frozenset(road.path))
    assert dedupe_road_paths([road], grid, _tier_rank) == []
