"""The drainage network's shape, measured.

Two halves.  The first builds networks by hand, where the right answer is known, and is
what makes the metrics trustworthy — without it, a later change to the generator could
move a number for reasons that have nothing to do with the terrain.  The second runs the
real pipeline and asks whether the rivers it makes look like rivers.
"""

import math

import pytest

from tests.worlds import build_world
from worldgen.analysis import drainage
from worldgen.core.hex import Hex
from worldgen.core.hex_grid import corner_neighbors, hex_corner_keys, neighbors
from worldgen.core.world_state import River, WorldState

# --- hand-built networks -----------------------------------------------------
#
# The networks are drawn as hex paths, which are easy to read, and laid onto corners.
# Corner (q, r, 0) stands for hex (q, r): the six corners of that kind two sides away from
# it are exactly the corners standing for the hex's six neighbours, so a hex step becomes
# two sides through the corner between, and every tree keeps its shape.


def c(hex_coord):
    """The corner standing for a hex."""
    return (hex_coord[0], hex_coord[1], 0)


def mid(a, b):
    """The corner between the corners standing for two neighbouring hexes."""
    (q, r), (bq, br) = a, b
    via = {
        (-1, 1): (q, r, 1),
        (0, 1): (q, r, 1),
        (0, -1): (q, r - 1, 1),
        (-1, 0): (q, r - 1, 1),
        (1, -1): (q + 1, r - 1, 1),
        (1, 0): (q + 1, r - 1, 1),
    }
    return via[(bq - q, br - r)]


def course(path):
    """A hex path laid onto corners."""
    out = [c(path[0])]
    for a, b in zip(path, path[1:], strict=False):
        out += [mid(a, b), c(b)]
    return out


def _world(rivers: list[list[tuple[int, int]]]) -> WorldState:
    """A world that is all land, carrying exactly the given river paths.

    Paths are given the way hydrology emits them: a tributary *ends on* the trunk corner
    it joins, so the junction is the one two paths have in common.
    """
    coords = {h for path in rivers for h in path}
    qs = [q for q, _ in coords]
    rs = [r for _, r in coords]
    state = WorldState.empty(seed=1, width=1, height=1)
    state.hexes = {}
    for q in range(min(qs) - 2, max(qs) + 3):
        for r in range(min(rs) - 2, max(rs) + 3):
            state.hexes[(q, r)] = Hex(coord=(q, r))
    state.rivers = [River(corners=course(path), flow_volume=1.0) for path in rivers]
    return state


def _comb(rows: int = 5, length: int = 10) -> list[list[tuple[int, int]]]:
    """Straight channels side by side, all flowing east, never meeting.

    The artefact this module exists to detect, built deliberately so the metric has a
    control to be checked against.
    """
    return [[(q, r) for q in range(length)] for r in range(rows)]


# A balanced order-3 tree: four headwaters pair into two order-2 links, which pair into
# one trunk.  Horton's ratio over it is exactly 2, which is what makes it a fixture.
#
# A corner has three neighbours, each between it and two of the hex's six, so on corners
# two streams arriving at a junction from neighbouring hexes would already have met one
# side upstream.  Each junction here takes its two inflows and its outflow from different
# pairs, so the three meet where the hex path says they do.
_TREE: list[list[tuple[int, int]]] = [
    [(-4, 6), (-3, 6), (-2, 6)],  # order 1
    [(-2, 8), (-2, 7), (-2, 6)],  # order 1
    [(0, 4), (1, 4), (2, 4)],  # order 1
    [(4, 2), (3, 3), (2, 4)],  # order 1
    [(-2, 6), (-1, 6), (0, 6)],  # order 2
    [(2, 4), (1, 5), (0, 6)],  # order 2
    [(0, 6), (0, 7), (0, 8)],  # order 3
]

# A trunk along a row and a tributary arriving from below it, meeting at (2, 0).
_Y = [[(0, 0), (1, 0), (2, 0)], [(2, 2), (2, 1), (2, 0)]]


def test_the_fixtures_are_actually_adjacent():
    """Guards the other fixtures: a path with a gap in it would measure nothing."""
    for path in _TREE + _comb():
        for a, b in zip(path, path[1:], strict=False):
            assert b in neighbors(a), f"{a} -> {b} is not a step"
        laid = course(path)
        for a, b in zip(laid, laid[1:], strict=False):
            assert b in corner_neighbors(a), f"{a} -> {b} is not one side"


def test_a_y_junction_is_one_confluence():
    state = _world(_Y)
    metrics = drainage.drainage_metrics(state)
    assert metrics.confluence_count == 1
    assert metrics.conflicts == 0


def test_a_tributary_ending_on_the_trunk_rebuilds_the_fork():
    """The split-river representation must put back together into one graph."""
    net = drainage.build_network(_world(_Y))
    assert sorted(net.upstream[c((2, 0))]) == sorted([mid((1, 0), (2, 0)), mid((2, 1), (2, 0))])
    assert net.outlets == frozenset({c((2, 0))})


def test_strahler_promotes_only_when_two_equal_orders_meet():
    """A first-order stream joining a second-order one leaves it second-order."""
    net = drainage.build_network(
        _world(
            [
                *_Y,  # 1 + 1 -> 2 at (2, 0)
                [(2, 0), (3, 0)],
                [(4, -1), (3, 0)],  # a 1 joining the 2
            ]
        )
    )
    assert net.order[c((2, 0))] == 2
    assert net.order[c((3, 0))] == 2, "a lone tributary must not promote the trunk"


def test_a_balanced_binary_tree_has_a_bifurcation_ratio_of_two():
    net = drainage.build_network(_world(_TREE))
    counts = drainage.links_by_order(net)
    assert counts == {1: 4, 2: 2, 3: 1}
    assert drainage.bifurcation_ratio(counts) == pytest.approx(2.0)


def test_the_bifurcation_ratio_is_undefined_below_three_orders():
    """One step is not evidence of a law — and no third order is itself the finding."""
    assert math.isnan(drainage.bifurcation_ratio({1: 2, 2: 1}))
    assert math.isnan(drainage.bifurcation_ratio({}))


def test_links_are_runs_not_corners():
    """A long straight headwater is one first-order link, however many sides it covers."""
    net = drainage.build_network(_world([[(q, 0) for q in range(12)]]))
    assert drainage.links_by_order(net) == {1: 1}


# --- the artefact detectors --------------------------------------------------


def test_a_comb_of_parallel_channels_scores_high():
    """The metric's control, and the reason it is not the headline.

    A comb of touching channels scores 0.25 here against 0.0 for the tree, but real
    generated maps score near zero on it — their channels are sparse enough that two
    neighbouring rivers sit several kilometres apart and never touch at all.  So this
    measures something true and rarely triggered; what the pipeline tests assert on is
    the branching family below.

    A straight course along hexsides zigzags between two headings 60 degrees apart, so
    its steps agree at most cos 30 = 0.866 on a direction, not 1.
    """
    metrics = drainage.drainage_metrics(_world(_comb()))
    assert metrics.parallel_pair_fraction > 0.2
    assert metrics.azimuth_concentration > 0.85
    assert metrics.confluence_count == 0
    assert math.isnan(metrics.bifurcation_ratio)


def test_a_dendritic_tree_scores_far_lower_than_a_comb():
    tree = drainage.drainage_metrics(_world(_TREE))
    comb = drainage.drainage_metrics(_world(_comb()))
    assert tree.parallel_pair_fraction < comb.parallel_pair_fraction / 2
    assert tree.joined_pair_fraction > comb.joined_pair_fraction
    assert tree.headwater_azimuth_concentration < comb.headwater_azimuth_concentration


def test_headwaters_of_a_tree_point_every_which_way():
    net = drainage.build_network(_world(_TREE))
    assert drainage.azimuth_concentration(net, orders={1}) < 0.45


def test_rivers_all_heading_one_way_concentrate():
    """`river_azimuth_concentration` reduces each river to a heading, so it can see two
    channels running side by side that never touch — which a pairs metric cannot."""
    assert drainage.river_azimuth_concentration(_world(_comb())) == pytest.approx(1.0)
    assert drainage.river_azimuth_concentration(_world(_TREE)) < 0.6


def test_the_first_order_ratio_is_defined_where_the_bifurcation_ratio_is_not():
    """A two-order network has no Horton ratio but still has a first step."""
    assert math.isnan(drainage.bifurcation_ratio({1: 12, 2: 1}))
    assert drainage.first_order_link_ratio({1: 12, 2: 1}) == 12.0
    assert drainage.first_order_link_ratio({1: 12}) == math.inf


def test_contradictory_edges_are_counted_not_swallowed():
    """Two receivers for one hex means something upstream is wrong; say so."""
    state = _world([[(0, 0), (1, 0)], [(0, 0), (0, 1)]])
    net = drainage.build_network(state)
    assert net.conflicts == 1


def test_a_loop_is_refused():
    # Round the six corners of one hex and back to the first.
    ring = hex_corner_keys((0, 0))
    state = _world([[(0, 0)]])
    state.rivers = [River(corners=[*ring, ring[0]], flow_volume=1.0)]
    net = drainage.build_network(state)
    assert net.conflicts == 1
    # The refused edge is the last one traced, the one that would have closed the loop,
    # so the corner it came from is left as the end of an open chain.
    assert net.outlets == frozenset({ring[-1]})


# --- the real pipeline -------------------------------------------------------

SEEDS = (42, 7, 1234)


@pytest.fixture(scope="module", params=SEEDS)
def hydro_metrics(request):
    """Drainage metrics for a 64x64 world, stopped after hydrology.

    64 and not smaller because branching is a question about *basins*, and a 48 km square
    does not hold one: its land comes to about 1100 km2 broken into many short coastal
    catchments, the largest around 100 km2, and a basin only shows as many stream orders
    as its area divides into channel-sized pieces.  At 48 km no threshold that still draws
    fordable watercourses reaches third order on every seed — the map is too small for the
    question, not badly drained.  Stopping after hydrology keeps each world under a second.
    """
    state = build_world(seed=request.param, width=64, height=64, until="HydrologyStage")
    return drainage.drainage_metrics(state)


def test_the_network_is_a_forest(hydro_metrics):
    assert hydro_metrics.conflicts == 0


def test_strahler_order_never_decreases_downstream():
    state = build_world(seed=42, width=64, height=64, until="HydrologyStage")
    net = drainage.build_network(state)
    for upper, lower in net.downstream.items():
        assert net.order[lower] >= net.order[upper]


def test_the_rebuilt_confluences_agree_with_the_hydrology_tag():
    """Two independent routes to the same number — the courses, and the corner tags.

    Inland only, on both sides.  Two rivers ending on the same shore have two upstream
    neighbours in the graph but have not met: they have both arrived somewhere.  Counting
    that as a fork would inflate every map with a coast on it.
    """
    state = build_world(seed=42, width=64, height=64, until="HydrologyStage")
    tagged = {k for k, tags in state.river_corners.items() if "confluence" in tags}
    net = drainage.build_network(state)
    rebuilt = drainage.confluences(net, drainage.channel_corners(state, net))
    assert rebuilt == tagged


def test_metrics_are_reproducible_for_a_seed():
    a = drainage.drainage_metrics(build_world(seed=7, width=64, height=64, until="HydrologyStage"))
    b = drainage.drainage_metrics(build_world(seed=7, width=64, height=64, until="HydrologyStage"))
    # Field by field rather than `a == b`, because the bifurcation ratio is legitimately
    # NaN on a network with no third order, and NaN is not equal to itself.
    for name in vars(a):
        left, right = getattr(a, name), getattr(b, name)
        if isinstance(left, float) and math.isnan(left):
            assert math.isnan(right), name
        else:
            assert left == right, name


def test_drainage_density_is_plausible(hydro_metrics):
    """A 1 km sample only resolves channels draining tens of km2, so this sits well below
    the 0.5-5 km/km2 a field survey of the same landscape would report."""
    assert 0.01 <= hydro_metrics.drainage_density <= 0.30


def test_touching_channels_still_join(hydro_metrics):
    """Not an artefact detector — a guard.

    Generated channels are sparse enough that the ones which do touch are nearly always
    the same river, so this reads near zero today and must keep doing so.  A change that
    made neighbouring channels run alongside each other instead of merging would show up
    here first.
    """
    assert hydro_metrics.parallel_pair_fraction <= 0.15


def test_tributaries_join_their_trunk_rather_than_running_beside_it(hydro_metrics):
    """Two rivers down one valley floor, a hex or two apart, that never meet.

    The share of tributaries running more than 5 km within 2 km of the river they join.
    Measured at 64x64 on seeds 42, 7 and 1234: 0.105, 0.119 and 0.297 with each corner at
    the floor (`corner_floor_blend` 0), which left a level lane a hex out from every river;
    0.067, 0.070 and 0.051 at the default 0.25.  What remains is mostly the parallel
    drainage of broad plains, which real landscapes have too.
    """
    assert hydro_metrics.long_side_by_side_fraction <= 0.15


def test_rivers_do_not_share_one_heading(hydro_metrics):
    """Passes today, and is here to stay honest about what is wrong.

    With N rivers an isotropic set still scores about 1/sqrt(N) by chance — 0.21 to 0.38
    at the counts these maps produce — and every seed measured between 0.04 and 0.37.  The
    channels are *not* collimated: what makes them read as parallel is that they never
    branch, which is what the tests below measure.
    """
    assert hydro_metrics.river_azimuth_concentration <= 0.6


# --- what is actually broken -------------------------------------------------
#
# Measured before any fix, at 48x48 and 64x64 on seeds 42, 7 and 1234:
#
#     seed  size    max order   N1/N2   confluence rate
#       42  48x48           1     inf             0.000
#        7  48x48           2    7.00             0.043
#     1234  48x48           2   12.00             0.000
#       42  64x64           2    9.50             0.019
#        7  64x64           2   11.00             0.011
#     1234  64x64           1     inf             0.000
#
# Every bound below is a round number well clear of those, not the measurement shaved by
# an epsilon.
#
# What fixed them was not erosion but `channel_min_discharge`, which turned out to decide
# the branching and not just the density: a basin shows only as many Strahler orders as
# its area divides into channel-sized pieces, and at 20,000 the basin-to-threshold ratio
# on a 64 km map was about five.  Adding a drainage-area term to the incision law moved
# none of these numbers measurably; lowering the threshold to 6,000 moved all four.


def test_the_network_reaches_a_third_order(hydro_metrics):
    """Two first-order streams make a second; two seconds make a third.

    Before the threshold change no map got past second order, which is another way of
    saying the network was a set of threads and not a tree.  Now 3 on every seed.
    """
    assert hydro_metrics.strahler_max >= 3


def test_first_order_streams_pair_up(hydro_metrics):
    """Horton's law puts N(1)/N(2) between about 3 and 5.

    Was 9 to infinity — a crowd of unbranched fingers that never paired.  Now 4.6 to 5.9,
    so the ceiling is 8: clear of the measurement, and still far under the old numbers.
    """
    assert 2.0 <= hydro_metrics.first_order_link_ratio <= 8.0


def test_the_bifurcation_ratio_is_plausible(hydro_metrics):
    """That it is a number at all is most of the point: the ratio needs three orders, and
    before the threshold change there were never three, so this was NaN on every seed.

    The band is wide deliberately.  Field networks cluster between 3 and 5, but a network
    with exactly three orders offers only two ratios to average and its top order is a
    single link, which pushes N(2)/N(3) up and the geometric mean with it — measured 5.7
    to 8.3 here.  Tightening the ceiling would be asserting that these maps hold bigger
    basins than they do.
    """
    assert not math.isnan(hydro_metrics.bifurcation_ratio), "fewer than three orders"
    assert 2.0 <= hydro_metrics.bifurcation_ratio <= 10.0


def test_the_network_has_confluences(hydro_metrics):
    """Was 0.000 to 0.020 at 64 km; now 0.036 to 0.056.

    A hand-built dendritic tree scores about 0.16, but that tree is all junction and no
    trunk; a real map spends most of its channel hexes running between forks.
    """
    assert hydro_metrics.confluence_rate >= 0.025
