"""Drainage on the corner graph, on hand-built worlds where the right answer is known."""

import numpy as np
import pytest

from worldgen.core.hex import Hex, TerrainClass
from worldgen.core.hex_grid import corner_hexes, corner_neighbors
from worldgen.stages.corner_drainage import (
    accumulate,
    build_network,
    drain_corner,
    flow_direction,
    is_lake_node,
    trace_streams,
)

N = 10


def _world(elevation, water=(), lakes=()):
    """An NxN axial grid; *water* hexes are sea, *lakes* closed or open lake hexes."""
    hexes = {}
    for q in range(N):
        for r in range(N):
            c = (q, r)
            if c in water:
                tc = TerrainClass.OPEN_WATER
            elif c in lakes:
                tc = TerrainClass.INLAND_WATER
            else:
                tc = TerrainClass.LAND
            hexes[c] = Hex(coord=c, elevation=float(elevation(q, r)), terrain_class=tc)
    return hexes


def _drain(hexes, closed=(), open_lakes=(), seed=0, wander=2.0):
    land = {c for c, h in hexes.items() if h.terrain_class == TerrainClass.LAND}
    ocean = {c for c, h in hexes.items() if h.terrain_class == TerrainClass.OPEN_WATER}
    heights = {c: h.elevation for c, h in hexes.items()}
    net = build_network(heights, land, ocean, set(closed), [list(x) for x in open_lakes])
    drainage = flow_direction(net, np.random.default_rng(seed), wander)
    sources = {}
    for c in land:
        d = drain_corner(c, net, drainage)
        if d is not None:
            sources[d] = sources.get(d, 0.0) + 1.0
    for node, comp in net.lake_hexes.items():
        sources[node] = sources.get(node, 0.0) + len(comp)
    drainage.acc = accumulate(drainage.flow, sources)
    return net, drainage, sources, land


SEA = {(0, r) for r in range(N)}


def _slope(q, r):
    # Rising away from the sea in the west, with a little cross-slope so it is not flat.
    return 10.0 * q + 0.3 * ((q * 7 + r * 3) % 5)


@pytest.fixture
def slope():
    return _drain(_world(_slope, water=SEA))


def test_every_corner_drains_downhill_to_somewhere(slope):
    net, drainage, _, _ = slope
    for node, nxt in drainage.flow.items():
        if node in net.terminal:
            assert nxt is None
            continue
        assert nxt is not None, f"{node} has nowhere to go"
        assert nxt in net.neighbors[node]
        assert drainage.order[nxt] < drainage.order[node]
        assert drainage.filled[nxt] <= drainage.filled[node]


def test_water_runs_only_between_land_hexes(slope):
    # A step along a side with water or the map edge on one hand would be a river drawn
    # along the shore or the frame.
    net, drainage, _, land = slope
    for node, nxt in drainage.flow.items():
        if nxt is None or is_lake_node(nxt):
            continue
        shared = set(corner_hexes(node)) & set(corner_hexes(nxt))
        assert len(shared) == 2 and shared <= land


def test_all_the_rain_leaves_the_map(slope):
    net, drainage, sources, _ = slope
    out = sum(drainage.acc[t] for t in net.terminal if t in drainage.acc)
    assert out == pytest.approx(sum(sources.values()))


def test_a_pit_is_filled_and_drained_rather_than_trapping_water():
    def pitted(q, r):
        return _slope(q, r) - (40.0 if (q, r) == (6, 4) else 0.0)

    net, drainage, sources, _ = _drain(_world(pitted, water=SEA))
    assert all(nxt is not None for n, nxt in drainage.flow.items() if n not in net.terminal)
    out = sum(drainage.acc[t] for t in net.terminal if t in drainage.acc)
    assert out == pytest.approx(sum(sources.values()))


def test_streams_are_chains_of_corners_ending_where_water_leaves_or_joins():
    net, drainage, _, _ = _drain(_world(_slope, water=SEA))
    streams = trace_streams(drainage, threshold=4.0)
    assert streams.paths
    on_a_course = {c for p in streams.paths for c in p[:-1]}
    for path in streams.paths:
        for a, b in zip(path, path[1:], strict=False):
            assert b in corner_neighbors(a)
            assert drainage.flow[a] == b
        last = path[-1]
        assert last in net.terminal or last in on_a_course or drainage.flow.get(last) is None
    # Every channel corner is on a course.
    covered = {c for p in streams.paths for c in p}
    assert streams.channel <= covered


def test_a_tributary_ends_on_the_trunk_and_the_trunk_carries_on():
    net, drainage, _, _ = _drain(_world(_slope, water=SEA))
    streams = trace_streams(drainage, threshold=2.0)
    ends = {p[-1] for p in streams.paths}
    junctions = {c for c, ups in streams.upstream.items() if len(ups) >= 2}
    assert junctions, "this slope should have at least one confluence"
    for j in junctions:
        assert j in ends
        # Exactly one course passes through the junction rather than ending on it.
        through = [p for p in streams.paths if j in p[:-1]]
        assert len(through) == 1


def test_the_same_seed_drains_the_same_way():
    a = _drain(_world(_slope, water=SEA), seed=3)[1].flow
    b = _drain(_world(_slope, water=SEA), seed=3)[1].flow
    assert a == b


def test_a_steep_exponent_takes_the_steepest_way_down():
    net, drainage, _, _ = _drain(_world(_slope, water=SEA), wander=200.0)
    for node, nxt in drainage.flow.items():
        if nxt is None or node in net.terminal or is_lake_node(nxt):
            continue
        lower = [n for n in net.neighbors[node] if drainage.order[n] < drainage.order[node]]
        if any(n in net.wet for n in lower):
            continue
        best = max(drainage.filled[node] - drainage.filled[n] for n in lower)
        assert drainage.filled[node] - drainage.filled[nxt] == pytest.approx(best)


LAKE = {(5, 4), (5, 5), (6, 4)}


def _basin(q, r):
    # The slope with a hollow at the lake, its rim lowest on the seaward side.
    base = _slope(q, r)
    return base - 30.0 if (q, r) in LAKE else base


def test_an_open_lake_collects_its_inflow_and_spills_it_onward():
    hexes = _world(_basin, water=SEA, lakes=LAKE)
    net, drainage, sources, _ = _drain(hexes, open_lakes=[LAKE])
    (lake,) = net.lake_hexes
    assert drainage.flow[lake] is not None
    assert drainage.flow[lake] == drainage.parent[lake]
    # The spill corner is on the lake shore and does not run straight back in.
    spill = drainage.flow[lake]
    assert lake in net.neighbors[spill]
    assert drainage.flow[spill] != lake
    # Everything still reaches the sea, the lake's water included.
    out = sum(drainage.acc[t] for t in net.terminal if t in drainage.acc)
    assert out == pytest.approx(sum(sources.values()))
    assert drainage.acc[spill] >= drainage.acc[lake]


def test_a_stream_ends_at_an_open_lake_and_a_new_one_leaves_it():
    hexes = _world(_basin, water=SEA, lakes=LAKE)
    net, drainage, _, _ = _drain(hexes, open_lakes=[LAKE])
    (lake,) = net.lake_hexes
    streams = trace_streams(drainage, threshold=2.0)
    spill = drainage.flow[lake]
    assert any(p[0] == spill for p in streams.paths)
    for p in streams.paths:
        assert not any(is_lake_node(c) for c in p)


def test_a_closed_lake_is_where_its_water_stops():
    hexes = _world(_basin, water=SEA, lakes=LAKE)
    net, drainage, sources, _ = _drain(hexes, closed=LAKE)
    shore = {c for c in net.wet if any(h in LAKE for h in corner_hexes(c))}
    assert shore and shore <= net.terminal
    into_lake = sum(drainage.acc[c] for c in shore)
    assert into_lake > 0


def test_a_map_too_dry_for_any_channel_keeps_its_largest_drainage_line():
    net, drainage, _, _ = _drain(_world(_slope, water=SEA))
    streams = trace_streams(drainage, threshold=1e9)
    assert len(streams.paths) == 1
    assert streams.paths[0][-1] in net.terminal
