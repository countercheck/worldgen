import statistics

import numpy as np
import pytest

from worldgen.core.config import WorldConfig
from worldgen.core.hex import TerrainClass
from worldgen.core.hex_grid import distance, neighbors
from worldgen.core.pipeline import GeneratorPipeline
from worldgen.core.world_state import River, WorldState
from worldgen.stages.elevation import ElevationStage
from worldgen.stages.erosion import ErosionStage
from worldgen.stages.hydrology import HydrologyStage, _split_at_confluences
from worldgen.stages.terrain_class import TerrainClassificationStage
from worldgen.stages.water_bodies import WaterBodiesStage

from .worlds import build_world


def _build_pipeline(seed: int = 42, width: int = 32, height: int = 32):
    cfg = WorldConfig(width=width, height=height, erosion_iterations=500)
    p = GeneratorPipeline(seed, cfg)
    p.add_stage(ElevationStage)
    p.add_stage(ErosionStage)
    p.add_stage(TerrainClassificationStage)
    p.add_stage(WaterBodiesStage)
    p.add_stage(HydrologyStage)
    return p


@pytest.fixture(scope="module")
def hydro_state():
    return _build_pipeline().run()


def test_river_flow_nonzero(hydro_state):
    river_hexes = [h for h in hydro_state.hexes.values() if h.river_flow > 0]
    assert len(river_hexes) > 0, "No river hexes found after hydrology stage"


def test_river_flow_normalized(hydro_state):
    for h in hydro_state.hexes.values():
        assert 0.0 <= h.river_flow <= 1.0, f"river_flow {h.river_flow} out of [0, 1]"


def test_river_paths_connected(hydro_state):
    for river in hydro_state.rivers:
        assert len(river.hexes) >= 2, "River has fewer than 2 hexes"
        for i in range(len(river.hexes) - 1):
            a, b = river.hexes[i], river.hexes[i + 1]
            assert b in neighbors(a), f"Non-adjacent hexes in river path: {a} -> {b}"


def test_rivers_reach_ocean(hydro_state):
    # Each river must terminate at ocean/lake, a grid border, OR a confluence with
    # another river (tributaries end AT the confluence hex, which is also part of the
    # higher-flow trunk, so the polylines visually connect).
    water_set = {
        coord
        for coord, h in hydro_state.hexes.items()
        if h.terrain_class in (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER)
    }
    # Build a set of all hexes that appear in any river so we can detect confluences.
    all_river_hexes: set[tuple[int, int]] = set()
    for river in hydro_state.rivers:
        all_river_hexes.update(river.hexes)

    w, h = hydro_state.width, hydro_state.height
    for river in hydro_state.rivers:
        mouth = river.hexes[-1]
        q, r = mouth
        on_border = q == 0 or q == w - 1 or r == 0 or r == h - 1
        reaches_water = any(n in water_set for n in neighbors(mouth)) or mouth in water_set
        # A tributary that stopped at a confluence: its last hex is adjacent to another
        # river's trunk hex that is either explicitly tagged as a confluence or carries
        # at least as much downstream flow as the tributary mouth (and is not itself a
        # headwater).
        mouth_flow = hydro_state.hexes[mouth].river_flow if mouth in hydro_state.hexes else 0.0
        at_confluence = any(
            n in all_river_hexes
            and n not in river.hexes
            and n in hydro_state.hexes
            and (
                "confluence" in hydro_state.hexes[n].tags
                or (
                    hydro_state.hexes[n].river_flow >= mouth_flow
                    and "headwater" not in hydro_state.hexes[n].tags
                )
            )
            for n in neighbors(mouth)
        )
        assert reaches_water or on_border or at_confluence, (
            f"River mouth {mouth} does not reach water body, grid border, or confluence"
        )


def test_flow_accumulates_downstream(hydro_state):
    # Flow accumulation (river_flow) must be non-decreasing along a river path —
    # each step downstream collects more water. Checks the accumulation invariant
    # without depending on the filled vs. actual elevation distinction.
    for river in hydro_state.rivers:
        river_hexes = [
            hydro_state.hexes[c]
            for c in river.hexes
            if c in hydro_state.hexes and hydro_state.hexes[c].river_flow > 0
        ]
        for i in range(len(river_hexes) - 1):
            flow_a = river_hexes[i].river_flow
            flow_b = river_hexes[i + 1].river_flow
            assert flow_b >= flow_a - 1e-9, (
                f"River_flow decreases downstream: {flow_a:.4f} -> {flow_b:.4f}"
            )


def test_tags_assigned(hydro_state):
    all_tags: set[str] = set()
    for h in hydro_state.hexes.values():
        all_tags.update(h.tags)
    assert "headwater" in all_tags, "No headwater tags found"
    assert "river_mouth" in all_tags, "No river_mouth tags found"


def test_river_tag_on_river_paths(hydro_state):
    # Every hex in a River path that is a land hex must carry the "river" tag.
    water_classes = {TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER}
    for river in hydro_state.rivers:
        for coord in river.hexes:
            if coord not in hydro_state.hexes:
                continue
            hx = hydro_state.hexes[coord]
            if hx.terrain_class in water_classes:
                continue
            assert "river" in hx.tags, f"River path hex {coord} missing 'river' tag"


def test_flow_volume(hydro_state):
    rivers = hydro_state.rivers
    assert all(0.0 < r.flow_volume <= 1.0 for r in rivers), "flow_volume out of (0, 1]"
    # flow_volume must reflect mouth accumulation, not headwater discharge.
    # Each river's flow_volume (normalized accumulation at its last land hex) must be
    # >= the river_flow of its headwater (the first hex in the path), because rivers
    # accumulate water as they flow downstream.
    for river in rivers:
        head = river.hexes[0]
        head_flow = hydro_state.hexes[head].river_flow if head in hydro_state.hexes else 0.0
        assert (
            river.flow_volume >= head_flow - 1e-9
        ), (  # 1e-9 tolerance for floating-point arithmetic
            f"flow_volume {river.flow_volume:.6f} < headwater river_flow {head_flow:.6f}; "
            "flow_volume must represent mouth discharge, not headwater"
        )


def test_no_border_edge_creep(hydro_state):
    # Rivers must not "creep" along the map edge: no river path should contain two
    # consecutive hexes that are both on the grid border.  This validates that the
    # border-land -> border-land flow termination in _flow_direction works correctly.
    w, h = hydro_state.width, hydro_state.height

    def on_border(coord):
        q, r = coord
        return q == 0 or q == w - 1 or r == 0 or r == h - 1

    for river in hydro_state.rivers:
        for i in range(len(river.hexes) - 1):
            a, b = river.hexes[i], river.hexes[i + 1]
            assert not (on_border(a) and on_border(b)), (
                f"River has consecutive border hexes at positions {i} and {i + 1}: {a} -> {b}"
            )


def test_reproducibility():
    s1 = _build_pipeline(seed=7).run()
    s2 = _build_pipeline(seed=7).run()
    for coord in s1.hexes:
        assert s1.hexes[coord].river_flow == s2.hexes[coord].river_flow, (
            f"river_flow differs at {coord} between identical seeds"
        )
        assert s1.hexes[coord].tags == s2.hexes[coord].tags, (
            f"hex tags differ at {coord} between identical seeds"
        )
    assert len(s1.rivers) == len(s2.rivers), "river count differs between identical seeds"
    for i, (r1, r2) in enumerate(zip(s1.rivers, s2.rivers, strict=True)):
        assert r1.hexes == r2.hexes, f"river[{i}] path differs between identical seeds"
        assert r1.flow_volume == r2.flow_volume, (
            f"river[{i}] flow_volume differs between identical seeds"
        )


def test_lake_drainage_merges_without_rewiring_existing_river():
    cfg = WorldConfig(width=5, height=5)
    stage = HydrologyStage(cfg, np.random.default_rng(0))
    ws = WorldState.empty(seed=3, width=5, height=5)

    lake = (2, 2)
    spillway = (2, 1)
    merge = (2, 0)
    downstream = (3, 0)
    # A river running into the lake, so the basin takes in more than it evaporates and
    # has to overflow.  Without it this one-hex lake collects only the rain that falls on
    # it, which in a temperate climate is exactly what evaporates off it again, and the
    # water balance correctly declines to give a closed basin an outflow — leaving
    # nothing for this test to look at.
    feeder = (2, 3)

    for hex_item in ws.hexes.values():
        hex_item.terrain_class = TerrainClass.LAND
        hex_item.elevation = 10.0
        hex_item.river_flow = 0.0
    ws.hexes[lake].terrain_class = TerrainClass.INLAND_WATER
    ws.hexes[lake].elevation = 0.0
    ws.hexes[spillway].elevation = 1.0

    river_set = {merge, downstream}
    flow_dir = {merge: downstream, downstream: None, spillway: None, feeder: lake}
    land = set(ws.hexes) - {lake}
    ocean: set[tuple[int, int]] = set()
    lakes = {lake}
    acc = {spillway: 1.0, merge: 5.0, downstream: 8.0, feeder: 20.0}
    filled = {coord: hex_item.elevation for coord, hex_item in ws.hexes.items()}
    filled[spillway] = 1.0

    stage._guided_path_to_ocean = lambda *args, **kwargs: [merge]
    stage._forced_exit_to_border = lambda *args, **kwargs: [merge]

    stage._ensure_lake_drainage(
        river_set=river_set,
        flow_dir=flow_dir,
        hexes=ws.hexes,
        land=land,
        ocean=ocean,
        lakes=lakes,
        acc=acc,
        filled=filled,
        on_border=ws.on_border,
    )

    assert flow_dir[spillway] == merge
    assert flow_dir[merge] == downstream
    # The channel is joined, not seized: its own course is untouched.  Its *flow* is not,
    # and must not be — a stream below a junction carries what both sides bring it.  The
    # basin takes in 21 (a 20-unit river plus the rain on its one hex) and evaporates 1,
    # so 20 arrive here where the channel carried 5.  This used to assert 5.0, holding the
    # junction to the smaller of the two and pouring the lake's throughput away at the
    # confluence.
    assert acc[merge] == 20.0


def test_no_shared_hexes_between_rivers(hydro_state):
    # Each land hex must appear in at most one River segment, EXCEPT confluence hexes.
    # A confluence hex is the last hex of a tributary and simultaneously an interior
    # hex of the higher-flow trunk — it is shared by design so the two polylines
    # visually connect.  Any shared hex that is NOT a tributary endpoint represents
    # genuine trunk duplication and is a bug.
    from collections import defaultdict

    hex_to_rivers: dict[tuple[int, int], list[int]] = defaultdict(list)
    land_terrain = {
        coord
        for coord, hx in hydro_state.hexes.items()
        if hx.terrain_class not in (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER)
    }
    for i, river in enumerate(hydro_state.rivers):
        for coord in river.hexes:
            if coord in land_terrain:
                hex_to_rivers[coord].append(i)

    tributary_endpoints = {
        river.hexes[-1] for river in hydro_state.rivers if river.hexes[-1] in land_terrain
    }
    shared = {k: v for k, v in hex_to_rivers.items() if len(v) > 1 and k not in tributary_endpoints}
    assert not shared, (
        f"{len(shared)} land hexes appear in multiple rivers outside confluence endpoints; "
        f"first offender: {next(iter(shared))} in rivers {next(iter(shared.values()))}"
    )


def _confluence_fixture():
    """A trunk and a tributary that merge at (2, 0), with hand-set accumulation.

    Accumulation at the confluence (10.0) is deliberately far above the tributary's
    own pre-merge accumulation (3.0) so the two can be told apart in flow_volume.
    """
    trunk = River(hexes=[(0, 0), (1, 0), (2, 0), (3, 0)], flow_volume=1.0)
    tributary = River(hexes=[(0, 2), (1, 1), (2, 0), (3, 0)], flow_volume=0.3)
    acc = {
        (0, 0): 1.0,
        (1, 0): 2.0,
        (2, 0): 10.0,
        (3, 0): 12.0,
        (0, 2): 1.0,
        (1, 1): 3.0,
    }
    return [trunk, tributary], set(acc), acc, 12.0


def test_split_at_confluences_tributary_ends_on_confluence_hex():
    """A trimmed tributary's last hex is the trunk's confluence hex, and nothing beyond."""
    rivers, land, acc, max_acc = _confluence_fixture()
    trunk, tributary = _split_at_confluences(rivers, land, acc, max_acc)

    # Original list order is preserved, and the higher-flow trunk keeps its full path.
    assert trunk.hexes == [(0, 0), (1, 0), (2, 0), (3, 0)]

    # The tributary now reaches the confluence so the two polylines visually connect.
    assert tributary.hexes[-1] == (2, 0)
    assert tributary.hexes == [(0, 2), (1, 1), (2, 0)]

    # ...but it must not duplicate any trunk hex downstream of the confluence.
    downstream = set(trunk.hexes[trunk.hexes.index((2, 0)) + 1 :])
    assert not downstream & set(tributary.hexes)


def test_split_at_confluences_tributary_keeps_pre_merge_flow_volume():
    """The shared confluence hex carries combined discharge; the tributary must not."""
    rivers, land, acc, max_acc = _confluence_fixture()
    _, tributary = _split_at_confluences(rivers, land, acc, max_acc)

    # acc[(1, 1)] is the last hex the tributary owns exclusively.
    assert tributary.flow_volume == pytest.approx(3.0 / max_acc)
    # The confluence's own accumulation already includes the trunk — using it here would
    # render the whole tributary at post-merge width.
    assert tributary.flow_volume != pytest.approx(acc[(2, 0)] / max_acc)


# ---------------------------------------------------------------------------
# Rivers entering from off the map
# ---------------------------------------------------------------------------
#
# An inlet must be a *land* hex on the border, so these worlds drop an edge from
# `continent_falloff_edges`.  With the default sea ring the whole border is ocean, there
# is no land for a river to enter through, and the feature correctly does nothing — which
# `test_inflow_needs_land_at_the_border` pins down.

_INFLOW_KW = dict(
    width=48,
    height=48,
    erosion_iterations=800,
    continent_falloff_edges=("south", "east", "west"),
)


def _inflow_world(**overrides):
    return build_world(seed=11, until="HydrologyStage", **{**_INFLOW_KW, **overrides})


def _sources(state):
    return [c for c, h in state.hexes.items() if "river_source_offmap" in h.tags]


@pytest.fixture(scope="module")
def inflow_state():
    return _inflow_world()


def test_inflow_river_enters_from_the_border(inflow_state):
    # The direct regression test for the trace loop: _build_rivers breaks on the border
    # at the top of its loop, so without the inlet exemption an inflow river would be a
    # single hex.
    sources = _sources(inflow_state)
    assert sources, "expected at least one off-map river source"

    entering = [r for r in inflow_state.rivers if r.hexes[0] in sources]
    assert entering, "no river starts at an off-map source"
    for river in entering:
        assert len(river.hexes) > 1, f"off-map river at {river.hexes[0]} is a one-hex stub"
        assert inflow_state.on_border(river.hexes[0])
        assert not inflow_state.on_border(river.hexes[1]), (
            "an inflow river's second hex is on the border: it is creeping along the edge "
            "rather than heading inland"
        )


def test_inflow_sources_are_never_water(inflow_state):
    # A river may not rise out of the sea or a lake.  Guarded in three places: candidates
    # are drawn from land, an inlet's downstream hex must be land too, and the source tag
    # is dropped if lake drainage later submerges the hex.
    water = (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER)
    for coord in _sources(inflow_state):
        assert inflow_state.hexes[coord].terrain_class not in water

    for river in inflow_state.rivers:
        source = river.hexes[0]
        assert inflow_state.hexes[source].terrain_class not in water, (
            f"river starts in water at {source}"
        )


def test_inflow_respects_min_separation(inflow_state):
    sources = _sources(inflow_state)
    separation = _INFLOW_KW.get(
        "river_inflow_min_separation", WorldConfig().river_inflow_min_separation
    )
    for i, a in enumerate(sources):
        for b in sources[i + 1 :]:
            assert distance(a, b) >= separation, f"inlets {a} and {b} share a valley"


def test_inflow_count_is_respected(inflow_state):
    assert len(_sources(inflow_state)) <= WorldConfig().river_inflow_count


def test_inflow_arrives_already_large(inflow_state):
    # The point of the feature: an off-map river is wide where it crosses the border,
    # not a trickle that grows. A spring inside the map starts with one hex of rain, so
    # any inlet must carry far more than the largest ordinary headwater does.
    sources = set(_sources(inflow_state))
    assert sources

    # Springs only.  A lake's outflow has no upstream *river* hex either, so it is tagged
    # a headwater too, and since it carries the whole basin's discharge it is often the
    # largest one on the map — which says nothing about whether an off-map river arrives
    # large, the thing being measured here.
    local_headwaters = [
        r.hexes[0]
        for r in inflow_state.rivers
        if r.hexes[0] not in sources
        and "headwater" in inflow_state.hexes[r.hexes[0]].tags
        and not any(
            inflow_state.hexes[n].terrain_class == TerrainClass.INLAND_WATER
            for n in neighbors(r.hexes[0])
            if n in inflow_state.hexes
        )
    ]
    assert local_headwaters

    weakest_inlet = min(inflow_state.hexes[c].river_flow for c in sources)
    strongest_local_source = max(inflow_state.hexes[c].river_flow for c in local_headwaters)
    assert weakest_inlet > strongest_local_source, (
        "an off-map river should cross the border already carrying its catchment"
    )


def test_inflow_disabled_produces_no_off_map_sources():
    state = _inflow_world(river_inflow_count=0)
    assert _sources(state) == []


def test_inflow_zero_matches_the_pre_feature_world():
    # river_inflow_count = 0 must leave hydrology exactly as it was before the feature.
    a = _inflow_world(river_inflow_count=0)
    b = _inflow_world(river_inflow_count=0, river_inflow_volume=0.9)
    assert [r.hexes for r in a.rivers] == [r.hexes for r in b.rivers]


def test_inflow_is_reproducible():
    a = _inflow_world()
    b = build_world(seed=11, until="HydrologyStage", **_INFLOW_KW)
    assert sorted(_sources(a)) == sorted(_sources(b))


def test_inflow_edges_select_the_side_water_arrives_from():
    # The north edge is the only one without the sea ring, so it is the only one that can
    # carry an inlet; asking for the south instead must yield none.
    north = _inflow_world(river_inflow_edges=("north",))
    assert _sources(north), "expected inlets on the north edge"
    for coord in _sources(north):
        _col, row = north.grid_index(coord)
        assert row == 0, f"inlet at {coord} is not on the north edge"

    south = _inflow_world(river_inflow_edges=("south",))
    assert _sources(south) == [], "the south edge is all sea and cannot carry an inlet"


def test_inflow_empty_edge_list_is_not_an_error():
    assert _sources(_inflow_world(river_inflow_edges=())) == []


def test_inflow_needs_land_at_the_border():
    # The default map rings itself with sea, so no river can enter it by land.  Enabled
    # but inert, not an error.
    state = build_world(
        seed=11, until="HydrologyStage", width=48, height=48, erosion_iterations=800
    )
    assert _sources(state) == []


def test_downstream_lengths_counts_land_hexes_to_the_outlet():
    # A straight chain of four land hexes draining into a fifth that is not land (the
    # sea, or off the map): each hex counts itself plus everything below it, and the
    # water hex is not counted.
    chain = [(0, 0), (1, 0), (2, 0), (3, 0)]
    outlet = (4, 0)
    flow_dir = {c: n for c, n in zip(chain, chain[1:] + [outlet], strict=True)}
    lengths = HydrologyStage._downstream_lengths(flow_dir, set(chain))
    assert [lengths[c] for c in chain] == [4, 3, 2, 1]


def test_downstream_lengths_shares_a_common_trunk():
    # Two headwaters joining a trunk. The memo has to give the trunk one value, not
    # recompute it per branch, and each branch counts itself on top of it.
    trunk = [(2, 0), (3, 0)]
    flow_dir = {
        (0, 0): (2, 0),
        (1, 0): (2, 0),
        (2, 0): (3, 0),
        (3, 0): (4, 0),
    }
    land = {(0, 0), (1, 0), *trunk}
    lengths = HydrologyStage._downstream_lengths(flow_dir, land)
    assert lengths[(3, 0)] == 1
    assert lengths[(2, 0)] == 2
    assert lengths[(0, 0)] == lengths[(1, 0)] == 3


def test_downstream_lengths_terminates_on_a_cycle():
    # flow_dir is cycle-free by construction, but the walk must not hang if that ever
    # stops being true.
    flow_dir = {(0, 0): (1, 0), (1, 0): (2, 0), (2, 0): (0, 0)}
    lengths = HydrologyStage._downstream_lengths(flow_dir, {(0, 0), (1, 0), (2, 0)})
    assert set(lengths) == {(0, 0), (1, 0), (2, 0)}
    assert all(v > 0 for v in lengths.values())


def test_inflow_min_length_rejects_short_courses():
    # A floor longer than the map can possibly offer leaves nothing eligible. Fewer
    # rivers than river_inflow_count asks for is the intended outcome — better than
    # importing one that leaves again a few hexes later.
    assert _sources(_inflow_world(river_inflow_min_length=0.95)) == []


def test_inflow_without_a_floor_still_places_rivers():
    # The floor is what costs inlets, so with it off the count should be met — this is
    # what pins the previous test on the floor rather than on the world having no
    # candidates at all.
    state = _inflow_world(river_inflow_min_length=0.0, river_inflow_length_bias=0.0)
    assert _sources(state)


def test_inflow_prefers_the_longer_course():
    # The whole point of the length weighting. Measured on the course the water actually
    # takes, walking the drawn rivers — a single River may be trimmed at a confluence
    # where a larger trunk claims the trunk hexes, so its polyline is not the full course.
    def course_length(state, source):
        downstream = {}
        for river in state.rivers:
            for a, b in zip(river.hexes, river.hexes[1:], strict=False):
                downstream.setdefault(a, b)
        seen, current, hops = set(), source, 0
        while current is not None and current not in seen:
            seen.add(current)
            hops += 1
            current = downstream.get(current)
        return hops

    def median_course(**overrides):
        lengths = []
        for seed in (3, 4, 5, 6):
            state = build_world(seed=seed, until="HydrologyStage", **{**_INFLOW_KW, **overrides})
            lengths += [course_length(state, c) for c in _sources(state)]
        return statistics.median(lengths) if lengths else 0

    biased = median_course()
    unbiased = median_course(river_inflow_length_bias=0.0, river_inflow_min_length=0.0)
    assert biased > unbiased, (
        f"length-biased inlets are no longer than unbiased ones ({biased} vs {unbiased})"
    )


def test_a_coastal_map_drains_to_the_sea():
    # Regression: a basin whose outflow route joined a channel that flowed back into the
    # same basin was left draining into itself, and the endorheic pass then reported it
    # closed.  On this config that swallowed almost every lake on the map — 95% of lake
    # hexes came out endorheic, including one inland sea of 5121 hexes on a 256x256 run.
    #
    # The map has an ocean along its south edge and slopes down to it, so the water has
    # somewhere to go and most of it should get there.  Some genuinely closed basins are
    # expected and wanted; a map that is nearly all closed basin is the bug.
    state = build_world(
        seed=1,
        until="HydrologyStage",
        width=80,
        height=80,
        erosion_iterations=600,
        grid_layout="offset",
        continent_falloff_edges=["south"],
        continent_shelf_variance=0.8,
        elevation_gradient=[0.0, -0.5],
    )
    lake = [h for h in state.hexes.values() if h.terrain_class == TerrainClass.INLAND_WATER]
    endorheic = [h for h in lake if "endorheic" in h.tags]
    assert lake, "expected this config to produce lakes"
    assert len(endorheic) / len(lake) < 0.5, (
        f"{len(endorheic)} of {len(lake)} lake hexes are endorheic on a map with a coast; "
        "basins are draining into themselves rather than to the sea"
    )


def test_a_lake_outflow_is_seeded_with_the_whole_basin_inflow():
    """The outflow carries the basin's throughput, not the spillway hex's own drainage.

    Asserted on the accumulation the routing seeds, because that is where the property
    lives.  Measuring it off the finished River objects does not work: confluence
    splitting cuts an outflow into segments at every junction, so the polyline starting
    at the spillway is not necessarily the one carrying the basin's water, and a rewrite
    of this test that tried came out reading a correct outlet as a fifteenth of its feed.
    """
    lake, spillway, merge, downstream = (2, 2), (2, 1), (2, 0), (3, 0)
    feeders = {(2, 3): 400.0, (1, 2): 60.0, (3, 2): 15.0}

    cfg = WorldConfig(width=5, height=5, endorheic_evaporation_scale=0.0)
    stage = HydrologyStage(cfg, np.random.default_rng(0))
    ws = WorldState.empty(seed=3, width=5, height=5)
    for hex_item in ws.hexes.values():
        hex_item.terrain_class = TerrainClass.LAND
        hex_item.elevation = 10.0
    ws.hexes[lake].terrain_class = TerrainClass.INLAND_WATER
    ws.hexes[lake].elevation = 0.0
    ws.hexes[spillway].elevation = 1.0

    filled = {coord: h.elevation for coord, h in ws.hexes.items()}
    filled[spillway] = 1.0
    acc = {spillway: 1.0, merge: 5.0, downstream: 8.0, **feeders}
    flow_dir = {merge: downstream, downstream: None, spillway: None}
    flow_dir.update(dict.fromkeys(feeders, lake))

    stage._guided_path_to_ocean = lambda *a, **k: [merge]
    stage._forced_exit_to_border = lambda *a, **k: [merge]
    stage._ensure_lake_drainage(
        river_set={merge, downstream},
        flow_dir=flow_dir,
        hexes=ws.hexes,
        land=set(ws.hexes) - {lake},
        ocean=set(),
        lakes={lake},
        acc=acc,
        filled=filled,
        on_border=ws.on_border,
    )

    # 475 units of river, plus the one hex of rain falling on the lake itself.
    assert acc[spillway] == pytest.approx(sum(feeders.values()) + 1.0)
    # And the old bug, stated as what it was: the spillway's own single hex of rain.
    assert acc[spillway] > 100.0


def _balance_world(**cfg_kw):
    """A one-hex lake with one river running into it, for water-balance tests.

    Deliberately synthetic: the balance is a ratio between what arrives and what
    evaporates, and a hand-built basin is the only way to put both sides of it where the
    test can see them.
    """
    cfg = WorldConfig(width=5, height=5, **cfg_kw)
    stage = HydrologyStage(cfg, np.random.default_rng(0))
    ws = WorldState.empty(seed=3, width=5, height=5)

    lake, spillway, merge, downstream, feeder = (2, 2), (2, 1), (2, 0), (3, 0), (2, 3)
    for hex_item in ws.hexes.values():
        hex_item.terrain_class = TerrainClass.LAND
        hex_item.elevation = 10.0
    ws.hexes[lake].terrain_class = TerrainClass.INLAND_WATER
    ws.hexes[lake].elevation = 0.0
    ws.hexes[spillway].elevation = 1.0

    filled = {coord: h.elevation for coord, h in ws.hexes.items()}
    filled[spillway] = 1.0
    stage._guided_path_to_ocean = lambda *a, **k: [merge]
    stage._forced_exit_to_border = lambda *a, **k: [merge]

    _rivers, outlet_of = stage._ensure_lake_drainage(
        river_set={merge, downstream},
        flow_dir={merge: downstream, downstream: None, spillway: None, feeder: lake},
        hexes=ws.hexes,
        land=set(ws.hexes) - {lake},
        ocean=set(),
        lakes={lake},
        acc={spillway: 1.0, merge: 5.0, downstream: 8.0, feeder: 20.0},
        filled=filled,
        on_border=ws.on_border,
    )
    return outlet_of[lake]


def test_a_basin_taking_in_more_than_it_evaporates_is_given_an_outlet():
    # 21 units arrive (a 20-unit river plus the rain on one hex of water); a temperate
    # hex of lake evaporates 1.  It has to overflow.
    assert _balance_world(regional_climate="temperate") is not None


def test_a_basin_that_evaporates_what_reaches_it_is_closed():
    # Same basin, same inflow, evaporation cranked past it: the water now leaves as
    # vapour and no channel is cut.  This is the Caspian, and it is the case the old
    # topological test could not express — it closed a basin when path-finding failed,
    # which is a fact about the terrain's shape, not about its water.
    assert _balance_world(endorheic_evaporation_scale=100.0) is None


def test_climate_alone_can_close_a_basin():
    # The evaporation rate is the region's climate, so the same terrain and the same
    # rivers give a draining lake in the cold and a closed one in the desert.
    inflow_scale = dict(endorheic_evaporation_scale=8.0)
    assert _balance_world(regional_climate="boreal", **inflow_scale) is not None
    assert _balance_world(regional_climate="arid", **inflow_scale) is None


def test_rain_shadow_does_not_change_how_much_rain_falls():
    # Only where it falls. The field averages one unit per land hex at any strength, so
    # river_flow_threshold and river_inflow_volume — both fractions — keep their meaning.
    from worldgen.stages.precipitation import rain_per_hex

    state = build_world(seed=11, until="WaterBodiesStage", width=48, height=48)
    land = {
        c
        for c, h in state.hexes.items()
        if h.terrain_class not in (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER)
    }
    for strength in (0.0, 0.25, 0.5, 1.0):
        cfg = WorldConfig(width=48, height=48, rain_shadow_strength=strength)
        rain = rain_per_hex(state, cfg, land)
        mean = sum(rain[c] for c in land) / len(land)
        assert mean == pytest.approx(1.0, abs=1e-9), f"strength {strength} moved the mean"


def test_rain_shadow_off_gives_every_hex_the_same_rain():
    from worldgen.stages.precipitation import rain_per_hex

    state = build_world(seed=11, until="WaterBodiesStage", width=48, height=48)
    land = {
        c
        for c, h in state.hexes.items()
        if h.terrain_class not in (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER)
    }
    rain = rain_per_hex(state, WorldConfig(rain_shadow_strength=0.0), land)
    assert set(rain.values()) == {1.0}


def test_rain_shadow_makes_rain_uneven():
    # The point of it: a windward slope collects more than a hex in the lee.
    from worldgen.stages.precipitation import rain_per_hex

    state = build_world(seed=11, until="WaterBodiesStage", width=48, height=48)
    land = {
        c
        for c, h in state.hexes.items()
        if h.terrain_class not in (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER)
    }
    flat = rain_per_hex(state, WorldConfig(rain_shadow_strength=0.0), land)
    shaped = rain_per_hex(state, WorldConfig(rain_shadow_strength=1.0), land)
    assert statistics.pstdev(shaped[c] for c in land) > statistics.pstdev(flat[c] for c in land)
    assert min(shaped[c] for c in land) < 1.0 < max(shaped[c] for c in land)


def test_a_drier_catchment_raises_a_smaller_river():
    """The point of the whole thing: less rain upstream means less water downstream.

    Tested on the accumulation directly rather than on a generated map's statistics.  The
    river *set* is the top fraction of hexes by flow, so redistributing rain changes which
    hexes are rivers as well as how much they carry, and a summary statistic over that set
    moves for reasons that have nothing to do with the property being claimed here.
    """
    chain = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
    flow_dir = {c: n for c, n in zip(chain, chain[1:], strict=False)}
    flow_dir[chain[-1]] = None
    land = set(chain)
    stage = HydrologyStage(WorldConfig(), np.random.default_rng(0))

    wet = stage._flow_accumulation(flow_dir, land, None, dict.fromkeys(chain, 1.0))
    # Same map, same catchment, but the headwaters sit in a rain shadow.
    shadowed = stage._flow_accumulation(
        flow_dir, land, None, {**dict.fromkeys(chain, 1.0), chain[0]: 0.1, chain[1]: 0.2}
    )

    mouth = chain[-1]
    assert wet[mouth] == pytest.approx(5.0)
    assert shadowed[mouth] == pytest.approx(3.3)
    assert shadowed[mouth] < wet[mouth]
