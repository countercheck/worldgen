import statistics

import numpy as np
import pytest

from worldgen.core.config import WorldConfig
from worldgen.core.hex import TerrainClass
from worldgen.core.hex_grid import corner_hexes, corner_neighbors, distance, side_hexes
from worldgen.core.pipeline import GeneratorPipeline
from worldgen.core.world_state import WorldState
from worldgen.stages.elevation import ElevationStage
from worldgen.stages.erosion import ErosionStage
from worldgen.stages.hydrology import HydrologyStage
from worldgen.stages.terrain_class import TerrainClassificationStage
from worldgen.stages.water_bodies import WaterBodiesStage

from .worlds import build_world

_WATER = (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER)


def _build_pipeline(seed: int = 42, width: int = 32, height: int = 32):
    cfg = WorldConfig(width=width, height=height)
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
    # A course runs along hexsides: each corner is one side from the next.
    assert hydro_state.rivers
    for river in hydro_state.rivers:
        assert len(river.corners) >= 2, "a course has fewer than two corners"
        for a, b in zip(river.corners, river.corners[1:], strict=False):
            assert b in corner_neighbors(a), f"corners {a} -> {b} are not one side apart"


def test_rivers_reach_ocean(hydro_state):
    # Every course ends where its water leaves the network — at the sea or a lake, or at
    # the map edge — or on a corner of a larger river it joins.
    hexes = hydro_state.hexes
    interior = {c for river in hydro_state.rivers for c in river.corners[:-1]}
    for river in hydro_state.rivers:
        end = river.corners[-1]
        around = corner_hexes(end)
        at_water = any(h in hexes and hexes[h].terrain_class in _WATER for h in around)
        at_edge = any(h not in hexes for h in around)
        joins = end in interior
        assert at_water or at_edge or joins, f"course ends at {end}, nowhere water can go"


def test_flow_accumulates_downstream(hydro_state):
    # Each side downstream carries at least what the side above it did.
    for river in hydro_state.rivers:
        catchments = [hydro_state.river_sides[s].catchment_km2 for s in river.sides()]
        for a, b in zip(catchments, catchments[1:], strict=False):
            assert b >= a - 1e-9, f"catchment falls downstream: {a:.2f} -> {b:.2f}"


def test_tags_assigned(hydro_state):
    corner_tags = {t for tags in hydro_state.river_corners.values() for t in tags}
    assert {"river_source", "river_mouth", "river_end"} <= corner_tags
    # Sources, mouths and confluences are points on the river, not places on the map.
    hex_tags = {t for h in hydro_state.hexes.values() for t in h.tags}
    assert not hex_tags & {"river", "headwater", "river_mouth", "river_source", "confluence"}


def test_both_banks_of_a_river_record_its_flow(hydro_state):
    for side, rs in hydro_state.river_sides.items():
        for h in side_hexes(side):
            hx = hydro_state.hexes[h]
            assert hx.river_flow > 0, f"bank {h} of river side {side} records no flow"
            assert hx.catchment_km2 >= rs.catchment_km2


def test_every_river_side_has_land_on_both_hands(hydro_state):
    # A side with water or the map edge beside it is a shoreline or the frame; a river
    # drawn there would run along the coast or the edge of the map.
    for side in hydro_state.river_sides:
        for h in side_hexes(side):
            assert h in hydro_state.hexes, f"river side {side} runs along the map edge"
            assert hydro_state.hexes[h].terrain_class not in _WATER, (
                f"river side {side} runs along the shore"
            )


def test_flow_volume(hydro_state):
    rivers = hydro_state.rivers
    assert all(0.0 < r.flow_volume <= 1.0 for r in rivers), "flow_volume out of (0, 1]"
    # Measured at the mouth, so at least what the course carries at its head.
    for river in rivers:
        head = hydro_state.river_sides[river.sides()[0]].flow
        assert river.flow_volume >= head - 1e-9


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
    assert [r.corners for r in s1.rivers] == [r.corners for r in s2.rivers]
    assert [r.flow_volume for r in s1.rivers] == [r.flow_volume for r in s2.rivers]
    assert s1.river_sides == s2.river_sides
    assert s1.river_corners == s2.river_corners


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
    # basin takes in 21 (a 20-unit river plus the rain on its one hex) and evaporates the
    # potential evapotranspiration off that one hex of surface, which in a temperate
    # region is 350 mm against 800 mm of mean rainfall — 0.4375 of a hex's worth.  So
    # 20.5625 arrive here where the channel carried 5.  This used to assert 5.0, holding
    # the junction to the smaller of the two and pouring the lake's throughput away at
    # the confluence.
    expected = 21.0 - cfg.pet_mm() / cfg.mean_precip_mm
    assert acc[merge] == pytest.approx(expected)


def test_no_side_belongs_to_two_rivers(hydro_state):
    # Each stretch of river is drawn once: a tributary ends on the trunk's corner and does
    # not run on down it.  So no side is in two courses, and a corner is shared only as the
    # last corner of a tributary.
    from collections import Counter

    sides = Counter(s for river in hydro_state.rivers for s in river.sides())
    assert all(n == 1 for n in sides.values())
    ends = {river.corners[-1] for river in hydro_state.rivers}
    corners = Counter(c for river in hydro_state.rivers for c in river.corners)
    shared = {c for c, n in corners.items() if n > 1}
    assert shared <= ends


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
    continent_falloff_edges=("south", "east", "west"),
)


def _inflow_world(**overrides):
    return build_world(seed=11, until="HydrologyStage", **{**_INFLOW_KW, **overrides})


def _sources(state):
    """The corners rivers enter the map at."""
    return sorted(c for c, tags in state.river_corners.items() if "river_source_offmap" in tags)


@pytest.fixture(scope="module")
def inflow_state():
    return _inflow_world()


def test_inflow_river_enters_from_the_border(inflow_state):
    # An inflow course starts one side in from the map edge and heads inland.
    sources = _sources(inflow_state)
    assert sources, "expected at least one off-map river source"
    hexes = inflow_state.hexes

    def at_edge(corner):
        return any(h not in hexes for h in corner_hexes(corner))

    entering = [r for r in inflow_state.rivers if r.corners[0] in sources]
    assert len(entering) == len(sources)
    for river in entering:
        first = river.corners[0]
        assert len(river.corners) > 2, f"off-map river at {first} is a stub"
        assert any(at_edge(n) for n in corner_neighbors(first)), "inlet is not by the edge"
        assert not at_edge(river.corners[1]), "inflow runs along the edge, not inland"


def test_inflow_sources_are_never_water(inflow_state):
    # A river may not rise out of the sea or a lake: every course starts on a side with
    # land on both hands.
    for river in inflow_state.rivers:
        for h in side_hexes(river.sides()[0]):
            assert inflow_state.hexes[h].terrain_class not in _WATER


def test_inflow_respects_min_separation(inflow_state):
    # Inlets are chosen on hexes `river_inflow_min_separation` apart, and each enters at a
    # corner of its hex — so two inlet corners' hexes may sit up to two closer than that.
    sources = _sources(inflow_state)
    separation = _INFLOW_KW.get(
        "river_inflow_min_separation", WorldConfig().river_inflow_min_separation
    )
    for i, a in enumerate(sources):
        for b in sources[i + 1 :]:
            gap = min(distance(x, y) for x in corner_hexes(a) for y in corner_hexes(b))
            assert gap >= separation - 2, f"inlets {a} and {b} share a valley"


def test_inflow_count_is_respected(inflow_state):
    assert len(_sources(inflow_state)) <= WorldConfig().river_inflow_count


def test_inflow_arrives_already_large(inflow_state):
    # The point of the feature: an off-map river is wide where it crosses the border, not
    # a trickle that grows.  So its first side carries more than any spring's does.
    sources = set(_sources(inflow_state))
    assert sources
    springs = {c for c, tags in inflow_state.river_corners.items() if "river_source" in tags}
    assert springs

    def head(river):
        return inflow_state.river_sides[river.sides()[0]].catchment_km2

    weakest_inlet = min(head(r) for r in inflow_state.rivers if r.corners[0] in sources)
    strongest_spring = max(head(r) for r in inflow_state.rivers if r.corners[0] in springs)
    assert weakest_inlet > strongest_spring, (
        "an off-map river should cross the border already carrying its catchment"
    )


def test_inflow_disabled_produces_no_off_map_sources():
    state = _inflow_world(river_inflow_count=0)
    assert _sources(state) == []


def test_inflow_zero_matches_the_pre_feature_world():
    # river_inflow_count = 0 must leave hydrology exactly as it was before the feature.
    a = _inflow_world(river_inflow_count=0)
    b = _inflow_world(river_inflow_count=0, river_inflow_volume=0.9)
    assert [r.corners for r in a.rivers] == [r.corners for r in b.rivers]


def test_inflow_is_reproducible():
    a = _inflow_world()
    b = build_world(seed=11, until="HydrologyStage", **_INFLOW_KW)
    assert sorted(_sources(a)) == sorted(_sources(b))


def test_inflow_edges_select_the_side_water_arrives_from():
    # The north edge is the only one without the sea ring, so it is the only one that can
    # carry an inlet; asking for the south instead must yield none.
    north = _inflow_world(river_inflow_edges=("north",))
    assert _sources(north), "expected inlets on the north edge"
    for corner in _sources(north):
        rows = [north.grid_index(h)[1] for h in corner_hexes(corner) if h in north.hexes]
        assert 0 in rows, f"inlet at {corner} is not on the north edge"

    south = _inflow_world(river_inflow_edges=("south",))
    assert _sources(south) == [], "the south edge is all sea and cannot carry an inlet"


def test_inflow_empty_edge_list_is_not_an_error():
    assert _sources(_inflow_world(river_inflow_edges=())) == []


def test_inflow_needs_land_at_the_border():
    # The default map rings itself with sea, so no river can enter it by land.  Enabled
    # but inert, not an error.
    state = build_world(seed=11, until="HydrologyStage", width=48, height=48)
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
            for a, b in zip(river.corners, river.corners[1:], strict=False):
                downstream.setdefault(a, b)
        seen, current, hops = set(), source, 0
        while current is not None and current not in seen:
            seen.add(current)
            hops += 1
            current = downstream.get(current)
        return hops

    def median_course(**overrides):
        lengths = []
        # Twelve seeds, not four.  Four gave about eight measurements a side, few enough
        # that the two medians could land on the same value and say nothing: they did
        # exactly that when `channel_min_discharge` dropped to 6,000 and the river set
        # changed under them.  Twelve gives about twenty-two a side and separates cleanly
        # — measured 8.5 biased against 7.0 unbiased — and the worlds stop after hydrology,
        # so the extra eight cost about five seconds.
        for seed in range(3, 15):
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
        erosion_droplets_per_hex=0.8,
        grid_layout="offset",
        continent_falloff_edges=["south"],
        continent_shelf_variance=0.8,
        elevation_gradient_m=[0.0, -850.0],
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


# ---------------------------------------------------------------------------
# One inlet, one course (tech-debt #153)
# ---------------------------------------------------------------------------
#
# Erosion chooses the inlets and hands each to hydrology with the course its river runs;
# hydrology adopts both.  They used to be chosen twice, by different rules, and agreed on
# none of ten: the imported catchment widened valleys the imported river never ran down.

_SHIPPED_KW = dict(width=48, height=48, model="organic", continent_falloff_edges=("south",))


def _with_handoff(monkeypatch, seed):
    """A world to hydrology, and the courses erosion handed it on the way."""
    from .worlds import build_pipeline

    seen = {}
    run = HydrologyStage.run

    def spy(self, state):
        seen["courses"] = list(state.metadata.get("inflow_courses", []))
        return run(self, state)

    monkeypatch.setattr(HydrologyStage, "run", spy)
    state = build_pipeline(seed=seed, until="HydrologyStage", **_SHIPPED_KW).run()
    return state, seen["courses"]


@pytest.mark.parametrize("seed", [11, 42])
def test_hydrology_imports_rivers_at_the_inlets_erosion_carved_for(monkeypatch, seed):
    state, courses = _with_handoff(monkeypatch, seed)
    assert courses, "the shipped config's open north edge should admit an inlet"
    assert set(_sources(state)) == {course[0] for _, course in courses}


@pytest.mark.parametrize("seed", [11, 42])
def test_an_imported_river_runs_the_course_erosion_carved(monkeypatch, seed):
    """Every side of the course, until it meets standing water, carries the import."""
    state, courses = _with_handoff(monkeypatch, seed)
    cfg = WorldConfig(**state.metadata["config"])
    land = [h for h in state.hexes.values() if h.terrain_class not in _WATER]
    imported = cfg.river_inflow_volume * len(land)
    from worldgen.core.hex_grid import side_joining

    for _, course in courses:
        for a, b in zip(course, course[1:], strict=False):
            if any(
                state.hexes[h].terrain_class in _WATER for h in corner_hexes(a) if h in state.hexes
            ):
                break
            side = state.river_sides.get(side_joining(a, b))
            assert side is not None, f"the imported river leaves its course at {a}"
            assert side.catchment_km2 >= imported


def test_the_handoff_is_not_kept_in_the_world():
    state = build_world(seed=11, until="HydrologyStage", **_SHIPPED_KW)
    assert "inflow_courses" not in state.metadata


def test_a_map_that_admits_no_inlet_is_untouched_by_the_inlet_settings():
    """The sea-ring default has no land on its border: asking for inlets changes nothing.

    Nothing is chosen, so nothing is drawn and nothing handed on, and the world is the
    one it was before inlets were chosen once rather than twice.
    """
    asked = build_world(seed=42, width=48, height=48, model="organic")
    not_asked = build_world(seed=42, width=48, height=48, model="organic", river_inflow_count=0)

    def world(state):
        out = state.to_dict()
        out["metadata"] = {k: v for k, v in out["metadata"].items() if k != "config"}
        return out

    assert world(asked) == world(not_asked)
