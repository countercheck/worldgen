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
        if h.terrain_class in (TerrainClass.OCEAN, TerrainClass.LAKE)
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
    water_classes = {TerrainClass.OCEAN, TerrainClass.LAKE}
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

    for hex_item in ws.hexes.values():
        hex_item.terrain_class = TerrainClass.FLAT
        hex_item.elevation = 10.0
        hex_item.river_flow = 0.0
    ws.hexes[lake].terrain_class = TerrainClass.LAKE
    ws.hexes[lake].elevation = 0.0
    ws.hexes[spillway].elevation = 1.0

    river_set = {merge, downstream}
    flow_dir = {merge: downstream, downstream: None, spillway: None}
    land = set(ws.hexes) - {lake}
    ocean: set[tuple[int, int]] = set()
    lakes = {lake}
    acc = {spillway: 1.0, merge: 5.0, downstream: 8.0}
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
    assert acc[merge] == 5.0


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
        if hx.terrain_class not in (TerrainClass.OCEAN, TerrainClass.LAKE)
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
    water = (TerrainClass.OCEAN, TerrainClass.LAKE)
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

    local_headwaters = [
        r.hexes[0]
        for r in inflow_state.rivers
        if r.hexes[0] not in sources and "headwater" in inflow_state.hexes[r.hexes[0]].tags
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
