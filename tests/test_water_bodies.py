from collections import deque

import pytest

from worldgen.core.config import WorldConfig
from worldgen.core.hex import TerrainClass
from worldgen.core.hex_grid import corner_hexes, neighbors
from worldgen.core.pipeline import GeneratorPipeline
from worldgen.stages.elevation import ElevationStage
from worldgen.stages.erosion import ErosionStage
from worldgen.stages.hydrology import HydrologyStage
from worldgen.stages.terrain_class import TerrainClassificationStage
from worldgen.stages.water_bodies import WaterBodiesStage


def _build_pipeline(seed: int = 42, width: int = 40, height: int = 40):
    cfg = WorldConfig(width=width, height=height)
    p = GeneratorPipeline(seed, cfg)
    p.add_stage(ElevationStage)
    p.add_stage(ErosionStage)
    p.add_stage(TerrainClassificationStage)
    p.add_stage(WaterBodiesStage)
    p.add_stage(HydrologyStage)
    return p


@pytest.fixture(scope="module")
def world():
    return _build_pipeline().run()


def _on_border(coord, w, h):
    q, r = coord
    return q == 0 or q == w - 1 or r == 0 or r == h - 1


def _bfs_component(seed, members):
    component = {seed}
    queue = deque([seed])
    while queue:
        coord = queue.popleft()
        for nbr in neighbors(coord):
            if nbr in members and nbr not in component:
                component.add(nbr)
                queue.append(nbr)
    return component


def _water_components(state, terrain_class):
    members = {c for c, h in state.hexes.items() if h.terrain_class == terrain_class}
    visited = set()
    components = []
    for seed in members:
        if seed in visited:
            continue
        comp = _bfs_component(seed, members)
        visited |= comp
        components.append(comp)
    return components


def test_ocean_bodies_touch_border(world):
    """Every connected OCEAN water body must include at least one map-edge hex."""
    w, h = world.width, world.height
    for comp in _water_components(world, TerrainClass.OPEN_WATER):
        assert any(_on_border(c, w, h) for c in comp), (
            f"OCEAN component of size {len(comp)} has no map-edge hex"
        )


def test_lake_bodies_no_border(world):
    """No LAKE hex may be on the map edge (lakes are entirely inland)."""
    w, h = world.width, world.height
    for coord, hx in world.hexes.items():
        if hx.terrain_class == TerrainClass.INLAND_WATER:
            assert not _on_border(coord, w, h), (
                f"LAKE hex {coord} is on the map edge — should be OCEAN"
            )


def test_all_water_classified(world):
    """Every hex below sea level is either OCEAN or LAKE.

    Sea level is zero: elevation is metres above it, so this is a statement about the
    world rather than a comparison against a per-map threshold.
    """
    sea = 0.0
    water_types = (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER)
    for coord, hx in world.hexes.items():
        if hx.elevation < sea:
            assert hx.terrain_class in water_types, (
                f"Hex {coord} with elevation {hx.elevation:.3f} < sea_level {sea} "
                f"has terrain_class {hx.terrain_class} (expected OCEAN or LAKE)"
            )


def _courses_downstream(world):
    """Corner -> the next corner downstream, over every drawn course."""
    down = {}
    for river in world.rivers:
        for a, b in zip(river.corners, river.corners[1:], strict=False):
            down.setdefault(a, b)
    return down


def _where_it_goes(world, start, lake_of, own):
    """Follow the courses from *start*: "ocean", "border", "closed" (a lake with no outlet),
    a lake index, or None."""
    down = _courses_downstream(world)
    seen, node = set(), start
    while node is not None and node not in seen:
        seen.add(node)
        for h in corner_hexes(node):
            if h not in world.hexes:
                return "border"
            hx = world.hexes[h]
            if hx.terrain_class == TerrainClass.OPEN_WATER:
                return "ocean"
            if h in lake_of and lake_of[h] != own and node != start:
                return lake_of[h]
            if hx.terrain_class == TerrainClass.INLAND_WATER and "endorheic" in hx.tags:
                return "closed"
        node = down.get(node)
    return None


def _outlets(world, comp):
    """Courses leaving a lake: those that start on its shore."""
    return [
        r.corners[0]
        for r in world.rivers
        if any(h in comp for h in corner_hexes(r.corners[0]))
        and "river_source" not in world.river_corners.get(r.corners[0], set())
    ]


def _open_lakes(world):
    lakes = _water_components(world, TerrainClass.INLAND_WATER)
    return [comp for comp in lakes if not any("endorheic" in world.hexes[c].tags for c in comp)]


def test_lake_has_outflow_river(world):
    """Every lake that is not a closed basin has a river leaving it, and that river reaches
    the sea, the map edge, or another lake — open, or closed like the Dead Sea at the end
    of the Jordan."""
    lakes = _open_lakes(world)
    if not lakes:
        pytest.skip("No draining lakes in this world — nothing to check")
    lake_of = {c: i for i, comp in enumerate(lakes) for c in comp}
    for i, comp in enumerate(lakes):
        outlets = _outlets(world, comp)
        assert outlets, f"LAKE (size {len(comp)}) has no river leaving it"
        assert any(_where_it_goes(world, o, lake_of, i) is not None for o in outlets), (
            f"LAKE (size {len(comp)})'s outflow goes nowhere"
        )


def test_lake_chain_terminates(world):
    """Following lake outflows from lake to lake must reach the sea, the map edge, or a
    closed lake that the water leaves only by evaporating."""
    lakes = _open_lakes(world)
    if not lakes:
        pytest.skip("No draining lakes in this world — nothing to check")
    lake_of = {c: i for i, comp in enumerate(lakes) for c in comp}
    for start in range(len(lakes)):
        visited, idx, end = {start}, start, None
        while end is None:
            goes = [_where_it_goes(world, o, lake_of, idx) for o in _outlets(world, lakes[idx])]
            if "ocean" in goes or "border" in goes or "closed" in goes:
                end = "out"
                break
            onward = [g for g in goes if isinstance(g, int) and g not in visited]
            if not onward:
                break
            idx = onward[0]
            visited.add(idx)
        assert end == "out", (
            f"LAKE component {start} (size {len(lakes[start])}) outflow chain does not "
            "reach the sea, the map edge or a closed lake"
        )


def test_water_body_reproducible():
    """Same seed produces identical OCEAN/LAKE classification."""
    s1 = _build_pipeline(seed=13).run()
    s2 = _build_pipeline(seed=13).run()
    for coord in s1.hexes:
        assert s1.hexes[coord].terrain_class == s2.hexes[coord].terrain_class, (
            f"terrain_class differs at {coord} between runs with same seed"
        )
