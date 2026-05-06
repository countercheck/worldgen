from collections import deque

import pytest

from worldgen.core.config import WorldConfig
from worldgen.core.hex import TerrainClass
from worldgen.core.hex_grid import neighbors
from worldgen.core.pipeline import GeneratorPipeline
from worldgen.stages.elevation import ElevationStage
from worldgen.stages.erosion import ErosionStage
from worldgen.stages.hydrology import HydrologyStage
from worldgen.stages.terrain_class import TerrainClassificationStage
from worldgen.stages.water_bodies import WaterBodiesStage


def _build_pipeline(seed: int = 42, width: int = 40, height: int = 40):
    cfg = WorldConfig(width=width, height=height, erosion_iterations=500)
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
    for comp in _water_components(world, TerrainClass.OCEAN):
        assert any(_on_border(c, w, h) for c in comp), (
            f"OCEAN component of size {len(comp)} has no map-edge hex"
        )


def test_lake_bodies_no_border(world):
    """No LAKE hex may be on the map edge (lakes are entirely inland)."""
    w, h = world.width, world.height
    for coord, hx in world.hexes.items():
        if hx.terrain_class == TerrainClass.LAKE:
            assert not _on_border(coord, w, h), (
                f"LAKE hex {coord} is on the map edge — should be OCEAN"
            )


def test_all_water_classified(world):
    """Every hex with elevation below sea_level is either OCEAN or LAKE."""
    sea = world.metadata.get("config", {}).get("sea_level", 0.45)
    water_types = (TerrainClass.OCEAN, TerrainClass.LAKE)
    for coord, hx in world.hexes.items():
        if hx.elevation < sea:
            assert hx.terrain_class in water_types, (
                f"Hex {coord} with elevation {hx.elevation:.3f} < sea_level {sea} "
                f"has terrain_class {hx.terrain_class} (expected OCEAN or LAKE)"
            )


def test_lake_has_outflow_river(world):
    """Each LAKE must have an outflow: a border river hex whose downstream path
    (non-decreasing river_flow, not re-entering this lake) reaches ocean or border."""
    lake_comps = _water_components(world, TerrainClass.LAKE)
    if not lake_comps:
        pytest.skip("No lakes in this world — nothing to check")

    w, h = world.width, world.height
    land = {
        c
        for c, hx in world.hexes.items()
        if hx.terrain_class not in (TerrainClass.OCEAN, TerrainClass.LAKE)
    }

    def downstream_reaches_terminal(start, comp_set):
        """BFS along non-decreasing river_flow from start, not through comp_set."""
        visited = {start} | comp_set
        queue = deque([start])
        while queue:
            coord = queue.popleft()
            q, r = coord
            if q == 0 or q == w - 1 or r == 0 or r == h - 1:
                return True
            for nbr in neighbors(coord):
                if nbr not in world.hexes or nbr in visited:
                    continue
                nhx = world.hexes[nbr]
                if nhx.terrain_class == TerrainClass.OCEAN:
                    return True
                # Accept reaching a *different* lake as a valid intermediate terminal
                if nhx.terrain_class == TerrainClass.LAKE and nbr not in comp_set:
                    return True
                # Follow downstream: non-decreasing river_flow moves toward accumulation
                if (
                    nbr in land
                    and nhx.river_flow >= world.hexes[coord].river_flow
                    and nhx.river_flow > 0
                ):
                    visited.add(nbr)
                    queue.append(nbr)
        return False

    for comp in lake_comps:
        comp_set = frozenset(comp)
        border_rivers = [
            nbr
            for c in comp
            for nbr in neighbors(c)
            if nbr in land and world.hexes[nbr].river_flow > 0
        ]
        assert border_rivers, f"LAKE (size {len(comp)}) has no adjacent river hex at all"
        assert any(downstream_reaches_terminal(r, comp_set) for r in border_rivers), (
            f"LAKE component (size {len(comp)}) has no outflow: "
            "no border river hex has a downstream path to ocean/border"
        )


def test_lake_chain_terminates(world):
    """Chain rule: following lake outflow rivers must reach ocean or border without cycles.

    For each lake component, BFS through river hexes (not re-entering this lake) to find
    the next water body.  If that next body is another lake, recurse; if it is ocean or
    border the chain terminates successfully.  Tracking visited lake indices detects cycles.
    """
    lake_comps = _water_components(world, TerrainClass.LAKE)
    if not lake_comps:
        pytest.skip("No lakes in this world — nothing to check")

    w, h = world.width, world.height
    ocean = {c for c, hx in world.hexes.items() if hx.terrain_class == TerrainClass.OCEAN}
    land = {
        c
        for c, hx in world.hexes.items()
        if hx.terrain_class not in (TerrainClass.OCEAN, TerrainClass.LAKE)
    }
    hex_to_lake_idx = {c: i for i, comp in enumerate(lake_comps) for c in comp}

    def follow_river_to_water(start, current_comp):
        """BFS from *start* through river hexes; return 'ocean', 'border', lake_idx, or None."""
        visited = set(current_comp) | {start}
        queue = deque([start])
        while queue:
            coord = queue.popleft()
            cq, cr = coord
            if cq == 0 or cq == w - 1 or cr == 0 or cr == h - 1:
                return "border"
            for nbr in neighbors(coord):
                if nbr not in world.hexes or nbr in visited:
                    continue
                nhx = world.hexes[nbr]
                if nhx.terrain_class == TerrainClass.OCEAN:
                    return "ocean"
                if nbr in hex_to_lake_idx:
                    return hex_to_lake_idx[nbr]
                if nbr in land and nhx.river_flow > 0:
                    visited.add(nbr)
                    queue.append(nbr)
        return None

    for start_idx, start_comp in enumerate(lake_comps):
        visited_lakes: set[int] = {start_idx}
        queue: deque[int] = deque([start_idx])
        found_terminal = False

        while queue and not found_terminal:
            idx = queue.popleft()
            comp = lake_comps[idx]
            border_rivers = [
                nbr
                for c in comp
                for nbr in neighbors(c)
                if nbr in land and world.hexes[nbr].river_flow > 0
            ]
            for r in border_rivers:
                result = follow_river_to_water(r, comp)
                if result in ("ocean", "border"):
                    found_terminal = True
                    break
                if isinstance(result, int) and result not in visited_lakes:
                    visited_lakes.add(result)
                    queue.append(result)

        assert found_terminal, (
            f"LAKE component {start_idx} (size {len(start_comp)}) outflow chain "
            "does not terminate at ocean or border (possible cycle or missing drainage)"
        )


def test_water_body_reproducible():
    """Same seed produces identical OCEAN/LAKE classification."""
    s1 = _build_pipeline(seed=13).run()
    s2 = _build_pipeline(seed=13).run()
    for coord in s1.hexes:
        assert s1.hexes[coord].terrain_class == s2.hexes[coord].terrain_class, (
            f"terrain_class differs at {coord} between runs with same seed"
        )
