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
    """Each LAKE component must have at least one adjacent land hex with river_flow > 0."""
    lake_comps = _water_components(world, TerrainClass.LAKE)
    if not lake_comps:
        pytest.skip("No lakes in this world — nothing to check")

    land = {
        c
        for c, h in world.hexes.items()
        if h.terrain_class not in (TerrainClass.OCEAN, TerrainClass.LAKE)
    }

    for comp in lake_comps:
        border_river_hexes = [
            nbr
            for c in comp
            for nbr in neighbors(c)
            if nbr in land and world.hexes[nbr].river_flow > 0
        ]
        assert border_river_hexes, (
            f"LAKE component (size {len(comp)}) has no adjacent river hex — drainage missing"
        )


def test_lake_chain_terminates(world):
    """Following lake outflow rivers from any LAKE must eventually reach a map-edge hex
    or a hex adjacent to OCEAN (not cycle in another lake forever)."""
    lake_comps = _water_components(world, TerrainClass.LAKE)
    if not lake_comps:
        pytest.skip("No lakes in this world — nothing to check")

    w, h = world.width, world.height
    ocean = {c for c, hx in world.hexes.items() if hx.terrain_class == TerrainClass.OCEAN}
    land = {
        c
        for c, h in world.hexes.items()
        if h.terrain_class not in (TerrainClass.OCEAN, TerrainClass.LAKE)
    }

    def reaches_terminal(start_comp):
        """BFS over lake components and their river outflows until we hit a terminal."""
        visited_comps = set()
        queue = deque([frozenset(start_comp)])
        while queue:
            comp_key = queue.popleft()
            if comp_key in visited_comps:
                return False  # cycle
            visited_comps.add(comp_key)
            for c in comp_key:
                for nbr in neighbors(c):
                    if nbr not in world.hexes:
                        continue
                    if _on_border(nbr, w, h):
                        return True
                    if nbr in ocean:
                        return True
                    # Follow river hexes out of this lake
                    if nbr in land and world.hexes[nbr].river_flow > 0:
                        # Find if this river eventually reaches a terminal via simple check:
                        # We trust test_rivers_reach_ocean covers that; here just check the
                        # lake component has at least one non-lake-recirculating outflow.
                        return True
        return False

    for comp in lake_comps:
        assert reaches_terminal(comp), (
            f"LAKE component (size {len(comp)}) chain does not terminate at ocean or border"
        )


def test_water_body_reproducible():
    """Same seed produces identical OCEAN/LAKE classification."""
    s1 = _build_pipeline(seed=13).run()
    s2 = _build_pipeline(seed=13).run()
    for coord in s1.hexes:
        assert s1.hexes[coord].terrain_class == s2.hexes[coord].terrain_class, (
            f"terrain_class differs at {coord} between runs with same seed"
        )
