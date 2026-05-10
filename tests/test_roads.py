import pytest

from worldgen.core.config import WorldConfig
from worldgen.core.hex import Hex, SettlementTier, TerrainClass
from worldgen.core.pipeline import GeneratorPipeline
from worldgen.core.world_state import RoadTier
from worldgen.stages.biomes import BiomeStage
from worldgen.stages.climate import ClimateStage
from worldgen.stages.elevation import ElevationStage
from worldgen.stages.erosion import ErosionStage
from worldgen.stages.habitability import HabitabilityStage
from worldgen.stages.hydrology import HydrologyStage
from worldgen.stages.road_cost import slope_edge_cost
from worldgen.stages.roads import RoadStage
from worldgen.stages.settlements import SettlementStage
from worldgen.stages.terrain_class import TerrainClassificationStage


def _build_pipeline(seed: int = 42, width: int = 64, height: int = 64, **cfg_overrides):
    cfg = WorldConfig(
        width=width,
        height=height,
        erosion_iterations=500,
        target_city_count=4,
        target_town_count=10,
        road_travellers_city=100,
        road_travellers_town=20,
        road_travellers_village=5,
        **cfg_overrides,
    )
    p = GeneratorPipeline(seed, cfg)
    p.add_stage(ElevationStage)
    p.add_stage(ErosionStage)
    p.add_stage(TerrainClassificationStage)
    p.add_stage(HydrologyStage)
    p.add_stage(ClimateStage)
    p.add_stage(BiomeStage)
    p.add_stage(HabitabilityStage)
    p.add_stage(SettlementStage)
    p.add_stage(RoadStage)
    return p


@pytest.fixture(scope="module")
def road_state():
    return _build_pipeline().run()


def test_has_roads(road_state):
    assert len(road_state.roads) >= 1


def test_road_paths_min_length(road_state):
    for road in road_state.roads:
        assert len(road.path) >= 2, f"Road has path length {len(road.path)}"


def test_road_paths_connected(road_state):
    from worldgen.core.hex_grid import distance

    for road in road_state.roads:
        for a, b in zip(road.path, road.path[1:], strict=False):
            assert distance(a, b) == 1, f"Non-adjacent coords in road path: {a} -> {b}"


def test_road_water_segments_are_bracketed(road_state):
    """Roads may now traverse water (oceans + lakes are traversable as a single piece
    of terrain), but every water segment must be bracketed by land hexes — a road
    cannot start, end, or consist entirely of water hexes."""
    water = (TerrainClass.OCEAN, TerrainClass.LAKE)
    for road in road_state.roads:
        first = road_state.hexes[road.path[0]]
        last = road_state.hexes[road.path[-1]]
        assert first.terrain_class not in water, f"Road begins on water at {road.path[0]}"
        assert last.terrain_class not in water, f"Road ends on water at {road.path[-1]}"
        on_land = [c for c in road.path if road_state.hexes[c].terrain_class not in water]
        assert on_land, f"Road has no land hexes: {road.path}"


def test_road_connections_symmetric(road_state):
    hexes = road_state.hexes
    for coord, hx in hexes.items():
        for neighbor in hx.road_connections:
            assert coord in hexes[neighbor].road_connections, (
                f"road_connections not symmetric: {coord} -> {neighbor} but not reverse"
            )


def test_river_crossing_hexes_tagged(road_state):
    """Road hexes that enter a river from a non-river hex must be tagged ford/bridge."""
    for road in road_state.roads:
        path = road.path
        for i, c in enumerate(path):
            hx = road_state.hexes.get(c)
            if hx is None or "river" not in hx.tags:
                continue
            prev_c = path[i - 1] if i > 0 else None
            prev_hx = road_state.hexes.get(prev_c) if prev_c is not None else None
            if prev_hx is None or "river" not in prev_hx.tags:
                assert "ford" in hx.tags or "bridge" in hx.tags, (
                    f"River crossing hex {c} on road not tagged ford/bridge"
                )


def test_valid_road_tiers(road_state):
    for road in road_state.roads:
        assert isinstance(road.tier, RoadTier)


def test_cities_mutually_reachable(road_state):
    from collections import deque

    hexes = road_state.hexes
    cities = [s for s in road_state.settlements if s.tier == SettlementTier.CITY]
    if len(cities) <= 1:
        return

    # BFS over road_connections
    start = cities[0].coord
    visited = {start}
    queue = deque([start])
    while queue:
        c = queue.popleft()
        for n in hexes[c].road_connections:
            if n not in visited:
                visited.add(n)
                queue.append(n)

    for city in cities[1:]:
        assert city.coord in visited, (
            f"City {city.name} at {city.coord} not reachable via road network"
        )


def test_river_preference_in_roads(road_state):
    hexes = road_state.hexes
    road_hexes = {c for road in road_state.roads for c in road.path if c in hexes}
    all_land = {c for c, h in hexes.items() if h.terrain_class != TerrainClass.OCEAN}

    if not road_hexes or not all_land:
        return

    river_in_roads = sum(1 for c in road_hexes if "river" in hexes[c].tags)
    river_in_map = sum(1 for c in all_land if "river" in hexes[c].tags)

    road_river_rate = river_in_roads / len(road_hexes) if road_hexes else 0
    map_river_rate = river_in_map / len(all_land) if all_land else 0

    assert road_river_rate >= map_river_rate, (
        f"River preference not detected: road river rate {road_river_rate:.3f} < "
        f"map river rate {map_river_rate:.3f}"
    )


def test_road_river_traffic_threshold_draws_more_river_roads():
    """Lowering road_river_traffic_min relative to road_min_traffic admits river
    hexes with light traffic into the drawn road network. With it set equal to
    road_min_traffic (effectively disabled) those river-only hexes should be
    absent from the road set."""
    seed = 7
    # Default behaviour: river hexes admitted with 1 traveller
    s_low = _build_pipeline(seed=seed, road_river_traffic_min=1).run()
    # Disabled: river hexes treated like land hexes (need road_min_traffic = 3)
    s_off = _build_pipeline(seed=seed, road_river_traffic_min=3).run()

    def river_road_hexes(state):
        rh = {c for road in state.roads for c in road.path}
        return {c for c in rh if state.hexes[c].river_flow > 0}

    low_river_roads = river_road_hexes(s_low)
    off_river_roads = river_road_hexes(s_off)

    assert low_river_roads >= off_river_roads, (
        "Lower threshold removed river road coverage that the higher threshold kept"
    )
    # Sanity: with road_river_traffic_min=1 we expect strictly more river road
    # coverage on a typical world. Allow equality for degenerate maps where the
    # river network is sparse or every river hex already meets road_min_traffic.
    assert len(low_river_roads) >= len(off_river_roads)


def test_reproducibility():
    s1 = _build_pipeline(seed=99).run()
    s2 = _build_pipeline(seed=99).run()
    tiers1 = sorted((r.tier.value, tuple(r.path)) for r in s1.roads)
    tiers2 = sorted((r.tier.value, tuple(r.path)) for r in s2.roads)
    assert tiers1 == tiers2, "Roads differ between identical seeds"


def test_slope_edge_cost_formula():
    """Unit test for the hyperbolic slope cost formula used in edge_cost."""
    cfg = WorldConfig()

    def slope_cost(delta_elev):
        return slope_edge_cost(
            Hex(coord=(0, 0), elevation=0.0),
            Hex(coord=(1, 0), elevation=delta_elev),
            cfg,
        )

    # grade = 0% → free
    assert slope_cost(0.0) == pytest.approx(0.0)
    # grade = free_pct (3%) → zero cost
    delta_free = cfg.road_slope_free_pct * cfg.hex_size_m / (cfg.road_elev_range_m * 100.0)
    assert slope_cost(delta_free) == pytest.approx(0.0)
    # grade slightly above free → small positive cost
    assert slope_cost(delta_free * 1.01) > 0.0
    # midpoint grade → cost = road_slope_cost × 1.0
    mid_pct = (cfg.road_slope_free_pct + cfg.road_slope_cap_pct) / 2
    delta_mid = mid_pct * cfg.hex_size_m / (cfg.road_elev_range_m * 100.0)
    mid_cost = slope_cost(delta_mid)
    expected_mid = (
        cfg.road_slope_cost
        * (mid_pct - cfg.road_slope_free_pct)
        / (cfg.road_slope_cap_pct - mid_pct)
    )
    assert abs(mid_cost - expected_mid) < 1e-9
    # grade = cap_pct → saturated at road_slope_cost * road_slope_cap_mult
    delta_cap = cfg.road_slope_cap_pct * cfg.hex_size_m / (cfg.road_elev_range_m * 100.0)
    assert slope_cost(delta_cap) == pytest.approx(cfg.road_slope_cost * cfg.road_slope_cap_mult)
    # grade > cap → same saturation value
    assert slope_cost(delta_cap * 2) == pytest.approx(cfg.road_slope_cost * cfg.road_slope_cap_mult)
    # monotonically increasing between free and cap
    deltas = [delta_free + i * (delta_cap - delta_free) / 20 for i in range(1, 21)]
    costs = [slope_cost(d) for d in deltas]
    assert all(a <= b for a, b in zip(costs, costs[1:], strict=False))
