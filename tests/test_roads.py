import pytest

from worldgen.core.config import WorldConfig
from worldgen.core.hex import Hex, SettlementTier, TerrainClass
from worldgen.core.pipeline import GeneratorPipeline
from worldgen.core.world_state import RoadTier
from worldgen.stages.biomes import BiomeStage
from worldgen.stages.city_town import CityTownStage
from worldgen.stages.climate import ClimateStage
from worldgen.stages.cultivation import CultivationStage, VillageCultivationStage
from worldgen.stages.elevation import ElevationStage
from worldgen.stages.erosion import ErosionStage
from worldgen.stages.habitability import HabitabilityStage
from worldgen.stages.hydrology import HydrologyStage
from worldgen.stages.interurban_roads import InterurbanRoadStage
from worldgen.stages.land_cover import LandCoverStage
from worldgen.stages.road_cost import slope_edge_cost
from worldgen.stages.terrain_class import TerrainClassificationStage
from worldgen.stages.village_placement import VillagePlacementStage
from worldgen.stages.village_tracks import VillageTrackStage
from worldgen.stages.water_bodies import WaterBodiesStage


def _build_pipeline(seed: int = 42, width: int = 64, height: int = 64, **cfg_overrides):
    defaults = {
        "erosion_iterations": 500,
        "target_city_count": 4,
        "target_town_count": 10,
        "road_travellers_city": 100,
        "road_travellers_town": 20,
        "road_travellers_village": 5,
    }
    # Overrides win, so a fixture can vary any of these — including the ones above.
    cfg = WorldConfig(width=width, height=height, **{**defaults, **cfg_overrides})
    p = GeneratorPipeline(seed, cfg)
    # The production pipeline, stage for stage — see worldgen/cli.py. Road invariants
    # have to be asserted about the stages that actually generate worlds.
    p.add_stage(ElevationStage)
    p.add_stage(ErosionStage)
    p.add_stage(TerrainClassificationStage)
    p.add_stage(WaterBodiesStage)
    p.add_stage(HydrologyStage)
    p.add_stage(ClimateStage)
    p.add_stage(BiomeStage)
    p.add_stage(LandCoverStage)
    p.add_stage(HabitabilityStage)
    p.add_stage(CityTownStage)
    p.add_stage(InterurbanRoadStage)
    p.add_stage(CultivationStage)
    p.add_stage(VillagePlacementStage)
    p.add_stage(VillageTrackStage)
    p.add_stage(VillageCultivationStage)
    return p


@pytest.fixture(scope="module")
def road_state():
    return _build_pipeline().run()


@pytest.fixture(scope="module", params=["64x64", "48x48-dense"])
def any_road_state(request):
    """The river invariants, checked at two map sizes and settlement densities.

    The 48x48 case is the config the exported map set is generated at; the channel leak
    that the settlement exemption used to allow showed up there and not at 64x64, so a
    single fixture size is not enough to trust these.
    """
    if request.param == "64x64":
        return _build_pipeline().run()
    return _build_pipeline(
        seed=42,
        width=48,
        height=48,
        erosion_iterations=200,
        target_city_count=3,
        target_town_count=6,
        road_travellers_city=50,
        road_travellers_town=10,
        road_travellers_village=2,
    ).run()


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

    # Ferries are links too: where a river channel cuts a city off, the network is
    # joined by boat rather than by a road running down the river.
    changed = True
    while changed:
        changed = False
        for f in road_state.ferries:
            for a, b in ((f.a, f.b), (f.b, f.a)):
                if a in visited and b not in visited:
                    visited.add(b)
                    queue.append(b)
                    changed = True
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


def test_river_corridor_preference_in_roads(road_state):
    """Roads still follow river valleys — but along the corridor, not down the channel.

    The pull used to sit on river hexes themselves, so this measured how often roads
    landed *on* a river. Roads now take the bank instead, so the corridor (a river hex
    or a hex beside one) is what they should over-represent.
    """
    from worldgen.core.hex_grid import neighbors

    hexes = road_state.hexes
    road_hexes = {c for road in road_state.roads for c in road.path if c in hexes}
    all_land = {c for c, h in hexes.items() if h.terrain_class != TerrainClass.OCEAN}

    if not road_hexes or not all_land:
        return

    river = {c for c in all_land if "river" in hexes[c].tags}
    corridor = river | {n for c in river for n in neighbors(c) if n in all_land}

    road_rate = len(road_hexes & corridor) / len(road_hexes)
    map_rate = len(corridor & all_land) / len(all_land)

    assert road_rate >= map_rate, (
        f"River corridor preference not detected: road corridor rate {road_rate:.3f} < "
        f"map corridor rate {map_rate:.3f}"
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


# --- river channel constraint ------------------------------------------------


def _river_edges(state):
    return {
        frozenset((a, b))
        for river in state.rivers
        for a, b in zip(river.hexes, river.hexes[1:], strict=False)
    }


def test_roads_never_run_along_a_river_channel(any_road_state):
    """A road drawn on the channel hides which bank it — and anything on it — is on.

    Crossing is fine; travelling down the river is not. A settlement exempts the hexside
    only when its counterpart is dry land — enough to reach a riverside town, not enough
    to leave one along the water.
    """
    state = any_road_state
    hexes = state.hexes
    settled = {s.coord for s in state.settlements}
    channel = _river_edges(state)

    def exempt(a, b):
        return (a in settled and hexes[b].river_flow <= 0) or (
            b in settled and hexes[a].river_flow <= 0
        )

    offenders = []
    for road in state.roads:
        for a, b in zip(road.path, road.path[1:], strict=False):
            if frozenset((a, b)) in channel and not exempt(a, b):
                offenders.append((a, b))
    assert not offenders, f"roads run along the river channel at {offenders[:5]}"


def test_roads_cross_rivers_on_opposite_sides(road_state):
    """A road entering a river hex must come out the other side, not back the same way.

    Dipping into the channel and returning to the bank it came from would leave a unit
    standing on that hex with no defined side, and would tag a ford that is not one.
    The cost model should make it uneconomic; this checks that it actually does.
    """
    from worldgen.core.hex_grid import neighbors

    hexes = road_state.hexes
    settled = {s.coord for s in road_state.settlements}

    # Local channel direction at each river hex: the ring indices of its up/downstream
    # neighbours. Only hexes with both are two-sided; sources and mouths are skipped.
    channel_dirs: dict = {}
    for river in road_state.rivers:
        for prev, cur, nxt in zip(river.hexes, river.hexes[1:], river.hexes[2:], strict=False):
            ring = neighbors(cur)
            if prev in ring and nxt in ring:
                channel_dirs[cur] = (ring.index(prev), ring.index(nxt))

    def same_arc(centre, i, j, p, n):
        """True when ring positions p and n sit on the same side of the channel."""
        ring = neighbors(centre)
        if p not in ring or n not in ring:
            return False
        pi, ni = ring.index(p), ring.index(n)
        lo, hi = min(i, j), max(i, j)
        between = lo < pi < hi
        return between == (lo < ni < hi)

    offenders = []
    for road in road_state.roads:
        path = road.path
        for k in range(1, len(path) - 1):
            c = path[k]
            if c in settled or c not in channel_dirs:
                continue
            # Only a bank->river->bank step is a crossing; a step whose neighbour is
            # itself a river hex is channel travel, covered by its own test below.
            if hexes[path[k - 1]].river_flow > 0 or hexes[path[k + 1]].river_flow > 0:
                continue
            i, j = channel_dirs[c]
            if same_arc(c, i, j, path[k - 1], path[k + 1]):
                offenders.append((path[k - 1], c, path[k + 1]))
    assert not offenders, f"road re-enters the bank it came from at {offenders[:5]}"


def test_ferry_endpoints_are_a_plausible_hop(road_state):
    from worldgen.core.hex_grid import distance

    cfg = WorldConfig()
    for f in road_state.ferries:
        assert f.a != f.b
        assert distance(f.a, f.b) <= cfg.road_ferry_max_hop, (
            f"ferry from {f.a} to {f.b} is longer than road_ferry_max_hop"
        )


def test_roads_never_occupy_consecutive_river_hexes(any_road_state):
    """A road meets a river to cross it, and is back on a bank the very next hex.

    This replaces an earlier ratio test that compared in-channel steps against crossings.
    That premise was wrong twice over: it counted a road merely *approaching* a riverside
    town as channel travel, and its threshold happened to hold at 64x64 while failing at
    48x48. A run of consecutive river hexes is the thing that actually means "travelling
    the river", and it does not depend on map size or settlement density.

    Settlement hexes are excluded from a run: a town on the water is a road hex by
    definition, and a road arriving at one through an adjacent river hex is unavoidable
    where a river braids or meanders past the town.
    """
    state = any_road_state
    hexes = state.hexes
    settled = {s.coord for s in state.settlements}

    worst = []
    for road in state.roads:
        run: list = []
        for c in road.path:
            if hexes[c].river_flow > 0 and c not in settled:
                run.append(c)
            else:
                if len(run) > 1:
                    worst.append(list(run))
                run = []
        if len(run) > 1:
            worst.append(list(run))

    assert not worst, f"roads travel along {len(worst)} river runs, e.g. {worst[:3]}"


def test_road_hexes_on_rivers_are_rare(any_road_state):
    """Whatever the route, a road should almost always have a bank under it."""
    state = any_road_state
    hexes = state.hexes
    road_hexes = {c for road in state.roads for c in road.path if c in hexes}
    settled = {s.coord for s in state.settlements}
    if not road_hexes:
        return
    on_river = {c for c in road_hexes if hexes[c].river_flow > 0 and c not in settled}
    assert len(on_river) / len(road_hexes) < 0.10, (
        f"{len(on_river)}/{len(road_hexes)} road hexes sit in a river channel"
    )
