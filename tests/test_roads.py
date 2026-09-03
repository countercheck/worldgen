import pytest

from tests.worlds import build_pipeline
from worldgen.core.config import WorldConfig
from worldgen.core.hex import Hex, SettlementTier, TerrainClass
from worldgen.core.hex_grid import road_polylines
from worldgen.core.world_state import ROAD_TIER_RANK, RoadTier, road_edge_key
from worldgen.stages.road_cost import slope_edge_cost

# Traveller counts are turned well down from production so the gravity simulation stays
# affordable in a test; the road_state fixture in conftest.py uses the same numbers.
_ROAD_DEFAULTS = {
    "target_city_count": 4,
    "target_town_count": 10,
    "road_travellers_city": 100,
    "road_travellers_town": 20,
    "road_travellers_village": 5,
}


def _build_pipeline(seed: int = 42, width: int = 64, height: int = 64, **cfg_overrides):
    """Overrides win, so a test can vary any default — including the ones above."""
    return build_pipeline(
        seed=seed, width=width, height=height, **{**_ROAD_DEFAULTS, **cfg_overrides}
    )


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
        target_city_count=3,
        target_town_count=6,
        road_travellers_city=50,
        road_travellers_town=10,
        road_travellers_village=2,
    ).run()


def test_has_roads(road_state):
    assert len(road_state.road_edges) >= 1


def test_every_road_edge_joins_two_neighbouring_hexes(road_state):
    from worldgen.core.hex_grid import distance

    for a, b in road_state.road_edges:
        assert distance(a, b) == 1, f"road edge between non-adjacent hexes: {a} -> {b}"


def test_road_edges_are_stored_under_one_canonical_key(road_state):
    """An edge is undirected, so (a, b) and (b, a) must not both exist and disagree."""
    for key in road_state.road_edges:
        assert key == road_edge_key(*key)


def test_the_drawn_network_never_starts_or_ends_on_water(road_state):
    """Roads may traverse water — oceans and lakes are one piece of terrain to the
    router — but a drawn leg is land only, so no polyline may begin or end wet."""
    water = (TerrainClass.OCEAN, TerrainClass.LAKE)
    for _, leg in road_polylines(road_state.road_edges, road_state.hexes):
        assert len(leg) >= 2
        for end in (leg[0], leg[-1]):
            assert road_state.hexes[end].terrain_class not in water, (
                f"drawn road leg terminates on water at {end}"
            )


def test_road_connections_symmetric(road_state):
    hexes = road_state.hexes
    for coord, hx in hexes.items():
        for neighbor in hx.road_connections:
            assert coord in hexes[neighbor].road_connections, (
                f"road_connections not symmetric: {coord} -> {neighbor} but not reverse"
            )


def test_river_crossing_hexes_tagged(road_state):
    """A river hex the road network reaches from dry land is a crossing, and is tagged."""
    for a, b in road_state.road_edges:
        for coord, other in ((a, b), (b, a)):
            hx = road_state.hexes.get(coord)
            other_hx = road_state.hexes.get(other)
            if hx is None or "river" not in hx.tags:
                continue
            if other_hx is not None and "river" in other_hx.tags:
                continue
            assert "ford" in hx.tags or "bridge" in hx.tags, (
                f"River crossing hex {coord} on road not tagged ford/bridge"
            )


def test_valid_road_tiers(road_state):
    for tier in road_state.road_edges.values():
        assert isinstance(tier, RoadTier)


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


def test_roads_route_when_river_flow_is_continuous():
    """river_flow_continuous must not change what counts as a river channel.

    In that mode HydrologyStage writes a flow value onto every draining land hex, not
    just the channels.  Anything in the road costs that identifies a river by
    `river_flow > 0` then calls the whole map a river: every land-to-land edge reads as
    channel travel and prices at infinity, and the network does not merely route badly,
    it fails to route at all.  The costs identify a channel by the "river" tag for this
    reason, so the two modes must produce the same roads.
    """
    plain = _build_pipeline(seed=42).run()
    continuous = _build_pipeline(seed=42, river_flow_continuous=True).run()

    assert continuous.road_edges, "no roads generated with river_flow_continuous"
    flowing = [h for h in continuous.hexes.values() if h.river_flow > 0]
    tagged = [h for h in continuous.hexes.values() if "river" in h.tags]
    assert len(flowing) > len(tagged) * 2, (
        "continuous mode should put flow on far more hexes than are tagged as river; "
        "if it does not, this test is no longer exercising the case it was written for"
    )

    def network(ws):
        return sorted((k, t.value) for k, t in ws.road_edges.items())

    assert network(continuous) == network(plain), (
        "roads differ between flow modes: something is still identifying a river channel "
        "by river_flow rather than by the tag"
    )


def test_river_corridor_preference_in_roads(road_state):
    """Roads still follow river valleys — but along the corridor, not down the channel.

    The pull used to sit on river hexes themselves, so this measured how often roads
    landed *on* a river. Roads now take the bank instead, so the corridor (a river hex
    or a hex beside one) is what they should over-represent.
    """
    from worldgen.core.hex_grid import neighbors

    hexes = road_state.hexes
    road_hexes = {c for edge in road_state.road_edges for c in edge if c in hexes}
    all_land = {c for c, h in hexes.items() if h.terrain_class != TerrainClass.OCEAN}

    if not road_hexes or not all_land:
        return

    river = {c for c in all_land if "river" in hexes[c].tags}
    # The banks, not the corridor. Measuring the corridor — banks *and* channel — asks
    # two questions at once and fails on the answer to the wrong one: roads decline river
    # hexes on purpose, at `road_river_hex_cost` and with channel travel excluded
    # outright, so a corridor measure demands they over-use the banks by enough to make
    # up for never touching the water. What `bank_discount` actually claims is narrower
    # and is what is checked here: given dry land, a road prefers the bank beside a river
    # to dry land away from one.
    dry = all_land - river
    banks = {n for c in river for n in neighbors(c) if n in dry}

    road_dry = road_hexes & dry
    if not road_dry or not banks:
        return

    road_rate = len(road_dry & banks) / len(road_dry)
    map_rate = len(banks) / len(dry)

    assert road_rate >= map_rate, (
        f"Riverbank preference not detected: roads run on a bank {road_rate:.3f} of the "
        f"time against {map_rate:.3f} of the dry land being bank"
    )


def test_roads_decline_the_channel_itself(road_state):
    """The other half of what the corridor measure used to conflate.

    A road beside a river is following the valley; a road *in* it has nowhere to be a
    bank of. So river hexes should be markedly under-used relative to how much of the
    land they make up.
    """
    hexes = road_state.hexes
    road_hexes = {c for edge in road_state.road_edges for c in edge if c in hexes}
    all_land = {c for c, h in hexes.items() if h.terrain_class != TerrainClass.OCEAN}
    river = {c for c in all_land if "river" in hexes[c].tags}
    if not road_hexes or not river:
        return

    road_rate = len(road_hexes & river) / len(road_hexes)
    map_rate = len(river) / len(all_land)
    assert road_rate < map_rate, (
        f"roads sit on the channel {road_rate:.3f} of the time against {map_rate:.3f} "
        "of the land being river — the channel is not being declined"
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
        rh = {c for edge in state.road_edges for c in edge}
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
    net1 = sorted((k, t.value) for k, t in s1.road_edges.items())
    net2 = sorted((k, t.value) for k, t in s2.road_edges.items())
    assert net1 == net2, "Roads differ between identical seeds"


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
    delta_free = cfg.road_slope_free_pct * cfg.hex_size_m / 100.0
    assert slope_cost(delta_free) == pytest.approx(0.0)
    # grade slightly above free → small positive cost
    assert slope_cost(delta_free * 1.01) > 0.0
    # midpoint grade → cost = road_slope_cost × 1.0
    mid_pct = (cfg.road_slope_free_pct + cfg.road_slope_cap_pct) / 2
    delta_mid = mid_pct * cfg.hex_size_m / 100.0
    mid_cost = slope_cost(delta_mid)
    expected_mid = (
        cfg.road_slope_cost
        * (mid_pct - cfg.road_slope_free_pct)
        / (cfg.road_slope_cap_pct - mid_pct)
    )
    assert abs(mid_cost - expected_mid) < 1e-9
    # grade = cap_pct → saturated at road_slope_cost * road_slope_cap_mult
    delta_cap = cfg.road_slope_cap_pct * cfg.hex_size_m / 100.0
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

    offenders = [
        (a, b) for a, b in state.road_edges if frozenset((a, b)) in channel and not exempt(a, b)
    ]
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

    # The graph answers this directly: a crossing is a river hex whose road edges reach
    # exactly two banks, and those two must lie on opposite arcs of the channel.
    road_adj: dict = {}
    for a, b in road_state.road_edges:
        road_adj.setdefault(a, set()).add(b)
        road_adj.setdefault(b, set()).add(a)

    offenders = []
    for c, (i, j) in channel_dirs.items():
        if c in settled or c not in road_adj:
            continue
        # Only a bank->river->bank crossing is in question; a neighbour that is itself a
        # river hex is channel travel, covered by its own test below.
        banks = [n for n in road_adj[c] if hexes[n].river_flow <= 0]
        if len(banks) != 2 or len(banks) != len(road_adj[c]):
            continue
        if same_arc(c, i, j, banks[0], banks[1]):
            offenders.append((banks[0], c, banks[1]))
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

    # A run of channel travel is an edge joining two river hexes: on the graph there is
    # no need to reconstruct the sequence, since the offending step *is* the edge.
    worst = [
        (a, b)
        for a, b in state.road_edges
        if hexes[a].river_flow > 0
        and hexes[b].river_flow > 0
        and a not in settled
        and b not in settled
    ]

    assert not worst, f"roads travel along {len(worst)} river runs, e.g. {worst[:3]}"


def test_road_hexes_on_rivers_are_rare(any_road_state):
    """Whatever the route, a road should almost always have a bank under it."""
    state = any_road_state
    hexes = state.hexes
    road_hexes = {c for edge in state.road_edges for c in edge if c in hexes}
    settled = {s.coord for s in state.settlements}
    if not road_hexes:
        return
    on_river = {c for c in road_hexes if hexes[c].river_flow > 0 and c not in settled}
    assert len(on_river) / len(road_hexes) < 0.10, (
        f"{len(on_river)}/{len(road_hexes)} road hexes sit in a river channel"
    )


def test_no_two_important_roads_run_side_by_side_for_long(road_state):
    """Two roads a kilometre apart are fine; two *highways* a kilometre apart are not.

    A hex is beside another road all the time — at junctions, at a town, where a valley
    route passes under a hillside one. What no map should show is a pair of roads of the
    same tier, at the same height, keeping each other company for miles: that is one road
    drawn twice, and it is what the pathfinder produces when it cannot find the same
    corridor twice running.

    So this measures the thing that matters rather than raw adjacency. A *parallel pair* is
    two road hexes that are neighbours with no edge between them; a *run* chains pairs that
    advance together. Tracks are exempt — a lane beside a road is a lane.
    """
    from worldgen.core.hex_grid import neighbors

    edges = road_state.road_edges
    hex_tier: dict = {}
    for (a, b), tier in edges.items():
        for c in (a, b):
            if ROAD_TIER_RANK[tier] > ROAD_TIER_RANK.get(hex_tier.get(c), -1):
                hex_tier[c] = tier
    important = {c for c, t in hex_tier.items() if t is not RoadTier.TRACK}

    pairs = {
        tuple(sorted((c, n)))
        for c in important
        for n in neighbors(c)
        if n in important and road_edge_key(c, n) not in edges and hex_tier[c] is hex_tier[n]
    }
    if not pairs:
        return

    adj: dict = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    def advances_to(pair):
        a, b = pair
        return {
            cand
            for a2 in adj.get(a, ())
            for b2 in adj.get(b, ())
            if (cand := tuple(sorted((a2, b2)))) != pair and cand in pairs
        }

    seen: set = set()
    worst: set = set()
    for start in pairs:
        if start in seen:
            continue
        stack, run = [start], set()
        while stack:
            x = stack.pop()
            if x in run:
                continue
            run.add(x)
            stack.extend(advances_to(x) - run)
        seen |= run
        if len(run) > len(worst):
            worst = run

    assert len(worst) <= 4, (
        f"{len(worst)} hexes of primary/secondary road run parallel to another of the same "
        f"tier — one road drawn twice. Near {sorted(worst)[0]}"
    )


def test_no_road_skirts_a_settlement_it_could_pass_through(road_state):
    """A road whose two ends both touch a town entered its ring and left without arriving.

    At 1 hex = 1 km that is a trunk road passing a market town at the width of one field.
    Bypasses are a motor-age idea; before that the road went through the town, which is
    half the reason the town is where it is.

    One hex, deliberately, and not more. At two the test stops discriminating — near a
    town most edges have both ends within two hexes, because they are the roads radiating
    *from* it — and bending those would zigzag every route through every settlement it
    passed. Measured on a 128x128 map, "both ends within r" covers 5% of the network at
    r=1, 12% at r=2 and 22% at r=3, while the road distance to a nearby town already
    equals the crow-flies distance at every radius out to four.
    """
    from worldgen.core.hex_grid import neighbors

    offenders = []
    for seat in {s.coord for s in road_state.settlements}:
        ring = set(neighbors(seat))
        offenders += [(seat, a, b) for a, b in road_state.road_edges if a in ring and b in ring]

    assert not offenders, (
        f"{len(offenders)} roads pass a settlement without entering it, e.g. "
        f"{offenders[0][1]}->{offenders[0][2]} around {offenders[0][0]}"
    )
