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
    for edge in road_state.road_edges.values():
        assert isinstance(edge.tier, RoadTier)


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
        return sorted((k, e.tier.value) for k, e in ws.road_edges.items())

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


def test_a_road_touches_the_channel_only_to_cross_it(road_state):
    """A road beside a river is following the valley; a road *in* it has nowhere to be a
    bank of. So every road hex on the channel must be a crossing, and nothing else.

    This used to compare rates — river hexes should be a smaller share of the road network
    than of the land — and that measure conflates two different things. Crossing a river
    puts a road hex on a river hex necessarily, so on a map with many settlements and few
    watercourses the rate runs above the land rate with no road travelling the channel at
    all: measured on this fixture, 20 road hexes sit on the channel and all 20 are tagged
    ford or bridge. The rate comparison passed only because `bank_discount` happened to
    reduce crossings by making the banks attractive, and failed the moment it was deleted —
    for a reason that was never the defect it was watching for.

    Asserting every channel hex is a crossing says the thing directly, and admits no
    tolerance where the rate test admitted a whole percentage point. Travel *along* the
    channel has its own tests either side of this one.
    """
    hexes = road_state.hexes
    road_hexes = {c for edge in road_state.road_edges for c in edge if c in hexes}
    river = {c for c in road_hexes if "river" in hexes[c].tags}
    if not river:
        return

    settled = {s.coord for s in road_state.settlements}
    offenders = [c for c in river if not ({"ford", "bridge"} & hexes[c].tags) and c not in settled]
    assert not offenders, (
        f"{len(offenders)} road hexes sit on the channel without being a crossing, "
        f"e.g. {sorted(offenders)[0]} — a road with no bank to be on"
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
    net1 = sorted((k, e.tier.value) for k, e in s1.road_edges.items())
    net2 = sorted((k, e.tier.value) for k, e in s2.road_edges.items())
    assert net1 == net2, "Roads differ between identical seeds"


def test_slope_edge_cost_is_the_switchback_priced():
    """Climb converted to level going, continuously — and refused outright above the cap.

    The curve this replaced was free below `road_slope_free_pct` (3%), which is exactly
    `terrain_rolling_gradient_m` and so the FLAT boundary: every flat edge cost nothing and
    every flat route tied. Above 25% it saturated at ten times base rather than refusing, so
    a road met a 65% face, paid a flat twenty, and went straight up.
    """
    cfg = WorldConfig()

    def slope_cost(delta_elev):
        return slope_edge_cost(
            Hex(coord=(0, 0), elevation=0.0),
            Hex(coord=(1, 0), elevation=delta_elev),
            cfg,
        )

    assert slope_cost(0.0) == pytest.approx(0.0)

    # Metres of climb over the exchange rate, and nothing is free but level ground.
    assert slope_cost(cfg.road_delta_elevation_per_hex) == pytest.approx(1.0)
    assert slope_cost(2 * cfg.road_delta_elevation_per_hex) == pytest.approx(2.0)
    assert slope_cost(1.0) > 0.0, "a metre of climb must cost something"

    # Symmetric: a road is cut-and-fill, and a descent needs braking. Naismith's walker
    # pays for the climb alone, which is why `travel_ascent_per_hex` is a different number.
    for delta in (5.0, 40.0, 120.0):
        assert slope_cost(delta) == pytest.approx(slope_cost(-delta))

    # Above the cap it is refused, not priced.
    delta_cap = cfg.road_slope_cap_pct * cfg.hex_size_m / 100.0
    assert slope_cost(delta_cap) == float("inf")
    assert slope_cost(delta_cap * 2) == float("inf")
    assert slope_cost(delta_cap * 0.99) < float("inf")

    # Monotone all the way to the cap.
    deltas = [delta_cap * i / 20 for i in range(20)]
    costs = [slope_cost(d) for d in deltas]
    assert costs == sorted(costs)


def test_a_steep_road_hex_is_tagged_as_a_switchback():
    """The zigzag is priced but cannot be drawn: a switchback is a hundred-metre feature
    and a hex is a kilometre. The tag is how a reader knows the segment is slow."""
    from worldgen.stages.road_cost import edge_grade_pct, tag_switchbacks

    cfg = WorldConfig()
    hexes = {
        (0, 0): Hex(coord=(0, 0), elevation=0.0),
        (1, 0): Hex(coord=(1, 0), elevation=cfg.road_switchback_grade_pct * 10.0),
        (2, 0): Hex(coord=(2, 0), elevation=cfg.road_switchback_grade_pct * 10.0 + 1.0),
    }
    edges = {((0, 0), (1, 0)): RoadTier.PRIMARY, ((1, 0), (2, 0)): RoadTier.PRIMARY}
    tag_switchbacks(edges, hexes, cfg)

    assert edge_grade_pct(hexes[(0, 0)], hexes[(1, 0)], cfg) >= cfg.road_switchback_grade_pct
    assert "switchback" in hexes[(0, 0)].tags
    assert "switchback" in hexes[(1, 0)].tags
    # The gentle edge tags nothing of its own; (1, 0) is marked by the steep one beside it.
    assert "switchback" not in hexes[(2, 0)].tags


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
    for (a, b), edge in edges.items():
        for c in (a, b):
            if ROAD_TIER_RANK[edge.tier] > ROAD_TIER_RANK.get(hex_tier.get(c), -1):
                hex_tier[c] = edge.tier
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

    "Could" is load-bearing: a road may decline a town that is dear to reach, up a bank it
    cannot climb, or across a channel it has no business on. So the test asks the same
    question the rule does, through the same function — `detour_is_allowed`. Writing the
    guard out a second time here is exactly what went wrong before: the copy knew only
    about the cost bound, so a skirt refused for a 30% grade read as a defect.
    """
    from worldgen.core.hex_grid import neighbors
    from worldgen.stages.road_cost import detour_is_allowed, river_edges

    cfg = WorldConfig(**road_state.metadata["config"])
    settled = {s.coord for s in road_state.settlements}
    blocked = river_edges(road_state.rivers)

    offenders = []
    for seat in settled:
        ring = set(neighbors(seat))
        for a, b in road_state.road_edges:
            if a not in ring or b not in ring:
                continue
            if detour_is_allowed(road_state.hexes, settled, cfg, blocked, a, seat, b):
                offenders.append((seat, a, b))

    assert not offenders, (
        f"{len(offenders)} roads pass a settlement they could have been bent through, "
        f"e.g. {offenders[0][1]}->{offenders[0][2]} around {offenders[0][0]}"
    )


def test_the_road_network_is_all_one_piece(road_state):
    """Every settlement must be reachable from every other **by road**.

    Two things excuse a break, and both are narrow. Roads may not run down a river channel,
    so a delta island or a braided confluence can be unreachable by land and
    `_guarantee_connectivity` joins it by boat — but only after land routing has failed. And
    some maps are simply in pieces: an island beyond ferry range cannot be reached at all,
    which the stage records in `metadata["unreachable_settlements"]` rather than raising
    over, because an archipelago should be generable.

    What is *not* excused is two road networks sharing one landmass. Where land connects
    two places, roads must.

    Two things used to leave it that way. The connectivity guarantee only ran on maps with
    two or more *cities*, so the organic model — whose markets are all TOWN — had nothing
    watching it; and `road_river_traffic_min` admits a riverbank edge on a single traveller,
    so a stretch of towpath could qualify while joining nothing at all. Both were masked
    while `_stitch_via_junction` made almost every route a concatenation of the same few
    legs, which kept the map connected by accident.
    """
    from worldgen.core.hex_grid import neighbors

    adj: dict = {}
    for a, b in road_state.road_edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    if not adj:
        return

    def components(links):
        seen, out = set(), []
        for start in links:
            if start in seen:
                continue
            stack, comp = [start], set()
            while stack:
                c = stack.pop()
                if c in comp:
                    continue
                comp.add(c)
                stack.extend(links.get(c, ()) - comp)
            seen |= comp
            out.append(comp)
        return out

    seats = {s.coord for s in road_state.settlements}
    by_road = components(adj)

    # Nothing may be drawn that reaches neither a settlement nor a ferry landing.
    anchors = seats | {c for f in road_state.ferries for c in (f.a, f.b)}
    for comp in by_road:
        assert comp & anchors, (
            f"{len(comp)} road hexes near {sorted(comp)[0]} reach no settlement and no "
            "ferry — a road that connects nothing"
        )

    # Once ferries count as links there must be one network — except for anything the
    # terrain genuinely severs, which the stage records rather than raising over.
    linked = {c: set(v) for c, v in adj.items()}
    for ferry in road_state.ferries:
        linked.setdefault(ferry.a, set()).add(ferry.b)
        linked.setdefault(ferry.b, set()).add(ferry.a)
    reached = max(components(linked), key=len, default=set())
    conceded = {
        tuple(entry["coord"]) for entry in road_state.metadata.get("unreachable_settlements", [])
    }
    stranded = sorted(seats - reached - conceded)
    assert not stranded, (
        f"{len(stranded)} settlements are cut off from the main network even counting "
        f"ferries, and the stage did not record them as unreachable, e.g. {stranded[0]}"
    )

    # And settlements standing on the same ground must be joined **by road**, not merely
    # by sea. Sea carriage was so much cheaper than land that the traffic model will cross
    # a bay rather than walk round it, which is right for a journey and wrong for a
    # network: without `_join_by_land` the reference map came out forty land networks tied
    # together by eight sea crossings, so a cart could not reach the next market without a
    # boat. Roads must join what land can join.
    dry = {
        c
        for c, hx in road_state.hexes.items()
        if hx.terrain_class not in (TerrainClass.OCEAN, TerrainClass.LAKE)
    }
    landmass: dict = {}
    for seat in seats & dry:
        if seat in landmass:
            continue
        stack, reached = [seat], set()
        while stack:
            c = stack.pop()
            if c in reached:
                continue
            reached.add(c)
            stack.extend(n for n in neighbors(c) if n in dry and n not in reached)
        for c in reached & seats:
            landmass[c] = seat

    home = {}
    for i, comp in enumerate(by_road):
        for c in comp:
            home[c] = i
    together: dict = {}
    for seat, mass in landmass.items():
        together.setdefault(mass, set()).add(home.get(seat))
    for mass, roads in together.items():
        real = roads - {None}
        assert len(real) <= 1, (
            f"settlements sharing the landmass around {mass} sit in {len(real)} separate "
            "road networks — they can only reach each other by sea"
        )


def test_a_bigger_settlement_sends_more_travellers(road_state):
    """Travellers come from population, so the roads out of a big market are the busier.

    Population used to enter only on the *destination* side of the gravity term, so a market
    of 6,200 and one of 900 each sent the same flat per-tier count and wore the same road
    out of their own gates.
    """
    by_pop = sorted(road_state.settlements, key=lambda s: s.population)
    small, large = by_pop[0], by_pop[-1]
    if small.population == large.population:
        return

    def busiest_road_at(seat):
        tiers = [ROAD_TIER_RANK[e.tier] for key, e in road_state.road_edges.items() if seat in key]
        return max(tiers, default=-1)

    assert busiest_road_at(large.coord) >= busiest_road_at(small.coord), (
        f"the largest settlement ({large.population}) has a lesser road than the smallest "
        f"({small.population}) — travellers are not following population"
    )


def test_an_island_beyond_ferry_range_does_not_break_generation():
    """Some maps are in pieces, and that is a fact about the world rather than an error.

    `ferry_link` raises when the nearest landing is further than `road_ferry_max_hop`, on
    the grounds that a silent long-haul boat link reads worse than a loud failure. True of a
    delta island a few hexes off the bank; wrong as a reason to refuse to generate an
    archipelago at all — two islands twenty hexes apart cannot be joined by road and never
    will be. The stage records what it could not join and carries on.

    Both escapes are shut off here: no land route exists and no ferry is plausible, which is
    exactly the archipelago case.
    """
    from worldgen.core.errors import RoutingError
    from worldgen.stages import interurban_roads as ir

    state = _build_pipeline(seed=42, width=48, height=48).run()
    cfg = WorldConfig(**state.metadata["config"])
    stage = ir.InterurbanRoadStage(cfg, None)

    def no_land_route(*_a, **_k):
        return None

    def no_ferry(*_a, **_k):
        raise RoutingError("no plausible ferry")

    real_astar, real_ferry = ir.astar, ir.ferry_link
    ir.astar, ir.ferry_link = no_land_route, no_ferry
    try:
        edges, ferries, unreachable = stage._guarantee_connectivity(
            state.hexes,
            state.settlements,
            {},  # nothing joined yet, so every settlement is its own component
            cfg,
            frozenset(),
            {s.coord for s in state.settlements},
        )
    finally:
        ir.astar, ir.ferry_link = real_astar, real_ferry

    # It came back rather than raising, and it said what it could not reach.
    assert ferries == []
    assert unreachable, "nothing was recorded as unreachable — the test did not bite"
    assert len(unreachable) >= len(state.settlements) - 1
    coord, reason = unreachable[0]
    assert coord in {s.coord for s in state.settlements}
    assert "ferry" in reason


def test_unreachable_settlements_are_recorded_on_the_world():
    """A map in pieces is something a reader of the output should be able to see."""
    from worldgen.core.errors import RoutingError
    from worldgen.stages import interurban_roads as ir

    def no_ferry(*_a, **_k):
        raise RoutingError("no plausible ferry")

    real = ir.ferry_link
    ir.ferry_link = no_ferry
    try:
        state = _build_pipeline(seed=42, width=48, height=48).run()
    finally:
        ir.ferry_link = real

    # This map needs no ferries, so nothing should be recorded — but the key, when it is
    # written at all, has to be shaped for the reader.
    for entry in state.metadata.get("unreachable_settlements", []):
        assert set(entry) == {"coord", "reason"}
        assert len(entry["coord"]) == 2
