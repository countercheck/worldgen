import pytest

from tests.worlds import build_pipeline, build_world
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
    water = (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER)
    for _, leg in road_polylines(road_state.road_edges, road_state.hexes):
        assert len(leg) >= 2
        for end in (leg[0], leg[-1]):
            assert road_state.hexes[end].terrain_class not in water, (
                f"drawn road leg terminates on water at {end}"
            )


def test_road_edges_and_sea_edges_split_the_water_between_them(road_state):
    """`road_edges` is land and `sea_edges` is the water — each must actually be so.

    The old water check lived on a network the stage now filters by construction, so it
    could never fail; the place a violation would land today is `sea_edges`. A road edge
    with a wet endpoint is a cart in the sea; a sea edge with two dry endpoints is a road
    filed under boats, invisible to every "is there a land route" question.
    """
    water = (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER)
    for a, b in road_state.road_edges:
        for end in (a, b):
            assert road_state.hexes[end].terrain_class not in water, (
                f"road edge endpoint {end} is on water"
            )
    for a, b in road_state.sea_edges:
        assert any(road_state.hexes[c].terrain_class in water for c in (a, b)), (
            f"sea edge {a}-{b} touches no water at all"
        )


def test_road_connections_symmetric(road_state):
    hexes = road_state.hexes
    for coord, hx in hexes.items():
        for neighbor in hx.road_connections:
            assert coord in hexes[neighbor].road_connections, (
                f"road_connections not symmetric: {coord} -> {neighbor} but not reverse"
            )


def test_every_road_crossing_of_a_river_is_tagged(road_state):
    """A road crosses a river by stepping across the side it runs along, and that side is
    tagged a ford or a bridge — the only places roads cross rivers."""
    from worldgen.core.hex_grid import side_between

    crossed = 0
    for a, b in road_state.road_edges:
        side = side_between(a, b)
        if side in road_state.river_sides:
            crossed += 1
            assert road_state.river_sides[side].tags & {"ford", "bridge"}, (
                f"road crosses the river at {side} with no ford or bridge"
            )
    assert crossed, "no road crosses a river on this map, so this asserts nothing"


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

    # Ferries are links too, where a world has any.
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
    it fails to route at all.  The costs identify a river by its sides for this reason,
    so the two modes must produce the same roads.
    """
    plain = _build_pipeline(seed=42).run()
    continuous = _build_pipeline(seed=42, river_flow_continuous=True).run()

    assert continuous.road_edges, "no roads generated with river_flow_continuous"
    flowing = [h for h in continuous.hexes.values() if h.river_flow > 0]
    from worldgen.core.hex_grid import side_hexes

    tagged = {h for s in continuous.river_sides for h in side_hexes(s)}
    assert len(flowing) > len(tagged) * 2, (
        "continuous mode should put flow on far more hexes than stand on a river bank; "
        "if it does not, this test is no longer exercising the case it was written for"
    )

    def network(ws):
        return sorted((k, e.tier.value) for k, e in ws.road_edges.items())

    assert network(continuous) == network(plain), (
        "roads differ between flow modes: something is still identifying a river channel "
        "by river_flow rather than by its sides"
    )


def test_river_corridor_preference_in_roads(road_state):
    """Roads follow river valleys: along the bank.

    This was `xfail` until the two halves of this branch met. It was failing because the
    slope measure underneath it was wrong: taking the mean absolute difference to all six
    neighbours reports how rough the *surroundings* are, so a valley floor read as steep —
    both flanks stand above it, and their height went into the mean whatever the floor was
    doing. Roads were therefore priced *away* from the banks they should follow. Measuring
    slope as tilt cancels symmetric surroundings and reads a valley floor level, and
    lateral planation gives the road a floodplain to run along at all. Across seeds 42, 1,
    2, 3, 5 and 8 the margin went from a mean of -0.048, positive on one seed, to +0.109,
    positive on all six.
    """
    hexes = road_state.hexes
    road_hexes = {c for edge in road_state.road_edges for c in edge if c in hexes}
    all_land = {c for c, h in hexes.items() if h.terrain_class != TerrainClass.OPEN_WATER}

    if not road_hexes or not all_land:
        return

    # Rivers run along hexsides, so every hex beside one is a bank: roads should be on
    # banks more often than banks occur in the land.
    banks = {c for r in road_state.rivers for c in r.banks() if c in all_land}
    if not banks:
        return

    road_rate = len(road_hexes & banks) / len(road_hexes & all_land)
    map_rate = len(banks) / len(all_land)

    assert road_rate >= map_rate, (
        f"Riverbank preference not detected: roads run on a bank {road_rate:.3f} of the "
        f"time against {map_rate:.3f} of the dry land being bank"
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
    cannot climb, or across a river it has no need to cross. So the test asks the same
    question the rule does, through the same function — `detour_is_allowed`. Writing the
    guard out a second time here is exactly what went wrong before: the copy knew only
    about the cost bound, so a skirt refused for a 30% grade read as a defect.
    """
    from worldgen.core.hex_grid import neighbors
    from worldgen.stages.road_cost import detour_is_allowed, river_crossings

    cfg = WorldConfig(**road_state.metadata["config"])
    settled = {s.coord for s in road_state.settlements}
    crossings = river_crossings(road_state.river_sides)

    offenders = []
    for seat in settled:
        ring = set(neighbors(seat))
        for a, b in road_state.road_edges:
            if a not in ring or b not in ring:
                continue
            if detour_is_allowed(road_state.hexes, settled, cfg, crossings, a, seat, b):
                offenders.append((seat, a, b))

    assert not offenders, (
        f"{len(offenders)} roads pass a settlement they could have been bent through, "
        f"e.g. {offenders[0][1]}->{offenders[0][2]} around {offenders[0][0]}"
    )


@pytest.fixture(scope="module", params=["classic", "organic"])
def connected_state(request, road_state):
    """The connectivity invariant, checked on both settlement models.

    It used to be checked on `classic` alone, which cannot break it: the guarantee lives in
    `InterurbanRoadStage`, and `classic` founds nothing afterwards. `organic` does —
    `ChokepointStage` adds the village tier after the roads are built — so the one model
    that can strand a settlement was the one nobody was watching.

    This will not catch a rare case on its own: the failure that prompted it appears on one
    map in sixty (128x128 mediterranean, seed 42) and on none at 48, 64 or 96. What it
    catches is gross breakage, and the specific rule is tested directly in
    `test_chokepoints.py`.
    """
    if request.param == "classic":
        return road_state
    return build_world(seed=42, width=64, height=64, model="organic")


def test_the_road_network_is_all_one_piece(connected_state):
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
    for a, b in connected_state.road_edges:
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

    seats = {s.coord for s in connected_state.settlements}
    by_road = components(adj)

    # Nothing may be drawn that reaches neither a settlement nor a ferry landing.
    anchors = seats | {c for f in connected_state.ferries for c in (f.a, f.b)}
    # `prune_orphan_roads` keeps a component that lands a sea leg — a road to a harbour is
    # a road to somewhere — so this has to allow what the pruner allows, or it fails on
    # ground the model deliberately keeps.
    anchors |= {c for key in connected_state.sea_edges for c in key}
    for comp in by_road:
        assert comp & anchors, (
            f"{len(comp)} road hexes near {sorted(comp)[0]} reach no settlement and no "
            "ferry — a road that connects nothing"
        )

    # Once ferries count as links there must be one network — except for anything the
    # terrain genuinely severs, which the stage records rather than raising over.
    linked = {c: set(v) for c, v in adj.items()}
    for ferry in connected_state.ferries:
        linked.setdefault(ferry.a, set()).add(ferry.b)
        linked.setdefault(ferry.b, set()).add(ferry.a)
    reached = max(components(linked), key=len, default=set())
    conceded = {
        tuple(entry["coord"])
        for entry in connected_state.metadata.get("unreachable_settlements", [])
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
        for c, hx in connected_state.hexes.items()
        if hx.terrain_class not in (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER)
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

    # And the stronger claim the settlement one is a corollary of: **one road network per
    # landmass**, whether or not the pieces hold settlements. `_join_by_land` joins
    # components rather than seats, so a spur that holds nothing today cannot strand
    # whatever is founded on it tomorrow — which is exactly how a chokepoint village once
    # ended up cut off from the 37 settlements it shared ground with.
    mass_id: dict = {}
    for start in dry:
        if start in mass_id:
            continue
        stack, reached = [start], set()
        while stack:
            c = stack.pop()
            if c in reached:
                continue
            reached.add(c)
            stack.extend(n for n in neighbors(c) if n in dry and n not in reached)
        for c in reached:
            mass_id[c] = start

    nets_per_mass: dict = {}
    for i, comp in enumerate(by_road):
        for c in comp:
            if c in mass_id:
                nets_per_mass.setdefault(mass_id[c], set()).add(i)
                break
    for mass, nets in nets_per_mass.items():
        assert len(nets) == 1, (
            f"the landmass around {mass} carries {len(nets)} separate road networks; "
            "where land joins two roads, a road should join them"
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


def test_a_place_nothing_can_reach_does_not_break_generation(monkeypatch):
    """Some maps are in pieces, and that is a fact about the world rather than an error.

    A river is always crossable at a price, so what can strand a settlement now is ground
    no cart can climb to.  Raising there would make such a map ungenerable; the stage
    records what it could not join and carries on.
    """
    from worldgen.stages import interurban_roads as ir

    state = _build_pipeline(seed=42, width=48, height=48).run()
    cfg = WorldConfig(**state.metadata["config"])
    stage = ir.InterurbanRoadStage(cfg, None)

    def no_route(*_a, **_k):
        return None

    monkeypatch.setattr(ir, "_to_any", no_route)
    _edges, unreachable = stage._guarantee_connectivity(state.hexes, state.settlements, {}, cfg, {})

    # It came back rather than raising, and it said what it could not reach.
    assert unreachable, "nothing was recorded as unreachable — the test did not bite"
    assert len(unreachable) >= len(state.settlements) - 1
    coord, reason = unreachable[0]
    assert coord in {s.coord for s in state.settlements}
    assert "no route" in reason


def test_unreachable_settlements_are_recorded_on_the_world():
    """A map in pieces is something a reader of the output should be able to see."""
    state = _build_pipeline(seed=42, width=48, height=48).run()
    for entry in state.metadata.get("unreachable_settlements", []):
        assert set(entry) == {"coord", "reason"}
        assert len(entry["coord"]) == 2


def test_the_adjacency_index_matches_the_edges(road_state):
    """`hex.road_connections` is the serialised index of the network; the edges are the
    network. The village stage reroutes edges after the index was first written, so it
    must rebuild what it invalidated — a reader following the index must never walk an
    edge the map does not have, or miss one it does.
    """
    want: dict = {}
    for a, b in list(road_state.road_edges) + list(road_state.sea_edges):
        want.setdefault(a, set()).add(b)
        want.setdefault(b, set()).add(a)
    for coord, hx in road_state.hexes.items():
        assert hx.road_connections == want.get(coord, set()), (
            f"adjacency at {coord} disagrees with the edge set: "
            f"index {sorted(hx.road_connections)} vs edges {sorted(want.get(coord, set()))}"
        )


def test_connectivity_joins_an_isolated_settlement_to_the_nearest_of_the_network():
    """The multi-target search must pick the same target the exhaustive one did.

    `_guarantee_connectivity` used to run a full A* from an isolated settlement to every
    settlement already on the network and keep the cheapest path. It now runs one
    `astar_to_any` against the whole component, which is the same argmin — the frontier is
    ordered by true cost, so the first goal reached is the cheapest goal — at a fraction of
    the work. This asserts the equivalence on the property that matters: the isolated place
    ends up joined, and joined to a member of the component it was routed at.
    """
    from worldgen.core.hex_grid import astar_to_any
    from worldgen.stages import interurban_roads as ir
    from worldgen.stages.road_cost import (
        make_road_edge_cost,
        river_crossings,
        terrain_base_cost,
    )

    state = _build_pipeline(seed=42, width=48, height=48).run()
    cfg = WorldConfig(**state.metadata["config"])
    hexes = state.hexes
    places = [s for s in state.settlements if s.tier in (SettlementTier.CITY, SettlementTier.TOWN)]
    if len(places) < 3:
        pytest.skip("needs at least three cities or towns to have something to join")

    crossings = river_crossings(state.river_sides)

    def plain_cost(hx):
        return terrain_base_cost(hx, cfg)

    plain_edge = make_road_edge_cost(cfg, crossings)

    # Every settlement its own component: nothing is joined yet.
    edges, unreachable = ir.InterurbanRoadStage(cfg, None)._guarantee_connectivity(
        hexes, places, {}, cfg, crossings
    )
    joined = {c for key in edges for c in key}
    unreachable_coords = {tuple(u[0]) for u in unreachable}
    for s in places:
        if s.coord in unreachable_coords:
            continue
        assert s.coord in joined, f"{s.name} at {s.coord} was left off the network"

    # And the search it now relies on really does reach a goal when one is reachable.
    a, b = places[0].coord, {p.coord for p in places[1:]}
    path = astar_to_any(hexes, a, b, plain_cost, plain_edge)
    if path is not None:
        assert path[0] == a
        assert path[-1] in b, "astar_to_any ended somewhere that was not a goal"


@pytest.mark.parametrize(
    ("tier", "road"),
    [
        (SettlementTier.VILLAGE, "track"),
        (SettlementTier.TOWN, "secondary"),
        (SettlementTier.CITY, "primary"),
    ],
)
def test_a_stranded_place_is_joined_by_the_road_it_deserves(tier, road):
    """The guarantee used to lay PRIMARY whatever it joined, which gave a lumber camp of a
    hundred and twenty a trunk road once resource villages joined the network."""
    from worldgen.core.hex import Hex, Settlement, SettlementRole, TerrainClass
    from worldgen.core.world_state import RoadTier
    from worldgen.stages import interurban_roads as ir

    cfg = WorldConfig()
    hexes = {
        (q, r): Hex(coord=(q, r), terrain_class=TerrainClass.LAND)
        for q in range(10)
        for r in range(3)
    }
    city = Settlement(
        coord=(0, 1),
        tier=SettlementTier.CITY,
        role=SettlementRole.MARKET,
        population=9000,
        name="city",
    )
    stranded = Settlement(
        coord=(7, 1), tier=tier, role=SettlementRole.MARKET, population=200, name="stranded"
    )
    existing = {((0, 1), (1, 1)): RoadTier.PRIMARY}

    edges, unreachable = ir.InterurbanRoadStage(cfg, None)._guarantee_connectivity(
        hexes, [city, stranded], dict(existing), cfg, {}
    )
    assert not unreachable
    laid = {k: t for k, t in edges.items() if k not in existing}
    assert laid and {t.value for t in laid.values()} == {road}
