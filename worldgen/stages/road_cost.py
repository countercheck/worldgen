from ..core.hex import TerrainClass
from ..core.hex_grid import neighbors, side_hexes
from ..core.world_state import ROAD_TIER_RANK, RoadTier, road_edge_key
from .riverside import side_gradients, side_span

WATER = (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER)


def delta_elevation(from_hx, to_hx) -> float:
    """Height gained from *from_hx* to *to_hx*, signed: negative where the road falls.

    One definition, so the number a `RoadEdge` carries and the number the cost is charged on
    cannot drift apart. The cost takes the absolute value; the sign is kept for the reader.
    """
    return to_hx.elevation - from_hx.elevation


def edge_grade_pct(from_hx, to_hx, cfg) -> float:
    """Percent grade between two adjacent hexes."""
    return abs(delta_elevation(from_hx, to_hx)) * 100.0 / cfg.hex_size_m


def grade_is_under_cap(from_hx, to_hx, cfg) -> bool:
    """True when edge grade is below the configured slope cap threshold."""
    return edge_grade_pct(from_hx, to_hx, cfg) < cfg.road_slope_cap_pct


def max_grade_cap_delta(cfg) -> float:
    """Elevation delta equivalent to the slope cap, for fast per-edge comparisons
    (avoids repeating the grade_is_under_cap division/multiplication per edge)."""
    return cfg.road_slope_cap_pct * cfg.hex_size_m / 100.0


def slope_edge_cost(from_hx, to_hx, cfg) -> float:
    """What the climb costs, as hexes of level going — the switchback, priced.

    At 1 hex = 1 km a road climbing 200 m is not a straight ramp; it is several kilometres
    of zigzag folded inside that hex. `road_delta_elevation_per_hex` is the exchange rate that says
    so, and the cost is continuous in the height difference rather than banded.

    Charged on the *absolute* height difference, so a descent costs exactly what the same
    climb would. That is the difference from `travel_ascent_per_hex`: a walker pays for the
    climb alone (Naismith), while a road is cut-and-fill and a steep descent needs braking
    and washes out.

    Above `road_slope_cap_pct` the edge is refused outright — a laden cart cannot climb 25%,
    and it should not be offered the option at a price. The curve this replaced saturated
    there instead, so a road met a 65% face, paid a flat twenty for it, and went straight up.

    Water pays nothing here. A boat notices the sea floor's gradient not at all, and
    charging it did two wrong things at once: every sea leg paid for the bathymetry under
    it, and a shelf dropping faster than the cap made a strait *impassable* — a cliff
    a keel never touches. `water_edge_cost` prices getting on and off the water; the
    water itself is level by definition.
    """
    if from_hx.terrain_class in WATER or to_hx.terrain_class in WATER:
        return 0.0
    if not grade_is_under_cap(from_hx, to_hx, cfg):
        return float("inf")
    return abs(delta_elevation(from_hx, to_hx)) / cfg.road_delta_elevation_per_hex


def terrain_base_cost(hx, cfg) -> float:
    """Base node cost by terrain class.

    Water (OCEAN/LAKE) returns the small `road_water_cost` rather than infinity;
    this lets pathfinding traverse water bodies as a single piece of terrain
    where embark/disembark costs (charged on edges) dominate the journey.

    A settlement hex costs nothing — it is a road segment carrying the most travellers
    there can be. Cleared ground, a bridge already built, an inn: passing through a town
    is easier than passing beside it, and a route should be drawn to one from a couple of
    hexes out rather than have to be bent into it afterwards.
    """
    if hx.settlement is not None:
        return 0.0
    tc = hx.terrain_class
    if tc in WATER:
        return cfg.road_water_cost
    # No steepness surcharge here.  The climb between two hexes is already priced on the
    # edge, from the actual metres of rise, so banding the hex and charging again billed
    # the same ascent twice — and billed it wrongly where the band and the grade
    # disagreed, taxing a level valley floor at escarpment rates because the bluff above
    # it was in the averaging window.
    return cfg.road_flat_cost


def water_edge_cost(from_hx, to_hx, cfg) -> float:
    """Embark/disembark cost for transitions between land and water hexes."""
    from_water = from_hx.terrain_class in WATER
    to_water = to_hx.terrain_class in WATER
    if from_water == to_water:
        return 0.0
    return cfg.road_embark_cost if to_water else cfg.road_disembark_cost


def river_crossings(river_sides) -> dict[frozenset, float]:
    """Every pair of hexes a river runs between, mapped to that river's flow there.

    Rivers run along hexsides, so a road crosses one exactly when it steps between the two
    hexes either side of a river side.  Keyed by the unordered pair so an edge cost can
    look it up from either end.
    """
    return {frozenset(side_hexes(side)): rs.flow for side, rs in river_sides.items()}


def river_crossing_edge_cost(from_hx, to_hx, cfg, crossings=None) -> float:
    """What it costs a road to cross a river: charged once, on the side the river runs along.

    A fixed term for the crossing itself — the ford's approaches or the bridge's
    abutments — and a term scaled by the river's flow, since a bigger river is a longer
    span.  *crossings* is `river_crossings` of the world's river sides; without it no step
    crosses anything.
    """
    if not crossings:
        return 0.0
    flow = crossings.get(frozenset((from_hx.coord, to_hx.coord)))
    if flow is None:
        return 0.0
    return cfg.road_river_crossing_base + cfg.road_river_crossing_flow * flow


def settlement_skirt_cost(from_hx, to_hx, cfg, ring) -> float:
    """What it costs to pass a town at one hex without going in.

    *ring* maps a hex to the seats it neighbours, so a shared entry means both ends of this
    edge touch the same settlement: the road enters the ring and leaves without arriving.

    This is the half of the settlement pull that works at one hex. A discount on the town
    itself cannot: the direct route and the detour both pay for the same two ring hexes, so
    the detour costs exactly what the town costs on top, and driving that to zero makes the
    detour a *tie* rather than a win. Ties are settled by heap order. Charging the skirt is
    what actually shifts the route.
    """
    if not ring:
        return 0.0
    shared = ring.get(from_hx.coord)
    if not shared or not (shared & ring.get(to_hx.coord, frozenset())):
        return 0.0
    return cfg.road_settlement_skirt_cost


def road_edge_cost(from_hx, to_hx, cfg, ring=None, crossings=None) -> float:
    """Combined edge-cost: slope + water embark/disembark + river crossing + town skirt."""
    return (
        slope_edge_cost(from_hx, to_hx, cfg)
        + water_edge_cost(from_hx, to_hx, cfg)
        + river_crossing_edge_cost(from_hx, to_hx, cfg, crossings)
        + settlement_skirt_cost(from_hx, to_hx, cfg, ring)
    )


def settlement_rings(seats) -> dict:
    """Hex -> the settlement seats it neighbours, for `settlement_skirt_cost`."""
    out: dict = {}
    for seat in seats:
        for n in neighbors(seat):
            out.setdefault(n, set()).add(seat)
    return {k: frozenset(v) for k, v in out.items()}


def make_road_edge_cost(cfg, crossings=None, ring=None):
    """Edge-cost closure over `road_edge_cost`, with the world's river crossings bound in.

    There is no longer anything a road may not do at a river.  When rivers occupied hexes a
    road could run down the channel, and had to be forbidden from it, so that the bank it
    was on stayed readable; a river along a hexside has no channel to run down, and a road
    beside it is simply on one bank.  Crossing is a step across the side, priced by
    `river_crossing_edge_cost`.
    """

    def edge_cost(from_hx, to_hx) -> float:
        return road_edge_cost(from_hx, to_hx, cfg, ring, crossings)

    return edge_cost


def tag_switchbacks(road_edges, hexes, cfg) -> None:
    """Mark road hexes whose grade is steep enough that the road must double back on itself.

    The cost model already charges for it — `slope_edge_cost` converts the climb into the
    level-going it really represents — but nothing in the output said so, and at this scale
    nothing can be drawn: a switchback is a hundred-metre feature and a hex is a kilometre.
    The tag is how a reader of the map, or a wargame counting movement, knows the segment is
    slow.

    Mutates `hex.tags` in place, like `tag_river_crossings`.
    """
    for a, b in road_edges:
        ha, hb = hexes.get(a), hexes.get(b)
        if ha is None or hb is None:
            continue
        if edge_grade_pct(ha, hb, cfg) >= cfg.road_switchback_grade_pct:
            ha.tags.add("switchback")
            hb.tags.add("switchback")


def tag_river_crossings(road_edges, state, cfg) -> None:
    """Tag the river sides the road network crosses: a bridge, or a ford where it can wade.

    Mutates the sides' tags in place.  What the water is decides it, not how busy the road
    is: a road over a river too big to wade (`side_span` over 1) needs a bridge whatever
    its tier — a track over a navigable river is not a track through it — and one over
    water that can be waded uses the ford, unless it is a primary road, which is worth a
    bridge anyway.  The result does not depend on the order routes were built in.

    `CrossingStage` (the organic model) tags its own fords and bridges before roads exist.
    A bridge a road crosses is left alone, since it is not demoted by carrying a quiet
    road.  A bridge no road crosses is untagged: `CrossingStage` marks where traffic would
    justify one, so markets can grow there, and a site the road network never reached was
    never built.  A ford stays either way — it is terrain, and needs nobody to build it.
    """
    side_of = {frozenset(side_hexes(side)): side for side in state.river_sides}
    best: dict = {}
    for key, tier in road_edges.items():
        side = side_of.get(frozenset(key))
        if side is None:
            continue
        if side not in best or ROAD_TIER_RANK[tier] > ROAD_TIER_RANK[best[side]]:
            best[side] = tier
    gradient = side_gradients(state) if best else {}
    for side, tier in best.items():
        rs = state.river_sides[side]
        if "bridge" in rs.tags:
            continue
        wadeable = "ford" in rs.tags or (
            side_span(rs.catchment_km2, gradient.get(side, 0.0), cfg) <= 1.0
        )
        if tier is RoadTier.PRIMARY or not wadeable:
            rs.tags.discard("ford")
            rs.tags.add("bridge")
        else:
            rs.tags.add("ford")
    for side, rs in state.river_sides.items():
        if side not in best:
            rs.tags.discard("bridge")


def pheromone_discount(base: float, traffic: float, cfg) -> float:
    """What a hex costs a traveller once earlier travellers have worn a path across it.

    Extracted so the shape can be measured.  It was inline in `InterurbanRoadStage`, and
    `_guarantee_city_connectivity`'s `plain_cost` quietly used a different formula.
    """
    return max(0.0, base - cfg.road_pheromone_factor * traffic)


def _keep_higher_tier(existing, incoming):
    """Merge rule for a tier map: a road laid here is never demoted by one bent onto it."""
    if existing is None or ROAD_TIER_RANK[incoming] > ROAD_TIER_RANK[existing]:
        return incoming
    return existing


def add_traffic(existing, incoming):
    """Merge rule for a traffic map: two roads meeting carry the sum of what each carried.

    This is why consolidation happens *before* tiering rather than after. Bending a bypass
    through a town merges two flows onto one pair of edges, and the merged edge should be
    ranked on what it now carries — two secondary roads meeting can make a primary. Taking
    the higher of two tiers after the fact cannot express that; adding the traffic and
    then cutting the percentiles does it for nothing.
    """
    return incoming if existing is None else existing + incoming


def detour_is_allowed(hexes, settled, cfg, crossings, a, seat, b) -> bool:
    """Could a road skirting *seat* on the edge (a, b) be bent through it instead?

    Two things forbid it, and they are why "no road skirts a town" is not an invariant on
    its own: the legs may not climb a grade a laden cart cannot, and may not cost more than
    `road_settlement_detour_max_mult` times the edge they replace — a road is allowed to
    decline a town that is dear to reach, and a river to cross is part of that cost.

    Split out so the rule has one statement. It was written twice, once here and once in the
    test that guards it, and the copies disagreed: the test knew only about the cost bound,
    so a skirt refused for a 30% grade or a channel crossing read as a defect.
    """

    # A road that already arrives at a settlement is not passing one by. Without this, two
    # settlements side by side bent a road back and forth for ever: routing it through the
    # one laid a leg skirting the other, and routing that through the other laid a leg
    # skirting the first. Every leg a bend lays ends at a settlement, so with this rule one
    # pass is final.
    if a in settled or b in settled:
        return False

    def leg_cost(start, end) -> float:
        return terrain_base_cost(hexes[end], cfg) + road_edge_cost(
            hexes[start], hexes[end], cfg, crossings=crossings
        )

    legs = ((a, seat), (seat, b))
    for start, end in legs:
        if end not in hexes or start not in hexes:
            return False
        if not grade_is_under_cap(hexes[start], hexes[end], cfg):
            return False
    direct = leg_cost(a, b)
    detour = leg_cost(a, seat) + leg_cost(seat, b)
    return not (direct > 0 and detour > direct * cfg.road_settlement_detour_max_mult)


def route_through_settlements(
    road_edges, hexes, settled, cfg, crossings=None, combine=_keep_higher_tier
) -> int:
    """Bend any road skirting a settlement so that it passes through it instead.

    A road whose two ends are both neighbours of a town enters the ring around it and
    leaves again without arriving — at 1 hex = 1 km, a trunk road passing a market town at
    the width of one field. Bypasses are a motor-age idea; before that the road went
    through the town, which is half the reason the town is where it is.

    So the edge (a, b) is replaced by (a, s) and (s, b): one hex longer, and the traffic
    now calls. Where those two already exist the bypass was pure redundancy and simply
    goes. A replacement may not climb a grade a laden cart cannot, or cost more than
    `road_settlement_detour_max_mult` times the edge
    it replaces — a road is allowed to decline a town that is dear to reach.

    *road_edges* maps an edge to whatever *combine* knows how to merge: traffic before
    tiering (`add_traffic`), or tiers after it (the default, which never demotes). Mutates
    it in place; returns how many bypasses were rerouted.
    """

    rerouted = 0
    for seat in settled:
        seat_hx = hexes.get(seat)
        if seat_hx is None or seat_hx.terrain_class in WATER:
            continue
        ring = set(neighbors(seat))
        for a, b in [(a, b) for a, b in road_edges if a in ring and b in ring]:
            if not detour_is_allowed(hexes, settled, cfg, crossings, a, seat, b):
                continue
            legs = ((a, seat), (seat, b))
            carried = road_edges.pop(road_edge_key(a, b))
            for start, end in legs:
                key = road_edge_key(start, end)
                road_edges[key] = combine(road_edges.get(key), carried)
            rerouted += 1
    return rerouted


def as_road_edges(tiers, hexes) -> dict:
    """Turn a key -> tier map into key -> `RoadEdge`, measuring each edge as it goes.

    The stages build with bare tiers because that is all the routing and tidying passes
    need. This is the one place the delta is measured, so the number a world carries is the
    number `slope_edge_cost` charged on.
    """
    from ..core.world_state import RoadEdge

    out = {}
    for (a, b), tier in tiers.items():
        ha, hb = hexes.get(a), hexes.get(b)
        delta = delta_elevation(ha, hb) if ha is not None and hb is not None else 0.0
        out[(a, b)] = RoadEdge(tier, delta)
    return out


def tier_near(road_edges, node, hops) -> RoadTier:
    """The highest tier of any road within *hops* edges of *node*, along the roads.

    TRACK where no road comes that close. Walked along the network rather than measured as
    the crow flies, so a trunk on the far side of a ridge is not "near".
    """
    adj: dict = {}
    for a, b in road_edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    best = RoadTier.TRACK
    seen, frontier = {node}, [node]
    for _ in range(hops):
        nxt = []
        for c in frontier:
            for n in adj.get(c, ()):
                tier = road_edges[road_edge_key(c, n)]
                if ROAD_TIER_RANK[tier] > ROAD_TIER_RANK[best]:
                    best = tier
                if n not in seen:
                    seen.add(n)
                    nxt.append(n)
        frontier = nxt
    return best


def fill_tier_gaps(road_edges, max_edges) -> int:
    """Promote a short lower-class stretch that a road of one class stops and resumes across.

    Tiers are cut per edge on traffic, so where two routes take neighbouring hexes for a
    step and then rejoin, the traffic splits between the branches and neither branch makes
    the cut: a primary road runs to a point, turns secondary for a kilometre, and carries on
    primary. Nobody builds a road like that.

    A gap is a path of at most *max_edges* lower-tier edges from an **end** of the tier —
    a node where exactly one road of that tier stops — to any part of that tier's network it
    is not already joined to. The other side may be an end too, the dip proper, or the
    middle of another road, where one trunk stops a hex short of meeting a second. Those two
    conditions are what separate a gap from a real junction. A secondary road leaving the
    middle of one trunk for the middle of another starts from no end, so it is a connector
    and keeps its class; and a lane between two points of the same primary road is a
    shortcut the traffic declined, not a break in it.

    Primary first, then secondary, so a secondary gap is judged on the primary network as
    it will be drawn. Mutates *road_edges* (key -> tier); returns how many edges changed.
    """
    if max_edges <= 0:
        return 0

    promoted = 0
    for tier in (RoadTier.PRIMARY, RoadTier.SECONDARY):
        rank = ROAD_TIER_RANK[tier]
        adj: dict = {}
        for a, b in road_edges:
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)

        def of_tier(a, b, rank=rank):
            return ROAD_TIER_RANK[road_edges[road_edge_key(a, b)]] >= rank

        # Union-find over the tier's own network, so a join made here counts for the next.
        parent: dict = {}

        def find(n, parent=parent):
            parent.setdefault(n, n)
            while parent[n] != n:
                parent[n] = parent[parent[n]]
                n = parent[n]
            return n

        on_tier = set()
        for a, b in road_edges:
            if of_tier(a, b):
                on_tier |= {a, b}
                parent[find(a)] = find(b)
        ends = sorted(n for n in on_tier if sum(of_tier(n, m) for m in adj[n]) == 1)
        end_set = set(ends)

        for start in ends:
            if start not in end_set:
                continue
            # Breadth-first over lower-tier edges, so the shortest gap is the one filled;
            # neighbours sorted so the same network always fills the same way.
            prev = {start: None}
            frontier = [start]
            found = None
            for _ in range(max_edges):
                nxt = []
                for c in frontier:
                    for n in sorted(adj[c]):
                        if n in prev or of_tier(c, n):
                            continue
                        prev[n] = c
                        if n in on_tier:
                            # Reaching the tier ends the search either way: another
                            # piece of it is the gap closed, and its own piece is a
                            # junction the path may not pass through.
                            if find(n) != find(start):
                                found = n
                                break
                            continue
                        nxt.append(n)
                    if found:
                        break
                if found or not nxt:
                    break
                frontier = nxt
            if found is None:
                continue
            c = found
            while prev[c] is not None:
                road_edges[road_edge_key(c, prev[c])] = tier
                promoted += 1
                c = prev[c]
            parent[find(found)] = find(start)
            end_set -= {start, found}

    return promoted


def prune_orphan_roads(road_edges, anchors) -> int:
    """Drop any part of the network that connects nothing.

    `road_river_traffic_min` admits a riverbank edge on a single traveller, so a stretch of
    towpath can qualify without joining anything — a five-hex road in the middle of a
    valley, reaching no settlement and no ferry. That is not a road, it is a residue of the
    threshold, and it is what leaves the network in more than one piece.

    *anchors* is what makes a component worth keeping: settlement seats, and the landings
    of any ferry, since a component reachable only by boat is legitimately separate on land.

    Mutates *road_edges*; returns how many edges were dropped.
    """
    adj: dict = {}
    for a, b in road_edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    seen: set = set()
    doomed: set = set()
    for start in adj:
        if start in seen:
            continue
        stack, comp = [start], set()
        while stack:
            c = stack.pop()
            if c in comp:
                continue
            comp.add(c)
            stack.extend(adj[c] - comp)
        seen |= comp
        if not (comp & anchors):
            doomed |= comp

    if not doomed:
        return 0
    dropped = [k for k in road_edges if k[0] in doomed or k[1] in doomed]
    for k in dropped:
        del road_edges[k]
    return len(dropped)
