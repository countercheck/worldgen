from collections import deque

from ..core.errors import RoutingError
from ..core.hex import TerrainClass
from ..core.hex_grid import astar, distance, neighbors
from ..core.world_state import ROAD_TIER_RANK, Ferry, RoadTier, road_edge_key

WATER = (TerrainClass.OCEAN, TerrainClass.LAKE)


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
    if tc == TerrainClass.ESCARPMENT:
        return cfg.road_escarpment_cost
    if tc == TerrainClass.STEEP:
        return cfg.road_steep_cost
    if tc == TerrainClass.ROLLING:
        return cfg.road_rolling_cost
    return cfg.road_flat_cost


def is_river(hx) -> bool:
    """True when this hex is a river channel.

    The "river" tag, not `river_flow > 0`.  With `river_flow_continuous` the hydrology
    stage writes a flow value onto every draining land hex, so a flow test calls the
    whole map a river: every land-to-land edge reads as channel travel and prices at
    infinity, which leaves the road network unroutable rather than merely mis-costed.
    Everything here that asks "is this the channel?" wants the tag; only the terms that
    scale by *how much* water flows should read `river_flow` itself.
    """
    return "river" in hx.tags


def river_hex_cost(hx, cfg) -> float:
    """Penalty for a road standing on a river hex rather than beside it.

    The channel exclusion in `make_road_edge_cost` only covers hexsides a river is
    actually drawn along.  A meander doubling back, or two braids running side by side,
    puts river hexes adjacent without such a hexside — and a road threading those is
    still in the water, with no bank to be on.  Charging the hex itself closes that gap:
    a crossing pays it once and stays affordable, travelling the channel pays it every
    step and stops being worth it.
    """
    return cfg.road_river_hex_cost if is_river(hx) else 0.0


def water_edge_cost(from_hx, to_hx, cfg) -> float:
    """Embark/disembark cost for transitions between land and water hexes."""
    from_water = from_hx.terrain_class in WATER
    to_water = to_hx.terrain_class in WATER
    if from_water == to_water:
        return 0.0
    return cfg.road_embark_cost if to_water else cfg.road_disembark_cost


def river_crossing_edge_cost(from_hx, to_hx, cfg) -> float:
    """Penalty on each land↔river edge, scaled by the larger river_flow.

    A perpendicular crossing of a 1-hex-wide river hits this twice (entering
    and leaving), so the configured base+flow values represent half of the
    total perpendicular crossing cost.
    """
    from_river = is_river(from_hx)
    to_river = is_river(to_hx)
    if from_river == to_river:
        return 0.0
    flow = max(from_hx.river_flow, to_hx.river_flow)
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


def road_edge_cost(from_hx, to_hx, cfg, ring=None) -> float:
    """Combined edge-cost: slope + water embark/disembark + river crossing + town skirt."""
    return (
        slope_edge_cost(from_hx, to_hx, cfg)
        + water_edge_cost(from_hx, to_hx, cfg)
        + river_crossing_edge_cost(from_hx, to_hx, cfg)
        + settlement_skirt_cost(from_hx, to_hx, cfg, ring)
    )


def settlement_rings(seats) -> dict:
    """Hex -> the settlement seats it neighbours, for `settlement_skirt_cost`."""
    out: dict = {}
    for seat in seats:
        for n in neighbors(seat):
            out.setdefault(n, set()).add(seat)
    return {k: frozenset(v) for k, v in out.items()}


def river_edges(rivers) -> set[frozenset]:
    """Every hexside a river runs along, as unordered coord pairs.

    A river is stored as an ordered hex sequence and drawn as a polyline through those
    centres, so consecutive pairs are exactly the segments the map shows.  Two river
    hexes that merely touch — a meander doubling back, or two different rivers running
    side by side — are not one of these, and a road may use that hexside freely.
    """
    out: set[frozenset] = set()
    for river in rivers:
        for a, b in zip(river.hexes, river.hexes[1:], strict=False):
            out.add(frozenset((a, b)))
    return out


def _settlement_exempt(hexes, settled, a, b) -> bool:
    """True when a settlement lets a road use the channel hexside between *a* and *b*.

    Only far enough to reach the settlement: the counterpart must be dry land, never
    another river hex.  Otherwise a town on the water becomes a licence to carry on down
    the channel one hex at a time.
    """
    a_hx, b_hx = hexes.get(a), hexes.get(b)
    if a in settled and b_hx is not None and not is_river(b_hx):
        return True
    return b in settled and a_hx is not None and not is_river(a_hx)


def make_road_edge_cost(cfg, blocked_edges=None, exempt_coords=frozenset(), ring=None):
    """Edge-cost closure, optionally forbidding the hexsides a river runs along.

    Crossing a river stays legal and stays priced by `river_crossing_edge_cost` — what
    the exclusion forbids is travelling *along* the channel, which would draw the road on
    top of the river and leave the side it runs on undefined.

    Settlement hexes are exempt, but only far enough to be *reached*: the exemption holds
    when the settlement's counterpart is dry land, not when it is another river hex.  A
    riverside town is functionally a road hex that happens to sit on the water, so a road
    must be able to arrive at one; letting it leave along the channel as well would hand
    back the very thing the exclusion exists to prevent, one hex at a time.  Reaching a
    town never needs a channel hexside anyway — a town with no dry neighbour at all is
    the sealed-off case, and ferries cover that.

    `blocked_edges=None` (or an empty set) drops the channel exclusion entirely, leaving
    only the ordinary node and edge costs.  Every stage passes the river's hexsides; the
    default exists for callers pricing a route where the exclusion does not apply.
    """

    def edge_cost(from_hx, to_hx) -> float:
        # Any hexside joining two river hexes is channel travel, not just one that is a
        # segment of a single river's polyline.  Where a river meanders back on itself,
        # or two rivers run side by side, the two hexes are adjacent without being
        # consecutive on either path — and a road stepping between them still reads as
        # running down the water, which is the thing the exclusion exists to stop.
        along_channel = is_river(from_hx) and is_river(to_hx)
        if along_channel or (
            blocked_edges and frozenset((from_hx.coord, to_hx.coord)) in blocked_edges
        ):
            exempt = (
                from_hx.coord in exempt_coords
                and not is_river(to_hx)
                or to_hx.coord in exempt_coords
                and not is_river(from_hx)
            )
            if not exempt:
                return float("inf")
        return road_edge_cost(from_hx, to_hx, cfg, ring)

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


def tag_river_crossings(road_edges, hexes) -> None:
    """Tag river hexes the road network crosses: ford, or bridge where the road is primary.

    Mutates `hex.tags` in place. Purely cosmetic (used by renderers); does
    not feed back into pathfinding cost.

    It used to walk each route in turn and promote a ford to a bridge the second time a
    route entered the same river hex — "more than one road uses this, so build a bridge".
    With the network stored as edges there are no routes to count, and no need to: edge
    tier already *is* how busy a crossing is, taken from the same traffic the old
    second-visit rule was standing in for. A primary crossing gets a bridge, a quieter one
    a ford, and the result no longer depends on the order routes happened to be built in.

    `CrossingStage` (the organic model) tags its own fords and bridges before roads exist;
    those are left alone, since a bridge is not demoted by carrying a quiet road.
    """
    incident: dict = {}
    for (a, b), tier in road_edges.items():
        for coord in (a, b):
            hx = hexes.get(coord)
            if hx is None or "river" not in hx.tags:
                continue
            # Only a road arriving from off the channel is a crossing; one running along
            # the bank is not, and roads may not run down a channel in any case.
            other = b if coord == a else a
            other_hx = hexes.get(other)
            if other_hx is not None and "river" in other_hx.tags:
                continue
            best = incident.get(coord)
            if best is None or ROAD_TIER_RANK[tier] > ROAD_TIER_RANK[best]:
                incident[coord] = tier

    for coord, tier in incident.items():
        tags = hexes[coord].tags
        if "bridge" in tags:
            continue
        if tier is RoadTier.PRIMARY:
            tags.discard("ford")
            tags.add("bridge")
        else:
            tags.add("ford")


def reachable_under_constraint(hexes, start, blocked, settled) -> set:
    """Hexes reachable from *start* without travelling along a river channel.

    Only the channel exclusion restricts movement here — water is traversable for roads
    (see `terrain_base_cost`), so a component is genuinely separate only where a river's
    own hexsides form a complete cut.  That is the delta-island case ferries exist for.
    """
    seen = {start}
    queue = deque([start])
    while queue:
        c = queue.popleft()
        for n in neighbors(c):
            if n not in hexes or n in seen:
                continue
            if frozenset((c, n)) in blocked and not _settlement_exempt(hexes, settled, c, n):
                continue
            seen.add(n)
            queue.append(n)
    return seen


def ferry_link(hexes, origin, label, main, cfg, blocked, settled, plain_cost, plain_edge):
    """Shortest plausible boat hop from *origin*'s land component to the road network.

    Returns `(Ferry, [paths])` — the crossing plus any road needed to reach its landing
    on the origin's side.  Endpoints are chosen by hex distance over sorted candidates,
    so the result is reproducible from the seed.

    Both landings must be dry land off the channel: a ferry is drawn as a pair of
    anchorages, and an anchorage means a shore you can stand a road end on.  Roads
    traverse ocean and lake hexes, and river hexes are road hexes wherever a road crosses
    one, so both components hold candidates that would put an anchor mid-channel or out
    at sea.  They are filtered out here rather than left to the renderer.

    Raises `RoutingError` when the gap exceeds `road_ferry_max_hop`: at that width a
    ferry is not a plausible reading of the map, and a silent long-haul boat link would
    be worse than a loud failure.
    """
    near = reachable_under_constraint(hexes, origin, blocked, settled)
    far = main - near
    if not far:
        raise RoutingError(
            f"{label} at {origin} is cut off and the road network has no hex outside "
            "its own component to ferry to."
        )

    def landings(coords):
        """Candidate endpoints: dry land, off the channel, so an anchorage can sit there."""
        return sorted(
            c
            for c in coords
            if (hx := hexes.get(c)) is not None
            and hx.terrain_class not in WATER
            and not is_river(hx)
        )

    near_landings, far_landings = landings(near), landings(far)
    if not near_landings or not far_landings:
        side = "its own side" if not near_landings else "the road network's side"
        raise RoutingError(
            f"{label} at {origin} is cut off by a river channel and {side} offers no dry "
            "land off the channel to land a ferry on."
        )

    best = None
    for a in near_landings:
        for b in far_landings:
            d = distance(a, b)
            if d <= cfg.road_ferry_max_hop and (best is None or d < best[0]):
                best = (d, a, b)
    if best is None:
        raise RoutingError(
            f"{label} at {origin} is cut off by a river channel and the nearest "
            f"reachable road hex is further than road_ferry_max_hop "
            f"({cfg.road_ferry_max_hop} hexes); no plausible ferry exists."
        )

    _, a, b = best
    paths = []
    if a != origin:
        p = astar(hexes, origin, a, plain_cost, plain_edge)
        if not p:
            raise RoutingError(f"{label} at {origin} cannot reach its own ferry landing at {a}.")
        paths.append(p)
    return Ferry(a=a, b=b), paths


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


def detour_is_allowed(hexes, settled, cfg, blocked, a, seat, b) -> bool:
    """Could a road skirting *seat* on the edge (a, b) be bent through it instead?

    Three things forbid it, and they are why "no road skirts a town" is not an invariant on
    its own: the legs may not cross a river channel they have no business on, may not climb
    a grade a laden cart cannot, and may not cost more than `road_settlement_detour_max_mult`
    times the edge they replace — a road is allowed to decline a town that is dear to reach.

    Split out so the rule has one statement. It was written twice, once here and once in the
    test that guards it, and the copies disagreed: the test knew only about the cost bound,
    so a skirt refused for a 30% grade or a channel crossing read as a defect.
    """

    def leg_cost(start, end) -> float:
        return (
            terrain_base_cost(hexes[end], cfg)
            + river_hex_cost(hexes[end], cfg)
            + road_edge_cost(hexes[start], hexes[end], cfg)
        )

    legs = ((a, seat), (seat, b))
    for start, end in legs:
        if end not in hexes or start not in hexes:
            return False
        if frozenset((start, end)) in blocked and not _settlement_exempt(
            hexes, settled, start, end
        ):
            return False
        if not grade_is_under_cap(hexes[start], hexes[end], cfg):
            return False
    direct = leg_cost(a, b)
    detour = leg_cost(a, seat) + leg_cost(seat, b)
    return not (direct > 0 and detour > direct * cfg.road_settlement_detour_max_mult)


def route_through_settlements(
    road_edges, hexes, settled, cfg, blocked=frozenset(), combine=_keep_higher_tier
) -> int:
    """Bend any road skirting a settlement so that it passes through it instead.

    A road whose two ends are both neighbours of a town enters the ring around it and
    leaves again without arriving — at 1 hex = 1 km, a trunk road passing a market town at
    the width of one field. Bypasses are a motor-age idea; before that the road went
    through the town, which is half the reason the town is where it is.

    So the edge (a, b) is replaced by (a, s) and (s, b): one hex longer, and the traffic
    now calls. Where those two already exist the bypass was pure redundancy and simply
    goes. A replacement may not cross a river channel it has no business on, climb a grade
    a laden cart cannot, or cost more than `road_settlement_detour_max_mult` times the edge
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
            if not detour_is_allowed(hexes, settled, cfg, blocked, a, seat, b):
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
