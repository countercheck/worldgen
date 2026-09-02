from collections import deque

from ..core.errors import RoutingError
from ..core.hex import TerrainClass
from ..core.hex_grid import astar, distance, neighbors
from ..core.world_state import Ferry

_WATER = (TerrainClass.OCEAN, TerrainClass.LAKE)


def edge_grade_pct(from_hx, to_hx, cfg) -> float:
    """Percent grade between two adjacent hexes."""
    delta = abs(to_hx.elevation - from_hx.elevation)
    return delta * cfg.road_elev_range_m * 100.0 / cfg.hex_size_m


def grade_is_under_cap(from_hx, to_hx, cfg) -> bool:
    """True when edge grade is below the configured slope cap threshold."""
    return edge_grade_pct(from_hx, to_hx, cfg) < cfg.road_slope_cap_pct


def max_grade_cap_delta(cfg) -> float:
    """Elevation delta equivalent to the slope cap, for fast per-edge comparisons
    (avoids repeating the grade_is_under_cap division/multiplication per edge)."""
    return cfg.road_slope_cap_pct * cfg.hex_size_m / (cfg.road_elev_range_m * 100.0)


def slope_edge_cost(from_hx, to_hx, cfg) -> float:
    """Grade-aware edge penalty for road pathfinding."""
    grade_pct = edge_grade_pct(from_hx, to_hx, cfg)
    if grade_pct <= cfg.road_slope_free_pct:
        return 0.0
    if grade_pct >= cfg.road_slope_cap_pct:
        return cfg.road_slope_cost * cfg.road_slope_cap_mult
    raw = (
        cfg.road_slope_cost
        * (grade_pct - cfg.road_slope_free_pct)
        / (cfg.road_slope_cap_pct - grade_pct)
    )
    return min(raw, cfg.road_slope_cost * cfg.road_slope_cap_mult)


def terrain_base_cost(hx, cfg) -> float:
    """Base node cost by terrain class.

    Water (OCEAN/LAKE) returns the small `road_water_cost` rather than infinity;
    this lets pathfinding traverse water bodies as a single piece of terrain
    where embark/disembark costs (charged on edges) dominate the journey.
    """
    tc = hx.terrain_class
    if tc in _WATER:
        return cfg.road_water_cost
    if tc == TerrainClass.MOUNTAIN:
        return cfg.road_mountain_cost
    if tc == TerrainClass.HILL:
        return cfg.road_hill_cost
    return cfg.road_flat_cost


def bank_discount(hx, hexes, cfg) -> float:
    """Scaled along-river discount, applied to the *bank* rather than the channel.

    Roads follow river valleys, but a road drawn down the channel itself hides which
    side of the river it — and anything standing on it — is on.  So the pull lives on
    the land hexes beside the river instead: a road runs along the bank, and the side it
    takes is a fact about the world rather than an accident of rendering.

    Scaled by the largest adjacent river's flow (bigger river → bigger pull), with the
    same `min_flow` floor as before so small headwater rivers keep a usable discount.
    River hexes themselves get nothing; visiting one is a crossing, not a route.
    """
    if hx.river_flow > 0:
        return 0.0
    flow = 0.0
    for n in neighbors(hx.coord):
        n_hx = hexes.get(n)
        if n_hx is not None and n_hx.river_flow > flow:
            flow = n_hx.river_flow
    if flow <= 0:
        return 0.0
    return cfg.road_bank_discount * max(flow, cfg.road_bank_discount_min_flow)


def river_hex_cost(hx, cfg) -> float:
    """Penalty for a road standing on a river hex rather than beside it.

    The channel exclusion in `make_road_edge_cost` only covers hexsides a river is
    actually drawn along.  A meander doubling back, or two braids running side by side,
    puts river hexes adjacent without such a hexside — and a road threading those is
    still in the water, with no bank to be on.  Charging the hex itself closes that gap:
    a crossing pays it once and stays affordable, travelling the channel pays it every
    step and stops being worth it.
    """
    return cfg.road_river_hex_cost if hx.river_flow > 0 else 0.0


def water_edge_cost(from_hx, to_hx, cfg) -> float:
    """Embark/disembark cost for transitions between land and water hexes."""
    from_water = from_hx.terrain_class in _WATER
    to_water = to_hx.terrain_class in _WATER
    if from_water == to_water:
        return 0.0
    return cfg.road_embark_cost if to_water else cfg.road_disembark_cost


def river_crossing_edge_cost(from_hx, to_hx, cfg) -> float:
    """Penalty on each land↔river edge, scaled by the larger river_flow.

    A perpendicular crossing of a 1-hex-wide river hits this twice (entering
    and leaving), so the configured base+flow values represent half of the
    total perpendicular crossing cost.
    """
    from_river = from_hx.river_flow > 0
    to_river = to_hx.river_flow > 0
    if from_river == to_river:
        return 0.0
    flow = max(from_hx.river_flow, to_hx.river_flow)
    return cfg.road_river_crossing_base + cfg.road_river_crossing_flow * flow


def road_edge_cost(from_hx, to_hx, cfg) -> float:
    """Combined edge-cost: slope + water embark/disembark + river crossing."""
    return (
        slope_edge_cost(from_hx, to_hx, cfg)
        + water_edge_cost(from_hx, to_hx, cfg)
        + river_crossing_edge_cost(from_hx, to_hx, cfg)
    )


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
    if a in settled and b_hx is not None and b_hx.river_flow <= 0:
        return True
    return b in settled and a_hx is not None and a_hx.river_flow <= 0


def make_road_edge_cost(cfg, blocked_edges=None, exempt_coords=frozenset()):
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
        if blocked_edges and frozenset((from_hx.coord, to_hx.coord)) in blocked_edges:
            exempt = (
                from_hx.coord in exempt_coords
                and to_hx.river_flow <= 0
                or to_hx.coord in exempt_coords
                and from_hx.river_flow <= 0
            )
            if not exempt:
                return float("inf")
        return road_edge_cost(from_hx, to_hx, cfg)

    return edge_cost


def tag_river_crossings(roads, hexes) -> None:
    """Tag river-entry hexes on each road as ford → bridge on second visit.

    Mutates `hex.tags` in place. Purely cosmetic (used by renderers); does
    not feed back into pathfinding cost.
    """
    for road in roads:
        path = road.path
        for i, c in enumerate(path):
            if c not in hexes:
                continue
            hx = hexes[c]
            if "river" not in hx.tags:
                continue
            prev_c = path[i - 1] if i > 0 else None
            prev_hx = hexes.get(prev_c) if prev_c is not None else None
            if (prev_hx is None or "river" not in prev_hx.tags) and "bridge" not in hx.tags:
                if "ford" not in hx.tags:
                    hx.tags.add("ford")
                else:
                    hx.tags.discard("ford")
                    hx.tags.add("bridge")


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
            and hx.terrain_class not in _WATER
            and hx.river_flow <= 0
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
