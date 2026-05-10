from ..core.hex import TerrainClass

_WATER = (TerrainClass.OCEAN, TerrainClass.LAKE)


def edge_grade_pct(from_hx, to_hx, cfg) -> float:
    """Percent grade between two adjacent hexes."""
    delta = abs(to_hx.elevation - from_hx.elevation)
    return delta * cfg.road_elev_range_m * 100.0 / cfg.hex_size_m


def grade_is_under_cap(from_hx, to_hx, cfg) -> bool:
    """True when edge grade is below the configured slope cap threshold."""
    return edge_grade_pct(from_hx, to_hx, cfg) < cfg.road_slope_cap_pct


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


def river_discount(hx, cfg) -> float:
    """Scaled along-river discount: bigger river → bigger pull on routes.

    Multiplier on `cfg.road_river_discount` is `max(river_flow, min_flow)` so
    that small headwater rivers retain a usable discount even when their
    normalised flow is barely above the river threshold.
    """
    if hx.river_flow <= 0:
        return 0.0
    flow = max(hx.river_flow, cfg.road_river_discount_min_flow)
    return cfg.road_river_discount * flow


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
            if hx.river_flow == 0:
                continue
            prev_c = path[i - 1] if i > 0 else None
            prev_hx = hexes.get(prev_c) if prev_c is not None else None
            if prev_hx is None or prev_hx.river_flow == 0:
                if "ford" not in hx.tags:
                    hx.tags.add("ford")
                else:
                    hx.tags.discard("ford")
                    hx.tags.add("bridge")
