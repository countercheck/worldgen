def slope_edge_cost(from_hx, to_hx, cfg) -> float:
    """Grade-aware edge penalty for road pathfinding."""
    delta = abs(to_hx.elevation - from_hx.elevation)
    grade_pct = delta * cfg.road_elev_range_m * 100.0 / cfg.hex_size_m
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
