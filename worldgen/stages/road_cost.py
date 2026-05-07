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
