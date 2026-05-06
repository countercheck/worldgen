import math
from dataclasses import dataclass, field
from pathlib import Path

from ..core.hex import SettlementTier
from ..core.hex_grid import axial_to_pixel, neighbors
from ..core.world_state import RoadTier, WorldState
from ..render.debug_viewer import BIOME_COLORS, LAND_COVER_COLORS, TERRAIN_COLORS


@dataclass
class SVGConfig:
    hex_size: float = 12.0
    padding: int = 20
    color_mode: str = "biome"  # "terrain" | "biome" | "land_cover" | "elevation"
    layers: set[str] = field(
        default_factory=lambda: {"terrain", "rivers", "roads", "settlements", "labels", "grid"}
    )
    style: str = "atlas"  # "atlas" | "topographic" | "wargame"
    contour_elevation_scale_m: float = 3000.0
    contour_min_m: float = 10.0
    contour_max_m: float = 300.0
    contour_max_stroke: float = 4.0


_ROAD_SVG = {
    RoadTier.PRIMARY: {"stroke": "#5c3d1e", "stroke-width": "2.0", "dasharray": None},
    RoadTier.SECONDARY: {"stroke": "#8b6914", "stroke-width": "1.2", "dasharray": None},
    RoadTier.TRACK: {"stroke": "#b8a070", "stroke-width": "0.6", "dasharray": "4 2"},
}


def _rgb_to_hex(r: float, g: float, b: float) -> str:
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def _hex_vertices(cx: float, cy: float, size: float) -> list[tuple[float, float]]:
    angles = [0, 60, 120, 180, 240, 300]
    return [
        (cx + size * math.cos(math.radians(a)), cy + size * math.sin(math.radians(a)))
        for a in angles
    ]


def _points_str(pts: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)


def _xml_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _get_hex_fill(h, color_mode: str) -> str:
    if color_mode == "terrain":
        rgb = TERRAIN_COLORS.get(h.terrain_class, (0.5, 0.5, 0.5))
    elif color_mode == "land_cover":
        rgb = (
            LAND_COVER_COLORS.get(h.land_cover, (0.5, 0.5, 0.5))
            if h.land_cover is not None
            else (0.5, 0.5, 0.5)
        )
    elif color_mode == "elevation":
        v = h.elevation
        rgb = (v, v, v)
    else:  # biome
        if h.biome is not None:
            rgb = BIOME_COLORS.get(h.biome, (0.5, 0.5, 0.5))
        else:
            rgb = TERRAIN_COLORS.get(h.terrain_class, (0.5, 0.5, 0.5))
    return _rgb_to_hex(*rgb[:3])


def _star_points(cx: float, cy: float, outer: float, inner: float, n: int = 5) -> str:
    pts = []
    for i in range(n * 2):
        r = outer if i % 2 == 0 else inner
        angle = math.radians(i * 180 / n - 90)
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    return _points_str(pts)


def render(ws: WorldState, config: SVGConfig | None = None) -> str:
    """Render WorldState as an SVG string."""
    if config is None:
        config = SVGConfig()

    if config.style == "topographic":
        color_mode = "elevation"
        layers = {"terrain", "rivers", "grid", "contours"}
    elif config.style == "wargame":
        color_mode = "terrain"
        layers = {"terrain", "roads", "settlements", "grid"}
    else:
        color_mode = config.color_mode
        layers = config.layers

    size = config.hex_size
    pad = config.padding

    all_pixels = [axial_to_pixel(coord, size) for coord in ws.hexes]
    if not all_pixels:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="0" height="0"></svg>'

    min_x = min(p[0] for p in all_pixels) - size
    min_y = min(p[1] for p in all_pixels) - size
    max_x = max(p[0] for p in all_pixels) + size
    max_y = max(p[1] for p in all_pixels) + size

    ox = -min_x + pad
    oy = -min_y + pad
    w = math.ceil(max_x - min_x + 2 * pad)
    h = math.ceil(max_y - min_y + 2 * pad)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
    ]

    if "terrain" in layers:
        out.append('  <g id="layer-terrain">')
        for hex_item in ws.hexes.values():
            px, py = axial_to_pixel(hex_item.coord, size)
            verts = _hex_vertices(px + ox, py + oy, size)
            fill = _get_hex_fill(hex_item, color_mode)
            out.append(f'    <polygon points="{_points_str(verts)}" fill="{fill}" stroke="none"/>')
        out.append("  </g>")

    if "grid" in layers:
        grid_lw = "2.0" if config.style == "wargame" else "0.5"
        out.append('  <g id="layer-grid">')
        for hex_item in ws.hexes.values():
            px, py = axial_to_pixel(hex_item.coord, size)
            verts = _hex_vertices(px + ox, py + oy, size)
            out.append(
                f'    <polygon points="{_points_str(verts)}" fill="none" stroke="#555555" stroke-width="{grid_lw}"/>'
            )
        out.append("  </g>")

    if "contours" in layers:
        scale = config.contour_elevation_scale_m
        min_m = config.contour_min_m
        max_m = config.contour_max_m
        max_stroke = config.contour_max_stroke
        if max_m <= min_m:
            raise ValueError(
                f"contour_max_m ({max_m!r}) must be greater than contour_min_m ({min_m!r})"
            )
        out.append('  <g id="layer-contours">')
        for coord, hex_item in ws.hexes.items():
            ca = axial_to_pixel(coord, size)
            for nbr_coord in neighbors(coord):
                if nbr_coord < coord:
                    continue
                nbr = ws.hexes.get(nbr_coord)
                if nbr is None:
                    continue
                diff_m = abs(hex_item.elevation - nbr.elevation) * scale
                if diff_m < min_m:
                    continue
                t = min((diff_m - min_m) / (max_m - min_m), 1.0)
                stroke = 0.3 + t * (max_stroke - 0.3)
                cb = axial_to_pixel(nbr_coord, size)
                mx = (ca[0] + cb[0]) / 2 + ox
                my = (ca[1] + cb[1]) / 2 + oy
                dx = cb[0] - ca[0]
                dy = cb[1] - ca[1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist == 0:
                    continue
                px = -dy / dist
                py = dx / dist
                half = size / 2
                x1, y1 = mx + px * half, my + py * half
                x2, y2 = mx - px * half, my - py * half
                out.append(
                    f'    <line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}"'
                    f' stroke="#333333" stroke-width="{stroke:.2f}" stroke-linecap="round"/>'
                )
        out.append("  </g>")

    if "rivers" in layers:
        out.append('  <g id="layer-rivers">')
        for river in ws.rivers:
            if len(river.hexes) < 2:
                continue
            pts = []
            for coord in river.hexes:
                px, py = axial_to_pixel(coord, size)
                pts.append((px + ox, py + oy))
            sw = max(0.5, min(4.0, river.flow_volume * 2))
            out.append(
                f'    <polyline points="{_points_str(pts)}" fill="none" stroke="#3a78c9"'
                f' stroke-width="{sw:.2f}" stroke-linecap="round" stroke-linejoin="round"/>'
            )
        out.append("  </g>")

    if "roads" in layers:
        out.append('  <g id="layer-roads">')
        for road in ws.roads:
            if len(road.path) < 2:
                continue
            pts = []
            for coord in road.path:
                px, py = axial_to_pixel(coord, size)
                pts.append((px + ox, py + oy))
            style = _ROAD_SVG[road.tier]
            da = f' stroke-dasharray="{style["dasharray"]}"' if style["dasharray"] else ""
            out.append(
                f'    <polyline points="{_points_str(pts)}" fill="none" stroke="{style["stroke"]}"'
                f' stroke-width="{style["stroke-width"]}" stroke-linecap="round"'
                f' stroke-linejoin="round"{da}/>'
            )
        out.append("  </g>")

    if "settlements" in layers:
        out.append('  <g id="layer-settlements">')
        for s in ws.settlements:
            px, py = axial_to_pixel(s.coord, size)
            cx, cy = px + ox, py + oy
            if s.tier == SettlementTier.CITY:
                pts = _star_points(cx, cy, outer=6.0, inner=2.5)
                out.append(
                    f'    <polygon points="{pts}" fill="gold" stroke="black" stroke-width="0.8"/>'
                )
            elif s.tier == SettlementTier.TOWN:
                r = 3.5
                out.append(
                    f'    <rect x="{cx - r:.2f}" y="{cy - r:.2f}" width="{2 * r:.2f}" height="{2 * r:.2f}"'
                    f' fill="white" stroke="black" stroke-width="0.8"/>'
                )
            else:
                out.append(
                    f'    <circle cx="{cx:.2f}" cy="{cy:.2f}" r="2.5" fill="white" stroke="black" stroke-width="0.8"/>'
                )
        out.append("  </g>")

    if "labels" in layers:
        out.append('  <g id="layer-labels" font-family="sans-serif" font-size="7" fill="black">')
        for s in ws.settlements:
            px, py = axial_to_pixel(s.coord, size)
            cx, cy = px + ox, py + oy - size - 2
            out.append(
                f'    <text x="{cx:.2f}" y="{cy:.2f}" text-anchor="middle">{_xml_escape(s.name)}</text>'
            )
        out.append("  </g>")

    out.append("</svg>")
    return "\n".join(out)


def save(ws: WorldState, path, config: SVGConfig | None = None) -> None:
    """Write SVG hex map to a file."""
    Path(path).write_text(render(ws, config), encoding="utf-8")
