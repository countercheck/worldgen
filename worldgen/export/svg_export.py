import math
from dataclasses import dataclass, field
from pathlib import Path

from ..core.hex import SettlementTier
from ..core.hex_grid import axial_to_pixel, neighbors, split_path_on_water
from ..core.world_state import RoadTier, WorldState
from ..render.debug_viewer import BIOME_COLORS, LAND_COVER_COLORS, TERRAIN_COLORS
from . import legend


@dataclass
class SVGConfig:
    hex_size: float = 12.0
    padding: int = 20
    color_mode: str = "biome"  # "terrain" | "biome" | "land_cover" | "elevation"
    layers: set[str] = field(
        default_factory=lambda: {
            "terrain",
            "rivers",
            "roads",
            "settlements",
            "labels",
            "grid",
            "legend",
        }
    )
    style: str = "atlas"  # "atlas" | "topographic" | "wargame"
    contour_elevation_scale_m: float = 3000.0
    contour_interval_m: float = 100.0
    contour_max_crossings: int = 5
    contour_max_stroke: float = 4.0
    legend_corner: str = "top-right"  # "top-right" | "bottom-left"
    legend_scale: float = 1.0  # multiplier on hex_size for legend glyph/text size


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


def _settlement_marker(tier: SettlementTier, cx: float, cy: float, scale: float = 1.0) -> str:
    """One settlement symbol centred on (cx, cy).

    Shared by the settlements layer and the legend so the two can never drift apart.
    *scale* multiplies the symbol geometry only — the hairline stroke stays constant so
    small legend glyphs keep a crisp outline.
    """
    if tier == SettlementTier.CITY:
        pts = _star_points(cx, cy, outer=6.0 * scale, inner=2.5 * scale)
        return f'<polygon points="{pts}" fill="gold" stroke="black" stroke-width="0.8"/>'
    if tier == SettlementTier.TOWN:
        r = 3.5 * scale
        return (
            f'<rect x="{cx - r:.2f}" y="{cy - r:.2f}" width="{2 * r:.2f}" height="{2 * r:.2f}"'
            f' fill="white" stroke="black" stroke-width="0.8"/>'
        )
    return (
        f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{2.5 * scale:.2f}"'
        f' fill="white" stroke="black" stroke-width="0.8"/>'
    )


def _legend_glyph(row: legend.LegendRow, cx: float, cy: float, g: float, color_mode: str) -> str:
    """SVG markup for one legend row's symbol, in a square box of side *g* centred on (cx, cy)."""
    if row.kind == "ramp":
        sw = g / len(legend.ELEVATION_RAMP)
        cells = "".join(
            f'<rect x="{cx - g / 2 + i * sw:.2f}" y="{cy - g / 4:.2f}"'
            f' width="{sw:.2f}" height="{g / 2:.2f}"'
            f' fill="{_rgb_to_hex(v, v, v)}" stroke="none"/>'
            for i, v in enumerate(legend.ELEVATION_RAMP)
        )
        return (
            f'{cells}<rect x="{cx - g / 2:.2f}" y="{cy - g / 4:.2f}"'
            f' width="{g:.2f}" height="{g / 2:.2f}"'
            f' fill="none" stroke="#555555" stroke-width="0.5"/>'
        )

    if row.kind == "fill":
        verts = _hex_vertices(cx, cy, g / 2)
        return (
            f'<polygon points="{_points_str(verts)}" fill="{_get_hex_fill(row.sample, color_mode)}"'
            f' stroke="#555555" stroke-width="0.5"/>'
        )

    if row.kind == "settlement":
        # Same helper as the settlements layer, so the symbols can never diverge.
        return _settlement_marker(row.sample, cx, cy, scale=g / legend.SYMBOL_BOX)

    if row.kind == "road":
        style = _ROAD_SVG[row.sample]
        stroke, width, dash = style["stroke"], style["stroke-width"], style["dasharray"]
    else:  # "river" — matches the rivers layer's colour
        stroke, width, dash = "#3a78c9", "2.0", None
    da = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{cx - g / 2:.2f}" y1="{cy:.2f}" x2="{cx + g / 2:.2f}" y2="{cy:.2f}"'
        f' stroke="{stroke}" stroke-width="{width}" stroke-linecap="round"{da}/>'
    )


def _legend_metrics(config: SVGConfig, rows: list[legend.LegendRow]) -> legend.Metrics:
    """Panel geometry from an advance-width estimate.

    Measured against DejaVu Sans, the widest legend labels come in around 0.55 em/char;
    0.6 leaves headroom for a wider substitute font rather than letting text spill past
    the panel edge.  (The PNG exporter measures glyphs exactly instead.)
    """
    font = config.hex_size * config.legend_scale * 0.8
    return legend.metrics(
        config.hex_size,
        config.legend_scale,
        len(rows),
        label_w=max(len(row.label) for row in rows) * font * 0.6,
        title_w=len("Legend") * font * 0.62,
    )


def _legend_svg(
    ws: WorldState,
    config: SVGConfig,
    rows: list[legend.LegendRow],
    m: legend.Metrics,
    color_mode: str,
    ox: float,
    oy: float,
    w: int,
    h: int,
) -> list[str]:
    """Legend panel tucked into one of the two empty corners of the canvas.

    Row selection, panel geometry and placement live in `legend`; this only draws them.
    """
    x, y = legend.placement(
        ws,
        config.hex_size,
        config.padding,
        config.legend_corner,
        ox,
        oy,
        w,
        h,
        m.width,
        m.height,
        margin=m.glyph,
        axial_to_pixel=axial_to_pixel,
    )

    out = [
        '  <g id="layer-legend">',
        f'    <rect x="{x:.2f}" y="{y:.2f}" width="{m.width:.2f}" height="{m.height:.2f}"'
        f' rx="{m.glyph / 3:.2f}"'
        f' fill="#ffffff" fill-opacity="0.92" stroke="#333333" stroke-width="1"/>',
        f'    <text x="{x + m.inner:.2f}" y="{y + m.inner + m.font:.2f}" font-family="sans-serif"'
        f' font-size="{m.font:.2f}" font-weight="bold" fill="black">Legend</text>',
    ]
    for i, row in enumerate(rows):
        cy = y + m.inner + m.title_h + i * m.row_h + m.row_h / 2
        out.append(f"    {_legend_glyph(row, x + m.inner + m.glyph / 2, cy, m.glyph, color_mode)}")
        out.append(
            f'    <text x="{x + m.inner + m.glyph + m.gap:.2f}" y="{cy + m.font * 0.35:.2f}"'
            f' font-family="sans-serif" font-size="{m.font:.2f}" fill="black">'
            f"{_xml_escape(row.label)}</text>"
        )
    out.append("  </g>")
    return out


def render(ws: WorldState, config: SVGConfig | None = None) -> str:
    """Render WorldState as an SVG string."""
    if config is None:
        config = SVGConfig()

    if config.style == "topographic":
        color_mode = "elevation"
        layers = {"terrain", "rivers", "grid", "contours", "legend"}
    elif config.style == "wargame":
        color_mode = "terrain"
        layers = {"terrain", "roads", "settlements", "grid", "legend"}
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

    # Size the legend before the canvas: on a small map the panel can be larger than the
    # map itself, and clamping alone would just crop it. Grow the canvas to fit instead.
    legend_rows: list[legend.LegendRow] = []
    legend_m: legend.Metrics | None = None
    if "legend" in layers:
        legend.validate(config.legend_corner, config.legend_scale)
        legend_rows = legend.rows(ws, color_mode, layers)
        if legend_rows:
            legend_m = _legend_metrics(config, legend_rows)
            w = max(w, math.ceil(legend_m.width + 2 * pad))
            h = max(h, math.ceil(legend_m.height + 2 * pad))

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
        interval = config.contour_interval_m
        max_n = config.contour_max_crossings
        max_stroke = config.contour_max_stroke
        if interval <= 0:
            raise ValueError(f"contour_interval_m must be positive, got {interval!r}")
        if max_n <= 0:
            raise ValueError(f"contour_max_crossings must be positive, got {max_n!r}")
        out.append('  <g id="layer-contours">')
        for coord, hex_item in ws.hexes.items():
            ca = axial_to_pixel(coord, size)
            for nbr_coord in neighbors(coord):
                if nbr_coord < coord:
                    continue
                nbr = ws.hexes.get(nbr_coord)
                if nbr is None:
                    continue
                lo_m = min(hex_item.elevation, nbr.elevation) * scale
                hi_m = max(hex_item.elevation, nbr.elevation) * scale
                n = int(hi_m / interval) - int(lo_m / interval)
                if n <= 0:
                    continue
                t = 1.0 if max_n == 1 else min((n - 1) / (max_n - 1), 1.0)
                stroke = 0.3 + t * (max_stroke - 0.3)
                v = round(187 * (1 - t) + 17 * t)
                color = f"#{v:02x}{v:02x}{v:02x}"
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
                    f' stroke="{color}" stroke-width="{stroke:.2f}" stroke-linecap="round"/>'
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
            style = _ROAD_SVG[road.tier]
            da = f' stroke-dasharray="{style["dasharray"]}"' if style["dasharray"] else ""
            for leg in split_path_on_water(road.path, ws.hexes):
                pts = []
                for coord in leg:
                    px, py = axial_to_pixel(coord, size)
                    pts.append((px + ox, py + oy))
                out.append(
                    f'    <polyline points="{_points_str(pts)}" fill="none"'
                    f' stroke="{style["stroke"]}" stroke-width="{style["stroke-width"]}"'
                    f' stroke-linecap="round" stroke-linejoin="round"{da}/>'
                )
        out.append("  </g>")

    if "settlements" in layers:
        out.append('  <g id="layer-settlements">')
        for s in ws.settlements:
            px, py = axial_to_pixel(s.coord, size)
            out.append(f"    {_settlement_marker(s.tier, px + ox, py + oy)}")
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

    # Legend last so it paints over anything that reaches into the corner.
    if legend_m is not None:
        out.extend(_legend_svg(ws, config, legend_rows, legend_m, color_mode, ox, oy, w, h))

    out.append("</svg>")
    return "\n".join(out)


def save(ws: WorldState, path, config: SVGConfig | None = None) -> None:
    """Write SVG hex map to a file."""
    Path(path).write_text(render(ws, config), encoding="utf-8")
