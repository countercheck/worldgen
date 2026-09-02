import math
from dataclasses import dataclass, field
from pathlib import Path

from ..core.hex import SettlementTier
from ..core.hex_grid import axial_to_pixel, dedupe_road_paths, neighbors
from ..core.world_state import ROAD_TIER_RANK, RoadTier, WorldState
from ..render.debug_viewer import BIOME_COLORS, LAND_COVER_COLORS, TERRAIN_COLORS
from . import legend, rivers


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
            "anchorages",
            "crossings",
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
    # Rivers widen downstream with the flow in each hex, so a headwater is visibly not
    # the trunk it feeds. 0 tracks flow continuously; a positive value quantises into
    # that many discrete widths, giving the stepped look of a stream-order map. The
    # exponent shapes the curve — flow is power-law distributed, so mapping it linearly
    # (1.0) draws almost every river at the minimum width.
    river_min_width: float = 1.0
    river_max_width: float = 5.0
    river_width_steps: int = 0
    river_width_exponent: float = 0.5
    # Rivers and roads are the features a reader traces across the map, so they are drawn
    # twice: a dark casing first, then the line itself on top of it.  The casing is what
    # makes them legible over terrain of any colour — an unoutlined brown road vanishes
    # into scrub and a blue river into deep forest.  Set to 0.0 for flat, uncased lines.
    feature_outline: float = 0.9
    river_color: str = "#2f6fbf"
    river_casing_color: str = "#0e2a50"
    road_casing_color: str = "#20140a"


_ROAD_SVG = {
    RoadTier.PRIMARY: {"stroke": "#4a2f14", "stroke-width": "3.0", "dasharray": None},
    RoadTier.SECONDARY: {"stroke": "#7a5a10", "stroke-width": "2.0", "dasharray": None},
    RoadTier.TRACK: {"stroke": "#8a6a34", "stroke-width": "1.2", "dasharray": "4 2"},
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


def _anchorage_marker(cx: float, cy: float, scale: float = 1.0) -> str:
    """Anchor symbol marking where a road embarks onto, or lands from, water.

    Ring, stem, crossbar and flukes — drawn rather than lifted from a font so it stays
    legible at map scale and needs no external asset.
    """
    s = 5.0 * scale
    return (
        f'<g fill="none" stroke="#1b3a5c" stroke-width="{max(0.6, 0.28 * s):.2f}"'
        f' stroke-linecap="round">'
        f'<circle cx="{cx:.2f}" cy="{cy - 0.80 * s:.2f}" r="{0.26 * s:.2f}"/>'
        f'<line x1="{cx:.2f}" y1="{cy - 0.54 * s:.2f}" x2="{cx:.2f}" y2="{cy + 0.92 * s:.2f}"/>'
        f'<line x1="{cx - 0.60 * s:.2f}" y1="{cy - 0.26 * s:.2f}"'
        f' x2="{cx + 0.60 * s:.2f}" y2="{cy - 0.26 * s:.2f}"/>'
        f'<path d="M {cx - 0.72 * s:.2f},{cy + 0.40 * s:.2f}'
        f' Q {cx:.2f},{cy + 1.16 * s:.2f} {cx + 0.72 * s:.2f},{cy + 0.40 * s:.2f}"/>'
        f"</g>"
    )


_CROSSING_INK = "#2b2118"


def _crossing_marker(kind: str, cx: float, cy: float, angle: float, scale: float = 1.0) -> str:
    """A ford or bridge, laid across the river rather than along it.

    Bridge: two parallel decks with abutments at each end — the conventional map symbol.
    Ford: the same span broken into dashes, reading as a way through the water rather
    than over it.  *angle* is the river's bearing; the symbol is rotated square to it.
    """
    s = 5.0 * scale
    w = max(0.6, 0.26 * s)
    half, sep = 0.78 * s, 0.30 * s
    turn = f' transform="rotate({angle + 90:.1f} {cx:.2f} {cy:.2f})"'
    body = [f'<g fill="none" stroke="{_CROSSING_INK}" stroke-width="{w:.2f}"{turn}>']
    dash = ' stroke-dasharray="1.6 1.4"' if kind == "ford" else ""
    for side in (-1, 1):
        y = cy + side * sep
        body.append(
            f'<line x1="{cx - half:.2f}" y1="{y:.2f}" x2="{cx + half:.2f}" y2="{y:.2f}"{dash}/>'
        )
    if kind == "bridge":
        # Abutments: short uprights closing the deck at both banks.
        for side in (-1, 1):
            x = cx + side * half
            body.append(
                f'<line x1="{x:.2f}" y1="{cy - sep * 1.7:.2f}"'
                f' x2="{x:.2f}" y2="{cy + sep * 1.7:.2f}"/>'
            )
    body.append("</g>")
    return "".join(body)


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


def _legend_glyph(
    row: legend.LegendRow,
    cx: float,
    cy: float,
    g: float,
    color_mode: str,
    river_color: str = "#2f6fbf",
) -> str:
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

    if row.kind == "anchorage":
        return _anchorage_marker(cx, cy, scale=g / legend.SYMBOL_BOX)

    if row.kind in ("ford", "bridge"):
        # Angle 90 so the legend shows the span horizontally, as it reads on the map.
        return _crossing_marker(row.kind, cx, cy, -90.0, scale=g / legend.SYMBOL_BOX)

    # Legend line widths scale with the glyph box, as the legend's symbols already do.
    glyph_scale = g / legend.SYMBOL_BOX
    if row.kind == "road":
        style = _ROAD_SVG[row.sample]
        stroke, dash = style["stroke"], style["dasharray"]
        width = f"{float(style['stroke-width']) * glyph_scale:.2f}"
        if dash:
            dash = " ".join(f"{float(v) * glyph_scale:.2f}" for v in dash.split())
    else:  # "river" — the configured colour, so key and map can never diverge
        stroke, width, dash = river_color, f"{2.0 * glyph_scale:.2f}", None
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
        out.append(
            "    "
            + _legend_glyph(
                row, x + m.inner + m.glyph / 2, cy, m.glyph, color_mode, config.river_color
            )
        )
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
        # Wargame maps are read to move units, so the features that gate movement —
        # rivers and the fords and bridges over them — are as important as the roads.
        layers = {
            "terrain",
            "rivers",
            "roads",
            "settlements",
            "grid",
            "anchorages",
            "crossings",
            "legend",
        }
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

    # Line widths are written for the reference hex; scale them to this export.
    line_scale = legend.stroke_scale(size)

    # Size the legend before the canvas: on a small map the panel can be larger than the
    # map itself, and clamping alone would just crop it. Grow the canvas to fit instead.
    if "rivers" in layers:
        rivers.validate(
            config.river_min_width,
            config.river_max_width,
            config.river_width_steps,
            config.river_width_exponent,
        )

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
        # Every casing first, then every line.  Drawing each river's casing immediately
        # under its own line would let a later river's casing paint over an earlier
        # river's line wherever two run close together, which is exactly the confluence
        # the reader most needs to follow.
        bands = []
        for river in ws.rivers:
            # Banded by per-hex flow, so a river grows downstream instead of being drawn
            # at one width taken from its mouth.
            for run, sw in rivers.width_bands(
                river,
                ws.hexes,
                config.river_min_width,
                config.river_max_width,
                config.river_width_steps,
                config.river_width_exponent,
            ):
                pts = []
                for coord in run:
                    px, py = axial_to_pixel(coord, size)
                    pts.append((px + ox, py + oy))
                bands.append((pts, sw * line_scale))

        outline = config.feature_outline * line_scale
        if outline > 0.0:
            out.append('    <g id="rivers-casing">')
            for pts, sw in bands:
                out.append(
                    f'      <polyline points="{_points_str(pts)}" fill="none"'
                    f' stroke="{config.river_casing_color}"'
                    f' stroke-width="{sw + 2.0 * outline:.2f}"'
                    f' stroke-linecap="round" stroke-linejoin="round"/>'
                )
            out.append("    </g>")
        out.append('    <g id="rivers-line">')
        for pts, sw in bands:
            out.append(
                f'      <polyline points="{_points_str(pts)}" fill="none"'
                f' stroke="{config.river_color}"'
                f' stroke-width="{sw:.2f}"'
                f' stroke-linecap="round" stroke-linejoin="round"/>'
            )
        out.append("    </g>")
        out.append("  </g>")

    if "roads" in layers:
        out.append('  <g id="layer-roads">')
        # Deduped and ordered by tier, so shared trunk segments are drawn once and a
        # track never paints over the primary road it branches from.
        legs = []
        for road, leg in dedupe_road_paths(ws.roads, ws.hexes, lambda r: ROAD_TIER_RANK[r.tier]):
            style = _ROAD_SVG[road.tier]
            da = ""
            if style["dasharray"]:
                # The dash pattern scales too, or a track's dashes crowd into a solid
                # line on a large export and stretch to gaps on a small one.
                dashes = " ".join(
                    f"{float(v) * line_scale:.2f}" for v in style["dasharray"].split()
                )
                da = f' stroke-dasharray="{dashes}"'
            pts = []
            for coord in leg:
                px, py = axial_to_pixel(coord, size)
                pts.append((px + ox, py + oy))
            legs.append((pts, float(style["stroke-width"]) * line_scale, style["stroke"], da))

        outline = config.feature_outline * line_scale
        if outline > 0.0:
            # Casings for every road before any road line, for the same reason as the
            # rivers: at a junction the branch's casing must not cut the trunk in half.
            # The casing is solid even under a dashed track — a dashed outline would
            # leave the track's own gaps unoutlined and it would break up again.
            out.append('    <g id="roads-casing">')
            for pts, lw, _stroke, _da in legs:
                out.append(
                    f'      <polyline points="{_points_str(pts)}" fill="none"'
                    f' stroke="{config.road_casing_color}"'
                    f' stroke-width="{lw + 2.0 * outline:.2f}"'
                    f' stroke-linecap="round" stroke-linejoin="round"/>'
                )
            out.append("    </g>")
        out.append('    <g id="roads-line">')
        for pts, lw, stroke, da in legs:
            out.append(
                f'      <polyline points="{_points_str(pts)}" fill="none"'
                f' stroke="{stroke}"'
                f' stroke-width="{lw:.2f}"'
                f' stroke-linecap="round" stroke-linejoin="round"{da}/>'
            )
        out.append("    </g>")
        out.append("  </g>")

    if "crossings" in layers:
        marks = legend.crossings(ws, axial_to_pixel, size)
        if marks:
            out.append('  <g id="layer-crossings">')
            for coord, kind, angle in marks:
                px, py = axial_to_pixel(coord, size)
                out.append(
                    f"    {_crossing_marker(kind, px + ox, py + oy, angle, scale=size / 12.0)}"
                )
            out.append("  </g>")

    if "anchorages" in layers:
        # One marker per shore point, however many routes embark there. Covers both
        # sea legs and ferry landings — the same thing as far as a reader is concerned.
        points = legend.anchorage_points(ws)
        if points:
            out.append('  <g id="layer-anchorages">')
            for coord in points:
                px, py = axial_to_pixel(coord, size)
                out.append(f"    {_anchorage_marker(px + ox, py + oy, scale=size / 12.0)}")
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
