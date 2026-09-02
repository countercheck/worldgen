import math
from dataclasses import dataclass, field

from PIL import Image, ImageDraw, ImageFont

from ..core.hex import SettlementTier
from ..core.hex_grid import axial_to_pixel, dedupe_road_paths, neighbors
from ..core.world_state import ROAD_TIER_RANK, RoadTier, WorldState
from ..render.debug_viewer import BIOME_COLORS, LAND_COVER_COLORS, TERRAIN_COLORS
from . import legend, rivers


@dataclass
class PNGConfig:
    hex_size: float = 12.0
    dpi: int = 150
    style: str = "atlas"  # "atlas" | "topographic" | "wargame"
    color_mode: str = "biome"  # "biome" | "terrain" | "land_cover" | "elevation"
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
    river_max_width: float = 4.0
    river_width_steps: int = 0
    river_width_exponent: float = 0.5


_ROAD_COLOR = {
    RoadTier.PRIMARY: (92, 61, 30),
    RoadTier.SECONDARY: (139, 105, 20),
    RoadTier.TRACK: (184, 160, 112),
}
_ROAD_WIDTH = {
    RoadTier.PRIMARY: 2,
    RoadTier.SECONDARY: 2,
    RoadTier.TRACK: 1,
}


def _rgb_int(r: float, g: float, b: float) -> tuple[int, int, int]:
    return (int(r * 255), int(g * 255), int(b * 255))


def _hex_verts(cx: float, cy: float, size: float) -> list[tuple[int, int]]:
    angles = [0, 60, 120, 180, 240, 300]
    return [
        (int(cx + size * math.cos(math.radians(a))), int(cy + size * math.sin(math.radians(a))))
        for a in angles
    ]


def _get_hex_fill(h, color_mode: str) -> tuple[int, int, int]:
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
    return _rgb_int(*rgb[:3])


def _star_pts(
    cx: float, cy: float, outer: float, inner: float, n: int = 5
) -> list[tuple[int, int]]:
    pts = []
    for i in range(n * 2):
        r = outer if i % 2 == 0 else inner
        angle = math.radians(i * 180 / n - 90)
        pts.append((int(cx + r * math.cos(angle)), int(cy + r * math.sin(angle))))
    return pts


def _draw_anchorage(draw: ImageDraw.ImageDraw, cx, cy, scale: float = 1.0):
    """Anchor symbol marking where a road embarks onto, or lands from, water.

    Ring, stem, crossbar and flukes — the same construction as the SVG exporter's.
    """
    s = 5.0 * scale
    color = (27, 58, 92)
    lw = max(1, round(0.28 * s))
    r = 0.26 * s
    draw.ellipse([cx - r, cy - 0.80 * s - r, cx + r, cy - 0.80 * s + r], outline=color, width=lw)
    draw.line([(cx, cy - 0.54 * s), (cx, cy + 0.92 * s)], fill=color, width=lw)
    draw.line(
        [(cx - 0.60 * s, cy - 0.26 * s), (cx + 0.60 * s, cy - 0.26 * s)], fill=color, width=lw
    )
    # PIL has no quadratic curve; the flukes are the lower half of an ellipse.
    draw.arc(
        [cx - 0.72 * s, cy - 0.30 * s, cx + 0.72 * s, cy + 1.00 * s],
        start=25,
        end=155,
        fill=color,
        width=lw,
    )


_CROSSING_INK = (43, 33, 24)


def _draw_crossing(draw: ImageDraw.ImageDraw, kind: str, cx, cy, angle: float, scale: float = 1.0):
    """A ford or bridge, laid across the river rather than along it.

    Same construction as the SVG exporter's. PIL cannot rotate a primitive, so the
    endpoints are rotated by hand about the centre.
    """
    s = 5.0 * scale
    lw = max(1, round(0.26 * s))
    half, sep = 0.78 * s, 0.30 * s
    rad = math.radians(angle + 90.0)
    cos_a, sin_a = math.cos(rad), math.sin(rad)

    def at(dx, dy):
        return (cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a)

    for side in (-1, 1):
        y = side * sep
        if kind == "ford":
            # Broken span: a way through the water rather than over it.
            steps = 5
            for i in range(steps):
                if i % 2:
                    continue
                x0 = -half + (2 * half) * i / steps
                x1 = -half + (2 * half) * (i + 1) / steps
                draw.line([at(x0, y), at(x1, y)], fill=_CROSSING_INK, width=lw)
        else:
            draw.line([at(-half, y), at(half, y)], fill=_CROSSING_INK, width=lw)

    if kind == "bridge":
        for side in (-1, 1):
            x = side * half
            draw.line([at(x, -sep * 1.7), at(x, sep * 1.7)], fill=_CROSSING_INK, width=lw)


def _draw_settlement(draw: ImageDraw.ImageDraw, tier: SettlementTier, cx, cy, scale: float = 1.0):
    """One settlement symbol centred on (cx, cy).

    Shared by the settlements layer and the legend so the two can never drift apart.
    """
    if tier == SettlementTier.CITY:
        draw.polygon(
            _star_pts(cx, cy, outer=6.0 * scale, inner=2.5 * scale),
            fill=(255, 215, 0),
            outline=(0, 0, 0),
        )
    elif tier == SettlementTier.TOWN:
        r = 4 * scale
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255), outline=(0, 0, 0))
    else:
        r = 3 * scale
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255), outline=(0, 0, 0))


def _dashed_line(draw: ImageDraw.ImageDraw, x1, y, x2, color, width, dash=4, gap=2):
    """Horizontal dashed run — PIL has no stroke-dasharray equivalent."""
    x = x1
    while x < x2:
        draw.line([(x, y), (min(x + dash, x2), y)], fill=color, width=width)
        x += dash + gap


def _draw_legend_glyph(draw: ImageDraw.ImageDraw, row, cx, cy, g: float, color_mode: str) -> None:
    """One legend row's symbol, in a square box of side *g* centred on (cx, cy)."""
    if row.kind == "ramp":
        sw = g / len(legend.ELEVATION_RAMP)
        for i, v in enumerate(legend.ELEVATION_RAMP):
            x0 = cx - g / 2 + i * sw
            draw.rectangle([x0, cy - g / 4, x0 + sw, cy + g / 4], fill=_rgb_int(v, v, v))
        draw.rectangle(
            [cx - g / 2, cy - g / 4, cx + g / 2, cy + g / 4], outline=(85, 85, 85), width=1
        )
    elif row.kind == "fill":
        draw.polygon(
            _hex_verts(cx, cy, g / 2),
            fill=_get_hex_fill(row.sample, color_mode),
            outline=(85, 85, 85),
        )
    elif row.kind == "settlement":
        _draw_settlement(draw, row.sample, cx, cy, scale=g / legend.SYMBOL_BOX)
    elif row.kind == "anchorage":
        _draw_anchorage(draw, cx, cy, scale=g / legend.SYMBOL_BOX)
    elif row.kind in ("ford", "bridge"):
        # Angle -90 so the legend shows the span horizontally, as it reads on the map.
        _draw_crossing(draw, row.kind, cx, cy, -90.0, scale=g / legend.SYMBOL_BOX)
    elif row.kind == "road":
        glyph_scale = g / legend.SYMBOL_BOX
        color = _ROAD_COLOR[row.sample]
        lw = max(1, round(_ROAD_WIDTH[row.sample] * glyph_scale))
        if row.sample == RoadTier.TRACK:
            _dashed_line(draw, cx - g / 2, cy, cx + g / 2, color, lw)
        else:
            draw.line([(cx - g / 2, cy), (cx + g / 2, cy)], fill=color, width=lw)
    else:  # "river" — matches the rivers layer's colour
        lw = max(1, round(2 * g / legend.SYMBOL_BOX))
        draw.line([(cx - g / 2, cy), (cx + g / 2, cy)], fill=(58, 120, 201), width=lw)


def _legend_font(config: PNGConfig):
    font_px = max(8, round(config.hex_size * config.legend_scale * 0.8))
    try:
        return ImageFont.load_default(size=font_px)
    except (TypeError, AttributeError, OSError):
        # Pillow < 10.1 has no scalable default; fall back to the fixed bitmap font.
        return ImageFont.load_default()


def _legend_metrics(config: PNGConfig, rows: list[legend.LegendRow], font) -> legend.Metrics:
    """Panel geometry from exact glyph measurement — PIL can measure, so no estimate."""
    scratch = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    def text_w(text: str) -> float:
        bbox = scratch.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0]

    return legend.metrics(
        config.hex_size,
        config.legend_scale,
        len(rows),
        label_w=max(text_w(row.label) for row in rows),
        title_w=text_w("Legend"),
    )


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    ws: WorldState,
    config: PNGConfig,
    rows: list[legend.LegendRow],
    m: legend.Metrics,
    font,
    color_mode: str,
    ox: float,
    oy: float,
    pad: float,
    width: int,
    height: int,
) -> None:
    """Legend panel tucked into one of the two empty corners of the canvas.

    Row selection, panel geometry and placement live in `legend`; this only draws them.
    """
    x, y = legend.placement(
        ws,
        config.hex_size,
        pad,
        config.legend_corner,
        ox,
        oy,
        width,
        height,
        m.width,
        m.height,
        margin=m.glyph,
        axial_to_pixel=axial_to_pixel,
    )

    draw.rectangle(
        [x, y, x + m.width, y + m.height], fill=(255, 255, 255), outline=(51, 51, 51), width=1
    )
    draw.text((x + m.inner, y + m.inner), "Legend", fill=(0, 0, 0), font=font)
    for i, row in enumerate(rows):
        cy = y + m.inner + m.title_h + i * m.row_h + m.row_h / 2
        _draw_legend_glyph(draw, row, x + m.inner + m.glyph / 2, cy, m.glyph, color_mode)
        bbox = draw.textbbox((0, 0), row.label, font=font)
        draw.text(
            (x + m.inner + m.glyph + m.gap, cy - (bbox[3] - bbox[1]) / 2 - bbox[1]),
            row.label,
            fill=(0, 0, 0),
            font=font,
        )


def render(ws: WorldState, config: PNGConfig | None = None) -> Image.Image:
    """Render WorldState as a PIL Image."""
    if config is None:
        config = PNGConfig()

    if config.style == "topographic":
        color_mode = "elevation"
        layers: set[str] = {"terrain", "rivers", "grid", "contours", "legend"}
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
    pad = 20

    all_pixels = [axial_to_pixel(coord, size) for coord in ws.hexes]
    if not all_pixels:
        return Image.new("RGB", (1, 1), (255, 255, 255))

    min_x = min(p[0] for p in all_pixels) - size
    min_y = min(p[1] for p in all_pixels) - size
    max_x = max(p[0] for p in all_pixels) + size
    max_y = max(p[1] for p in all_pixels) + size

    ox = -min_x + pad
    oy = -min_y + pad
    width = math.ceil(max_x - min_x + 2 * pad)
    height = math.ceil(max_y - min_y + 2 * pad)

    # Size the legend before the canvas: on a small map the panel can be larger than the
    # map itself, and clamping alone would just crop it. Grow the canvas to fit instead.
    # Line widths are written for the reference hex; scale them to this export.
    line_scale = legend.stroke_scale(size)

    if "rivers" in layers:
        rivers.validate(
            config.river_min_width,
            config.river_max_width,
            config.river_width_steps,
            config.river_width_exponent,
        )

    legend_rows: list[legend.LegendRow] = []
    legend_m: legend.Metrics | None = None
    legend_font = None
    if "legend" in layers:
        legend.validate(config.legend_corner, config.legend_scale)
        legend_rows = legend.rows(ws, color_mode, layers)
        if legend_rows:
            legend_font = _legend_font(config)
            legend_m = _legend_metrics(config, legend_rows, legend_font)
            width = max(width, math.ceil(legend_m.width + 2 * pad))
            height = max(height, math.ceil(legend_m.height + 2 * pad))

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    if "terrain" in layers:
        for hex_item in ws.hexes.values():
            px, py = axial_to_pixel(hex_item.coord, size)
            verts = _hex_verts(px + ox, py + oy, size)
            fill = _get_hex_fill(hex_item, color_mode)
            draw.polygon(verts, fill=fill)

    if "grid" in layers:
        grid_lw = 2 if config.style == "wargame" else 1
        for hex_item in ws.hexes.values():
            px, py = axial_to_pixel(hex_item.coord, size)
            verts = _hex_verts(px + ox, py + oy, size)
            draw.polygon(verts, outline=(80, 80, 80), width=grid_lw)

    if "contours" in layers:
        scale = config.contour_elevation_scale_m
        interval = config.contour_interval_m
        max_n = config.contour_max_crossings
        max_stroke = config.contour_max_stroke
        if interval <= 0:
            raise ValueError(f"contour_interval_m must be positive, got {interval!r}")
        if max_n <= 0:
            raise ValueError(f"contour_max_crossings must be positive, got {max_n!r}")
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
                stroke = max(1, round(0.3 + t * (max_stroke - 0.3)))
                v = round(187 * (1 - t) + 17 * t)
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
                x1, y1 = int(mx + px * half), int(my + py * half)
                x2, y2 = int(mx - px * half), int(my - py * half)
                draw.line([(x1, y1), (x2, y2)], fill=(v, v, v), width=stroke)

    if "rivers" in layers:
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
                    pts.append((int(px + ox), int(py + oy)))
                draw.line(pts, fill=(58, 120, 201), width=max(1, round(sw * line_scale)))

    if "roads" in layers:
        # Deduped and ordered by tier, so shared trunk segments are drawn once and a
        # track never paints over the primary road it branches from.
        for road, leg in dedupe_road_paths(ws.roads, ws.hexes, lambda r: ROAD_TIER_RANK[r.tier]):
            pts = []
            for coord in leg:
                px, py = axial_to_pixel(coord, size)
                pts.append((int(px + ox), int(py + oy)))
            draw.line(
                pts,
                fill=_ROAD_COLOR[road.tier],
                width=max(1, round(_ROAD_WIDTH[road.tier] * line_scale)),
            )

    if "crossings" in layers:
        for coord, kind, angle in legend.crossings(ws, axial_to_pixel, size):
            px, py = axial_to_pixel(coord, size)
            _draw_crossing(draw, kind, px + ox, py + oy, angle, scale=size / 12.0)

    if "anchorages" in layers:
        # One marker per shore point, however many routes embark there. Covers both
        # sea legs and ferry landings — the same thing as far as a reader is concerned.
        for coord in legend.anchorage_points(ws):
            px, py = axial_to_pixel(coord, size)
            _draw_anchorage(draw, px + ox, py + oy, scale=size / 12.0)

    if "settlements" in layers:
        for s in ws.settlements:
            px, py = axial_to_pixel(s.coord, size)
            _draw_settlement(draw, s.tier, px + ox, py + oy)

    if "labels" in layers:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        for s in ws.settlements:
            px, py = axial_to_pixel(s.coord, size)
            cx, cy = int(px + ox), int(py + oy)
            bbox = draw.textbbox((0, 0), s.name, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text((cx - tw // 2, cy - int(size) - th - 2), s.name, fill=(0, 0, 0), font=font)

    # Legend last so it paints over anything that reaches into the corner.
    if legend_m is not None:
        _draw_legend(
            draw,
            ws,
            config,
            legend_rows,
            legend_m,
            legend_font,
            color_mode,
            ox,
            oy,
            pad,
            width,
            height,
        )

    return img


def save(ws: WorldState, path, config: PNGConfig | None = None) -> None:
    """Write PNG hex map to a file."""
    if config is None:
        config = PNGConfig()
    img = render(ws, config)
    img.save(str(path), dpi=(config.dpi, config.dpi))
