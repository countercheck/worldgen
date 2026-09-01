import math
from dataclasses import dataclass, field

from PIL import Image, ImageDraw, ImageFont

from ..core.hex import SettlementTier
from ..core.hex_grid import axial_to_pixel, neighbors, split_path_on_water
from ..core.world_state import RoadTier, WorldState
from ..render.debug_viewer import BIOME_COLORS, LAND_COVER_COLORS, TERRAIN_COLORS
from . import legend


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
            "legend",
        }
    )
    contour_elevation_scale_m: float = 3000.0
    contour_interval_m: float = 100.0
    contour_max_crossings: int = 5
    contour_max_stroke: float = 4.0
    legend_corner: str = "top-right"  # "top-right" | "bottom-left"
    legend_scale: float = 1.0  # multiplier on hex_size for legend glyph/text size


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
    elif row.kind == "road":
        color, lw = _ROAD_COLOR[row.sample], _ROAD_WIDTH[row.sample]
        if row.sample == RoadTier.TRACK:
            _dashed_line(draw, cx - g / 2, cy, cx + g / 2, color, lw)
        else:
            draw.line([(cx - g / 2, cy), (cx + g / 2, cy)], fill=color, width=lw)
    else:  # "river" — matches the rivers layer's colour
        draw.line([(cx - g / 2, cy), (cx + g / 2, cy)], fill=(58, 120, 201), width=2)


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
        layers = {"terrain", "roads", "settlements", "grid", "legend"}
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
            if len(river.hexes) < 2:
                continue
            pts = []
            for coord in river.hexes:
                px, py = axial_to_pixel(coord, size)
                pts.append((int(px + ox), int(py + oy)))
            lw = max(1, min(4, int(river.flow_volume * 2) + 1))
            draw.line(pts, fill=(58, 120, 201), width=lw)

    if "roads" in layers:
        for road in ws.roads:
            for leg in split_path_on_water(road.path, ws.hexes):
                pts = []
                for coord in leg:
                    px, py = axial_to_pixel(coord, size)
                    pts.append((int(px + ox), int(py + oy)))
                draw.line(pts, fill=_ROAD_COLOR[road.tier], width=_ROAD_WIDTH[road.tier])

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
