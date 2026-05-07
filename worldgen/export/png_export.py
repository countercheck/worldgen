import math
from dataclasses import dataclass, field

from PIL import Image, ImageDraw, ImageFont

from ..core.hex import SettlementTier
from ..core.hex_grid import axial_to_pixel, neighbors
from ..core.world_state import RoadTier, WorldState
from ..render.debug_viewer import BIOME_COLORS, LAND_COVER_COLORS, TERRAIN_COLORS


@dataclass
class PNGConfig:
    hex_size: float = 12.0
    dpi: int = 150
    style: str = "atlas"  # "atlas" | "topographic" | "wargame"
    color_mode: str = "biome"  # "biome" | "terrain" | "land_cover" | "elevation"
    layers: set[str] = field(
        default_factory=lambda: {"terrain", "rivers", "roads", "settlements", "labels", "grid"}
    )
    contour_elevation_scale_m: float = 3000.0
    contour_interval_m: float = 100.0
    contour_max_crossings: int = 5
    contour_max_stroke: float = 4.0


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


def render(ws: WorldState, config: PNGConfig | None = None) -> Image.Image:
    """Render WorldState as a PIL Image."""
    if config is None:
        config = PNGConfig()

    if config.style == "topographic":
        color_mode = "elevation"
        layers: set[str] = {"terrain", "rivers", "grid", "contours"}
    elif config.style == "wargame":
        color_mode = "terrain"
        layers = {"terrain", "roads", "settlements", "grid"}
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
            if len(road.path) < 2:
                continue
            pts = []
            for coord in road.path:
                px, py = axial_to_pixel(coord, size)
                pts.append((int(px + ox), int(py + oy)))
            draw.line(pts, fill=_ROAD_COLOR[road.tier], width=_ROAD_WIDTH[road.tier])

    if "settlements" in layers:
        for s in ws.settlements:
            px, py = axial_to_pixel(s.coord, size)
            cx, cy = px + ox, py + oy
            if s.tier == SettlementTier.CITY:
                pts = _star_pts(cx, cy, outer=6.0, inner=2.5)
                draw.polygon(pts, fill=(255, 215, 0), outline=(0, 0, 0))
            elif s.tier == SettlementTier.TOWN:
                r = 4
                draw.rectangle(
                    [cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255), outline=(0, 0, 0)
                )
            else:
                r = 3
                draw.ellipse(
                    [cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255), outline=(0, 0, 0)
                )

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

    return img


def save(ws: WorldState, path, config: PNGConfig | None = None) -> None:
    """Write PNG hex map to a file."""
    if config is None:
        config = PNGConfig()
    img = render(ws, config)
    img.save(str(path), dpi=(config.dpi, config.dpi))
