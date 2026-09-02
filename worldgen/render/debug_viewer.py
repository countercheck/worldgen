import math
from pathlib import Path

import matplotlib as mpl

from ..core.hex import Biome, LandCover, SettlementTier, TerrainClass
from ..core.hex_grid import axial_to_pixel, dedupe_road_paths
from ..core.world_state import ROAD_TIER_RANK, RoadTier, WorldState

TERRAIN_COLORS = {
    TerrainClass.OCEAN: (0.2, 0.4, 0.8),
    TerrainClass.LAKE: (0.35, 0.6, 0.85),
    TerrainClass.COAST: (0.9, 0.8, 0.4),
    TerrainClass.FLAT: (0.4, 0.8, 0.4),
    TerrainClass.HILL: (0.7, 0.6, 0.3),
    TerrainClass.MOUNTAIN: (0.5, 0.5, 0.5),
}

BIOME_COLORS = {
    Biome.OCEAN: (0.2, 0.4, 0.8),
    Biome.TUNDRA: (0.9, 0.95, 0.95),
    Biome.BOREAL: (0.3, 0.5, 0.3),
    Biome.TEMPERATE_FOREST: (0.2, 0.6, 0.2),
    Biome.GRASSLAND: (0.6, 0.8, 0.3),
    Biome.SHRUBLAND: (0.8, 0.7, 0.3),
    Biome.DESERT: (0.95, 0.9, 0.6),
    Biome.TROPICAL: (0.1, 0.5, 0.1),
    Biome.WETLAND: (0.4, 0.6, 0.4),
    Biome.ALPINE: (0.7, 0.7, 0.7),
}

LAND_COVER_COLORS = {
    LandCover.OPEN_WATER: (0.255, 0.412, 0.882),
    LandCover.BOG: (0.333, 0.420, 0.184),
    LandCover.MARSH: (0.420, 0.557, 0.420),
    LandCover.DENSE_FOREST: (0.102, 0.290, 0.102),
    LandCover.WOODLAND: (0.227, 0.478, 0.227),
    LandCover.SCRUB: (0.545, 0.455, 0.333),
    LandCover.OPEN: (0.784, 0.847, 0.439),
    LandCover.TUNDRA: (0.690, 0.769, 0.769),
    LandCover.DESERT: (0.824, 0.706, 0.549),
    LandCover.ALPINE: (0.627, 0.627, 0.627),
    LandCover.BARE_ROCK: (0.376, 0.376, 0.376),
}

_ROAD_STYLE = {
    RoadTier.PRIMARY: {"stroke": "#5c3d1e", "stroke-width": "2.0"},
    RoadTier.SECONDARY: {"stroke": "#8b6914", "stroke-width": "1.2"},
    RoadTier.TRACK: {"stroke": "#b8a070", "stroke-width": "0.6", "dasharray": "4 2"},
}


def _get_color_biome(h) -> tuple[float, float, float]:
    return BIOME_COLORS.get(h.biome, (0.5, 0.5, 0.5))


def _get_color_terrain(h) -> tuple[float, float, float]:
    return TERRAIN_COLORS[h.terrain_class]


def _hex_vertices(cx: float, cy: float, size: float) -> list[tuple[float, float]]:
    angles = [0, 60, 120, 180, 240, 300]
    return [
        (cx + size * math.cos(math.radians(a)), cy + size * math.sin(math.radians(a)))
        for a in angles
    ]


def _points_str(pts: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)


def _rgb_to_hex(r: float, g: float, b: float) -> str:
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def _star_points(cx: float, cy: float, outer: float, inner: float, n: int = 5) -> str:
    pts = []
    for i in range(n * 2):
        r = outer if i % 2 == 0 else inner
        angle = math.radians(i * 180 / n - 90)
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    return _points_str(pts)


def _color_getter(attribute: str):
    """Returns (get_color, settlement_overlay, road_overlay) for an attribute name."""
    if attribute == "biome":
        return _get_color_biome, False, False
    if attribute == "terrain_class":
        return _get_color_terrain, False, False
    if attribute == "elevation":
        cmap = mpl.colormaps["terrain"]
        return (lambda h: cmap(h.elevation)), False, False
    if attribute == "moisture":
        cmap = mpl.colormaps["Blues"]
        return (lambda h: cmap(h.moisture)), False, False
    if attribute == "temperature":
        cmap = mpl.colormaps["RdYlBu_r"]
        return (lambda h: cmap(h.temperature)), False, False
    if attribute == "river_flow":
        cmap = mpl.colormaps["Blues"]

        def get_color(h):
            if h.terrain_class in (TerrainClass.OCEAN, TerrainClass.LAKE):
                return TERRAIN_COLORS[h.terrain_class]
            return cmap(min(h.river_flow * 3, 1.0))

        return get_color, False, False
    if attribute == "habitability":
        cmap = mpl.colormaps["YlGn"]
        return (lambda h: cmap(h.habitability)), False, False
    if attribute == "settlements":
        return _get_color_biome, True, False
    if attribute == "roads":
        return _get_color_biome, False, True
    if attribute == "land_cover":

        def get_color(h):
            if h.land_cover is None:
                return (0.5, 0.5, 0.5)
            return LAND_COVER_COLORS.get(h.land_cover, (0.5, 0.5, 0.5))

        return get_color, False, False
    if attribute == "cultivation":
        _cultivated = (0.831, 0.643, 0.298)

        def get_color(h):
            if h.land_cover is None:
                return (0.5, 0.5, 0.5)
            base = LAND_COVER_COLORS.get(h.land_cover, (0.5, 0.5, 0.5))
            return _cultivated if h.cultivated else base

        return get_color, False, False
    raise ValueError(f"Unknown attribute: {attribute}")


def render_svg(state: WorldState, attribute: str, hex_size: float = 20) -> str:
    """Render hex map colored by attribute as an SVG string."""
    get_color, settlement_overlay, road_overlay = _color_getter(attribute)

    hex_items = list(state.hexes.values())
    if not hex_items:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="0" height="0"></svg>'

    pad = hex_size
    pixels = [axial_to_pixel(h.coord, hex_size) for h in hex_items]
    min_x = min(p[0] for p in pixels) - pad
    min_y = min(p[1] for p in pixels) - pad
    max_x = max(p[0] for p in pixels) + pad
    max_y = max(p[1] for p in pixels) + pad
    ox = -min_x
    oy = -min_y
    w = math.ceil(max_x - min_x)
    h = math.ceil(max_y - min_y)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        f'  <rect width="{w}" height="{h}" fill="white"/>',
        '  <g id="layer-hexes">',
    ]
    for hex_item, (px, py) in zip(hex_items, pixels, strict=True):
        verts = _hex_vertices(px + ox, py + oy, hex_size)
        fill = _rgb_to_hex(*get_color(hex_item)[:3])
        out.append(
            f'    <polygon points="{_points_str(verts)}" fill="{fill}"'
            f' stroke="gray" stroke-width="0.5"/>'
        )
    out.append("  </g>")

    if settlement_overlay:
        out.append('  <g id="layer-settlements">')
        for s in state.settlements:
            px, py = axial_to_pixel(s.coord, hex_size)
            cx, cy = px + ox, py + oy
            if s.tier == SettlementTier.CITY:
                pts = _star_points(cx, cy, outer=7.0, inner=3.0)
                out.append(
                    f'    <polygon points="{pts}" fill="gold" stroke="black" stroke-width="0.8"/>'
                )
            elif s.tier == SettlementTier.TOWN:
                r = 4.0
                out.append(
                    f'    <rect x="{cx - r:.2f}" y="{cy - r:.2f}" width="{2 * r:.2f}"'
                    f' height="{2 * r:.2f}" fill="white" stroke="black" stroke-width="0.8"/>'
                )
            else:
                out.append(
                    f'    <circle cx="{cx:.2f}" cy="{cy:.2f}" r="2.5" fill="white"'
                    f' stroke="black" stroke-width="0.8"/>'
                )
        out.append("  </g>")

    if road_overlay:
        out.append('  <g id="layer-roads">')
        # Iterating RoadTier drew PRIMARY first and TRACK last, so a track painted over
        # the primary road it branches from. dedupe_road_paths awards each shared edge to
        # its highest tier and returns them in ascending order, so primaries land on top.
        for road, leg in dedupe_road_paths(
            state.roads, state.hexes, lambda r: ROAD_TIER_RANK[r.tier]
        ):
            style = _ROAD_STYLE[road.tier]
            da = f' stroke-dasharray="{style["dasharray"]}"' if "dasharray" in style else ""
            pts = [(px + ox, py + oy) for px, py in (axial_to_pixel(c, hex_size) for c in leg)]
            out.append(
                f'    <polyline points="{_points_str(pts)}" fill="none"'
                f' stroke="{style["stroke"]}" stroke-width="{style["stroke-width"]}"'
                f' stroke-linecap="round" stroke-linejoin="round"{da}/>'
            )
        out.append("  </g>")

    out.append(
        f'  <text x="8" y="18" font-family="sans-serif" font-size="14">World Map — {attribute}</text>'
    )
    out.append("</svg>")
    return "\n".join(out)


def render(state: WorldState, attribute: str, output_path: str, hex_size: float = 20) -> None:
    """Render hex map colored by attribute and write it to output_path as SVG."""
    Path(output_path).write_text(render_svg(state, attribute, hex_size), encoding="utf-8")
