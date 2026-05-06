import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt

from ..core.hex import Biome, LandCover, SettlementTier, TerrainClass
from ..core.hex_grid import axial_to_pixel
from ..core.world_state import RoadTier, WorldState

TERRAIN_COLORS = {
    TerrainClass.OCEAN: (0.2, 0.4, 0.8),
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


def _get_color_biome(h: WorldState) -> tuple[float, float, float]:
    return BIOME_COLORS.get(h.biome, (0.5, 0.5, 0.5))


def _get_color_terrain(h: WorldState) -> tuple[float, float, float]:
    return TERRAIN_COLORS[h.terrain_class]


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
    RoadTier.PRIMARY: {"color": "#5c3d1e", "lw": 2.0, "zorder": 3},
    RoadTier.SECONDARY: {"color": "#8b6914", "lw": 1.2, "zorder": 2},
    RoadTier.TRACK: {"color": "#b8a070", "lw": 0.6, "zorder": 1},
}

_SETTLEMENT_STYLE = {
    SettlementTier.CITY: ("*", 14, "gold"),
    SettlementTier.TOWN: ("s", 8, "white"),
    SettlementTier.VILLAGE: ("o", 5, "white"),
}


def render(state: WorldState, attribute: str, output_path: str, hex_size: float = 20):
    """Render hex map colored by attribute."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_aspect("equal")

    settlement_overlay = False
    road_overlay = False

    if attribute == "biome":
        get_color = _get_color_biome
    elif attribute == "terrain_class":
        get_color = _get_color_terrain
    elif attribute == "elevation":
        cmap = plt.cm.get_cmap("terrain")

        def get_color(h: WorldState):  # noqa: F811
            return cmap(h.elevation)
    elif attribute == "moisture":
        cmap = plt.cm.get_cmap("Blues")

        def get_color(h: WorldState):  # noqa: F811
            return cmap(h.moisture)
    elif attribute == "temperature":
        cmap = plt.cm.get_cmap("RdYlBu_r")

        def get_color(h: WorldState):  # noqa: F811
            return cmap(h.temperature)
    elif attribute == "river_flow":
        cmap = plt.cm.get_cmap("Blues")

        def get_color(h: WorldState):  # noqa: F811
            return cmap(min(h.river_flow * 3, 1.0))
    elif attribute == "habitability":
        cmap = plt.cm.get_cmap("YlGn")

        def get_color(h: WorldState):  # noqa: F811
            return cmap(h.habitability)
    elif attribute == "settlements":
        get_color = _get_color_biome
        settlement_overlay = True
    elif attribute == "roads":
        get_color = _get_color_biome
        road_overlay = True
    elif attribute == "land_cover":

        def get_color(h):  # noqa: F811
            if h.land_cover is None:
                return (0.5, 0.5, 0.5)
            return LAND_COVER_COLORS.get(h.land_cover, (0.5, 0.5, 0.5))
    elif attribute == "cultivation":
        _CULTIVATED = (0.831, 0.643, 0.298)

        def get_color(h):  # noqa: F811
            if h.land_cover is None:
                return (0.5, 0.5, 0.5)
            base = LAND_COVER_COLORS.get(h.land_cover, (0.5, 0.5, 0.5))
            return _CULTIVATED if h.cultivated else base
    else:
        raise ValueError(f"Unknown attribute: {attribute}")

    for hex_item in state.hexes.values():
        x, y = axial_to_pixel(hex_item.coord, hex_size)
        color = get_color(hex_item)

        vertices = _hex_vertices(x, y, hex_size)
        polygon = patches.Polygon(vertices, facecolor=color, edgecolor="gray", linewidth=0.5)
        ax.add_patch(polygon)

    if settlement_overlay:
        for s in state.settlements:
            x, y = axial_to_pixel(s.coord, hex_size)
            marker, size, color = _SETTLEMENT_STYLE[s.tier]
            ax.plot(
                x,
                y,
                marker,
                markersize=size,
                color=color,
                markeredgecolor="black",
                markeredgewidth=0.8,
            )

    if road_overlay:
        for road in state.roads:
            pixel_coords = [axial_to_pixel(coord, hex_size) for coord in road.path]
            if pixel_coords:
                xs, ys = map(list, zip(*pixel_coords, strict=False))
            else:
                xs, ys = [], []
            style = _ROAD_STYLE[road.tier]
            ax.plot(
                xs,
                ys,
                color=style["color"],
                lw=style["lw"],
                zorder=style["zorder"],
                solid_capstyle="round",
            )

    ax.autoscale_view()
    ax.set_title(f"World Map — {attribute}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()


def _hex_vertices(x: float, y: float, size: float) -> list[tuple[float, float]]:
    """Get 6 vertices of flat-top hex."""
    angles = [0, 60, 120, 180, 240, 300]
    return [(x + size * np.cos(np.radians(a)), y + size * np.sin(np.radians(a))) for a in angles]
