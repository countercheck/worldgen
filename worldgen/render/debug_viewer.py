import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt

from ..core.hex import Biome, TerrainClass
from ..core.hex_grid import axial_to_pixel
from ..core.world_state import WorldState

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


def render(state: WorldState, attribute: str, output_path: str, hex_size: float = 20):
    """Render hex map colored by attribute."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_aspect("equal")

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
        get_color = _get_color_terrain
    elif attribute == "habitability":
        cmap = plt.cm.get_cmap("YlGn")

        def get_color(h: WorldState):  # noqa: F811
            return cmap(h.habitability)
    else:
        raise ValueError(f"Unknown attribute: {attribute}")

    for hex_item in state.hexes.values():
        x, y = axial_to_pixel(hex_item.coord, hex_size)
        color = get_color(hex_item)

        vertices = _hex_vertices(x, y, hex_size)
        polygon = patches.Polygon(vertices, facecolor=color, edgecolor="gray", linewidth=0.5)
        ax.add_patch(polygon)

    if attribute == "river_flow":
        _draw_rivers(ax, state, hex_size)

    ax.autoscale_view()
    ax.set_title(f"World Map — {attribute}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()


def _draw_rivers(ax, state: WorldState, hex_size: float) -> None:
    """Overlay river paths as lines scaled by flow volume."""
    for river in state.rivers:
        xs = []
        ys = []
        for coord in river.hexes:
            if coord in state.hexes:
                x, y = axial_to_pixel(coord, hex_size)
                xs.append(x)
                ys.append(y)
        if len(xs) < 2:
            continue
        width = 0.5 + river.flow_volume * 4.0
        ax.plot(
            xs,
            ys,
            color=(0.1, 0.4, 0.8),
            linewidth=width,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=2,
        )


def _hex_vertices(x: float, y: float, size: float) -> list[tuple[float, float]]:
    """Get 6 vertices of flat-top hex."""
    angles = [0, 60, 120, 180, 240, 300]
    return [(x + size * np.cos(np.radians(a)), y + size * np.sin(np.radians(a))) for a in angles]
