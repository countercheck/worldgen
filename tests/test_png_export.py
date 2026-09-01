import pytest
from PIL import Image

from worldgen.core.hex import (
    Biome,
    LandCover,
    Settlement,
    SettlementRole,
    SettlementTier,
    TerrainClass,
)
from worldgen.core.world_state import River, Road, RoadTier, WorldState
from worldgen.export.png_export import PNGConfig, render, save


def _small_world() -> WorldState:
    ws = WorldState.empty(seed=99, width=4, height=4)
    h = ws.hexes[(0, 0)]
    h.biome = Biome.GRASSLAND
    h.terrain_class = TerrainClass.FLAT
    h.land_cover = LandCover.OPEN
    h.elevation = 0.5
    ws.settlements = [
        Settlement(
            coord=(1, 1),
            tier=SettlementTier.CITY,
            role=SettlementRole.MARKET,
            population=5000,
            name="Ironhaven",
        ),
        Settlement(
            coord=(2, 2),
            tier=SettlementTier.TOWN,
            role=SettlementRole.PORT,
            population=800,
            name="Saltmere",
        ),
        Settlement(
            coord=(3, 1),
            tier=SettlementTier.VILLAGE,
            role=SettlementRole.AGRICULTURAL,
            population=120,
            name="Millbrook",
        ),
    ]
    ws.rivers = [River(hexes=[(0, 0), (1, 0), (2, 0)], flow_volume=1.5)]
    ws.roads = [Road(path=[(1, 1), (2, 1), (3, 1)], tier=RoadTier.PRIMARY)]
    return ws


def test_produces_pil_image():
    ws = _small_world()
    img = render(ws)
    assert isinstance(img, Image.Image)


def test_dimensions_nonzero():
    ws = _small_world()
    img = render(ws)
    assert img.width > 0
    assert img.height > 0


def test_mode_is_rgb():
    ws = _small_world()
    img = render(ws)
    assert img.mode == "RGB"


def test_style_presets():
    ws = _small_world()
    for style in ("atlas", "topographic", "wargame"):
        config = PNGConfig(style=style)
        img = render(ws, config)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"
        assert img.width > 0
        assert img.height > 0


def test_save_creates_file(tmp_path):
    ws = _small_world()
    path = tmp_path / "world.png"
    save(ws, str(path))
    assert path.exists()


def test_saved_file_is_valid_png(tmp_path):
    ws = _small_world()
    path = tmp_path / "world.png"
    save(ws, str(path))
    img = Image.open(path)
    assert img.width > 0
    assert img.height > 0


def test_empty_world_returns_image():
    ws = WorldState(seed=1, width=0, height=0)
    img = render(ws)
    assert isinstance(img, Image.Image)


def test_hex_size_affects_dimensions():
    ws = _small_world()
    small = render(ws, PNGConfig(hex_size=6.0))
    large = render(ws, PNGConfig(hex_size=20.0))
    assert large.width > small.width
    assert large.height > small.height


def test_layer_toggle_terrain_only():
    ws = _small_world()
    config = PNGConfig(layers={"terrain"})
    img = render(ws, config)
    assert isinstance(img, Image.Image)


def test_contours_layer_renders():
    ws = WorldState.empty(seed=1, width=4, height=4)
    # Use known-adjacent hexes: (1, 0) is a neighbor of (0, 0).
    ws.hexes[(0, 0)].elevation = 0.0
    ws.hexes[(1, 0)].elevation = 0.5  # 1500 m diff → contour line drawn
    # Render only the contours layer so the background stays plain white.
    config = PNGConfig(layers={"contours"})
    img = render(ws, config)
    assert isinstance(img, Image.Image)
    assert img.width > 0
    # At least one pixel should be non-white (the contour line itself).
    pixels = list(img.getdata())
    assert any(p != (255, 255, 255) for p in pixels), "expected contour pixels to be drawn"


def test_topographic_style_includes_contours():
    ws = _small_world()
    config = PNGConfig(style="topographic")
    img = render(ws, config)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"


def test_contours_flat_world_no_lines():
    ws = WorldState.empty(seed=1, width=4, height=4)
    for h in ws.hexes.values():
        h.elevation = 0.5
    # Flat world: contour layer runs but draws nothing — all pixels stay white.
    config = PNGConfig(layers={"contours"})
    img = render(ws, config)
    assert isinstance(img, Image.Image)
    pixels = list(img.getdata())
    assert all(p == (255, 255, 255) for p in pixels), "expected no contour pixels for flat world"


def test_contours_reject_nonpositive_max_crossings():
    ws = _small_world()
    with pytest.raises(ValueError, match="contour_max_crossings must be positive"):
        render(ws, PNGConfig(layers={"contours"}, contour_max_crossings=0))


# --- legend ------------------------------------------------------------------


def _sheared_world() -> WorldState:
    """A world wide enough that the axial shear opens up real corner space."""
    ws = WorldState.empty(seed=7, width=32, height=32)
    ws.rivers = [River(hexes=[(0, 0), (1, 0), (2, 0)], flow_volume=1.5)]
    ws.roads = [Road(path=[(1, 1), (2, 1), (3, 1)], tier=RoadTier.PRIMARY)]
    ws.settlements = [
        Settlement(
            coord=(4, 4),
            tier=SettlementTier.CITY,
            role=SettlementRole.MARKET,
            population=5000,
            name="Ironhaven",
        )
    ]
    return ws


def _panel_bbox(img: Image.Image, no_legend: Image.Image):
    """Bounding box of the pixels the legend adds, as (min_x, min_y, max_x, max_y)."""
    from PIL import ImageChops

    diff = ImageChops.difference(img.convert("RGB"), no_legend.convert("RGB"))
    return diff.getbbox()


@pytest.mark.parametrize("corner", ["top-right", "bottom-left"])
def test_legend_covers_no_terrain(corner):
    """The panel must land in blank canvas, not on top of hexes."""
    ws = _sheared_world()
    layers = {"terrain", "rivers", "roads", "settlements", "grid"}
    plain = render(ws, PNGConfig(layers=set(layers)))
    with_legend = render(ws, PNGConfig(layers=layers | {"legend"}, legend_corner=corner))
    box = _panel_bbox(with_legend, plain)
    assert box is not None, "legend drew nothing"
    # Every pixel the legend replaced must have been blank white canvas before.
    region = plain.convert("RGB").crop(box)
    colors = {c for _, c in region.getcolors(maxcolors=1 << 20)}
    assert colors == {(255, 255, 255)}, f"legend painted over map content in the {corner} corner"


@pytest.mark.parametrize("corner", ["top-right", "bottom-left"])
def test_legend_hugs_the_map_edge(corner):
    """Clearing the diagonal is not enough — the panel must also stay next to the map.

    The empty triangle is enormous on a wide map, so simply jamming the panel into the
    canvas corner would leave it stranded far from any hex.  The panel is placed flush
    against the bounding diagonal instead; this pins that gap to roughly the one-hex
    margin the placement reserves.
    """
    ws = _sheared_world()
    hex_size = 12.0
    layers = {"terrain", "grid"}
    plain = render(ws, PNGConfig(hex_size=hex_size, layers=layers))
    with_legend = render(
        ws, PNGConfig(hex_size=hex_size, layers=layers | {"legend"}, legend_corner=corner)
    )
    x0, y0, x1, y1 = _panel_bbox(with_legend, plain)
    px = plain.convert("RGB").load()

    def first_painted_row(rows):
        for y in rows:
            if any(px[x, y] != (255, 255, 255) for x in range(x0, x1)):
                return y
        return None

    if corner == "top-right":
        row = first_painted_row(range(y1, plain.height))
        gap = None if row is None else row - y1
    else:
        row = first_painted_row(range(y0, 0, -1))
        gap = None if row is None else y0 - row
    assert gap is not None, "no map content in the legend's columns"
    assert gap > 0, f"legend panel is not clear of the map ({gap}px)"
    assert gap < 4 * hex_size, f"legend panel is adrift, {gap}px from the nearest hex"


def test_legend_in_default_layers():
    """The stock PNGConfig draws a legend; dropping the layer is what removes it."""
    ws = _sheared_world()
    default = render(ws)
    without = render(ws, PNGConfig(layers=PNGConfig().layers - {"legend"}))
    box = _panel_bbox(default, without)
    assert box is not None, "default config drew no legend"
    # With the layer off, that same region is untouched canvas.
    region = without.convert("RGB").crop(box)
    colors = {c for _, c in region.getcolors(maxcolors=1 << 20)}
    assert colors == {(255, 255, 255)}


def test_legend_corners_are_on_opposite_sides():
    ws = _sheared_world()
    layers = {"terrain", "grid", "legend"}
    plain = render(ws, PNGConfig(layers={"terrain", "grid"}))
    tr = _panel_bbox(render(ws, PNGConfig(layers=layers, legend_corner="top-right")), plain)
    bl = _panel_bbox(render(ws, PNGConfig(layers=layers, legend_corner="bottom-left")), plain)
    assert tr[0] > bl[0]  # top-right is further right
    assert tr[1] < bl[1]  # top-right is further up


def test_legend_scale_grows_the_panel():
    ws = _sheared_world()
    layers = {"terrain", "grid", "legend"}
    plain = render(ws, PNGConfig(layers={"terrain", "grid"}))
    small = _panel_bbox(render(ws, PNGConfig(layers=layers, legend_scale=1.0)), plain)
    large = _panel_bbox(render(ws, PNGConfig(layers=layers, legend_scale=2.0)), plain)
    assert large[2] - large[0] > small[2] - small[0]
    assert large[3] - large[1] > small[3] - small[1]


def test_legend_rejects_unknown_corner():
    with pytest.raises(ValueError, match="legend_corner must be"):
        render(_small_world(), PNGConfig(legend_corner="middle"))


def test_legend_rejects_nonpositive_scale():
    with pytest.raises(ValueError, match="legend_scale must be positive"):
        render(_small_world(), PNGConfig(legend_scale=0))


def test_legend_skipped_for_empty_world():
    img = render(WorldState(seed=1, width=0, height=0))
    assert img.size == (1, 1)
