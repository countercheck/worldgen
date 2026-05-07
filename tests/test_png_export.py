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
