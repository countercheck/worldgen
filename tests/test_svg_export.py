from worldgen.core.hex import (
    Biome,
    LandCover,
    Settlement,
    SettlementRole,
    SettlementTier,
    TerrainClass,
)
from worldgen.core.world_state import River, Road, RoadTier, WorldState
from worldgen.export.svg_export import SVGConfig, render, save


def _small_world() -> WorldState:
    ws = WorldState.empty(seed=99, width=4, height=4)
    h = ws.hexes[(0, 0)]
    h.biome = Biome.GRASSLAND
    h.terrain_class = TerrainClass.FLAT
    h.land_cover = LandCover.OPEN
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


def test_valid_svg():
    ws = _small_world()
    svg = render(ws)
    assert svg.startswith("<svg")
    assert svg.rstrip().endswith("</svg>")


def test_default_layers_present():
    ws = _small_world()
    svg = render(ws)
    assert 'id="layer-terrain"' in svg
    assert 'id="layer-rivers"' in svg
    assert 'id="layer-roads"' in svg
    assert 'id="layer-settlements"' in svg
    assert 'id="layer-labels"' in svg
    assert 'id="layer-grid"' in svg


def test_layer_toggle():
    ws = _small_world()
    config = SVGConfig(layers={"terrain"})
    svg = render(ws, config)
    assert 'id="layer-terrain"' in svg
    assert 'id="layer-settlements"' not in svg
    assert 'id="layer-rivers"' not in svg


def test_settlement_names_in_output():
    ws = _small_world()
    svg = render(ws)
    assert "Ironhaven" in svg
    assert "Saltmere" in svg
    assert "Millbrook" in svg


def test_style_presets_produce_svg():
    ws = _small_world()
    for style in ("atlas", "topographic", "wargame"):
        config = SVGConfig(style=style)
        svg = render(ws, config)
        assert svg.startswith("<svg")
        assert len(svg) > 100


def test_topographic_omits_labels():
    ws = _small_world()
    config = SVGConfig(style="topographic")
    svg = render(ws, config)
    assert 'id="layer-labels"' not in svg


def test_wargame_omits_labels():
    ws = _small_world()
    config = SVGConfig(style="wargame")
    svg = render(ws, config)
    assert 'id="layer-labels"' not in svg


def test_river_stroke_present():
    ws = _small_world()
    svg = render(ws)
    assert 'id="layer-rivers"' in svg
    assert "polyline" in svg


def test_road_stroke_present():
    ws = _small_world()
    svg = render(ws)
    assert 'id="layer-roads"' in svg
    assert "#5c3d1e" in svg  # PRIMARY road color


def test_empty_world_returns_valid_svg():
    ws = WorldState(seed=1, width=0, height=0)
    svg = render(ws)
    assert "svg" in svg


def test_save_creates_file(tmp_path):
    ws = _small_world()
    path = tmp_path / "world.svg"
    save(ws, path)
    assert path.exists()
    content = path.read_text()
    assert "<svg" in content


def test_all_settlement_tiers_rendered():
    ws = _small_world()
    svg = render(ws)
    assert "gold" in svg  # city star
    assert "<rect" in svg  # town square
    assert "<circle" in svg  # village circle


def test_contours_layer_produces_lines():
    ws = WorldState.empty(seed=1, width=4, height=4)
    # Set up a steep elevation gradient between two known-adjacent hexes.
    # (1, 0) is a neighbor of (0, 0) per the axial hex grid.
    ws.hexes[(0, 0)].elevation = 0.0
    ws.hexes[(1, 0)].elevation = 0.5  # 1500 m diff at scale 3000
    config = SVGConfig(layers={"contours"})
    svg = render(ws, config)
    assert 'id="layer-contours"' in svg
    assert "<line" in svg


def test_contours_below_threshold_omitted():
    ws = WorldState.empty(seed=1, width=4, height=4)
    # All hexes at same elevation → no contour lines drawn.
    for h in ws.hexes.values():
        h.elevation = 0.5
    config = SVGConfig(layers={"contours"})
    svg = render(ws, config)
    assert 'id="layer-contours"' in svg
    assert "<line" not in svg


def test_topographic_style_includes_contours():
    ws = _small_world()
    config = SVGConfig(style="topographic")
    svg = render(ws, config)
    assert 'id="layer-contours"' in svg


def test_contour_stroke_scales_with_elevation_diff():
    import re

    ws = WorldState.empty(seed=1, width=4, height=4)
    # (0,0)↔(1,0) and (2,0)↔(3,0) are adjacent pairs in the axial grid.
    # Pair A: 1 threshold crossing (100 m)
    ws.hexes[(0, 0)].elevation = 0.0  # 0 m
    ws.hexes[(1, 0)].elevation = 0.05  # 150 m → crosses 100 m
    # Pair B: 4 threshold crossings (100, 200, 300, 400 m)
    ws.hexes[(2, 0)].elevation = 0.0  # 0 m
    ws.hexes[(3, 0)].elevation = 0.15  # 450 m → crosses 100–400 m
    config = SVGConfig(layers={"contours"})
    svg = render(ws, config)
    widths = [float(m) for m in re.findall(r'stroke-width="([\d.]+)"', svg)]
    assert len(widths) >= 2
    assert min(widths) < max(widths)  # more crossings → thicker


def test_contour_darkness_scales_with_crossings():
    import re

    ws = WorldState.empty(seed=1, width=4, height=4)
    # Pair A: 1 crossing → light gray
    ws.hexes[(0, 0)].elevation = 0.0
    ws.hexes[(1, 0)].elevation = 0.05  # 150 m
    # Pair B: saturated crossings → near-black
    ws.hexes[(2, 0)].elevation = 0.0
    ws.hexes[(3, 0)].elevation = 0.5  # 1500 m → 15 crossings
    config = SVGConfig(layers={"contours"})
    svg = render(ws, config)
    colors = re.findall(r'stroke="(#[0-9a-f]{6})"', svg)
    grays = [int(c[1:3], 16) for c in colors]  # red channel == gray value
    assert min(grays) < max(grays)  # more crossings → darker
