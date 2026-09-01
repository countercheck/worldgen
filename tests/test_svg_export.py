import pytest

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


def test_contour_single_crossing_uses_lightest_min_styling():
    import re

    ws = WorldState.empty(seed=1, width=4, height=4)
    ws.hexes[(0, 0)].elevation = 0.0
    ws.hexes[(1, 0)].elevation = 0.05  # exactly one threshold crossing at 100 m
    svg = render(ws, SVGConfig(layers={"contours"}, contour_max_crossings=5))
    m = re.search(r'stroke="(#[0-9a-f]{6})" stroke-width="([\d.]+)"', svg)
    assert m is not None
    # For n=1, normalization pins t=0, yielding the configured minimum contour style:
    # light gray (#bbbbbb from 187) and base stroke width (0.30 from the 0.3 floor).
    assert m.group(1) == "#bbbbbb"
    assert m.group(2) == "0.30"


def test_contour_max_crossings_one_saturates_first_crossing():
    import re

    ws = WorldState.empty(seed=1, width=4, height=4)
    ws.hexes[(0, 0)].elevation = 0.0
    ws.hexes[(1, 0)].elevation = 0.05  # one threshold crossing
    svg = render(ws, SVGConfig(layers={"contours"}, contour_max_crossings=1))
    m = re.search(r'stroke="(#[0-9a-f]{6})" stroke-width="([\d.]+)"', svg)
    assert m is not None
    # max_n=1 is the saturation case: the first crossing uses darkest color and max stroke.
    assert m.group(1) == "#111111"
    assert m.group(2) == "4.00"


def test_contours_reject_nonpositive_max_crossings():
    ws = _small_world()
    with pytest.raises(ValueError, match="contour_max_crossings must be positive"):
        render(ws, SVGConfig(layers={"contours"}, contour_max_crossings=0))


# --- legend ------------------------------------------------------------------


def _legend_panel(svg: str) -> tuple[float, float, float, float]:
    """(x, y, width, height) of the legend's backing panel."""
    import re

    body = svg.split('<g id="layer-legend">')[1]
    m = re.search(r'<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)"', body)
    assert m is not None, "legend panel rect not found"
    return tuple(float(v) for v in m.groups())  # type: ignore[return-value]


def _terrain_boxes(svg: str) -> list[tuple[float, float, float, float]]:
    """Bounding box per drawn hex, as (min_x, min_y, max_x, max_y)."""
    import re

    body = svg.split('<g id="layer-terrain">')[1].split("</g>")[0]
    boxes = []
    for pts in re.findall(r'<polygon points="([^"]+)"', body):
        xs = [float(p.split(",")[0]) for p in pts.split()]
        ys = [float(p.split(",")[1]) for p in pts.split()]
        boxes.append((min(xs), min(ys), max(xs), max(ys)))
    return boxes


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


def test_legend_layer_present_by_default():
    svg = render(_small_world())
    assert 'id="layer-legend"' in svg
    assert ">Legend</text>" in svg


def test_legend_layer_can_be_disabled():
    svg = render(_small_world(), SVGConfig(layers={"terrain"}))
    assert 'id="layer-legend"' not in svg


@pytest.mark.parametrize("corner", ["top-right", "bottom-left"])
def test_legend_sits_in_empty_corner(corner):
    """The panel must not cover any hex — that is the whole point of corner placement."""
    ws = _sheared_world()
    svg = render(ws, SVGConfig(legend_corner=corner))
    lx, ly, lw, lh = _legend_panel(svg)
    overlapping = [
        b
        for b in _terrain_boxes(svg)
        if b[2] > lx and b[0] < lx + lw and b[3] > ly and b[1] < ly + lh
    ]
    assert not overlapping, f"legend panel covers {len(overlapping)} hexes in the {corner} corner"


@pytest.mark.parametrize("corner", ["top-right", "bottom-left"])
def test_legend_hugs_the_map_edge(corner):
    """Clearing the diagonal is not enough — the panel must also stay next to the map.

    The empty triangle is enormous on a wide map, so simply jamming the panel into the
    canvas corner would leave it stranded thousands of pixels from any hex.  The panel is
    placed flush against the bounding diagonal instead; this pins that gap to roughly the
    one-hex margin the placement reserves.
    """
    ws = _sheared_world()
    hex_size = 12.0
    svg = render(ws, SVGConfig(hex_size=hex_size, legend_corner=corner))
    lx, ly, lw, lh = _legend_panel(svg)
    # Hexes in the same columns as the panel, i.e. the ones it is placed against.
    under = [b for b in _terrain_boxes(svg) if b[2] > lx and b[0] < lx + lw]
    assert under, "no hexes share the legend's columns"
    gap = (
        min(b[1] for b in under) - (ly + lh)
        if corner == "top-right"
        else ly - max(b[3] for b in under)
    )
    assert gap > 0, f"legend panel is not clear of the map ({gap:.1f}px)"
    assert gap < 4 * hex_size, f"legend panel is adrift, {gap:.1f}px from the nearest hex"


@pytest.mark.parametrize("corner", ["top-right", "bottom-left"])
def test_legend_stays_inside_canvas(corner):
    import re

    ws = _sheared_world()
    svg = render(ws, SVGConfig(legend_corner=corner))
    cw, ch = (int(v) for v in re.search(r'width="(\d+)" height="(\d+)"', svg).groups())
    lx, ly, lw, lh = _legend_panel(svg)
    assert lx >= 0 and ly >= 0
    assert lx + lw <= cw and ly + lh <= ch


def test_legend_corners_are_on_opposite_sides():
    ws = _sheared_world()
    tr = _legend_panel(render(ws, SVGConfig(legend_corner="top-right")))
    bl = _legend_panel(render(ws, SVGConfig(legend_corner="bottom-left")))
    assert tr[0] > bl[0]  # top-right is further right
    assert tr[1] < bl[1]  # top-right is further up


def test_legend_lists_only_road_tiers_present():
    ws = _small_world()  # PRIMARY only
    body = render(ws).split('<g id="layer-legend">')[1]
    assert "Primary road" in body
    assert "Secondary road" not in body
    assert "Track road" not in body


def test_legend_lists_settlement_tiers_present():
    body = render(_small_world()).split('<g id="layer-legend">')[1]
    for label in ("City", "Town", "Village"):
        assert f">{label}</text>" in body


def test_legend_omits_rows_for_disabled_layers():
    ws = _small_world()
    body = render(ws, SVGConfig(layers={"terrain", "legend"})).split('<g id="layer-legend">')[1]
    assert "Primary road" not in body
    assert ">River</text>" not in body
    assert ">City</text>" not in body


def test_legend_uses_elevation_ramp_in_topographic_style():
    ws = _small_world()
    body = render(ws, SVGConfig(style="topographic")).split('<g id="layer-legend">')[1]
    assert "elevation" in body  # the ramp row, not per-category swatches
    assert ">Grassland</text>" not in body


def test_legend_biome_categories_reflect_world_content():
    ws = _small_world()
    body = render(ws).split('<g id="layer-legend">')[1]
    assert ">Grassland</text>" in body  # the one biome set in _small_world
    assert ">Desert</text>" not in body


def test_legend_scale_grows_the_panel():
    ws = _sheared_world()
    small = _legend_panel(render(ws, SVGConfig(legend_scale=1.0)))
    large = _legend_panel(render(ws, SVGConfig(legend_scale=2.0)))
    assert large[2] > small[2] and large[3] > small[3]


def test_legend_symbols_match_map_symbols():
    """Legend glyphs come from the same helper as the map, so a city is a gold star."""
    ws = _small_world()
    body = render(ws).split('<g id="layer-legend">')[1]
    assert 'fill="gold"' in body  # city star
    assert "#5c3d1e" in body  # PRIMARY road color, same as the roads layer
    assert "#3a78c9" in body  # river blue, same as the rivers layer


def test_legend_rejects_unknown_corner():
    with pytest.raises(ValueError, match="legend_corner must be"):
        render(_small_world(), SVGConfig(legend_corner="middle"))


def test_legend_rejects_nonpositive_scale():
    with pytest.raises(ValueError, match="legend_scale must be positive"):
        render(_small_world(), SVGConfig(legend_scale=0))


def test_legend_skipped_for_empty_world():
    svg = render(WorldState(seed=1, width=0, height=0))
    assert 'id="layer-legend"' not in svg


def test_legend_fits_a_canvas_smaller_than_itself():
    """A 4x4 map is smaller than its own legend; the canvas must grow to fit the panel.

    Clamping cannot rescue a panel larger than the image it sits in — it just crops the
    last rows — so the exporter sizes the legend before the canvas.
    """
    import re

    ws = _small_world()
    svg = render(ws)
    cw, ch = (int(v) for v in re.search(r'width="(\d+)" height="(\d+)"', svg).groups())
    lx, ly, lw, lh = _legend_panel(svg)
    assert lx >= 0 and ly >= 0
    assert lx + lw <= cw, f"panel overflows canvas width by {lx + lw - cw:.1f}px"
    assert ly + lh <= ch, f"panel overflows canvas height by {ly + lh - ch:.1f}px"


@pytest.mark.parametrize("corner", ["top-right", "bottom-left"])
def test_placement_reserves_the_full_hex_support(corner):
    """Placement must reserve the hex polygon's own reach past its centre, not its centre.

    Both regions are separated by the diagonal `d = y - x/sqrt(3)`, so they are disjoint
    exactly when their `d` ranges are — a sharper test than bounding boxes, which
    overestimate a hexagon and go blunt as the panel shrinks.  A small `legend_scale`
    shrinks the safety margin so an under-reserved support cannot hide behind it.
    """
    import math
    import re

    ws = _sheared_world()
    svg = render(ws, SVGConfig(legend_scale=0.15, legend_corner=corner))
    lx, ly, lw, lh = _legend_panel(svg)

    def d(x, y):
        return y - x / math.sqrt(3)

    panel_ds = [d(x, y) for x in (lx, lx + lw) for y in (ly, ly + lh)]
    body = svg.split('<g id="layer-terrain">')[1].split("</g>")[0]
    hex_ds = [
        d(float(p.split(",")[0]), float(p.split(",")[1]))
        for pts in re.findall(r'<polygon points="([^"]+)"', body)
        for p in pts.split()
    ]

    if corner == "top-right":
        assert max(panel_ds) < min(hex_ds), (
            f"panel reaches d={max(panel_ds):.2f}, terrain starts at d={min(hex_ds):.2f}"
        )
    else:
        assert min(panel_ds) > max(hex_ds), (
            f"panel reaches d={min(panel_ds):.2f}, terrain ends at d={max(hex_ds):.2f}"
        )
