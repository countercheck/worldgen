import pytest

from tests.worlds import lay_road
from worldgen.core.hex import (
    Biome,
    LandCover,
    Settlement,
    SettlementRole,
    SettlementTier,
    TerrainClass,
)
from worldgen.core.world_state import Ferry, River, RoadTier, WorldState
from worldgen.export.svg_export import SVGConfig, render, save


def _small_world() -> WorldState:
    ws = WorldState.empty(seed=99, width=4, height=4)
    h = ws.hexes[(0, 0)]
    h.biome = Biome.GRASSLAND
    h.terrain_class = TerrainClass.LAND
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
    lay_road(ws, [(1, 1), (2, 1), (3, 1)], RoadTier.PRIMARY)
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
    assert "#4a2f14" in svg  # PRIMARY road color


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
    ws.hexes[(1, 0)].elevation = 1500.0  # metres: a wall of contours
    config = SVGConfig(layers={"contours"})
    svg = render(ws, config)
    assert 'id="layer-contours"' in svg
    assert "<line" in svg


def test_contours_below_threshold_omitted():
    ws = WorldState.empty(seed=1, width=4, height=4)
    # All hexes at same elevation → no contour lines drawn.
    for h in ws.hexes.values():
        h.elevation = 1500.0
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
    ws.hexes[(1, 0)].elevation = 150.0  # crosses 100 m
    # Pair B: 4 threshold crossings (100, 200, 300, 400 m)
    ws.hexes[(2, 0)].elevation = 0.0  # 0 m
    ws.hexes[(3, 0)].elevation = 450.0  # crosses 100-400 m
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
    ws.hexes[(1, 0)].elevation = 150.0
    # Pair B: saturated crossings → near-black
    ws.hexes[(2, 0)].elevation = 0.0
    ws.hexes[(3, 0)].elevation = 1500.0  # 15 crossings
    config = SVGConfig(layers={"contours"})
    svg = render(ws, config)
    colors = re.findall(r'stroke="(#[0-9a-f]{6})"', svg)
    grays = [int(c[1:3], 16) for c in colors]  # red channel == gray value
    assert min(grays) < max(grays)  # more crossings → darker


def test_contour_single_crossing_uses_lightest_min_styling():
    import re

    ws = WorldState.empty(seed=1, width=4, height=4)
    ws.hexes[(0, 0)].elevation = 0.0
    ws.hexes[(1, 0)].elevation = 150.0  # exactly one threshold crossing at 100 m
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
    ws.hexes[(1, 0)].elevation = 150.0  # one threshold crossing
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
    lay_road(ws, [(1, 1), (2, 1), (3, 1)], RoadTier.PRIMARY)
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
    assert "#4a2f14" in body  # PRIMARY road color, same as the roads layer
    assert "#2f6fbf" in body  # river blue, same as the rivers layer


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


# --- roads and anchorages ----------------------------------------------------


def _roads_group(svg: str) -> str:
    # The line subgroup, not the whole layer: the layer now opens with a casing subgroup
    # drawn under every road, and these tests are about the roads as drawn on top of it.
    return svg.split('<g id="roads-line">')[1].split("</g>")[0]


def _water_crossing_world() -> WorldState:
    """A road that puts to sea mid-route: two land legs, two shore points."""
    ws = WorldState.empty(seed=99, width=6, height=3)
    for r in range(3):
        ws.hexes[(3, r)].terrain_class = TerrainClass.OPEN_WATER
    lay_road(ws, [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1)], RoadTier.PRIMARY)
    return ws


def _branching_world() -> WorldState:
    """A track branching off a primary road, sharing the first two edges with it."""
    ws = WorldState.empty(seed=99, width=5, height=3)
    lay_road(ws, [(0, 1), (1, 1), (2, 1), (3, 1)], RoadTier.PRIMARY)
    lay_road(ws, [(0, 1), (1, 1), (2, 1), (2, 2)], RoadTier.TRACK)
    return ws


def test_shared_road_segments_drawn_once():
    """A branching track must not stack a second polyline on the shared trunk."""
    import re

    body = _roads_group(render(_branching_world()))
    polylines = re.findall(r'<polyline points="([^"]+)"', body)
    assert len(polylines) == 2
    points = [p.split() for p in polylines]
    drawn = [frozenset((a, b)) for pts in points for a, b in zip(pts, pts[1:], strict=False)]
    assert len(drawn) == len(set(drawn)), "an edge was drawn twice"


def test_primary_road_drawn_after_the_track_that_branches_off_it():
    """Paint order is the fix for overdraw — the primary road has to land on top."""
    body = _roads_group(render(_branching_world()))
    assert body.index('stroke="#8a6a34"') < body.index('stroke="#4a2f14"')


def test_anchorage_layer_marks_both_shores():
    svg = render(_water_crossing_world())
    assert 'id="layer-anchorages"' in svg
    group = svg.split('<g id="layer-anchorages">')[1].split("  </g>")[0]
    assert group.count('stroke="#1b3a5c"') == 2


def test_anchorage_layer_absent_when_no_road_meets_water():
    """_small_world's road stays on land, so there is nothing to mark."""
    assert 'id="layer-anchorages"' not in render(_small_world())


def test_anchorage_layer_can_be_disabled():
    ws = _water_crossing_world()
    svg = render(ws, SVGConfig(layers={"terrain", "roads"}))
    assert 'id="layer-anchorages"' not in svg


def test_anchorage_markers_sit_on_the_land_shores():
    """The markers belong on the two shore hexes, not on the water between them.

    Absolute positions carry the layout offset, so compare the gap between the two
    markers with the gap between the shore hexes (2,1) and (4,1) themselves.
    """
    import re

    from worldgen.core.hex_grid import axial_to_pixel

    hex_size = 12.0
    svg = render(_water_crossing_world(), SVGConfig(hex_size=hex_size, layers={"anchorages"}))
    group = svg.split('<g id="layer-anchorages">')[1].split("  </g>")[0]
    drawn_x = sorted(float(x) for x in re.findall(r'<circle cx="([\d.-]+)"', group))
    assert len(drawn_x) == 2
    shore_gap = axial_to_pixel((4, 1), hex_size)[0] - axial_to_pixel((2, 1), hex_size)[0]
    assert abs((drawn_x[1] - drawn_x[0]) - shore_gap) < 0.01


def test_legend_lists_anchorage_when_a_road_meets_water():
    body = render(_water_crossing_world()).split('<g id="layer-legend">')[1]
    assert ">Anchorage</text>" in body


def test_legend_omits_anchorage_when_no_road_meets_water():
    body = render(_small_world()).split('<g id="layer-legend">')[1]
    assert ">Anchorage</text>" not in body


def test_wargame_style_includes_anchorages():
    svg = render(_water_crossing_world(), SVGConfig(style="wargame"))
    assert 'id="layer-anchorages"' in svg


def test_ferry_landings_draw_anchorages():
    """A ferry stands in for a road where a river channel cuts the network in two."""
    ws = _small_world()
    ws.ferries = [Ferry(a=(0, 1), b=(2, 1))]
    svg = render(ws)
    assert 'id="layer-anchorages"' in svg
    group = svg.split('<g id="layer-anchorages">')[1].split("  </g>")[0]
    assert group.count('stroke="#1b3a5c"') == 2


def test_ferry_puts_an_anchorage_row_in_the_legend():
    ws = _small_world()
    ws.ferries = [Ferry(a=(0, 1), b=(2, 1))]
    body = render(ws).split('<g id="layer-legend">')[1]
    assert ">Anchorage</text>" in body


# --- fords and bridges -------------------------------------------------------


def _crossing_world() -> WorldState:
    """A river with a road crossing it: one hex tagged ford, one tagged bridge."""
    ws = WorldState.empty(seed=5, width=5, height=5)
    ws.rivers = [River(hexes=[(2, 0), (2, 1), (2, 2), (2, 3)], flow_volume=1.0)]
    for r in range(4):
        ws.hexes[(2, r)].river_flow = 0.8
        ws.hexes[(2, r)].tags.add("river")
    ws.hexes[(2, 1)].tags.add("ford")
    ws.hexes[(2, 2)].tags.add("bridge")
    lay_road(ws, [(1, 1), (2, 1), (3, 1)], RoadTier.PRIMARY)
    return ws


def _crossings_group(svg: str) -> str:
    return svg.split('<g id="layer-crossings">')[1].split("\n  </g>")[0]


def test_crossings_layer_draws_a_symbol_per_tagged_hex():
    svg = render(_crossing_world())
    assert 'id="layer-crossings"' in svg
    group = _crossings_group(svg)
    assert group.count('stroke="#2b2118"') == 2


def test_ford_is_dashed_and_bridge_is_not():
    """The two must be tellable apart at a glance, not just present."""
    group = _crossings_group(render(_crossing_world()))
    marks = group.strip().splitlines()
    ford = next(m for m in marks if "stroke-dasharray" in m)
    bridge = next(m for m in marks if "stroke-dasharray" not in m)
    # The bridge carries abutments (4 lines) where the ford is just the broken span (2).
    assert bridge.count("<line") == 4
    assert ford.count("<line") == 2


def test_crossings_are_rotated_square_to_the_river():
    """A span drawn along the current would read as a second river."""
    import re

    group = _crossings_group(render(_crossing_world()))
    angles = [float(a) for a in re.findall(r"rotate\(([-\d.]+)", group)]
    assert angles, "crossing symbols are not rotated at all"
    # River runs down a column here; the span must not be parallel to it.
    assert all(abs((a % 180) - 90.0) > 1.0 for a in angles), angles


def test_crossings_layer_can_be_disabled():
    svg = render(_crossing_world(), SVGConfig(layers={"terrain", "roads"}))
    assert 'id="layer-crossings"' not in svg


def test_crossings_layer_absent_when_nothing_is_tagged():
    assert 'id="layer-crossings"' not in render(_small_world())


def test_legend_lists_ford_and_bridge():
    body = render(_crossing_world()).split('<g id="layer-legend">')[1]
    assert ">Ford</text>" in body
    assert ">Bridge</text>" in body


def test_legend_omits_crossings_not_present():
    """Only the kinds the map actually contains earn a row."""
    ws = _crossing_world()
    ws.hexes[(2, 2)].tags.discard("bridge")
    body = render(ws).split('<g id="layer-legend">')[1]
    assert ">Ford</text>" in body
    assert ">Bridge</text>" not in body


def test_wargame_style_draws_roads_rivers_and_crossings():
    """The wargame preset exists to be read while moving units."""
    svg = render(_crossing_world(), SVGConfig(style="wargame"))
    for layer in ("roads", "rivers", "crossings", "settlements", "terrain"):
        assert f'id="layer-{layer}"' in svg, f"wargame style is missing the {layer} layer"


def test_anchorage_not_drawn_for_an_undrawable_land_leg():
    """A one-hex land leg draws no road, so its shore must not carry a lone anchor.

    `split_path_on_water` discards runs shorter than two hexes — marking their shore
    left anchors sitting on the coast with nothing attached.
    """
    ws = WorldState.empty(seed=3, width=5, height=3)
    for r in range(3):
        ws.hexes[(2, r)].terrain_class = TerrainClass.OPEN_WATER
    # (1,1) is a single land hex before the water: no polyline, so no anchorage.
    lay_road(ws, [(1, 1), (2, 1), (3, 1), (4, 1)], RoadTier.PRIMARY)
    svg = render(ws, SVGConfig(layers={"roads", "anchorages"}))
    group = svg.split('<g id="layer-anchorages">')[1].split("  </g>")[0]
    assert group.count('stroke="#1b3a5c"') == 1  # only the drawn (3,1)-(4,1) leg's shore


# --- river widths ------------------------------------------------------------


def _flowing_river_world() -> WorldState:
    """One river gaining flow from headwater to mouth."""
    ws = WorldState.empty(seed=11, width=6, height=3)
    path = [(q, 1) for q in range(6)]
    for i, c in enumerate(path):
        ws.hexes[c].river_flow = 0.05 + i * 0.19
    ws.rivers = [River(hexes=path, flow_volume=1.0)]
    return ws


def _river_widths(svg: str) -> list[float]:
    import re

    body = svg.split('<g id="rivers-line">')[1].split("</g>")[0]
    return [float(w) for w in re.findall(r'stroke-width="([\d.]+)"', body)]


def test_river_is_drawn_at_several_widths():
    """A river should visibly grow downstream, not be one width taken from its mouth."""
    widths = _river_widths(render(_flowing_river_world()))
    assert len(set(widths)) > 1, f"river drawn at a single width: {widths}"


def test_river_widths_increase_downstream():
    widths = _river_widths(render(_flowing_river_world()))
    assert widths == sorted(widths), f"river narrows downstream: {widths}"


def test_river_widths_respect_the_configured_range():
    svg = render(_flowing_river_world(), SVGConfig(river_min_width=2.0, river_max_width=7.0))
    for w in _river_widths(svg):
        assert 2.0 <= w <= 7.0


def test_one_width_step_draws_a_uniform_river():
    svg = render(_flowing_river_world(), SVGConfig(river_width_steps=1))
    assert len(set(_river_widths(svg))) == 1


def test_bigger_river_is_drawn_wider_than_a_smaller_one():
    """Two rivers on one map must be rankable by eye."""
    ws = _flowing_river_world()
    trickle = [(q, 0) for q in range(4)]
    for c in trickle:
        ws.hexes[c].river_flow = 0.05
    ws.rivers.append(River(hexes=trickle, flow_volume=0.05))
    widths = _river_widths(render(ws))
    assert max(widths) > min(widths)


def test_river_width_settings_are_validated():
    with pytest.raises(ValueError, match="river_width_steps must be >= 0"):
        render(_flowing_river_world(), SVGConfig(river_width_steps=-1))
    with pytest.raises(ValueError, match="river_max_width must be >="):
        render(_flowing_river_world(), SVGConfig(river_min_width=5.0, river_max_width=1.0))


# --- line widths scale with hex size -----------------------------------------


def _road_widths(svg: str) -> list[float]:
    import re

    body = _roads_group(svg)
    return [float(w) for w in re.findall(r'stroke-width="([\d.]+)"', body)]


def test_river_widths_scale_with_hex_size():
    """A 30px hex must not get hairline rivers, nor a 6px hex drown in them."""
    ws = _flowing_river_world()
    small = _river_widths(render(ws, SVGConfig(hex_size=12.0)))
    large = _river_widths(render(ws, SVGConfig(hex_size=24.0)))
    assert len(small) == len(large)
    for a, b in zip(small, large, strict=True):
        assert b == pytest.approx(a * 2, rel=0.02)


def test_road_widths_scale_with_hex_size():
    ws = _branching_world()
    small = _road_widths(render(ws, SVGConfig(hex_size=12.0)))
    large = _road_widths(render(ws, SVGConfig(hex_size=36.0)))
    assert len(small) == len(large)
    for a, b in zip(small, large, strict=True):
        assert b == pytest.approx(a * 3, rel=0.02)


def test_track_dashes_scale_with_hex_size():
    """Unscaled dashes crowd into a solid line on a big export."""
    import re

    ws = _branching_world()

    def dashes(hex_size):
        body = _roads_group(render(ws, SVGConfig(hex_size=hex_size)))
        m = re.search(r'stroke-dasharray="([\d. ]+)"', body)
        assert m is not None, "no dashed track found"
        return [float(v) for v in m.group(1).split()]

    small, large = dashes(12.0), dashes(24.0)
    for a, b in zip(small, large, strict=True):
        assert b == pytest.approx(a * 2, rel=0.02)


def test_reference_hex_size_leaves_widths_unchanged():
    """12px is the reference the widths are written for, so it must scale by exactly 1."""
    body = _roads_group(render(_branching_world(), SVGConfig(hex_size=12.0)))
    assert 'stroke-width="3.00"' in body  # PRIMARY, as configured
    assert 'stroke-width="1.20"' in body  # TRACK


def test_legend_line_glyphs_scale_with_legend_scale():
    import re

    ws = _small_world()

    def widths(scale):
        body = render(ws, SVGConfig(legend_scale=scale)).split('<g id="layer-legend">')[1]
        return [float(w) for w in re.findall(r'stroke-width="([\d.]+)"', body)]

    assert max(widths(2.0)) > max(widths(1.0))


def test_elevation_fills_are_legal_colours():
    """Metre elevations must be normalised before they reach a colour channel.

    Fed raw, a 585 m hex formatted as `#2469a2469a2469a` and a below-sea hex went
    negative — every fill in elevation mode must be a well-formed six-digit colour,
    whatever the map's vertical span.
    """
    import re

    ws = WorldState.empty(seed=1, width=4, height=4)
    for i, h in enumerate(ws.hexes.values()):
        h.elevation = -120.0 + i * 90.0  # metres, spanning below and above sea level

    svg = render(ws, SVGConfig(color_mode="elevation", layers={"terrain"}))
    fills = re.findall(r'fill="([^"]+)"', svg)
    assert fills
    for fill in fills:
        assert re.fullmatch(r"#[0-9a-f]{6}|none", fill), f"illegal fill {fill!r}"
