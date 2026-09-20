import pytest

# Both fixtures here were near-identical 32x32 worlds differing only in a traveller
# count no rendering test reads. They are one memoised `small_state` now; see
# tests/conftest.py.


def test_render_roads_produces_file(small_state, tmp_path):
    from worldgen.render.debug_viewer import render

    out = tmp_path / "roads.svg"
    render(small_state, "roads", str(out))
    assert out.exists() and out.stat().st_size > 0
    assert out.read_text().startswith("<svg")


def test_render_unknown_attribute_raises(small_state, tmp_path):
    from worldgen.render.debug_viewer import render

    with pytest.raises(ValueError, match="Unknown attribute"):
        render(small_state, "nonexistent", str(tmp_path / "x.svg"))


def test_render_land_cover_produces_file(small_state, tmp_path):
    from worldgen.render.debug_viewer import render

    out = tmp_path / "land_cover.svg"
    render(small_state, "land_cover", str(out))
    assert out.exists() and out.stat().st_size > 0


def test_render_cultivation_produces_file(small_state, tmp_path):
    from worldgen.render.debug_viewer import render

    out = tmp_path / "cultivation.svg"
    render(small_state, "cultivation", str(out))
    assert out.exists() and out.stat().st_size > 0


def test_debug_viewer_paints_the_primary_road_over_the_branching_track():
    """Iterating RoadTier drew tracks last, so a branch painted over its own trunk."""
    import re

    from tests.worlds import lay_road
    from worldgen.core.world_state import RoadTier, WorldState
    from worldgen.render.debug_viewer import render_svg

    ws = WorldState.empty(seed=1, width=5, height=3)
    lay_road(ws, [(0, 1), (1, 1), (2, 1), (3, 1)], RoadTier.PRIMARY)
    lay_road(ws, [(0, 1), (1, 1), (2, 1), (2, 2)], RoadTier.TRACK)
    body = render_svg(ws, "roads").split('<g id="layer-roads">')[1].split("</g>")[0]

    assert body.index('stroke="#b8a070"') < body.index('stroke="#5c3d1e"')
    # The shared trunk is drawn once, by the primary road only.
    polylines = re.findall(r'<polyline points="([^"]+)"', body)
    assert len(polylines) == 2
    edges = [
        frozenset((a, b))
        for pts in (p.split() for p in polylines)
        for a, b in zip(pts, pts[1:], strict=False)
    ]
    assert len(edges) == len(set(edges)), "an edge was drawn twice"


def test_render_drainage_marks_confluences_and_scales_by_order():
    """The plate must show branching, not just wet hexes."""
    from worldgen.core.hex import Hex
    from worldgen.core.world_state import River, WorldState
    from worldgen.render.debug_viewer import render_svg

    state = WorldState.empty(seed=1, width=1, height=1)
    state.hexes = {(q, r): Hex(coord=(q, r)) for q in range(-1, 4) for r in range(-2, 2)}
    state.rivers = [
        River(hexes=[(0, 0), (1, 0), (2, 0)], flow_volume=1.0),
        River(hexes=[(1, -1), (2, -1), (2, 0)], flow_volume=1.0),
        River(hexes=[(2, 0), (3, 0)], flow_volume=1.0),
    ]
    svg = render_svg(state, "drainage")
    assert 'id="layer-drainage"' in svg
    assert 'fill="#d95f02"' in svg, "the confluence should be marked"
    widths = {
        line.split('stroke-width="')[1].split('"')[0]
        for line in svg.splitlines()
        if "polyline" in line
    }
    assert len(widths) >= 2, f"trunk and tributary should differ in width: {widths}"


def test_render_drainage_produces_a_file(small_state, tmp_path):
    from worldgen.render.debug_viewer import render

    out = tmp_path / "drainage.svg"
    render(small_state, "drainage", str(out))
    assert out.exists() and out.stat().st_size > 0


def test_the_drainage_plate_agrees_with_its_own_caption(small_state):
    """The plate prints its own numbers, so what it draws has to be what it counts.

    Two ways this drifted before: the markers were drawn for every junction in the graph
    while the caption counted land junctions only, so a hex where two rivers merely arrive
    at the same lake got a dot and no tally; and one-hex links were dropped from the
    drawing while still feeding the N1/N2 and bifurcation figures in the caption.
    """
    import re

    from worldgen.analysis import build_network, drainage_metrics, links_by_order
    from worldgen.render.debug_viewer import render_svg

    svg = render_svg(small_state, "drainage")
    metrics = drainage_metrics(small_state)

    caption = re.search(r"confluences (\d+)", svg)
    assert caption is not None, "the plate should state its own numbers"
    assert int(caption.group(1)) == metrics.confluence_count

    markers = svg.count('fill="#d95f02"')
    assert markers == metrics.confluence_count, (
        f"plate draws {markers} confluence markers, caption counts {metrics.confluence_count}"
    )

    # Every link counted is a link drawn — as a polyline, or as a dot where a one-hex link
    # has no direction to draw. The confluence markers are the other circles, so take them
    # off before comparing.
    drawn = svg.count("<polyline") + svg.count("<circle") - markers
    assert drawn == sum(links_by_order(build_network(small_state)).values())
