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
