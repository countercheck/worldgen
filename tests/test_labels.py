"""Label placement: rank by size, rivers along the water, nothing on top of anything."""

import pytest

from tests.worlds import build_world
from worldgen.core.hex import SettlementTier
from worldgen.core.hex_grid import axial_to_pixel
from worldgen.export import png_export, svg_export
from worldgen.export.labels import estimate_width, place_labels

_SIZE = 12.0


@pytest.fixture(scope="module")
def world():
    return build_world(model="classic", width=64, height=64)


def _centre(coord):
    return axial_to_pixel(coord, _SIZE)


def _box(lb):
    w, h = estimate_width(lb.text, lb.size, lb.bold)
    return (lb.x - w / 2, lb.y - h / 2, lb.x + w / 2, lb.y + h / 2)


def test_no_two_horizontal_labels_overlap(world):
    placed = [lb for lb in place_labels(world, _centre, _SIZE) if not lb.angle]
    boxes = [_box(lb) for lb in placed]
    for i, (a0, b0, a1, b1) in enumerate(boxes):
        for x0, y0, x1, y1 in boxes[i + 1 :]:
            assert a1 <= x0 or a0 >= x1 or b1 <= y0 or b0 >= y1


def test_a_label_never_sits_on_a_settlement_marker(world):
    placed = [lb for lb in place_labels(world, _centre, _SIZE) if not lb.river]
    for lb in placed:
        x0, y0, x1, y1 = _box(lb)
        for s in world.settlements:
            x, y = _centre(s.coord)
            assert not (x0 < x < x1 and y0 < y < y1), (lb.text, s.name)


def test_size_says_rank(world):
    placed = {lb.text: lb for lb in place_labels(world, _centre, _SIZE)}
    size = {}
    for s in world.settlements:
        if s.name in placed:
            size.setdefault(s.tier, set()).add(placed[s.name].size)
    if SettlementTier.CITY in size and SettlementTier.TOWN in size:
        assert min(size[SettlementTier.CITY]) > max(size[SettlementTier.TOWN])
    if SettlementTier.TOWN in size and SettlementTier.VILLAGE in size:
        assert min(size[SettlementTier.TOWN]) > max(size[SettlementTier.VILLAGE])


def test_every_city_is_labelled(world):
    """Cities are set first, so a crowded map drops villages before it drops them."""
    placed = {lb.text for lb in place_labels(world, _centre, _SIZE)}
    for s in world.settlements:
        if s.tier is SettlementTier.CITY:
            assert s.name in placed


def test_river_labels_name_named_rivers_and_read_left_to_right(world):
    rivers = [lb for lb in place_labels(world, _centre, _SIZE) if lb.river]
    names = {r.name for r in world.rivers if r.name}
    for lb in rivers:
        assert lb.text in names
        assert lb.italic
        assert -90 < lb.angle <= 90


def test_no_river_label_without_the_river(world):
    assert not [lb for lb in place_labels(world, _centre, _SIZE, rivers=False) if lb.river]


def test_the_svg_carries_the_city_names_bold(world):
    svg = svg_export.render(world)
    for s in world.settlements:
        if s.tier is SettlementTier.CITY:
            assert f">{svg_export._xml_escape(s.name)}</text>" in svg
    assert 'font-weight="bold"' in svg


def test_the_png_draws_labels_without_error(world):
    img = png_export.render(world)
    assert img.width > 0
