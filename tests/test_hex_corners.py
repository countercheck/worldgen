"""Corners and sides of the hex grid: one name each, checked against the drawn geometry."""

import math

import pytest

from worldgen.core.hex_grid import (
    axial_to_pixel,
    corner_hexes,
    corner_id,
    corner_neighbors,
    corner_of,
    corner_sides,
    corner_to_pixel,
    hex_corner_keys,
    hex_range,
    hex_side_keys,
    neighbors,
    parse_corner,
    parse_side,
    side_between,
    side_corners,
    side_hexes,
    side_id,
)

SIZE = 10.0
HEXES = hex_range((0, 0), 4) + [(-7, 3), (5, -9), (11, 2)]


def _drawn_corner(coord, k):
    cx, cy = axial_to_pixel(coord, SIZE)
    a = math.radians(60 * k)
    return cx + SIZE * math.cos(a), cy + SIZE * math.sin(a)


def _same(p, q):
    return math.dist(p, q) < 1e-9


@pytest.mark.parametrize("coord", HEXES)
def test_a_corner_name_is_where_the_hex_draws_that_corner(coord):
    for k in range(6):
        assert _same(corner_to_pixel(corner_of(coord, k), SIZE), _drawn_corner(coord, k))


@pytest.mark.parametrize("coord", HEXES)
def test_a_hex_has_six_distinct_corners_and_sides(coord):
    assert len(set(hex_corner_keys(coord))) == 6
    assert len(set(hex_side_keys(coord))) == 6
    assert all(c[2] in (0, 1) for c in hex_corner_keys(coord))
    assert all(s[2] in (0, 1, 2) for s in hex_side_keys(coord))


@pytest.mark.parametrize("coord", HEXES)
def test_three_hexes_meet_at_every_corner(coord):
    for corner in hex_corner_keys(coord):
        hexes = corner_hexes(corner)
        assert len(set(hexes)) == 3
        assert coord in hexes
        # Each of them names this corner too, and each is a neighbour of the other two.
        for h in hexes:
            assert corner in hex_corner_keys(h)
            assert all(o in neighbors(h) for o in hexes if o != h)
        p = corner_to_pixel(corner, SIZE)
        assert all(math.dist(p, axial_to_pixel(h, SIZE)) == pytest.approx(SIZE) for h in hexes)


def test_every_corner_is_owned_by_exactly_one_hex():
    # Two owned corners per hex is exactly enough: a patch of the grid names no corner
    # twice, and every hex's corners are all among the names its patch produces.
    patch = hex_range((0, 0), 6)
    owned = [(q, r, k) for q, r in patch for k in (0, 1)]
    assert len(owned) == len(set(owned))
    inner = hex_range((0, 0), 5)
    assert {c for h in inner for c in hex_corner_keys(h)} <= set(owned)


@pytest.mark.parametrize("coord", HEXES)
def test_corner_neighbours_are_one_side_away_and_mutual(coord):
    for corner in hex_corner_keys(coord):
        p = corner_to_pixel(corner, SIZE)
        near = corner_neighbors(corner)
        assert len(set(near)) == 3
        for n in near:
            assert math.dist(p, corner_to_pixel(n, SIZE)) == pytest.approx(SIZE)
            assert corner in corner_neighbors(n)


@pytest.mark.parametrize("coord", HEXES)
def test_a_side_is_named_the_same_from_either_hex(coord):
    for n in neighbors(coord):
        side = side_between(coord, n)
        assert side == side_between(n, coord)
        assert set(side_hexes(side)) == {coord, n}
        assert side in hex_side_keys(coord) and side in hex_side_keys(n)


@pytest.mark.parametrize("coord", HEXES)
def test_a_side_runs_between_the_two_hexes(coord):
    for i, side in enumerate(hex_side_keys(coord)):
        a, b = side_corners(side)
        assert b in corner_neighbors(a)
        # Both ends are corners of both hexes, and the side is drawn where hex side i is.
        for h in side_hexes(side):
            assert {a, b} <= set(hex_corner_keys(h))
        drawn = {_drawn_corner(coord, i), _drawn_corner(coord, i + 1)}
        ends = {corner_to_pixel(a, SIZE), corner_to_pixel(b, SIZE)}
        assert all(any(_same(e, d) for d in drawn) for e in ends)


@pytest.mark.parametrize("coord", HEXES)
def test_corner_sides_join_the_corner_to_each_neighbour(coord):
    for corner in hex_corner_keys(coord):
        for n, side in zip(corner_neighbors(corner), corner_sides(corner), strict=True):
            assert set(side_corners(side)) == {corner, n}


def test_side_between_refuses_hexes_that_do_not_touch():
    with pytest.raises(ValueError):
        side_between((0, 0), (2, 0))


def test_string_keys_round_trip():
    for coord in HEXES:
        for c in hex_corner_keys(coord):
            assert parse_corner(corner_id(c)) == c
        for s in hex_side_keys(coord):
            assert parse_side(side_id(s)) == s
    assert side_id((-3, 4, 2)) == "-3,4,2"
    with pytest.raises(ValueError):
        parse_corner("0,0,2")
    with pytest.raises(ValueError):
        parse_side("0,0,3")
