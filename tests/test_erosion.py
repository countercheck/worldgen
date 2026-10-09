"""Erosion's helpers, called directly.

`erosion.py` reads 72% covered and that figure is misleading: `_drop_particle`,
`_deposit_delta` and the rest of the droplet loop are `@numba.njit`, and coverage.py
cannot instrument a JIT-compiled function.  Run the same suite under
`NUMBA_DISABLE_JIT=1` and `erosion.py` jumps to 96% — the droplet code is exercised
heavily, the tracer simply cannot see it.  CI does not measure it that way because the
suite takes 75x longer without numba.

What is genuinely untested is the plain-Python scaffolding around the hot loops: the
guard clauses that decide whether the widening runs at all, and where an off-map river
is admitted.  Those are what this file covers, by calling the functions rather than
generating a world and hoping the branch is reached.
"""

import numpy as np

from worldgen.core.world_state import WorldState
from worldgen.stages.erosion import _inflow_mouths, _neighbour_table, _widen_valleys


def _sloping_shelf(w=12, h=12):
    """Land falling away from the west edge, so a west inlet has somewhere to run."""
    arr = np.zeros((w, h))
    for i in range(w):
        for j in range(h):
            arr[i, j] = 1.0 - 0.05 * i
    state = WorldState.empty(seed=1, width=w, height=h)
    return arr, state, _neighbour_table(state, w, h)


# --- _inflow_mouths -------------------------------------------------------------


def test_no_inflows_are_admitted_when_none_are_asked_for():
    arr, state, nbrs = _sloping_shelf()
    assert _inflow_mouths(arr, 0.0, state, nbrs, ("west",), 0, 3) == []


def test_inflows_are_capped_at_the_count_asked_for():
    """The ranking is by drop, so without a cap the whole west edge would qualify."""
    arr, state, nbrs = _sloping_shelf()
    unlimited = _inflow_mouths(arr, 0.0, state, nbrs, ("west",), 99, 1)
    assert len(unlimited) > 2, "fixture is meant to offer several candidate mouths"

    capped = _inflow_mouths(arr, 0.0, state, nbrs, ("west",), 2, 1)
    assert len(capped) == 2
    assert capped == unlimited[:2], "the cap should keep the best, not an arbitrary two"


def test_inflows_only_come_from_the_edges_named():
    arr, state, nbrs = _sloping_shelf()
    w, h = arr.shape
    for i, j in _inflow_mouths(arr, 0.0, state, nbrs, ("west",), 4, 1):
        assert i == 0, f"mouth at {(i, j)} is not on the west edge"


# --- _widen_valleys guards ------------------------------------------------------


def _one_channel(w=15, h=9, col=7, flow=50.0):
    arr = np.full((w, h), 0.5)
    arr[col, :] = 0.3
    discharge = np.ones((w, h))
    discharge[col, :] = flow
    return arr, discharge


def test_widening_is_a_no_op_with_no_land():
    """An all-ocean map has no valley to widen and must not raise."""
    arr, discharge = _one_channel()
    before = arr.copy()
    _widen_valleys(
        arr,
        discharge,
        sea_level=99.0,
        width_max=6.0,
        width_exponent=0.6,
        floor_slope=0.001,
        max_relief=0.5,
        channel_fraction=0.05,
    )
    assert (arr == before).all()


def test_widening_is_a_no_op_with_no_discharge():
    """Before the first accumulation pass every cell carries nothing."""
    arr, _ = _one_channel()
    before = arr.copy()
    _widen_valleys(
        arr,
        np.zeros_like(arr),
        sea_level=0.0,
        width_max=6.0,
        width_exponent=0.6,
        floor_slope=0.001,
        max_relief=0.5,
        channel_fraction=0.05,
    )
    assert (arr == before).all()


def test_widening_is_a_no_op_when_width_max_is_zero():
    arr, discharge = _one_channel()
    before = arr.copy()
    _widen_valleys(
        arr,
        discharge,
        sea_level=0.0,
        width_max=0.0,
        width_exponent=0.6,
        floor_slope=0.001,
        max_relief=0.5,
        channel_fraction=0.05,
    )
    assert (arr == before).all()


# --- the size dependence recorded as tech-debt issue #116 ---------------------


def _many_channels(w=41, h=9, cols=(6, 13, 20, 27, 34), flow=50.0, trunk=None):
    """Several equal channels, optionally beside one carrying far more.

    *trunk* stands for a river entering from off the map: `_inflow_mouths` seeds it with
    `river_inflow_volume x land area`, which is far more than any river the map raises
    for itself.
    """
    arr = np.full((w, h), 0.5)
    discharge = np.ones((w, h))
    for c in cols:
        arr[c, :] = 0.3
        discharge[c, :] = flow
    if trunk is not None:
        arr[w - 3, :] = 0.3
        discharge[w - 3, :] = trunk
    return arr, discharge


def _channels_given_a_belt(arr, discharge, cols):
    before = arr.copy()
    _widen_valleys(
        arr,
        discharge,
        sea_level=0.0,
        width_max=6.0,
        width_exponent=0.6,
        floor_slope=0.001,
        max_relief=0.5,
        channel_fraction=0.4,
    )
    return [c for c in cols if (arr[c - 1, :] < before[c - 1, :]).any()]


def test_every_ordinary_channel_gets_a_floodplain_when_none_dominates():
    cols = (6, 13, 20, 27, 34)
    arr, discharge = _many_channels(cols=cols)
    assert _channels_given_a_belt(arr, discharge, cols) == list(cols)


def test_an_imported_trunk_river_does_not_strip_the_other_channels():
    """One very large channel must not cost the ordinary ones their floodplains.

    This is the whole of item 19 in eight lines. `_widen_valleys` sizes each belt as
    `width_max * (flow / flow.max()) ** width_exponent` and drops the channel entirely
    when that falls under one cell, so the belt a river gets depends on the largest
    river anywhere on the map rather than on its own size. An off-map inflow is seeded
    with a catchment no on-map river can match, and every other valley goes bare.

    Flipping this to a pass is the acceptance test for the fix.
    """
    cols = (6, 13, 20, 27, 34)
    arr, discharge = _many_channels(cols=cols, trunk=5_000.0)
    assert _channels_given_a_belt(arr, discharge, cols) == list(cols)


def _belt_widths(arr, discharge, cols):
    """How many cells of row 4 each channel's widening lowered, by channel."""
    before = arr.copy()
    _widen_valleys(
        arr,
        discharge,
        sea_level=0.0,
        width_max=6.0,
        width_exponent=0.6,
        floor_slope=0.001,
        max_relief=0.5,
        channel_fraction=0.4,
    )
    cut = before[:, 4] - arr[:, 4] > 1e-12
    return [int(cut[c - 3 : c + 4].sum()) for c in cols]


def test_a_belt_is_sized_by_its_own_river_not_the_largest_on_the_map():
    """Adding a far bigger river elsewhere leaves every other belt exactly as wide.

    A floodplain is laid down by its own river wandering, so its width is a matter of
    that river's discharge.  Sized against the largest flow on the map instead, the belt
    a channel got depended on what else happened to be on the map with it.
    """
    cols = (6, 13, 20, 27, 34)
    alone = _belt_widths(*_many_channels(cols=cols), cols)
    beside_a_trunk = _belt_widths(*_many_channels(cols=cols, trunk=5_000.0), cols)
    assert all(width > 0 for width in alone)
    assert beside_a_trunk == alone


def test_a_belt_is_the_same_width_on_a_bigger_map():
    """The same channel on a map three times as wide gets the same belt.

    A bigger map raises a bigger trunk of its own, without any river from off the map:
    the larger map here carries one ten times the size, as a native trunk would be.
    """
    small = _belt_widths(*_many_channels(w=15, cols=(7,)), (7,))
    big = _belt_widths(*_many_channels(w=45, cols=(7,), trunk=500.0), (7,))
    assert small[0] > 0
    assert big == small
