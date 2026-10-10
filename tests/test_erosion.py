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
import pytest

from worldgen.core.world_state import WorldState
from worldgen.stages.erosion import _choose_inlets, _course, _drain, _widen_valleys


def _sloping_shelf(w=12, h=12):
    """Land falling away from the west edge, so a west inlet has somewhere to run."""
    arr = np.zeros((w, h))
    for i in range(w):
        for j in range(h):
            arr[i, j] = 1.0 - 0.05 * i + 0.001 * j
    state = WorldState.empty(seed=1, width=w, height=h)
    net, drainage = _drain(arr, 0.0, state, {}, 2.0, np.random.default_rng(0))
    return arr, state, net, drainage


def _choose(arr, state, net, drainage, rng=None, **over):
    kw = dict(edges=("west",), count=2, separation=1, min_length_km=1.0, length_bias=2.0)
    kw.update(over)
    return _choose_inlets(arr, 0.0, state, net, drainage, rng or np.random.default_rng(3), **kw)


# --- _choose_inlets -------------------------------------------------------------


def test_no_inflows_are_admitted_when_none_are_asked_for():
    assert _choose(*_sloping_shelf(), count=0) == []


def test_inflows_are_capped_at_the_count_asked_for():
    shelf = _sloping_shelf()
    assert len(_choose(*shelf, count=99)) > 2, "fixture is meant to offer several inlets"
    assert len(_choose(*shelf, count=2)) == 2


def test_inflows_only_come_from_the_edges_named():
    for i, _j in _choose(*_sloping_shelf(), count=4):
        assert i == 0, "an inlet off the west edge was admitted"


def test_an_inlet_has_a_course_at_least_as_long_as_asked():
    """Length is the rule: a hex whose water meets the sea a step later is no inlet."""
    arr, state, net, drainage = _sloping_shelf()
    from worldgen.stages.corner_drainage import inlet_corner

    for cell in _choose(arr, state, net, drainage, count=4, min_length_km=4.0):
        head = inlet_corner(state.coord_at(*cell), net, drainage)
        assert (len(_course(drainage, head)) - 1) / 3**0.5 >= 4.0
    assert _choose(arr, state, net, drainage, min_length_km=1000.0) == []


def test_no_candidate_means_no_draw():
    """A map whose border is all sea admits no inlet, and must not move the generator.

    Every later draw the stage makes would shift with it, and the sea-ring default map —
    which never admits an inlet — would stop being the world it was.
    """
    shelf = _sloping_shelf()
    rng = np.random.default_rng(5)
    before = rng.bit_generator.state
    assert _choose(*shelf, rng=rng, edges=("east",), min_length_km=1000.0) == []
    assert rng.bit_generator.state == before


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

    *trunk* stands for a river entering from off the map: erosion seeds it with
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


# --- the load an off-map river brings (tech-debt #119) ---------------------------

_SEA = 0.5


def _valley(w=30, h=11, sea_from=24, belt=True):
    """A river down the middle row of a gentle slope into the sea, with its valley belt.

    The sea reaches the east edge, so it is the open sea. Discharge is a trunk's along the
    middle row and a trickle elsewhere; the belt is three cells either side of it, credited
    as `_widen_valleys` credits one, most beside the channel.
    """
    arr = np.zeros((w, h))
    mid = h // 2
    for i in range(w):
        arr[i, :] = 0.9 - 0.01 * i if i < sea_from else 0.3
        if i < sea_from:
            arr[i, :] += 0.002 * np.abs(np.arange(h) - mid)
    discharge = np.ones((w, h))
    discharge[:, mid] = 1_000.0
    meander = np.zeros((w, h))
    if belt:
        for j in range(h):
            d = abs(j - mid)
            if 0 < d <= 3:
                meander[:sea_from, j] = 1.0 - d / 4.0
    return arr, discharge, meander, WorldState.empty(seed=1, width=w, height=h)


def _stage(**over):
    from worldgen.core.config import WorldConfig
    from worldgen.stages.erosion import ErosionStage

    cfg = WorldConfig(valley_channel_fraction=0.02, **over)
    return ErosionStage(cfg, np.random.default_rng(0)), cfg


def _bring(arr, discharge, meander, state, volume=100.0, **over):
    """Run an imported river in at the west edge; returns what changed and the record."""
    from worldgen.stages.erosion import _drain_to_the_sea, _inlet_reaches

    stage, cfg = _stage(**over)
    net, drainage = _drain_to_the_sea(arr, _SEA, state, {}, 2.0, np.random.default_rng(0))
    mouth = (0, arr.shape[1] // 2)
    reaches, sea = _inlet_reaches(arr, _SEA, state, net, drainage, mouth)
    before = arr.copy()
    deposition = np.zeros_like(arr)
    stage._load_inlets(
        arr, deposition, state, net, drainage, {mouth: volume}, _SEA, meander, discharge
    )
    span = cfg.max_elevation_m + cfg.seabed_depth_m
    load = cfg.erosion_inlet_yield_m / span * volume
    return before, deposition, reaches, sea, load, cfg, state


def test_the_river_lays_a_fixed_share_of_its_load_at_every_reach():
    """The load falls away geometrically down the course: f of what is left, each reach."""
    before, deposition, reaches, sea, load, cfg, _ = _bring(*_valley())
    assert len(reaches) > 5 and sea is not None
    f = cfg.erosion_inlet_drop_fraction
    on_land = deposition[before >= _SEA].sum()
    assert on_land == pytest.approx(load * (1.0 - (1.0 - f) ** len(reaches)))


def test_no_channel_bed_rises():
    """The bed is scoured, not built up — neither this river's nor any other's within reach."""
    arr, discharge, meander, state = _valley()
    _, deposition, reaches, _, _, _, _ = _bring(arr, discharge, meander, state)
    mid = arr.shape[1] // 2
    assert all(deposition[c] == 0.0 for c in reaches)
    assert (deposition[:, mid] == 0.0).all()


def test_a_cell_rises_no_more_than_the_reaches_that_can_reach_it_lay():
    """Each reach lays at most `f` of the load; a cell is in the floodplain only of reaches
    within the belt's half-width of it, so that bounds its rise — with no tuned clamp."""
    before, deposition, reaches, _, load, cfg, _ = _bring(*_valley())
    f, r = cfg.erosion_inlet_drop_fraction, int(cfg.valley_width_max)
    w, h = before.shape
    for i in range(w):
        for j in range(h):
            if before[i, j] < _SEA:
                continue
            near = sum(1 for a, b in reaches if abs(a - i) + abs(b - j) <= r)
            assert deposition[i, j] <= f * load * near + 1e-12


def test_a_gorge_with_no_belt_lays_on_its_banks():
    before, deposition, reaches, _, _, _, _ = _bring(*_valley(belt=False))
    raised = {(i, j) for i, j in zip(*np.nonzero(deposition), strict=True) if before[i, j] >= _SEA}
    assert raised
    for i, j in raised:
        assert any(abs(i - a) + abs(j - b) == 1 for a, b in reaches), f"{(i, j)} is no bank"


def test_what_reaches_the_sea_fans_out_below_the_waterline():
    before, deposition, _, sea, _, _, _ = _bring(*_valley())
    water = before < _SEA
    assert deposition[water].sum() > 0.0
    after = before + deposition
    assert (after[water] < _SEA).all(), "the fan was built onto the waterline"
    assert deposition[sea] == deposition[water].max(), "the fan is thickest at the mouth"


def test_the_load_is_conserved():
    """Laid on land, laid at sea, or recorded as gone to the deep: nothing else."""
    arr, discharge, meander, state = _valley()
    before, deposition, _, _, load, cfg, state = _bring(arr, discharge, meander, state)
    span = cfg.max_elevation_m + cfg.seabed_depth_m
    deep = state.metadata["inlet_sediment_to_deep_m_km2"] / span
    assert deposition.sum() + deep == pytest.approx(load, rel=1e-4)
    assert state.metadata["inlet_sediment_off_map_m_km2"] == 0.0


def test_a_fan_with_no_room_sends_its_load_to_the_deep():
    from worldgen.stages.erosion import _fan_out

    arr, _, _, state = _valley()
    deposition = np.zeros_like(arr)
    left = _fan_out(arr, deposition, state, (24, 5), _SEA, 1_000.0, radius=2)
    assert left == pytest.approx(1_000.0 - deposition.sum())
    assert left > 0.0
    assert (arr[24:, :] < _SEA).all()
    assert deposition[28:, :].sum() == 0.0, "the fan went past its radius"


def test_a_river_that_leaves_the_map_takes_its_load_with_it():
    arr, discharge, meander, state = _valley(sea_from=30)
    _, deposition, _, sea, load, cfg, state = _bring(arr, discharge, meander, state)
    assert sea is None
    span = cfg.max_elevation_m + cfg.seabed_depth_m
    off = state.metadata["inlet_sediment_off_map_m_km2"] / span
    assert deposition.sum() + off == pytest.approx(load, rel=1e-4)


def test_a_hollow_on_the_course_is_crossed_and_left_as_it_was():
    """No reach in a hollow: filled to its spill, one flooded 800 km2 around it."""
    arr, discharge, meander, state = _valley()
    arr[10:13, 3:8] -= 0.05  # a basin across the valley
    before, deposition, reaches, sea, _, _, _ = _bring(arr, discharge, meander, state)
    assert sea is not None
    assert not any(10 <= i < 13 and 3 <= j < 8 for i, j in reaches)
