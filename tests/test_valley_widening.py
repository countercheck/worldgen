import numpy as np
import pytest

from worldgen.core.world_state import WorldState
from worldgen.stages.erosion import _neighbour_table, _widen_valleys


def _ridge_with_notch(w=21, h=9, notch_col=10):
    """A slope with a single one-cell notch cut down it — a droplet-carved V."""
    arr = np.zeros((w, h))
    for i in range(w):
        for j in range(h):
            arr[i, j] = 0.6 - 0.02 * j
    arr[notch_col, :] = 0.30
    discharge = np.ones((w, h))
    discharge[notch_col, :] = 50.0
    return arr, discharge


def test_widening_planes_a_floor_out_from_the_channel():
    arr, discharge = _ridge_with_notch()
    before = arr.copy()
    _widen_valleys(arr, discharge, 0.0, 4.0, 0.4, 0.001, 0.5, 0.05)
    row = 4
    # The notch itself is untouched; its neighbours are cut down toward it.
    assert arr[10, row] == before[10, row]
    assert arr[9, row] < before[9, row]
    assert arr[11, row] < before[11, row]
    # And the floor is flat-ish rather than a V: the step from the channel to its
    # neighbour is far smaller than the drop the notch used to have.
    assert arr[9, row] - arr[10, row] < (before[9, row] - before[10, row]) / 4


def test_widening_never_raises_ground():
    arr, discharge = _ridge_with_notch()
    before = arr.copy()
    _widen_valleys(arr, discharge, 0.0, 4.0, 0.4, 0.001, 0.5, 0.05)
    assert (arr <= before + 1e-12).all()


def test_widening_stops_at_a_wall():
    # A gorge: the notch is walled by ground standing far above it. Lateral planation
    # cannot take a bluff down, so the valley stays as narrow as the walls make it.
    arr, discharge = _ridge_with_notch()
    arr[:, :] = 0.9
    arr[10, :] = 0.30
    before = arr.copy()
    _widen_valleys(arr, discharge, 0.0, 6.0, 0.4, 0.001, 0.02, 0.05)
    assert (arr == before).all(), "planed through a wall it should not have shifted"


def test_widening_is_bounded_by_relief_not_only_width():
    # Same terrain, same width budget: a bigger relief allowance cuts more.
    arr_a, disch = _ridge_with_notch()
    arr_b = arr_a.copy()
    _widen_valleys(arr_a, disch, 0.0, 6.0, 0.4, 0.001, 0.02, 0.05)
    _widen_valleys(arr_b, disch, 0.0, 6.0, 0.4, 0.001, 0.5, 0.05)
    assert arr_b.sum() < arr_a.sum()


def test_widening_disabled_leaves_the_field_alone():
    arr, discharge = _ridge_with_notch()
    before = arr.copy()
    _widen_valleys(arr, discharge, 0.0, 0.0, 0.4, 0.001, 0.5, 0.05)
    assert (arr == before).all()


def test_a_bigger_channel_gets_a_wider_valley():
    # Reach scales with a channel's share of the largest flow on the map, so the two have
    # to be weighed against each other in one field — a lone channel is always the biggest
    # there is, whatever number it carries.
    w, h = 41, 9
    arr = np.zeros((w, h))
    for i in range(w):
        for j in range(h):
            arr[i, j] = 0.6 - 0.02 * j
    discharge = np.ones((w, h))
    for col, flow in ((10, 50.0), (30, 5.0)):
        arr[col, :] = 0.30
        discharge[col, :] = flow
    before = arr.copy()

    _widen_valleys(arr, discharge, 0.0, 6.0, 0.4, 0.001, 0.5, 0.05)

    cut = before[:, 4] - arr[:, 4] > 1e-9
    assert int(cut[:21].sum()) > int(cut[21:].sum())


def test_sink_filling_lets_drainage_cross_a_pit():
    """Without it, accumulation dies in the first depression and no trunk river forms.

    That was most of why the carved channels coincided with only a third of the rivers
    hydrology found: a pitted surface gives a scatter of short segments, not a network.
    """
    from worldgen.core.world_state import WorldState
    from worldgen.stages.erosion import _grid_flow_accumulation, _neighbour_table

    w = h = 9
    state = WorldState.empty(seed=1, width=w, height=h)
    arr = np.zeros((w, h))
    for i in range(w):
        for j in range(h):
            arr[i, j] = 1.0 - 0.05 * j  # drains toward high j
    arr[4, 4] = 0.2  # a pit partway down

    acc = _grid_flow_accumulation(arr, 0.0, _neighbour_table(state, w, h))
    # Water reaching the pit must carry on past it rather than stopping there.
    assert acc[:, 8].sum() > acc[4, 4], "drainage never left the pit"


def test_widening_produces_flat_ground_beside_the_channel():
    """The property the whole pass exists for, stated directly.

    Not measured as the rise of the ground at a distance from the river: cutting a floor
    lowers it, which leaves the valley walls standing *higher* above the water, so that
    figure rises even as the floodplain broadens.  It answers how incised a channel is,
    not how wide its floor is.  Count the ground at the river's own level instead.
    """
    arr, discharge = _ridge_with_notch(w=21, h=9, notch_col=10)
    row = 4
    river = arr[10, row]

    def flat_beside(field):
        return sum(1 for i in range(21) if abs(field[i, row] - river) <= 0.01)

    before = flat_beside(arr)
    _widen_valleys(arr, discharge, 0.0, 4.0, 0.4, 0.001, 0.5, 0.05)
    assert flat_beside(arr) > before


def test_an_off_map_catchment_makes_the_channel_below_it_a_trunk():
    """Seeding a mouth is what gives an imported river a river's valley.

    Widening scales its reach by discharge, and accumulation starts every cell at one hex
    of rain, so without this an off-map trunk is measured as the trickle its first few
    on-map hexes would raise — and gets a trickle's valley, however much water hydrology
    later says crosses the border there.
    """
    from worldgen.core.world_state import WorldState
    from worldgen.stages.erosion import _grid_flow_accumulation, _neighbour_table

    w = h = 9
    state = WorldState.empty(seed=1, width=w, height=h)
    arr = np.zeros((w, h))
    for i in range(w):
        for j in range(h):
            arr[i, j] = 1.0 - 0.05 * j  # drains toward high j
    table = _neighbour_table(state, w, h)
    mouth = (4, 0)

    plain = _grid_flow_accumulation(arr, 0.0, table)
    seeded = _grid_flow_accumulation(arr, 0.0, table, {mouth: 500.0})

    assert seeded[mouth] > plain[mouth] * 10
    # And it travels: the catchment has to reach the sea, not stop at the border hex.
    assert seeded[:, 8].sum() > plain[:, 8].sum() + 400


# --- channel incision --------------------------------------------------------


def _ramp(w=12, h=5, drop=0.02):
    """A plane tilting east, well above sea level, with no depressions in it."""
    arr = np.zeros((w, h), dtype=float)
    for i in range(w):
        arr[i, :] = 0.9 - i * drop
    return arr


def _incise(arr, acc, state, **over):
    from worldgen.stages.erosion import _grid_receivers, _incise_channels

    neighbours = _neighbour_table(state, *arr.shape)
    receivers, order = _grid_receivers(arr, 0.0, neighbours)
    kwargs = dict(
        m_per_pass=12.0,
        area_exponent=0.5,
        slope_exponent=1.0,
        reference_km2=500.0,
        reference_slope=0.01,
        min_gradient_m=0.01,
        max_cut_m=40.0,
    )
    kwargs.update(over)
    _incise_channels(arr, acc, receivers, order, 0.0, 1700.0, **kwargs)
    return receivers


def test_a_big_catchment_cuts_below_the_ground_beside_it():
    """The whole point: without this a trunk can never get deeper than its hillslopes."""
    arr = _ramp()
    state = WorldState.empty(seed=1, width=12, height=5)
    acc = np.ones_like(arr)
    acc[:, 2] = 800.0  # one row draining a real catchment
    before = arr.copy()
    _incise(arr, acc, state)
    channel_cut = before[6, 2] - arr[6, 2]
    hillslope_cut = before[6, 1] - arr[6, 1]
    assert channel_cut > 10 * hillslope_cut, (
        f"a 800 km2 channel should outcut 1 km2 of hillslope by far: "
        f"{channel_cut:.5f} vs {hillslope_cut:.5f}"
    )
    assert arr[6, 2] < arr[6, 1], "the channel should end up below the ground beside it"


def test_incision_never_inverts_the_flow():
    """No cell may be cut below its own receiver, or the next sink fill has to undo it."""
    arr = _ramp()
    state = WorldState.empty(seed=1, width=12, height=5)
    acc = np.full_like(arr, 400.0)
    receivers = _incise(arr, acc, state)
    for cell, receiver in receivers.items():
        assert arr[cell] > arr[receiver], f"{cell} was cut to or below {receiver}"


def test_a_zero_area_exponent_reverts_to_slope_alone():
    """The knob that turns the new term off has to actually turn it off."""
    arr_area, arr_flat = _ramp(), _ramp()
    state = WorldState.empty(seed=1, width=12, height=5)
    acc = np.ones_like(arr_area)
    acc[:, 2] = 800.0
    _incise(arr_area, acc, state)
    _incise(arr_flat, acc, state, area_exponent=0.0)
    assert arr_flat[6, 2] == pytest.approx(arr_flat[6, 1]), "no area term, no contrast"
    assert arr_area[6, 2] < arr_area[6, 1]


def test_incision_is_disabled_at_zero_metres():
    arr = _ramp()
    state = WorldState.empty(seed=1, width=12, height=5)
    before = arr.copy()
    _incise(arr, np.full_like(arr, 900.0), state, m_per_pass=0.0)
    assert np.array_equal(arr, before)


def test_one_cell_cannot_lose_more_than_the_cap():
    """A cliff inherited from the noise must not become a chasm in one pass."""
    arr = _ramp(drop=0.0)
    arr[3, :] = 0.9
    arr[4:, :] = 0.1  # a 1360 m step
    state = WorldState.empty(seed=1, width=12, height=5)
    before = arr.copy()
    _incise(arr, np.full_like(arr, 5000.0), state, max_cut_m=40.0)
    worst_m = float((before - arr).max()) * 1700.0
    assert worst_m <= 40.0 + 1e-6, f"cut {worst_m:.1f} m in one pass"
