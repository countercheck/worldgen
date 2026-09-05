import numpy as np

from worldgen.stages.erosion import _widen_valleys


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
