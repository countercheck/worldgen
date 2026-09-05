"""Unit tests for the shared river width model in worldgen.export.rivers."""

import pytest

from worldgen.core.hex import Hex, TerrainClass
from worldgen.core.world_state import River
from worldgen.export import rivers


def _grid(flows: dict) -> dict:
    """A grid of (q, 0) hexes carrying the given per-hex river flows."""
    return {
        (q, 0): Hex(coord=(q, 0), terrain_class=TerrainClass.LAND, river_flow=f)
        for q, f in flows.items()
    }


def _river(length: int, flow_volume: float = 1.0) -> River:
    return River(hexes=[(q, 0) for q in range(length)], flow_volume=flow_volume)


def test_uniform_flow_is_one_band():
    """Nothing to say about flow means one run, not one per hex."""
    hexes = _grid({q: 0.5 for q in range(5)})
    bands = rivers.width_bands(_river(5), hexes, 1.0, 4.0, 4)
    assert len(bands) == 1
    run, width = bands[0]
    assert run == [(q, 0) for q in range(5)]
    assert 1.0 <= width <= 4.0


def test_width_grows_downstream():
    """The whole point: a headwater must not be drawn like the trunk it feeds."""
    hexes = _grid({0: 0.05, 1: 0.3, 2: 0.6, 3: 0.95})
    bands = rivers.width_bands(_river(4), hexes, 1.0, 4.0, 4)
    widths = [w for _, w in bands]
    assert widths == sorted(widths), f"width should not shrink downstream: {widths}"
    assert widths[0] < widths[-1]


def test_runs_are_contiguous_and_share_their_joins():
    """A change of width must not leave a gap in the drawn line."""
    hexes = _grid({0: 0.05, 1: 0.05, 2: 0.95, 3: 0.95})
    bands = rivers.width_bands(_river(4), hexes, 1.0, 4.0, 4)
    assert len(bands) == 2
    first, second = bands[0][0], bands[1][0]
    assert first[-1] == second[0], "runs must share the vertex they meet at"


def test_every_segment_is_drawn_exactly_once():
    hexes = _grid({0: 0.05, 1: 0.4, 2: 0.4, 3: 0.99, 4: 0.99})
    bands = rivers.width_bands(_river(5), hexes, 1.0, 4.0, 4)
    drawn = [frozenset((a, b)) for run, _ in bands for a, b in zip(run, run[1:], strict=False)]
    expected = {frozenset(((q, 0), (q + 1, 0))) for q in range(4)}
    assert len(drawn) == len(set(drawn)) == len(expected)
    assert set(drawn) == expected


def test_widths_span_the_configured_range():
    hexes = _grid({0: 0.01, 1: 0.99})
    bands = rivers.width_bands(_river(2), hexes, 2.0, 9.0, 4)
    for _, w in bands:
        assert 2.0 <= w <= 9.0


def test_single_step_uses_the_max_width():
    """One band means one width, and it should be the visible one."""
    hexes = _grid({0: 0.1, 1: 0.9})
    bands = rivers.width_bands(_river(2), hexes, 1.0, 4.0, 1)
    assert [w for _, w in bands] == [4.0]


def test_more_steps_give_more_distinct_widths():
    hexes = _grid({q: q / 9 for q in range(10)})
    coarse = {w for _, w in rivers.width_bands(_river(10), hexes, 1.0, 4.0, 2)}
    fine = {w for _, w in rivers.width_bands(_river(10), hexes, 1.0, 4.0, 5)}
    assert len(fine) > len(coarse)


def test_falls_back_to_flow_volume_where_a_hex_carries_no_flow():
    """Drainage tails have no per-hex flow of their own; the river still gets drawn."""
    hexes = _grid({0: 0.0, 1: 0.0})
    bands = rivers.width_bands(_river(2, flow_volume=0.9), hexes, 1.0, 4.0, 4)
    assert len(bands) == 1
    assert bands[0][1] > 1.0, "fallback should not collapse to the thinnest band"


def test_offgrid_coords_do_not_crash():
    bands = rivers.width_bands(_river(3, flow_volume=0.5), {}, 1.0, 4.0, 4)
    assert len(bands) == 1


def test_river_shorter_than_two_hexes_draws_nothing():
    assert rivers.width_bands(River(hexes=[(0, 0)], flow_volume=1.0), {}, 1.0, 4.0, 4) == []
    assert rivers.width_bands(River(hexes=[], flow_volume=1.0), {}, 1.0, 4.0, 4) == []


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"steps": -1}, "river_width_steps must be >= 0"),
        ({"min_width": 0}, "river_min_width must be positive"),
        ({"min_width": 5.0, "max_width": 2.0}, "river_max_width must be >="),
        ({"exponent": 0.0}, "river_width_exponent must be positive"),
        ({"exponent": -1.0}, "river_width_exponent must be positive"),
    ],
)
def test_validate_rejects_bad_settings(kwargs, match):
    args = {"min_width": 1.0, "max_width": 4.0, "steps": 4, "exponent": 0.5}
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        rivers.validate(args["min_width"], args["max_width"], args["steps"], args["exponent"])


def test_validate_accepts_the_defaults():
    from worldgen.export.png_export import PNGConfig
    from worldgen.export.svg_export import SVGConfig

    for cfg in (SVGConfig(), PNGConfig()):
        rivers.validate(
            cfg.river_min_width,
            cfg.river_max_width,
            cfg.river_width_steps,
            cfg.river_width_exponent,
        )


# --- continuous scaling (steps == 0, the default) ----------------------------


def test_continuous_width_tracks_the_hex_flow_exactly():
    """No buckets: the width is the hex's own flow mapped onto the width range."""
    hexes = _grid({0: 0.25, 1: 0.25})
    bands = rivers.width_bands(_river(2), hexes, 1.0, 5.0, 0, exponent=1.0)
    assert bands[0][1] == pytest.approx(1.0 + 4.0 * 0.25)


def test_continuous_gives_a_distinct_width_per_flow_value():
    """Four different flows band into fewer widths; continuous keeps all four."""
    # All five flows sit below 0.25, so banding at 4 steps puts every segment in band 0.
    hexes = _grid({0: 0.05, 1: 0.10, 2: 0.15, 3: 0.20, 4: 0.24})
    banded = {w for _, w in rivers.width_bands(_river(5), hexes, 1.0, 4.0, 4, exponent=1.0)}
    smooth = {w for _, w in rivers.width_bands(_river(5), hexes, 1.0, 4.0, 0, exponent=1.0)}
    assert len(banded) == 1, "these flows all fall in one band"
    assert len(smooth) == 4, "continuous scaling should separate every flow step"


def test_continuous_still_grows_downstream():
    hexes = _grid({0: 0.05, 1: 0.2, 2: 0.5, 3: 0.9})
    widths = [w for _, w in rivers.width_bands(_river(4), hexes, 1.0, 4.0, 0)]
    assert widths == sorted(widths)
    assert len(set(widths)) == 3


def test_continuous_stays_within_the_configured_range():
    hexes = _grid({0: 0.0001, 1: 1.0})
    for _, w in rivers.width_bands(_river(2), hexes, 1.5, 6.0, 0):
        assert 1.5 <= w <= 6.0


def test_flow_above_one_is_clamped():
    """Normalisation should keep flow in [0, 1]; a stray value must not draw wider."""
    assert rivers.width_for_flow(4.0, 1.0, 4.0, 0) == 4.0
    assert rivers.width_for_flow(-1.0, 1.0, 4.0, 0) == 1.0


def test_continuous_and_banded_agree_at_the_extremes():
    assert rivers.width_for_flow(0.0, 1.0, 4.0, 0) == rivers.width_for_flow(0.0, 1.0, 4.0, 4)
    assert rivers.width_for_flow(1.0, 1.0, 4.0, 0) == rivers.width_for_flow(1.0, 1.0, 4.0, 4)


# --- the width curve (exponent) ----------------------------------------------


def test_typical_flows_are_not_all_drawn_at_the_minimum():
    """The defect the exponent exists for.

    river_flow is accumulation over the basin maximum, so the median river hex sits
    near 0.02.  Mapped linearly that is 0.5% of the width range — every stream but the
    trunk draws as the same hairline.
    """
    median_flow = 0.02
    span = 4.0 - 0.5

    def share_of_range(exponent: float) -> float:
        return (rivers.width_for_flow(median_flow, 0.5, 4.0, 0, exponent) - 0.5) / span

    assert share_of_range(1.0) < 0.05, "linear mapping pins a typical river to min_width"
    assert share_of_range(0.5) > 0.10, "the curve must lift a typical river clear of min_width"


def test_the_default_exponent_is_the_curve_not_the_linear_map():
    assert rivers.width_for_flow(0.25, 1.0, 5.0, 0) == pytest.approx(
        rivers.width_for_flow(0.25, 1.0, 5.0, 0, exponent=0.5)
    )
    assert rivers.width_for_flow(0.25, 1.0, 5.0, 0) != rivers.width_for_flow(
        0.25, 1.0, 5.0, 0, exponent=1.0
    )


def test_the_curve_is_monotonic_and_fixes_the_endpoints():
    """Reshaping the middle must not reorder rivers or overflow the range."""
    for exponent in (0.25, 0.5, 1.0, 2.0):
        widths = [rivers.width_for_flow(f / 20, 1.0, 4.0, 0, exponent=exponent) for f in range(21)]
        assert widths == sorted(widths)
        assert widths[0] == pytest.approx(1.0)
        assert widths[-1] == pytest.approx(4.0)


def test_the_curve_applies_under_banding_too():
    """Banding buckets the curved flow, so small streams spread across bands as well."""
    # All three flows sit below 0.25, so linear banding at 4 steps buckets them together.
    hexes = _grid({0: 0.01, 1: 0.05, 2: 0.2})
    linear = {w for _, w in rivers.width_bands(_river(3), hexes, 1.0, 4.0, 4, exponent=1.0)}
    curved = {w for _, w in rivers.width_bands(_river(3), hexes, 1.0, 4.0, 4, exponent=0.5)}
    assert len(linear) == 1, "linear banding puts all three in the bottom band"
    assert len(curved) > 1


def test_continuous_costs_more_polylines_than_banding():
    """The trade the setting exists for: fidelity against output size."""
    hexes = _grid({q: q / 19 for q in range(20)})
    smooth = rivers.width_bands(_river(20), hexes, 1.0, 4.0, 0)
    banded = rivers.width_bands(_river(20), hexes, 1.0, 4.0, 4)
    assert len(smooth) > len(banded)
