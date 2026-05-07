import pytest

from worldgen.core.config import WorldConfig


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"hex_size_m": 0.0}, "hex_size_m"),
        ({"road_elev_range_m": 0.0}, "road_elev_range_m"),
        (
            {"road_slope_free_pct": 10.0, "road_slope_cap_pct": 10.0},
            "road_slope_cap_pct",
        ),
        ({"settlement_min_reachable": 0}, "settlement_min_reachable"),
    ],
)
def test_world_config_validates_new_road_and_settlement_fields(kwargs, message):
    with pytest.raises(ValueError, match=message):
        WorldConfig(**kwargs)
