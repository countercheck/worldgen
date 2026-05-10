import json

import pytest
import yaml

from worldgen.core.config import WorldConfig


def test_yaml_roundtrip(tmp_path):
    cfg = WorldConfig(width=64, height=48, base_moisture=0.1, elevation_gradient=(0.3, -0.2))
    out = str(tmp_path / "cfg.yaml")
    cfg.to_yaml(out)
    loaded = WorldConfig.from_yaml(out)
    assert loaded.width == 64
    assert loaded.height == 48
    assert loaded.base_moisture == pytest.approx(0.1)
    assert loaded.elevation_gradient == pytest.approx((0.3, -0.2))


def test_from_yaml_ignores_export_block(tmp_path):
    data = {"width": 32, "export": {"style": "topographic", "hex_size": 8.0}}
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(data))
    cfg = WorldConfig.from_yaml(str(p))
    assert cfg.width == 32


def test_from_yaml_empty_file_uses_defaults(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("")
    cfg = WorldConfig.from_yaml(str(p))
    assert cfg.width == WorldConfig().width


def test_from_yaml_requires_mapping_root(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("- not\n- a\n- mapping\n")
    with pytest.raises(ValueError, match="mapping/object"):
        WorldConfig.from_yaml(str(p))


def test_from_yaml_wind_direction_is_tuple(tmp_path):
    data = {"wind_direction": [0.0, 1.0]}
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(data))
    cfg = WorldConfig.from_yaml(str(p))
    assert isinstance(cfg.wind_direction, tuple)
    assert cfg.wind_direction == (0.0, 1.0)


def test_from_yaml_elevation_gradient_is_tuple(tmp_path):
    data = {"elevation_gradient": [0.5, -0.3]}
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(data))
    cfg = WorldConfig.from_yaml(str(p))
    assert isinstance(cfg.elevation_gradient, tuple)
    assert cfg.elevation_gradient == pytest.approx((0.5, -0.3))


def test_from_json_wind_direction_is_tuple(tmp_path):
    data = {"wind_direction": [0.0, 1.0]}
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(data))
    cfg = WorldConfig.from_json(str(p))
    assert isinstance(cfg.wind_direction, tuple)


@pytest.mark.parametrize("key", ["wind_direction", "elevation_gradient"])
def test_yaml_tuple_fields_require_two_numeric_values(tmp_path, key):
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump({key: None}))
    with pytest.raises(ValueError, match=key):
        WorldConfig.from_yaml(str(p))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"wind_direction": (1.0,)}, "wind_direction"),
        ({"wind_direction": ("east", 0.0)}, "wind_direction"),
        ({"elevation_gradient": (0.5,)}, "elevation_gradient"),
        ({"elevation_gradient": (0.1, "north")}, "elevation_gradient"),
    ],
)
def test_world_config_validates_vector_fields(kwargs, message):
    with pytest.raises(ValueError, match=message):
        WorldConfig(**kwargs)


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
        ({"moisture_bleed_passes": -1}, "moisture_bleed_passes"),
        ({"moisture_bleed_strength": -0.1}, "moisture_bleed_strength"),
        ({"moisture_bleed_strength": 1.1}, "moisture_bleed_strength"),
    ],
)
def test_world_config_validates_new_road_and_settlement_fields(kwargs, message):
    with pytest.raises(ValueError, match=message):
        WorldConfig(**kwargs)
