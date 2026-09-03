import json

import pytest
import yaml

from worldgen.core.config import WorldConfig


def test_yaml_roundtrip(tmp_path):
    cfg = WorldConfig(
        width=64, height=48, base_precip_mm=50.0, elevation_gradient_m=(300.0, -200.0)
    )
    out = str(tmp_path / "cfg.yaml")
    cfg.to_yaml(out)
    loaded = WorldConfig.from_yaml(out)
    assert loaded.width == 64
    assert loaded.height == 48
    assert loaded.base_precip_mm == pytest.approx(50.0)
    assert loaded.elevation_gradient_m == pytest.approx((300.0, -200.0))


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
    data = {"elevation_gradient_m": [0.5, -0.3]}
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(data))
    cfg = WorldConfig.from_yaml(str(p))
    assert isinstance(cfg.elevation_gradient_m, tuple)
    assert cfg.elevation_gradient_m == pytest.approx((0.5, -0.3))


def test_from_json_wind_direction_is_tuple(tmp_path):
    data = {"wind_direction": [0.0, 1.0]}
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(data))
    cfg = WorldConfig.from_json(str(p))
    assert isinstance(cfg.wind_direction, tuple)


@pytest.mark.parametrize("key", ["wind_direction", "elevation_gradient_m"])
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
        ({"elevation_gradient_m": (0.5,)}, "elevation_gradient_m"),
        ({"elevation_gradient_m": (0.1, "north")}, "elevation_gradient_m"),
    ],
)
def test_world_config_validates_vector_fields(kwargs, message):
    with pytest.raises(ValueError, match=message):
        WorldConfig(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"hex_size_m": 0.0}, "hex_size_m"),
        ({"max_elevation_m": 0.0}, "max_elevation_m"),
        ({"seabed_depth_m": 0.0}, "seabed_depth_m"),
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


# --- unknown and retired keys ------------------------------------------------


def test_unknown_key_raises_value_error_not_type_error(tmp_path):
    """A typo used to reach the dataclass constructor as a TypeError.

    The CLI catches ValueError, so that surfaced as a raw traceback rather than a message.
    """
    path = tmp_path / "typo.yaml"
    path.write_text("width: 32\nhex_size_mm: 1000\n")
    with pytest.raises(ValueError, match="Unknown config setting"):
        WorldConfig.from_yaml(str(path))


def test_unknown_key_suggests_the_nearest_real_setting(tmp_path):
    path = tmp_path / "typo.yaml"
    path.write_text("hex_size_mm: 1000\n")
    with pytest.raises(ValueError, match="did you mean 'hex_size_m'"):
        WorldConfig.from_yaml(str(path))


def test_unknown_key_with_no_near_match_still_names_the_key(tmp_path):
    path = tmp_path / "junk.yaml"
    path.write_text("qqqqzzzz: 1\n")
    with pytest.raises(ValueError, match="'qqqqzzzz'"):
        WorldConfig.from_yaml(str(path))


def test_unknown_key_in_json_is_also_a_value_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"width": 32, "not_a_setting": 1}')
    with pytest.raises(ValueError, match="Unknown config setting"):
        WorldConfig.from_json(str(path))


def test_retired_key_warns_and_still_loads(tmp_path, monkeypatch):
    """A config written against an older version must keep working."""
    from worldgen.core import config as config_module

    monkeypatch.setitem(config_module._RETIRED_FIELDS, "old_knob", "use new_knob instead.")
    path = tmp_path / "old.yaml"
    path.write_text("width: 32\nold_knob: 5\n")

    with pytest.warns(DeprecationWarning, match="old_knob"):
        cfg = WorldConfig.from_yaml(str(path))
    assert cfg.width == 32


def test_shipped_default_config_loads():
    """`init-config` copies this file verbatim, so it must parse against the dataclass.

    It is hand-synced with the dataclass, so nothing else would catch a key that was
    renamed in one place and not the other.
    """
    from pathlib import Path

    import worldgen

    shipped = Path(worldgen.__file__).parent / "default_config.yaml"
    assert WorldConfig.from_yaml(str(shipped)).width > 0
