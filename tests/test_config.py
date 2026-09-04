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


# --- unknown and retired keys ------------------------------------------------


def test_unknown_key_raises_value_error_not_type_error(tmp_path):
    """A typo used to reach the dataclass constructor as a TypeError.

    The CLI catches ValueError, so that surfaced as a raw traceback rather than a message.
    """
    path = tmp_path / "typo.yaml"
    path.write_text("width: 32\nsea_levl: 0.3\n")
    with pytest.raises(ValueError, match="Unknown config setting"):
        WorldConfig.from_yaml(str(path))


def test_unknown_key_suggests_the_nearest_real_setting(tmp_path):
    path = tmp_path / "typo.yaml"
    path.write_text("sea_levl: 0.3\n")
    with pytest.raises(ValueError, match="did you mean 'sea_level'"):
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


# --- heightmap import --------------------------------------------------------


def test_heightmap_fields_roundtrip_through_yaml(tmp_path):
    cfg = WorldConfig(
        heightmap_path="maps/coast.png",
        heightmap_mode="coastline",
        heightmap_land_threshold=0.4,
        heightmap_invert=True,
        heightmap_coast_falloff=True,
    )
    out = str(tmp_path / "hm.yaml")
    cfg.to_yaml(out)
    loaded = WorldConfig.from_yaml(out)

    assert loaded.heightmap_path == "maps/coast.png"
    assert loaded.heightmap_mode == "coastline"
    assert loaded.heightmap_land_threshold == pytest.approx(0.4)
    assert loaded.heightmap_invert is True
    assert loaded.heightmap_coast_falloff is True


def test_unset_heightmap_path_roundtrips_as_null(tmp_path):
    out = str(tmp_path / "none.yaml")
    WorldConfig().to_yaml(out)
    assert WorldConfig.from_yaml(out).heightmap_path is None


def test_heightmap_path_accepts_a_path_object():
    """A programmatic caller reaches for Path; every dump downstream wants a string."""
    from pathlib import Path

    cfg = WorldConfig(heightmap_path=Path("a") / "b.png")
    assert cfg.heightmap_path == str(Path("a") / "b.png")
    # `metadata["config"]` is the dataclass dict, serialised verbatim into world.json.
    json.dumps(cfg.__dict__)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"heightmap_mode": "nonsense"}, "heightmap_mode"),
        ({"heightmap_land_threshold": 1.5}, "heightmap_land_threshold"),
        ({"heightmap_land_threshold": -0.1}, "heightmap_land_threshold"),
        ({"heightmap_path": ""}, "heightmap_path"),
        ({"heightmap_path": 3}, "heightmap_path"),
    ],
)
def test_world_config_validates_heightmap_fields(kwargs, message):
    with pytest.raises(ValueError, match=message):
        WorldConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"river_inflow_count": -1}, "river_inflow_count"),
        ({"river_inflow_volume": -0.1}, "river_inflow_volume"),
        ({"river_inflow_min_separation": -1}, "river_inflow_min_separation"),
        ({"river_inflow_edges": ["nrth"]}, "river_inflow_edges"),
        ({"river_inflow_edges": [3]}, "river_inflow_edges"),
        # The shared edge parser names the setting that was actually wrong, so a typo in
        # one edge list is never reported against the other.
        ({"continent_falloff_edges": ["nrth"]}, "continent_falloff_edges"),
    ],
)
def test_world_config_validates_river_inflow_fields(kwargs, message):
    with pytest.raises(ValueError, match=message):
        WorldConfig(**kwargs)


def test_river_inflow_edges_are_canonicalised():
    # Same normalisation the falloff edges get: a comma-separated string, any case, any
    # order, deduplicated, and stable so two equivalent configs compare equal.
    assert WorldConfig(river_inflow_edges="West, north ,west").river_inflow_edges == (
        "north",
        "west",
    )
    assert WorldConfig(river_inflow_edges=[]).river_inflow_edges == ()
