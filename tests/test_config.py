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
        ({"road_delta_elevation_per_hex": 0.0}, "road_delta_elevation_per_hex"),
        (
            {"road_switchback_grade_pct": 30.0, "road_slope_cap_pct": 25.0},
            "road_switchback_grade_pct",
        ),
        ({"settlement_min_reachable": 0}, "settlement_min_reachable"),
        ({"moisture_bleed_passes": -1}, "moisture_bleed_passes"),
        ({"moisture_bleed_strength": -0.1}, "moisture_bleed_strength"),
        ({"moisture_bleed_strength": 1.1}, "moisture_bleed_strength"),
        # The ground goes bare above the treeline, not below it.
        ({"biome_snowline_temp_c": 0.0}, "biome_snowline_temp_c"),
        ({"biome_snowline_temp_c": -2.0}, "biome_snowline_temp_c"),
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


def _config_yaml_keys(path):
    """Top-level setting names in a config file, excluding the `export` section."""
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f)
    return {k for k in raw if k != "export"}


def _documented_config_files():
    """The two hand-written config files, both of which must track the dataclass.

    `default_config.yaml` is what `init-config` copies for a new user; the root
    `worldgen.yaml` is the working config the README and the verification commands
    point at. Neither is generated, so only a test keeps them honest.
    """
    from pathlib import Path

    import worldgen

    package = Path(worldgen.__file__).parent
    return [package / "default_config.yaml", package.parent / "worldgen.yaml"]


@pytest.mark.parametrize("path", _documented_config_files(), ids=lambda p: p.name)
def test_shipped_config_loads(path):
    """`init-config` copies this file verbatim, so it must parse against the dataclass."""
    assert WorldConfig.from_yaml(str(path)).width > 0


@pytest.mark.parametrize("path", _documented_config_files(), ids=lambda p: p.name)
def test_shipped_config_documents_every_setting(path):
    """Every `WorldConfig` field must appear in the shipped configs.

    Parsing alone is not enough, and the difference is not academic: the root
    `worldgen.yaml` went twenty-one retired keys and forty-nine missing ones out of date
    across the units work while still loading cleanly every time, because a retired key
    only warns and a missing one silently takes the dataclass default. A config file that
    loads but documents none of the settings that matter is worse than one that fails, so
    the coverage is asserted rather than the parse.
    """
    from dataclasses import fields

    documented = _config_yaml_keys(path)
    declared = {f.name for f in fields(WorldConfig)}

    missing = sorted(declared - documented)
    assert not missing, f"{path.name} does not document: {', '.join(missing)}"


@pytest.mark.parametrize("path", _documented_config_files(), ids=lambda p: p.name)
def test_shipped_config_has_no_retired_or_unknown_settings(path):
    """...and nothing in them may be a setting the dataclass no longer has.

    A retired key loads with a warning, so a stale file is quiet at runtime; this is what
    makes it audible.
    """
    from dataclasses import fields

    from worldgen.core.config import _RENAMED_FIELDS, _RETIRED_FIELDS

    documented = _config_yaml_keys(path)
    declared = {f.name for f in fields(WorldConfig)}

    retired = sorted(documented & (set(_RETIRED_FIELDS) | set(_RENAMED_FIELDS)))
    assert not retired, f"{path.name} still sets retired settings: {', '.join(retired)}"

    unknown = sorted(documented - declared)
    assert not unknown, f"{path.name} sets settings that do not exist: {', '.join(unknown)}"


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
        ({"river_inflow_length_bias": -0.1}, "river_inflow_length_bias"),
        ({"river_inflow_min_length": -0.1}, "river_inflow_min_length"),
        # The shared edge parser names the setting that was actually wrong, so a typo in
        # one edge list is never reported against the other.
        ({"continent_falloff_edges": ["nrth"]}, "continent_falloff_edges"),
    ],
)
def test_world_config_validates_river_inflow_fields(kwargs, message):
    with pytest.raises(ValueError, match=message):
        WorldConfig(**kwargs)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("erosion_inlet_yield_m", -0.1),
        ("erosion_inlet_fan_radius", -1),
        ("erosion_inlet_drop_fraction", -0.1),
        ("erosion_inlet_drop_fraction", 1.1),
    ],
)
def test_the_inlet_load_settings_are_validated(name, value):
    with pytest.raises(ValueError, match=name):
        WorldConfig(**{name: value})


@pytest.mark.parametrize("value", [0.0, -150.0])
def test_valley_width_reference_must_be_positive(value):
    """It divides every channel's discharge, so zero or less has no meaning."""
    with pytest.raises(ValueError, match="valley_width_reference_km2"):
        WorldConfig(valley_width_reference_km2=value)


def test_river_inflow_edges_are_canonicalised():
    # Same normalisation the falloff edges get: a comma-separated string, any case, any
    # order, deduplicated, and stable so two equivalent configs compare equal.
    assert WorldConfig(river_inflow_edges="West, north ,west").river_inflow_edges == (
        "north",
        "west",
    )
    assert WorldConfig(river_inflow_edges=[]).river_inflow_edges == ()


def test_world_config_validates_evaporation_scale():
    with pytest.raises(ValueError, match="endorheic_evaporation_scale"):
        WorldConfig(endorheic_evaporation_scale=-0.1)


def test_every_climate_evaporates_something_and_the_warm_ones_evaporate_more():
    """The basin water balance reads potential evapotranspiration, not a per-climate knob.

    It is the same term the runoff model takes off the top of the rain, which is what
    stops the basins and the rivers disagreeing about the climate — and being keyed on
    temperature it orders itself: a desert basin loses more than a subarctic one without
    anybody tabulating that fact.
    """
    from worldgen.core.config import CLIMATE_CONTEXTS

    pets = {name: WorldConfig(regional_climate=name).pet_mm() for name in CLIMATE_CONTEXTS}
    assert all(v > 0 for v in pets.values()), pets
    # Warmer regions must evaporate more, or the balance has the world backwards.
    assert pets["boreal"] < pets["temperate"] < pets["mediterranean"] < pets["arid"], pets


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_world_config_validates_rain_shadow_strength(value):
    with pytest.raises(ValueError, match="rain_shadow_strength"):
        WorldConfig(rain_shadow_strength=value)


# Seasonal runoff (tech-debt #120). Pike's curve is non-linear in rain, so a year whose rain
# is bunched into a season or two must shed more than the same rain spread evenly.


def test_an_even_year_runs_off_exactly_as_the_annual_curve_says():
    cfg = WorldConfig(season_shares=(0.25, 0.25, 0.25, 0.25))
    for precip in (50.0, 200.0, 800.0, 3000.0):
        assert cfg.seasonal_runoff_mm(precip) == pytest.approx(cfg.runoff_mm(precip))


def test_bunched_rain_runs_off_more_than_the_same_rain_spread_evenly():
    even = WorldConfig(season_shares=(0.25, 0.25, 0.25, 0.25))
    bunched = WorldConfig(season_shares=(0.05, 0.05, 0.10, 0.80))
    for precip in (300.0, 800.0, 2000.0):
        assert bunched.seasonal_runoff_mm(precip) > even.seasonal_runoff_mm(precip)


def test_seasonal_runoff_never_exceeds_the_rain_and_never_drops_below_the_floor():
    for name in ("arid", "mediterranean", "temperate", "boreal", "tropical"):
        cfg = WorldConfig(regional_climate=name)
        for precip in (0.0, 10.0, cfg.mean_precip_mm, 5000.0):
            got = cfg.seasonal_runoff_mm(precip)
            assert got >= cfg.min_runoff_mm, name
            assert got <= max(precip, cfg.min_runoff_mm) + 1e-9, name


def test_the_floor_applies_to_the_year_not_to_each_season():
    """A season with no rain earns no runoff. Flooring each season would hand a desert a
    quarter of `min_runoff_mm` for every dry season, on top of what its wet one shed."""
    cfg = WorldConfig(season_shares=(0.0, 0.0, 0.0, 1.0), min_runoff_mm=25.0)
    wet_season_alone = cfg._pike_runoff_mm(800.0, cfg.pet_mm()) / 4
    assert cfg.seasonal_runoff_mm(200.0) == pytest.approx(max(25.0, wet_season_alone))


def test_seasonal_runoff_defaults_to_the_climate_mean():
    cfg = WorldConfig(regional_climate="mediterranean")
    assert cfg.seasonal_runoff_mm() == cfg.seasonal_runoff_mm(cfg.mean_precip_mm)


def test_dry_summer_and_monsoon_climates_gain_runoff_and_even_ones_do_not():
    """What #120 is for: the climates whose rain is bunched shed more than the annual curve
    credited them with; the ones whose rain is even barely move."""

    def gain(name):
        cfg = WorldConfig(regional_climate=name, min_runoff_mm=0.0)
        return cfg.seasonal_runoff_mm() / cfg.runoff_mm(cfg.mean_precip_mm)

    assert gain("mediterranean") > 1.1
    assert gain("tropical") > 1.05
    assert gain("temperate") == pytest.approx(1.0, abs=0.02)
    assert gain("boreal") == pytest.approx(1.0, abs=0.02)
