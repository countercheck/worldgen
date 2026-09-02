import pytest

from worldgen.core.config import WorldConfig
from worldgen.core.hex import TerrainClass
from worldgen.core.pipeline import GeneratorPipeline
from worldgen.core.world_state import WorldState
from worldgen.stages.climate import ClimateStage
from worldgen.stages.elevation import ElevationStage
from worldgen.stages.erosion import ErosionStage
from worldgen.stages.hydrology import HydrologyStage
from worldgen.stages.terrain_class import TerrainClassificationStage


def _build_pipeline(seed: int = 42, width: int = 32, height: int = 32):
    cfg = WorldConfig(width=width, height=height)
    p = GeneratorPipeline(seed, cfg)
    p.add_stage(ElevationStage)
    p.add_stage(ErosionStage)
    p.add_stage(TerrainClassificationStage)
    p.add_stage(HydrologyStage)
    p.add_stage(ClimateStage)
    return p


@pytest.fixture(scope="module")
def climate_state():
    return _build_pipeline().run()


def test_temperature_is_a_plausible_annual_mean(climate_state):
    for h in climate_state.hexes.values():
        assert -60.0 <= h.temperature <= 50.0, (
            f"temperature {h.temperature:.1f} C is not a plausible mean annual value"
        )


def test_moisture_is_a_plausible_annual_rainfall(climate_state):
    for h in climate_state.hexes.values():
        assert 0.0 <= h.moisture <= 12000.0, (
            f"moisture {h.moisture:.0f} mm is not a plausible annual rainfall"
        )


def test_standing_water_is_not_short_of_rain(climate_state):
    for h in climate_state.hexes.values():
        if h.terrain_class == TerrainClass.OCEAN:
            assert h.moisture == pytest.approx(
                climate_state.metadata["config"]["mean_precip_mm"]
            ), f"water hex has {h.moisture:.0f} mm — it should not read as the driest ground"


def test_mountains_colder_than_flat(climate_state):
    # Sample mountain and flat hexes at similar latitudes; mountains must be colder on average.
    height = climate_state.height
    mountain_temps = []
    flat_temps = []
    for (_, r), h in climate_state.hexes.items():
        mid = height * 0.3 < r < height * 0.7
        if not mid:
            continue
        if h.terrain_class == TerrainClass.STEEP:
            mountain_temps.append(h.temperature)
        elif h.terrain_class == TerrainClass.FLAT:
            flat_temps.append(h.temperature)

    if mountain_temps and flat_temps:
        assert sum(mountain_temps) / len(mountain_temps) < sum(flat_temps) / len(flat_temps), (
            "Mountain hexes not colder than flat hexes at similar latitude"
        )


def test_rain_shadow_present(climate_state):
    # Measured against the barrier the air has already had to climb, not against the
    # terrain class beside it. Orographic lift keys on elevation above sea level, so what
    # casts a shadow is *high* ground. Selecting on the terrain class worked only while
    # MOUNTAIN meant "steep or high"; the classes are bands of gradient now, and a steep
    # hex can sit at any altitude. Testing the mechanism is both truer and less brittle
    # than testing a proxy that has stopped holding.
    cfg = WorldConfig(**climate_state.metadata["config"])
    water = (TerrainClass.OCEAN, TerrainClass.LAKE)

    # Wind blows east by default, so upwind is west along a row.
    sheltered, exposed = [], []
    for r in range(climate_state.height):
        barrier = 0.0
        for q in range(climate_state.width):
            h = climate_state.hexes.get((q, r))
            if h is None:
                continue
            if h.terrain_class not in water:
                behind = (barrier - cfg.sea_level) > 0.30
                (sheltered if behind else exposed).append(h.moisture)
            barrier = max(barrier, h.elevation)

    assert sheltered, "no land lies behind high ground — the map has no barrier to test"
    assert exposed, "no land lies in front of high ground"

    wet = sum(exposed) / len(exposed)
    dry = sum(sheltered) / len(sheltered)
    # As a ratio rather than a difference in millimetres, so the test says the same thing
    # about a desert as about a rainforest.
    assert dry < 0.9 * wet, (
        f"Rain shadow not detected: land behind high ground averages {dry:.0f} mm against "
        f"{wet:.0f} mm in front of it. Erosion wearing the barriers down is the usual "
        f"cause — see erosion_droplets_per_hex."
    )


def test_reproducibility():
    s1 = _build_pipeline(seed=7).run()
    s2 = _build_pipeline(seed=7).run()
    for coord in s1.hexes:
        assert s1.hexes[coord].temperature == s2.hexes[coord].temperature, (
            f"temperature differs at {coord} between identical seeds"
        )
        assert s1.hexes[coord].moisture == s2.hexes[coord].moisture, (
            f"moisture differs at {coord} between identical seeds"
        )


def _mean_land_temperature(state) -> float:
    temps = [h.temperature for h in state.hexes.values() if h.terrain_class != TerrainClass.OCEAN]
    return sum(temps) / len(temps) if temps else 0.0


def test_mean_temperature_shifts_the_map():
    """A warmer region should come out warmer."""

    def run_with_base(base: float):
        cfg = WorldConfig(width=32, height=32, mean_temperature_c=base)
        p = GeneratorPipeline(42, cfg)
        p.add_stage(ElevationStage)
        p.add_stage(ErosionStage)
        p.add_stage(TerrainClassificationStage)
        p.add_stage(HydrologyStage)
        p.add_stage(ClimateStage)
        return p.run()

    cold_state = run_with_base(2.0)
    warm_state = run_with_base(22.0)
    assert _mean_land_temperature(cold_state) < _mean_land_temperature(warm_state), (
        "a higher mean_temperature_c did not produce a warmer map"
    )


def test_mean_temperature_preserves_latitude_shape():
    """Changing mean_temperature_c should shift temperatures but preserve the
    relative latitude ordering — equatorial hexes warmer than polar ones."""

    def run_with_base(base: float):
        cfg = WorldConfig(
            width=32,
            height=32,
            mean_temperature_c=base,
            latitude_temp_range_c=8.0,  # large enough to distinguish rows
        )
        p = GeneratorPipeline(42, cfg)
        p.add_stage(ElevationStage)
        p.add_stage(ErosionStage)
        p.add_stage(TerrainClassificationStage)
        p.add_stage(HydrologyStage)
        p.add_stage(ClimateStage)
        return p.run()

    for base in (2.0, 22.0):
        state = run_with_base(base)
        height = state.height
        polar_temps = [
            h.temperature
            for (_, r), h in state.hexes.items()
            if h.terrain_class != TerrainClass.OCEAN and r < height * 0.15
        ]
        equatorial_temps = [
            h.temperature
            for (_, r), h in state.hexes.items()
            if h.terrain_class != TerrainClass.OCEAN and height * 0.4 < r < height * 0.6
        ]
        if polar_temps and equatorial_temps:
            assert sum(equatorial_temps) / len(equatorial_temps) > sum(polar_temps) / len(
                polar_temps
            ), f"With mean_temperature_c={base}, equatorial hexes are not warmer than polar hexes"


def test_mean_temperature_validation():
    """A mean annual temperature outside anything Earth offers should raise."""
    with pytest.raises(ValueError, match="mean_temperature_c"):
        WorldConfig(mean_temperature_c=-99.0)
    with pytest.raises(ValueError, match="mean_temperature_c"):
        WorldConfig(mean_temperature_c=120.0)


def _mean_land_moisture(state) -> float:
    from worldgen.core.hex import TerrainClass

    vals = [
        h.moisture
        for h in state.hexes.values()
        if h.terrain_class not in (TerrainClass.OCEAN, TerrainClass.LAKE)
    ]
    return sum(vals) / len(vals) if vals else 0.0


def test_base_precip_shifts_mean_upward():
    """A positive rainfall bias should make the region wetter."""

    def run(base: float):
        cfg = WorldConfig(width=32, height=32, base_precip_mm=base)
        p = GeneratorPipeline(42, cfg)
        p.add_stage(ElevationStage)
        p.add_stage(ErosionStage)
        p.add_stage(TerrainClassificationStage)
        p.add_stage(HydrologyStage)
        p.add_stage(ClimateStage)
        return p.run()

    dry = run(0.0)
    wet = run(300.0)
    assert _mean_land_moisture(dry) < _mean_land_moisture(wet), (
        "a positive base_precip_mm did not raise mean land rainfall"
    )


def test_base_precip_has_no_artificial_ceiling():
    """Rainfall in millimetres has no upper bound to clamp to.

    It used to be clipped at a normalised 1.0, which silently swallowed any bias that
    would have pushed a wet region wetter.
    """

    cfg = WorldConfig(width=32, height=32, base_precip_mm=600.0)
    p = GeneratorPipeline(42, cfg)
    p.add_stage(ElevationStage)
    p.add_stage(ErosionStage)
    p.add_stage(TerrainClassificationStage)
    p.add_stage(HydrologyStage)
    p.add_stage(ClimateStage)
    state = p.run()
    for h in state.hexes.values():
        assert h.moisture >= 600.0, (
            f"a 600 mm bias left a hex at {h.moisture:.0f} mm — the bias was clipped away"
        )


def test_moisture_bleed_requires_river_tag():
    cfg = WorldConfig(
        width=3,
        height=1,
        sea_level=2.0,
        moisture_bleed_passes=1,
        moisture_bleed_strength=0.5,
    )
    stage = ClimateStage(cfg, None)

    state = WorldState.empty(seed=1, width=3, height=1)
    for hx in state.hexes.values():
        hx.terrain_class = TerrainClass.FLAT
        hx.elevation = 0.0
    state.hexes[(0, 0)].elevation = 1.0
    state.hexes[(0, 0)].river_flow = 1.0

    without_tag = stage.run(state)
    assert without_tag.hexes[(1, 0)].moisture == pytest.approx(0.0)

    tagged_state = WorldState.empty(seed=1, width=3, height=1)
    for hx in tagged_state.hexes.values():
        hx.terrain_class = TerrainClass.FLAT
        hx.elevation = 0.0
    tagged_state.hexes[(0, 0)].elevation = 1.0
    tagged_state.hexes[(0, 0)].river_flow = 1.0
    tagged_state.hexes[(0, 0)].tags.add("river")

    with_tag = stage.run(tagged_state)
    assert with_tag.hexes[(1, 0)].moisture > without_tag.hexes[(1, 0)].moisture


def test_latitude_temp_range_validation():
    """A negative spread between the map's edges is meaningless."""
    with pytest.raises(ValueError, match="latitude_temp_range_c"):
        WorldConfig(latitude_temp_range_c=-0.01)


def test_the_old_zero_to_one_temperature_settings_still_load(tmp_path):
    """A config written against the 0-1 axis must say so, not fail obscurely."""
    path = tmp_path / "old.yaml"
    path.write_text("base_temperature: 0.5\naltitude_lapse_rate: 0.4\nbiome_cold_temp: 0.25\n")
    with pytest.warns(DeprecationWarning, match="Celsius"):
        cfg = WorldConfig.from_yaml(str(path))
    assert cfg.mean_temperature_c == 10.0, "should fall back to the climate's real mean"


def test_erosion_dose_does_not_wash_the_rain_shadow_away():
    """The coupling that makes `erosion_droplets_per_hex` a climate setting too.

    Orographic lift is `elevation - sea_level`, and erosion wears high ground down, so
    weather and rain shadow pull against each other: enough droplets to cut floodplains
    also flatten the barriers that make a leeward side dry. At eight per hex the high
    ground is all but gone — 0-4% of land stands 0.30 above sea level, against 13-19% at
    the default — and the shadow closes to nothing on a small map. This pins the default
    on the right side of that, so raising it for flatter country cannot silently cost the
    map its dry country.
    """

    def shadow(dose):
        cfg = WorldConfig(width=48, height=48, erosion_droplets_per_hex=dose)
        p = GeneratorPipeline(42, cfg)
        for stage in (ElevationStage, ErosionStage, TerrainClassificationStage, ClimateStage):
            p.add_stage(stage)
        state = p.run()

        water = (TerrainClass.OCEAN, TerrainClass.LAKE)
        sheltered, exposed = [], []
        for r in range(state.height):
            barrier = 0.0
            for q in range(state.width):
                h = state.hexes.get((q, r))
                if h is None:
                    continue
                if h.terrain_class not in water:
                    behind = (barrier - cfg.sea_level) > 0.30
                    (sheltered if behind else exposed).append(h.moisture)
                barrier = max(barrier, h.elevation)
        if not sheltered or not exposed:
            return 0.0
        wet = sum(exposed) / len(exposed)
        dry = sum(sheltered) / len(sheltered)
        # How much drier the sheltered side is, as a fraction. Unit-independent, so this
        # keeps meaning the same thing whatever units moisture is carried in.
        return (wet - dry) / wet if wet > 0 else 0.0

    at_default = shadow(WorldConfig().erosion_droplets_per_hex)
    assert at_default > 0.15, (
        f"the default erosion dose leaves the sheltered side only {at_default:.0%} drier "
        "than the exposed one; the rain shadow has been eroded away"
    )
    assert at_default > shadow(8.0), (
        "a heavier dose should weaken the shadow — if it does not, this test is no "
        "longer exercising the coupling it was written for"
    )
