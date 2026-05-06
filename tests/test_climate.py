import pytest

from worldgen.core.config import WorldConfig
from worldgen.core.hex import TerrainClass
from worldgen.core.pipeline import GeneratorPipeline
from worldgen.stages.climate import ClimateStage
from worldgen.stages.elevation import ElevationStage
from worldgen.stages.erosion import ErosionStage
from worldgen.stages.hydrology import HydrologyStage
from worldgen.stages.terrain_class import TerrainClassificationStage


def _build_pipeline(seed: int = 42, width: int = 32, height: int = 32):
    cfg = WorldConfig(width=width, height=height, erosion_iterations=500)
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


def test_temperature_in_range(climate_state):
    for h in climate_state.hexes.values():
        assert 0.0 <= h.temperature <= 1.0, f"temperature {h.temperature} out of [0, 1]"


def test_moisture_in_range(climate_state):
    for h in climate_state.hexes.values():
        assert 0.0 <= h.moisture <= 1.0, f"moisture {h.moisture} out of [0, 1]"


def test_ocean_moisture_is_one(climate_state):
    for h in climate_state.hexes.values():
        if h.terrain_class == TerrainClass.OCEAN:
            assert h.moisture == 1.0, f"ocean hex moisture {h.moisture} != 1.0"


def test_mountains_colder_than_flat(climate_state):
    # Sample mountain and flat hexes at similar latitudes; mountains must be colder on average.
    height = climate_state.height
    mountain_temps = []
    flat_temps = []
    for (_, r), h in climate_state.hexes.items():
        mid = height * 0.3 < r < height * 0.7
        if not mid:
            continue
        if h.terrain_class == TerrainClass.MOUNTAIN:
            mountain_temps.append(h.temperature)
        elif h.terrain_class == TerrainClass.FLAT:
            flat_temps.append(h.temperature)

    if mountain_temps and flat_temps:
        assert sum(mountain_temps) / len(mountain_temps) < sum(flat_temps) / len(flat_temps), (
            "Mountain hexes not colder than flat hexes at similar latitude"
        )


def test_rain_shadow_present(climate_state):
    # Compare moisture on windward vs. leeward sides of mountain ranges.
    # Default wind blows east (positive q direction). For each mountain column,
    # windward neighbors (lower q) should have higher average moisture than leeward (higher q).
    windward_avg = []
    leeward_avg = []
    for (q, r), h in climate_state.hexes.items():
        if h.terrain_class != TerrainClass.MOUNTAIN:
            continue
        windward = climate_state.hexes.get((q - 1, r))
        leeward = climate_state.hexes.get((q + 1, r))
        if (
            windward
            and leeward
            and windward.terrain_class != TerrainClass.OCEAN
            and leeward.terrain_class != TerrainClass.OCEAN
        ):
            windward_avg.append(windward.moisture)
            leeward_avg.append(leeward.moisture)

    if windward_avg and leeward_avg:
        assert sum(windward_avg) / len(windward_avg) > sum(leeward_avg) / len(leeward_avg), (
            "Rain shadow not detected: windward side is not wetter than leeward"
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


def test_base_temperature_shifts_mean_upward():
    """Higher base_temperature should produce a higher mean land temperature."""

    def run_with_base(base: float):
        cfg = WorldConfig(width=32, height=32, erosion_iterations=500, base_temperature=base)
        p = GeneratorPipeline(42, cfg)
        p.add_stage(ElevationStage)
        p.add_stage(ErosionStage)
        p.add_stage(TerrainClassificationStage)
        p.add_stage(HydrologyStage)
        p.add_stage(ClimateStage)
        return p.run()

    cold_state = run_with_base(0.2)
    warm_state = run_with_base(0.8)
    assert _mean_land_temperature(cold_state) < _mean_land_temperature(warm_state), (
        "Higher base_temperature did not produce higher mean land temperature"
    )


def test_base_temperature_preserves_latitude_shape():
    """Changing base_temperature should shift temperatures but preserve the
    relative latitude ordering — equatorial hexes warmer than polar ones."""

    def run_with_base(base: float):
        cfg = WorldConfig(
            width=32,
            height=32,
            erosion_iterations=500,
            base_temperature=base,
            latitude_temp_range=0.3,  # large enough to distinguish rows
        )
        p = GeneratorPipeline(42, cfg)
        p.add_stage(ElevationStage)
        p.add_stage(ErosionStage)
        p.add_stage(TerrainClassificationStage)
        p.add_stage(HydrologyStage)
        p.add_stage(ClimateStage)
        return p.run()

    for base in (0.3, 0.7):
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
            ), f"With base_temperature={base}, equatorial hexes are not warmer than polar hexes"


def test_base_temperature_validation():
    """base_temperature outside [0, 1] should raise ValueError."""
    with pytest.raises(ValueError, match="base_temperature"):
        WorldConfig(base_temperature=-0.1)
    with pytest.raises(ValueError, match="base_temperature"):
        WorldConfig(base_temperature=1.1)


def test_latitude_temp_range_validation():
    """latitude_temp_range outside [0, 1] should raise ValueError."""
    with pytest.raises(ValueError, match="latitude_temp_range"):
        WorldConfig(latitude_temp_range=-0.01)
    with pytest.raises(ValueError, match="latitude_temp_range"):
        WorldConfig(latitude_temp_range=1.1)
