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
    assert wet > dry + 0.05, (
        f"Rain shadow not detected: land behind high ground averages {dry:.2f} moisture "
        f"against {wet:.2f} in front of it. Erosion wearing the barriers down is the "
        f"usual cause — see erosion_droplets_per_hex."
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
        cfg = WorldConfig(width=32, height=32, base_temperature=base)
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


def _mean_land_moisture(state) -> float:
    from worldgen.core.hex import TerrainClass

    vals = [
        h.moisture
        for h in state.hexes.values()
        if h.terrain_class not in (TerrainClass.OCEAN, TerrainClass.LAKE)
    ]
    return sum(vals) / len(vals) if vals else 0.0


def test_base_moisture_shifts_mean_upward():
    """Positive base_moisture should raise mean land moisture."""

    def run(base: float):
        cfg = WorldConfig(width=32, height=32, base_moisture=base)
        p = GeneratorPipeline(42, cfg)
        p.add_stage(ElevationStage)
        p.add_stage(ErosionStage)
        p.add_stage(TerrainClassificationStage)
        p.add_stage(HydrologyStage)
        p.add_stage(ClimateStage)
        return p.run()

    dry = run(0.0)
    wet = run(0.3)
    assert _mean_land_moisture(dry) < _mean_land_moisture(wet), (
        "Positive base_moisture did not raise mean land moisture"
    )


def test_base_moisture_clamps_to_unit_interval():
    """base_moisture = 1.0 should not push moisture above 1.0."""

    cfg = WorldConfig(width=32, height=32, base_moisture=1.0)
    p = GeneratorPipeline(42, cfg)
    p.add_stage(ElevationStage)
    p.add_stage(ErosionStage)
    p.add_stage(TerrainClassificationStage)
    p.add_stage(HydrologyStage)
    p.add_stage(ClimateStage)
    state = p.run()
    for h in state.hexes.values():
        assert h.moisture <= 1.0, f"moisture {h.moisture} exceeded 1.0 with base_moisture=1.0"


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
    """latitude_temp_range outside [0, 1] should raise ValueError."""
    with pytest.raises(ValueError, match="latitude_temp_range"):
        WorldConfig(latitude_temp_range=-0.01)
    with pytest.raises(ValueError, match="latitude_temp_range"):
        WorldConfig(latitude_temp_range=1.1)


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
        return sum(exposed) / len(exposed) - sum(sheltered) / len(sheltered)

    at_default = shadow(WorldConfig().erosion_droplets_per_hex)
    assert at_default > 0.10, (
        f"the default erosion dose leaves only a {at_default:.2f} moisture contrast "
        "across high ground; the rain shadow has been eroded away"
    )
    assert at_default > shadow(8.0), (
        "a heavier dose should weaken the shadow — if it does not, this test is no "
        "longer exercising the coupling it was written for"
    )
