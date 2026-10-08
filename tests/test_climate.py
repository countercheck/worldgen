import pytest

from worldgen.core.config import WorldConfig
from worldgen.core.hex import TerrainClass, TerrainLabel, terrain_labels
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
        if h.terrain_class == TerrainClass.OPEN_WATER:
            assert h.moisture == pytest.approx(
                climate_state.metadata["config"]["mean_precip_mm"]
            ), f"water hex has {h.moisture:.0f} mm — it should not read as the driest ground"


def test_the_two_seasons_are_the_year(climate_state):
    """The split says when the rain falls, never how much: the halves sum to the year, the
    wet half is the wetter, and every hex divides by the region's share."""
    share = climate_state.metadata["config"]["wet_season_share"]
    for h in climate_state.hexes.values():
        assert h.wet_season_precip_mm + h.dry_season_precip_mm == pytest.approx(h.moisture)
        assert h.wet_season_precip_mm >= h.dry_season_precip_mm
        assert h.wet_season_precip_mm == pytest.approx(h.moisture * share)


def test_each_climate_has_its_own_seasons():
    """Summer drought is what makes the Mediterranean pastoral on rainfall that would
    plough in Kent, so its year must be the more bunched of the two."""
    from worldgen.core.config import CLIMATE_CONTEXTS

    for name in CLIMATE_CONTEXTS:
        assert 0.5 <= WorldConfig(regional_climate=name).wet_season_share <= 1.0
    assert (
        WorldConfig(regional_climate="mediterranean").wet_season_share
        > WorldConfig(regional_climate="temperate").wet_season_share
    )
    with pytest.raises(ValueError, match="wet_season_share"):
        WorldConfig(wet_season_share=0.4)


def test_a_shore_takes_no_rain_from_the_sea_beside_it(monkeypatch):
    """A lee shore is not wetter than its windward shore because of the sea behind it.

    The orographic sweep leaves open water holding its *carrier* value — saturated air, not
    rainfall — and the smear used to blend that into every coast, windward and lee alike
    (tech-debt #123). Change what the water holds, then, and no land hex's rain may move:
    the smear reads land only.
    """
    import copy

    import worldgen.stages.climate as climate_module
    from worldgen.stages.precipitation import orographic_pattern

    cfg = WorldConfig(width=32, height=32)
    p = GeneratorPipeline(42, cfg)
    for stage in (ElevationStage, ErosionStage, TerrainClassificationStage, HydrologyStage):
        p.add_stage(stage)
    before = p.run()
    water = (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER)
    coast = [c for c, h in before.hexes.items() if h.terrain_class is TerrainClass.COAST]
    assert coast, "the map has no shore to test"

    def run_with_sea_holding(value):
        def pattern(state, config):
            out = orographic_pattern(state, config)
            for c, h in state.hexes.items():
                if h.terrain_class in water:
                    out[c] = value
            return out

        monkeypatch.setattr(climate_module, "orographic_pattern", pattern)
        return ClimateStage(cfg, None).run(copy.deepcopy(before))

    saturated, dry = run_with_sea_holding(1.0), run_with_sea_holding(0.0)
    for c, h in saturated.hexes.items():
        if h.terrain_class not in water:
            assert h.moisture == pytest.approx(dry.hexes[c].moisture), (
                f"{c} ({h.terrain_class.value}) gets {h.moisture:.0f} mm beside a saturated "
                f"sea and {dry.hexes[c].moisture:.0f} mm beside a dry one: the smear is "
                "reading the water"
            )


def test_mountains_colder_than_flat(climate_state):
    # Sample mountain and flat hexes at similar latitudes; mountains must be colder on average.
    height = climate_state.height
    mountain_temps = []
    flat_temps = []
    labels = terrain_labels(climate_state)
    for (_, r), h in climate_state.hexes.items():
        mid = height * 0.3 < r < height * 0.7
        if not mid:
            continue
        if labels[h.coord] is TerrainLabel.STEEP:
            mountain_temps.append(h.temperature)
        elif labels[h.coord] is TerrainLabel.FLAT:
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
    water = (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER)

    land = [h.elevation for h in climate_state.hexes.values() if h.terrain_class not in water]
    land.sort()
    high = land[int(len(land) * 0.8)]

    # Wind blows east by default, so upwind is west along a row.
    sheltered, exposed = [], []
    for r in range(climate_state.height):
        barrier = 0.0
        for q in range(climate_state.width):
            h = climate_state.hexes.get((q, r))
            if h is None:
                continue
            if h.terrain_class not in water:
                behind = barrier > high
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
    temps = [
        h.temperature for h in state.hexes.values() if h.terrain_class != TerrainClass.OPEN_WATER
    ]
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
            if h.terrain_class != TerrainClass.OPEN_WATER and r < height * 0.15
        ]
        equatorial_temps = [
            h.temperature
            for (_, r), h in state.hexes.items()
            if h.terrain_class != TerrainClass.OPEN_WATER and height * 0.4 < r < height * 0.6
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
        if h.terrain_class not in (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER)
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


def test_moisture_bleed_requires_a_river():
    """A river wets the ground beside it at or below its own level, and nothing else does."""
    from worldgen.core.hex_grid import side_between
    from worldgen.core.world_state import RiverSide

    cfg = WorldConfig(width=3, height=1, moisture_bleed_passes=1, moisture_bleed_strength=0.5)
    stage = ClimateStage(cfg, None)

    def world(with_river):
        state = WorldState.empty(seed=1, width=3, height=1)
        for hx in state.hexes.values():
            hx.terrain_class = TerrainClass.LAND
            hx.elevation = 0.0
        if with_river:
            state.river_sides[side_between((0, 0), (1, 0))] = RiverSide(100.0, 1.0)
        return stage.run(state)

    dry = world(False).hexes[(1, 0)].moisture
    assert world(True).hexes[(1, 0)].moisture > dry, (
        "the bleed did nothing: a river beside a hex should wet it"
    )


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

    Orographic lift is height above sea level, and erosion wears high ground down, so
    weather and rain shadow pull against each other: enough droplets to cut floodplains
    also flatten the barriers that make a leeward side dry. At eight per hex the high
    ground is all but gone — 0-4% of land stands 0.30 above sea level, against 13-19% at
    the default — and the shadow closes to nothing on a small map. This pins the default
    on the right side of that, so raising it for flatter country cannot silently cost the
    map its dry country.

    Pinned as the coupling rather than an absolute: the default's shadow must be at least
    half as large again as the heavy dose's. On this 48x48 map it is 10.5% against 5.6%.
    It used to be asserted as above 15%, measured at 28% against 14% — but most of that
    28% was not shadow. The climate smear blended the sea's carrier value into the coasts
    (tech-debt #123), and the exposed side, which takes the windward shore and most of the
    land near open water, read that as rain. Over land only, the exposed side drops from
    920 mm to 859 and the sheltered side rises from 659 to 769, and the gap between the
    two doses is what is left to measure.

    With `elevation_profile` off: the profile puts the land back on its heights after
    erosion, so under it the dose no longer lowers the high ground at all, and the
    coupling this pins only exists in the noise's own relief.
    """

    def shadow(dose):
        cfg = WorldConfig(
            width=48, height=48, erosion_droplets_per_hex=dose, elevation_profile="none"
        )
        p = GeneratorPipeline(42, cfg)
        for stage in (ElevationStage, ErosionStage, TerrainClassificationStage, ClimateStage):
            p.add_stage(stage)
        state = p.run()

        water = (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER)
        land = [h.elevation for h in state.hexes.values() if h.terrain_class not in water]
        land.sort()
        high = land[int(len(land) * 0.8)]
        sheltered, exposed = [], []
        for r in range(state.height):
            barrier = 0.0
            for q in range(state.width):
                h = state.hexes.get((q, r))
                if h is None:
                    continue
                if h.terrain_class not in water:
                    behind = barrier > high
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
    at_heavy = shadow(8.0)
    assert at_default > 0.0, "the default map has no rain shadow at all"
    assert at_default > 1.5 * at_heavy, (
        f"the default erosion dose leaves the sheltered side {at_default:.0%} drier than "
        f"the exposed one, against {at_heavy:.0%} at eight droplets a hex: the default is "
        "no longer clearly on the side of the coupling that keeps its dry country"
    )
