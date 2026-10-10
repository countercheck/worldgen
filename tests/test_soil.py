"""Soil quality: what the ground could support, before anything is done with it.

The rules are unit-testable on their own — each arm answers one question — so most of this
tests the arms directly and only then checks that a whole map comes out looking like the
country it is named after.
"""

import collections

from tests.worlds import build_world
from worldgen.core.config import WorldConfig, halves_as_seasons, season_split
from worldgen.core.hex import SOIL_RANK, Biome, Hex, LandCover, SoilQuality, TerrainClass
from worldgen.stages.riverside import Rivers, river_index
from worldgen.stages.soil import is_alluvium, rainfall_soil, slope_soil

# --- the slope arm -----------------------------------------------------------


def test_slope_reads_off_the_terrain_bands():
    """No thresholds of its own: a hex is 1 km across, so the gradient bands already say
    what a plough and a cart can manage, and a second set could only disagree."""
    cfg = WorldConfig()
    assert slope_soil(cfg.terrain_escarpment_gradient_m, cfg) is SoilQuality.UNUSABLE
    assert slope_soil(cfg.terrain_steep_gradient_m, cfg) is SoilQuality.GRAZING
    assert slope_soil(cfg.terrain_steep_gradient_m - 1, cfg) is SoilQuality.ARABLE
    assert slope_soil(0.0, cfg) is SoilQuality.ARABLE


def test_slope_never_improves_with_steepness():
    cfg = WorldConfig()
    ranks = [SOIL_RANK[slope_soil(g, cfg)] for g in range(0, 400, 10)]
    assert ranks == sorted(ranks, reverse=True)


# --- the rainfall arm --------------------------------------------------------


def test_rainfall_fails_differently_at_each_end():
    """Too dry and too wet are not the same failure, and one symmetric rule cannot say so.

    Under the dry-farming limit nothing is grown at all. Between that and the arable band
    you get steppe — grass will grow and a crop will not — which is grazing. Above the band
    the ground is leached and waterlogged, which is poor *arable*, not pasture: calling a
    rainforest "grazing" was the tell that the first version of this rule was wrong.
    """
    cfg = WorldConfig()
    assert even_year(100, cfg) is SoilQuality.UNUSABLE
    assert even_year(320, cfg) is SoilQuality.GRAZING
    assert even_year(700, cfg) is SoilQuality.ARABLE
    assert even_year(1800, cfg) is SoilQuality.MARGINAL
    assert even_year(3200, cfg) is SoilQuality.UNUSABLE


def even_year(annual_mm, cfg):
    """A year without seasons: each season gets a quarter of the rain."""
    return rainfall_soil(season_split(annual_mm, (0.25, 0.25, 0.25, 0.25)), cfg)


def seasonal_year(annual_mm, wet_season_share, cfg):
    """#145's year of two halves, as four seasons: the wet half in autumn and winter, the
    dry half in spring and summer (a temperate region's wettest pair)."""
    shares, _, _ = halves_as_seasons(wet_season_share, "temperate")
    return rainfall_soil(season_split(annual_mm, shares), cfg)


# The two halves' growing seasons in `seasonal_year`.
IN_WET = ("autumn", "winter")
IN_DRY = ("spring", "summer")


def test_rainfall_is_best_across_the_arable_band():
    """The band is `biome_dry_precip_mm`..`biome_wet_precip_mm`, the same pair BiomeStage
    classifies on, so the two systems cannot drift apart about what counts as wet. Soil
    reads each season against a quarter of it, and in a year without seasons that is
    exactly the annual band."""
    cfg = WorldConfig()
    for mm in (cfg.biome_dry_precip_mm, 700, cfg.biome_wet_precip_mm):
        assert even_year(mm, cfg) is SoilQuality.ARABLE
    assert even_year(cfg.biome_dry_precip_mm - 1, cfg) is not SoilQuality.ARABLE
    assert even_year(cfg.biome_wet_precip_mm + 1, cfg) is not SoilQuality.ARABLE


def test_rainfall_is_monotonic_toward_the_band():
    """Drier than the band should only improve as it gets wetter, and the reverse above."""
    cfg = WorldConfig()
    dry = [SOIL_RANK[even_year(mm, cfg)] for mm in range(0, 400, 20)]
    wet = [SOIL_RANK[even_year(mm, cfg)] for mm in range(1020, 3200, 100)]
    assert dry == sorted(dry)
    assert wet == sorted(wet, reverse=True)


def _pr145_rainfall_soil(wet_mm, dry_mm, cfg, in_wet, groundwater_mm=0.0):
    """#145's rule over two half-years, kept here as the reference the seasons must match."""
    carried = cfg.soil_water_carryover * max(0.0, wet_mm - dry_mm)
    growing = wet_mm if in_wet else dry_mm + carried
    dry = 2.0 * (growing + groundwater_mm)
    wet = 2.0 * (wet_mm - carried)
    if dry < cfg.soil_dry_farming_min_precip_mm:
        return SoilQuality.UNUSABLE
    if dry < cfg.biome_dry_precip_mm:
        return SoilQuality.GRAZING
    if wet <= cfg.biome_wet_precip_mm:
        return SoilQuality.ARABLE
    if wet < cfg.food_drowned_precip_mm:
        return SoilQuality.MARGINAL
    return SoilQuality.UNUSABLE


def test_four_seasons_holding_two_halves_read_as_the_halves_did():
    """The calibration: a year whose four shares are #145's two halves, each split evenly
    over two seasons, reads exactly as #145 read it, for every climate's region, share,
    carryover, growing half and oasis."""
    for climate in ("temperate", "mediterranean", "arid", "tropical", "boreal"):
        for share in (0.5, 0.55, 0.65, 0.75, 0.8, 0.95):
            shares, wet_pair, dry_pair = halves_as_seasons(share, climate)
            for in_wet in (True, False):
                for carryover in (0.0, 0.3, 0.5):
                    cfg = WorldConfig(
                        regional_climate=climate,
                        season_shares=shares,
                        growing_seasons=wet_pair if in_wet else dry_pair,
                        soil_water_carryover=carryover,
                    )
                    for annual in range(50, 4000, 37):
                        for groundwater in (0.0, 250.0):
                            seasons = season_split(annual, shares)
                            old = _pr145_rainfall_soil(
                                annual * share, annual - annual * share, cfg, in_wet, groundwater
                            )
                            assert rainfall_soil(seasons, cfg, groundwater) is old, (
                                climate,
                                share,
                                in_wet,
                                carryover,
                                annual,
                                groundwater,
                            )


def test_the_dry_arm_reads_the_growing_seasons():
    """A crop fails for want of water in the seasons it grows in, so the same year farms
    better grown in its wet seasons than in its dry ones, and where there are several the
    dry arm reads their mean: a dry summer is made up by a wet spring."""
    seasons = (300.0, 10.0, 200.0, 290.0)  # 800 mm with a summer drought

    def grown_in(*names):
        return rainfall_soil(seasons, WorldConfig(growing_seasons=names))

    assert grown_in("summer") is SoilQuality.GRAZING
    assert grown_in("spring") is SoilQuality.ARABLE
    assert SOIL_RANK[grown_in("spring", "summer")] >= SOIL_RANK[grown_in("summer")]
    # Only the growing seasons are read: moving rain between two seasons the crop does not
    # grow in, without making either the wettest or the driest, changes nothing.
    shifted = (300.0, 10.0, 290.0, 200.0)
    cfg = WorldConfig(growing_seasons=("summer",))
    assert rainfall_soil(shifted, cfg) is rainfall_soil(seasons, cfg)


def test_a_drier_dry_season_farms_worse_on_the_same_rain():
    """The Mediterranean's summer drought, as a rule: bunching the same annual rain into
    one season can only cost the dry arm, never help it.

    480 mm falling evenly ploughs; the same 480 mm with three quarters of it in winter
    leaves a summer of steppe, which is grazing, even with the winter's water the ground
    carries over.
    """
    cfg = WorldConfig(growing_seasons=IN_DRY)
    assert even_year(480, cfg) is SoilQuality.ARABLE
    assert seasonal_year(480, 0.75, cfg) is SoilQuality.GRAZING
    shares = [0.5 + 0.05 * i for i in range(11)]
    for carryover in (0.0, cfg.soil_water_carryover, 0.49):
        c = WorldConfig(growing_seasons=IN_DRY, soil_water_carryover=carryover)
        for annual in range(300, 1000, 25):
            ranks = [SOIL_RANK[seasonal_year(annual, s, c)] for s in shares]
            assert ranks == sorted(ranks, reverse=True), (
                f"at {annual} mm a drier dry season made the ground better: {ranks}"
            )


def test_a_wetter_wet_season_leaches_sooner():
    """And the wet arm reads the wettest season: the same rain bunched into part of the
    year leaches ground that would have been arable had it fallen evenly."""
    cfg = WorldConfig()
    assert even_year(950, cfg) is SoilQuality.ARABLE
    assert seasonal_year(950, 0.75, cfg) is SoilQuality.MARGINAL
    # A single peak quarter does it as surely as a wet half-year.
    assert rainfall_soil(season_split(950, (0.15, 0.55, 0.2, 0.1)), cfg) is (SoilQuality.MARGINAL)


def test_a_crop_grown_in_the_rains_never_ploughs_worse():
    """Where crops grow in the wet seasons the drought falls on a fallow one, so at equal
    rain and shares a monsoon or taiga year is never worse ground than a Mediterranean one
    — and bunching the rain cannot make a wet-season crop thirsty."""
    in_wet = WorldConfig(growing_seasons=IN_WET)
    in_dry = WorldConfig(growing_seasons=IN_DRY)
    for annual in range(100, 3400, 25):
        for share in (0.5, 0.55, 0.65, 0.75, 0.9, 1.0):
            wet = SOIL_RANK[seasonal_year(annual, share, in_wet)]
            dry = SOIL_RANK[seasonal_year(annual, share, in_dry)]
            assert wet >= dry, (annual, share)
    # 480 mm bunched into a summer monsoon still ploughs; into a winter it is grazing.
    assert seasonal_year(480, 0.75, in_wet) is SoilQuality.ARABLE
    assert seasonal_year(480, 0.75, in_dry) is SoilQuality.GRAZING


def test_each_climate_grows_its_crops_in_its_own_season():
    """The Mediterranean grows through its summer drought; the taiga and the monsoon in
    their rains."""

    def in_wettest(cfg):
        wettest = max(range(4), key=lambda i: cfg.season_shares[i])
        return wettest in cfg.growing_season_indices

    assert not in_wettest(WorldConfig(regional_climate="mediterranean"))
    assert "summer" in WorldConfig(regional_climate="mediterranean").growing_seasons
    assert in_wettest(WorldConfig(regional_climate="boreal"))
    assert in_wettest(WorldConfig(regional_climate="tropical"))
    assert WorldConfig(regional_climate="boreal", growing_seasons=("winter",))


def test_the_ground_carries_the_rains_into_the_drought():
    """Carryover only ever narrows the gap between the seasons — it eases the drought and
    the leaching both — so more of it never makes ground worse, and a perfect store (0.5)
    reads any year of two halves as an even one."""
    for annual in range(200, 3200, 50):
        for share in (0.6, 0.75, 0.9, 1.0):
            bare = seasonal_year(annual, share, WorldConfig(soil_water_carryover=0.0))
            stored = seasonal_year(annual, share, WorldConfig(soil_water_carryover=0.3))
            assert SOIL_RANK[stored] >= SOIL_RANK[bare], (annual, share)
            perfect = WorldConfig(soil_water_carryover=0.5)
            assert seasonal_year(annual, share, perfect) is even_year(annual, perfect)


# --- alluvium ----------------------------------------------------------------


def _river_pair(catchment_km2, gradient_drop_m=0.0):
    """A hex in the valley of a river of the given catchment, and the rivers index saying so.

    `slope` is measured by TerrainClassificationStage and read from the hex, so a hand-built
    hex has to state it rather than leave it to be re-derived.
    """
    here = (0, 0)
    hx = Hex(coord=here, elevation=100.0, slope=gradient_drop_m)
    return here, hx, Rivers(near={here: catchment_km2})


def test_alluvium_wants_a_river_too_big_to_wade():
    """`ford_max_catchment_km2` is the threshold, asked from the other side.

    A stream draining a few tens of square kilometres is ankle deep and a step across; a
    river you cannot wade is one that floods and lays down silt. Reusing the fording figure
    means the map cannot hold one opinion about a river's size for crossing it and another
    for what it deposits.
    """
    cfg = WorldConfig()
    coord, hx, rivers = _river_pair(cfg.ford_max_catchment_km2 + 1)
    assert is_alluvium(coord, hx, cfg, rivers)
    coord, hx, rivers = _river_pair(cfg.ford_max_catchment_km2 - 1)
    assert not is_alluvium(coord, hx, cfg, rivers)


def test_alluvium_wants_ground_the_river_can_spread_over():
    """Silt settles where the water slows and spreads. A torrent in a gorge cuts."""
    cfg = WorldConfig()
    coord, hx, rivers = _river_pair(500.0, gradient_drop_m=cfg.terrain_rolling_gradient_m * 3)
    assert not is_alluvium(coord, hx, cfg, rivers)


def test_a_hex_with_no_river_near_it_is_not_alluvium():
    cfg = WorldConfig()
    assert not is_alluvium((0, 0), Hex(coord=(0, 0), elevation=100.0), cfg, Rivers())


# --- whole maps --------------------------------------------------------------


def _soiled(**over):
    return build_world(
        seed=42,
        width=96,
        height=96,
        model="organic",
        until="SoilStage",
        continent_falloff_edges=("south",),
        **over,
    )


def _shares(state):
    land = [h for h in state.hexes.values() if h.terrain_class is not TerrainClass.OPEN_WATER]
    counts = collections.Counter(h.soil for h in land)
    return {k: v / len(land) for k, v in counts.items()}, land


def test_prime_is_scarce_and_always_on_a_river():
    """Floodplain is rare, and it is the one class that comes from the drainage.

    A rule that made a third of the map prime would not be describing alluvium.
    """
    state = _soiled(regional_climate="temperate")
    cfg = WorldConfig(**state.metadata["config"])
    shares, _ = _shares(state)
    assert 0.0 < shares.get(SoilQuality.PRIME, 0.0) < 0.15, (
        f"prime is {shares.get(SoilQuality.PRIME, 0.0):.1%} of the map"
    )
    rivers = river_index(state, cfg)
    for coord, hx in state.hexes.items():
        if hx.soil is SoilQuality.PRIME:
            assert is_alluvium(coord, hx, cfg, rivers), f"{coord} is prime off a floodplain"


def test_water_and_wetland_take_no_soil_class():
    """Neither is ploughland, so neither is described as ploughland.

    They are valued in their own right by `potential_food` — the sea as a fishery, a fen as
    a fen — and a bog that came out PRIME for being flat beside a big river would be the
    model contradicting itself.
    """
    for hx in _soiled(regional_climate="temperate").hexes.values():
        if hx.terrain_class in (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER) or hx.biome in (
            Biome.OCEAN,
            Biome.WETLAND,
        ):
            assert hx.soil is SoilQuality.UNUSABLE


def test_above_the_treeline_nothing_grows_at_any_rainfall():
    state = _soiled(regional_climate="boreal")
    for hx in state.hexes.values():
        if hx.biome in (Biome.ALPINE, Biome.TUNDRA):
            assert hx.soil is SoilQuality.UNUSABLE


def test_the_taiga_grows_no_wheat():
    """Podzol is poor ground however flat it is and however much rain falls on it.

    Without the cold cap a boreal map comes out 13% arable, which is wheat in the taiga —
    and it lifted the region's food by 42%, so this is load-bearing rather than cosmetic.
    """
    state = _soiled(regional_climate="boreal")
    cfg = WorldConfig(**state.metadata["config"])
    for hx in state.hexes.values():
        if hx.temperature < cfg.biome_cold_temp_c:
            assert SOIL_RANK[hx.soil] <= SOIL_RANK[SoilQuality.MARGINAL]


def test_each_climate_comes_out_as_the_country_it_is_named_after():
    """One rule set, and the region decides the answer rather than a per-climate table.

    Phrased as comparisons between climates rather than "temperate is mostly arable",
    because which single class wins moves with map size — a smaller map is rougher, so a
    96x96 temperate map is 36% grazing to 28% arable where a 128x128 is 33% to 28%. The
    ordering between climates holds at either size, and the ordering is the claim.
    """
    shares = {
        climate: _shares(_soiled(regional_climate=climate))[0]
        for climate in ("temperate", "mediterranean", "arid", "tropical")
    }

    def has(climate, *classes):
        return sum(shares[climate].get(c, 0.0) for c in classes)

    farmland = (SoilQuality.ARABLE, SoilQuality.PRIME)
    summary = "; ".join(
        f"{c}: "
        + ", ".join(f"{k.value} {v:.0%}" for k, v in sorted(s.items(), key=lambda kv: -kv[1]))
        for c, s in shares.items()
    )
    assert has("temperate", *farmland) == max(has(c, *farmland) for c in shares), (
        f"temperate is not the best farmland of the four. {summary}"
    )
    assert has("mediterranean", SoilQuality.GRAZING) > has("temperate", SoilQuality.GRAZING), (
        f"mediterranean should be the more pastoral. {summary}"
    )
    assert has("arid", SoilQuality.UNUSABLE) == max(has(c, SoilQuality.UNUSABLE) for c in shares), (
        f"a desert should be the emptiest of the four. {summary}"
    )
    assert has("tropical", SoilQuality.MARGINAL) > has("temperate", SoilQuality.MARGINAL), (
        f"the tropics should be the more leached. {summary}"
    )


def test_good_soil_carries_wildwood():
    """The point of separating soil from cover.

    A temperate map used to be half open grass, which had the causality backwards: grass is
    what you get after clearing or on thin soil. Prime and arable ground should be under
    trees until somebody clears it.
    """
    state = build_world(
        seed=42,
        width=96,
        height=96,
        model="organic",
        until="LandCoverStage",
        continent_falloff_edges=("south",),
        regional_climate="temperate",
    )
    wooded = {LandCover.WOODLAND, LandCover.DENSE_FOREST}
    good = [
        h
        for h in state.hexes.values()
        if h.soil in (SoilQuality.PRIME, SoilQuality.ARABLE)
        and h.terrain_class is not TerrainClass.OPEN_WATER
    ]
    assert good
    under_trees = sum(1 for h in good if h.land_cover in wooded)
    assert under_trees / len(good) > 0.9, (
        f"only {under_trees / len(good):.0%} of the good soil is wooded"
    )


def test_same_seed_same_soil():
    a, b = _soiled(regional_climate="temperate"), _soiled(regional_climate="temperate")
    assert {c: h.soil for c, h in a.hexes.items()} == {c: h.soil for c, h in b.hexes.items()}
