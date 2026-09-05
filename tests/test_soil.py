"""Soil quality: what the ground could support, before anything is done with it.

The rules are unit-testable on their own — each arm answers one question — so most of this
tests the arms directly and only then checks that a whole map comes out looking like the
country it is named after.
"""

import collections

from tests.worlds import build_world
from worldgen.core.config import WorldConfig
from worldgen.core.hex import SOIL_RANK, Biome, Hex, LandCover, SoilQuality, TerrainClass
from worldgen.core.hex_grid import neighbors
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
    assert rainfall_soil(100, cfg) is SoilQuality.UNUSABLE
    assert rainfall_soil(320, cfg) is SoilQuality.GRAZING
    assert rainfall_soil(700, cfg) is SoilQuality.ARABLE
    assert rainfall_soil(1800, cfg) is SoilQuality.MARGINAL
    assert rainfall_soil(3200, cfg) is SoilQuality.UNUSABLE


def test_rainfall_is_best_across_the_arable_band():
    """The band is `biome_dry_precip_mm`..`biome_wet_precip_mm`, the same pair BiomeStage
    classifies on, so the two systems cannot drift apart about what counts as wet."""
    cfg = WorldConfig()
    for mm in (cfg.biome_dry_precip_mm, 700, cfg.biome_wet_precip_mm):
        assert rainfall_soil(mm, cfg) is SoilQuality.ARABLE


def test_rainfall_is_monotonic_toward_the_band():
    """Drier than the band should only improve as it gets wetter, and the reverse above."""
    cfg = WorldConfig()
    dry = [SOIL_RANK[rainfall_soil(mm, cfg)] for mm in range(0, 400, 20)]
    wet = [SOIL_RANK[rainfall_soil(mm, cfg)] for mm in range(1020, 3200, 100)]
    assert dry == sorted(dry)
    assert wet == sorted(wet, reverse=True)


# --- alluvium ----------------------------------------------------------------


def _river_pair(catchment_km2, gradient_drop_m=0.0):
    """A gentle hex beside one river hex of the given catchment."""
    here = (0, 0)
    there = neighbors(here)[0]
    hexes = {
        # `slope` is measured by TerrainClassificationStage and read from the hex, so a
        # hand-built pair has to state it rather than leave it to be re-derived.
        here: Hex(coord=here, elevation=100.0, slope=gradient_drop_m),
        there: Hex(
            coord=there,
            elevation=100.0 - gradient_drop_m,
            catchment_km2=catchment_km2,
            slope=gradient_drop_m,
            tags={"river"},
        ),
    }
    return here, hexes


def test_alluvium_wants_a_river_too_big_to_wade():
    """`ford_max_catchment_km2` is the threshold, asked from the other side.

    A stream draining a few tens of square kilometres is ankle deep and a step across; a
    river you cannot wade is one that floods and lays down silt. Reusing the fording figure
    means the map cannot hold one opinion about a river's size for crossing it and another
    for what it deposits.
    """
    cfg = WorldConfig()
    big, hexes = _river_pair(cfg.ford_max_catchment_km2 + 1)
    assert is_alluvium(big, hexes[big], hexes, cfg)

    small, hexes = _river_pair(cfg.ford_max_catchment_km2 - 1)
    assert not is_alluvium(small, hexes[small], hexes, cfg)


def test_alluvium_wants_ground_the_river_can_spread_over():
    """Silt settles where the water slows and spreads. A torrent in a gorge cuts."""
    cfg = WorldConfig()
    coord, hexes = _river_pair(500.0, gradient_drop_m=cfg.terrain_rolling_gradient_m * 3)
    assert not is_alluvium(coord, hexes[coord], hexes, cfg)


def test_a_hex_with_no_river_near_it_is_not_alluvium():
    cfg = WorldConfig()
    hexes = {(0, 0): Hex(coord=(0, 0), elevation=100.0)}
    assert not is_alluvium((0, 0), hexes[(0, 0)], hexes, cfg)


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
    for coord, hx in state.hexes.items():
        if hx.soil is SoilQuality.PRIME:
            assert is_alluvium(coord, hx, state.hexes, cfg), f"{coord} is prime off a floodplain"


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
