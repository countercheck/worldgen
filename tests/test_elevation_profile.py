"""`elevation_profile`, the closed hollows it leaves, and rivers that wander.

The profile sets the share of land at each height before erosion and holds it through
erosion; the hollows that shaping leaves above sea level become lakes or wetland; and
water picks its way downhill at random, weighted to the steepest.
"""

import numpy as np
import pytest

from tests.worlds import build_world
from worldgen.core.config import ELEVATION_PROFILES, WorldConfig
from worldgen.core.hex import TerrainClass
from worldgen.core.hex_grid import neighbors
from worldgen.stages.elevation import apply_profile, profile_heights

_KW = {"width": 48, "height": 48}
_WATER = (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER)


def _land_heights(ws) -> np.ndarray:
    return np.array([h.elevation for h in ws.hexes.values() if h.elevation > 0.0])


# -- the profile itself -------------------------------------------------------------------


def test_profile_heights_have_the_mode_and_median_asked_for():
    heights = profile_heights(20001, 50.0, 150.0, 1500.0)
    assert np.all(np.diff(heights) >= 0)
    assert np.median(heights) == pytest.approx(150.0, rel=0.05)
    counts, edges = np.histogram(heights, bins=np.arange(0, 400, 10))
    assert 30 <= edges[np.argmax(counts)] <= 70
    assert heights[-1] <= 1500.0


def test_apply_profile_keeps_rank_sea_and_coast():
    rng = np.random.default_rng(1)
    arr = rng.normal(200.0, 300.0, size=(30, 30))
    out = apply_profile(arr, WorldConfig())
    land = arr > 0.0
    assert np.array_equal(out > 0.0, land)
    assert np.array_equal(out[~land], arr[~land])
    assert np.array_equal(
        np.argsort(arr[land], kind="stable"), np.argsort(out[land], kind="stable")
    )


def test_none_keeps_the_noise():
    arr = np.linspace(-100, 900, 100).reshape(10, 10)
    assert np.array_equal(apply_profile(arr, WorldConfig(elevation_profile="none")), arr)


@pytest.mark.parametrize("profile", sorted(ELEVATION_PROFILES))
def test_a_finished_map_has_its_profiles_median(profile):
    """Erosion takes off a share of every height; the profile is put back after it."""
    _, median = ELEVATION_PROFILES[profile]
    ws = build_world(until="ErosionStage", elevation_profile=profile, **_KW)
    assert np.median(_land_heights(ws)) == pytest.approx(median, rel=0.1)


def test_erosion_keeps_the_land_share_it_was_given():
    before = build_world(until="ElevationStage", **_KW)
    after = build_world(until="ErosionStage", **_KW)
    share = lambda ws: np.mean([h.elevation > 0.0 for h in ws.hexes.values()])  # noqa: E731
    assert share(after) == pytest.approx(share(before), abs=0.02)


def test_custom_profile_uses_its_own_numbers():
    ws = build_world(
        until="ErosionStage",
        elevation_profile="custom",
        elevation_profile_mode_m=300.0,
        elevation_profile_median_m=600.0,
        # High enough that the cut at the ceiling barely touches the tail, which would
        # otherwise pull the median down.
        max_elevation_m=6000.0,
        **_KW,
    )
    assert np.median(_land_heights(ws)) == pytest.approx(600.0, rel=0.1)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"elevation_profile": "alpine"}, "unknown elevation_profile"),
        (
            {
                "elevation_profile": "custom",
                "elevation_profile_mode_m": 200.0,
                "elevation_profile_median_m": 100.0,
            },
            "custom elevation_profile needs",
        ),
        ({"river_wander_exponent": -1.0}, "river_wander_exponent"),
        ({"lake_min_hexes": 0}, "lake_min_hexes"),
    ],
)
def test_bad_settings_are_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        WorldConfig(**kwargs)


def test_the_old_curve_setting_is_retired_with_a_pointer(tmp_path):
    config = tmp_path / "old.yaml"
    config.write_text("elevation_hypsometry_exponent: 2.0\n", encoding="utf-8")
    with pytest.warns(DeprecationWarning, match="elevation_profile"):
        WorldConfig.from_yaml(str(config))


# -- hollows ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def watered():
    return build_world(until="HydrologyStage", width=64, height=64)


def _spill_levels(ws) -> dict:
    """The level each hex must fill to before water runs out of it."""
    import heapq

    hexes = ws.hexes
    level = {c: h.elevation for c, h in hexes.items()}
    water = {c for c, h in hexes.items() if h.terrain_class in _WATER}
    seen = water | {c for c in hexes if ws.on_border(c)}
    heap = [(level[c], c) for c in seen]
    heapq.heapify(heap)
    while heap:
        e, c = heapq.heappop(heap)
        for n in neighbors(c):
            if n in hexes and n not in seen:
                seen.add(n)
                level[n] = max(level[n], e)
                heapq.heappush(heap, (level[n], n))
    return level


def test_no_hollow_big_and_deep_enough_for_a_lake_is_left_dry(watered):
    cfg = WorldConfig()
    level = _spill_levels(watered)
    under = {c for c, h in watered.hexes.items() if level[c] - h.elevation > 1e-9}
    seen = set()
    for start in under:
        if start in seen:
            continue
        comp, stack = {start}, [start]
        while stack:
            for n in neighbors(stack.pop()):
                if n in under and n not in comp:
                    comp.add(n)
                    stack.append(n)
        seen |= comp
        depth = max(level[c] - watered.hexes[c].elevation for c in comp)
        assert not (len(comp) >= cfg.lake_min_hexes and depth >= cfg.lake_min_depth_m)


def test_a_map_with_hollows_gets_lakes_or_wetland():
    ws = build_world(until="BiomeStage", width=64, height=64)
    lakes = sum(h.terrain_class == TerrainClass.INLAND_WATER for h in ws.hexes.values())
    hollows = sum("hollow" in h.tags for h in ws.hexes.values())
    assert lakes + hollows > 0


def test_a_dry_region_fills_no_hollows():
    """Only a basin below sea level holds water there; a hollow above it stays dry."""
    ws = build_world(until="WaterBodiesStage", lake_min_runoff_mm=1e9, width=64, height=64)
    assert not any(
        h.terrain_class == TerrainClass.INLAND_WATER and h.elevation > 0.0
        for h in ws.hexes.values()
    )


# -- wandering ----------------------------------------------------------------------------


def _longest_straight_run(path) -> int:
    best = cur = 1
    for a, b, c in zip(path, path[1:], path[2:], strict=False):
        same = (b[0] - a[0], b[1] - a[1]) == (c[0] - b[0], c[1] - b[1])
        cur = cur + 1 if same else 1
        best = max(best, cur)
    return best


def test_rivers_still_run_downhill_to_water_when_they_wander(watered):
    hexes = watered.hexes
    for river in watered.rivers:
        for a, b in zip(river.hexes, river.hexes[1:], strict=False):
            assert b in neighbors(a)
        end = river.hexes[-1]
        assert (
            hexes[end].terrain_class in _WATER
            or watered.on_border(end)
            or any(hexes[n].terrain_class in _WATER for n in neighbors(end) if n in hexes)
            or any(end in r.hexes[:-1] for r in watered.rivers if r is not river)
        )


def test_wandering_breaks_up_straight_runs():
    """The same map routed by steepest descent alone has longer straight runs."""
    runs = {}
    for power in (1.0, 50.0):
        ws = build_world(until="HydrologyStage", river_wander_exponent=power, width=64, height=64)
        runs[power] = sum(_longest_straight_run(r.hexes) for r in ws.rivers) / len(ws.rivers)
    assert runs[1.0] < runs[50.0]
