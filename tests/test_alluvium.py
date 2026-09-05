"""Alluvium: the sediment the erosion model moves, recorded rather than discarded.

The assertions are structural, as everywhere else here — where the silt ends up relative
to the water and the slope, never how much of it there is at a named hex.
"""

from collections import deque

import numpy as np
import pytest

from worldgen.core.config import WorldConfig
from worldgen.core.hex import TerrainClass
from worldgen.core.hex_grid import neighbors
from worldgen.core.pipeline import GeneratorPipeline
from worldgen.core.world_state import WorldState
from worldgen.stages import default_stages
from worldgen.stages.elevation import ElevationStage
from worldgen.stages.erosion import ErosionStage
from worldgen.stages.habitability import potential_food
from worldgen.stages.terrain_class import TerrainClassificationStage

WATER = (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER)


def _eroded(seed: int = 42, size: int = 48, **overrides) -> WorldState:
    cfg = WorldConfig(width=size, height=size, **overrides)
    p = GeneratorPipeline(seed, cfg)
    p.add_stage(ElevationStage)
    p.add_stage(ErosionStage)
    p.add_stage(TerrainClassificationStage)
    return p.run()


def _full(seed: int = 42, size: int = 48) -> WorldState:
    p = GeneratorPipeline(seed, WorldConfig(width=size, height=size))
    for stage in default_stages():
        p.add_stage(stage)
    return p.run()


def _land(state: WorldState) -> dict:
    return {c: h for c, h in state.hexes.items() if h.terrain_class not in WATER}


def _distance_to_river(state: WorldState, land: dict) -> dict:
    """Hops across land from each hex to the nearest river hex."""
    dist = {c: 0 for c in land if "river" in state.hexes[c].tags}
    queue = deque(dist)
    while queue:
        c = queue.popleft()
        for n in neighbors(c):
            if n in land and n not in dist:
                dist[n] = dist[c] + 1
                queue.append(n)
    return dist


@pytest.fixture(scope="module")
def eroded():
    return _eroded()


def test_alluvium_within_unit_interval(eroded):
    for h in eroded.hexes.values():
        assert 0.0 <= h.alluvium <= 1.0, f"alluvium {h.alluvium} at {h.coord} outside [0, 1]"


def test_alluvium_only_on_land(eroded):
    """Sediment under the sea is seabed, not soil, and nothing downstream can farm it."""
    for h in eroded.hexes.values():
        if h.elevation < 0.0:
            assert h.alluvium == 0.0, f"submerged hex {h.coord} carries alluvium {h.alluvium}"


def test_some_land_is_alluvial(eroded):
    """The field must actually be populated — a run of all zeros would satisfy every
    other assertion here while recording nothing at all."""
    deep = [h for h in _land(eroded).values() if h.alluvium > 0.5]
    assert deep, "no hex on the map carries appreciable alluvium"


def test_alluvium_is_not_everywhere(eroded):
    """Floodplain is a small fraction of a continent.  If most of the map is silt the
    normalisation has collapsed and the field distinguishes nothing."""
    land = _land(eroded)
    fraction = sum(1 for h in land.values() if h.alluvium > 0.5) / len(land)
    assert fraction < 0.35, f"{fraction:.0%} of land is deep alluvium; that is not a floodplain"


def test_alluvium_sits_on_gentle_ground(eroded):
    """Silt settles where water slows.  It cannot be characteristic of steep ground —
    that is where it washes off from."""
    land = list(_land(eroded).values())
    deep = [h.slope for h in land if h.alluvium > 0.5]
    rest = [h.slope for h in land if h.alluvium <= 0.5]
    assert deep and rest
    assert np.mean(deep) < np.mean(rest), (
        f"alluvial ground (mean slope {np.mean(deep):.4f}) is not gentler than the rest "
        f"({np.mean(rest):.4f})"
    )


@pytest.mark.parametrize("seed", [42, 5, 7])
def test_alluvium_falls_away_from_the_rivers(seed):
    """The profile of a floodplain: deepest at the channel, thinning to the valley wall.

    Against `HydrologyStage`'s rivers, not erosion's own channels.  Erosion computes its
    drainage separately by design (see docs/ARCHITECTURE.md), so this is the test that the
    two agree — silt laid down against one network and measured against the other.
    """
    state = _full(seed)
    land = _land(state)
    dist = _distance_to_river(state, land)

    means = []
    for d in range(4):
        vals = [h.alluvium for c, h in land.items() if dist.get(c) == d]
        assert vals, f"no land at distance {d} from a river"
        means.append(float(np.mean(vals)))

    assert means == sorted(means, reverse=True), (
        f"alluvium does not thin away from the channel: {means}"
    )
    assert means[0] > 2 * means[3], f"alluvium is barely concentrated on the rivers at all: {means}"


def test_erosion_strips_the_uplands():
    """Ground that loses more than it gains keeps no soil.

    Tested on the function rather than on a generated map, because there is no field on a
    hex that says "upland": `slope` is the mean gradient to the neighbours, so a level
    valley floor at the foot of a bluff measures as steep as the bluff does — the same
    trap that put a floodplain in the mountain band before the steepness classes were
    removed.  What the netting actually claims is this, and this is exact.
    """
    from worldgen.stages.erosion import _normalise_alluvium

    deposition = np.array([[-0.5, 0.0, 0.5]])
    out = _normalise_alluvium(
        deposition,
        meander=np.zeros((1, 3)),
        land=np.ones((1, 3), dtype=bool),
        quantile=0.98,
        floodplain_gain=0.0,
        smoothing=0.0,
    )
    assert out[0, 0] == 0.0, "a cell scoured net of deposit kept soil"
    assert out[0, 1] == 0.0
    assert out[0, 2] > 0.0, "a cell silted net of scour kept none"


def test_floodplain_gain_of_zero_leaves_only_deposition():
    """The two sources are separable, and the belts are the larger of them."""
    with_belts = _eroded()
    without = _eroded(alluvium_floodplain_gain=0.0)

    a = np.array([h.alluvium for h in _land(with_belts).values()])
    b = np.array([h.alluvium for h in _land(without).values()])
    assert b.mean() < a.mean(), "disabling the floodplain term did not reduce alluvium"
    assert b.max() > 0.0, "droplet deposition alone recorded nothing"


def test_alluvium_does_not_move_the_coastline():
    """Recording sediment must not change the ground.  Alluvium is read off the erosion
    model, not fed back into it, so the elevations have to come out bit-identical."""
    baseline = _eroded(alluvium_floodplain_gain=0.0, alluvium_smoothing=0.0)
    other = _eroded()
    for coord, h in baseline.hexes.items():
        assert h.elevation == other.hexes[coord].elevation, (
            f"elevation at {coord} changed with the alluvium settings"
        )


def test_reproducible():
    a, b = _eroded(seed=11), _eroded(seed=11)
    for coord, h in a.hexes.items():
        assert h.alluvium == b.hexes[coord].alluvium, f"alluvium differs at {coord}"


# --- what reads it ----------------------------------------------------------------


def test_alluvium_raises_food_value():
    """Silt makes good ground better.

    Read off `potential_food`, which keys on soil: rainfall enters once, in `SoilStage`,
    where it chooses the class. The alluvium the erosion model measured is a separate
    fact from the soil class — a multiplier on it, not a term inside it.
    """
    from worldgen.core.hex import LandCover, SoilQuality

    cfg = WorldConfig()
    state = WorldState.empty(seed=1, width=2, height=1)

    bare, silted = state.hexes[(0, 0)], state.hexes[(1, 0)]
    for hx in (bare, silted):
        hx.land_cover = LandCover.OPEN
        hx.soil = SoilQuality.ARABLE
    silted.alluvium = 1.0

    assert potential_food(silted, cfg) > potential_food(bare, cfg)


def test_alluvium_does_not_make_barren_ground_fertile():
    """The bonus is multiplicative on purpose: deltas carry the deepest silt on the map,
    and a flat bonus would have turned a bare-rock river mouth into farmland.

    UNUSABLE is the ground that carries the claim — desert, bare rock, above the treeline,
    drowned. `soil_value` scores it zero, and no multiplier lifts a zero.
    """
    from worldgen.core.hex import LandCover, SoilQuality

    cfg = WorldConfig()
    state = WorldState.empty(seed=1, width=1, height=1)
    hx = state.hexes[(0, 0)]
    hx.land_cover = LandCover.BARE_ROCK
    hx.soil = SoilQuality.UNUSABLE
    hx.alluvium = 1.0
    assert potential_food(hx, cfg) == 0.0, "barren ground scored as farmland once silted"


def test_alluvium_survives_a_round_trip(tmp_path):
    from worldgen.export.json_export import load, save

    state = _eroded(size=24)
    path = tmp_path / "world.json"
    save(state, str(path))
    back = load(str(path))
    for coord, h in state.hexes.items():
        assert back.hexes[coord].alluvium == h.alluvium, f"alluvium lost at {coord}"


def test_older_files_load_without_alluvium():
    """A pre-1.5 file never measured it, and 0.0 is the honest answer — unlike slope it
    cannot be recovered from the elevations."""
    state = _eroded(size=12)
    data = state.to_dict()
    data["version"] = "1.4"
    for hd in data["hexes"]:
        hd.pop("alluvium")
    back = WorldState.from_dict(data)
    assert all(h.alluvium == 0.0 for h in back.hexes.values())


def test_config_validation():
    for bad in ({"alluvium_floodplain_gain": -0.1}, {"alluvium_smoothing": -1.0}):
        name = next(iter(bad))
        with pytest.raises(ValueError, match=name):
            WorldConfig(**bad)
    for bad_quantile in (0.0, 1.5):
        with pytest.raises(ValueError, match="alluvium_quantile"):
            WorldConfig(alluvium_quantile=bad_quantile)
