"""The map's vocabulary is derived, not stored.

Mountain, hill and flat are bands on `Hex.slope` produced for drawing and reading. The
generator does not carry them, which is the point: as stored classes they were read by six
stages in place of the terrain, and a level floodplain beside a bluff came out a mountain.
"""

import pytest

from worldgen.core.hex import Hex, TerrainClass, TerrainLabel, terrain_label, terrain_labels
from worldgen.core.world_state import WorldState

HILL, MOUNTAIN = 0.02, 0.04


def _hex(slope, terrain_class=TerrainClass.LAND):
    return Hex(coord=(0, 0), slope=slope, terrain_class=terrain_class)


@pytest.mark.parametrize(
    "slope,expected",
    [
        (0.0, TerrainLabel.FLAT),
        (0.019, TerrainLabel.FLAT),
        (0.02, TerrainLabel.HILL),
        (0.04, TerrainLabel.HILL),
        (0.041, TerrainLabel.MOUNTAIN),
        (0.5, TerrainLabel.MOUNTAIN),
    ],
)
def test_slope_bands_into_the_expected_word(slope, expected):
    assert terrain_label(_hex(slope), HILL, MOUNTAIN) is expected


@pytest.mark.parametrize(
    "terrain_class,expected",
    [
        (TerrainClass.OCEAN, TerrainLabel.OCEAN),
        (TerrainClass.LAKE, TerrainLabel.LAKE),
        (TerrainClass.COAST, TerrainLabel.COAST),
    ],
)
def test_water_and_shore_outrank_steepness(terrain_class, expected):
    # A shore is a shore however steep, which is what the old classes said by ordering.
    assert terrain_label(_hex(0.9, terrain_class), HILL, MOUNTAIN) is expected


def test_the_generator_no_longer_carries_a_steepness_class():
    assert not {"flat", "hill", "mountain"} & {m.value for m in TerrainClass}


def test_labels_use_the_thresholds_the_world_was_generated_with():
    ws = WorldState.empty(seed=1, width=2, height=1)
    for h in ws.hexes.values():
        h.slope = 0.03
    ws.metadata["config"] = {"terrain_hill_gradient": 0.05, "terrain_mountain_gradient": 0.10}
    assert set(terrain_labels(ws).values()) == {TerrainLabel.FLAT}

    ws.metadata["config"] = {"terrain_hill_gradient": 0.01, "terrain_mountain_gradient": 0.02}
    assert set(terrain_labels(ws).values()) == {TerrainLabel.MOUNTAIN}
