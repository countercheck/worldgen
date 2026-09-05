"""The map's vocabulary is derived, not stored.

Flat, rolling, steep and escarpment are bands on `Hex.slope` produced for drawing and
reading. The generator does not carry them, which is the point: as stored classes they
were read by six stages in place of the terrain, and a level floodplain beside a bluff
came out an escarpment.
"""

import pytest

from worldgen.core.hex import Hex, TerrainClass, TerrainLabel, terrain_label, terrain_labels
from worldgen.core.world_state import WorldState

# Metres of rise per kilometre, as the config quotes them.
ROLLING, STEEP, ESCARPMENT = 30.0, 100.0, 250.0
BANDS = (ROLLING, STEEP, ESCARPMENT)


def _hex(slope, terrain_class=TerrainClass.LAND):
    return Hex(coord=(0, 0), slope=slope, terrain_class=terrain_class)


@pytest.mark.parametrize(
    "slope,expected",
    [
        (0.0, TerrainLabel.FLAT),
        (29.9, TerrainLabel.FLAT),
        (30.0, TerrainLabel.ROLLING),
        (99.9, TerrainLabel.ROLLING),
        (100.0, TerrainLabel.STEEP),
        (249.9, TerrainLabel.STEEP),
        (250.0, TerrainLabel.ESCARPMENT),
        (900.0, TerrainLabel.ESCARPMENT),
    ],
)
def test_slope_bands_into_the_expected_word(slope, expected):
    assert terrain_label(_hex(slope), *BANDS) is expected


@pytest.mark.parametrize(
    "terrain_class,expected",
    [
        (TerrainClass.OPEN_WATER, TerrainLabel.OCEAN),
        (TerrainClass.INLAND_WATER, TerrainLabel.LAKE),
        (TerrainClass.COAST, TerrainLabel.COAST),
    ],
)
def test_water_and_shore_outrank_steepness(terrain_class, expected):
    # A shore is a shore however steep, which is what the old classes said by ordering.
    assert terrain_label(_hex(900.0, terrain_class), *BANDS) is expected


def test_the_generator_no_longer_carries_a_steepness_class():
    words = {"flat", "rolling", "steep", "escarpment", "hill", "mountain"}
    assert not words & {m.value for m in TerrainClass}


def test_labels_use_the_thresholds_the_world_was_generated_with():
    ws = WorldState.empty(seed=1, width=2, height=1)
    for h in ws.hexes.values():
        h.slope = 50.0

    ws.metadata["config"] = {
        "terrain_rolling_gradient_m": 80.0,
        "terrain_steep_gradient_m": 200.0,
        "terrain_escarpment_gradient_m": 400.0,
    }
    assert set(terrain_labels(ws).values()) == {TerrainLabel.FLAT}

    ws.metadata["config"] = {
        "terrain_rolling_gradient_m": 10.0,
        "terrain_steep_gradient_m": 20.0,
        "terrain_escarpment_gradient_m": 40.0,
    }
    assert set(terrain_labels(ws).values()) == {TerrainLabel.ESCARPMENT}
