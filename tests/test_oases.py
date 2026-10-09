"""Oases: groundwater surfacing in the desert, sited by SoilStage on its own generator."""

import pytest

from tests.worlds import build_world
from worldgen.core.config import WorldConfig
from worldgen.core.hex import SOIL_RANK, Biome
from worldgen.core.hex_grid import distance
from worldgen.stages.oases import OASIS_TAG


def _arid(**over):
    return build_world(
        seed=42,
        width=64,
        height=64,
        model="organic",
        until="SoilStage",
        **dict(regional_climate="arid", **over),
    )


@pytest.fixture(scope="module")
def arid():
    return _arid()


def _springs(state):
    return sorted(c for c, h in state.hexes.items() if OASIS_TAG in h.tags)


def test_a_desert_has_a_few_oases(arid):
    cfg = WorldConfig(**arid.metadata["config"])
    desert = sum(1 for h in arid.hexes.values() if h.biome is Biome.DESERT)
    springs = _springs(arid)
    assert springs, "an arid map with no oasis on it"
    assert len(springs) <= round(cfg.oasis_per_1000_km2 * desert / 1000)


def test_same_seed_same_oases(arid):
    again = _arid()
    assert _springs(again) == _springs(arid)
    assert {c: h.groundwater_mm for c, h in again.hexes.items()} == {
        c: h.groundwater_mm for c, h in arid.hexes.items()
    }


def test_no_oasis_outside_the_desert(arid):
    for c, h in arid.hexes.items():
        if h.groundwater_mm > 0 or OASIS_TAG in h.tags:
            assert h.biome is Biome.DESERT, f"{c} is watered from below but is {h.biome}"
    temperate = build_world(seed=42, width=64, height=64, model="organic", until="SoilStage")
    assert not _springs(temperate)
    assert all(h.groundwater_mm == 0 for h in temperate.hexes.values())


def test_oases_stand_apart(arid):
    """Two oases' watered ground never touches: each is its own spring."""
    cfg = WorldConfig(**arid.metadata["config"])
    springs = _springs(arid)
    for i, a in enumerate(springs):
        for b in springs[i + 1 :]:
            assert distance(a, b) >= 2 * cfg.oasis_radius + 2


def test_an_oasis_farms_better_than_the_same_ground_without_it(arid):
    """The only thing an oasis changes is the ground it waters, and only for the better."""
    dry = _arid(oasis_per_1000_km2=0.0)
    assert not _springs(dry)
    better = 0
    for c, h in arid.hexes.items():
        before, after = SOIL_RANK[dry.hexes[c].soil], SOIL_RANK[h.soil]
        if h.groundwater_mm > 0:
            assert after >= before, f"{c} farms worse for its oasis"
            better += after > before
        else:
            assert after == before, f"{c} changed soil without any groundwater"
    assert better, "no oasis made its ground any better"


def test_oasis_settings_are_validated():
    for name in ("oasis_per_1000_km2", "oasis_groundwater_mm", "oasis_radius"):
        with pytest.raises(ValueError, match=name):
            WorldConfig(**{name: -1})
