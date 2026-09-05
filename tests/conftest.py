"""Shared fixtures. The builders themselves live in `tests/worlds.py`.

Worlds are memoised for the session: several modules want the same 64x64 seed-42 world,
and generating it once rather than once per module is what keeps the suite fast enough to
run on every commit.
"""

import pytest

from tests.worlds import build_world

# --- the standard worlds -----------------------------------------------------
# Named for what they are for, not for their dimensions, so a size can change in one place.


@pytest.fixture(scope="session")
def settle_state():
    """Full pipeline at 64x64 — settlements, roads and cultivation all present."""
    return build_world(target_city_count=4, target_town_count=12)


@pytest.fixture(scope="session")
def road_state():
    """Full pipeline at 64x64 with traveller counts turned down for speed."""
    return build_world(
        target_city_count=4,
        target_town_count=10,
    )


@pytest.fixture(scope="session")
def cult_state():
    """Full pipeline at 64x64 with small cultivation radii, so farmland does not blanket."""
    return build_world(
        target_city_count=3,
        target_town_count=8,
        cultivation_city_radius=6,
        cultivation_town_radius=3,
        cultivation_village_radius=2,
    )


@pytest.fixture(scope="session")
def hab_state():
    """Stops after HabitabilityStage — scoring tests need nothing downstream of it."""
    return build_world(width=48, height=48, until="HabitabilityStage")


@pytest.fixture(scope="session")
def small_state():
    """32x32, everything present. Cheap enough for the rendering tests to use freely."""
    return build_world(
        width=32,
        height=32,
        target_city_count=2,
        target_town_count=4,
    )
