"""`tests/worlds.fork_world` builds the world a full build makes, or refuses to.

The slow knob-turning tests compare worlds built by forking a shared head, so a fork that
drifted from a full build would have them asserting about worlds the generator never
makes — and passing. These hold it to a full build on a small world, field for field, and
check that the guard on the head actually bites.
"""

import json

import pytest

from tests.worlds import build_pipeline, fork_world


def _snapshot(state):
    """Everything the JSON carries, plus the journeys it leaves out."""
    return json.dumps(state.to_dict(), sort_keys=True), sorted(state.journeys.items())


_SMALL = {"seed": 7, "width": 32, "height": 32, "model": "organic"}


@pytest.mark.parametrize(
    ("at", "varied"),
    [
        ("ChokepointStage", {"chokepoint_min_road_tier": "track"}),
        ("LandUseStage", {"clearing_margin": 0.8}),
        ("CityPromotionStage", {"city_min_draw": 14.0}),
    ],
)
def test_a_fork_is_the_full_build(at, varied):
    full = build_pipeline(**_SMALL, **varied).run()
    assert _snapshot(fork_world(at, varied, **_SMALL)) == _snapshot(full)


def test_a_fork_after_a_stage_that_reads_the_knob_is_refused():
    """`clearing_margin` is `LandUseStage`'s: forking after it would share a head built
    with the wrong margin, so the head build fails instead."""
    with pytest.raises(AssertionError, match="clearing_margin"):
        fork_world("CityPromotionStage", {"clearing_margin": 0.8}, **_SMALL)
