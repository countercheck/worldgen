"""The pipeline itself: its composition, its ordering, and its reproducibility.

This file was empty. The stage list was asserted nowhere, so a stage could be dropped from
the CLI, or reordered into a position where it read an attribute its producer had not yet
written, without a single test noticing.
"""

import pytest

from tests.worlds import build_pipeline
from worldgen.core.world_state import WorldState
from worldgen.stages import MODELS, default_stages


def test_every_advertised_model_builds():
    for model in MODELS:
        assert default_stages(model), f"{model} produced an empty stage list"


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="Unknown pipeline model"):
        default_stages("nonexistent")


def test_cli_and_tests_share_one_stage_list():
    """The regression this file exists for.

    Five copies of the stage list used to be maintained by hand — one in the CLI and one
    in each of four test modules. Both now read the shared registry, so this asserts the
    CLI has not quietly grown a private list again.

    Either entry point counts: `stages_for` is `default_stages` resolved against a config,
    and both bottom out in the same tuple.
    """
    import inspect

    from worldgen import cli

    # `generate` is a click Command; the function it wraps is its callback.
    source = inspect.getsource(cli.generate.callback)
    assert "stages_for(" in source or "default_stages(" in source, (
        "cli.generate no longer builds from the shared stage registry"
    )
    assert ".add_stage(ElevationStage)" not in source, "cli.generate has an inline stage list"


def test_until_treats_the_two_elevation_stages_as_one_slot(tmp_path):
    """`build_pipeline`'s docstring advertises varying anything alongside `until`.

    Adding `heightmap_path` swaps the class in that slot, so an existing
    `until="ElevationStage"` call would otherwise start failing on a name that is an
    implementation detail of the swap.
    """
    import numpy as np
    from PIL import Image

    path = tmp_path / "hm.png"
    Image.fromarray(np.full((32, 32), 200, np.uint8)).save(path)

    imported = build_pipeline(until="ElevationStage", heightmap_path=str(path), width=16, height=16)
    assert [cls.__name__ for cls, _ in imported.stages] == ["ImageElevationStage"]

    generated = build_pipeline(until="ElevationStage", width=16, height=16)
    assert [cls.__name__ for cls, _ in generated.stages] == ["ElevationStage"]

    with pytest.raises(ValueError, match="No stage named"):
        build_pipeline(until="NoSuchStage", width=16, height=16)


def test_stage_order_is_load_bearing():
    """Each of these consumes what the one before it produces."""
    names = [s.__name__ for s in default_stages("classic")]
    order = [
        "ElevationStage",  # terrain classification needs elevation
        "TerrainClassificationStage",  # water bodies need a terrain class
        "WaterBodiesStage",  # hydrology needs to know the sea
        "HydrologyStage",  # climate's orographic pass needs rivers
        "ClimateStage",  # biomes need temperature and moisture
        "BiomeStage",  # land cover is derived from biome
        "LandCoverStage",  # habitability scores land cover
        "HabitabilityStage",  # placement ranks on habitability
        "CityTownStage",  # roads join cities and towns
        "InterurbanRoadStage",  # villages are sited along road corridors
        "VillagePlacementStage",  # tracks connect the villages
        "VillageTrackStage",
    ]
    positions = [names.index(n) for n in order]
    assert positions == sorted(positions), f"Stage order violated: {names}"


def test_pipeline_runs_and_populates_a_world():
    # 48x48 rather than 32x32: `continent_shelf_hexes` is capped at a quarter of the
    # shorter side, so a 32-hex map is mostly coastal shelf and cannot muster the
    # `settlement_min_reachable` hexes of connected interior a settlement needs. It is
    # not a landscape, and asserting settlements on one asserts an accident.
    state = build_pipeline(width=48, height=48).run()
    assert isinstance(state, WorldState)
    assert len(state.hexes) == 48 * 48
    assert state.settlements, "a 48x48 world produced no settlements"
    assert state.roads, "a 48x48 world produced no roads"


def test_until_truncates_the_run():
    """`until` must stop the pipeline, not merely run it and discard the tail."""
    state = build_pipeline(width=32, height=32, until="HabitabilityStage").run()
    assert not state.settlements, "stages after HabitabilityStage still ran"
    assert any(h.habitability_city > 0 for h in state.hexes.values())


def test_until_rejects_a_stage_not_in_the_pipeline():
    with pytest.raises(ValueError, match="No stage named"):
        build_pipeline(until="NoSuchStage")


def test_same_seed_same_world():
    """The project's central promise: one integer reproduces the whole map."""
    kwargs = {"width": 32, "height": 32}
    a = build_pipeline(seed=7, **kwargs).run()
    b = build_pipeline(seed=7, **kwargs).run()

    assert [h.elevation for h in a.hexes.values()] == [h.elevation for h in b.hexes.values()]
    assert sorted(s.coord for s in a.settlements) == sorted(s.coord for s in b.settlements)
    assert sorted(tuple(r.path) for r in a.roads) == sorted(tuple(r.path) for r in b.roads)


def test_different_seed_different_world():
    kwargs = {"width": 32, "height": 32}
    a = build_pipeline(seed=7, **kwargs).run()
    b = build_pipeline(seed=8, **kwargs).run()
    assert [h.elevation for h in a.hexes.values()] != [h.elevation for h in b.hexes.values()]


def test_seed_is_recorded_in_metadata():
    state = build_pipeline(seed=123, width=32, height=32).run()
    assert state.seed == 123
    assert state.metadata["seed"] == 123
