"""World builders shared by the test suite.

Four test modules each carried a private `_build_pipeline` that restated the production
stage list, stage for stage.  Five copies of one ordering meant a stage added to the CLI
but missed in a fixture would leave those tests asserting about a pipeline nobody runs.
Everything here builds on `worldgen.stages.stages_for`, which is the same list the CLI
uses — `default_stages` resolved against the config, so a fixture that sets a heightmap
gets the image importer in place of the noise stage exactly as a real run would.

Worlds are memoised for the session.  Several modules want the same 64x64 seed-42 world,
and generating it once rather than once per module is what keeps the suite fast enough to
run on every commit.
"""

from worldgen.core.config import WorldConfig
from worldgen.core.pipeline import GeneratorPipeline
from worldgen.stages import stages_for


def build_pipeline(
    seed: int = 42,
    width: int = 64,
    height: int = 64,
    model: str = "classic",
    until: str | None = None,
    **cfg_overrides,
) -> GeneratorPipeline:
    """A pipeline over the production stage list.

    *until* stops after the stage of that class name, for tests that only care about an
    early attribute and should not pay for settlement and road generation.  Overrides win
    the production defaults, so a caller can vary anything it needs to.

    The two elevation stages are interchangeable as far as *until* is concerned: a caller
    that sets `heightmap_path` gets `ImageElevationStage` in the slot, and asking to stop
    after "ElevationStage" means the same thing either way.  Without that, adding a
    heightmap to an existing `until="ElevationStage"` call would fail on a name that is
    an implementation detail of the swap.
    """
    cfg = WorldConfig(width=width, height=height, **cfg_overrides)

    stages = stages_for(cfg, model)
    if until is not None:
        names = [s.__name__ for s in stages]
        if until not in names:
            aliases = {"ElevationStage", "ImageElevationStage"}
            swapped = aliases - {until} if until in aliases else set()
            match = next((n for n in names if n in swapped), None)
            if match is None:
                raise ValueError(f"No stage named {until!r} in the {model} pipeline: {names}")
            until = match
        stages = stages[: names.index(until) + 1]

    pipeline = GeneratorPipeline(seed, cfg)
    for stage in stages:
        pipeline.add_stage(stage)
    return pipeline


_WORLD_CACHE: dict = {}


def build_world(
    seed: int = 42,
    width: int = 64,
    height: int = 64,
    model: str = "classic",
    until: str | None = None,
    **cfg_overrides,
):
    """Memoised `build_pipeline(...).run()`.

    The cache key is the full argument set, so two callers asking for different configs
    get different worlds — but the common 64x64 seed-42 case is generated once.  Callers
    must not mutate the result; anything that needs to mutate should build its own.
    """

    def _freeze(value):
        if isinstance(value, list):
            return tuple(_freeze(v) for v in value)
        if isinstance(value, dict):
            return tuple((k, _freeze(v)) for k, v in sorted(value.items()))
        return value

    frozen_overrides = tuple(sorted((k, _freeze(v)) for k, v in cfg_overrides.items()))
    key = (seed, width, height, model, until, frozen_overrides)
    if key not in _WORLD_CACHE:
        _WORLD_CACHE[key] = build_pipeline(
            seed=seed,
            width=width,
            height=height,
            model=model,
            until=until,
            **cfg_overrides,
        ).run()
    return _WORLD_CACHE[key]


def lay_road(ws, path, tier):
    """Write a hex path into the world as a route of *tier*.

    Mirrors what `InterurbanRoadStage` does on the way out, which a fixture has to or it
    tests a shape the generator never produces: one tier per edge, the higher tier winning
    where two routes overlap, and an edge with a foot in the water filed under `sea_edges`
    rather than `road_edges`, because it is a sea leg and not a road.
    """
    from worldgen.core.hex import TerrainClass
    from worldgen.core.world_state import ROAD_TIER_RANK, road_edge_key

    water = (TerrainClass.OCEAN, TerrainClass.LAKE)
    for a, b in zip(path, path[1:], strict=False):
        a, b = tuple(a), tuple(b)
        key = road_edge_key(a, b)
        wet = any(ws.hexes[c].terrain_class in water for c in (a, b))
        book = ws.sea_edges if wet else ws.road_edges
        if ROAD_TIER_RANK[tier] > ROAD_TIER_RANK.get(book.get(key), -1):
            book[key] = tier
        ws.hexes[a].road_connections.add(b)
        ws.hexes[b].road_connections.add(a)
    return ws
