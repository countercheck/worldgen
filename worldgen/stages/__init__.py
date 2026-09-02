"""Pipeline stage registry.

The stage list lived inline in `cli.py` and was then copied, stage for stage, into four
test modules.  Five copies of one ordering is four too many: a stage added to the CLI but
not to a test fixture makes the tests assert against a pipeline nobody runs.  This is the
single definition both use.

Imports are deliberately inside `default_stages` rather than at module scope.  `cli.py`
imported every stage lazily so that `worldgen --help` does not pay for numpy and the whole
stage tree; keeping the imports in the function body preserves that.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.pipeline import GeneratorStage

MODELS = ("classic",)


def default_stages(model: str = "classic") -> tuple[type["GeneratorStage"], ...]:
    """The production pipeline, in run order.

    *model* selects which settlement and road model to use.  Only ``classic`` — the
    habitability-ranked placement with a single gravity road pass — exists today; the
    haulage-based model is added alongside it so the two can be compared in the debug
    viewer before either is retired.
    """
    if model not in MODELS:
        raise ValueError(f"Unknown pipeline model {model!r}. Supported: {', '.join(MODELS)}")

    from .biomes import BiomeStage
    from .city_town import CityTownStage
    from .climate import ClimateStage
    from .cultivation import CultivationStage, VillageCultivationStage
    from .elevation import ElevationStage
    from .erosion import ErosionStage
    from .habitability import HabitabilityStage
    from .hydrology import HydrologyStage
    from .interurban_roads import InterurbanRoadStage
    from .land_cover import LandCoverStage
    from .terrain_class import TerrainClassificationStage
    from .village_placement import VillagePlacementStage
    from .village_tracks import VillageTrackStage
    from .water_bodies import WaterBodiesStage

    stages = (
        ElevationStage,
        ErosionStage,
        TerrainClassificationStage,
        WaterBodiesStage,
        HydrologyStage,
        ClimateStage,
        BiomeStage,
        LandCoverStage,
        HabitabilityStage,
        # Villages need roads and cultivation to exist before VillagePlacementStage will
        # site them, so this ordering is load-bearing, not incidental.
        CityTownStage,
        InterurbanRoadStage,
        CultivationStage,
        VillagePlacementStage,
        VillageTrackStage,
        VillageCultivationStage,
    )
    return stages


def stages_for(config, model: str = "classic") -> tuple[type["GeneratorStage"], ...]:
    """`default_stages(model)`, resolved against a config.

    Today that means one substitution: a world with `heightmap_path` set reads its terrain
    from that image instead of generating it, so `ImageElevationStage` takes
    `ElevationStage`'s place.

    The swap is positional and the tuple keeps its length, which is what makes an imported
    world comparable with a generated one.  `GeneratorPipeline.run` draws a child seed per
    stage from the parent stream before constructing it, so as long as the count does not
    change, hydrology, climate and settlement all see exactly the seed they would have.

    Kept separate from `default_stages` rather than folded into it: that function is the
    declarative statement of what the pipeline *is*, and both the ordering test and
    `build_pipeline`'s `until=` look stages up in it by name.
    """
    stages = default_stages(model)
    if not getattr(config, "heightmap_path", None):
        return stages

    from .elevation import ElevationStage
    from .image_elevation import ImageElevationStage

    return tuple(ImageElevationStage if s is ElevationStage else s for s in stages)
