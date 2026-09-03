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

MODELS = ("classic", "organic")


def default_stages(model: str = "classic") -> tuple[type["GeneratorStage"], ...]:
    """The production pipeline, in run order.

    *model* selects which settlement and road model to use.

    ``classic`` ranks hexes on habitability and places a configured number of cities and
    towns at a fixed minimum separation, then sprinkles villages.

    ``organic`` derives the hierarchy from pre-industrial haulage economics: markets go
    where the most surplus can reach them inside a day's return, and their number follows
    the land rather than a target.  It models the countryside as a productive surface
    rather than a list of hamlets, so it runs no village stages; it is still being built
    stage by stage, so the classic road stages run over the new settlements for now.  The
    two live side by side so they can be compared in the debug viewer before either is
    retired.
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
    from .soil import SoilStage
    from .terrain_class import TerrainClassificationStage
    from .village_placement import VillagePlacementStage
    from .village_tracks import VillageTrackStage
    from .water_bodies import WaterBodiesStage

    # Terrain first, and in strict dependency order: each of these reads what the one
    # before it wrote.
    physical = (
        ElevationStage,
        ErosionStage,
        TerrainClassificationStage,
        WaterBodiesStage,
        HydrologyStage,
        ClimateStage,
        BiomeStage,
        # Soil before cover, because cover depends on it: good ground carries wildwood
        # until somebody clears it, so what grows on a hex follows from what the hex is.
        # Soil itself needs the gradient, the drainage, the rainfall and the cold biomes,
        # which is every stage above.
        SoilStage,
        LandCoverStage,
        HabitabilityStage,
    )

    if model == "organic":
        from .chokepoints import ChokepointStage
        from .cities import CityPromotionStage
        from .crossings import CrossingStage
        from .land_use import LandUseStage
        from .markets import MarketStage

        return physical + (
            # Crossings before markets, deliberately: a bridging point is the cheapest
            # ground in a district to reach from both banks, so it should be a reason a
            # market grows there rather than something noticed afterwards.
            CrossingStage,
            MarketStage,
            # Land use immediately after siting, and it founds the markets. A market is
            # worth what its countryside actually sends, and until the countryside has
            # been put to use that is not known — so `MarketStage` plants and allocates,
            # and this clears, sizes and counts who lives on the land.
            LandUseStage,
            # Cities before roads: promotion changes populations, and population is what
            # decides how many travellers a place sends.
            CityPromotionStage,
            # Interim: the classic road stages still run over the new settlements, so
            # there is something to look at in the viewer. Markets are all TOWN tier for
            # now, which is what InterurbanRoadStage expects. Trade roads replace this
            # in turn.
            #
            # The three classic village stages are deliberately absent. They exist to serve
            # a village tier built the other way round: every hex clearing a habitability
            # bar becomes a hamlet, which buried 74 markets under 835 settlements on a
            # 128x128 temperate map, so what the viewer showed was mostly not the haulage
            # model. `ChokepointStage` below is the organic village tier — gated on holding
            # something rather than sprinkled.
            #
            # The win is legibility, not speed: the three stages cost 0.8 s of a 15.2 s
            # pipeline. InterurbanRoadStage is 12.3 s of what remains.
            InterurbanRoadStage,
            # Chokepoints after roads, and that ordering is the whole idea. A chokepoint
            # is not a good site that happens to have traffic; it is a bad site that has
            # traffic anyway, and only the built network can say which crossings carry
            # any. They sit on the road by construction, so nothing has to be recut.
            ChokepointStage,
        )

    return physical + (
        # Villages need roads and cultivation to exist before VillagePlacementStage will
        # site them, so this ordering is load-bearing, not incidental.
        CityTownStage,
        InterurbanRoadStage,
        CultivationStage,
        VillagePlacementStage,
        VillageTrackStage,
        VillageCultivationStage,
    )


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
