from pathlib import Path

import click

from .core.config import WorldConfig
from .core.pipeline import GeneratorPipeline


@click.group()
def cli():
    """Procedural world generator."""
    pass


@cli.command()
@click.option("--seed", type=int, default=42, help="Random seed")
@click.option("--config", type=str, default=None, help="Config JSON file")
@click.option("--output-dir", type=str, default="./output", help="Output directory")
@click.option("--width", type=int, default=None, help="Map width in hexes")
@click.option("--height", type=int, default=None, help="Map height in hexes")
def generate(seed: int, config: str, output_dir: str, width: int, height: int):
    """Generate a world."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cfg = WorldConfig.from_json(config) if config else WorldConfig()

    if width:
        cfg.width = width
    if height:
        cfg.height = height

    from .stages.biomes import BiomeStage
    from .stages.city_town import CityTownStage
    from .stages.climate import ClimateStage
    from .stages.cultivation import CultivationStage, VillageCultivationStage
    from .stages.elevation import ElevationStage
    from .stages.erosion import ErosionStage
    from .stages.habitability import HabitabilityStage
    from .stages.hydrology import HydrologyStage
    from .stages.interurban_roads import InterurbanRoadStage
    from .stages.land_cover import LandCoverStage
    from .stages.terrain_class import TerrainClassificationStage
    from .stages.village_placement import VillagePlacementStage
    from .stages.village_tracks import VillageTrackStage

    click.echo(f"Generating world with seed {seed}...")
    click.echo(f"  Size: {cfg.width}×{cfg.height}")

    pipeline = GeneratorPipeline(seed, cfg)
    (
        pipeline.add_stage(ElevationStage)
        .add_stage(ErosionStage)
        .add_stage(TerrainClassificationStage)
        .add_stage(HydrologyStage)
        .add_stage(ClimateStage)
        .add_stage(BiomeStage)
        .add_stage(LandCoverStage)
        .add_stage(HabitabilityStage)
        .add_stage(CityTownStage)
        .add_stage(InterurbanRoadStage)
        .add_stage(CultivationStage)
        .add_stage(VillagePlacementStage)
        .add_stage(VillageTrackStage)
        .add_stage(VillageCultivationStage)
    )
    state = pipeline.run()

    click.echo("Writing output...")
    cfg.to_json(str(output_path / "config.json"))

    from .export.json_export import save as save_json
    from .render.debug_viewer import render as render_debug

    save_json(state, str(output_path / "world.json"))

    render_debug(state, "elevation", str(output_path / "elevation.png"))
    render_debug(state, "terrain_class", str(output_path / "terrain_class.png"))
    render_debug(state, "river_flow", str(output_path / "river_flow.png"))
    render_debug(state, "temperature", str(output_path / "temperature.png"))
    render_debug(state, "moisture", str(output_path / "moisture.png"))
    render_debug(state, "biome", str(output_path / "biome.png"))
    render_debug(state, "habitability", str(output_path / "habitability.png"))
    render_debug(state, "settlements", str(output_path / "settlements.png"))
    render_debug(state, "roads", str(output_path / "roads.png"))
    render_debug(state, "land_cover", str(output_path / "land_cover.png"))
    render_debug(state, "cultivation", str(output_path / "cultivation.png"))

    click.echo("✓ Done")


@cli.command(name="render")
@click.option("--input", type=str, required=True, help="Input world.json file")
@click.option(
    "--attribute",
    type=str,
    default="terrain_class",
    help="Attribute to render: elevation, terrain_class, river_flow, temperature, moisture, "
    "biome, habitability, settlements, roads, land_cover, cultivation",
)
@click.option("--output", type=str, required=True, help="Output PNG file")
def render_map(input: str, attribute: str, output: str):
    """Render a saved world from world.json."""
    from .export.json_export import load as load_json
    from .render.debug_viewer import render as render_debug

    click.echo(f"Loading {input}...")
    state = load_json(input)
    click.echo(f"Rendering {attribute}...")
    render_debug(state, attribute, output)
    click.echo(f"✓ Saved to {output}")


@cli.command()
def presets():
    """List available presets."""
    presets_dir = Path(__file__).parent.parent / "presets"
    if presets_dir.exists():
        for preset in sorted(presets_dir.glob("*.json")):
            click.echo(f"  {preset.stem}")
    else:
        click.echo("No presets found")


if __name__ == "__main__":
    cli()
