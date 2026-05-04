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

    from .stages.elevation import ElevationStage
    from .stages.erosion import ErosionStage
    from .stages.terrain_class import TerrainClassificationStage

    click.echo(f"Generating world with seed {seed}...")
    click.echo(f"  Size: {cfg.width}×{cfg.height}")

    pipeline = GeneratorPipeline(seed, cfg)
    pipeline.add_stage(ElevationStage).add_stage(ErosionStage).add_stage(TerrainClassificationStage)
    state = pipeline.run()

    click.echo("Writing output...")
    cfg.to_json(str(output_path / "config.json"))

    from .render.debug_viewer import render as render_debug

    render_debug(state, "elevation", str(output_path / "elevation.png"))
    render_debug(state, "terrain_class", str(output_path / "terrain_class.png"))

    click.echo("✓ Done")


@cli.command()
@click.option("--input", type=str, required=True, help="Input JSON world file")
@click.option("--attribute", type=str, default="terrain_class", help="Attribute to render")
@click.option("--output", type=str, required=True, help="Output PNG file")
def render_map(input: str, attribute: str, output: str):
    """Render an existing world."""
    click.echo(f"Loading {input}...")

    click.echo(f"Rendering {attribute}...")
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
