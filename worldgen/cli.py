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

    if config:
        if config.lower().endswith((".yaml", ".yml")):
            cfg = WorldConfig.from_yaml(config)
        else:
            cfg = WorldConfig.from_json(config)
    else:
        cfg = WorldConfig()

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
    from .stages.water_bodies import WaterBodiesStage

    click.echo(f"Generating world with seed {seed}...")
    click.echo(f"  Size: {cfg.width}×{cfg.height}")

    pipeline = GeneratorPipeline(seed, cfg)
    (
        pipeline.add_stage(ElevationStage)
        .add_stage(ErosionStage)
        .add_stage(TerrainClassificationStage)
        .add_stage(WaterBodiesStage)
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


_ATTRIBUTES = [
    "elevation",
    "terrain_class",
    "river_flow",
    "temperature",
    "moisture",
    "biome",
    "habitability",
    "settlements",
    "roads",
    "land_cover",
    "cultivation",
]


@cli.command(name="render")
@click.option("--input", "input_path", type=str, required=True, help="Input world.json file")
@click.option(
    "--attribute",
    type=click.Choice(_ATTRIBUTES, case_sensitive=False),
    default="terrain_class",
    show_default=True,
    help="Attribute to render.",
)
@click.option("--output", type=str, required=True, help="Output PNG file")
def render_map(input_path: str, attribute: str, output: str):
    """Render a saved world from world.json."""
    from .export.json_export import load as load_json
    from .render.debug_viewer import render as render_debug

    click.echo(f"Loading {input_path}...")
    try:
        state = load_json(input_path)
        click.echo(f"Rendering {attribute}...")
        render_debug(state, attribute, output)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"✓ Saved to {output}")


_STYLES = ["atlas", "topographic", "wargame"]
_COLOR_MODES = ["biome", "terrain", "land_cover", "elevation"]
_DEFAULT_LAYERS = {"terrain", "rivers", "roads", "settlements", "labels", "grid"}
_ALLOWED_LAYERS = _DEFAULT_LAYERS | {"contours"}


def _parse_layers_value(value, source: str) -> set[str]:
    if isinstance(value, str):
        parsed = [layer.strip() for layer in value.split(",") if layer.strip()]
    elif isinstance(value, (list, tuple, set)):
        parsed = []
        for layer in value:
            if not isinstance(layer, str):
                raise click.ClickException(
                    f"{source} must be a list of strings or a comma-separated string."
                )
            stripped = layer.strip()
            if stripped:
                parsed.append(stripped)
    else:
        raise click.ClickException(
            f"{source} must be a list of strings or a comma-separated string."
        )

    unknown = set(parsed) - _ALLOWED_LAYERS
    if unknown:
        allowed = ", ".join(sorted(_ALLOWED_LAYERS))
        raise click.ClickException(
            f"Unknown layer(s): {', '.join(sorted(unknown))}. Allowed: {allowed}"
        )
    return set(parsed)


def _load_export_section(config_path: str) -> dict:
    if config_path.lower().endswith((".yaml", ".yml")):
        import yaml

        with open(config_path) as f:
            raw = yaml.safe_load(f)
    else:
        import json as _json

        with open(config_path) as f:
            raw = _json.load(f)

    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise click.ClickException("Config root must be a mapping/object.")

    export_section = raw.get("export", {})
    if export_section is None:
        return {}
    if not isinstance(export_section, dict):
        raise click.ClickException("'export' section must be a mapping/object.")
    return export_section


@cli.command(name="export")
@click.option("--input", "input_path", type=str, required=True, help="Input world.json file")
@click.option("--output", type=str, required=True, help="Output SVG file")
@click.option("--config", "config_path", type=str, default=None, help="Config YAML/JSON file")
@click.option(
    "--style",
    type=click.Choice(_STYLES, case_sensitive=False),
    default=None,
    help="Visual style preset (overrides config file). Choices: atlas, topographic, wargame.",
)
@click.option(
    "--color-mode",
    type=click.Choice(_COLOR_MODES, case_sensitive=False),
    default=None,
    help="Hex fill color source (overrides config file). Choices: biome, terrain, land_cover, elevation.",
)
@click.option(
    "--layers",
    default=None,
    help="Comma-separated layers to include (overrides config file). "
    "Choices: terrain,rivers,roads,settlements,labels,grid,contours",
)
@click.option(
    "--hex-size", type=float, default=None, help="Hex size in pixels (overrides config file)."
)
@click.option(
    "--padding", type=int, default=None, help="Border padding in pixels (overrides config file)."
)
def export_svg(
    input_path: str,
    output: str,
    config_path: str | None,
    style: str | None,
    color_mode: str | None,
    layers: str | None,
    hex_size: float | None,
    padding: int | None,
) -> None:
    """Export a saved world as an SVG hex map."""
    from .export.json_export import load as load_json
    from .export.svg_export import SVGConfig
    from .export.svg_export import save as save_svg

    # Start with SVGConfig defaults, then override with config file, then CLI flags
    svg_kwargs: dict = {
        "style": "atlas",
        "color_mode": "biome",
        "hex_size": 12.0,
        "padding": 20,
        "layers": set(_DEFAULT_LAYERS),
    }

    if config_path:
        export_section = _load_export_section(config_path)
        for key in (
            "style",
            "color_mode",
            "hex_size",
            "padding",
            "contour_elevation_scale_m",
            "contour_interval_m",
            "contour_max_crossings",
            "contour_max_stroke",
        ):
            if key in export_section:
                svg_kwargs[key] = export_section[key]
        if "layers" in export_section:
            svg_kwargs["layers"] = _parse_layers_value(
                export_section["layers"], "export.layers in config"
            )

    # CLI flags override config file (only when explicitly provided)
    if style is not None:
        svg_kwargs["style"] = style
    if color_mode is not None:
        svg_kwargs["color_mode"] = color_mode
    if hex_size is not None:
        svg_kwargs["hex_size"] = hex_size
    if padding is not None:
        svg_kwargs["padding"] = padding
    if layers is not None:
        svg_kwargs["layers"] = _parse_layers_value(layers, "--layers")

    unknown_cfg = svg_kwargs["layers"] - _ALLOWED_LAYERS
    if unknown_cfg:
        allowed = ", ".join(sorted(_ALLOWED_LAYERS))
        raise click.ClickException(
            f"Unknown layer(s) in config: {', '.join(sorted(unknown_cfg))}. Allowed: {allowed}"
        )

    click.echo(f"Loading {input_path}...")
    try:
        state = load_json(input_path)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    cfg = SVGConfig(**svg_kwargs)
    save_svg(state, output, cfg)
    click.echo(f"✓ Saved to {output}")


@cli.command(name="init-config")
@click.option(
    "--output",
    type=str,
    default="worldgen.yaml",
    show_default=True,
    help="Path to write the default config file.",
)
@click.option("--force", is_flag=True, default=False, help="Overwrite existing file.")
def init_config(output: str, force: bool) -> None:
    """Write the default annotated worldgen.yaml to disk."""
    out = Path(output)
    if out.exists() and not force:
        raise click.ClickException(f"{output} already exists. Use --force to overwrite.")
    out.parent.mkdir(parents=True, exist_ok=True)
    template = Path(__file__).parent / "default_config.yaml"
    import shutil

    shutil.copyfile(template, out)
    click.echo(f"✓ Written to {output}")


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
