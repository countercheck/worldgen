from pathlib import Path

import click

from .core.config import HEIGHTMAP_MODES, WorldConfig
from .core.hex_grid import GRID_LAYOUTS
from .core.pipeline import GeneratorPipeline
from .stages import MODELS, stages_for


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
@click.option(
    "--grid-layout",
    type=click.Choice(GRID_LAYOUTS, case_sensitive=False),
    default=None,
    help=(
        "Grid shape. 'axial' draws a leaning parallelogram; 'offset' draws a rectangle "
        "with ragged north and south edges."
    ),
)
@click.option(
    "--model",
    type=click.Choice(MODELS, case_sensitive=False),
    default="classic",
    show_default=True,
    help="Settlement and road model to run.",
)
@click.option(
    "--heightmap",
    type=str,
    default=None,
    help=(
        "Read the terrain from an image instead of generating it. Replaces the noise "
        "elevation stage; everything downstream is unchanged."
    ),
)
@click.option(
    "--heightmap-mode",
    type=click.Choice(HEIGHTMAP_MODES, case_sensitive=False),
    default=None,
    help=(
        "How to read --heightmap. 'elevation' treats it as a greyscale heightmap; "
        "'coastline' treats it as a land/sea stencil and fills it with generated terrain."
    ),
)
def generate(
    seed: int,
    config: str,
    output_dir: str,
    width: int,
    height: int,
    grid_layout: str,
    model: str,
    heightmap: str,
    heightmap_mode: str,
):
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
    if grid_layout:
        cfg.grid_layout = grid_layout.lower()
    if heightmap:
        cfg.heightmap_path = heightmap
    if heightmap_mode:
        cfg.heightmap_mode = heightmap_mode.lower()

    click.echo(f"Generating world with seed {seed}...")
    click.echo(f"  Size: {cfg.width}×{cfg.height} ({cfg.grid_layout})")
    click.echo(f"  Model: {model}")
    if cfg.heightmap_path:
        click.echo(f"  Heightmap: {cfg.heightmap_path} ({cfg.heightmap_mode})")

    pipeline = GeneratorPipeline(seed, cfg)
    for stage in stages_for(cfg, model):
        pipeline.add_stage(stage)
    try:
        state = pipeline.run()
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo("Writing output...")
    cfg.to_json(str(output_path / "config.json"))

    from .export.json_export import save as save_json
    from .render.debug_viewer import render as render_debug

    save_json(state, str(output_path / "world.json"))

    render_debug(state, "elevation", str(output_path / "elevation.svg"))
    render_debug(state, "terrain_class", str(output_path / "terrain_class.svg"))
    render_debug(state, "river_flow", str(output_path / "river_flow.svg"))
    render_debug(state, "temperature", str(output_path / "temperature.svg"))
    render_debug(state, "moisture", str(output_path / "moisture.svg"))
    render_debug(state, "biome", str(output_path / "biome.svg"))
    render_debug(state, "habitability_city", str(output_path / "habitability_city.svg"))
    render_debug(state, "habitability_town", str(output_path / "habitability_town.svg"))
    render_debug(state, "habitability_village", str(output_path / "habitability_village.svg"))
    render_debug(state, "settlements", str(output_path / "settlements.svg"))
    render_debug(state, "roads", str(output_path / "roads.svg"))
    render_debug(state, "land_cover", str(output_path / "land_cover.svg"))
    render_debug(state, "cultivation", str(output_path / "cultivation.svg"))
    render_debug(state, "territory", str(output_path / "territory.svg"))

    click.echo("✓ Done")


_ATTRIBUTES = [
    "elevation",
    "terrain_class",
    "river_flow",
    "temperature",
    "moisture",
    "biome",
    "habitability",
    "habitability_city",
    "habitability_town",
    "habitability_village",
    "settlements",
    "roads",
    "land_cover",
    "cultivation",
    "territory",
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
@click.option("--output", type=str, required=True, help="Output SVG file")
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
_DEFAULT_LAYERS = {
    "terrain",
    "rivers",
    "roads",
    "settlements",
    "labels",
    "grid",
    "anchorages",
    "crossings",
    "legend",
}
_ALLOWED_LAYERS = _DEFAULT_LAYERS | {"contours"}
_CONFIG_LAYERS_SOURCE = "export.layers in config"


def _parse_layers_value(
    value: str | list[str] | tuple[str, ...] | set[str], source: str
) -> set[str]:
    """Parse layer config from CSV string or iterable and validate allowed names."""
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
    """Load and validate the optional top-level `export` config mapping."""
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
            "legend_corner",
            "legend_scale",
            "river_min_width",
            "river_max_width",
            "river_width_steps",
            "river_width_exponent",
            "feature_outline",
            "river_color",
            "river_casing_color",
            "road_casing_color",
        ):
            if key in export_section:
                svg_kwargs[key] = export_section[key]
        if "layers" in export_section:
            svg_kwargs["layers"] = _parse_layers_value(
                export_section["layers"], _CONFIG_LAYERS_SOURCE
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
    try:
        save_svg(state, output, cfg)
    except ValueError as exc:
        raise click.ClickException(f"Bad export setting: {exc}") from exc
    click.echo(f"✓ Saved to {output}")


@cli.command(name="import-heightmap")
@click.option("--input", "input_path", type=str, required=True, help="Input image file")
@click.option("--output-dir", type=str, default="./output", help="Output directory")
@click.option("--config", "config_path", type=str, default=None, help="Config YAML/JSON file")
@click.option(
    "--mode",
    type=click.Choice(HEIGHTMAP_MODES, case_sensitive=False),
    default=None,
    help=(
        "'elevation' reads the image as a greyscale heightmap; 'coastline' reads it as a "
        "land/sea stencil and fills it with generated terrain. Defaults to the config's "
        "heightmap_mode, itself 'elevation'."
    ),
)
@click.option("--width", type=int, default=None, help="Map width in hexes")
@click.option("--height", type=int, default=None, help="Map height in hexes")
@click.option(
    "--grid-layout",
    type=click.Choice(GRID_LAYOUTS, case_sensitive=False),
    default=None,
    help=(
        "Grid shape. 'offset' is usually the one you want here: it draws a rectangle, so "
        "an imported image is not sheared into a parallelogram."
    ),
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help=(
        "Seeds the terrain in 'coastline' mode. In 'elevation' mode the image decides "
        "everything and this only stamps the world's seed."
    ),
)
def import_heightmap(
    input_path: str,
    output_dir: str,
    config_path: str,
    mode: str,
    width: int,
    height: int,
    grid_layout: str,
    seed: int,
) -> None:
    """Convert an image into elevation, without generating anything else.

    Runs the import and terrain classification alone, so you can check where the coast
    actually landed before paying for a full world.  Erosion does not run, which means
    this is also the faithful path: the elevations written here are exactly what the
    image says, where `generate` would renormalise them.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        if config_path:
            if config_path.lower().endswith((".yaml", ".yml")):
                cfg = WorldConfig.from_yaml(config_path)
            else:
                cfg = WorldConfig.from_json(config_path)
        else:
            cfg = WorldConfig()
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if width:
        cfg.width = width
    if height:
        cfg.height = height
    if grid_layout:
        cfg.grid_layout = grid_layout.lower()
    cfg.heightmap_path = input_path
    if mode:
        # Only when actually given, so a `heightmap_mode` from --config is not silently
        # overwritten by this option's default — the same rule `generate` follows.
        cfg.heightmap_mode = mode.lower()

    click.echo(f"Importing {input_path} as {cfg.heightmap_mode}...")
    click.echo(f"  Size: {cfg.width}×{cfg.height} ({cfg.grid_layout})")

    from .stages.image_elevation import ImageElevationStage
    from .stages.terrain_class import TerrainClassificationStage

    pipeline = GeneratorPipeline(seed, cfg)
    pipeline.add_stage(ImageElevationStage).add_stage(TerrainClassificationStage)
    try:
        state = pipeline.run()
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    from .export.json_export import save as save_json
    from .render.debug_viewer import render as render_debug

    save_json(state, str(output_path / "world.json"))
    render_debug(state, "elevation", str(output_path / "elevation.svg"))
    render_debug(state, "terrain_class", str(output_path / "terrain_class.svg"))
    click.echo(f"✓ Written to {output_dir}")


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
