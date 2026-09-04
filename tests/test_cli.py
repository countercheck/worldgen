import pytest
import yaml
from click.testing import CliRunner

from worldgen.cli import cli
from worldgen.core.hex import (
    Biome,
    LandCover,
    Settlement,
    SettlementRole,
    SettlementTier,
    TerrainClass,
)
from worldgen.core.world_state import River, Road, RoadTier, WorldState
from worldgen.export.json_export import save as save_json


def _small_world() -> WorldState:
    ws = WorldState.empty(seed=1, width=4, height=4)
    for h in ws.hexes.values():
        h.biome = Biome.GRASSLAND
        h.terrain_class = TerrainClass.FLAT
        h.land_cover = LandCover.OPEN
        h.temperature = 0.5
        h.moisture = 0.5
    ws.settlements = [
        Settlement(
            coord=(1, 1),
            tier=SettlementTier.CITY,
            role=SettlementRole.MARKET,
            population=5000,
            name="Ironhaven",
        )
    ]
    ws.rivers = [River(hexes=[(0, 0), (1, 0)], flow_volume=1.0)]
    ws.roads = [Road(path=[(1, 1), (2, 1)], tier=RoadTier.PRIMARY)]
    return ws


@pytest.fixture()
def world_json(tmp_path) -> str:
    ws = _small_world()
    path = str(tmp_path / "world.json")
    save_json(ws, path)
    return path


def test_export_default(world_json, tmp_path):
    out = str(tmp_path / "world.svg")
    result = CliRunner().invoke(cli, ["export", "--input", world_json, "--output", out])
    assert result.exit_code == 0, result.output
    with open(out) as f:
        content = f.read()
    assert content.startswith("<svg")
    assert content.rstrip().endswith("</svg>")
    assert 'id="layer-contours"' not in content


def test_export_style_topographic(world_json, tmp_path):
    out = str(tmp_path / "topo.svg")
    result = CliRunner().invoke(
        cli, ["export", "--input", world_json, "--output", out, "--style", "topographic"]
    )
    assert result.exit_code == 0, result.output
    with open(out) as f:
        content = f.read()
    assert 'id="layer-labels"' not in content
    assert 'id="layer-terrain"' in content


def test_export_style_wargame(world_json, tmp_path):
    out = str(tmp_path / "wargame.svg")
    result = CliRunner().invoke(
        cli, ["export", "--input", world_json, "--output", out, "--style", "wargame"]
    )
    assert result.exit_code == 0, result.output
    with open(out) as f:
        content = f.read()
    assert 'id="layer-roads"' in content
    assert 'id="layer-labels"' not in content


def test_export_custom_layers(world_json, tmp_path):
    out = str(tmp_path / "custom.svg")
    result = CliRunner().invoke(
        cli,
        ["export", "--input", world_json, "--output", out, "--layers", "terrain,rivers"],
    )
    assert result.exit_code == 0, result.output
    with open(out) as f:
        content = f.read()
    assert 'id="layer-terrain"' in content
    assert 'id="layer-rivers"' in content
    assert 'id="layer-roads"' not in content
    assert 'id="layer-settlements"' not in content


def test_export_contours_layer_allowed(world_json, tmp_path):
    out = str(tmp_path / "contours.svg")
    result = CliRunner().invoke(
        cli, ["export", "--input", world_json, "--output", out, "--layers", "contours"]
    )
    assert result.exit_code == 0, result.output
    with open(out) as f:
        content = f.read()
    assert 'id="layer-contours"' in content


def test_export_hex_size(world_json, tmp_path):
    out = str(tmp_path / "big.svg")
    result = CliRunner().invoke(
        cli, ["export", "--input", world_json, "--output", out, "--hex-size", "24"]
    )
    assert result.exit_code == 0, result.output
    with open(out) as f:
        content = f.read()
    assert "<svg" in content


def test_export_missing_input(tmp_path):
    out = str(tmp_path / "world.svg")
    result = CliRunner().invoke(cli, ["export", "--output", out])
    assert result.exit_code != 0


def test_export_missing_output(world_json):
    result = CliRunner().invoke(cli, ["export", "--input", world_json])
    assert result.exit_code != 0


def test_export_bad_style(world_json, tmp_path):
    out = str(tmp_path / "world.svg")
    result = CliRunner().invoke(
        cli, ["export", "--input", world_json, "--output", out, "--style", "fantasy"]
    )
    assert result.exit_code != 0


def test_export_bad_layer(world_json, tmp_path):
    out = str(tmp_path / "world.svg")
    result = CliRunner().invoke(
        cli, ["export", "--input", world_json, "--output", out, "--layers", "terrain,typo"]
    )
    assert result.exit_code != 0
    assert "typo" in result.output
    assert "Allowed" in result.output


def test_export_layers_with_whitespace(world_json, tmp_path):
    """Whitespace around layer names should be stripped and accepted."""
    out = str(tmp_path / "world.svg")
    result = CliRunner().invoke(
        cli,
        ["export", "--input", world_json, "--output", out, "--layers", "terrain, rivers"],
    )
    assert result.exit_code == 0, result.output


def test_export_layers_empty_entries(world_json, tmp_path):
    """Empty entries from trailing/double commas should be dropped silently."""
    out = str(tmp_path / "world.svg")
    result = CliRunner().invoke(
        cli,
        ["export", "--input", world_json, "--output", out, "--layers", "terrain,,rivers"],
    )
    assert result.exit_code == 0, result.output


def test_export_help_shows_layers_option():
    result = CliRunner().invoke(cli, ["export", "--help"])
    assert result.exit_code == 0, result.output
    assert "--layers" in result.output
    assert "contours" in result.output


def test_export_reads_export_block_from_yaml_config(world_json, tmp_path):
    out = str(tmp_path / "from-config.svg")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.dump(
            {
                "export": {
                    "style": "atlas",
                    "color_mode": "terrain",
                    "hex_size": 9.0,
                    "padding": 7,
                    "layers": ["terrain", "contours"],
                    "contour_elevation_scale_m": 2500.0,
                    "contour_interval_m": 50.0,
                    "contour_max_crossings": 3,
                    "contour_max_stroke": 2.0,
                }
            }
        )
    )

    result = CliRunner().invoke(
        cli,
        ["export", "--input", world_json, "--output", out, "--config", str(cfg_path)],
    )
    assert result.exit_code == 0, result.output
    with open(out) as f:
        content = f.read()
    assert 'id="layer-contours"' in content


def test_export_cli_flags_override_config_values(world_json, tmp_path):
    out = str(tmp_path / "override.svg")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.dump({"export": {"style": "atlas", "layers": ["terrain", "contours"]}})
    )

    result = CliRunner().invoke(
        cli,
        [
            "export",
            "--input",
            world_json,
            "--output",
            out,
            "--config",
            str(cfg_path),
            "--style",
            "wargame",
            "--layers",
            "terrain,grid",
        ],
    )
    assert result.exit_code == 0, result.output
    with open(out) as f:
        content = f.read()
    assert 'id="layer-contours"' not in content
    assert 'id="layer-grid"' in content
    assert 'id="layer-roads"' in content
    assert 'id="layer-settlements"' in content


def test_export_config_layers_can_be_comma_separated_string(world_json, tmp_path):
    out = str(tmp_path / "config-layers-string.svg")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.dump({"export": {"layers": "terrain, contours"}}))
    result = CliRunner().invoke(
        cli,
        ["export", "--input", world_json, "--output", out, "--config", str(cfg_path)],
    )
    assert result.exit_code == 0, result.output
    with open(out) as f:
        content = f.read()
    assert 'id="layer-terrain"' in content
    assert 'id="layer-contours"' in content


def test_export_config_layers_must_be_list_or_string(world_json, tmp_path):
    out = str(tmp_path / "bad.svg")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.dump({"export": {"layers": 123}}))
    result = CliRunner().invoke(
        cli,
        ["export", "--input", world_json, "--output", out, "--config", str(cfg_path)],
    )
    assert result.exit_code != 0
    assert "export.layers in config" in result.output


def test_init_config_writes_nested_output(tmp_path):
    out = tmp_path / "nested" / "worldgen.yaml"
    result = CliRunner().invoke(cli, ["init-config", "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "elevation_gradient" in out.read_text()


def test_init_config_refuses_overwrite_without_force(tmp_path):
    out = tmp_path / "worldgen.yaml"
    out.write_text("original")
    result = CliRunner().invoke(cli, ["init-config", "--output", str(out)])
    assert result.exit_code == 1
    assert "already exists" in result.output
    assert out.read_text() == "original"


def test_init_config_force_overwrites_file(tmp_path):
    out = tmp_path / "worldgen.yaml"
    out.write_text("original")
    result = CliRunner().invoke(cli, ["init-config", "--output", str(out), "--force"])
    assert result.exit_code == 0, result.output
    assert "elevation_gradient" in out.read_text()


# --- importing an image ------------------------------------------------------


def _heightmap(tmp_path, name="hm.png", size=64):
    """A north-dark, south-bright ramp — enough structure to see a coast in."""
    import numpy as np
    from PIL import Image

    ramp = np.linspace(0, 255, size, dtype="uint8")
    path = tmp_path / name
    Image.fromarray(np.repeat(ramp[:, None], size, 1)).save(path)
    return str(path)


@pytest.mark.parametrize("mode", ["elevation", "coastline"])
def test_import_heightmap_writes_a_loadable_world(tmp_path, mode):
    out_dir = tmp_path / f"out-{mode}"
    result = CliRunner().invoke(
        cli,
        [
            "import-heightmap",
            "--input",
            _heightmap(tmp_path, f"{mode}.png"),
            "--output-dir",
            str(out_dir),
            "--mode",
            mode,
            "--width",
            "16",
            "--height",
            "14",
            "--grid-layout",
            "offset",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out_dir / "elevation.svg").exists()
    assert (out_dir / "terrain_class.svg").exists()

    from worldgen.export.json_export import load as load_json

    state = load_json(str(out_dir / "world.json"))
    assert len(state.hexes) == 16 * 14
    elevations = {h.elevation for h in state.hexes.values()}
    assert len(elevations) > 1, "every hex came out identical — nothing was imported"


def test_import_heightmap_mode_defaults_to_the_config(tmp_path):
    """`--mode` used to carry a non-None default and overwrite the config unconditionally,
    so `heightmap_mode: coastline` in a config file was silently ignored — and `generate`
    disagreed with `import-heightmap` about the same setting."""
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"heightmap_mode": "coastline", "width": 16, "height": 16}))
    img = _heightmap(tmp_path, "modes.png")

    from_config = CliRunner().invoke(
        cli,
        [
            "import-heightmap",
            "--input",
            img,
            "--config",
            str(cfg),
            "--output-dir",
            str(tmp_path / "a"),
        ],
    )
    assert from_config.exit_code == 0, from_config.output
    assert "as coastline" in from_config.output

    overridden = CliRunner().invoke(
        cli,
        [
            "import-heightmap",
            "--input",
            img,
            "--config",
            str(cfg),
            "--mode",
            "elevation",
            "--output-dir",
            str(tmp_path / "b"),
        ],
    )
    assert overridden.exit_code == 0, overridden.output
    assert "as elevation" in overridden.output, "an explicit --mode must still win"


def test_import_heightmap_reports_a_bad_path_cleanly(tmp_path):
    """A missing file is a message, not a traceback."""
    missing = tmp_path / "nope.png"
    result = CliRunner().invoke(
        cli,
        ["import-heightmap", "--input", str(missing), "--output-dir", str(tmp_path / "o")],
    )
    assert result.exit_code != 0
    assert "nope.png" in result.output


@pytest.mark.parametrize("command", ["generate", "import-heightmap"])
@pytest.mark.parametrize("kind", ["missing", "malformed"])
def test_bad_config_is_a_message_not_a_traceback(tmp_path, command, kind):
    """`from_yaml` raises FileNotFoundError and yaml.YAMLError, neither a ValueError.

    Both used to escape the per-command `except ValueError` as raw tracebacks — and
    `generate` had no handling at all — so the shared loader owns the whole family now.
    """
    cfg = tmp_path / "bad.yaml"
    if kind == "malformed":
        cfg.write_text("width: [unclosed")

    args = ["--config", str(cfg), "--output-dir", str(tmp_path / "o")]
    if command == "import-heightmap":
        args = ["import-heightmap", "--input", _heightmap(tmp_path, "cfg-err.png"), *args]
    else:
        args = ["generate", *args]

    result = CliRunner().invoke(cli, args)
    assert result.exit_code != 0
    assert "bad.yaml" in result.output, (
        f"expected a clean message naming the config, got: {result.output!r} "
        f"(exception: {result.exception!r})"
    )


def test_generate_accepts_a_heightmap(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"erosion_iterations": 0, "target_city_count": 1}))
    out_dir = tmp_path / "gen"
    result = CliRunner().invoke(
        cli,
        [
            "generate",
            "--seed",
            "3",
            "--config",
            str(cfg),
            "--output-dir",
            str(out_dir),
            "--width",
            "24",
            "--height",
            "21",
            "--grid-layout",
            "offset",
            "--heightmap",
            _heightmap(tmp_path, "gen.png", size=96),
            "--heightmap-mode",
            "coastline",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Heightmap:" in result.output
    assert (out_dir / "world.json").exists()


def test_generate_reports_a_bad_heightmap_cleanly(tmp_path):
    result = CliRunner().invoke(
        cli,
        [
            "generate",
            "--output-dir",
            str(tmp_path / "gen"),
            "--width",
            "12",
            "--height",
            "12",
            "--heightmap",
            str(tmp_path / "absent.png"),
        ],
    )
    assert result.exit_code != 0
    assert "absent.png" in result.output
