import pytest
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


def test_export_help_layers_default_text():
    result = CliRunner().invoke(cli, ["export", "--help"])
    assert result.exit_code == 0, result.output
    normalized = " ".join(result.output.split())
    assert "default: terrain,rivers,roads,settlements,labels,grid" in normalized, result.output
