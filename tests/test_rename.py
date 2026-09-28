"""Naming a finished world again: `stages.rename` and the `worldgen rename` command.

The world goes through world.json first, as it does for a user, so a field the naming
stage reads but the file drops would show up here as names that differ.
"""

import json

import pytest
from click.testing import CliRunner

from tests.worlds import build_world
from worldgen.cli import cli
from worldgen.core.config import WorldConfig
from worldgen.core.world_state import WorldState
from worldgen.stages import rename

_KW = {"width": 64, "height": 64}


def _saved(model):
    """The world as world.json holds it, and a fresh copy each call: rename edits it."""
    return json.loads(json.dumps(build_world(model=model, **_KW).to_dict()))


def _names(ws):
    return (
        [(s.coord, s.name, s.culture, s.etymology) for s in ws.settlements],
        [r.name for r in ws.rivers],
        ws.metadata["cultures"],
    )


def _config(data, **overrides):
    return WorldConfig.from_dict({**data["metadata"]["config"], **overrides})


@pytest.mark.parametrize("model", ["classic", "organic"])
def test_its_own_seed_and_config_give_back_its_own_names(model):
    data = _saved(model)
    original = WorldState.from_dict(data)
    again = rename(WorldState.from_dict(data), _config(data), data["seed"])
    assert _names(again) == _names(original)


def test_other_packs_rename_the_places_and_leave_the_ground():
    data = _saved("classic")
    original = WorldState.from_dict(data)
    cfg = _config(data, naming_packs=("norse", "welsh"), naming_substrate_pack="latin")
    renamed = rename(WorldState.from_dict(data), cfg, data["seed"])

    assert [(s.coord, s.tier, s.population) for s in renamed.settlements] == [
        (s.coord, s.tier, s.population) for s in original.settlements
    ]
    assert renamed.road_edges == original.road_edges
    assert [s.name for s in renamed.settlements] != [s.name for s in original.settlements]
    packs = [c["pack"] for c in renamed.metadata["cultures"]]
    assert packs[:2] == ["norse", "welsh"] and packs[-1] == "latin"
    assert renamed.metadata["config"]["naming_packs"] == ("norse", "welsh")


def test_another_seed_is_another_naming_and_is_recorded():
    data = _saved("classic")
    renamed = rename(WorldState.from_dict(data), _config(data), data["seed"] + 1)
    assert [s.name for s in renamed.settlements] != [
        s.name for s in WorldState.from_dict(data).settlements
    ]
    assert renamed.metadata["naming_seed"] == data["seed"] + 1
    assert renamed.metadata["seed"] == data["seed"]


def test_a_river_the_new_naming_leaves_unnamed_loses_its_old_name():
    data = _saved("classic")
    assert any(r["name"] for r in data["rivers"])
    cfg = _config(data, naming_river_min_catchment_km2=1e12)
    renamed = rename(WorldState.from_dict(data), cfg, data["seed"])
    assert not any(r.name for r in renamed.rivers)


# -- the command --------------------------------------------------------------------------


def _world_file(tmp_path):
    path = tmp_path / "world.json"
    path.write_text(json.dumps(_saved("classic")), encoding="utf-8")
    return path


def _run(*args):
    return CliRunner().invoke(cli, ["rename", *map(str, args)])


def test_the_command_writes_a_renamed_world(tmp_path):
    src, out = _world_file(tmp_path), tmp_path / "out" / "renamed.json"
    result = _run("--input", src, "--output", out, "--packs", "hive,khuzdul", "--seed", 9)
    assert result.exit_code == 0, result.output
    renamed = json.loads(out.read_text(encoding="utf-8"))
    assert [c["pack"] for c in renamed["metadata"]["cultures"]][:2] == ["hive", "khuzdul"]
    assert renamed["metadata"]["naming_seed"] == 9
    assert "Hive" in result.output


def test_the_command_uses_only_the_naming_settings_of_a_config(tmp_path):
    config = tmp_path / "naming.yaml"
    config.write_text("width: 200\nnaming_cultures: 2\nnaming_packs: [french]\n", "utf-8")
    src, out = _world_file(tmp_path), tmp_path / "renamed.json"
    result = _run("--input", src, "--output", out, "--config", config)
    assert result.exit_code == 0, result.output
    renamed = json.loads(out.read_text(encoding="utf-8"))
    assert renamed["metadata"]["config"]["width"] == _KW["width"]
    regional = [c for c in renamed["metadata"]["cultures"] if c["role"] == "regional"]
    assert len(regional) == 2 and regional[0]["pack"] == "french"


def test_substrate_none_leaves_the_rivers_to_the_regions(tmp_path):
    src, out = _world_file(tmp_path), tmp_path / "renamed.json"
    result = _run("--input", src, "--output", out, "--substrate-pack", "none")
    assert result.exit_code == 0, result.output
    roles = [c["role"] for c in json.loads(out.read_text("utf-8"))["metadata"]["cultures"]]
    assert "substrate" not in roles


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("--packs", "klingon"), "unknown culture pack 'klingon'"),
        (("--cultures", 0), "naming_cultures is 0"),
        (("--packs", "norse,norse"), "names a pack twice"),
    ],
)
def test_the_command_reports_a_bad_request_as_a_message(tmp_path, args, message):
    src = _world_file(tmp_path)
    result = _run("--input", src, "--output", tmp_path / "x.json", *args)
    assert result.exit_code != 0
    assert message in result.output
    assert "Traceback" not in result.output
