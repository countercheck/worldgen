"""Culture pack files: the validator's messages, the loader's folders, and the hash.

A pack is written by hand, often by someone who has never read this code, so the thing
worth testing about a bad one is that the message names the file and the field.
"""

import copy
from importlib import resources

import pytest
import yaml

from tests.worlds import build_pipeline
from worldgen.export.culture_packs import load_packs
from worldgen.naming.packs import CulturePackError, pack_hash, parse_pack


def _english() -> dict:
    text = resources.files("worldgen.naming.packs").joinpath("english.yaml").read_text("utf-8")
    return yaml.safe_load(text)


def _fails(data, match):
    with pytest.raises(CulturePackError, match=match):
        parse_pack(data, "mine.yaml")


def test_a_shipped_pack_parses():
    pack = parse_pack(_english(), "english.yaml")
    assert pack.key == "english"
    assert pack.heads["ford"] == ("ford",)


def test_a_missing_meaning_is_named():
    data = _english()
    del data["heads"]["ford"]
    _fails(data, r"^mine\.yaml: heads is missing 'ford'")


def test_an_unknown_meaning_is_named():
    data = _english()
    data["quals"]["dragon"] = ["wyrm"]
    _fails(data, r"quals has unknown 'dragon'")


def test_an_unknown_top_level_key_is_named():
    data = _english()
    data["colour"] = "red"
    _fails(data, r"the file has unknown key 'colour'")


def test_a_template_using_the_wrong_placeholder_is_named():
    data = _english()
    data["templates"]["founder"] = ["{q}{h}"]
    _fails(data, r"templates\.founder\[0\] uses \{q\}; a founder template may use only")


def test_a_broken_template_is_named():
    data = _english()
    data["templates"]["compound"] = ["{q{h}"]
    _fails(data, r"templates\.compound\[0\] is not a valid template")


def test_ranked_templates_need_three_ranks():
    data = _english()
    data["ranked"] = {"compound": [["{q}{h}"], ["{q}{h}"]]}
    _fails(data, r"ranked\.compound must list exactly three")


def test_an_agreeing_qualifier_needs_endings():
    data = _english()
    data["quals"]["white"] = ["whit~"]
    _fails(data, r"agreement\.endings is not given")


def test_an_agreeing_pack_needs_a_gender_for_every_head():
    data = _english()
    data["quals"]["white"] = ["whit~"]
    data["agreement"] = {"genders": {"ford": "m"}, "endings": {"m": "e"}}
    _fails(data, r"agreement\.genders has no gender for head word")


def test_a_bare_on_is_caught_as_yaml_reading_a_boolean():
    """The commonest way a hand-written pack goes wrong: `particles: [on]` is a list
    holding true, because YAML 1.1 reads a bare on as a boolean."""
    data = yaml.safe_load("particles: [on, upon]")
    pack = _english()
    pack["spelling"]["particles"] = data["particles"]
    _fails(pack, r"spelling\.particles\[0\] is true, not a word.*put the word in quotes")


def test_an_unknown_name_style_is_named():
    data = _english()
    data["proper_names"] = {"style": "runes"}
    _fails(data, r"proper_names\.style is 'runes'; use 'lists' or 'syllables'")


def test_a_syllable_rank_naming_an_unknown_register_is_named():
    data = yaml.safe_load(
        resources.files("worldgen.naming.packs").joinpath("hive.yaml").read_text("utf-8")
    )
    data["proper_names"]["by_rank"][2] = ["hiss"]
    _fails(data, r"proper_names\.by_rank\[2\] names unknown register 'hiss'")


def test_a_fix_that_does_not_compile_is_named():
    data = _english()
    data["spelling"]["fixes"] = [["(", ""]]
    _fails(data, r"spelling\.fixes\[0\] pattern '\(' does not compile")


def test_the_hash_follows_content_not_formatting():
    data = _english()
    same = copy.deepcopy(data)
    changed = copy.deepcopy(data)
    changed["heads"]["ford"] = ["wath"]
    assert pack_hash(data) == pack_hash(same)
    assert pack_hash(data) != pack_hash(changed)


# -- loading ------------------------------------------------------------------------------


def test_every_built_in_pack_ships_and_loads():
    files = [
        e.name
        for e in resources.files("worldgen.naming.packs").iterdir()
        if e.name.endswith(".yaml")
    ]
    packs = load_packs()
    assert len(files) == len(packs) >= 17
    assert all(p.source == "builtin" for p in packs.values())


def _write(folder, name, data):
    folder.mkdir(exist_ok=True)
    (folder / name).write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")


def test_a_user_folder_adds_and_overrides_packs(tmp_path):
    mine = _english()
    mine["heads"]["ford"] = ["wath"]
    _write(tmp_path / "packs", "english.yaml", mine)
    new = _english()
    new["key"], new["name"] = "mercian", "Mercian"
    _write(tmp_path / "packs", "mercian.yaml", new)

    packs = load_packs((str(tmp_path / "packs"),))
    assert packs["english"].source == "user"
    assert packs["english"].pack.heads["ford"] == ("wath",)
    assert packs["mercian"].pack.name == "Mercian"
    assert packs["norse"].source == "builtin"
    assert packs["english"].hash != load_packs()["english"].hash


def test_a_comment_in_a_pack_file_does_not_change_its_hash(tmp_path):
    text = resources.files("worldgen.naming.packs").joinpath("english.yaml").read_text("utf-8")
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "english.yaml").write_text("# a note\n" + text, encoding="utf-8")
    assert load_packs((str(tmp_path / "a"),))["english"].hash == load_packs()["english"].hash


def test_a_missing_folder_is_named(tmp_path):
    with pytest.raises(CulturePackError, match="naming_pack_dirs: .* is not a folder"):
        load_packs((str(tmp_path / "nowhere"),))


def test_two_packs_with_one_key_in_one_folder_are_refused(tmp_path):
    _write(tmp_path / "p", "a.yaml", _english())
    _write(tmp_path / "p", "b.yaml", _english())
    with pytest.raises(CulturePackError, match="key 'english' is already used"):
        load_packs((str(tmp_path / "p"),))


def test_the_stage_uses_a_user_pack_and_records_where_it_came_from(tmp_path):
    new = _english()
    new["key"], new["name"] = "mercian", "Mercian"
    _write(tmp_path / "packs", "mercian.yaml", new)
    ws = build_pipeline(
        width=32,
        height=32,
        naming_packs=("mercian",),
        naming_pack_dirs=(str(tmp_path / "packs"),),
    ).run()
    record = ws.metadata["cultures"][0]
    assert record["pack"] == "mercian"
    assert record["pack_source"] == "user"
    assert len(record["pack_hash"]) == 12
    assert any(s.culture == "Mercian" for s in ws.settlements)


def test_the_authoring_guides_worked_example_is_a_valid_pack():
    """docs/CULTURE_PACKS.md tells a reader the example is valid as it stands."""
    from pathlib import Path

    guide = (Path(__file__).parent.parent / "docs" / "CULTURE_PACKS.md").read_text("utf-8")
    block = guide.split("## A complete small pack")[1].split("```yaml")[1].split("```")[0]
    pack = parse_pack(yaml.safe_load(block), "mercian.yaml")
    assert pack.key == "mercian"


def test_the_cli_reports_a_bad_pack_as_a_message_not_a_traceback(tmp_path):
    from click.testing import CliRunner

    from worldgen.cli import cli

    config = tmp_path / "cfg.yaml"
    config.write_text("naming_packs: [klingon]\n", encoding="utf-8")
    result = CliRunner().invoke(
        cli,
        [
            "generate",
            "--seed",
            "1",
            "--config",
            str(config),
            "--width",
            "16",
            "--height",
            "16",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code != 0
    assert "unknown culture pack 'klingon'" in result.output
    assert "Traceback" not in result.output
