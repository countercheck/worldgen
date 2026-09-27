"""Hand-written culture packs: complete, well-formed, and honoured by the stage."""

import numpy as np
import pytest

from tests.worlds import build_world
from worldgen.core.config import CULTURE_PACKS, WorldConfig
from worldgen.naming import GENERICS, SPECIFICS, Qualifier
from worldgen.naming.packs import PACKS, PackCulture

_ALL = sorted(PACKS)


def test_the_config_knows_every_pack_and_no_others():
    assert tuple(sorted(CULTURE_PACKS)) == tuple(_ALL)


@pytest.mark.parametrize("key", _ALL)
def test_a_pack_has_words_for_every_meaning(key):
    pack = PACKS[key]
    assert set(pack.heads) == set(GENERICS)
    assert set(pack.quals) == set(SPECIFICS)
    for words in (*pack.heads.values(), *pack.quals.values()):
        assert words and all(words)


@pytest.mark.parametrize("key", _ALL)
def test_every_name_a_pack_can_make_is_capitalised_and_whole(key):
    """Every head with every qualifier, every template: none raises, none comes out blank
    or lower-case, and no stray template brace or agreement mark survives."""
    culture = PackCulture(PACKS[key])
    rng = np.random.default_rng(0)
    qualifiers = [None, Qualifier("proper", "Vassa", "on"), Qualifier("proper", "Tarin", "of")]
    qualifiers += [Qualifier("meaning", k) for k in SPECIFICS]
    for generic in GENERICS:
        for q in qualifiers:
            for _ in range(4):
                name = culture.place_name(generic, q, rng)
                assert name and name[0].isupper(), (generic, q, name)
                assert not set("{}~") & set(name), name


def test_slavic_adjectives_agree_with_their_noun():
    pack = PACKS["slavic"]
    culture = PackCulture(pack)
    rng = np.random.default_rng(1)
    seen = 0
    for generic in GENERICS:
        for _ in range(10):
            name = culture.place_name(generic, Qualifier("meaning", "oak"), rng)
            if " " in name:
                adjective, noun = name.split(" ", 1)
                assert adjective.endswith(pack.endings[pack.genders[noun.lower()]]), name
                seen += 1
    assert seen


@pytest.mark.parametrize("key", [k for k in _ALL if PACKS[k].endings])
def test_every_head_in_an_agreeing_pack_has_a_gender_it_has_an_ending_for(key):
    pack = PACKS[key]
    heads = {w for words in pack.heads.values() for w in words}
    assert heads <= set(pack.genders), sorted(heads - set(pack.genders))
    assert set(pack.genders.values()) <= set(pack.endings)


def test_an_adjective_agrees_and_a_noun_phrase_is_not_run_into_the_head():
    """Fuente Blanca or Fuenteblanca, but Fuente del Roble — never Fuentedel Roble."""
    culture = PackCulture(PACKS["spanish"])
    rng = np.random.default_rng(7)
    for _ in range(30):
        white = culture.place_name("spring", Qualifier("meaning", "white"), rng)
        oak = culture.place_name("spring", Qualifier("meaning", "oak"), rng)
        assert white.endswith(("Blanca", "blanca")), white
        assert oak.split(" ")[0] in ("Fuente", "Fontana"), oak


def test_latin_puts_a_founder_and_a_river_in_the_genitive():
    culture = PackCulture(PACKS["latin"])
    rng = np.random.default_rng(8)
    assert culture.place_name("market", Qualifier("proper", "Iulius", "of"), rng) == "Forum Iulii"
    assert culture.place_name("bridge", Qualifier("proper", "Vassa", "on"), rng) == "Pons Vassae"


def test_french_particles_stay_lower_case():
    culture = PackCulture(PACKS["french"])
    rng = np.random.default_rng(2)
    name = culture.place_name("ford", Qualifier("meaning", "oak"), rng)
    assert "-du-" in name or "-aux-" in name


@pytest.mark.parametrize("key", _ALL)
def test_proper_names_are_stable_and_step_on_when_asked(key):
    culture = PackCulture(PACKS[key])
    assert culture.proper_name("river:3:0", 2) == culture.proper_name("river:3:0", 2)
    assert culture.proper_name("people", 2) == PACKS[key].name
    # Raising the count is how the stage asks for a different name; far enough past the
    # end of the list it must still be getting new ones.
    names = {culture.proper_name("person:x", n) for n in range(1, 400)}
    assert len(names) > 100


def test_the_config_refuses_a_pack_it_does_not_know():
    with pytest.raises(ValueError, match="unknown culture pack"):
        WorldConfig(naming_packs=("klingon",))
    with pytest.raises(ValueError, match="unknown culture pack"):
        WorldConfig(naming_substrate_pack="klingon")


def test_the_config_refuses_more_packs_than_regions_and_repeats():
    with pytest.raises(ValueError, match="naming_cultures"):
        WorldConfig(naming_cultures=1, naming_packs=("english", "norse"))
    with pytest.raises(ValueError, match="twice"):
        WorldConfig(naming_packs=("english", "english"))


_PACKED = {"naming_packs": ("english", "norse"), "naming_substrate_pack": "welsh"}


@pytest.fixture(scope="module")
def packed():
    return build_world(model="classic", width=64, height=64, **_PACKED)


def test_the_stage_names_regions_in_their_packs(packed):
    roles = {c["name"]: c for c in packed.metadata["cultures"]}
    assert roles["English"]["pack"] == "english"
    assert roles["Norse"]["pack"] == "norse"
    assert roles["Welsh"]["role"] == "substrate"
    assert {s.culture for s in packed.settlements} <= set(roles)


def test_rivers_take_the_substrate_packs_names(packed):
    welsh = PackCulture(PACKS["welsh"])
    possible = set(welsh._rivers())
    named = [r.name for r in packed.rivers if r.name]
    assert named
    assert all(n.casefold() in {p.casefold() for p in possible} for n in named), named


def test_choosing_packs_leaves_the_invented_regions_alone(packed):
    """Region three is invented either way, and speaks the same language either way."""
    plain = build_world(model="classic", width=64, height=64)
    assert plain.metadata["cultures"][2]["name"] == packed.metadata["cultures"][2]["name"]
    assert [s.coord for s in plain.settlements] == [s.coord for s in packed.settlements]


def test_khuzdul_keeps_the_part_after_a_hyphen_lower_case():
    """Khazad-dûm, not Khazad-Dûm."""
    culture = PackCulture(PACKS["khuzdul"])
    rng = np.random.default_rng(4)
    hyphenated = [culture.place_name("hall", Qualifier("meaning", "red"), rng) for _ in range(20)]
    hyphenated = [n for n in hyphenated if "-" in n]
    assert hyphenated
    assert all(n.split("-", 1)[1][0].islower() for n in hyphenated), hyphenated


def test_quenya_drops_the_diaeresis_inside_a_compound():
    """Ninquë alone, but Ninquelondë: ë is only written at the end of a word."""
    culture = PackCulture(PACKS["quenya"])
    rng = np.random.default_rng(5)
    for _ in range(20):
        name = culture.place_name("harbour", Qualifier("meaning", "white"), rng)
        assert "ë" not in name[:-1], name


def test_the_hive_names_a_queens_seat_with_ceremony_and_an_outpost_tersely():
    culture = PackCulture(PACKS["hive"])
    rng = np.random.default_rng(6)
    phrases = ("-of-the-", "-Under-", "-Beside-")
    for _ in range(20):
        queens = culture.place_name("burgh", Qualifier("meaning", "marsh"), rng, rank=0)
        outpost = culture.place_name("enclosure", Qualifier("meaning", "marsh"), rng, rank=2)
        assert any(p in queens for p in phrases), queens
        assert not any(p in outpost for p in phrases), outpost


def test_the_hive_tongue_buzzes_for_queens_and_clicks_for_workers():
    """The two registers share no telltale letters: a buzz has no stops, a click no hum."""
    from worldgen.naming.packs.hive import tongue

    for i in range(50):
        buzz = tongue(f"person:{i}", 2, rank=0)
        click = tongue(f"person:{i}", 2, rank=2)
        assert not set("k'qxc") & set(buzz), buzz
        assert not set("vrsmngo") & set(click), click


def test_the_hive_glosses_a_founder_as_a_swarm():
    culture = PackCulture(PACKS["hive"])
    assert culture.founder_gloss.format(p="Tchix", h="larder") == "larder of the Tchix swarm"
