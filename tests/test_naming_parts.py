"""The pieces names are made from: reading a site, inventing a language, dividing the map
among peoples, and keeping names apart.

Each is checked on its own, on hand-built ground where the answer is known, so a failure
here says which piece broke rather than that some town somewhere came out oddly named.
"""

import numpy as np
import pytest

from worldgen.core.hex import (
    Hex,
    LandCover,
    Settlement,
    SettlementRole,
    SettlementTier,
    TerrainClass,
)
from worldgen.core.hex_grid import neighbors
from worldgen.naming import (
    GENERICS,
    SPECIFICS,
    Language,
    NameRegistry,
    Qualifier,
    add_direction,
    culture_regions,
    edit_distance,
    read_site,
)
from worldgen.naming.site import Site


def _patch(radius: int = 3) -> dict:
    hexes = {}
    for q in range(-radius, radius + 1):
        for r in range(-radius, radius + 1):
            if abs(q + r) <= radius:
                hexes[(q, r)] = Hex(coord=(q, r), land_cover=LandCover.OPEN)
    return hexes


def _village(role=SettlementRole.AGRICULTURAL, tier=SettlementTier.VILLAGE):
    return Settlement((0, 0), tier, role, 300, "placeholder")


def _read(hexes, settlement=None):
    return read_site(hexes, settlement or _village(), hill_relief_m=100.0, high_elevation_m=800.0)


# -- site ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tag", "generic"),
    [
        ("ford", "ford"),
        ("bridge", "bridge"),
        ("river_mouth", "mouth"),
        ("confluence", "confluence"),
        ("cataract", "falls"),
        ("pass", "pass"),
    ],
)
def test_a_tagged_crossing_or_landmark_is_a_head_the_site_can_be_named_for(tag, generic):
    hexes = _patch()
    hexes[(0, 0)].tags.add(tag)
    site = _read(hexes)
    assert generic in site.generics
    # And it outweighs the plain "farm" every village could be called.
    assert site.generics[generic] > site.generics["steading"]


@pytest.mark.parametrize(
    ("role", "generic"),
    [
        (SettlementRole.PORT, "harbour"),
        (SettlementRole.MINING, "mine"),
        (SettlementRole.LUMBER, "clearing"),
        (SettlementRole.MARKET, "market"),
    ],
)
def test_what_a_settlement_is_for_is_a_head_too(role, generic):
    assert generic in _read(_patch(), _village(role)).generics


@pytest.mark.parametrize("tier", list(SettlementTier))
def test_every_tier_has_a_head_whatever_its_ground(tier):
    site = _read(_patch(), _village(tier=tier))
    assert site.generics
    assert set(site.generics) <= set(GENERICS)


def test_a_marsh_next_door_qualifies_a_site_that_was_itself_drained():
    hexes = _patch()
    for n in neighbors((0, 0)):
        hexes[n].land_cover = LandCover.MARSH
    assert "marsh" in _read(hexes).specifics


def test_water_next_door_makes_a_shore():
    hexes = _patch()
    hexes[neighbors((0, 0))[0]].terrain_class = TerrainClass.INLAND_WATER
    assert "shore" in _read(hexes).generics


def test_a_site_commanding_the_ground_below_is_a_hill():
    hexes = _patch()
    assert "hill" not in _read(hexes).generics
    hexes[(0, 0)].relief = 150.0
    assert "hill" in _read(hexes).generics


def test_every_specific_read_is_one_the_vocabulary_knows():
    hexes = _patch()
    hexes[(0, 0)].relief = 300.0
    hexes[(0, 0)].elevation = 1200.0
    hexes[(0, 0)].tags.add("endorheic_shore")
    site = _read(hexes, _village(tier=SettlementTier.CITY))
    assert set(site.specifics) <= set(SPECIFICS)


@pytest.mark.parametrize(
    ("anchor", "direction"),
    [((0, 3), "north"), ((0, -3), "south"), ((-3, 0), "east"), ((3, 0), "west")],
)
def test_a_place_is_named_for_the_side_of_its_bigger_neighbour_it_lies_on(anchor, direction):
    site = Site((0, 0))
    add_direction(site, anchor)
    assert direction in site.specifics


# -- language ------------------------------------------------------------------------------


def test_one_seed_invents_one_language():
    a = Language.generate(np.random.default_rng(5))
    b = Language.generate(np.random.default_rng(5))
    assert a.name == b.name
    assert [a.word(k) for k in GENERICS] == [b.word(k) for k in GENERICS]


def test_different_seeds_invent_different_languages():
    names = {Language.generate(np.random.default_rng(s)).name for s in range(10)}
    assert len(names) > 5


def test_a_meaning_is_the_same_word_every_time_it_is_asked_for():
    lang = Language.generate(np.random.default_rng(1))
    first = lang.word("ford")
    for key in SPECIFICS:
        lang.word(key)
    assert lang.word("ford") == first


def test_the_word_for_a_meaning_does_not_depend_on_when_it_was_first_asked_for():
    """Seeded from the meaning, so a stage asking in a different order gets the same
    language — collisions aside, which a fresh language asked for two words never has."""
    a = Language.generate(np.random.default_rng(9))
    b = Language.generate(np.random.default_rng(9))
    a.word("ford")
    a_marsh = a.word("marsh")
    assert b.word("marsh") == a_marsh


def test_no_two_meanings_share_a_word():
    lang = Language.generate(np.random.default_rng(3))
    words = [lang.word(k) for k in [*GENERICS, *SPECIFICS]]
    assert len(set(words)) == len(words)


@pytest.mark.parametrize("seed", range(12))
def test_a_language_spells_only_with_its_own_letters(seed):
    rng = np.random.default_rng(seed)
    lang = Language.generate(rng)
    alphabet = lang.alphabet() | {" ", "-"}
    for generic in GENERICS:
        for q in (None, Qualifier("meaning", "oak"), Qualifier("meaning", "north")):
            name = lang.place_name(generic, q, rng)
            assert set(name) <= alphabet, (name, sorted(set(name) - alphabet))
            assert name[0].isupper()


def test_a_proper_name_is_used_as_spelled_by_whoever_spelled_it():
    """A town on a river keeps the river's name in the older people's spelling."""
    rng = np.random.default_rng(2)
    lang = Language.generate(rng)
    for _ in range(20):
        name = lang.place_name("ford", Qualifier("proper", "Vassa", "on"), rng)
        assert "Vassa" in name or "vassa" in name


def test_a_proper_name_is_never_a_single_letter():
    for seed in range(20):
        lang = Language.generate(np.random.default_rng(seed))
        for i in range(10):
            assert len(lang.proper_name(f"river:{i}", 2)) >= 3


# -- regions -------------------------------------------------------------------------------


def _strip(length: int, wall_at: int | None = None) -> dict:
    """A single row of land hexes, optionally with one high ridge across it."""
    hexes = {}
    for q in range(length):
        hexes[(q, 0)] = Hex(coord=(q, 0), elevation=2000.0 if q == wall_at else 0.0)
    return hexes


def _regions(hexes, n, homes=None):
    return culture_regions(
        hexes,
        n,
        np.random.default_rng(0),
        climb_m=150.0,
        river_cost=8.0,
        great_rivers=frozenset(),
        water_cost=2.0,
        homes=homes,
    )


def test_every_hex_belongs_to_some_people():
    hexes = _patch(6)
    owner = _regions(hexes, 3)
    assert set(owner) == set(hexes)
    assert set(owner.values()) == {0, 1, 2}


def test_each_people_holds_one_connected_territory():
    hexes = _patch(8)
    owner = _regions(hexes, 4)
    for culture in set(owner.values()):
        held = {c for c, o in owner.items() if o == culture}
        start = next(iter(held))
        seen, stack = {start}, [start]
        while stack:
            for n in neighbors(stack.pop()):
                if n in held and n not in seen:
                    seen.add(n)
                    stack.append(n)
        assert seen == held


def test_a_frontier_falls_on_a_ridge_rather_than_halfway():
    """Two homelands at the ends of a strip: without a ridge they meet in the middle;
    with one a quarter of the way along, they meet on it."""
    ends = [(0, 0), (19, 0)]
    flat = _regions(_strip(20), 2, ends)
    ridged = _regions(_strip(20, wall_at=5), 2, ends)
    assert flat[(4, 0)] == flat[(8, 0)] == 0
    assert ridged[(4, 0)] == 0
    assert ridged[(6, 0)] == ridged[(8, 0)] == 1


def test_homelands_are_spread_apart():
    """Farthest-point placement: on a strip, two homelands can never share a half."""
    owner = _regions(_strip(20), 2)
    assert owner[(0, 0)] != owner[(19, 0)]


# -- registry ------------------------------------------------------------------------------


def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3
    assert edit_distance("same", "same") == 0


def test_a_repeat_is_refused_however_it_is_capitalised_or_hyphenated():
    reg = NameRegistry(min_distance=1)
    reg.add("Ash-Ford")
    assert not reg.accepts("ashford")
    assert not reg.accepts("Ash Ford")
    assert reg.accepts("Oxford")


def test_a_near_miss_is_refused_as_a_misprint():
    reg = NameRegistry(min_distance=2)
    reg.add("Tharnosvik")
    assert not reg.accepts("Tharnasvik")
    assert reg.accepts("Umbergate")
