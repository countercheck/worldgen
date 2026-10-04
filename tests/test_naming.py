"""NamingStage over real worlds: every place named, no two alike, names that mean their site.

Both models, because each founds settlements its own way and both end in this stage.
"""

import re

import pytest

from tests.worlds import build_pipeline, build_world
from worldgen.core.hex_grid import distance
from worldgen.core.world_state import WorldState
from worldgen.naming import GLOSS, read_site

# The names settlements are founded with, which naming is there to replace.
_PLACEHOLDER = re.compile(r"_(city|town|village|market|port|mine|lumber|pass|bridge)_\d+$")

_KW = {"width": 64, "height": 64}


@pytest.fixture(scope="module", params=["classic", "organic"])
def world(request):
    return build_world(model=request.param, **_KW)


def test_no_placeholder_survives(world):
    assert world.settlements
    left = [s.name for s in world.settlements if _PLACEHOLDER.search(s.name)]
    assert not left, left[:5]


def test_no_two_places_share_a_name(world):
    keys = [s.name.casefold() for s in world.settlements]
    keys += [r.name.casefold() for r in world.rivers if r.name]
    assert len(keys) == len(set(keys))


def test_every_settlement_belongs_to_a_culture_the_world_records(world):
    known = {c["name"] for c in world.metadata["cultures"] if c["role"] == "regional"}
    assert known
    for s in world.settlements:
        assert s.culture in known, s


def test_a_name_means_something_about_its_site(world):
    """The etymology's head is one of the heads the ground offered."""
    cfg = world.metadata["config"]
    for s in world.settlements:
        site = read_site(
            world.hexes, s, cfg["naming_hill_relief_m"], cfg["naming_high_elevation_m"]
        )
        heads = {GLOSS[g] for g in site.generics}
        assert s.etymology
        assert any(s.etymology.endswith(h) or f"{h} on the " in s.etymology for h in heads), (
            s.name,
            s.etymology,
            sorted(heads),
        )


def test_a_town_named_for_a_river_names_a_river_that_is_there(world):
    by_name = {r.name: r for r in world.rivers if r.name}
    for s in world.settlements:
        if " on the " not in s.etymology:
            continue
        river = by_name[s.etymology.rsplit(" on the ", 1)[1]]
        assert min(distance(s.coord, c) for c in river.banks()) <= 1, (s.name, s.etymology)


def test_rivers_are_named_largest_first_down_to_the_threshold(world):
    cfg = world.metadata["config"]
    hexes = world.hexes
    for r in world.rivers:
        mouth = max(hexes[c].catchment_km2 for c in r.banks() if c in hexes)
        if mouth < cfg["naming_river_min_catchment_km2"]:
            assert r.name == ""


def test_the_same_seed_gives_the_same_names():
    a = build_pipeline(model="organic", **_KW).run()
    b = build_pipeline(model="organic", **_KW).run()
    assert [(s.coord, s.name, s.etymology) for s in a.settlements] == [
        (s.coord, s.name, s.etymology) for s in b.settlements
    ]
    assert [r.name for r in a.rivers] == [r.name for r in b.rivers]


@pytest.mark.parametrize("model", ["classic", "organic"])
def test_naming_changes_nothing_but_names(model):
    """Appended last, it draws the last child seed, so the world before it is untouched.
    A world with naming turned off is the same world, placeholder-named."""
    named = build_world(model=model, **_KW)
    bare = build_world(model=model, naming_cultures=0, **_KW)
    assert [(s.coord, s.tier, s.role, s.population) for s in named.settlements] == [
        (s.coord, s.tier, s.role, s.population) for s in bare.settlements
    ]
    assert named.road_edges == bare.road_edges
    assert all(_PLACEHOLDER.search(s.name) for s in bare.settlements)
    assert "cultures" not in bare.metadata


def test_names_survive_a_round_trip(world):
    back = WorldState.from_dict(world.to_dict())
    assert [(s.name, s.culture, s.etymology) for s in back.settlements] == [
        (s.name, s.culture, s.etymology) for s in world.settlements
    ]
    assert [r.name for r in back.rivers] == [r.name for r in world.rivers]


def test_one_culture_names_everything_in_one_language():
    w = build_world(model="organic", naming_cultures=1, naming_substrate=False, **_KW)
    assert len({s.culture for s in w.settlements}) == 1
    assert [c["role"] for c in w.metadata["cultures"]] == ["regional"]
