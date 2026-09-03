"""The village tier: places that exist because the traffic has no way round them.

Every claim here is about the *gate* rather than about a count. A village is founded where
a chokepoint and real traffic coincide, and sized from what the markets above it left
behind — so what these test is that both halves of the gate bind, and that the tier takes
nothing from the tier above.
"""

import pytest

from tests.worlds import build_pipeline, build_world
from worldgen.core.config import WorldConfig
from worldgen.core.hex import Hex, SettlementTier
from worldgen.core.hex_grid import neighbors
from worldgen.core.world_state import ROAD_TIER_RANK, RoadTier
from worldgen.stages.chokepoints import BRIDGE, PASS, is_pass, saddle_relief_m

# 96x96 with the gate opened to `track`, rather than the 128x128 and `secondary` that ship.
# At production's settings this map grows a single village, and one instance cannot show
# that a rule binds. Opening the gate one tier gives ten — two of them on passes — for
# 2.4 s instead of 40, and every rule under test is the same rule.
_CHOKE_DEFAULTS = {
    "regional_climate": "temperate",
    "continent_falloff_edges": ("south",),
    "chokepoint_min_road_tier": "track",
}


def _world(**over):
    """Memoised, so the several tests wanting the same world pay for it once."""
    return build_world(seed=42, width=96, height=96, model="organic", **{**_CHOKE_DEFAULTS, **over})


def _villages(state):
    return [s for s in state.settlements if s.tier is SettlementTier.VILLAGE]


@pytest.fixture(scope="module")
def choke_world():
    return _world()


# --- reading a saddle off the ring -------------------------------------------


def _ring_world(centre_m: float, ring_m: list[float]):
    """A hex and its six neighbours at the given elevations, in ring order."""
    hexes = {(0, 0): Hex(coord=(0, 0), elevation=centre_m)}
    for coord, elev in zip(neighbors((0, 0)), ring_m, strict=True):
        hexes[coord] = Hex(coord=coord, elevation=elev)
    return hexes


def test_a_saddle_reports_the_lesser_of_its_two_flanks():
    """A pass is only as walled as its weaker side.

    High ground east and west, open ground north and south: you cannot go over the walls,
    so you come through the middle. If one wall is a cliff and the other a gentle rise you
    walk over the rise, which is why this reports the smaller of the two.
    """
    hexes = _ring_world(100.0, [400.0, 400.0, 20.0, 250.0, 250.0, 20.0])
    assert saddle_relief_m((0, 0), hexes) == pytest.approx(150.0)


@pytest.mark.parametrize(
    ("name", "centre", "ring"),
    [
        ("summit", 500.0, [100.0] * 6),
        ("pit", 100.0, [500.0] * 6),
        ("hillside", 300.0, [500.0, 500.0, 500.0, 100.0, 100.0, 100.0]),
    ],
)
def test_ground_that_is_not_a_saddle_reports_nothing(name, centre, ring):
    """One run of high ground is a hillside; none at all is a summit or a pit.

    Only two or more runs is a col, and it is the runs that matter rather than the count of
    high neighbours — a hillside can have three and is still somewhere you walk around.
    """
    assert saddle_relief_m((0, 0), _ring_world(centre, ring)) == 0.0, name


def test_a_pass_is_walled_by_ground_the_terrain_bands_call_impassable():
    """The threshold is `terrain_steep_gradient_m`, not a setting of its own.

    A hex is 1 km across, so that figure is already the gradient at which this map says
    "pack animals, terraces, no wheels". A separate knob could only disagree with the
    terrain bands about what a road cannot climb.
    """
    cfg = WorldConfig()
    walled = _ring_world(100.0, [400.0, 400.0, 20.0, 400.0, 400.0, 20.0])
    gentle = _ring_world(100.0, [160.0, 160.0, 20.0, 160.0, 160.0, 20.0])
    assert is_pass((0, 0), walled, cfg)
    assert not is_pass((0, 0), gentle, cfg), (
        f"a saddle walled by {160.0 - 100.0:.0f} m is not a pass at "
        f"terrain_steep_gradient_m={cfg.terrain_steep_gradient_m}"
    )


# --- the gate ----------------------------------------------------------------


def test_every_village_holds_a_chokepoint(choke_world):
    """The first half of the gate. No village is founded on ordinary ground."""
    for s in _villages(choke_world):
        hx = choke_world.hexes[s.coord]
        beside_bridge = any(
            BRIDGE in choke_world.hexes[n].tags
            for n in neighbors(s.coord)
            if n in choke_world.hexes
        )
        assert PASS in hx.tags or BRIDGE in hx.tags or beside_bridge, (
            f"village at {s.coord} holds no bridge and no pass"
        )


def test_every_village_carries_traffic(choke_world):
    """The second half. A bridge on a farm track is a plank, not a town."""
    cfg = WorldConfig(**choke_world.metadata["config"])
    floor = ROAD_TIER_RANK[RoadTier(cfg.chokepoint_min_road_tier)]
    for s in _villages(choke_world):
        carried = [e.tier for key, e in choke_world.road_edges.items() if s.coord in key]
        assert carried, f"village at {s.coord} is on no road at all"
        assert max(ROAD_TIER_RANK[t] for t in carried) >= floor, (
            f"village at {s.coord} carries only {[t.value for t in carried]}"
        )


def test_a_stricter_gate_founds_fewer():
    """Traffic is what makes a crossing worth a settlement, so demanding more gives less."""
    counts = [
        len(_villages(_world(chokepoint_min_road_tier=tier)))
        for tier in ("track", "secondary", "primary")
    ]
    assert counts == sorted(counts, reverse=True), (
        f"village counts by gate track/secondary/primary were {counts}"
    )


def test_thin_country_grows_none():
    """No traffic, no toll village.

    An arid map has bridges on it and roads over them; what it has not got is enough
    moving over them to pay for a settlement at the crossing. At the shipped gate rather
    than this module's loosened one, because that is the claim — a desert grows no bridge
    towns, not that it grows none if you also refuse to count farm tracks.
    """
    arid = _world(regional_climate="arid", chokepoint_min_road_tier="secondary")
    assert not _villages(arid)


# --- what it takes from the tier above ---------------------------------------


def test_villages_take_nothing_from_the_markets(choke_world):
    """The invariant that keeps the peasantry from being counted twice.

    Villages are sized from *residual* surplus — the fraction `usable_fraction` says a
    market could not haul — so founding one cannot make the market above it any smaller,
    and the classic model's habit of sprinkling hamlets over ground already feeding a town
    cannot recur here.
    """
    before = build_world(
        seed=42,
        width=96,
        height=96,
        model="organic",
        until="InterurbanRoadStage",
        **_CHOKE_DEFAULTS,
    )
    was = {s.coord: s.population for s in before.settlements}
    now = {
        s.coord: s.population
        for s in choke_world.settlements
        if s.tier is not SettlementTier.VILLAGE
    }
    assert now == was, "a market changed size when the village tier was founded"


def test_a_village_is_a_village_and_not_a_small_market(choke_world):
    """Different reach, different order of size.

    A market gathers a day's cart; a village gathers a morning's walk out to its fields.
    The gap should be visible without squinting, and no village may outgrow the median
    market or the tier is not a tier.
    """
    villages = _villages(choke_world)
    towns = [s for s in choke_world.settlements if s.tier is SettlementTier.TOWN]
    assert villages and towns
    median_town = sorted(t.population for t in towns)[len(towns) // 2]
    assert max(v.population for v in villages) < median_town


def test_no_village_is_a_farmstead(choke_world):
    """`chokepoint_min_draw` reads directly as people: 0.25 of food is a hundred of them.

    The floor is applied to the real catchment draw rather than to the estimate planting
    ranks on, which is what makes that relation exact rather than approximate — allowing
    only the 0.9 low end of the size jitter.
    """
    cfg = WorldConfig(**choke_world.metadata["config"])
    smallest = round(cfg.chokepoint_min_draw * cfg.people_per_food * 0.9)
    for s in _villages(choke_world):
        assert s.population >= smallest, (
            f"village at {s.coord} has {s.population} people, under the {smallest} floor"
        )


# --- structure ---------------------------------------------------------------


def test_villages_need_no_road_built_for_them(choke_world):
    """Founded on the network by construction, which is why this runs after the roads.

    Nothing has to be recut around them — a settlement tier that perturbed the road model
    would change the traffic that justified it in the first place.
    """
    on_network = {c for key in choke_world.road_edges for c in key}
    for s in _villages(choke_world):
        assert s.coord in on_network


def test_one_settlement_to_a_hex(choke_world):
    coords = [s.coord for s in choke_world.settlements]
    assert len(coords) == len(set(coords))
    for s in choke_world.settlements:
        assert choke_world.hexes[s.coord].settlement is s


def test_same_seed_same_villages():
    a = _world()
    b = build_pipeline(seed=42, width=96, height=96, model="organic", **_CHOKE_DEFAULTS).run()
    assert sorted((s.coord, s.population) for s in _villages(a)) == sorted(
        (s.coord, s.population) for s in _villages(b)
    )
