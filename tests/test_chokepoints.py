"""The village tier: places that exist because the traffic has no way round them.

Every claim here is about the *gate* rather than about a count. A village is founded where
a chokepoint and real traffic coincide, and sized from what the markets above it left
behind — so what these test is that both halves of the gate bind, and that the tier takes
nothing from the tier above.
"""

import numpy as np
import pytest

from tests.worlds import build_pipeline, build_world
from worldgen.core.config import WorldConfig
from worldgen.core.hex import (
    SOIL_RANK,
    Hex,
    Settlement,
    SettlementRole,
    SettlementTier,
    SoilQuality,
)
from worldgen.core.hex_grid import neighbors
from worldgen.core.world_state import (
    ROAD_TIER_RANK,
    RoadEdge,
    RoadTier,
    WorldState,
    road_edge_key,
)
from worldgen.stages.chokepoints import (
    BRIDGE,
    PASS,
    ChokepointStage,
    is_pass,
    saddle_relief_m,
)

# 112x112 at seed 1, with the gate opened to `track` rather than the `secondary` that
# ships. At production's settings this map grows no village at all — once a bridge has to
# be genuinely crossed to hold anything, secondary-tier crossings are rare — and a rule
# cannot be shown to bind on an empty tier. Opening the gate one tier gives one genuine
# village in each climate, and every rule under test is the same rule.
#
# Moved here from 96x96 seed 42, which this branch emptied. Not a regression: roads follow
# riverbanks now instead of cutting across channels, which is the corridor property
# `test_river_corridor_preference_in_roads` had been carrying as an xfail. On the old
# fixture the bridges a road actually crosses fell from 4 to 2 and road hexes standing on
# a river from 28 to 18, while the bridges *tagged* rose from 41 to 58 — more rivers, less
# reason to cross them. So the village tier is thinner everywhere by design, and a fixture
# has to be somewhere with enough genuine crossings for the rules to have a subject. The
# residual surplus is untouched by any of this: 511 before, 520 after.
_CHOKE_SEED = 1
_CHOKE_SIZE = 112
_CHOKE_DEFAULTS = {
    "regional_climate": "temperate",
    "continent_falloff_edges": ("south",),
    "chokepoint_min_road_tier": "track",
}


def _world(**over):
    """Memoised, so the several tests wanting the same world pay for it once."""
    return build_world(
        seed=_CHOKE_SEED,
        width=_CHOKE_SIZE,
        height=_CHOKE_SIZE,
        model="organic",
        **{**_CHOKE_DEFAULTS, **over},
    )


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


def test_a_wall_with_a_gap_in_it_is_not_a_wall():
    """A flank is only as walled as its lowest hex.

    High ground both sides, but one wall carries a low gap: you cross at the gap, so the
    relief on offer is the gap's, not the outcrop's beside it. Scoring a flank by its
    highest hex called this a 300 m pass; it is a 50 m rise anyone walks over.
    """
    hexes = _ring_world(100.0, [400.0, 150.0, 20.0, 400.0, 400.0, 20.0])
    assert saddle_relief_m((0, 0), hexes) == pytest.approx(50.0)


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
    """The first half of the gate. No village is founded on ordinary ground.

    A bridge only counts if the drawn network actually goes over it — at least two road
    edges, one onto each bank. `CrossingStage` tags candidate sites before any road
    exists, and a tag nothing crosses holds nothing.
    """
    degree: dict = {}
    for a, b in choke_world.road_edges:
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1

    def crossed(coord):
        return BRIDGE in choke_world.hexes[coord].tags and degree.get(coord, 0) >= 2

    for s in _villages(choke_world):
        hx = choke_world.hexes[s.coord]
        beside_bridge = any(crossed(n) for n in neighbors(s.coord) if n in choke_world.hexes)
        assert PASS in hx.tags or crossed(s.coord) or beside_bridge, (
            f"village at {s.coord} holds no crossed bridge and no pass"
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


def test_a_spur_that_joins_no_settlements_is_not_a_chokepoint():
    """The third half of the gate, and the one a generated map will rarely show you.

    Road tiers are percentile cuts — a fixed *share* of edges comes out secondary however
    little uses them — so a short spur can hold a secondary road on no traffic at all.
    `prune_orphan_roads` keeps such a spur where it lands a sea leg, because a road to a
    harbour is a road to somewhere. A village founded on one is a landing place rather than
    a chokepoint, and it ends up cut off by land from every settlement it shares ground
    with, which is the invariant `_join_by_land` exists to hold.

    Built by hand rather than sampled, because the case is rare: it appears on one map in
    sixty (128x128 mediterranean, seed 42) and on none at 48, 64 or 96, so a test that
    generates worlds and looks would have gone on passing.
    """
    cfg = WorldConfig(width=12, height=12, chokepoint_min_road_tier="secondary")
    state = WorldState.empty(1, cfg.width, cfg.height, cfg.grid_layout)

    # A through road joining two settlements, and a spur joining nothing. Both carry a
    # bridge, and both are secondary.
    through = [(1, 1), (2, 1), (3, 1)]
    spur = [(8, 8), (9, 8)]
    for a, b in zip(through, through[1:], strict=False):
        state.road_edges[road_edge_key(a, b)] = RoadEdge(RoadTier.SECONDARY, 0.0)
    state.road_edges[road_edge_key(*spur)] = RoadEdge(RoadTier.SECONDARY, 0.0)
    for coord in (through[1], spur[0]):
        state.hexes[coord].tags.add(BRIDGE)

    for coord in (through[0], through[2]):
        s = Settlement(
            coord=coord,
            tier=SettlementTier.TOWN,
            role=SettlementRole.MARKET,
            population=500,
            name="market",
        )
        state.settlements.append(s)
        state.hexes[coord].settlement = s

    stage = ChokepointStage(cfg, np.random.default_rng(0))
    candidates = stage._candidates(state, cfg)
    assert through[1] in candidates, "the bridge on the through road should be a candidate"
    assert spur[0] not in candidates, (
        f"the bridge on a spur joining no settlements was accepted: {candidates}"
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


def test_thin_country_grows_few_villages_and_only_on_its_good_ground():
    """A desert is mostly empty, and what lives in it lives on the exceptions.

    This used to assert an arid map grew no villages at all, and before soil existed that
    was true because every desert hex was worth exactly nothing. The soil model gives an
    arid map its exceptions — alluvium along the rivers it has, and the odd well-watered
    corner the orography leaves — so the better claim is not that nobody lives there but
    that nobody lives on the sand.

    Compared at the `track` gate, where villages exist to compare. It once ran a tier
    higher, which read as the stronger claim and was the weaker one: the arid world grew
    zero villages there, so the soil loop below asserted over an empty list for as long
    as it existed — and once bridges had to be genuinely crossed, *both* worlds came out
    empty at `secondary` and the count comparison was 0 < 0. The bound is `<=` because
    the village tier is thin everywhere on this map: one genuine crossing each. What the
    desert must never do is out-village the well-watered country, and what its villages
    must never do is stand on sand — and the arid world does grow one here, on a pass on
    arable ground, so the soil assertion finally has a subject.
    """
    arid = _world(regional_climate="arid")
    temperate = _world()
    arid_villages = _villages(arid)
    assert len(arid_villages) <= len(_villages(temperate)), (
        "a desert should carry no more bridge villages than well-watered country"
    )
    assert arid_villages, (
        "the arid track-gate world is expected to grow a village; if terrain changes "
        "emptied it, move this test somewhere the soil claim has a subject again"
    )
    for v in arid_villages:
        soil = arid.hexes[v.coord].soil
        assert SOIL_RANK[soil] >= SOIL_RANK[SoilQuality.ARABLE], (
            f"village at {v.coord} stands on {soil.value} ground in a desert"
        )


# --- what it takes from the tier above ---------------------------------------


def test_villages_take_nothing_from_the_markets(choke_world):
    """The invariant that keeps the peasantry from being counted twice.

    Villages are sized from *residual* surplus — the fraction `usable_fraction` says a
    market could not haul — so founding one cannot make the market above it any smaller,
    and the classic model's habit of sprinkling hamlets over ground already feeding a town
    cannot recur here.
    """
    before = build_world(
        seed=_CHOKE_SEED,
        width=_CHOKE_SIZE,
        height=_CHOKE_SIZE,
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
    b = build_pipeline(
        seed=_CHOKE_SEED,
        width=_CHOKE_SIZE,
        height=_CHOKE_SIZE,
        model="organic",
        **_CHOKE_DEFAULTS,
    ).run()
    assert sorted((s.coord, s.population) for s in _villages(a)) == sorted(
        (s.coord, s.population) for s in _villages(b)
    )


def test_a_bridge_no_road_crosses_founds_nothing():
    """`CrossingStage` tags candidate sites before any road exists, and most are never
    built at. A road hex beside such a tag is not a bridgehead: the bridge only holds
    anything if the drawn network actually goes over it. Before this was checked, six of
    seven villages on the 96x96 fixture stood at bridges with no road edge at all.
    """
    cfg = WorldConfig(width=12, height=12, chokepoint_min_road_tier="secondary")
    state = WorldState.empty(1, cfg.width, cfg.height, cfg.grid_layout)

    road = [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]
    for a, b in zip(road, road[1:], strict=False):
        state.road_edges[road_edge_key(a, b)] = RoadEdge(RoadTier.SECONDARY, 0.0)

    # One bridge the road runs over, and one tagged beside the road that nothing crosses.
    state.hexes[(2, 1)].tags.add(BRIDGE)  # two road edges: crossed
    state.hexes[(4, 2)].tags.add(BRIDGE)  # neighbours (4, 1); no road edge at all

    for coord in (road[0], road[-1]):
        s = Settlement(
            coord=coord,
            tier=SettlementTier.TOWN,
            role=SettlementRole.MARKET,
            population=500,
            name="market",
        )
        state.settlements.append(s)
        state.hexes[coord].settlement = s

    stage = ChokepointStage(cfg, np.random.default_rng(0))
    candidates = stage._candidates(state, cfg)

    assert (2, 1) in candidates, "the bridge the road crosses is the chokepoint"
    assert (3, 1) in candidates, "the bank beside a crossed bridge is a bridgehead"
    assert (4, 1) not in candidates, (
        "a road hex beside a bridge nothing crosses was accepted as a bridgehead"
    )
