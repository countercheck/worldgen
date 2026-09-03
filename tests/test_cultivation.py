from tests.worlds import build_pipeline
from worldgen.core.config import WorldConfig
from worldgen.core.hex import LandCover, SettlementTier
from worldgen.core.hex_grid import distance, grade_reachable_count
from worldgen.stages.road_cost import grade_is_under_cap

_RESISTANT = {
    LandCover.BOG,
    LandCover.MARSH,
    LandCover.BARE_ROCK,
    LandCover.ALPINE,
    LandCover.TUNDRA,
    LandCover.DESERT,
    LandCover.OPEN_WATER,
}

# Small radii so farmland does not blanket the map and `test_default_uncultivated` has
# something to assert.  The cult_state fixture in conftest.py uses the same numbers.
_CULT_DEFAULTS = {
    "target_city_count": 3,
    "target_town_count": 8,
    "cultivation_city_radius": 6,
    "cultivation_town_radius": 3,
    "cultivation_village_radius": 2,
}


def _build_pipeline(seed: int = 42, width: int = 64, height: int = 64):
    return build_pipeline(seed=seed, width=width, height=height, **_CULT_DEFAULTS)


def test_default_uncultivated(cult_state):
    """Most land hexes should be uncultivated (wilderness)."""
    land = [h for h in cult_state.hexes.values() if h.land_cover != LandCover.OPEN_WATER]
    cultivated_count = sum(1 for h in land if h.cultivated)
    # Wilderness should dominate — cultivated should be a minority
    assert cultivated_count < len(land), "All land is cultivated — expected wilderness"


def test_resistant_hexes_never_cultivated(cult_state):
    for coord, h in cult_state.hexes.items():
        if h.land_cover in _RESISTANT:
            assert not h.cultivated, (
                f"Resistant hex {coord} (land_cover={h.land_cover}) was marked cultivated"
            )


def test_cities_have_cultivation_nearby(cult_state):
    hexes = cult_state.hexes
    cities = [s for s in cult_state.settlements if s.tier == SettlementTier.CITY]
    for city in cities:
        nearby = [
            hexes[n]
            for n in hexes
            if distance(n, city.coord) <= 6 and hexes[n].land_cover not in _RESISTANT
        ]
        cultivated_nearby = [h for h in nearby if h.cultivated]
        assert cultivated_nearby, f"No cultivated hexes near city at {city.coord}"


def test_villages_on_frontier_or_road(cult_state):
    hexes = cult_state.hexes
    from worldgen.core.hex_grid import neighbors as nbrs

    villages = [s for s in cult_state.settlements if s.tier == SettlementTier.VILLAGE]
    for v in villages:
        hx = hexes[v.coord]
        # Village should be: cultivated with uncultivated neighbor (frontier), OR road-adjacent
        on_frontier = hx.cultivated and any(
            not hexes[n].cultivated
            for n in nbrs(v.coord)
            if n in hexes and hexes[n].land_cover != LandCover.OPEN_WATER
        )
        road_adjacent = bool(hx.road_connections) or any(
            hexes[n].road_connections for n in nbrs(v.coord) if n in hexes
        )
        assert on_frontier or road_adjacent, (
            f"Village at {v.coord} is neither on cultivation frontier nor road-adjacent"
        )


def test_villages_meet_reachability_guard(cult_state):
    cfg = WorldConfig(**cult_state.metadata["config"])
    hexes = cult_state.hexes
    villages = [s for s in cult_state.settlements if s.tier == SettlementTier.VILLAGE]
    for village in villages:
        reachable = grade_reachable_count(
            village.coord,
            hexes,
            lambda a, b: grade_is_under_cap(a, b, cfg),
            cfg.settlement_min_reachable,
        )
        assert reachable >= cfg.settlement_min_reachable, (
            f"Village at {village.coord} is in a topographic pocket ({reachable} reachable hexes)"
        )


def test_village_separation(cult_state):
    villages = [s for s in cult_state.settlements if s.tier == SettlementTier.VILLAGE]
    for i, a in enumerate(villages):
        for b in villages[i + 1 :]:
            d = distance(a.coord, b.coord)
            assert d >= 3, f"Villages {a.name} and {b.name} too close: {d} < 3"


def test_villages_have_track_connection(cult_state):
    """Every village must be joined to the road network by `VillageTrackStage`.

    It used to assert the village was an endpoint of a *TRACK* road specifically. With one
    tier per edge that is no longer the right question: a lane laid onto ground the trunk
    network already covers keeps the higher tier, so a village on the highway is correctly
    not on a track. What has to hold is that it is connected at all.
    """
    track_endpoints: set = {c for edge in cult_state.road_edges for c in edge}

    villages = [s for s in cult_state.settlements if s.tier == SettlementTier.VILLAGE]
    for v in villages:
        assert v.coord in track_endpoints, (
            f"Village at {v.coord} is not an endpoint of any TRACK road — "
            "VillageTrackStage may have failed to connect it"
        )


def test_has_all_tiers(cult_state):
    tiers = {s.tier for s in cult_state.settlements}
    assert SettlementTier.CITY in tiers
    assert SettlementTier.TOWN in tiers
    assert SettlementTier.VILLAGE in tiers


def test_reproducibility():
    s1 = _build_pipeline(seed=55).run()
    s2 = _build_pipeline(seed=55).run()
    cult1 = sorted(c for c, h in s1.hexes.items() if h.cultivated)
    cult2 = sorted(c for c, h in s2.hexes.items() if h.cultivated)
    assert cult1 == cult2, "Cultivated hexes differ between identical seeds"
