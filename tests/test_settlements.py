from tests.worlds import build_pipeline, build_world
from worldgen.core.config import WorldConfig
from worldgen.core.hex import Biome, Hex, SettlementRole, SettlementTier, TerrainClass
from worldgen.stages.city_town import _assign_role as assign_city_town_role


def test_has_settlements(settle_state):
    assert len(settle_state.settlements) >= 1


def test_has_each_tier(settle_state):
    tiers = {s.tier for s in settle_state.settlements}
    assert SettlementTier.CITY in tiers, "No cities placed"
    assert SettlementTier.TOWN in tiers, "No towns placed"
    assert SettlementTier.VILLAGE in tiers, "No villages placed"


def test_settlements_on_land(settle_state):
    for s in settle_state.settlements:
        hx = settle_state.hexes[s.coord]
        assert hx.terrain_class != TerrainClass.OPEN_WATER, (
            f"Settlement {s.name} placed on ocean hex"
        )


def test_city_separation(settle_state):
    from worldgen.core.hex_grid import distance

    cities = [s for s in settle_state.settlements if s.tier == SettlementTier.CITY]
    cfg_sep = settle_state.metadata["config"]["city_min_separation"]
    for i, a in enumerate(cities):
        for b in cities[i + 1 :]:
            d = distance(a.coord, b.coord)
            assert d >= cfg_sep, f"Cities {a.name} and {b.name} too close: {d} < {cfg_sep}"


def test_town_separation(settle_state):
    from worldgen.core.hex_grid import distance

    towns = [s for s in settle_state.settlements if s.tier == SettlementTier.TOWN]
    cfg_sep = settle_state.metadata["config"]["town_min_separation"]
    for i, a in enumerate(towns):
        for b in towns[i + 1 :]:
            d = distance(a.coord, b.coord)
            assert d >= cfg_sep, f"Towns {a.name} and {b.name} too close: {d} < {cfg_sep}"


def test_village_separation(settle_state):
    from worldgen.core.hex_grid import distance

    villages = [s for s in settle_state.settlements if s.tier == SettlementTier.VILLAGE]
    for i, a in enumerate(villages):
        for b in villages[i + 1 :]:
            d = distance(a.coord, b.coord)
            assert d >= 3, f"Villages {a.name} and {b.name} too close: {d} < 3"


def test_hex_settlement_backref(settle_state):
    for s in settle_state.settlements:
        hx = settle_state.hexes[s.coord]
        assert hx.settlement is s, f"hex at {s.coord} does not reference settlement {s.name}"


def test_valid_tiers_and_roles(settle_state):
    for s in settle_state.settlements:
        assert isinstance(s.tier, SettlementTier)
        assert isinstance(s.role, SettlementRole)


def test_positive_population(settle_state):
    for s in settle_state.settlements:
        assert s.population > 0


def test_reproducibility():
    kwargs = {"seed": 13, "target_city_count": 4, "target_town_count": 12}
    s1 = build_world(**kwargs)
    s2 = build_pipeline(**kwargs).run()
    names1 = sorted(s.name for s in s1.settlements)
    names2 = sorted(s.name for s in s2.settlements)
    assert names1 == names2, "Settlement names differ between identical seeds"
    coords1 = sorted(s.coord for s in s1.settlements)
    coords2 = sorted(s.coord for s in s2.settlements)
    assert coords1 == coords2, "Settlement coords differ between identical seeds"


def _make_port_role_test_hexes():
    center = Hex(coord=(0, 0), biome=Biome.GRASSLAND)
    river_neighbor = Hex(coord=(1, 0), biome=Biome.GRASSLAND, river_flow=1.0)
    hexes = {center.coord: center, river_neighbor.coord: river_neighbor}
    return center, river_neighbor, hexes


def _navigable_catchment(cfg):
    """The catchment a river hex needs before a boat floats on it, plus a margin."""
    return cfg.navigable_min_discharge / cfg.runoff_mm(cfg.mean_precip_mm) * 1.1


def test_city_town_port_role_requires_a_river_tag():
    """Adjacency to water alone is not a port — the neighbour must carry the river tag.

    `river_flow` is set on the neighbour throughout: with `river_flow_continuous` the
    hydrology stage writes a flow value onto every draining land hex, so flow alone says
    nothing about whether a hex is a channel.
    """
    cfg = WorldConfig()
    center, river_neighbor, hexes = _make_port_role_test_hexes()
    river_neighbor.catchment_km2 = _navigable_catchment(cfg)

    assert assign_city_town_role(center.coord, center, hexes, cfg) is not SettlementRole.PORT

    river_neighbor.tags.add("river")
    assert assign_city_town_role(center.coord, center, hexes, cfg) is SettlementRole.PORT


def test_city_town_port_role_requires_water_a_boat_can_use():
    """A headwater brook is not a port, however tagged.

    This is the claim that separates a port from a riverside town. Nearly every settlement
    on a generated map stands on water of some kind, so a role that fired on any river tag
    labelled 43% of the land a port and told nobody anything. The discharge test is what
    makes the role rare enough to mean something.
    """
    cfg = WorldConfig()
    center, brook, hexes = _make_port_role_test_hexes()
    brook.tags.add("river")

    brook.catchment_km2 = 1.0
    assert assign_city_town_role(center.coord, center, hexes, cfg) is not SettlementRole.PORT

    brook.catchment_km2 = _navigable_catchment(cfg)
    assert assign_city_town_role(center.coord, center, hexes, cfg) is SettlementRole.PORT


def test_city_town_port_role_agrees_with_the_harbour_bonus():
    """The site scored as a harbour is the settlement labelled a port.

    Both now read `haulage.navigable`. They used to answer differently — habitability on
    discharge, the role on adjacency — so a hamlet could be a `PORT` on water the site
    score gave no harbour credit for.
    """
    from worldgen.stages.haulage import navigable

    cfg = WorldConfig()
    for catchment in (1.0, _navigable_catchment(cfg)):
        center, river_neighbor, hexes = _make_port_role_test_hexes()
        river_neighbor.tags.add("river")
        river_neighbor.catchment_km2 = catchment
        nbrs = [river_neighbor]
        scored_harbour = navigable(center, cfg) or any(navigable(n, cfg) for n in nbrs)
        is_port = assign_city_town_role(center.coord, center, hexes, cfg) is SettlementRole.PORT
        assert scored_harbour == is_port, f"disagreement at catchment {catchment}"
