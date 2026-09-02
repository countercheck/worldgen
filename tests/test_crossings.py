"""Where rivers can be crossed, and what that does to the districts around them."""

import pytest

from tests.worlds import build_pipeline
from worldgen.core.config import WorldConfig
from worldgen.core.hex import Hex
from worldgen.core.hex_grid import distance, neighbors
from worldgen.stages.crossings import BRIDGE, FORD, river_span
from worldgen.stages.road_cost import is_river

_CROSSING = {FORD, BRIDGE}


def _crossing_world(seed=42, width=64, height=64, stop="CrossingStage", **overrides):
    p = build_pipeline(
        seed=seed, width=width, height=height, model="organic", until=stop, **overrides
    )
    return p.run()


@pytest.fixture(scope="module")
def crossed():
    return _crossing_world()


@pytest.fixture(scope="module")
def settled():
    return _crossing_world(stop="MarketStage")


# --- catchment area is a quantity, not a rank --------------------------------


def test_catchment_area_is_recorded(crossed):
    """Hydrology normalises river_flow away; the area it was normalised from is kept."""
    river = [h for h in crossed.hexes.values() if is_river(h)]
    assert river, "no rivers on the fixture map"
    assert all(h.catchment_km2 > 0 for h in river)


def test_catchment_area_grows_downstream(crossed):
    """A river only ever collects more; that is what makes the figure physical.

    Measured over the land hexes only — a river's path ends on the water body it empties
    into, and the sea drains nothing.
    """
    water = ("ocean", "lake")
    checked = 0
    for river in crossed.rivers:
        areas = [
            crossed.hexes[c].catchment_km2
            for c in river.hexes
            if c in crossed.hexes and crossed.hexes[c].terrain_class.value not in water
        ]
        if len(areas) > 2:
            checked += 1
            assert areas[-1] >= areas[0], f"catchment shrank downstream: {areas}"
    assert checked, "no river was long enough to check"


def test_bigger_maps_really_do_have_bigger_rivers():
    """The defect this field exists to fix.

    `river_flow` is normalised against the largest accumulation present, so every map has
    a 1.0 whatever size its rivers are, and thresholds on it mean different things at
    different sizes. Catchment area is comparable: a larger region drains a larger trunk.
    """
    small = _crossing_world(width=48, height=48)
    large = _crossing_world(width=128, height=128, erosion_iterations=2000)

    def biggest(ws, field):
        return max(getattr(h, field) for h in ws.hexes.values() if is_river(h))

    assert biggest(small, "river_flow") == pytest.approx(biggest(large, "river_flow"), abs=0.01), (
        "river_flow should top out at 1.0 on both — that is what makes it a rank"
    )
    assert biggest(large, "catchment_km2") > 2 * biggest(small, "catchment_km2"), (
        "catchment area should show the larger map's larger trunk river"
    )


# --- river_span --------------------------------------------------------------


def test_span_is_one_at_the_wading_limit():
    cfg = WorldConfig()
    hx = Hex(coord=(0, 0), catchment_km2=cfg.ford_max_catchment_km2)
    assert river_span(hx, cfg) == pytest.approx(1.0)


def test_span_follows_the_square_root_of_area():
    """Width goes as the root of discharge — the same exponent the river renderer uses."""
    cfg = WorldConfig()
    quad = Hex(coord=(0, 0), catchment_km2=cfg.ford_max_catchment_km2 * 4)
    assert river_span(quad, cfg) == pytest.approx(2.0)


def test_span_depends_only_on_the_river_not_on_the_map():
    """The whole point: the same stream reads the same on any map."""
    cfg = WorldConfig()
    hx = Hex(coord=(0, 0), catchment_km2=300.0)
    assert river_span(hx, cfg) == river_span(Hex(coord=(9, 9), catchment_km2=300.0), cfg)


# --- fords are physical ------------------------------------------------------


def test_fords_are_exactly_the_wadeable_reaches(crossed):
    limit = crossed.metadata["config"]["ford_max_catchment_km2"]
    for hx in crossed.hexes.values():
        if not is_river(hx):
            continue
        if FORD in hx.tags:
            assert hx.catchment_km2 <= limit, "tagged a ford on water too big to wade"
        elif hx.catchment_km2 <= limit:
            raise AssertionError(f"wadeable reach at {hx.coord} was not tagged a ford")


def test_a_ford_needs_nobody_to_want_it(crossed):
    """Fords are terrain, not capital — they do not depend on pressure at all."""
    barren = _crossing_world(bridge_pressure_per_span=1e9)
    before = sum(1 for h in crossed.hexes.values() if FORD in h.tags)
    after = sum(1 for h in barren.hexes.values() if FORD in h.tags)
    assert before == after
    assert not any(BRIDGE in h.tags for h in barren.hexes.values())


# --- bridges are capital -----------------------------------------------------


def test_bridges_only_span_water_too_big_to_wade(crossed):
    limit = crossed.metadata["config"]["ford_max_catchment_km2"]
    for hx in crossed.hexes.values():
        if BRIDGE in hx.tags:
            assert hx.catchment_km2 > limit, "bridged a stream that could be waded"


def test_a_dearer_bridge_needs_more_traffic():
    lenient = _crossing_world(
        width=128, height=128, erosion_iterations=2000, bridge_pressure_per_span=2.0
    )
    strict = _crossing_world(
        width=128, height=128, erosion_iterations=2000, bridge_pressure_per_span=8.0
    )

    def bridges(ws):
        return [h for h in ws.hexes.values() if BRIDGE in h.tags]

    assert len(bridges(strict)) < len(bridges(lenient))
    # And the ones that survive are the better-served, not merely the smaller.
    assert max(river_span(h, WorldConfig()) for h in bridges(lenient)) >= max(
        river_span(h, WorldConfig()) for h in bridges(strict)
    )


def test_crossings_keep_their_distance(crossed):
    sep = crossed.metadata["config"]["crossing_min_separation"]
    bridges = [c for c, h in crossed.hexes.items() if BRIDGE in h.tags]
    for i, a in enumerate(bridges):
        for b in bridges[i + 1 :]:
            assert distance(a, b) > sep, f"bridges at {a} and {b} are within sight"


def test_no_bridge_beside_a_ford(crossed):
    """Nobody pays for a structure where the water can already be waded nearby."""
    sep = crossed.metadata["config"]["crossing_min_separation"]
    fords = [c for c, h in crossed.hexes.items() if FORD in h.tags]
    for coord, hx in crossed.hexes.items():
        if BRIDGE in hx.tags:
            assert all(distance(coord, f) > sep for f in fords)


# --- what crossings do to the map --------------------------------------------


def test_most_of_a_river_is_still_an_obstacle(crossed):
    """If a river were crossable everywhere it would not bound anything."""
    river = [h for h in crossed.hexes.values() if is_river(h)]
    crossable = [h for h in river if h.tags & _CROSSING]
    assert len(crossable) < len(river), "every river hex is crossable"


def test_markets_favour_crossings(settled):
    """The Oxford effect, as a measurement.

    A bridging point is the cheapest ground in a district to reach from both banks, so
    deciding crossings before settlement should pull markets onto them. Compared against
    the rate that would arise if markets ignored crossings entirely.
    """
    hexes = settled.hexes

    def at_crossing(coord):
        return bool(hexes[coord].tags & _CROSSING) or any(
            n in hexes and (hexes[n].tags & _CROSSING) for n in neighbors(coord)
        )

    land = [c for c, h in hexes.items() if h.terrain_class.value not in ("ocean", "lake")]
    base_rate = sum(1 for c in land if at_crossing(c)) / len(land)
    market_rate = sum(1 for s in settled.settlements if at_crossing(s.coord)) / len(
        settled.settlements
    )
    assert market_rate > base_rate, (
        f"markets sit at crossings {market_rate:.0%} of the time against a background "
        f"rate of {base_rate:.0%} — crossings are not attracting settlement"
    )


def test_crossings_let_a_catchment_reach_the_far_bank(settled):
    """A district should span the river where it can be crossed and stop where it cannot."""
    hexes = settled.hexes
    spanning = 0
    for coord, hx in hexes.items():
        if not (hx.tags & _CROSSING) or hx.territory is None:
            continue
        owners = {
            hexes[n].territory
            for n in neighbors(coord)
            if n in hexes and hexes[n].territory is not None
        }
        if len(owners) == 1 and hx.territory in owners:
            spanning += 1
    assert spanning, "no catchment holds ground on both sides of a crossing"


def test_same_seed_same_crossings():
    a = _crossing_world(seed=99)
    b = _crossing_world(seed=99)
    assert sorted(c for c, h in a.hexes.items() if h.tags & _CROSSING) == sorted(
        c for c, h in b.hexes.items() if h.tags & _CROSSING
    )
