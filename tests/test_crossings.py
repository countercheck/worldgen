"""Where rivers can be crossed, and what that does to the districts around them."""

import pytest

from tests.worlds import build_pipeline, lay_river
from worldgen.core.config import WorldConfig
from worldgen.core.hex_grid import distance, neighbors, side_hexes
from worldgen.core.world_state import WorldState
from worldgen.stages.crossings import BRIDGE, FORD
from worldgen.stages.haulage import catchment_carries_a_barge
from worldgen.stages.riverside import SIDE_KM, side_gradients, side_span

_CROSSING = {FORD, BRIDGE}


def _crossing_world(seed=42, width=64, height=64, stop="CrossingStage", **overrides):
    p = build_pipeline(
        seed=seed, width=width, height=height, model="organic", until=stop, **overrides
    )
    return p.run()


def _banks(ws):
    """The land hexes on a bank of some river."""
    return [ws.hexes[h] for s in ws.river_sides for h in side_hexes(s) if h in ws.hexes]


@pytest.fixture(scope="module")
def crossed():
    return _crossing_world()


@pytest.fixture(scope="module")
def settled():
    # Through `LandUseStage`, not merely `MarketStage`: siting and sizing are separate
    # stages now, and a world stopped at the first has catchments but no settlements.
    return _crossing_world(stop="LandUseStage")


# --- catchment area is a quantity, not a rank --------------------------------


def test_catchment_area_is_recorded(crossed):
    """Hydrology normalises river_flow away; the area it was normalised from is kept."""
    river = _banks(crossed)
    assert river, "no rivers on the fixture map"
    assert all(h.catchment_km2 > 0 for h in river)


def test_catchment_area_grows_downstream(crossed):
    """A river only ever collects more; that is what makes the figure physical."""
    checked = 0
    for river in crossed.rivers:
        areas = [crossed.river_sides[s].catchment_km2 for s in river.sides()]
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
    large = _crossing_world(width=128, height=128)

    def biggest(ws, field):
        return max(getattr(h, field) for h in _banks(ws))

    assert biggest(small, "river_flow") == pytest.approx(biggest(large, "river_flow"), abs=0.01), (
        "river_flow should top out at 1.0 on both — that is what makes it a rank"
    )
    assert biggest(large, "catchment_km2") > 2 * biggest(small, "catchment_km2"), (
        "catchment area should show the larger map's larger trunk river"
    )


# --- side_span ---------------------------------------------------------------


def test_span_is_one_at_the_wading_limit():
    cfg = WorldConfig()
    assert side_span(cfg.ford_max_catchment_km2, 0.0, cfg) == pytest.approx(1.0)


def test_span_follows_the_square_root_of_area():
    """Width goes as the root of discharge — the same exponent the river renderer uses."""
    cfg = WorldConfig()
    assert side_span(cfg.ford_max_catchment_km2 * 4, 0.0, cfg) == pytest.approx(2.0)


# --- steep ground is harder to cross -----------------------------------------


def test_fast_water_is_harder_to_cross_than_slack_water():
    """Same discharge, different gradient: velocity is what takes your feet."""
    cfg = WorldConfig()
    area = cfg.ford_max_catchment_km2
    assert side_span(area, 120.0, cfg) > side_span(area, 0.0, cfg)


def test_gradient_scales_the_span_as_configured():
    """One `crossing_relief_m` of fall per kilometre doubles the difficulty."""
    cfg = WorldConfig()
    assert side_span(cfg.ford_max_catchment_km2, cfg.crossing_relief_m, cfg) == pytest.approx(2.0)


def test_a_steep_trickle_can_be_harder_than_a_slack_river():
    """Size alone does not decide it — which is the point of folding relief in."""
    cfg = WorldConfig()
    area = cfg.ford_max_catchment_km2
    assert side_span(area * 0.5, 200.0, cfg) > side_span(area * 2, 0.0, cfg)


def test_the_gradient_is_read_along_the_river_not_across_it():
    """The distinction that a first attempt got wrong.

    Measuring the spread of surrounding ground reports how tall the valley is, not how
    fast the water runs — it called all but two reaches on a 64x64 map unfordable.  The
    gradient comes from the fall along the course alone: raising the banks beside a river
    changes nothing.
    """
    ws = WorldState.empty(seed=1, width=8, height=8)
    river = lay_river(ws, [(1, 3), (2, 3), (3, 3), (4, 3), (5, 3)])
    for i, side in enumerate(river.sides()):
        ws.river_sides[side].drop_m = 3.0 * i
    before = side_gradients(ws)
    for hx in ws.hexes.values():
        hx.elevation += 250.0 if hx.coord[1] != 3 else 0.0
    assert side_gradients(ws) == before
    # Three sides around each: 3 m a side at the head is half of one gap and one gap.
    first, second = river.sides()[:2]
    assert before[first] == pytest.approx((0.0 + 3.0) / (2 * SIDE_KM))
    assert before[second] == pytest.approx((0.0 + 3.0 + 6.0) / (3 * SIDE_KM))


# --- fords are physical ------------------------------------------------------


def _spans(ws):
    cfg = WorldConfig(**ws.metadata["config"])
    grad = side_gradients(ws)
    return {
        s: side_span(rs.catchment_km2, grad.get(s, 0.0), cfg) for s, rs in ws.river_sides.items()
    }


def test_fords_are_exactly_the_reaches_that_can_be_waded(crossed):
    """A ford is any reach no harder than the limit case: wading size on level ground —
    and, rarely, a slack reach of a major river a little past it."""
    cfg = WorldConfig(**crossed.metadata["config"])
    for side, span in _spans(crossed).items():
        rs = crossed.river_sides[side]
        if FORD in rs.tags:
            rare = catchment_carries_a_barge(rs.catchment_km2, cfg)
            limit = cfg.rare_ford_max_span if rare else 1.0
            assert span <= limit, f"ford at {side} on a reach of span {span:.2f}"
        else:
            assert span > 1.0, f"wadeable reach at {side} was not tagged a ford"


@pytest.fixture(scope="module")
def big_rivers():
    return _crossing_world(width=128, height=128)


def _rare_fords(ws):
    cfg = WorldConfig(**ws.metadata["config"])
    spans = _spans(ws)
    return [
        s
        for s, rs in ws.river_sides.items()
        if FORD in rs.tags and spans[s] > 1.0 and catchment_carries_a_barge(rs.catchment_km2, cfg)
    ]


def test_a_major_river_has_a_few_fords_far_apart(big_rivers):
    """Rare, so a ford on a big river is a place worth marching to, not a formality."""
    cfg = WorldConfig(**big_rivers.metadata["config"])
    rare = _rare_fords(big_rivers)
    major = [
        rs
        for rs in big_rivers.river_sides.values()
        if catchment_carries_a_barge(rs.catchment_km2, cfg)
    ]
    assert rare, "no major river on the fixture has a ford"
    assert len(rare) < 0.05 * len(major), f"{len(rare)} fords on {len(major)} major sides"
    for i, a in enumerate(rare):
        for b in rare[i + 1 :]:
            gap = min(distance(x, y) for x in side_hexes(a) for y in side_hexes(b))
            assert gap > cfg.rare_ford_separation, f"fords at {a} and {b} are {gap} apart"


def test_rare_fords_can_be_turned_off():
    ws = _crossing_world(width=128, height=128, rare_ford_max_span=1.0)
    assert not _rare_fords(ws)


def test_a_brooks_ford_does_not_stand_in_for_a_bridge_over_the_trunk(big_rivers):
    """Only a ford over water of the bridge's own size makes the bridge needless."""
    cfg = WorldConfig(**big_rivers.metadata["config"])
    sep = cfg.crossing_min_separation
    fords = [(s, rs) for s, rs in big_rivers.river_sides.items() if FORD in rs.tags]
    for side, rs in big_rivers.river_sides.items():
        if BRIDGE not in rs.tags:
            continue
        need = rs.catchment_km2 * cfg.ford_serves_bridge_fraction
        for f, frs in fords:
            near = min(distance(x, y) for x in side_hexes(side) for y in side_hexes(f))
            assert near > sep or frs.catchment_km2 < need, (
                f"bridge at {side} beside a ford at {f} over water of its own size"
            )
    any_ford = _crossing_world(width=128, height=128, ford_serves_bridge_fraction=0.0)

    def bridges(ws):
        return sum(1 for r in ws.river_sides.values() if BRIDGE in r.tags)

    assert bridges(big_rivers) > bridges(any_ford), (
        "counting every brook's ford against a bridge should cost bridges"
    )


def test_some_small_water_is_still_unfordable_because_it_is_steep(crossed):
    """If size alone decided it, folding relief into the span would be doing nothing."""
    cfg = WorldConfig(**crossed.metadata["config"])
    steep_and_small = [
        rs
        for rs in crossed.river_sides.values()
        if FORD not in rs.tags and rs.catchment_km2 <= cfg.ford_max_catchment_km2
    ]
    assert steep_and_small, "every unfordable reach is unfordable purely on size"


def test_a_ford_needs_nobody_to_want_it(crossed):
    """Fords are terrain, not capital — they do not depend on pressure at all."""
    barren = _crossing_world(bridge_pressure_per_span=1e9)
    before = sum(1 for rs in crossed.river_sides.values() if FORD in rs.tags)
    after = sum(1 for rs in barren.river_sides.values() if FORD in rs.tags)
    assert before == after
    assert not any(BRIDGE in rs.tags for rs in barren.river_sides.values())


def test_crossings_live_on_sides_only(crossed):
    """A ford is where a river is crossed, which is a side; no hex carries one."""
    assert not any(h.tags & _CROSSING for h in crossed.hexes.values())


# --- bridges are capital -----------------------------------------------------


def test_bridges_only_span_reaches_that_cannot_be_waded(crossed):
    spans = _spans(crossed)
    for side, rs in crossed.river_sides.items():
        if BRIDGE in rs.tags:
            assert spans[side] > 1.0, "bridged a wadeable reach"


def test_a_dearer_bridge_needs_more_traffic():
    lenient = _crossing_world(width=128, height=128, bridge_pressure_per_span=2.0)
    strict = _crossing_world(width=128, height=128, bridge_pressure_per_span=8.0)

    def bridges(ws):
        spans = _spans(ws)
        return [spans[s] for s, rs in ws.river_sides.items() if BRIDGE in rs.tags]

    assert len(bridges(strict)) < len(bridges(lenient))
    # And the ones that survive are the better-served, not merely the smaller.
    assert max(bridges(lenient)) >= max(bridges(strict), default=0.0)


def _bridged(ws):
    return [s for s, rs in ws.river_sides.items() if BRIDGE in rs.tags]


def _gap(a, b):
    return min(distance(x, y) for x in side_hexes(a) for y in side_hexes(b))


def test_crossings_keep_their_distance(crossed):
    sep = crossed.metadata["config"]["crossing_min_separation"]
    bridges = _bridged(crossed)
    for i, a in enumerate(bridges):
        for b in bridges[i + 1 :]:
            assert _gap(a, b) > sep, f"bridges at {a} and {b} are within sight"


# --- what crossings do to the map --------------------------------------------


def test_most_of_a_river_is_still_an_obstacle(crossed):
    """If a river were crossable everywhere it would not bound anything."""
    crossable = [rs for rs in crossed.river_sides.values() if rs.tags & _CROSSING]
    assert len(crossable) < len(crossed.river_sides), "every river side is crossable"


def test_markets_favour_crossings(settled):
    """The Oxford effect, as a measurement.

    A bridging point is the cheapest ground in a district to reach from both banks, so
    deciding crossings before settlement should pull markets onto them. Compared against
    the rate that would arise if markets ignored crossings entirely.
    """
    hexes = settled.hexes
    banks = {
        h for s, rs in settled.river_sides.items() if rs.tags & _CROSSING for h in side_hexes(s)
    }

    def at_crossing(coord):
        return coord in banks or any(n in banks for n in neighbors(coord))

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
    for side, rs in settled.river_sides.items():
        if not rs.tags & _CROSSING:
            continue
        owners = {hexes[h].territory for h in side_hexes(side)}
        if len(owners) == 1 and None not in owners:
            spanning += 1
    assert spanning, "no catchment holds ground on both sides of a crossing"


def test_same_seed_same_crossings():
    a = _crossing_world(seed=99)
    b = _crossing_world(seed=99)
    assert a.river_sides == b.river_sides
