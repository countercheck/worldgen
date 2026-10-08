"""What a world's rivers mean to the stages that read the ground beside them.

Rivers run along hexsides, but almost everything downstream of hydrology — people walking
to market, cargo by barge, a site's worth, the soil, a place's name — is worked out hex by
hex.  This is where the river sides are read into hex terms once, so every stage asks the
same question the same way: what it costs to get across, where a boat can go, and which
hexes lie beside a river and what is there.
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field

from ..core.hex import HexCoord, TerrainClass
from ..core.hex_grid import (
    Corner,
    Side,
    corner_hexes,
    hex_corner_keys,
    neighbors,
    side_corners,
    side_hexes,
)
from ..core.world_state import WorldState

WATER = (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER)

# A hexside is the side of a kilometre-wide hex: 1/sqrt(3) km of river.
SIDE_KM = 1.0 / math.sqrt(3.0)


def side_gradients(state: WorldState) -> dict[Side, float]:
    """How fast the water falls along each river side, in metres per kilometre.

    Measured along the river's own course — the fall over the side and the sides either
    side of it — because that is what sets the velocity, and velocity is what decides
    whether a reach can be waded.  Slack water spreads and braids into shallows; the same
    discharge running fast will take your feet from under you at half the depth.  Three
    sides rather than one because a corner stands at its lowest hex, so the fall over any
    single side comes in lumps: nothing for a side or two across one hex's foot, then all
    of it at once.

    Deliberately not the spread of the ground beside the river.  An earlier version
    measured highest neighbour against lowest, which sounds like the same question and is
    not: a river runs in a valley, so that figure reports how tall the valley sides are.
    It came out at a median of 255 m on a 64x64 map and called all but two reaches
    unfordable — but a river winding down a broad vale with hills either side is
    perfectly wadeable at the water's edge.  What the crossing cares about is the channel,
    not the skyline.
    """
    out: dict[Side, float] = {}
    for river in state.rivers:
        sides = river.sides()
        drops = [state.river_sides[s].drop_m if s in state.river_sides else 0.0 for s in sides]
        for i, side in enumerate(sides):
            lo, hi = max(0, i - 1), min(len(sides), i + 2)
            out[side] = sum(drops[lo:hi]) / ((hi - lo) * SIDE_KM)
    return out


def side_span(catchment_km2: float, gradient_m_per_km: float, cfg) -> float:
    """How hard a stretch of river is to get across, in multiples of the easiest wadeable one.

    Two things make it hard, and they multiply rather than compete.

    **How much water.**  Catchment area is the physical input, and width goes as the
    square root of discharge by hydraulic geometry — the same exponent the river renderer
    uses (`river_width_exponent: 0.5`), so the two agree about what a big river looks like.
    Deliberately not the flow rank: that is normalised against the largest accumulation on
    the map, so a threshold on it meant different things at different map sizes.

    **How fast it runs** (`side_gradients`).  A steep reach is also an incised one, and at
    a kilometre to the hex what defeats a bridge is rarely the span but the approaches,
    which then have to be cut.  So gradient makes a reach behave like a bigger river for
    both purposes, which is why one number serves fording, bridging, and the cost of
    getting across where there is no crossing at all.
    """
    if cfg.ford_max_catchment_km2 <= 0:
        return 0.0
    width = (catchment_km2 / cfg.ford_max_catchment_km2) ** 0.5
    return width * (1.0 + gradient_m_per_km / cfg.crossing_relief_m)


@dataclass
class Rivers:
    """What a world's rivers mean to somebody moving goods or themselves across it.

    Rivers run along hexsides, but people and cargo move hex to hex, so this reads the
    river sides into the terms a hex-to-hex search can use:

    *   **crossing** — what it costs to get across the river between two hexes on foot,
        keyed by the unordered pair either side of each river side (`ford_cost`).
    *   **reaches** — the navigable stretches of river each hex lies beside.  A reach is a
        run of sides a barge can use, joined corner to corner and cut at every cataract;
        a boat goes from one hex to the next only if both lie beside the same reach.
    *   **portage** — the hexes beside a cataract.  Nothing afloat passes them, so a cargo
        coming down the river lands above the falls and loads again below.
    *   **floats** — hexes beside a river carrying enough to drive logs.
    *   **bridged** — the hex pairs either side of a river `CrossingStage` bridged: water
        too big to wade, crossed only where somebody built over it, and tolled there.

    And for the stages that read what a place is like:

    *   **beside** — every land hex a river runs along one of the sides of, with the
        largest catchment it has there.  Both banks: a river along a hexside has a hex on
        either hand, and each is as much "on the river" as the other.
    *   **near** — the same over the river's valley floor: the banks, and every hex next to
        a bank that stands no higher than the higher bank.  What used to be a river hex and
        its ring, re-centred on a river that runs between hexes.
    *   **confluence** — land hexes at a corner where two rivers meet.
    *   **features** — per hex, what the rivers beside it carry that a visitor would name
        the place for: a ford, a bridge, a cataract, a river mouth, a confluence, a spring.
    *   **great** — the hex pairs either side of a river too big to cross casually, for
        anything that treats a great river as a frontier.
    """

    crossing: dict[frozenset[HexCoord], float] = field(default_factory=dict)
    reaches: dict[HexCoord, frozenset[int]] = field(default_factory=dict)
    reach_corners: dict[int, frozenset[Corner]] = field(default_factory=dict)
    portage: frozenset[HexCoord] = frozenset()
    floats: frozenset[HexCoord] = frozenset()
    beside: dict[HexCoord, float] = field(default_factory=dict)
    near: dict[HexCoord, float] = field(default_factory=dict)
    confluence: frozenset[HexCoord] = frozenset()
    features: dict[HexCoord, frozenset[str]] = field(default_factory=dict)
    side_catchment: dict[frozenset[HexCoord], float] = field(default_factory=dict)
    bridged: frozenset[frozenset[HexCoord]] = frozenset()

    def afloat(self, hx) -> bool:
        """True where a boat can be: open water, a lake, or the bank of a navigable reach."""
        return hx.terrain_class in WATER or hx.coord in self.reaches

    def joined(self, a_hx, b_hx) -> bool:
        """True where a boat goes from *a* to *b* without landing.

        Water to water; a bank to the water its reach flows into, at a corner the two
        share; or two banks of the same reach.
        """
        a_wet, b_wet = a_hx.terrain_class in WATER, b_hx.terrain_class in WATER
        if a_wet and b_wet:
            return True
        if a_wet != b_wet:
            water, bank = (a_hx, b_hx) if a_wet else (b_hx, a_hx)
            shared = set(hex_corner_keys(water.coord)) & set(hex_corner_keys(bank.coord))
            return any(shared & self.reach_corners[r] for r in self.reaches.get(bank.coord, ()))
        return bool(
            self.reaches.get(a_hx.coord, frozenset()) & self.reaches.get(b_hx.coord, frozenset())
        )


def river_index(state, cfg) -> Rivers:
    """Read *state*'s river sides and corners into a `Rivers`."""
    gradient = side_gradients(state)
    runoff = cfg.runoff_mm(cfg.mean_precip_mm)
    crossing: dict[frozenset[HexCoord], float] = {}
    navigable_sides = []
    portage: set[HexCoord] = set()
    floats: set[HexCoord] = set()
    beside: dict[HexCoord, float] = {}
    near: dict[HexCoord, float] = {}
    features: dict[HexCoord, set[str]] = defaultdict(set)
    side_catchment: dict[frozenset[HexCoord], float] = {}
    bridged: set[frozenset[HexCoord]] = set()
    for side, rs in sorted(state.river_sides.items()):
        pair = frozenset(side_hexes(side))
        side_catchment[pair] = rs.catchment_km2
        for h in pair:
            if h in state.hexes and state.hexes[h].terrain_class not in WATER:
                beside[h] = max(beside.get(h, 0.0), rs.catchment_km2)
                features[h] |= rs.tags & _SIDE_FEATURES
        # The valley floor: the banks and what lies beside them no higher than the
        # higher bank.  Ground above both banks is valley side, not floor.
        banks = [h for h in pair if h in state.hexes]
        top = max((state.hexes[h].elevation for h in banks), default=0.0)
        for h in {n for bank in banks for n in (bank, *neighbors(bank))}:
            hx = state.hexes.get(h)
            if hx is not None and hx.terrain_class not in WATER and hx.elevation <= top + 1e-6:
                near[h] = max(near.get(h, 0.0), rs.catchment_km2)
        if "bridge" in rs.tags and "ford" not in rs.tags:
            bridged.add(pair)
        if rs.tags & {"ford", "bridge"}:
            crossing[pair] = cfg.crossing_use_cost
        else:
            span = side_span(rs.catchment_km2, gradient.get(side, 0.0), cfg)
            crossing[pair] = cfg.travel_ford_cost * span
        discharge = rs.catchment_km2 * runoff  # as `haulage.catchment_carries_a_barge`
        if "cataract" in rs.tags:
            portage |= pair
        elif discharge >= cfg.navigable_min_discharge:
            navigable_sides.append(side)
        if discharge >= cfg.timber_float_min_discharge:
            floats |= pair

    # Reaches: navigable sides joined where they share a corner.
    root = {s: s for s in navigable_sides}

    def find(s):
        while root[s] != s:
            root[s] = root[root[s]]
            s = root[s]
        return s

    at_corner: dict[Corner, list[Side]] = defaultdict(list)
    for s in navigable_sides:
        for c in side_corners(s):
            at_corner[c].append(s)
    for sides in at_corner.values():
        for other in sides[1:]:
            ra, rb = find(sides[0]), find(other)
            if ra != rb:
                root[max(ra, rb)] = min(ra, rb)
    ids = {r: i for i, r in enumerate(sorted({find(s) for s in navigable_sides}))}

    reaches: dict[HexCoord, set[int]] = defaultdict(set)
    corners: dict[int, set[Corner]] = defaultdict(set)
    for s in navigable_sides:
        r = ids[find(s)]
        corners[r] |= set(side_corners(s))
        for h in side_hexes(s):
            if h in state.hexes and h not in portage:
                reaches[h].add(r)

    # What happens at the corners: the hexes meeting at a source, a mouth or a confluence
    # all stand at it.
    confluence: set[HexCoord] = set()
    for corner, tags in state.river_corners.items():
        named = {_CORNER_FEATURES[t] for t in tags if t in _CORNER_FEATURES}
        for h in corner_hexes(corner):
            if h in state.hexes and state.hexes[h].terrain_class not in WATER:
                features[h] |= named
                if "confluence" in tags:
                    confluence.add(h)

    return Rivers(
        crossing=crossing,
        reaches={h: frozenset(v) for h, v in reaches.items()},
        reach_corners={r: frozenset(v) for r, v in corners.items()},
        portage=frozenset(portage),
        floats=frozenset(floats),
        beside=beside,
        near=near,
        confluence=frozenset(confluence),
        features={h: frozenset(v) for h, v in features.items() if v},
        side_catchment=side_catchment,
        bridged=frozenset(bridged),
    )


# What a side or a corner carries that a place beside it can be named for.
_SIDE_FEATURES = frozenset({"ford", "bridge", "cataract", "rapids"})
_CORNER_FEATURES = {
    "river_mouth": "river_mouth",
    "confluence": "confluence",
    "river_source": "headwater",
}


def waterside(coord, hx, hexes, rivers: Rivers) -> bool:
    """A site on water a boat can use, or a step from it: on or beside the bank of a
    navigable reach, or on a shore.

    A step, because a river runs between hexes: its quay is on a bank, and the town that
    handles it may stand on the bank or on the ground behind — what used to be a river hex
    and its ring.  One predicate for the harbour bonus a site is scored with and the port role a
    settlement there is given, so the place scored as a harbour is the place labelled one.
    """
    if hx.terrain_class in WATER:
        return False
    return rivers.afloat(hx) or any(
        n in hexes and rivers.afloat(hexes[n]) for n in neighbors(coord)
    )
