"""Where a river can be got across, decided before anything is built on the map.

A river is not uniformly crossable and never was.  Most of its length is an obstacle; a
few places are not, and those places are why towns are where they are — Oxford, Frankfurt,
Innsbruck are all named for their crossing.  Deciding crossings *before* settlement rather
than tagging them afterwards is what lets that causality run the right way round: the
bridging point exists first, and the market grows on it because it is the cheapest ground
in the district to reach from both banks.

Two different things put a crossing somewhere, and they are not interchangeable:

**Fordability is physical.**  A shallow braided reach can be waded by anyone at no cost to
anybody.  What makes a reach shallow is low discharge and a slack gradient — a steep reach
of the same river is a gorge, and a big one is deep whatever its bed is doing.  So fords
are free, are decided entirely by terrain, and need no one to want them.

**A bridge is capital.**  Where the water is too big to wade, somebody has to pay for a
structure, and they only do that where enough traffic will use it.  That is the fixed
`road_river_crossing_base` term doing its proper work: not as a toll charged uniformly
along every watercourse, but as a threshold that a particular site either clears or does
not.  The pressure that clears it is the surplus lying within reach on either bank — a
bridge to nowhere does not get built.

Everywhere else the river stays a barrier, which is what makes a trunk river bound a
market catchment instead of being invisible to it.
"""

import math

from ..core.hex import TerrainClass
from ..core.hex_grid import Side, hex_range, side_hexes
from ..core.pipeline import GeneratorStage
from ..core.world_state import WorldState
from .habitability import potential_food
from .hydrology import mirror_on_band

FORD = "ford"
BRIDGE = "bridge"

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


def crossing_pressure(side: Side, surplus: dict, radius: int) -> float:
    """How much there is on either side worth connecting: the surplus within *radius* of
    either bank."""
    around = {c for bank in side_hexes(side) for c in hex_range(bank, radius)}
    return sum(surplus.get(c, 0.0) for c in around)


def _near(side: Side, radius: int) -> set:
    return {c for bank in side_hexes(side) for c in hex_range(bank, radius)}


class CrossingStage(GeneratorStage):
    """Tags every river side that can be crossed, as a ford or as a bridge."""

    def run(self, state: WorldState) -> WorldState:
        hexes = state.hexes
        cfg = self.config

        surplus = {
            coord: potential_food(hx, cfg) * cfg.marketable_surplus_fraction
            for coord, hx in hexes.items()
            if hx.terrain_class not in (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER)
        }

        # sorted() throughout: which of two equally good sites gets the bridge decides
        # where a market later grows, so it must not depend on dict ordering.
        sides = state.river_sides
        if not sides:
            return state
        gradient = side_gradients(state)
        span = {
            s: side_span(rs.catchment_km2, gradient.get(s, 0.0), cfg) for s, rs in sides.items()
        }
        river = sorted(sides)

        # A ford is any reach no harder to cross than the limit case: a stream at the
        # wading size on level ground. Steep water of the same size does not qualify.
        fords = [s for s in river if span[s] <= 1.0]
        for side in fords:
            sides[side].tags.add(FORD)

        # Bridges: only where the water cannot be waded, and only where enough lies on
        # either bank to be worth the capital. The threshold scales with discharge —
        # a wider river is a dearer structure and needs more traffic to justify it.
        sep = cfg.crossing_min_separation
        taken: set = set()
        for side in fords:
            taken |= _near(side, sep)

        candidates = []
        for side in river:
            if FORD in sides[side].tags or any(b in taken for b in side_hexes(side)):
                continue
            needed = cfg.bridge_pressure_per_span * span[side]
            pressure = crossing_pressure(side, surplus, cfg.crossing_pressure_radius)
            if pressure >= needed:
                candidates.append((pressure - needed, side))

        # Best-served site first, then suppress its neighbours: nobody builds two bridges
        # within sight of each other, and the surplus that justified one is the same
        # surplus that would have justified the next.
        candidates.sort(key=lambda x: (-x[0], x[1]))
        for _, side in candidates:
            if any(b in taken for b in side_hexes(side)):
                continue
            sides[side].tags.add(BRIDGE)
            taken |= _near(side, sep)

        # Until the stages after this read crossings off the sides, each is mirrored onto
        # the hex the river is drawn on.
        mirror_on_band(state, (FORD, BRIDGE))
        return state
