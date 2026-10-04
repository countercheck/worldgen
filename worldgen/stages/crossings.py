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

from ..core.hex import TerrainClass
from ..core.hex_grid import Side, hex_range, side_hexes
from ..core.pipeline import GeneratorStage
from ..core.world_state import WorldState
from .habitability import potential_food
from .haulage import catchment_carries_a_barge
from .riverside import side_gradients, side_span

FORD = "ford"
BRIDGE = "bridge"


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

        # Past that size a river is too big to wade, almost everywhere. Almost: a big
        # river spreading slack and shallow over a gravel bed can be got across on foot,
        # and those few places are the fords armies were marched to. So the slackest
        # reaches a little past the wading size are fords too — the easiest first, and
        # never two within `rare_ford_separation` of each other, which is what keeps them
        # rare. Only a major river — one that floats a barge — qualifies: anything smaller
        # is waded wherever it is not steep, and a small stream too steep to wade is a
        # gorge, with no slack reach to find. Roads add more fords later, where they cross
        # (`tag_river_crossings`).
        rare = sorted(
            (span[s], s)
            for s in river
            if 1.0 < span[s] <= cfg.rare_ford_max_span
            and catchment_carries_a_barge(sides[s].catchment_km2, cfg)
        )
        near_rare: set = set()
        for _, side in rare:
            if any(b in near_rare for b in side_hexes(side)):
                continue
            fords.append(side)
            near_rare |= _near(side, cfg.rare_ford_separation)

        for side in fords:
            sides[side].tags.add(FORD)

        # Bridges: only where the water cannot be waded, and only where enough lies on
        # either bank to be worth the capital. The threshold scales with discharge —
        # a wider river is a dearer structure and needs more traffic to justify it.
        #
        # A ford nearby makes a bridge needless only over water of its own size. A brook
        # forded beside a trunk river gets nobody across the trunk; counting it would
        # leave a well-watered map, fords on every brook, with hardly a bridge on it.
        sep = cfg.crossing_min_separation
        forded: dict = {}  # hex -> the largest catchment forded within `sep` of it
        for side in fords:
            km2 = sides[side].catchment_km2
            for c in _near(side, sep):
                forded[c] = max(forded.get(c, 0.0), km2)
        taken: set = set()

        def served(side) -> bool:
            km2 = sides[side].catchment_km2 * cfg.ford_serves_bridge_fraction
            return any(b in taken or forded.get(b, 0.0) >= km2 for b in side_hexes(side))

        candidates = []
        for side in river:
            if FORD in sides[side].tags or served(side):
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
            if served(side):
                continue
            sides[side].tags.add(BRIDGE)
            taken |= _near(side, sep)

        return state
