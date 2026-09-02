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
from ..core.hex_grid import hex_range
from ..core.pipeline import GeneratorStage
from ..core.world_state import WorldState
from .habitability import food_value
from .road_cost import is_river

FORD = "ford"
BRIDGE = "bridge"


def river_span(hx, cfg) -> float:
    """How wide the water is, in multiples of the widest that can be waded.

    Catchment area is the physical input — 1.0 means a stream at the very limit of
    wading, 4.0 a river four times that width.  Width goes as the square root of
    discharge by hydraulic geometry, which is the same exponent the river renderer uses
    (`river_width_exponent: 0.5`), so the two agree about what a big river looks like.

    Deliberately not built on `river_flow`.  That is normalised against the largest
    accumulation on the map, so it is a rank: every map has a 1.0 however small its rivers
    really are, and a threshold on it meant different things at different sizes — 0.15
    caught almost no river at 64x64 (median flow 0.31) and over half at 128x128 (median
    0.08), on the same landscape.  Catchment area says the same thing on any map.
    """
    if cfg.ford_max_catchment_km2 <= 0:
        return 0.0
    return (hx.catchment_km2 / cfg.ford_max_catchment_km2) ** 0.5


def crossing_pressure(coord, surplus: dict, radius: int) -> float:
    """How much there is on either side worth connecting.

    Summed over the whole neighbourhood rather than split into banks: telling one bank
    from the other on a hex grid needs the river's local direction, and a river running
    through good country has good country on both sides of it.  The simplification costs
    little and keeps this a single cheap pass.
    """
    return sum(surplus.get(c, 0.0) for c in hex_range(coord, radius))


class CrossingStage(GeneratorStage):
    """Tags every river hex that can be crossed, as a ford or as a bridge."""

    def run(self, state: WorldState) -> WorldState:
        hexes = state.hexes
        cfg = self.config

        surplus = {
            coord: food_value(hx, cfg, cfg.biome_dry_moist, cfg.biome_wet_moist)
            * cfg.marketable_surplus_fraction
            for coord, hx in hexes.items()
            if hx.terrain_class not in (TerrainClass.OCEAN, TerrainClass.LAKE)
        }

        # sorted() throughout: which of two equally good sites gets the bridge decides
        # where a market later grows, so it must not depend on dict ordering.
        river = sorted(c for c, hx in hexes.items() if is_river(hx))
        if not river:
            return state

        fords = [c for c in river if hexes[c].catchment_km2 <= cfg.ford_max_catchment_km2]
        for coord in fords:
            hexes[coord].tags.add(FORD)

        # Bridges: only where the water cannot be waded, and only where enough lies on
        # either side to be worth the capital. The threshold scales with discharge —
        # a wider river is a dearer structure and needs more traffic to justify it.
        taken = set()
        for coord in fords:
            taken |= set(hex_range(coord, cfg.crossing_min_separation))

        candidates = []
        for coord in river:
            if coord in taken or FORD in hexes[coord].tags:
                continue
            needed = cfg.bridge_pressure_per_span * river_span(hexes[coord], cfg)
            pressure = crossing_pressure(coord, surplus, cfg.crossing_pressure_radius)
            if pressure >= needed:
                candidates.append((pressure - needed, coord))

        # Best-served site first, then suppress its neighbours: nobody builds two bridges
        # within sight of each other, and the surplus that justified one is the same
        # surplus that would have justified the next.
        candidates.sort(key=lambda x: (-x[0], x[1]))
        for _, coord in candidates:
            if coord in taken:
                continue
            hexes[coord].tags.add(BRIDGE)
            taken |= set(hex_range(coord, cfg.crossing_min_separation))

        return state
