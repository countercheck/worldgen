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
from ..core.hex_grid import hex_range, neighbors
from ..core.pipeline import GeneratorStage
from ..core.world_state import WorldState
from .habitability import food_value
from .road_cost import is_river

FORD = "ford"
BRIDGE = "bridge"


def channel_drop_m(hx, hexes, cfg) -> float:
    """How far the water falls over this reach, in metres per kilometre of channel.

    Measured *along* the channel — the drop to the lowest neighbouring river hex — because
    that is what sets the velocity, and velocity is what decides whether a reach can be
    waded.  Slack water spreads and braids into shallows; the same discharge running fast
    will take your feet from under you at half the depth.

    Deliberately not the spread of the surrounding ground.  An earlier version measured
    highest neighbour against lowest, which sounds like the same question and is not: a
    river runs in a valley, so that figure reports how tall the valley sides are.  It came
    out at a median of 255 m on a 64x64 map and called all but two reaches unfordable —
    but a river winding down a broad vale with hills either side is perfectly wadeable at
    the water's edge.  What the crossing cares about is the channel, not the skyline.
    """
    lowest = hx.elevation
    for n in neighbors(hx.coord):
        n_hx = hexes.get(n)
        if n_hx is not None and is_river(n_hx):
            lowest = min(lowest, n_hx.elevation)
    return max(0.0, hx.elevation - lowest) * cfg.road_elev_range_m


def river_span(hx, hexes, cfg) -> float:
    """How hard this reach is to get across, in multiples of the easiest wadeable one.

    Two things make it hard, and they multiply rather than compete.

    **How much water.**  Catchment area is the physical input, and width goes as the
    square root of discharge by hydraulic geometry — the same exponent the river renderer
    uses (`river_width_exponent: 0.5`), so the two agree about what a big river looks like.
    Deliberately not `river_flow`: that is normalised against the largest accumulation on
    the map, so it is a rank rather than a quantity, and a threshold on it meant different
    things at different map sizes.

    **How fast it runs.**  A slack reach spreads and braids into shallows you can wade;
    the same discharge falling steeply concentrates into water that will take your feet
    from under you at half the depth.  A steep reach is also an incised one, and at a
    kilometre to the hex what defeats a bridge is rarely the span but the approaches,
    which then have to be cut.  So gradient makes a reach behave like a bigger river for
    both purposes, which is why one number serves fording, bridging, and the cost of
    getting across where there is no crossing at all.
    """
    if cfg.ford_max_catchment_km2 <= 0:
        return 0.0
    width = (hx.catchment_km2 / cfg.ford_max_catchment_km2) ** 0.5
    drop = channel_drop_m(hx, hexes, cfg)
    return width * (1.0 + drop / cfg.crossing_relief_m)


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

        # A ford is any reach no harder to cross than the limit case: a stream at the
        # wading size on level ground. Steep water of the same size does not qualify.
        fords = [c for c in river if river_span(hexes[c], hexes, cfg) <= 1.0]
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
            needed = cfg.bridge_pressure_per_span * river_span(hexes[coord], hexes, cfg)
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
