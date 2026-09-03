"""Pre-industrial transport economics: how far goods can move before they stop being worth
moving, and who can therefore reach what.

The target period is pre-rail and pre-motor, and in that world the binding constraint on
where people live and how large a place can grow is the cost of shifting bulk goods —
chiefly grain.  A draught team eats as it walks, so a cargo hauled far enough overland is
consumed by its own transport before it arrives; water carriage is roughly an order of
magnitude cheaper per tonne-kilometre, which is why the large pre-industrial cities sit on
navigable water and inland ones stay small.

Every range here is a travel-cost budget rather than a distance, so terrain shortens it
automatically.  A hex of level ground costs one unit wherever it lies and relief enters
only as ascent, so a valley floor is cheap however high it sits and a ridge is dear to
cross.  Catchments therefore come out valley-shaped with watersheds as boundaries, which
is the whole point of costing them rather than drawing discs.

Pure functions and one Dijkstra, in the shape of `road_cost.py`: no stage class, no world
mutation, so it can be unit-tested on synthetic grids.
"""

import heapq

from ..core.hex_grid import neighbors
from .crossings import river_span
from .road_cost import WATER, is_river


def usable_fraction(cost: float, range_limit: float) -> float:
    """How much of a cargo's value survives being hauled *cost* far.

    Linear to **exactly zero** at `range_limit`, which is the substance of the model: this
    is not a soft decay that leaves a trace of value at any distance, it is the point at
    which the team has eaten the load.  One constant therefore sets both the reach and the
    falloff, and there is no second knob to disagree with the first.
    """
    if range_limit <= 0.0:
        return 0.0
    if cost <= 0.0:
        return 1.0
    if cost >= range_limit:
        return 0.0
    return 1.0 - cost / range_limit


def navigable(hx, cfg) -> bool:
    """True where a boat can carry bulk: open water, or a river big enough to float one.

    Judged on discharge — catchment area times runoff depth — rather than on `river_flow`,
    which is normalised against the largest accumulation on the map and so says only how
    this river compares with its neighbours.  On the old test every map had a navigable
    trunk by construction, however small its rivers really were.  Discharge means the same
    thing everywhere: an arid region's biggest watercourse can now simply fail to float a
    boat, which is the point.
    """
    if hx.terrain_class in WATER:
        return True
    if not is_river(hx):
        return False
    discharge = hx.catchment_km2 * cfg.runoff_mm(cfg.mean_precip_mm)
    return discharge >= cfg.navigable_min_discharge


def haulage_range(hx, cfg) -> float:
    """The distance bulk goods can travel from *hx* before they are worth nothing.

    Water multiplies it.  Diocletian's Price Edict prices land carriage at roughly 55x sea
    and 11x river for the same tonne-kilometre, so the multiplier — not the absolute land
    range — is the well-attested half of this pair.
    """
    if navigable(hx, cfg):
        return cfg.haulage_range_land * cfg.haulage_range_water_mult
    return cfg.haulage_range_land


def make_travel_cost(hexes, cfg):
    """Node and edge cost closures for people and goods moving over the ground.

    Terrain and slope only.  Every river term in `road_cost.py` is deliberately left out,
    because they answer a question about *roads* rather than about travel:

    - `river_hex_cost` (12.0) prices a road out of threading a channel.  It is larger than
      the whole 10.0 market-day budget, so including it made a single river hex an
      absolute barrier and catchments came out covering a quarter of the map.
    - The channel exclusion in `make_road_edge_cost` prices river-to-river edges at
      infinity for the same reason.  Applied to a catchment it severs one along every
      watercourse — the exact inverse of the truth, since a river valley is the best land
      and the thing that holds a district together.
    - `river_hex_cost` keeps roads off the channel so the side a road runs on stays
      readable.  Nothing to do with how far a farmer walks.

    River crossings are charged, but by `ford_cost` below rather than by
    `river_crossing_edge_cost`.  Those are the same idea weighted differently, and the
    difference is the point.  A road's crossing cost is dominated by its `_base` term —
    the fixed capital of building any bridge at all, which is what stops bridges being
    thrown across every stream.  Somebody walking to market pays no capital: the crossing
    either exists or they ford it, and what decides that is how much water is in the way
    and how steep the ground is around it.  So travel drops the base and keeps only what
    scales with the reach itself.

    `terrain_base_cost` is left out too, and this is the one that actually mattered.  It
    charges 3x on a hill and 10x on a mountain — the cost of *cutting a road* through
    them.  A person walking a level kilometre of hill country covers it in the time they
    would cover a level kilometre of plain; what costs them is the climbing, and ascent is
    charged separately below.  Using both double-counts the same terrain, and since the
    generated maps run about a quarter hill and a quarter mountain, the median step came
    out at 3.0 against a 10.0 day budget: markets reached three hexes instead of ten, and
    catchments covered a third of the land they should.

    So a hex of level ground is one unit wherever it lies, and relief enters only through
    ascent.  A high plateau is walkable, which is right; a ridge is dear to cross, which
    is also right, and is what makes catchments break at watersheds.

    Slope is charged by Naismith's rule rather than by `slope_edge_cost`, for the same
    reason: that curve prices the difficulty of grading a road and saturates at ten times
    base cost.  Naismith's figure is that a fixed amount of ascent costs about as much as
    a set distance on the level, and only ascent counts — walking downhill is free.  Hence
    a linear charge on climb, with no saturation and no descent term.

    Water is impassable rather than cheap.  A road crosses water because a road is a
    route; a catchment is ground somebody works, and leaving the sea traversable at
    `road_water_cost` would let one coastal settlement claim an entire strait.  Fishing is
    handled by `fishery_rim`, which is bounded by the land that does the fishing.
    """

    def node_cost(hx) -> float:
        if hx.terrain_class in WATER:
            return float("inf")
        return cfg.road_flat_cost

    def edge_cost(from_hx, to_hx) -> float:
        if to_hx.terrain_class in WATER or from_hx.terrain_class in WATER:
            return float("inf")
        return ascent_cost(from_hx, to_hx, cfg) + ford_cost(from_hx, to_hx, hexes, cfg)

    return node_cost, edge_cost


def make_bulk_cost(hexes, cfg):
    """Node and edge cost closures for *bulk goods*, which travel by water where they can.

    The counterpart to `make_travel_cost`, and the difference between them is the whole of
    what separates a city from a market.  That function makes water impassable on purpose —
    a catchment is ground somebody works, and a traversable sea would let one coastal
    settlement claim a strait.  A cargo is not a farmer: it goes by ship, and before the
    railway that was the only way to move anything heavy any distance at all.

    Diocletian's Price Edict puts land carriage at roughly fifty-five times sea and eleven
    times river for the same tonne-kilometre.  `haulage_range_water_mult` stands in for both
    at fifteen, applied as a *divisor on the step* rather than a larger budget, so the reach
    it buys runs along the water rather than in a circle around the port.

    Boarding is not free.  `haulage_transship_cost` is charged once at each land-water
    transition, and without it the sea is a teleport: at a fifteenth the cost per hex, a
    cargo once afloat crosses the map for nothing and every coastal market reaches every
    other, which flattens the tier it is supposed to create.  A quay is real capital, and
    charging it is what makes a short hop not worth the trouble while a long haul plainly
    is.
    """
    node_cost, edge_cost = make_travel_cost(hexes, cfg)
    mult = cfg.haulage_range_water_mult

    def bulk_node(hx) -> float:
        if navigable(hx, cfg):
            return cfg.road_flat_cost / mult
        return node_cost(hx)

    def bulk_edge(from_hx, to_hx) -> float:
        afloat_from, afloat_to = navigable(from_hx, cfg), navigable(to_hx, cfg)
        if afloat_from != afloat_to:
            return cfg.haulage_transship_cost  # over the quay, one way or the other
        if afloat_from:
            return 0.0  # already afloat: no ascent, and a navigable river is a road
        return edge_cost(from_hx, to_hx)

    return bulk_node, bulk_edge


def ford_cost(from_hx, to_hx, hexes, cfg) -> float:
    """What it costs to get across a watercourse on foot.

    Charged on each land-river edge, so a perpendicular crossing pays twice — the same
    shape as `river_crossing_edge_cost`, and for the same reason.

    Where `CrossingStage` has already put a ford or a bridge, this is nearly nothing: the
    crossing exists and you walk over it.  Everywhere else it scales with `river_span` and
    has no fixed term, because somebody walking to market pays no capital — a slack
    headwater is a step across and barely registers, a big or fast-running reach is most of
    a day each way.  That is what makes a river bound a catchment along its length while
    the district still reaches across at the one place it can.
    """
    if is_river(from_hx) == is_river(to_hx):
        return 0.0
    channel = from_hx if is_river(from_hx) else to_hx
    if "ford" in channel.tags or "bridge" in channel.tags:
        return cfg.crossing_use_cost
    return cfg.travel_ford_cost * river_span(channel, hexes, cfg)


def ascent_cost(from_hx, to_hx, cfg) -> float:
    """Naismith: a fixed climb costs as much as a set distance on the level.

    `travel_ascent_per_hex` metres of ascent are charged as one hex of flat walking.
    Descent is free — the rule counts climb only, and a catchment that charged for going
    downhill would refuse to follow a valley, which is the one direction it should.
    """
    climb = to_hx.elevation - from_hx.elevation
    if climb <= 0.0:
        return 0.0
    return climb / cfg.travel_ascent_per_hex


def allocate_catchments(hexes, seats, budget: float, cfg):
    """Assign each land hex to the seat that can reach it most cheaply.

    One multi-source Dijkstra over the travel-cost field, stopping at *budget*.  Scales
    with hex count rather than seat count, so two hundred seats cost the same as thirty —
    which is what makes it affordable to re-run whenever the cost field changes.

    Returns `(owner, cost)` keyed by coord, covering only hexes inside somebody's budget.

    Ties break on `(cost, coord, owner)`, so a hex equidistant between two seats always
    goes to the same one regardless of dict ordering.  Determinism here is load-bearing:
    the catchments decide populations, which decide traffic, which decides the roads.
    """
    seats = sorted(seats)
    if not seats or budget <= 0.0:
        return {}, {}

    node_cost, edge_cost = make_travel_cost(hexes, cfg)

    owner: dict = {}
    cost: dict = {}
    heap = [(0.0, seat, seat) for seat in seats if seat in hexes]
    heapq.heapify(heap)

    while heap:
        d, coord, seat = heapq.heappop(heap)
        if coord in owner:
            continue
        owner[coord] = seat
        cost[coord] = d

        hx = hexes[coord]
        for n in neighbors(coord):
            if n in owner:
                continue
            n_hx = hexes.get(n)
            if n_hx is None:
                continue
            step = node_cost(n_hx) + edge_cost(hx, n_hx)
            if step == float("inf"):
                continue
            nd = d + step
            if nd < budget:
                heapq.heappush(heap, (nd, n, seat))

    return owner, cost


def fishery_rim(hexes, owner: dict, cost: dict) -> tuple[dict, dict]:
    """Extend each catchment onto the water its land touches.

    A coastal settlement fishes, and `food_value` already scores open water for exactly
    that reason.  But letting the catchment walk *across* water would hand one settlement
    a whole sea, so the rim is granted rather than traversed: a claimed land hex donates
    its adjacent unclaimed water at its own cost, and the water goes no further.

    Returns fresh dicts; the inputs are not mutated.
    """
    out_owner = dict(owner)
    out_cost = dict(cost)

    for coord in sorted(owner):
        for n in neighbors(coord):
            if n in out_owner:
                continue
            n_hx = hexes.get(n)
            if n_hx is None or n_hx.terrain_class not in WATER:
                continue
            out_owner[n] = owner[coord]
            out_cost[n] = cost[coord]

    return out_owner, out_cost


def gather(values: dict, owner: dict, cost: dict, range_limit: float) -> dict:
    """Total haulage-weighted value each seat can draw from what it owns.

    The one arithmetic every tier shares: a village's production, a market's surplus draw,
    a city's bulk supply.  What changes between them is the range limit and what is being
    summed, never the shape of the sum.
    """
    totals: dict = {}
    for coord, seat in owner.items():
        value = values.get(coord, 0.0)
        if value <= 0.0:
            continue
        weight = usable_fraction(cost[coord], range_limit)
        if weight <= 0.0:
            continue
        totals[seat] = totals.get(seat, 0.0) + value * weight
    return totals


def settleable(hexes, cfg) -> set:
    """Hexes that could carry a settlement at all, before any scoring.

    The same exclusions `HabitabilityStage` scores to zero — you do not found a village on
    open water, a mountain face, or a bog — kept in one place so the two cannot drift.
    """
    from ..core.hex import STEEP_LAND, Biome

    return {
        coord
        for coord, hx in hexes.items()
        if hx.terrain_class not in WATER
        and hx.terrain_class not in STEEP_LAND
        and hx.biome is not Biome.WETLAND
    }
