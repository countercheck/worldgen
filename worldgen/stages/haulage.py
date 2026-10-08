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

from ..core.hex import HexCoord, TerrainClass
from ..core.hex_grid import neighbors
from .riverside import WATER, Rivers


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


def navigable(hx, cfg, rivers: Rivers) -> bool:
    """True where a boat can carry bulk: open water, or the bank of a river big enough to
    float one (`Rivers.afloat`).

    Judged on discharge — catchment area times runoff depth — rather than on the flow rank,
    which is normalised against the largest accumulation on the map and so says only how
    this river compares with its neighbours.  On a rank every map had a navigable trunk by
    construction, however small its rivers really were.  Discharge means the same thing
    everywhere: an arid region's biggest watercourse can simply fail to float a boat, which
    is the point.
    """
    return rivers.afloat(hx)


def catchment_carries_a_barge(catchment_km2: float, cfg) -> bool:
    """Enough water to float a barge, whatever the reach is doing.

    The navigability test less the cataract, because `CataractStage` has to ask this of a
    reach to decide whether it is a cataract — a small steep river is only a brook.
    """
    discharge = catchment_km2 * cfg.runoff_mm(cfg.mean_precip_mm)
    return discharge >= cfg.navigable_min_discharge


def floatable(hx, cfg, rivers: Rivers) -> bool:
    """True where timber can be floated: open water, or beside a river carrying enough to
    drive logs.

    A lower bar than `navigable`. Logs were driven and rafted down rivers far too small for
    a barge, but not down a brook: `timber_float_min_discharge` is that bar, in the same
    km2 x mm of discharge.
    """
    return hx.terrain_class in WATER or hx.coord in rivers.floats


def haulage_range(hx, cfg, rivers: Rivers) -> float:
    """The distance bulk goods can travel from *hx* before they are worth nothing.

    Water multiplies it.  Diocletian's Price Edict prices land carriage at 28-56x sea and
    6-11x river for the same tonne-kilometre, so the multiplier — not the absolute land
    range — is the well-attested half of this pair.
    """
    if navigable(hx, cfg, rivers):
        return cfg.haulage_range_land * cfg.haulage_range_water_mult
    return cfg.haulage_range_land


def make_travel_cost(hexes, cfg, rivers: Rivers | None = None):
    """Node and edge cost closures for people and goods moving over the ground.

    Terrain and slope only: the river term in `road_cost.py` is deliberately left out,
    because it answers a question about *roads* rather than about travel.

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
        return ascent_cost(from_hx, to_hx, cfg) + ford_cost(from_hx, to_hx, rivers)

    return node_cost, edge_cost


def make_bulk_cost(hexes, cfg, rivers: Rivers | None = None):
    """Node and edge cost closures for *bulk goods*, which travel by water where they can.

    The counterpart to `make_travel_cost`, and the difference between them is the whole of
    what separates a city from a market.  That function makes water impassable on purpose —
    a catchment is ground somebody works, and a traversable sea would let one coastal
    settlement claim a strait.  A cargo is not a farmer: it goes by ship, and before the
    railway that was the only way to move anything heavy any distance at all.

    Diocletian's Price Edict puts land carriage at 28-56 times sea and 6-11 times river for
    the same tonne-kilometre (Duncan-Jones: sea 1, river 4.9, wagon 28, pack animal 56).  `haulage_range_water_mult` stands in for both
    at fifteen, applied as a *divisor on the step* rather than a larger budget, so the reach
    it buys runs along the water rather than in a circle around the port.

    Boarding is not free.  `haulage_transship_cost` is charged once at each land-water
    transition, and without it the sea is a teleport: at a fifteenth the cost per hex, a
    cargo once afloat crosses the map for nothing and every coastal market reaches every
    other, which flattens the tier it is supposed to create.  A quay is real capital, and
    charging it is what makes a short hop not worth the trouble while a long haul plainly
    is.

    Rivers run along hexsides, so a barge is "on" a hex when it lies beside a navigable
    reach (`Rivers`).  Two such hexes are one voyage only if they share the reach: two
    rivers side by side, or one river either side of a cataract, mean landing and loading
    again, and are charged both.
    """
    if rivers is None:
        raise ValueError("bulk haulage needs the world's rivers: pass river_index(state, cfg)")
    node_cost, edge_cost = make_travel_cost(hexes, cfg, rivers)
    mult = cfg.haulage_range_water_mult
    # A toll on a bridge is a cost of carriage like any other, and a cargo loses value
    # linearly to nothing at `haulage_range_land` (`usable_fraction`): so a toll of a share
    # *s* weighs exactly what *s* times that range of haul does, and a carter with a ford
    # nearer than that goes round.
    toll_cost = cfg.toll_bridge_share * cfg.haulage_range_land

    def landing(hx) -> float:
        # A sea-going ship wants a harbour; a barge ties up at a bank, and a lake boat is
        # the same inland craft, so a river or lake landing costs the river rate.
        if hx.terrain_class is TerrainClass.OPEN_WATER:
            return cfg.haulage_transship_cost
        return cfg.haulage_river_transship_cost

    def bulk_node(hx) -> float:
        if rivers.afloat(hx):
            return cfg.road_flat_cost / mult
        return node_cost(hx)

    def bulk_edge(from_hx, to_hx) -> float:
        afloat_from, afloat_to = rivers.afloat(from_hx), rivers.afloat(to_hx)
        if afloat_from != afloat_to:
            # Over the quay, one way or the other, priced by the water it meets.
            return landing(from_hx if afloat_from else to_hx)
        if afloat_from:
            if rivers.joined(from_hx, to_hx):
                return 0.0  # already afloat: no ascent, and a navigable river is a road
            # Afloat on both but not on the same water: ashore and afloat again.
            return landing(from_hx) + edge_cost(from_hx, to_hx) + landing(to_hx)
        if toll_cost and frozenset((from_hx.coord, to_hx.coord)) in rivers.bridged:
            return edge_cost(from_hx, to_hx) + toll_cost  # over a tolled bridge
        return edge_cost(from_hx, to_hx)

    return bulk_node, bulk_edge


def bulk_routes(
    hexes,
    seats,
    cfg,
    budget: float | None = None,
    rivers: Rivers | None = None,
    step_allowed=None,
) -> tuple[dict[HexCoord, float], dict[HexCoord, HexCoord]]:
    """Cost of hauling bulk to the nearest of *seats* from anywhere within `haulage_range_land`.

    A Dijkstra over `make_bulk_cost` rather than `make_travel_cost`. That distinction is the
    whole of what a city is: travel cost makes water impassable, because a catchment is
    ground somebody works, while a cargo goes by ship. Using the wrong one does not fail
    loudly — it silently makes every city inland, because the reach then measures nothing
    but how central a place is on land.

    Returns cost keyed by coord, over hexes within budget, and each hex's next step toward
    the seat — the route a cargo from there takes, which transshipment walks. With several
    seats it answers "how cheaply can a cargo from here reach any of them". *budget*
    defaults to `haulage_range_land`, the range of grain; a cargo worth more per ton, like
    smelted ore, is worth carrying further.

    *step_allowed*, given, is asked of every step in the direction the cargo travels —
    `step_allowed(from_hx, to_hx)` — and a step it refuses is not taken. That is how the
    river trade keeps to going downstream (`river_trade.river_trade_flows`); the cost of a
    step it allows is unchanged.
    """
    node_cost, edge_cost = make_bulk_cost(hexes, cfg, rivers)
    if budget is None:
        budget = cfg.haulage_range_land

    cost: dict[HexCoord, float] = {seat: 0.0 for seat in seats}
    toward: dict[HexCoord, HexCoord] = {}
    heap = [(0.0, seat) for seat in sorted(seats)]
    heapq.heapify(heap)
    while heap:
        d, coord = heapq.heappop(heap)
        if d > cost.get(coord, float("inf")):
            continue
        hx = hexes[coord]
        for n in neighbors(coord):
            n_hx = hexes.get(n)
            if n_hx is None:
                continue
            # The search expands outward from the seat, but the cargo travels the other
            # way — so each relaxation prices the step *n -> here*, toward the seat.
            # Getting the edge direction wrong does not fail loudly: slope is the only
            # asymmetric term, so it silently inflates the draw of every place the country
            # rises toward.
            if step_allowed is not None and not step_allowed(n_hx, hx):
                continue
            step = node_cost(n_hx) + edge_cost(n_hx, hx)
            if step == float("inf"):
                continue
            nd = d + step
            if nd < budget and nd < cost.get(n, float("inf")):
                cost[n] = nd
                toward[n] = coord
                heapq.heappush(heap, (nd, n))
    return cost, toward


def ford_cost(from_hx, to_hx, rivers: Rivers | None) -> float:
    """What it costs to get across a watercourse on foot.

    Rivers run along hexsides, so this is charged once, on the step across the side.
    Where `CrossingStage` or a road has put a ford or a bridge there, it is nearly nothing:
    the crossing exists and you walk over it.  Everywhere else it scales with the reach's
    span (`crossings.side_span`) and has no fixed term, because somebody walking to market
    pays no capital — a slack headwater is a step across and barely registers, a big or
    fast-running reach is most of a day.  That is what makes a river bound a catchment
    along its length while the district still reaches across at the one place it can.
    """
    if rivers is None:
        return 0.0
    return rivers.crossing.get(frozenset((from_hx.coord, to_hx.coord)), 0.0)


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


def allocate_catchments(hexes, seats, budget: float, cfg, rivers: Rivers | None = None):
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

    node_cost, edge_cost = make_travel_cost(hexes, cfg, rivers)

    owner: dict[HexCoord, HexCoord] = {}
    cost: dict[HexCoord, float] = {}
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


def fishery_rim(
    hexes, owner: dict[HexCoord, HexCoord], cost: dict[HexCoord, float]
) -> tuple[dict[HexCoord, HexCoord], dict[HexCoord, float]]:
    """Extend each catchment onto the water its land touches.

    A coastal settlement fishes, and `food_value` already scores open water for exactly
    that reason.  But letting the catchment walk *across* water would hand one settlement
    a whole sea, so the rim is granted rather than traversed: a claimed land hex donates
    its adjacent unclaimed water at its own cost, and the water goes no further.

    Each water hex goes to the *cheapest* adjacent claimant — the boats put out from the
    nearest shore that works the water, not from whichever claimant happens to sort
    first. It used to go to the lowest-coordinate neighbour, which handed fisheries to
    the wrong market wherever two catchments met at a shore. Ties break on the donor's
    coordinate, so the grant is deterministic whatever order the claims were laid.

    Returns fresh dicts; the inputs are not mutated.
    """
    out_owner = dict(owner)
    out_cost = dict(cost)

    best: dict[HexCoord, tuple[tuple[float, HexCoord], HexCoord]] = {}
    for coord in sorted(owner):
        claim = (cost[coord], coord)
        for n in neighbors(coord):
            if n in out_owner:
                continue
            n_hx = hexes.get(n)
            if n_hx is None or n_hx.terrain_class not in WATER:
                continue
            if n not in best or claim < best[n][0]:
                best[n] = (claim, owner[coord])

    for n, ((donor_cost, _), seat) in best.items():
        out_owner[n] = seat
        out_cost[n] = donor_cost

    return out_owner, out_cost


def gather(
    values: dict[HexCoord, float],
    owner: dict[HexCoord, HexCoord],
    cost: dict[HexCoord, float],
    range_limit: float,
) -> dict[HexCoord, float]:
    """Total haulage-weighted value each seat can draw from what it owns.

    The one arithmetic every tier shares: a village's production, a market's surplus draw,
    a city's bulk supply.  What changes between them is the range limit and what is being
    summed, never the shape of the sum.
    """
    totals: dict[HexCoord, float] = {}
    for coord, seat in owner.items():
        value = values.get(coord, 0.0)
        if value <= 0.0:
            continue
        weight = usable_fraction(cost[coord], range_limit)
        if weight <= 0.0:
            continue
        totals[seat] = totals.get(seat, 0.0) + value * weight
    return totals


def settleable(hexes, cfg) -> set[HexCoord]:
    """Hexes that could carry a settlement at all, before any scoring.

    The same exclusions `HabitabilityStage` scores to zero — you do not found a village on
    open water, a mountain face, or a bog — kept in one place so the two cannot drift.
    """
    from ..core.hex import Biome, is_steep

    return {
        coord
        for coord, hx in hexes.items()
        if hx.terrain_class not in WATER
        and not is_steep(hx, cfg.terrain_steep_gradient_m)
        and hx.biome is not Biome.WETLAND
    }
