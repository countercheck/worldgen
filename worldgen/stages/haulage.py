"""Pre-industrial transport economics: how far goods can move before they stop being worth
moving, and who can therefore reach what.

The target period is pre-rail and pre-motor, and in that world the binding constraint on
where people live and how large a place can grow is the cost of shifting bulk goods —
chiefly grain.  A draught team eats as it walks, so a cargo hauled far enough overland is
consumed by its own transport before it arrives; water carriage is roughly an order of
magnitude cheaper per tonne-kilometre, which is why the large pre-industrial cities sit on
navigable water and inland ones stay small.

Every range here is a travel-cost budget rather than a distance, so terrain shortens it
automatically: with `road_mountain_cost` ten times `road_flat_cost`, a two-and-a-half unit
field budget dies on the first mountain hex and runs freely down a valley.  Catchments come
out valley-shaped with ridges as boundaries, which is the point of costing them this way
instead of drawing discs.

Pure functions and one Dijkstra, in the shape of `road_cost.py`: no stage class, no world
mutation, so it can be unit-tested on synthetic grids.
"""

import heapq

from ..core.hex import TerrainClass
from ..core.hex_grid import neighbors
from .road_cost import (
    WATER,
    is_river,
    make_road_edge_cost,
    river_edges,
    river_hex_cost,
    terrain_base_cost,
)


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

    Headwaters are not navigable at any period; the flow threshold is what separates a
    stream you ford from a river you ship grain down.
    """
    if hx.terrain_class in WATER:
        return True
    return is_river(hx) and hx.river_flow >= cfg.navigable_river_flow


def haulage_range(hx, cfg) -> float:
    """The distance bulk goods can travel from *hx* before they are worth nothing.

    Water multiplies it.  Diocletian's Price Edict prices land carriage at roughly 55x sea
    and 11x river for the same tonne-kilometre, so the multiplier — not the absolute land
    range — is the well-attested half of this pair.
    """
    if navigable(hx, cfg):
        return cfg.haulage_range_land * cfg.haulage_range_water_mult
    return cfg.haulage_range_land


def make_travel_cost(hexes, cfg, rivers=None, exempt=frozenset()):
    """Node and edge cost closures for moving goods overland.

    Reuses the road cost primitives verbatim so that a route a cart would take and a route
    a road would take are priced by the same rules.  Two deliberate omissions:

    `bank_discount` is left out.  It exists to pull *roads* onto riverbanks so the side a
    road runs on stays readable, which has nothing to do with how far a farmer walks to a
    field, and applying it here would make catchments hug rivers for a rendering reason.

    Water is impassable rather than cheap.  Roads cross water because a road is a route;
    a catchment is ground somebody works, and leaving the sea traversable at
    `road_water_cost` would let one coastal settlement claim an entire strait.  Fishing is
    handled by `fishery_rim` instead, which is bounded by the land that does the fishing.
    """
    blocked = river_edges(rivers) if rivers else None
    road_edge = make_road_edge_cost(cfg, blocked, exempt)

    def node_cost(hx) -> float:
        if hx.terrain_class in WATER:
            return float("inf")
        return terrain_base_cost(hx, cfg) + river_hex_cost(hx, cfg)

    def edge_cost(from_hx, to_hx) -> float:
        if to_hx.terrain_class in WATER or from_hx.terrain_class in WATER:
            return float("inf")
        return road_edge(from_hx, to_hx)

    return node_cost, edge_cost


def allocate_catchments(hexes, seats, budget: float, cfg, rivers=None):
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

    node_cost, edge_cost = make_travel_cost(hexes, cfg, rivers, exempt=frozenset(seats))

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
    from ..core.hex import Biome

    return {
        coord
        for coord, hx in hexes.items()
        if hx.terrain_class not in WATER
        and hx.terrain_class != TerrainClass.MOUNTAIN
        and hx.biome is not Biome.WETLAND
    }
