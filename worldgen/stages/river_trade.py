"""Long-haul trade down the rivers: the interior's goods carried to the coast (tech-debt #125).

Provisioning is short-haul — a market feeds the city within `haulage_range_land` — and
manufactures between cities mostly take routes that avoid the falls. Neither carries the
trade the historical portage towns lived on: Aswan, Louisville, the fall-line towns of the
American east coast stood where the long trade *down* a river system had to land and load
again. So every town or city on a navigable reach and off the coast sends a share of its
worth downriver to the coast, and pays its way at every quay it changes mode at and every
portage it is carried round (tech-debt #142).

The routing is pure and kept apart from `CityPromotionStage`, which applies it and charges
each flow the quays and tolls on its path: every `RiverFlow` carries its whole path.
"""

from dataclasses import dataclass

from ..core.hex import HexCoord, SettlementTier, TerrainClass
from ..core.hex_grid import corner_hexes, hex_range, neighbors, side_corners
from ..core.world_state import WorldState
from .haulage import bulk_routes, usable_fraction
from .riverside import Rivers


@dataclass(frozen=True)
class RiverFlow:
    """One town's trade down its river: from *origin* to the coastal *dest*, along *path*.

    *path* runs origin to destination inclusive, hex by hex, so anything charged where a
    cargo passes — a quay, a portage, a toll — can be read off it. *people* is the worth
    of the cargo in the people it feeds, after the haul has eaten its share.
    """

    origin: HexCoord
    dest: HexCoord
    people: float
    path: tuple[HexCoord, ...]


def coastal(coord: HexCoord, hexes) -> bool:
    """On the sea or a step from it: where a river's trade is handed to ships."""
    return any(
        n in hexes and hexes[n].terrain_class is TerrainClass.OPEN_WATER
        for n in (coord, *neighbors(coord))
    )


def downstream_rank(state: WorldState, rivers: Rivers) -> dict[HexCoord, float]:
    """How far down its river system each hex lies, as the catchment draining past it.

    Catchment only grows downstream, so a cargo that never steps to a lower rank never
    goes upstream. A bank or a portage takes the largest catchment beside it; a lake takes
    the largest river touching it, which is its outlet — so a boat crosses a lake to the
    river leaving it but not up one feeding it; the sea is below everything. Ground off the
    rivers has no rank and is left out.
    """
    hexes = state.hexes
    rank: dict[HexCoord, float] = {
        h: c for h, c in rivers.beside.items() if h in rivers.reaches or h in rivers.portage
    }
    lake: dict[HexCoord, float] = {}
    for side, rs in state.river_sides.items():
        for corner in side_corners(side):
            for h in corner_hexes(corner):
                hx = hexes.get(h)
                if hx is not None and hx.terrain_class is TerrainClass.INLAND_WATER:
                    lake[h] = max(lake.get(h, 0.0), rs.catchment_km2)
    # A lake is one body: every hex of it ranks with its outlet.
    seen: set[HexCoord] = set()
    for start in sorted(lake):
        if start in seen:
            continue
        body, stack = [], [start]
        seen.add(start)
        while stack:
            h = stack.pop()
            body.append(h)
            for n in neighbors(h):
                n_hx = hexes.get(n)
                if (
                    n not in seen
                    and n_hx is not None
                    and n_hx.terrain_class is TerrainClass.INLAND_WATER
                ):
                    seen.add(n)
                    stack.append(n)
        top = max(lake.get(h, 0.0) for h in body)
        for h in body:
            rank[h] = top
    for h, hx in hexes.items():
        if hx.terrain_class is TerrainClass.OPEN_WATER:
            rank[h] = float("inf")
    return rank


def river_trade_flows(state: WorldState, cfg, rivers: Rivers) -> list[RiverFlow]:
    """Every inland river town's trade down to the coast, cheapest landfall first.

    Origins are the towns and cities on a navigable reach or a step from one, and off the
    coast; destinations are the towns and cities on it. One multi-source search from the
    coast over `make_bulk_cost`, within `haulage_range_land` × `river_trade_range_mult`,
    restricted to what a boat going downstream can do:

    *   afloat to afloat only on the same water (`Rivers.joined`) — no carrying a cargo
        across to a different river;
    *   onto and off a portage beside a cataract, and along it — the falls are walked
        round, but nothing else is;
    *   never to a lower `downstream_rank` — never upstream;
    *   from the origin onto the river beside it, and from the water onto the quay of the
        coastal town at the end.

    Each origin ships `river_trade_share` of its people's worth to the coastal town it
    reaches most cheaply, less what the haul eats (`usable_fraction`).
    """
    share = cfg.river_trade_share
    if share <= 0.0:
        return []
    hexes = state.hexes
    live = [s for s in state.settlements if s.tier in (SettlementTier.TOWN, SettlementTier.CITY)]
    coast = sorted(s.coord for s in live if coastal(s.coord, hexes))
    origins = sorted(
        s.coord
        for s in live
        if not coastal(s.coord, hexes) and any(n in rivers.reaches for n in hex_range(s.coord, 1))
    )
    if not coast or not origins:
        return []

    rank = downstream_rank(state, rivers)
    seats, starts = set(coast), set(origins)

    def allowed(from_hx, to_hx) -> bool:
        a, b = from_hx.coord, to_hx.coord
        if a not in rank:
            # Ashore: only an origin loading onto the river beside it.
            return a in starts and b in rank
        if b in seats:
            return True  # landfall at the coastal town
        if b not in rank or rank[b] < rank[a]:
            return False
        if rivers.afloat(from_hx) and rivers.afloat(to_hx):
            return rivers.joined(from_hx, to_hx)
        return True  # onto, along or off a portage

    budget = cfg.haulage_range_land * cfg.river_trade_range_mult
    cost, toward = bulk_routes(
        hexes, coast, cfg, budget=budget, rivers=rivers, step_allowed=allowed
    )

    by_coord = {s.coord: s for s in state.settlements}
    flows: list[RiverFlow] = []
    for origin in origins:
        if origin not in cost or origin in seats:
            continue
        path = [origin]
        while path[-1] not in seats:
            path.append(toward[path[-1]])
        people = by_coord[origin].population * share * usable_fraction(cost[origin], budget)
        if people > 0.0:
            flows.append(RiverFlow(origin, path[-1], people, tuple(path)))
    return flows


__all__ = ["RiverFlow", "coastal", "downstream_rank", "river_trade_flows"]
