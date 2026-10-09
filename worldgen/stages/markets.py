"""Market centres, planted where the most surplus can reach them in a day.

A market town exists because the countryside around it needs somewhere to sell a surplus
and buy what it does not grow, and because a person can get there, do business, and be
home before dark.  Bracton put the figure at 6 2/3 miles — a third of a twenty-mile day
out and a third back — and English market towns do cluster at 10-15 km.  That distance,
costed over real terrain, is the whole of what decides where these go.

Two things follow that the ranked-placement model could not express.

The count is emergent.  Nothing here says how many markets to make; planting stops when
the best remaining site is not worth a market.  Rich lowland carries them densely, poor
upland sparsely, and a mountainous map simply has fewer — where `target_town_count` would
have insisted on the same two dozen regardless.

The countryside is a surface, not a list of hamlets.  A market's draw is the surplus of
its catchment, and integrating over the food field gives the same number as enumerating
the ~900 hamlets a 128x128 map would historically hold, almost none of which would earn a
glyph.  So the peasantry is present in the arithmetic and absent from the settlement list.
"""

import heapq
import math

import numpy as np

from ..core import routing
from ..core.hex import HexCoord
from ..core.hex_grid import grade_reachable_count, hex_range, ring
from ..core.pipeline import GeneratorStage
from ..core.routing import CostField
from ..core.world_state import WorldState
from .habitability import potential_food, site_bonus
from .haulage import (
    allocate_catchments,
    fishery_rim,
    settleable,
    travel_field,
    usable_fraction,
)
from .riverside import WATER, river_index
from .road_cost import grade_is_under_cap

# Float slop when comparing a recomputed score against the heap's next-best.  Without it,
# a score that recomputes to bit-identical value can compare as stale against itself and
# the loop re-pushes forever.
_EPS = 1e-9


def day_reach(coord: HexCoord, radius: float, field: CostField) -> list[tuple[HexCoord, float]]:
    """`[(hex, weight)]` for every hex a catchment rooted at *coord* would draw on.

    The walk `allocate_catchments` makes from a single seat — a Dijkstra over the travel
    cost, stopping at *radius* — then the water beside that land at its cheapest donor's
    cost, as `fishery_rim` grants it. Each hex is weighted `usable_fraction(cost, radius)`,
    the share of its surplus that survives the haul, and hexes worth nothing are left out.
    *field* is `travel_field`'s arrays, built once per world. Travel makes water
    impassable, so the hexes it prices at `inf` to enter are the water the rim is cast on.
    """
    grid = field.grid
    cost, done, rim = routing.day_walk(
        grid.nbr, field.node, field.edge, field.node == np.inf, grid.index[coord], float(radius)
    )
    # Index order is coordinate order, so these come out sorted as the old lists were.
    land, wet = np.flatnonzero(done), np.flatnonzero(rim < np.inf)
    out = zip(
        grid.coords(land) + grid.coords(wet),
        cost[land].tolist() + rim[wet].tolist(),
        strict=True,
    )
    weighed = ((c, usable_fraction(v, radius)) for c, v in out)
    return [(c, w) for c, w in weighed if w > 0.0]


class MarketStage(GeneratorStage):
    """Plants market centres and gives each the catchment it can draw on.

    Siting only. `LandUseStage` founds and sizes them, because a market is worth what its
    countryside actually sends and nothing has been cleared yet at this point in the
    pipeline. Both halves read the same catchments — this stage writes them to
    `hex.territory` — so nothing has to be recomputed, and population has one owner rather
    than a provisional value that a later stage silently overwrites.

    Planting reads *potential* food, which is the honest surface for the question it asks:
    a settler picks land for what it will yield once worked, not for the wildwood standing
    on it today.
    """

    def run(self, state: WorldState) -> WorldState:
        hexes = state.hexes
        rivers = river_index(state, self.config)
        cfg = self.config

        surplus = {
            coord: potential_food(hx, cfg) * cfg.marketable_surplus_fraction
            for coord, hx in hexes.items()
        }

        seats = self._plant(hexes, surplus, cfg, rivers)
        if not seats:
            return state

        owner, cost = allocate_catchments(hexes, seats, cfg.market_day_radius, cfg, rivers)
        owner, cost = fishery_rim(hexes, owner, cost)
        for coord, seat in owner.items():
            hexes[coord].territory = seat
            hexes[coord].territory_cost = cost[coord]

        state.metadata["market_seats"] = sorted(seats)
        return state

    # -- planting -------------------------------------------------------------

    def _plant(self, hexes, surplus, cfg, rivers) -> list[HexCoord]:
        """Lazy-greedy siting against a depleting surplus surface, scored on the day-reach.

        A candidate is scored on exactly the ground a catchment rooted there would walk:
        a single-source Dijkstra over `make_travel_cost`, bounded by `market_day_radius` —
        the same walk `allocate_catchments` makes — plus the fishery rim on the terms
        `fishery_rim` grants it (each water hex beside reached land, at its cheapest
        donor's cost). Each hex counts at its haulage weight, `usable_fraction`, which is
        what the market will later be sized on; and planting depletes exactly what was
        scored. So a ridge or an estuary beside a candidate lowers its score, where the
        plain hex disc this replaced counted the far side as if it could be walked.

        Depletion only ever *reduces* a site's score, so lazy greedy is exact: an entry
        popped off the heap whose recomputed score still beats the next best is provably
        the true maximum. The heap is seeded with the plain ring disc as an upper bound —
        every step costs at least `road_flat_cost`, so a hex at ring d costs at least d
        steps' worth and weighs no more than a hex at that cost; a rim water hex sits one
        ring beyond its donor, hence one ring of slack — so the Dijkstra only runs on
        candidates that pop.
        """
        radius = cfg.market_day_radius
        # The cheapest a step can be: terrain charges `road_flat_cost` per hex entered,
        # and ascent and fording only ever add to it.
        step = cfg.road_flat_cost
        # Land reaches no further than ring ceil(radius / step) - 1; the rim one ring more.
        rings = math.ceil(radius / step) if step > 0.0 else 0
        offsets = [ring((0, 0), d) for d in range(rings + 1)]
        remaining = dict(surplus)
        field = travel_field(hexes, cfg, rivers)

        def weight(c: float) -> float:
            return usable_fraction(c, radius)

        reach_cache: dict[HexCoord, list[tuple[HexCoord, float]]] = {}

        def reach(coord):
            if coord not in reach_cache:
                reach_cache[coord] = day_reach(coord, radius, field)
            return reach_cache[coord]

        bonus_cache: dict[HexCoord, float] = {}

        def bonus(coord):
            if coord not in bonus_cache:
                bonus_cache[coord] = 1.0 + site_bonus(coord, hexes[coord], hexes, cfg, rivers)
            return bonus_cache[coord]

        def score(coord):
            total = 0.0
            for c, w in reach(coord):
                value = remaining.get(c)
                if value:
                    total += value * w
            return total * bonus(coord)

        def bound(coord):
            """The ring disc at the same weights: never below `score` (see docstring)."""
            if step <= 0.0:  # no floor on a step, so no disc bounds the reach
                return score(coord)
            q, r = coord
            total = 0.0
            for d, ring_offsets in enumerate(offsets):
                w_land = weight(d * step)
                w_water = weight(max(d - 1, 0) * step)
                for dq, dr in ring_offsets:
                    n = (q + dq, r + dr)
                    value = remaining.get(n)
                    if value:
                        w = w_water if hexes[n].terrain_class in WATER else w_land
                        total += value * w
            return total * bonus(coord)

        grade_cache: dict[HexCoord, int] = {}

        def reachable(coord):
            """Deferred to acceptance: it is the dear test, and most candidates never pop."""
            if coord not in grade_cache:
                grade_cache[coord] = grade_reachable_count(
                    coord,
                    hexes,
                    lambda a, b: grade_is_under_cap(a, b, cfg),
                    cfg.settlement_min_reachable,
                )
            return grade_cache[coord]

        # sorted() rather than dict order: heap ties break on the coord tuple, so the same
        # terrain always yields the same markets whatever order the hexes were built in.
        candidates = sorted(settleable(hexes, cfg))
        heap = [(-bound(c), c) for c in candidates]
        heapq.heapify(heap)

        seats: list[HexCoord] = []
        suppressed: set[HexCoord] = set()

        while heap:
            _, coord = heapq.heappop(heap)
            # Order matters: suppression, then recompute, then staleness, then the floor.
            # Testing the floor before the staleness check ends the loop on the first
            # suppressed hex popped, which truncates the map at the first market.
            if coord in suppressed:
                continue
            current = score(coord)
            if heap and current < -heap[0][0] - _EPS:
                heapq.heappush(heap, (-current, coord))
                continue
            if current < cfg.market_viability_floor:
                break
            if reachable(coord) < cfg.settlement_min_reachable:
                continue

            seats.append(coord)
            suppressed |= set(hex_range(coord, cfg.market_min_separation))
            for c, w in reach(coord):
                if c in remaining:
                    remaining[c] *= 1.0 - w

        return seats


__all__ = ["MarketStage", "day_reach", "usable_fraction"]
