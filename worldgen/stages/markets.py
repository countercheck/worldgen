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

from ..core.hex_grid import grade_reachable_count, hex_range, ring
from ..core.pipeline import GeneratorStage
from ..core.world_state import WorldState
from .habitability import potential_food, site_bonus
from .haulage import allocate_catchments, fishery_rim, settleable, usable_fraction
from .road_cost import grade_is_under_cap

# Float slop when comparing a recomputed score against the heap's next-best.  Without it,
# a score that recomputes to bit-identical value can compare as stale against itself and
# the loop re-pushes forever.
_EPS = 1e-9


def depletion_kernel(radius: float, decay: float) -> list[tuple[int, float]]:
    """Offsets and weights for the share of surplus a market takes at each distance.

    `1 / (1 + d/decay)` rather than a hard claim over a disc.  A market that took all the
    surplus it reached would fix spacing at exactly one radius, which is `city_min_separation`
    again wearing a different hat.  Taking a decaying *share* leaves enough at the margin
    for a second market where the country is rich and nothing where it is poor, so spacing
    grades with the land instead of being a constant.

    Returned as `(ring_index, share)` so the caller walks precomputed ring offsets once.
    """
    return [(d, 1.0 / (1.0 + d / decay)) for d in range(int(radius) + 1)]


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
        cfg = self.config

        surplus = {
            coord: potential_food(hx, cfg) * cfg.marketable_surplus_fraction
            for coord, hx in hexes.items()
        }

        seats = self._plant(hexes, surplus, cfg)
        if not seats:
            return state

        owner, cost = allocate_catchments(hexes, seats, cfg.market_day_radius, cfg)
        owner, cost = fishery_rim(hexes, owner, cost)
        for coord, seat in owner.items():
            hexes[coord].territory = seat
            hexes[coord].territory_cost = cost[coord]

        state.metadata["market_seats"] = sorted(seats)
        return state

    # -- planting -------------------------------------------------------------

    def _plant(self, hexes, surplus, cfg) -> list:
        """Lazy-greedy siting against a depleting surplus surface.

        Depletion only ever *reduces* a site's score, so the score function is monotone
        non-increasing and lazy greedy is exact, not approximate: an entry popped off the
        heap whose recomputed score still beats the next best is provably the true maximum.
        Without that, siting means rescoring every candidate after every plant.
        """
        kernel = depletion_kernel(cfg.market_day_radius, cfg.market_kernel_decay)
        offsets = [ring((0, 0), d) for d, _ in kernel]
        remaining = dict(surplus)

        def score(coord):
            q, r = coord
            total = 0.0
            for (_, share), ring_offsets in zip(kernel, offsets, strict=True):
                for dq, dr in ring_offsets:
                    value = remaining.get((q + dq, r + dr))
                    if value:
                        total += value * share
            return total * (1.0 + site_bonus(coord, hexes[coord], hexes, cfg))

        reach_cache: dict = {}

        def reachable(coord):
            """Deferred to acceptance: it is the dear test, and most candidates never pop."""
            if coord not in reach_cache:
                reach_cache[coord] = grade_reachable_count(
                    coord,
                    hexes,
                    lambda a, b: grade_is_under_cap(a, b, cfg),
                    cfg.settlement_min_reachable,
                )
            return reach_cache[coord]

        # sorted() rather than dict order: heap ties break on the coord tuple, so the same
        # terrain always yields the same markets whatever order the hexes were built in.
        candidates = sorted(settleable(hexes, cfg))
        heap = [(-score(c), c) for c in candidates]
        heapq.heapify(heap)

        seats: list = []
        suppressed: set = set()

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
            q, r = coord
            for (_, share), ring_offsets in zip(kernel, offsets, strict=True):
                for dq, dr in ring_offsets:
                    n = (q + dq, r + dr)
                    if n in remaining:
                        remaining[n] *= 1.0 - share

        return seats


__all__ = ["MarketStage", "depletion_kernel", "usable_fraction"]
