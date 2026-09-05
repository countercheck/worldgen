"""Cities, promoted from the markets that can be fed from furthest away.

A market town is bounded by what a cart fetches in a day, which is why they come out at a
uniform size whatever the country is like: fertility decides how *many* there are, not how
big each one grows.  Nothing in that model can produce a city, because a city is not a
large market.  It is a place fed from beyond a day's reach.

What makes that possible is bulk haulage, and what makes bulk haulage possible is water.
Diocletian's Price Edict puts land carriage at roughly fifty-five times sea and eleven
times river for the same tonne-kilometre, so the range over which a place can be
provisioned is not a property of the place — it is a property of what lies around it.  A
town on a navigable river or a sheltered coast draws on fifteen times the reach of one the
same size inland, and that single multiplier is the whole of the difference.

So this stage founds nothing.  It asks of each market how much *other* markets' surplus
can reach it, promotes the ones that clear `city_min_draw`, and moves the surplus it
absorbs off the markets that sent it.  The size gap between a port and an inland town is
produced by one constant rather than by a rule that says ports are bigger.
"""

import heapq

from ..core.hex import SettlementTier
from ..core.hex_grid import neighbors
from ..core.pipeline import GeneratorStage
from ..core.world_state import WorldState
from .habitability import actual_food
from .haulage import gather, make_bulk_cost, usable_fraction


class CityPromotionStage(GeneratorStage):
    """Promote markets that can be provisioned from beyond a day's reach."""

    def run(self, state: WorldState) -> WorldState:
        hexes = state.hexes
        cfg = self.config

        markets = [s for s in state.settlements if s.tier is SettlementTier.TOWN]
        if len(markets) < 2:
            return state

        draw = self._market_draw(hexes, cfg)
        reach = {s.coord: self._bulk_reach(hexes, s.coord, cfg) for s in markets}

        promoted, absorbed = self._promote(markets, draw, reach, cfg)
        if not promoted:
            return state

        self._resize(state, markets, absorbed, promoted, cfg)
        return state

    # -- what each market already gathers -------------------------------------

    @staticmethod
    def _market_draw(hexes, cfg) -> dict:
        """Each market's day-range surplus, read back off the territory it was given.

        `MarketStage` wrote `territory` and `territory_cost` onto every hex it claimed, so
        the catchments do not need walking again — this is `gather` over what is already
        recorded, and it agrees with the number that set each market's population.
        """
        surplus = {
            coord: actual_food(hx, cfg) * cfg.marketable_surplus_fraction
            for coord, hx in hexes.items()
        }
        owner = {c: hx.territory for c, hx in hexes.items() if hx.territory is not None}
        cost = {c: hexes[c].territory_cost for c in owner}
        return gather(surplus, owner, cost, cfg.market_day_radius)

    # -- how far bulk can come ------------------------------------------------

    @staticmethod
    def _bulk_reach(hexes, seat, cfg) -> dict:
        """Cost of hauling bulk to *seat* from anywhere within `haulage_range_land`.

        A second Dijkstra, over `make_bulk_cost` rather than `make_travel_cost`. That
        distinction is the whole stage: travel cost makes water impassable, because a
        catchment is ground somebody works, while a cargo goes by ship. Using the wrong one
        here does not fail loudly — it silently makes every city inland, because the reach
        then measures nothing but how central a market is on land.

        Returns cost keyed by coord, over hexes within budget.
        """
        node_cost, edge_cost = make_bulk_cost(hexes, cfg)
        budget = cfg.haulage_range_land

        cost: dict = {seat: 0.0}
        heap = [(0.0, seat)]
        while heap:
            d, coord = heapq.heappop(heap)
            if d > cost.get(coord, float("inf")):
                continue
            hx = hexes[coord]
            for n in neighbors(coord):
                n_hx = hexes.get(n)
                if n_hx is None:
                    continue
                # The search expands outward from the seat, but the cargo travels the
                # other way — so each relaxation prices the step *n -> here*, toward the
                # seat. Getting the edge direction wrong does not fail loudly: slope is
                # the only asymmetric term, so it silently inflates the draw of every
                # market the country rises toward.
                step = node_cost(n_hx) + edge_cost(n_hx, hx)
                if step == float("inf"):
                    continue
                nd = d + step
                if nd < budget and nd < cost.get(n, float("inf")):
                    cost[n] = nd
                    heapq.heappush(heap, (nd, n))
        return cost

    # -- promotion ------------------------------------------------------------

    def _promote(self, markets, draw, reach, cfg):
        """Greedily promote the market that can draw most, then move that surplus off.

        Suppression by depletion rather than by a separation disc, exactly as the markets
        themselves are planted: a promoted city takes a distance-weighted share of what
        each market it reaches can send, and what is left is what a second city would find.
        Two ports on one estuary therefore cannot both count the same hinterland, and the
        second is smaller for it rather than being forbidden.
        """
        seats = sorted(s.coord for s in markets)
        remaining = dict(draw)
        promoted: list = []
        absorbed: dict = {}

        while True:
            best, best_take = None, None
            for seat in seats:
                if seat in absorbed:
                    continue
                take = {
                    other: remaining.get(other, 0.0)
                    * usable_fraction(reach[seat][other], cfg.haulage_range_land)
                    for other in seats
                    if other != seat and other in reach[seat]
                }
                total = sum(take.values())
                if best_take is None or total > sum(best_take.values()):
                    best, best_take = seat, take

            if best is None or sum(best_take.values()) < cfg.city_min_draw:
                break

            promoted.append(best)
            absorbed[best] = best_take
            for other, taken in best_take.items():
                remaining[other] = max(0.0, remaining.get(other, 0.0) - taken)

        return promoted, absorbed

    # -- sizing ---------------------------------------------------------------

    def _resize(self, state, markets, absorbed, promoted, cfg):
        """Move the absorbed surplus onto the cities and off the markets that sent it.

        Conserved, deliberately. A city is not new food; it is the same countryside
        feeding a different place, so the map's total population barely moves while its
        distribution changes completely. A market in a city's shadow shrinks by what it
        sends, which is why a great port has quiet towns around it rather than peers.

        Applied as a *delta* on the population each settlement already has, not a
        recomputation from the draw, for two reasons that are really one. A settlement
        promotion never touched must come out of this stage byte-identical — recomputing
        rewrote every market from the un-jittered draw, silently stripping the founding
        jitter off the whole tier. And a promoted seat may itself have been drawn on by a
        city promoted before it, so its own draw is not all still its own: charging every
        seat exactly what was taken from it is what makes the books balance instead of
        counting the overlap twice.
        """
        by_coord = {s.coord: s for s in markets}
        lost: dict = {}
        for take in absorbed.values():
            for other, taken in take.items():
                lost[other] = lost.get(other, 0.0) + taken

        for coord, settlement in by_coord.items():
            delta = -lost.get(coord, 0.0)
            if coord in absorbed:
                delta += sum(absorbed[coord].values())
                settlement.tier = SettlementTier.CITY
                settlement.name = settlement.name.replace("_market_", "_city_")
            if delta:
                settlement.population = max(
                    1, round(settlement.population + delta * cfg.people_per_food)
                )

        state.metadata["cities"] = sorted(promoted)


__all__ = ["CityPromotionStage"]
