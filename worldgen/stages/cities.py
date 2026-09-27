"""Cities, promoted from the markets that can be fed from furthest away.

A market town is bounded by what a cart fetches in a day, which is why they come out at a
uniform size whatever the country is like: fertility decides how *many* there are, not how
big each one grows.  Nothing in that model can produce a city, because a city is not a
large market.  It is a place fed from beyond a day's reach.

What makes that possible is bulk haulage, and what makes bulk haulage possible is water.
Diocletian's Price Edict puts land carriage at 28-56 times sea and 6-11 times river for
the same tonne-kilometre, so the range over which a place can be provisioned is not a
property of the place — it is a property of what lies around it.  A town on a navigable
river or a sheltered coast draws on fifteen times the reach of one the same size inland,
and that single multiplier is the whole of the difference.

So this stage founds nothing.  It asks of each market how much *other* markets' surplus
can reach it, promotes the ones that clear `city_min_draw`, and moves the surplus it
absorbs off the markets that sent it.  The size gap between a port and an inland town is
produced by one constant rather than by a rule that says ports are bigger.
"""

from ..core.hex import SettlementTier, TerrainClass
from ..core.hex_grid import distance, hex_range
from ..core.pipeline import GeneratorStage
from ..core.world_state import WorldState
from .habitability import actual_food
from .haulage import bulk_routes, gather, navigable, usable_fraction

# How a cargo travels on a hex, ranked so the quay of a change is the lower-ranked side:
# the land hex where a cart meets a boat, or the river hex where a barge meets a ship.
_LAND, _INLAND_WATER, _SEA = 0, 1, 2


def _mode(hx, cfg) -> int:
    if hx.terrain_class is TerrainClass.OPEN_WATER:
        return _SEA
    return _INLAND_WATER if navigable(hx, cfg) else _LAND


class CityPromotionStage(GeneratorStage):
    """Promote markets that can be provisioned from beyond a day's reach."""

    def run(self, state: WorldState) -> WorldState:
        hexes = state.hexes
        cfg = self.config

        markets = [s for s in state.settlements if s.tier is SettlementTier.TOWN]
        if len(markets) < 2:
            return state

        draw = self._market_draw(hexes, cfg)
        routes = {s.coord: self._bulk_routes(hexes, s.coord, cfg) for s in markets}
        reach = {coord: cost for coord, (cost, _) in routes.items()}

        promoted, _ = self._promote(markets, draw, reach, cfg)
        if promoted:
            # Promotion decides which markets are cities; how big each grows is decided by
            # where the countryside's surplus actually goes, which is the allocation.
            absorbed = self._allocate(markets, promoted, draw, reach, cfg)
            toward = {coord: way for coord, (_, way) in routes.items()}
            self._resize(state, markets, absorbed, draw, promoted, cfg, toward)
            self._trade(state, cfg)

        # The second road to a city: a town grown past `city_min_population`, which after
        # transshipment is an entrepôt that feeds nobody's hinterland but handles everyone's.
        if cfg.city_min_population > 0:
            for s in markets:
                if s.tier is SettlementTier.TOWN and s.population >= cfg.city_min_population:
                    s.tier = SettlementTier.CITY
                    s.name = s.name.replace("_market_", "_city_")
                    promoted.append(s.coord)
        if promoted:
            state.metadata["cities"] = sorted(promoted)
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
        """Cost of hauling bulk to *seat* from anywhere within `haulage_range_land`."""
        return CityPromotionStage._bulk_routes(hexes, seat, cfg)[0]

    @staticmethod
    def _bulk_routes(hexes, seat, cfg) -> tuple[dict, dict]:
        """`bulk_routes` to one seat: the cost of hauling bulk there, and the way."""
        return bulk_routes(hexes, [seat], cfg)

    # -- transshipment --------------------------------------------------------

    @staticmethod
    def _break_points(source, seat, toward, hexes, cfg) -> list:
        """The quays a cargo from *source* to *seat* crosses: every change in how it travels.

        Cart to boat, boat to cart, and barge to ship where a navigable river meets the sea.
        The quay is the lower-ranked side of the change — the land hex, or the river hex at
        the mouth — because that is where the warehouses and the porters stand.
        """
        quays = []
        here = source
        while here != seat:
            nxt = toward.get(here)
            if nxt is None:
                break
            a, b = _mode(hexes[here], cfg), _mode(hexes[nxt], cfg)
            if a != b:
                quays.append(here if a < b else nxt)
            here = nxt
        return quays

    @staticmethod
    def _handler(quay, seats, radius):
        """The settlement that handles cargo at *quay*: the nearest within *radius*, if any."""
        near = [(distance(quay, c), c) for c in hex_range(quay, radius) if c in seats]
        return min(near)[1] if near else None

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
            best, best_take, best_total = None, None, -1.0
            for seat in seats:
                if seat in absorbed:
                    continue
                offered = {
                    other: remaining.get(other, 0.0)
                    * usable_fraction(reach[seat][other], cfg.haulage_range_land)
                    for other in seats
                    if other != seat and other in reach[seat]
                }
                total = sum(offered.values())
                if total > best_total:
                    best, best_total = seat, total
                    best_take = self._nearest_share(offered, reach[seat], total, cfg)

            if best is None or best_total < cfg.city_min_draw:
                break

            promoted.append(best)
            absorbed[best] = best_take
            for other, taken in best_take.items():
                remaining[other] = max(0.0, remaining.get(other, 0.0) - taken)

        return promoted, absorbed

    @staticmethod
    def _allocate(markets, promoted, draw, reach, cfg) -> dict:
        """Where each market's shipped surplus goes: split between the cities by their pull.

        Grain went where it fetched most after carriage, not to the nearest buyer. A big city
        has more buyers and pays more, so it outbids a small one even from further off —
        London drew on Norfolk and Yorkshire past towns that were nearer the farms — while
        carriage still means the nearest city takes most. So a city's pull on a market is
        its size times `usable_fraction` of the haul, and the market's shipment is split in
        proportion to pull raised to `city_pull_sharpness`: very high sends everything to the
        best-paying city, 0 splits it evenly, and in between the nearest city takes most with
        a tail going to the great city further off.

        A market ships `city_draw_share` of its surplus, and each city receives its share of
        that times the part that survives the haul. Cities themselves ship nothing: they are
        where the food goes.

        Pull follows the size a city grows to, and that size follows the pull — which is how
        a capital comes to dominate. Measured on founding sizes alone every candidate is a
        market of a few hundred, size separates nothing, and distance decides it all: the
        largest city came out barely bigger than the second. So the split is worked out
        `city_pull_rounds` times, each round pulling with the sizes the last one produced.
        """
        cities = sorted(set(promoted))
        start = {s.coord: s.population for s in markets}
        survives = {
            m: {
                c: usable_fraction(reach[c][m], cfg.haulage_range_land)
                for c in cities
                if m in reach[c] and reach[c][m] < cfg.haulage_range_land
            }
            for m in sorted(start)
            if m not in cities and draw.get(m, 0.0) > 0.0
        }

        size = {c: float(start[c]) for c in cities}
        absorbed: dict = {}
        for _ in range(max(1, cfg.city_pull_rounds)):
            absorbed = {c: {} for c in cities}
            for m, offers in survives.items():
                pull = {c: (f * size[c]) ** cfg.city_pull_sharpness for c, f in offers.items()}
                total = sum(pull.values())
                if total <= 0.0:
                    continue
                shipped = draw[m] * cfg.city_draw_share
                for c, p in pull.items():
                    taken = shipped * (p / total) * offers[c]
                    if taken > 0.0:
                        absorbed[c][m] = taken
            # The people who would follow that food, as `_resize` will move them.
            size = {
                c: start[c] + sum(start[m] * min(1.0, t / draw[m]) for m, t in absorbed[c].items())
                for c in cities
            }
        return absorbed

    @staticmethod
    def _nearest_share(offered, cost, total, cfg) -> dict:
        """What a city actually takes of what is *offered*: `city_draw_share` of it, nearest first.

        Promotion is judged on everything that can reach a place; the take is capped. A city
        that took all of it drained the half of the map its water reach covers, and every
        city after the first was sized on leftovers. Filling from the cheapest source first
        means the markets at the gates feed the city outright and the far ones keep their
        surplus for a city of their own.
        """
        budget = total * cfg.city_draw_share
        take: dict = {}
        for other in sorted(offered, key=lambda o: (cost[o], o)):
            if budget <= 0.0:
                break
            amount = min(offered[other], budget)
            take[other] = amount
            budget -= amount
        return take

    # -- sizing ---------------------------------------------------------------

    def _resize(self, state, markets, absorbed, draw, promoted, cfg, toward=None):
        """Move the absorbed surplus onto the cities and off the markets that sent it.

        Conserved, deliberately. A city is not new food; it is the same countryside
        feeding a different place, so the map's total population barely moves while its
        distribution changes completely. A market in a city's shadow shrinks by what it
        sends, which is why a great port has quiet towns around it rather than peers.

        The transfer is expressed as a **share of the population each settlement actually
        has**, not as the food figure converted afresh. Those are not the same quantity:
        founding sizes carry a jitter, so a market's population is its draw times
        `people_per_food` times something either side of one, and charging it the
        un-jittered `taken * people_per_food` bills a market that jittered *down* for more
        people than live there. It stayed invisible while no city took more than about
        four fifths of any market's draw — the shortfall was smaller than the jitter — and
        appeared the moment bulk reach grew: one market of 511 was charged 517 and landed
        on -6, which the floor then showed as a town of one person.

        Working in population throughout fixes it by construction rather than by widening
        the floor. A settlement can never lose more than it has, because the shares taken
        from any one market sum to at most its whole draw; the jitter survives in
        proportion, a market that ships half its surplus keeping half of whatever it was
        founded with; and the books balance exactly, since every city is credited the same
        figures its sources are debited.

        Applied as a *delta* on the population each settlement already has, not a
        recomputation from the draw, for two reasons that are really one. A settlement
        promotion never touched must come out of this stage byte-identical — recomputing
        rewrote every market from the un-jittered draw, silently stripping the founding
        jitter off the whole tier. And a promoted seat may itself have been drawn on by a
        city promoted before it, so its own draw is not all still its own: charging every
        seat exactly what was taken from it is what makes the books balance instead of
        counting the overlap twice.

        Given *toward*, each city's routes home, every cargo also pays its way through the
        ports it changes mode at on the way: `transship_share` of the people it feeds stay
        with the settlement handling each quay between its source and the city, off the
        city's gain. Still conserved — the handlers
        eat out of the cargo they handle — so an entrepôt grows on trade that feeds somebody
        else, which is how a river mouth can outgrow the hinterland it stands in.
        """
        by_coord = {s.coord: s for s in markets}
        start = {coord: s.population for coord, s in by_coord.items()}

        def moved(source: tuple, taken: float) -> float:
            """The people who follow *taken* worth of surplus off *source*."""
            total = draw.get(source, 0.0)
            if total <= 0.0 or source not in start:
                return 0.0
            return start[source] * min(1.0, taken / total)

        share = cfg.transship_share if toward is not None else 0.0
        handled: dict = {}
        # Every flow, as [origin q, r, destination q, r, people it feeds, kind], for the
        # road stage to carry: freight wears roads as travellers do.
        freight: list = []
        # Quays nobody stands at, keyed (quay, seat): the cargo through each and the people
        # it would support. The city keeps them for now; `ResourceStage` founds a port on
        # the busiest and moves them there.
        unhandled: dict = {}

        lost: dict = {}
        gained: dict = {}
        for seat, take in absorbed.items():
            for other, taken in take.items():
                people = moved(other, taken)
                lost[other] = lost.get(other, 0.0) + people
                gained[seat] = gained.get(seat, 0.0) + people
                if toward is not None and people > 0.0:
                    freight.append([*other, *seat, round(people, 3), "food"])
                if share <= 0.0:
                    continue
                paid = 0.0
                for quay in self._break_points(other, seat, toward[seat], state.hexes, cfg):
                    handler = self._handler(quay, by_coord, cfg.transship_radius)
                    # Loading at its own market is part of what the market already is, and
                    # unloading at the city is part of the city: only the places between
                    # earn a living off the trade.
                    if handler in (seat, other):
                        continue
                    cut = min(people * share, people - paid)
                    if handler is None:
                        food, kept = unhandled.get((quay, seat), (0.0, 0.0))
                        unhandled[(quay, seat)] = (food + taken, kept + cut)
                        paid += cut
                        continue
                    gained[handler] = gained.get(handler, 0.0) + cut
                    gained[seat] -= cut
                    paid += cut
                    handled[handler] = handled.get(handler, 0.0) + taken

        for coord, settlement in by_coord.items():
            if coord in absorbed:
                settlement.tier = SettlementTier.CITY
                settlement.name = settlement.name.replace("_market_", "_city_")
            delta = gained.get(coord, 0.0) - lost.get(coord, 0.0)
            if delta:
                settlement.population = max(1, round(settlement.population + delta))

        state.metadata["cities"] = sorted(promoted)
        if handled:
            state.metadata["transshipment"] = [
                [q, r, round(food, 3)] for (q, r), food in sorted(handled.items())
            ]
        if unhandled:
            state.metadata["unhandled_quays"] = [
                [q, r, sq, sr, round(food, 3), round(people, 3)]
                for ((q, r), (sq, sr)), (food, people) in sorted(unhandled.items())
            ]
        if freight:
            state.metadata["freight"] = freight

    # -- manufactured trade ---------------------------------------------------

    def _trade(self, state, cfg) -> None:
        """Manufactured goods between the cities, paying their way through every quay.

        Raw goods run from the countryside to a city; what the cities make runs between
        them, and that is the trade the great entrepôts and portage towns lived on — the
        provisioning alone is too short-haul to pass through much of anything. Each city
        puts `manufactured_trade_share` of its people's worth into trade, split between the
        cities it can reach by their size times the share of a cargo that survives the
        haul. Manufactures are worth more per ton than grain, so they go
        `manufactured_range_mult` times as far.

        Each shipment follows its bulk route and pays `transship_share` of itself at every
        change of mode on the way, as provisioning does, to whoever stands at the quay —
        drawn half from each of the two cities, so the books still balance. Quays nobody
        stands at are left for `ResourceStage` to found a port on, as provisioning's are.
        """
        if cfg.manufactured_trade_share <= 0.0:
            return
        hexes = state.hexes
        cities = sorted(s.coord for s in state.settlements if s.tier is SettlementTier.CITY)
        if len(cities) < 2:
            return
        by_coord = {s.coord: s for s in state.settlements}
        pop = {c: by_coord[c].population for c in cities}
        budget = cfg.haulage_range_land * cfg.manufactured_range_mult
        routes = {c: bulk_routes(hexes, [c], cfg, budget=budget) for c in cities}

        handled = {(q, r): f for q, r, f in state.metadata.get("transshipment", [])}
        unhandled: dict = {}
        for q, r, sq, sr, food, people in state.metadata.get("unhandled_quays", []):
            unhandled[((q, r), (sq, sr))] = (food, people)
        freight = state.metadata.setdefault("freight", [])
        delta: dict = {}

        for origin in cities:
            weight = {}
            for dest in cities:
                cost = routes[dest][0].get(origin)
                if dest != origin and cost is not None:
                    weight[dest] = pop[dest] * usable_fraction(cost, budget)
            weight = {d: w for d, w in weight.items() if w > 0.0}
            total = sum(weight.values())
            if total <= 0.0:
                continue
            out = pop[origin] * cfg.manufactured_trade_share
            for dest, w in sorted(weight.items()):
                volume = out * w / total
                freight.append([*origin, *dest, round(volume, 3), "goods"])
                food = volume / cfg.people_per_food
                paid = 0.0
                quays = self._break_points(origin, dest, routes[dest][1], hexes, cfg)
                for quay in quays:
                    handler = self._handler(quay, by_coord, cfg.transship_radius)
                    if handler in (origin, dest):
                        continue
                    cut = min(volume * cfg.transship_share, volume - paid)
                    paid += cut
                    if handler is None:
                        for seat in (origin, dest):
                            f, p = unhandled.get((quay, seat), (0.0, 0.0))
                            unhandled[(quay, seat)] = (f + food / 2, p + cut / 2)
                        continue
                    delta[handler] = delta.get(handler, 0.0) + cut
                    delta[origin] = delta.get(origin, 0.0) - cut / 2
                    delta[dest] = delta.get(dest, 0.0) - cut / 2
                    handled[handler] = handled.get(handler, 0.0) + food

        for coord, d in delta.items():
            s = by_coord[coord]
            s.population = max(1, round(s.population + d))
        if handled:
            state.metadata["transshipment"] = [
                [q, r, round(f, 3)] for (q, r), f in sorted(handled.items())
            ]
        if unhandled:
            state.metadata["unhandled_quays"] = [
                [q, r, sq, sr, round(f, 3), round(p, 3)]
                for ((q, r), (sq, sr)), (f, p) in sorted(unhandled.items())
            ]


__all__ = ["CityPromotionStage"]
