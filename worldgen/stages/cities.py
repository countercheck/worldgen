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

from typing import Any, NamedTuple

from ..core.hex import SOIL_RANK, HexCoord, SettlementTier, TerrainClass
from ..core.hex_grid import distance, hex_range
from ..core.pipeline import GeneratorStage
from ..core.world_state import WorldState
from .habitability import actual_food
from .haulage import bulk_routes, gather, navigable, usable_fraction
from .river_trade import river_trade_flows
from .riverside import WATER, river_index

# How a cargo travels on a hex: by cart, by barge on a river or a lake, or by ship.
_LAND, _INLAND_WATER, _SEA = 0, 1, 2

# What a cargo pays for on its way (tech-debt #142). A quay is a change of mode; a bridge
# is water too big to wade with one way over it; a portage is the walk round a cataract.
# Further kinds — a pass, a desert crossing and its watering stops, a strait — slot in as
# another kind with its own share and its own test in `_charges`, and nothing that pays a
# charge or founds a town on an unpaid one needs to change.
QUAY, BRIDGE, PORTAGE = "quay", "bridge", "portage"


class Charge(NamedTuple):
    """One place a cargo pays on its way: *share* of it, to whoever stands within *radius*
    of *site*. A *landing* is a quay, where the cargo is handled rather than tolled; one
    nobody stands at is left for a port, where a toll is left for a toll town."""

    site: HexCoord
    kind: str
    share: float
    radius: int
    landing: bool = False


def _mode(hx, cfg, rivers) -> int:
    if hx.terrain_class is TerrainClass.OPEN_WATER:
        return _SEA
    return _INLAND_WATER if navigable(hx, cfg, rivers) else _LAND


class CityPromotionStage(GeneratorStage):
    """Promote markets that can be provisioned from beyond a day's reach."""

    def run(self, state: WorldState) -> WorldState:
        hexes = state.hexes
        cfg = self.config

        markets = [s for s in state.settlements if s.tier is SettlementTier.TOWN]
        if len(markets) < 2:
            return state

        draw = self._market_draw(hexes, cfg)
        rivers = river_index(state, cfg)
        routes = {s.coord: self._bulk_routes(hexes, s.coord, cfg, rivers) for s in markets}
        reach = {coord: cost for coord, (cost, _) in routes.items()}

        promoted, _ = self._promote(markets, draw, reach, cfg)
        if promoted:
            # Promotion decides which markets are cities; how big each grows is decided by
            # where the countryside's surplus actually goes, which is the allocation.
            absorbed = self._allocate(markets, promoted, draw, reach, cfg)
            toward = {coord: way for coord, (_, way) in routes.items()}
            self._resize(state, markets, absorbed, draw, promoted, cfg, toward)
            self._trade(state, cfg)
        self._river_trade(state, cfg)

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
    def _market_draw(hexes, cfg) -> dict[HexCoord, float]:
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
    def _bulk_reach(hexes, seat, cfg, rivers) -> dict[HexCoord, float]:
        """Cost of hauling bulk to *seat* from anywhere within `haulage_range_land`."""
        return CityPromotionStage._bulk_routes(hexes, seat, cfg, rivers)[0]

    @staticmethod
    def _bulk_routes(
        hexes, seat, cfg, rivers
    ) -> tuple[dict[HexCoord, float], dict[HexCoord, HexCoord]]:
        """`bulk_routes` to one seat: the cost of hauling bulk there, and the way."""
        return bulk_routes(hexes, [seat], cfg, rivers=rivers)

    # -- transshipment --------------------------------------------------------

    @staticmethod
    def _step_quays(here, nxt, hexes, cfg, rivers) -> list[HexCoord]:
        """The quays on one step of a route: where the cargo changes how it travels."""
        a_hx, b_hx = hexes[here], hexes[nxt]
        a, b = _mode(a_hx, cfg, rivers), _mode(b_hx, cfg, rivers)
        if a != b:
            a_wet, b_wet = a_hx.terrain_class in WATER, b_hx.terrain_class in WATER
            if a_wet and b_wet:
                return [here if a < b else nxt]
            if a_wet or b_wet:
                return [nxt if a_wet else here]
            return [here if a > b else nxt]  # the bank, not the field
        if a == _INLAND_WATER and not rivers.joined(a_hx, b_hx):
            return [here, nxt]
        return []

    @classmethod
    def _break_points(cls, source, seat, toward, hexes, cfg, rivers) -> list[HexCoord]:
        """The quays a cargo from *source* to *seat* crosses: every change in how it travels.

        Cart to boat, boat to cart, and barge to ship where a navigable river meets the sea.
        The quay is where the warehouses and the porters stand, which is dry land: a river
        runs along a hexside, so a barge ties up at the bank hex beside it, and that bank is
        the quay whether the cargo arrived by cart or leaves by ship.  A lake or the sea is
        met at the land hex beside it.  Two banks that are not the same water — two rivers
        side by side — mean landing and loading again, and both are quays.
        """
        quays = []
        here = source
        while here != seat:
            nxt = toward.get(here)
            if nxt is None:
                break
            quays += cls._step_quays(here, nxt, hexes, cfg, rivers)
            here = nxt
        return quays

    @classmethod
    def _charges(cls, source, seat, toward, hexes, cfg, rivers) -> list[Charge]:
        """Every place a cargo from *source* to *seat* pays on its way, in order.

        **Quays** (`_break_points`) pay `transship_share` to whoever stands within
        `transship_radius`.

        **Portages.** A cargo coming down past a cataract lands above it and loads again
        below. With `toll_portage_share` above 0 the walk round the falls pays that share
        once, at the first portage hex it sets foot on, to whoever stands within
        `toll_radius`, and the two landings either side pay nothing: one charge, not three.
        At 0 there is no portage toll, and the two landings are quays like any other.

        **Bridges.** A step over a side `CrossingStage` bridged — water too big to wade —
        pays `toll_bridge_share` at the bridge, to whoever stands within `toll_radius`. The
        site is the bank with the better soil, as a bridge village grows on the farmland
        rather than the sand; a boat passing along the river under it pays nothing.
        """
        quay_r, toll_r = cfg.transship_radius, cfg.toll_radius
        single = cfg.toll_portage_share > 0.0
        out: list[Charge] = []
        run: list[HexCoord] = []  # the portage hexes of the falls being walked round
        landed = False  # whether that walk began or ended with a landing
        here = source
        while here != seat:
            nxt = toward.get(here)
            if nxt is None:
                break
            at_falls = single and (here in rivers.portage or nxt in rivers.portage)
            if not at_falls and run:
                if landed:
                    out.append(Charge(run[0], PORTAGE, cfg.toll_portage_share, toll_r))
                run, landed = [], False
            if at_falls:
                run += [h for h in (here, nxt) if h in rivers.portage and h not in run]
            for quay in cls._step_quays(here, nxt, hexes, cfg, rivers):
                if single and at_falls:
                    landed = True
                else:
                    out.append(Charge(quay, QUAY, cfg.transship_share, quay_r, True))
            if cfg.toll_bridge_share > 0.0 and frozenset((here, nxt)) in rivers.bridged:
                a_hx, b_hx = hexes[here], hexes[nxt]
                if not (rivers.afloat(a_hx) and rivers.afloat(b_hx) and rivers.joined(a_hx, b_hx)):
                    site = max(
                        (here, nxt), key=lambda c: (SOIL_RANK.get(hexes[c].soil, 0), c == here)
                    )
                    out.append(Charge(site, BRIDGE, cfg.toll_bridge_share, toll_r))
            here = nxt
        if run and landed:
            out.append(Charge(run[0], PORTAGE, cfg.toll_portage_share, toll_r))
        return out

    def _pay(self, charges, origin, dest, by_coord, volume):
        """Who takes what of a cargo of *volume*: `(charge, handler, cut)` for each charge
        somebody other than the two ends collects, or nobody does (*handler* `None`).

        Neither end pays itself: loading at its own market is part of what the market
        already is, and unloading at the city is part of the city — and a town tolling its
        own exports over its own bridge would be keeping what it already had. Only the
        places between earn a living off the trade. The cuts never add up to more than the
        cargo.
        """
        paid = 0.0
        out = []
        for charge in charges:
            handler = self._handler(charge.site, by_coord, charge.radius)
            if handler in (origin, dest):
                continue
            cut = min(volume * charge.share, volume - paid)
            paid += cut
            out.append((charge, handler, cut))
        return out

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
        promoted: list[HexCoord] = []
        absorbed: dict[HexCoord, dict[HexCoord, float]] = {}

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
    def _allocate(markets, promoted, draw, reach, cfg) -> dict[HexCoord, dict[HexCoord, float]]:
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
        absorbed: dict[HexCoord, dict[HexCoord, float]] = {}
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
    def _nearest_share(offered, cost, total, cfg) -> dict[HexCoord, float]:
        """What a city actually takes of what is *offered*: `city_draw_share` of it, nearest first.

        Promotion is judged on everything that can reach a place; the take is capped. A city
        that took all of it drained the half of the map its water reach covers, and every
        city after the first was sized on leftovers. Filling from the cheapest source first
        means the markets at the gates feed the city outright and the far ones keep their
        surplus for a city of their own.
        """
        budget = total * cfg.city_draw_share
        take: dict[HexCoord, float] = {}
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
        city's gain — and a toll at every portage and bridge it passes (`_charges`). Still
        conserved — the handlers eat out of the cargo they handle — so an entrepôt grows on
        trade that feeds somebody else, which is how a river mouth can outgrow the
        hinterland it stands in.
        """
        rivers = river_index(state, cfg)
        by_coord = {s.coord: s for s in markets}
        start = {coord: s.population for coord, s in by_coord.items()}

        def moved(source: HexCoord, taken: float) -> float:
            """The people who follow *taken* worth of surplus off *source*."""
            total = draw.get(source, 0.0)
            if total <= 0.0 or source not in start:
                return 0.0
            return start[source] * min(1.0, taken / total)

        # Every flow, as [origin q, r, destination q, r, people it feeds, kind], for the
        # road stage to carry: freight wears roads as travellers do.
        freight: list[list[int | float | str]] = []
        # What each quay and toll point handled, and the ones nobody stands at: the city
        # keeps those for now, and `ResourceStage` founds a port or a toll town on the
        # busiest and moves them there.
        books = _Books()

        lost: dict[HexCoord, float] = {}
        gained: dict[HexCoord, float] = {}
        for seat, take in absorbed.items():
            for other, taken in take.items():
                people = moved(other, taken)
                lost[other] = lost.get(other, 0.0) + people
                gained[seat] = gained.get(seat, 0.0) + people
                if toward is not None and people > 0.0:
                    freight.append([*other, *seat, round(people, 3), "food"])
                if toward is None:
                    continue
                # A charge of nothing is no charge: with `transship_share` at 0 the cargo
                # changes hands at no quay at all.
                charges = [
                    c
                    for c in self._charges(other, seat, toward[seat], state.hexes, cfg, rivers)
                    if c.share > 0.0
                ]
                for charge, handler, cut in self._pay(charges, other, seat, by_coord, people):
                    if handler is None:
                        books.unpaid(charge, seat, taken, cut)
                        continue
                    gained[handler] = gained.get(handler, 0.0) + cut
                    gained[seat] -= cut
                    books.paid(charge, handler, taken, cut)

        for coord, settlement in by_coord.items():
            if coord in absorbed:
                settlement.tier = SettlementTier.CITY
                settlement.name = settlement.name.replace("_market_", "_city_")
            delta = gained.get(coord, 0.0) - lost.get(coord, 0.0)
            if delta:
                settlement.population = max(1, round(settlement.population + delta))

        state.metadata["cities"] = sorted(promoted)
        books.write(state, by_coord)
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

        Each shipment follows its bulk route and pays every charge on it (`_charges`: the
        quays, portages and bridges) as provisioning does, to whoever stands at each —
        drawn half from each of the two cities, so the books still balance. Charges nobody
        collects are left for `ResourceStage` to found a port or a toll town on.
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
        rivers = river_index(state, cfg)
        routes = {c: bulk_routes(hexes, [c], cfg, budget=budget, rivers=rivers) for c in cities}

        books = _Books.read(state)
        freight = state.metadata.setdefault("freight", [])

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
                charges = self._charges(origin, dest, routes[dest][1], hexes, cfg, rivers)
                self._split(charges, origin, dest, volume, by_coord, books, cfg)

        books.write(state, by_coord)

    # -- long-haul river trade ------------------------------------------------

    def _river_trade(self, state, cfg) -> None:
        """The interior's trade down its rivers to the coast, paying its way at every quay.

        `river_trade_flows` finds each inland river town's way downstream to the coastal
        town it reaches most cheaply, and what its cargo is worth. Here each one pays every
        charge on its stored path (`_charges`) — a quay at each change of mode, the toll at
        each portage, a bridge where it crosses one ashore — to whoever stands at each,
        drawn half from each end, exactly as `_trade` does; charges nobody collects are
        left for `ResourceStage`. Recorded as freight of kind `river`, which wears no road:
        it went by water. Each flow's whole path is kept in `metadata["river_trade"]`.
        """
        if cfg.river_trade_share <= 0.0:
            return
        rivers = river_index(state, cfg)
        flows = river_trade_flows(state, cfg, rivers)
        if not flows:
            return
        hexes = state.hexes
        by_coord = {s.coord: s for s in state.settlements}
        books = _Books.read(state)
        freight = state.metadata.setdefault("freight", [])
        routes: list[list[Any]] = []

        for flow in flows:
            origin, dest, volume = flow.origin, flow.dest, flow.people
            freight.append([*origin, *dest, round(volume, 3), "river"])
            routes.append([*origin, *dest, round(volume, 3), [list(h) for h in flow.path]])
            toward = dict(zip(flow.path, flow.path[1:], strict=False))
            charges = self._charges(origin, dest, toward, hexes, cfg, rivers)
            self._split(charges, origin, dest, volume, by_coord, books, cfg)

        state.metadata["river_trade"] = routes
        books.write(state, by_coord)

    def _split(self, charges, origin, dest, volume, by_coord, books, cfg) -> None:
        """Pay a flow between two settlements its *charges*, drawn half from each end.

        Manufactures and the river trade both go this way: the shipment is the two ends'
        business, so what the places between keep comes off both. Charges nobody collects
        are left on the books for `ResourceStage` to found a settlement on.
        """
        food = volume / cfg.people_per_food
        for charge, handler, cut in self._pay(charges, origin, dest, by_coord, volume):
            if handler is None:
                for seat in (origin, dest):
                    books.unpaid(charge, seat, food / 2, cut / 2)
                continue
            delta = books.delta
            delta[handler] = delta.get(handler, 0.0) + cut
            delta[origin] = delta.get(origin, 0.0) - cut / 2
            delta[dest] = delta.get(dest, 0.0) - cut / 2
            books.paid(charge, handler, food, cut)


class _Books:
    """What the trade stages owe and have paid, read from and written back to metadata.

    `transshipment` is the cargo each settlement handled at a quay or a portage;
    `unhandled_quays` and `unhandled_tolls` are the charges nobody stood near, which
    `ResourceStage` founds ports and toll towns on; `tolls` is what each settlement
    collected at each toll point, as `[site q, r, kind, collector q, r, food, people]`.
    """

    def __init__(self) -> None:
        self.handled: dict[HexCoord, float] = {}
        self.unhandled: dict[tuple[HexCoord, HexCoord], tuple[float, float]] = {}
        self.untolled: dict[tuple[HexCoord, HexCoord, str], tuple[float, float]] = {}
        self.tolls: dict[tuple[HexCoord, str, HexCoord], tuple[float, float]] = {}
        self.delta: dict[HexCoord, float] = {}

    @classmethod
    def read(cls, state) -> "_Books":
        books = cls()
        md = state.metadata
        books.handled = {(q, r): f for q, r, f in md.get("transshipment", [])}
        for q, r, sq, sr, food, people in md.get("unhandled_quays", []):
            books.unhandled[((q, r), (sq, sr))] = (food, people)
        for q, r, sq, sr, food, people, kind in md.get("unhandled_tolls", []):
            books.untolled[((q, r), (sq, sr), kind)] = (food, people)
        for q, r, kind, cq, cr, food, people in md.get("tolls", []):
            books.tolls[((q, r), kind, (cq, cr))] = (food, people)
        return books

    def unpaid(self, charge, seat, food, people) -> None:
        if charge.landing:
            f, p = self.unhandled.get((charge.site, seat), (0.0, 0.0))
            self.unhandled[(charge.site, seat)] = (f + food, p + people)
        else:
            key = (charge.site, seat, charge.kind)
            f, p = self.untolled.get(key, (0.0, 0.0))
            self.untolled[key] = (f + food, p + people)

    def paid(self, charge, handler, food, people) -> None:
        if charge.kind in (QUAY, PORTAGE):
            self.handled[handler] = self.handled.get(handler, 0.0) + food
        if charge.kind != QUAY:
            key = (charge.site, charge.kind, handler)
            f, p = self.tolls.get(key, (0.0, 0.0))
            self.tolls[key] = (f + food, p + people)

    def write(self, state, by_coord) -> None:
        for coord, d in self.delta.items():
            s = by_coord[coord]
            s.population = max(1, round(s.population + d))
        md = state.metadata
        if self.handled:
            md["transshipment"] = [
                [q, r, round(f, 3)] for (q, r), f in sorted(self.handled.items())
            ]
        if self.unhandled:
            md["unhandled_quays"] = [
                [q, r, sq, sr, round(f, 3), round(p, 3)]
                for ((q, r), (sq, sr)), (f, p) in sorted(self.unhandled.items())
            ]
        if self.untolled:
            md["unhandled_tolls"] = [
                [q, r, sq, sr, round(f, 3), round(p, 3), kind]
                for ((q, r), (sq, sr), kind), (f, p) in sorted(self.untolled.items())
            ]
        if self.tolls:
            md["tolls"] = [
                [q, r, kind, cq, cr, round(f, 3), round(p, 3)]
                for ((q, r), kind, (cq, cr)), (f, p) in sorted(self.tolls.items())
            ]


__all__ = ["CityPromotionStage"]
