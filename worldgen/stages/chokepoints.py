"""Settlements that exist because the traffic has to come through them.

The tier below the market, and the only one the organic model has room for. Markets are
sited by what a countryside can send them; nothing below that scale earns a glyph on
economic grounds, because the peasantry is already in the arithmetic as a productive
surface rather than as a list of hamlets. What does earn one is a place where the road has
no choice — a bridgehead, or a saddle the ground either side forbids going round. Oxford,
Cambridge, Frankfurt, Innsbruck: every one of them is named for its crossing.

That is why this runs *after* the roads rather than before. A chokepoint is not a good
site that happens to have traffic; it is a bad site that has traffic anyway, and only the
built network can say which crossings carry any. It also means these settlements perturb
nothing: they are on the road by construction, so no route has to be recut around them.

Two kinds, and on this terrain they are wildly unequal.

**Bridgeheads** are common, because a road crosses water constantly.

**Passes** are rare, and that is the model working rather than failing. Roads price ascent
— `slope_edge_cost` charges every metre of climb — so a router offered a way round a ridge
takes it. A pass settlement therefore appears only where the ground leaves no way round,
which on 1500 m of relief is a couple of places on a map and on flat country is none.
"""

from ..core.hex import Settlement, SettlementTier
from ..core.hex_grid import distance, hex_range, neighbors
from ..core.pipeline import GeneratorStage
from ..core.world_state import ROAD_TIER_RANK, RoadTier, WorldState
from .city_town import _assign_role
from .habitability import actual_food
from .haulage import allocate_catchments, gather, settleable, usable_fraction

PASS = "pass"
BRIDGE = "bridge"


def saddle_relief_m(coord, hexes) -> float:
    """How far the ground rises either side of a saddle, in metres. 0.0 if not a saddle.

    A saddle is the shape of a pass: low ground on two sides of you and high ground on the
    other two. On a hex grid that reads directly off the ring — walk the six neighbours in
    order and count the runs of ground standing above you. One run is a hillside, none is
    a summit or a pit, and two or more is a col.

    The figure returned is the *lesser* of the two flanks, because a pass is only as walled
    as its weaker side: if one flank is a cliff and the other a gentle rise, you walk over
    the rise and the saddle is not a chokepoint at all. And a flank, in turn, is only as
    walled as its *lowest* hex — a run of [500, 40] is crossed at the 40, whatever stands
    beside it. Scoring a flank by its highest hex called a wall with a gap in it a wall,
    which is how a hillside with one proud outcrop could read as impassable ground.
    """
    hx = hexes[coord]
    ring = [hexes.get(n) for n in neighbors(coord)]
    if any(n is None for n in ring):
        return 0.0  # the map edge; there is no ring to walk
    high = [n.elevation > hx.elevation for n in ring]
    if all(high) or not any(high):
        return 0.0
    start = next(i for i in range(6) if high[i] and not high[i - 1])
    runs: list[list[float]] = []
    current: list[float] = []
    for step in range(6):
        i = (start + step) % 6
        if high[i]:
            current.append(ring[i].elevation)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    if len(runs) < 2:
        return 0.0
    return min(min(run) for run in runs) - hx.elevation


def is_pass(coord, hexes, cfg) -> bool:
    """A saddle walled by ground a cart cannot cross.

    The threshold is `terrain_steep_gradient_m` rather than a setting of its own, and
    deliberately: a hex is 1 km across, so that figure is already the gradient at which
    this map calls ground STEEP — "pack animals, terraces, no wheels". Ground the flanks
    of a saddle rise at least that fast is ground a road will not climb, which is the
    entire content of the claim that the traffic must come through here. A separate knob
    could only ever disagree with the terrain bands about what counts as impassable.
    """
    return saddle_relief_m(coord, hexes) >= cfg.terrain_steep_gradient_m


def residual_surplus(hexes, cfg) -> dict:
    """The marketable surplus the markets did not take.

    `gather` weights every hex by `usable_fraction` of the distance to its market, which
    falls to exactly zero at `market_day_radius` — so the fraction a market left behind is
    the complement of that, and ground outside every catchment was never drawn on at all.
    Reading it back this way is what stops the tier below double-counting the tier above:
    a village on a market's doorstep finds nothing left and is not founded, without any
    rule saying villages may not stand near markets.
    """
    out = {}
    for coord, hx in hexes.items():
        surplus = actual_food(hx, cfg) * cfg.marketable_surplus_fraction
        if surplus <= 0.0:
            continue
        if hx.territory is not None:
            surplus *= 1.0 - usable_fraction(hx.territory_cost, cfg.market_day_radius)
        if surplus > 0.0:
            out[coord] = surplus
    return out


class ChokepointStage(GeneratorStage):
    """Founds the village tier on bridgeheads and passes that carry real traffic."""

    def run(self, state: WorldState) -> WorldState:
        hexes = state.hexes
        cfg = self.config

        for coord in hexes:
            if is_pass(coord, hexes, cfg):
                hexes[coord].tags.add(PASS)

        candidates = self._candidates(state, cfg)
        if not candidates:
            return state

        residual = residual_surplus(hexes, cfg)
        seats = self._plant(candidates, residual, state, cfg)
        if not seats:
            return state

        # Twice, so the floor is applied to the figure it names rather than to an estimate
        # of it. Planting has to score on hex distance — the exclusive partition does not
        # exist until the seats do — but what decides whether a place is a village is the
        # draw off its real catchment, and the two differ by however rough the ground is.
        # The second pass re-partitions among the survivors, so a village is not left
        # holding the smaller catchment it had while a rejected neighbour was still in.
        draw = self._draw(hexes, seats, residual, cfg)
        seats = [s for s in seats if draw.get(s, 0.0) >= cfg.chokepoint_min_draw]
        if not seats:
            return state
        draw = self._draw(hexes, seats, residual, cfg)

        state.settlements.extend(self._found(seats, draw, hexes, cfg))
        return state

    @staticmethod
    def _draw(hexes, seats, residual, cfg) -> dict:
        """What each seat can actually fetch off its own fields."""
        owner, cost = allocate_catchments(hexes, seats, cfg.rural_field_radius, cfg)
        return gather(residual, owner, cost, cfg.rural_field_radius)

    # -- what may be one -------------------------------------------------------

    def _candidates(self, state, cfg) -> list:
        """Ground that holds a chokepoint *and* carries traffic over it.

        Both halves are needed. A bridge on a farm track is a plank, not a town; and a
        busy road crossing open country is passing through nowhere in particular.

        And the road has to be part of the network that joins settlements to each other.
        `road_tier` alone cannot say that, because the tiers are percentile cuts — a fixed
        *share* of edges is secondary however little uses them — so a short spur can hold a
        secondary road on no traffic at all. `prune_orphan_roads` keeps such a spur where it
        lands a sea leg, since a road to a harbour is a road to somewhere; but a settlement
        founded on one is a landing place, not a chokepoint, and it would be cut off from
        every other settlement by land while sharing their ground.
        """
        hexes = state.hexes
        floor = ROAD_TIER_RANK[RoadTier(cfg.chokepoint_min_road_tier)]
        carries = set()
        for (a, b), edge in state.road_edges.items():
            if ROAD_TIER_RANK[edge.tier] >= floor:
                carries.add(a)
                carries.add(b)
        carries &= self._through_network(state)

        # A bridge only holds anything if a road actually goes over it. `CrossingStage`
        # tags its bridges before any road exists — they are candidate sites, and most of
        # them are never built at. Judged by the drawn network: a crossed bridge carries
        # at least two road edges, one onto each bank; a tagged hex with one edge is a
        # road that ends at the water, and one with none is a proposal nobody took up.
        # Without this test the tier founded villages beside phantom crossings — on one
        # 96x96 fixture, six of seven stood at bridges no road touched.
        degree: dict = {}
        for a, b in state.road_edges:
            degree[a] = degree.get(a, 0) + 1
            degree[b] = degree.get(b, 0) + 1

        def crossed(coord) -> bool:
            return degree.get(coord, 0) >= 2

        occupied = {s.coord for s in state.settlements}
        held = set()
        for coord in carries:
            hx = hexes.get(coord)
            if hx is None:
                continue
            if PASS in hx.tags or (BRIDGE in hx.tags and crossed(coord)):
                held.add(coord)
            elif any(
                BRIDGE in hexes[n].tags and crossed(n) for n in neighbors(coord) if n in hexes
            ):
                held.add(coord)  # the bridgehead, which is where the town stands

        return sorted(held & settleable(hexes, cfg) - occupied)

    @staticmethod
    def _through_network(state) -> set:
        """Road hexes on a component that already joins settlements to each other.

        A component holding fewer than two settlements carries no journey between them,
        whatever tier its edges came out at.
        """
        adj: dict = {}
        for a, b in state.road_edges:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)

        seats = {s.coord for s in state.settlements}
        through: set = set()
        seen: set = set()
        for start in adj:
            if start in seen:
                continue
            stack, comp = [start], set()
            while stack:
                c = stack.pop()
                if c in comp:
                    continue
                comp.add(c)
                stack.extend(adj[c] - comp)
            seen |= comp
            if len(comp & seats) >= 2:
                through |= comp
        return through

    # -- planting --------------------------------------------------------------

    def _plant(self, candidates, residual, state, cfg) -> list:
        """Greedy over what is left, claiming a village's fields outright as it goes.

        A hard claim rather than the decaying share markets use, because the two are
        answering different questions. Markets compete for the same countryside, so each
        may take only a share and the rest supports the next one along; that is what makes
        market spacing follow the land. The fields around a village are not shared with
        anybody — they are walked out to and back from daily — so what one takes, the next
        does not get. There are only a few dozen candidates, so this rescores exhaustively
        rather than needing the lazy-greedy machinery.
        """
        remaining = dict(residual)
        field = int(cfg.rural_field_radius)

        def score(coord):
            """The same weighted sum `gather` uses, on hex distance rather than real cost.

            Only the ordering matters here — the floor is applied afterwards to the true
            draw — so this needs to rank candidates, not to measure them. Weighting by
            `usable_fraction` anyway keeps that ranking close to the one the real
            catchments produce: scoring the raw disc would rate a seat with distant fields
            level with one whose fields are at the gate.
            """
            total = 0.0
            for n in hex_range(coord, field):
                value = remaining.get(n)
                if value:
                    total += value * usable_fraction(distance(coord, n), cfg.rural_field_radius)
            return total

        # Existing settlements block the ground around them, so a village cannot be
        # planted in a market's gateway. The economics would mostly prevent it anyway —
        # there is no residual surplus that close to a market — but a bridge on the town's
        # own doorstep is the town's bridge whatever the arithmetic says.
        blocked = set()
        for s in state.settlements:
            blocked |= set(hex_range(s.coord, cfg.chokepoint_min_separation))

        seats: list = []
        pool = list(candidates)
        while pool:
            best = max(pool, key=lambda c: (score(c), c))
            if score(best) <= 0.0:
                break  # nothing left for anybody; the rest are on ground already claimed
            pool.remove(best)
            if best in blocked:
                continue
            seats.append(best)
            blocked |= set(hex_range(best, cfg.chokepoint_min_separation))
            for n in hex_range(best, field):
                remaining.pop(n, None)
        return seats

    # -- founding --------------------------------------------------------------

    def _found(self, seats, draw, hexes, cfg) -> list:
        """Sized by the fields it can work, the same arithmetic as every other tier.

        What differs is only the range: `rural_field_radius`, the daily walk out to the
        fields, against the market's day-return and the city's bulk haul. A village is not
        a small market — it is a place whose reach is a morning's walk.
        """
        jitter = self.rng.uniform(0.9, 1.1, size=len(seats))
        out = []
        for i, coord in enumerate(sorted(seats)):
            hx = hexes[coord]
            population = max(1, round(draw.get(coord, 0.0) * cfg.people_per_food * jitter[i]))
            kind = PASS if PASS in hx.tags else BRIDGE
            s = Settlement(
                coord=coord,
                tier=SettlementTier.VILLAGE,
                role=_assign_role(coord, hx, hexes),
                population=population,
                name=f"{hx.biome.name.lower()}_{kind}_{i}",
            )
            hx.settlement = s
            out.append(s)
        return out


__all__ = ["ChokepointStage", "is_pass", "residual_surplus", "saddle_relief_m"]
