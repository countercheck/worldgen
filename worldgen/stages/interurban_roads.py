from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from ..core import routing
from ..core.hex import HexCoord, SettlementRole, SettlementTier, TerrainClass
from ..core.hex_grid import distance, neighbors
from ..core.pipeline import GeneratorStage
from ..core.routing import Grid
from ..core.world_state import ROAD_TIER_RANK, RoadTier, WorldState, road_edge_key
from .road_cost import (
    add_traffic,
    as_road_edges,
    fill_tier_gaps,
    prune_orphan_roads,
    river_crossings,
    road_edge_costs,
    road_node_costs,
    route_through_settlements,
    settlement_rings,
    tag_river_crossings,
    tag_switchbacks,
    tier_near,
)

# The road the connectivity guarantee lays to a stranded place, by the best place on it.
_ROAD_FOR_TIER = {
    SettlementTier.CITY: RoadTier.PRIMARY,
    SettlementTier.TOWN: RoadTier.SECONDARY,
    SettlementTier.VILLAGE: RoadTier.TRACK,
}


# The village roles `ResourceStage` founds before the roads, which the roads must reach.
_FOUNDED = frozenset(
    {
        SettlementRole.MINING,
        SettlementRole.LUMBER,
        SettlementRole.BRIDGE,
        SettlementRole.PORTAGE,
        SettlementRole.CARAVANSARY,
    }
)


class InterurbanRoadStage(GeneratorStage):
    """Builds PRIMARY and SECONDARY roads between cities and towns only.

    Runs before village placement so that villages can use road corridors
    as placement candidates.
    """

    def run(self, state: WorldState) -> WorldState:
        hexes = state.hexes
        cfg = self.config
        # Cities and towns, and the villages `ResourceStage` founds: a mine or a lumber camp
        # has to be reached by road or its ore and timber go nowhere, and a toll village
        # lives on the traffic going past it. Other villages are placed after this stage
        # and joined by stages of their own.
        settlements = [
            s
            for s in state.settlements
            if s.tier in (SettlementTier.CITY, SettlementTier.TOWN) or s.role in _FOUNDED
        ]
        if not settlements:
            return state

        edge_traffic: dict[tuple[HexCoord, HexCoord], float] = defaultdict(float)
        canonical_routes: dict[tuple[HexCoord, HexCoord], list[HexCoord]] = {}
        journeys: dict[tuple[HexCoord, HexCoord], tuple[float, tuple[HexCoord, ...]]] = {}
        # The network as it grows, so a route can aim at the road rather than the town.
        net_adj: dict[HexCoord, set[HexCoord]] = defaultdict(set)
        # Costing the road home means a Dijkstra over the network per route, and the
        # network barely changes once the trunks are down — so keep the answer and rebuild
        # only when an edge has actually been added since it was worked out.
        net_version = [0]
        home_cache: dict[HexCoord, tuple[int, np.ndarray, np.ndarray]] = {}

        # Where the rivers run between hexes: a step across one is a crossing, and pays.
        crossings = river_crossings(state.river_sides)
        settled = {s.coord for s in state.settlements}
        # Which seats each hex neighbours, so an edge can be charged for skirting one.
        ring = settlement_rings(settled)

        # A hex costs its terrain, worn down by the traffic over it (`pheromone_discount`).
        # No discount for running beside a river. Roads follow valleys because valleys are
        # the low, level, well-watered ground that leads somewhere, not because a rule pays
        # them to.
        net = _Network.of(hexes, cfg, crossings, ring)

        # Travellers come from population rather than from tier, so a market of 6,200 wears
        # a deeper road out of its gates than one of 900. Population used to enter only on
        # the destination side of the gravity term, which made every origin equally busy.
        travellers = []
        for s in settlements:
            n = min(
                cfg.road_travellers_max, max(1, round(s.population * cfg.road_travellers_per_pop))
            )
            travellers.extend([s] * n)
        # Busiest first, rather than shuffled. The pheromone means order decides which route
        # gets worn first and which ones then snap onto it, so a random order had minor
        # journeys laying down track for trunk routes to follow. Sorting by the traffic a
        # settlement emits builds the trunks first and lets the rest tributary into them.
        # Ties keep list order, which is settlement order, so this stays deterministic.
        travellers.sort(key=lambda s: -s.population)

        pop_arr = [float(s.population) for s in settlements]
        coords_arr = [s.coord for s in settlements]
        n_s = len(settlements)
        s_index = {s.coord: i for i, s in enumerate(settlements)}

        def journey(origin, dest, n) -> None:
            """Route *origin* to *dest* and wear the road by *n* journeys along it."""
            key = (min(origin, dest), max(origin, dest))
            if key in canonical_routes:
                path = canonical_routes[key]
            else:
                path = self._route(net, origin, dest, home_cache, net_version[0])
                if path is None or len(path) < 2:
                    return
                canonical_routes[key] = path

            count, _ = journeys.get(key, (0.0, ()))
            journeys[key] = (count + n, tuple(path))
            for c in path:
                net.wear(c, n)
            for a, b in zip(path, path[1:], strict=False):
                edge_traffic[road_edge_key(a, b)] += n
                if b not in net_adj[a]:
                    net_adj[a].add(b)
                    net_adj[b].add(a)
                    net.join(a, net_adj[a])
                    net.join(b, net_adj[b])
                    net_version[0] += 1

        for origin_s in travellers:
            oi = s_index[origin_s.coord]
            dists = [max(1, distance(origin_s.coord, c)) for c in coords_arr]
            weights = [
                pop_arr[j] / (dists[j] ** cfg.road_gravity_exponent) if j != oi else 0.0
                for j in range(n_s)
            ]
            total_w = sum(weights)
            if total_w == 0:
                continue
            probs = [w / total_w for w in weights]
            di = int(self.rng.choice(n_s, p=probs))
            journey(origin_s.coord, coords_arr[di], 1.0)

        # Freight wears roads as travellers do. Every flow the settlement stages recorded
        # puts journeys on its route in proportion to the people it feeds, at two rates.
        # Raw goods — provisioning from market to city, ore from the mines — are bulk that
        # went short distances or by water, and wear the spokes into each city at
        # `road_raw_freight_per_person`. Manufactures between cities are the carrier and
        # wagon trade the main roads were built for, at `road_goods_freight_per_person`.
        # With one rate, provisioning outweighed the goods trade and its spokes took the
        # primary tier from the roads between cities. Timber is left off: it floats, and a
        # lumber camp's road is its river. Busiest first, as the travellers are.
        rate = {"goods": cfg.road_goods_freight_per_person}
        rate["food"] = rate["ore"] = cfg.road_raw_freight_per_person
        flows = state.metadata.get("freight", [])
        for oq, or_, dq, dr, people, kind in sorted(flows, key=lambda f: (-f[4], f[:4])):
            if kind in rate:
                n = people * rate[kind]
                origin, dest = (oq, or_), (dq, dr)
                if n > 0.0 and origin in hexes and dest in hexes and origin != dest:
                    journey(origin, dest, n)

        # Tier is a property of an edge, not of a journey.  It used to be taken per hex and
        # then collapsed onto whole routes by `_path_min_tier`, which handed a 157-hex route
        # the weakest tier any hex on it earned — one quiet hex demoted a trunk road end to
        # end, and a map came out 1,935 secondary against 6 primary.
        #
        # Consolidation happens here, on the traffic, and not after the tiers are cut.
        # Bending a bypass through a town merges two flows onto one pair of edges, and the
        # merged edge has to be ranked on what it now carries — two secondary roads meeting
        # at a market can make a primary. Taking the higher of two tiers afterwards cannot
        # express that; adding the traffic first and cutting the percentiles after does it
        # for nothing.
        route_through_settlements(edge_traffic, hexes, settled, cfg, crossings, combine=add_traffic)

        # A step along a riverbank uses the lower `road_river_traffic_min` threshold, so
        # that well-trafficked banks become drawn roads (towpaths, river roads).  Along the
        # bank is both hexes beside the same river and not across it: a step across is a
        # crossing, and earns no towpath.
        banks_of: dict[HexCoord, set[int]] = defaultdict(set)
        for i, river in enumerate(state.rivers):
            for c in river.banks():
                banks_of[c].add(i)

        def eligible_edge(key) -> bool:
            t = edge_traffic[key]
            if t >= cfg.road_min_traffic:
                return True
            a, b = key
            towpath = bool(banks_of[a] & banks_of[b]) and frozenset(key) not in crossings
            return towpath and t >= cfg.road_river_traffic_min

        eligible = sorted(
            (k for k in edge_traffic if eligible_edge(k)),
            key=lambda k: (-edge_traffic[k], k),
        )
        # `road_min_traffic` decides what is a road; the percentiles only decide how it is
        # drawn. They used to do both, which is why raising it from 3 to 1000 moved road
        # coverage by 0.4% of the map — it shrank the eligible set, and the percentiles
        # promptly re-cut the same fractions of whatever survived. Everything eligible is
        # now drawn, and a quiet lane is a TRACK rather than nothing at all.
        road_edges: dict[tuple[HexCoord, HexCoord], RoadTier] = {}
        if eligible:
            p_cut = max(1, round(len(eligible) * cfg.road_primary_pct))
            s_cut = max(
                p_cut + 1,
                round(len(eligible) * (cfg.road_primary_pct + cfg.road_secondary_pct)),
            )
            for i, key in enumerate(eligible):
                if i < p_cut:
                    road_edges[key] = RoadTier.PRIMARY
                elif i < s_cut:
                    road_edges[key] = RoadTier.SECONDARY
                else:
                    road_edges[key] = RoadTier.TRACK

        # Every settlement, not just the cities. It used to run only when a map had two
        # or more of them, so the organic model — whose markets are all TOWN — had nothing
        # guaranteeing its network was in one piece. It came out connected anyway only
        # because stitching made almost every route a concatenation of the same few legs;
        # with routes pathfound independently the map broke into two components.
        if len(settlements) > 1:
            road_edges, unreachable = self._guarantee_connectivity(
                hexes, settlements, road_edges, cfg, crossings
            )
            if unreachable:
                # Kept on the world rather than logged away: a map in pieces is a fact a
                # reader of the output should be able to see.
                state.metadata.setdefault("unreachable_settlements", []).extend(
                    {"coord": list(coord), "reason": reason} for coord, reason in unreachable
                )

        # An edge with a foot in the water is a sea leg, not a road. Splitting them here
        # rather than at draw time is what makes "is there a land route" a question the
        # world can answer: half this network by hex count is water, and by land alone the
        # reference map is forty networks tied together by eight crossings.
        sea_edges = {
            key: tier
            for key, tier in road_edges.items()
            if any(
                hexes[c].terrain_class in (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER)
                for c in key
            )
        }
        for key in sea_edges:
            del road_edges[key]

        # Where land can join two places, roads should. Sea carriage is so much cheaper
        # than land that routes will cross a bay rather than walk round it, which is right
        # for a journey and wrong for a network: it left the reference map as forty land
        # networks tied together by eight sea crossings, so a cart could not get from one
        # market to the next without a boat. This adds what the traffic model declined to.
        self._join_by_land(hexes, settlements, road_edges, cfg, crossings)

        # Last of all, on tiers: the connectivity guarantee and the land join both lay
        # roads of their own, and either can skirt a town like any other route.
        route_through_settlements(road_edges, hexes, settled, cfg, crossings)

        # Tiers are cut per edge, so where two routes part for a hex or two and rejoin, the
        # traffic splits between the branches and a trunk road steps down and back up again.
        # A road does not change class for a kilometre and change back; fill the dip.
        fill_tier_gaps(road_edges, cfg.road_tier_gap_max_edges)

        anchors = settled | {c for f in state.ferries for c in (f.a, f.b)}
        # A land network reaching no settlement is a residue of the traffic threshold; one
        # reaching only a shore is a road to a harbour, which is a road to somewhere.
        anchors |= {c for key in sea_edges for c in key}
        prune_orphan_roads(road_edges, anchors)

        for a, b in list(road_edges) + list(sea_edges):
            if a in hexes and b in hexes:
                hexes[a].road_connections.add(b)
                hexes[b].road_connections.add(a)

        tag_river_crossings(road_edges, state, cfg)
        tag_switchbacks(road_edges, hexes, cfg)

        # Re-score habitability near roads so VillagePlacementStage benefits.  Only the
        # village score: cities and towns are already sited by this point, and a road
        # they caused should not retroactively flatter the ground it runs over.
        road_hex_set = {c for edge in road_edges for c in edge}
        for coord, hx in hexes.items():
            if hx.settlement is not None:
                continue
            if hx.terrain_class in (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER):
                continue
            if any(n in road_hex_set for n in neighbors(coord)):
                hx.habitability_village = min(1.0, hx.habitability_village + 0.2)

        # The tiers were enough to build with; what goes out carries the delta elevation
        # too, signed in the direction of the key, so nothing downstream has to rebuild the
        # cost model to know how slow a segment is.
        state.road_edges = as_road_edges(road_edges, hexes, edge_traffic)
        state.sea_edges = as_road_edges(sea_edges, hexes, edge_traffic)
        # And the journeys themselves, which the tiers were cut from: a place on a busy
        # road lives on the people going past it, so the stages after this one need to
        # know how many there were, not only how the road was drawn.
        for i in np.flatnonzero(net.worn).tolist():
            hexes[net.grid.coord(i)].traffic = round(float(net.traffic[i]), 3)
        state.journeys = journeys
        return state

    def _route(self, net, origin, dest, cache, version):
        """The journey from *origin* to *dest*: a new leg, then the road that already goes there.

        A traveller bound for a town does not need a road of their own all the way, they need
        to reach the road that serves it. So the search runs against every hex from which
        *dest* is already reachable along roads that exist, and stops at whichever it
        touches first; the rest of the journey is that road.

        This is what stops the network being a mat. Pathing all the way to the seat every
        time had each route find its own line, and A* — whose heuristic assumes 1.0 a step
        and so misprices anything cheaper — could not reliably find the same line twice, so
        routes ran beside one another instead of joining. Aiming at the network means a
        route *joins* it by construction rather than by the pathfinder's good luck.

        It is also much cheaper, because the search ends at the first road it meets rather
        than at the far end of the map: 217k node expansions against 1,329k, and a road
        stage of 3.8s against 9.6s, while covering 11.8% of the land instead of 19.3%.

        The road home is *costed*, and weighed against the cost of reaching each possible
        join. Stopping at whichever road hex is cheapest to reach is not the same as the
        cheapest journey: a traveller would join at their own doorstep and follow the network
        however far round it went, so no road was ever built between two places the network
        already joined badly, and the graph came out very nearly a tree.

        That weighing is why the search runs without a heuristic. `goal_cost` is real cost
        and the heuristic counts hexes at 1.0 apiece, which road travel is far below, so the
        two together make the search abandon at the first expansion and take the network
        route every time. Dijkstra is slower and is the price of the comparison meaning
        anything.
        """
        # Every hex from which dest is reachable on the network so far, with the way back
        # and what it costs. Empty for the first traveller, who therefore paths the whole
        # way and becomes the road that everyone after them joins.
        grid = net.grid
        cached = cache.get(dest)
        if cached is not None and cached[0] == version:
            tree, home_cost = cached[1], cached[2]
        else:
            tree, home_cost = routing.along(
                net.adj, grid.nbr, net.node, net.traffic, net.factor, net.edge, grid.index[dest]
            )
            cache[dest] = (version, tree, home_cost)

        # No short circuit when the origin is already on the network. It is tempting — there
        # is a road home, so take it — but that is `_stitch_via_junction`'s mistake in
        # another guise, committing to an existing route without weighing it against a
        # direct one. The origin is itself a goal reached at no cost, so the network route
        # is the search's opening candidate and is beaten only if striking out pays.
        leg = _to_any(
            grid,
            net.node,
            net.edge,
            origin,
            home_cost < np.inf,
            traffic=net.traffic,
            factor=net.factor,
            goal_cost=home_cost,
        )
        if leg is None:
            return None
        return leg + routing.walk_back(grid, tree, int(tree[grid.index[leg[-1]]]))[::-1]

    @staticmethod
    def _join_by_land(hexes, places, road_edges, cfg, crossings):
        """Join everything that shares a landmass into one road network.

        The traffic model has no reason to build these: a traveller crossing a bay is doing
        the sensible thing, since sea carriage cost a fraction of land carriage. But a
        network in which neighbouring markets can only be reached by boat is not a road
        network, and a wargame cannot march down it.

        Joins **road components**, not merely settlements. Those are not the same thing and
        the difference bit: a spur that lands a sea leg is kept by `prune_orphan_roads` —
        a road to a harbour is a road to somewhere — but it holds no settlement, so a
        settlement-only rule had nothing to join it to. `ChokepointStage` then founded a
        village on such a spur, and the village came out cut off by land from the 37
        settlements it shared ground with. Joining the networks means anything founded
        later is on the one network by construction, whatever founds it.

        A settlement on no road at all is treated as a component of one, so this subsumes
        the rule it replaces.

        Land only, deliberately — the cost function refuses water outright, so this cannot
        satisfy itself with the sea leg that already exists.
        """
        water = (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER)
        dry = {c for c, hx in hexes.items() if hx.terrain_class not in water}

        grid = Grid.of(hexes)
        wet = grid.column(hexes, lambda hx: hx.terrain_class in water, np.bool_, False)
        land_cost = np.where(wet, np.inf, road_node_costs(hexes, grid, cfg))
        land_edge = road_edge_costs(hexes, grid, cfg, crossings)

        # What the ground itself connects, ignoring roads entirely.
        mass_of: dict[HexCoord, HexCoord] = {}
        for seed in dry:
            if seed in mass_of:
                continue
            stack, reached = [seed], set()
            while stack:
                c = stack.pop()
                if c in reached:
                    continue
                reached.add(c)
                stack.extend(n for n in neighbors(c) if n in dry and n not in reached)
            for c in reached:
                mass_of[c] = seed

        # The things that need joining: every road component, plus any settlement standing
        # on no road at all.
        adj: dict[HexCoord, set[HexCoord]] = defaultdict(set)
        for a, b in road_edges:
            adj[a].add(b)
            adj[b].add(a)
        units: list[set[HexCoord]] = []
        seen: set[HexCoord] = set()
        for node in adj:
            if node in seen:
                continue
            stack, comp = [node], set()
            while stack:
                c = stack.pop()
                if c in comp:
                    continue
                comp.add(c)
                stack.extend(adj[c] - comp)
            seen |= comp
            on_land = comp & dry
            if on_land:
                units.append(on_land)
        for place in places:
            if place.coord in dry and place.coord not in seen:
                units.append({place.coord})

        by_mass: dict[HexCoord, list[set[HexCoord]]] = defaultdict(list)
        for unit in units:
            by_mass[mass_of[next(iter(unit))]].append(unit)

        added = 0
        for group in by_mass.values():
            if len(group) < 2:
                continue
            # Largest first, so the small pieces are drawn onto the trunk rather than the
            # trunk onto a spur.
            group.sort(key=len, reverse=True)
            joined = set(group[0])
            goals = grid.mask(joined)
            for unit in group[1:]:
                if unit & joined:
                    continue
                # From the piece to whatever part of the network is cheapest to reach,
                # rather than to the hex that happens to be nearest as the crow flies.
                # The nearest one can be unreachable — across a gorge, or up a grade the
                # road rules refuse — and the old version silently gave up when it was,
                # which is how a settlement could stay stranded with a route ten hexes away.
                best = None
                for src in sorted(unit):
                    path = _to_any(grid, land_cost, land_edge, src, goals)
                    if path and len(path) > 1 and (best is None or len(path) < len(best)):
                        best = path
                if best is None:
                    continue
                # The join carries on the roads it meets at either end, so it takes the
                # lower of the two. It used to be a TRACK always, and the commonest join is
                # a trunk road whose sea leg was just split off above — which left a primary
                # road running to a shore, a lane round the bay, and the primary road again
                # on the far side. Read within a couple of hexes of each end rather than at
                # the end hex itself: the cheapest place to leave a piece is often a lane a
                # hex off the trunk's end, and `fill_tier_gaps` closes that hex. Not the
                # best road anywhere on each piece, which made a primary of every join
                # whose piece had a trunk somewhere in it — 164 edges on seed 42.
                tier = min(
                    tier_near(road_edges, best[0], 2),
                    tier_near(road_edges, best[-1], 2),
                    key=ROAD_TIER_RANK.__getitem__,
                )
                for a, b in zip(best, best[1:], strict=False):
                    road_edges.setdefault(road_edge_key(a, b), tier)
                joined |= set(best) | unit
                goals[[grid.index[c] for c in (*best, *unit)]] = True
                added += 1
        return added

    def _guarantee_connectivity(self, hexes, places, road_edges, cfg, crossings):
        """Join any settlement the traffic model left off the network.

        Adjacency is the drawn network itself.  It used to be rebuilt from whichever
        canonical routes contributed a tier, which was a second, subtly different answer to
        "what counts as a road" — with edges as the stored form there is only one.
        """
        road_adj: dict[HexCoord, set[HexCoord]] = defaultdict(set)
        for a, b in road_edges:
            road_adj[a].add(b)
            road_adj[b].add(a)

        place_coords = {s.coord for s in places}

        def bfs_component(start):
            visited = {start}
            queue = deque([start])
            while queue:
                c = queue.popleft()
                for n in road_adj.get(c, set()):
                    if n not in visited:
                        visited.add(n)
                        queue.append(n)
            return visited

        visited_global: set[HexCoord] = set()
        components = []
        for cc in place_coords:
            if cc in visited_global:
                continue
            comp = bfs_component(cc)
            visited_global |= comp
            components.append(comp)
        if not components:
            return road_edges, []
        main = max(components, key=len)

        grid = Grid.of(hexes)
        plain_cost = road_node_costs(hexes, grid, cfg)
        plain_edge = road_edge_costs(hexes, grid, cfg, crossings)

        def adopt(path, tier) -> None:
            """Lay *path* into the network at *tier*, without demoting anything."""
            for a, b in zip(path, path[1:], strict=False):
                road_adj[a].add(b)
                road_adj[b].add(a)
                road_edges.setdefault(road_edge_key(a, b), tier)

        def tier_for(coord) -> RoadTier:
            """The road a stranded piece of the network deserves: by the best place on it.

            This used to be PRIMARY always, from when only a city could be stranded; once
            the resource villages joined the network it was handing a lumber camp of 120 a
            trunk road. The piece is judged on everything it holds, because a village can
            stand between the main network and a city cut off behind it.
            """
            piece = bfs_component(coord) | {coord}
            best = max(
                (_ROAD_FOR_TIER[s.tier] for s in places if s.coord in piece),
                key=ROAD_TIER_RANK.__getitem__,
                default=RoadTier.TRACK,
            )
            return best

        # Settlements nothing can reach. Reported, not raised.
        unreachable: list[tuple[HexCoord, str]] = []
        max_iter = len(places) * 2
        for _ in range(max_iter):
            isolated = [s for s in places if s.coord not in main]
            if not isolated:
                break
            progressed = False
            for iso in isolated:
                # One search against the whole main component, not one per settlement in
                # it. `_to_any` with no residual is a Dijkstra that stops at the first
                # goal it reaches, which — the frontier being ordered by true cost — is
                # the cheapest goal; that is exactly the argmin the old loop computed by
                # running a full A* to every candidate and throwing all but one away.
                #
                # The old shape was quadratic in settlements and it dominated the stage:
                # at 160x160 organic it was 1,645 A* runs and 74 s of a 101 s stage, and
                # the cost grew as the main component did, so every settlement joined made
                # the next one more expensive to join.
                best_path = _to_any(
                    grid, plain_cost, plain_edge, iso.coord, grid.mask(main & place_coords)
                )
                if best_path:
                    adopt(best_path, tier_for(iso.coord))
                    main |= bfs_component(iso.coord)
                    progressed = True
                    break
            if not progressed:
                # No route to any settlement in the main component, over land or water: a
                # river is always crossable at a price, so what is left is ground no cart
                # can climb to.  Some maps simply are in pieces, and that is a fact about the
                # world rather than a failure of routing; raising would make such a map
                # ungenerable, which is worse than one that honestly shows two networks.
                iso = isolated[0]
                unreachable.append((iso.coord, f"{iso.name} has no route to the network"))
                # Treat its component as settled so the loop moves on to the next one
                # rather than trying the same search again every pass.
                main |= bfs_component(iso.coord)
                main.add(iso.coord)

        return road_edges, unreachable


@dataclass
class _Network:
    """The road stage's search state: the costs, the traffic wearing them down, the roads.

    Traffic is an array rather than a dict so the searches can read it; `worn` records
    which hexes any journey has crossed, the hexes the stage writes a `traffic` to.
    """

    grid: Grid
    node: np.ndarray
    edge: np.ndarray
    factor: float
    traffic: np.ndarray
    worn: np.ndarray
    # Per hex, the directions of the roads out of it, in the order its adjacency set
    # iterates. `routing.along` breaks ties in that order, as the dict search it replaced
    # did, so the order is copied from the set rather than recomputed.
    adj: np.ndarray

    @classmethod
    def of(cls, hexes, cfg, crossings, ring) -> "_Network":
        grid = Grid.of(hexes)
        return cls(
            grid=grid,
            node=road_node_costs(hexes, grid, cfg),
            edge=road_edge_costs(hexes, grid, cfg, crossings, ring),
            factor=float(cfg.road_pheromone_factor),
            traffic=np.zeros(grid.size),
            worn=np.zeros(grid.size, np.bool_),
            adj=np.full((grid.size, 6), -1, np.int64),
        )

    def wear(self, coord: HexCoord, n: float) -> None:
        i = self.grid.index[coord]
        self.traffic[i] += n
        self.worn[i] = True

    def join(self, coord: HexCoord, roads: set[HexCoord]) -> None:
        """Copy *coord*'s road neighbours across, in the order the set holds them."""
        row = self.adj[self.grid.index[coord]]
        row[:] = -1
        for k, other in enumerate(roads):
            row[k] = self.grid.direction(coord, other)


def _to_any(grid, node, edge, start, goals, traffic=None, factor=0.0, goal_cost=None):
    """`hex_grid.astar_to_any` (with no `aim`) over cost arrays: the path to the best goal.

    *goals* is a mask. With *goal_cost* each goal is weighed by what remains from it; without,
    the first goal reached is the cheapest. *traffic* wears the node costs as the road
    network's pheromone does.
    """
    if start not in grid.index or not goals.any():
        return None
    s = grid.index[start]
    if goals[s]:
        return [start]
    mode = routing.FIRST_GOAL if goal_cost is None else routing.GOAL_COST
    came, best = routing.to_any(
        grid.nbr,
        node,
        np.zeros(0) if traffic is None else traffic,
        factor,
        edge,
        s,
        goals,
        np.zeros(0) if goal_cost is None else goal_cost,
        mode,
    )
    return None if best < 0 else routing.walk_back(grid, came, best)
