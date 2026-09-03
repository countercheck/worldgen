from collections import defaultdict, deque
from heapq import heappop, heappush

from ..core.errors import RoutingError
from ..core.hex import SettlementTier, TerrainClass
from ..core.hex_grid import astar, astar_to_any, distance, neighbors
from ..core.pipeline import GeneratorStage
from ..core.world_state import Ferry, RoadTier, WorldState, road_edge_key
from .road_cost import (
    add_traffic,
    bank_discount,
    ferry_link,
    is_river,
    make_road_edge_cost,
    pheromone_discount,
    prune_orphan_roads,
    river_edges,
    river_hex_cost,
    route_through_settlements,
    settlement_rings,
    tag_river_crossings,
    terrain_base_cost,
)


class InterurbanRoadStage(GeneratorStage):
    """Builds PRIMARY and SECONDARY roads between cities and towns only.

    Runs before village placement so that villages can use road corridors
    as placement candidates.
    """

    def run(self, state: WorldState) -> WorldState:
        hexes = state.hexes
        cfg = self.config
        # Only city and town settlements participate
        settlements = [
            s for s in state.settlements if s.tier in (SettlementTier.CITY, SettlementTier.TOWN)
        ]
        if not settlements:
            return state

        hex_traffic: dict = defaultdict(float)
        edge_traffic: dict = defaultdict(float)
        canonical_routes: dict = {}
        # The network as it grows, so a route can aim at the road rather than the town.
        net_adj: dict = defaultdict(set)
        # Costing the road home means a Dijkstra over the network per route, and the
        # network barely changes once the trunks are down — so keep the answer and rebuild
        # only when an edge has actually been added since it was worked out.
        net_version = [0]
        home_cache: dict = {}

        # Hexsides the rivers run along — roads may cross a river but never travel down
        # it, so the bank a road takes stays readable. Settlement hexes are exempt.
        blocked = river_edges(state.rivers)
        settled = {s.coord for s in state.settlements}
        # Which seats each hex neighbours, so an edge can be charged for skirting one.
        ring = settlement_rings(settled)

        def node_cost(hx):
            base = terrain_base_cost(hx, cfg) + river_hex_cost(hx, cfg)
            base *= 1.0 - bank_discount(hx, hexes, cfg)
            return pheromone_discount(base, hex_traffic[hx.coord], cfg)

        edge_cost = make_road_edge_cost(cfg, blocked, settled, ring)

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
            dest_coord = coords_arr[di]

            key = (min(origin_s.coord, dest_coord), max(origin_s.coord, dest_coord))
            if key in canonical_routes:
                path = canonical_routes[key]
            else:
                path = self._route(
                    hexes,
                    origin_s.coord,
                    dest_coord,
                    net_adj,
                    node_cost,
                    edge_cost,
                    home_cache,
                    net_version[0],
                )
                if path is None or len(path) < 2:
                    continue
                canonical_routes[key] = path

            for c in path:
                hex_traffic[c] += 1.0
            for a, b in zip(path, path[1:], strict=False):
                edge_traffic[road_edge_key(a, b)] += 1.0
                if b not in net_adj[a]:
                    net_adj[a].add(b)
                    net_adj[b].add(a)
                    net_version[0] += 1

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
        route_through_settlements(edge_traffic, hexes, settled, cfg, blocked, combine=add_traffic)

        # River edges use the lower `road_river_traffic_min` threshold so that
        # well-trafficked riverbanks become drawn roads (towpaths, river roads).
        def eligible_edge(key) -> bool:
            t = edge_traffic[key]
            if t >= cfg.road_min_traffic:
                return True
            # By the tag, not by `river_flow > 0`. Under `river_flow_continuous` hydrology
            # writes a flow value onto every draining land hex, so a flow test calls the
            # whole map a river and admits every quiet edge on it. The costs have always
            # read the tag for this reason; eligibility was still reading the flow, and
            # only got away with it while stitching kept traffic high enough everywhere
            # that almost nothing sat in the one-to-two band where the two disagree.
            on_river = any(c in hexes and is_river(hexes[c]) for c in key)
            return on_river and t >= cfg.road_river_traffic_min

        eligible = sorted(
            (k for k in edge_traffic if eligible_edge(k)),
            key=lambda k: (-edge_traffic[k], k),
        )
        # `road_min_traffic` decides what is a road; the percentiles only decide how it is
        # drawn. They used to do both, which is why raising it from 3 to 1000 moved road
        # coverage by 0.4% of the map — it shrank the eligible set, and the percentiles
        # promptly re-cut the same fractions of whatever survived. Everything eligible is
        # now drawn, and a quiet lane is a TRACK rather than nothing at all.
        road_edges: dict = {}
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
            road_edges, ferries, unreachable = self._guarantee_connectivity(
                hexes, settlements, road_edges, cfg, blocked, settled
            )
            state.ferries.extend(ferries)
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
            if any(hexes[c].terrain_class in (TerrainClass.OCEAN, TerrainClass.LAKE) for c in key)
        }
        for key in sea_edges:
            del road_edges[key]

        # Where land can join two places, roads should. Sea carriage is so much cheaper
        # than land that routes will cross a bay rather than walk round it, which is right
        # for a journey and wrong for a network: it left the reference map as forty land
        # networks tied together by eight sea crossings, so a cart could not get from one
        # market to the next without a boat. This adds what the traffic model declined to.
        self._join_by_land(hexes, settlements, road_edges, cfg, blocked, settled)

        # Last of all, on tiers: the connectivity guarantee and the land join both lay
        # roads of their own, and either can skirt a town like any other route.
        route_through_settlements(road_edges, hexes, settled, cfg, blocked)

        anchors = settled | {c for f in state.ferries for c in (f.a, f.b)}
        # A land network reaching no settlement is a residue of the traffic threshold; one
        # reaching only a shore is a road to a harbour, which is a road to somewhere.
        anchors |= {c for key in sea_edges for c in key}
        prune_orphan_roads(road_edges, anchors)

        for a, b in list(road_edges) + list(sea_edges):
            if a in hexes and b in hexes:
                hexes[a].road_connections.add(b)
                hexes[b].road_connections.add(a)

        tag_river_crossings(road_edges, hexes)

        # Re-score habitability near roads so VillagePlacementStage benefits.  Only the
        # village score: cities and towns are already sited by this point, and a road
        # they caused should not retroactively flatter the ground it runs over.
        road_hex_set = {c for edge in road_edges for c in edge}
        for coord, hx in hexes.items():
            if hx.settlement is not None:
                continue
            if hx.terrain_class in (TerrainClass.OCEAN, TerrainClass.LAKE):
                continue
            if any(n in road_hex_set for n in neighbors(coord)):
                hx.habitability_village = min(1.0, hx.habitability_village + 0.2)

        state.road_edges = road_edges
        state.sea_edges = sea_edges
        return state

    def _route(self, hexes, origin, dest, net_adj, node_cost, edge_cost, cache, version):
        """The journey from *origin* to *dest*: a new leg, then the road that already goes there.

        A traveller bound for a town does not need a road of his own all the way, he needs
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
        cheapest journey: a traveller would join at his own doorstep and follow the network
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
        # way and becomes the road that everyone after him joins.
        cached = cache.get(dest)
        if cached is not None and cached[0] == version:
            tree, home_cost = cached[1], cached[2]
        else:
            tree, home_cost = self._road_home_tree(hexes, dest, net_adj, node_cost, edge_cost)
            cache[dest] = (version, tree, home_cost)

        def road_home(node):
            out = []
            while node is not None:
                out.append(node)
                node = tree[node]
            return out

        # No short circuit when the origin is already on the network. It is tempting — there
        # is a road home, so take it — but that is `_stitch_via_junction`'s mistake in
        # another guise, committing to an existing route without weighing it against a
        # direct one. The origin is itself a goal reached at no cost, so the network route
        # is the search's opening candidate and is beaten only if striking out pays.
        leg = astar_to_any(hexes, origin, set(tree), node_cost, edge_cost, goal_cost=home_cost)
        if leg is None:
            return None
        return leg + road_home(tree[leg[-1]])

    @staticmethod
    def _join_by_land(hexes, places, road_edges, cfg, blocked, settled):
        """Join settlements that share a landmass but not a road, by road.

        The traffic model has no reason to build these: a traveller crossing a bay is doing
        the sensible thing, since sea carriage cost a fraction of land carriage. But a
        network in which neighbouring markets can only be reached by boat is not a road
        network, and a wargame cannot march down it.

        Land only, deliberately — the cost function refuses water outright, so this cannot
        satisfy itself with the sea leg that already exists.
        """
        water = (TerrainClass.OCEAN, TerrainClass.LAKE)

        def land_cost(hx):
            if hx.terrain_class in water:
                return float("inf")
            return terrain_base_cost(hx, cfg) + river_hex_cost(hx, cfg)

        land_edge = make_road_edge_cost(cfg, blocked, settled)

        # Which settlements the ground itself connects, ignoring roads entirely.
        dry = {c for c, hx in hexes.items() if hx.terrain_class not in water}
        seats = [s.coord for s in places if s.coord in dry]
        landmass: dict = {}
        for seat in seats:
            if seat in landmass:
                continue
            stack, seen = [seat], set()
            while stack:
                c = stack.pop()
                if c in seen:
                    continue
                seen.add(c)
                stack.extend(n for n in neighbors(c) if n in dry and n not in seen)
            for c in seen & set(seats):
                landmass[c] = seat

        def road_component(start, edges):
            adj: dict = defaultdict(set)
            for a, b in edges:
                adj[a].add(b)
                adj[b].add(a)
            seen, stack = {start}, [start]
            while stack:
                c = stack.pop()
                for n in adj[c]:
                    if n not in seen:
                        seen.add(n)
                        stack.append(n)
            return seen

        by_land: dict = defaultdict(list)
        for seat in seats:
            by_land[landmass[seat]].append(seat)

        added = 0
        for group in by_land.values():
            if len(group) < 2:
                continue
            joined = road_component(group[0], road_edges)
            for seat in group[1:]:
                if seat in joined:
                    continue
                target = min(
                    (c for c in joined if c in dry), key=lambda c: distance(seat, c), default=None
                )
                if target is None:
                    continue
                path = astar(hexes, seat, target, land_cost, land_edge)
                if not path or len(path) < 2:
                    continue
                for a, b in zip(path, path[1:], strict=False):
                    road_edges.setdefault(road_edge_key(a, b), RoadTier.TRACK)
                joined |= set(path)
                added += 1
        return added

    @staticmethod
    def _road_home_tree(hexes, dest, net_adj, node_cost, edge_cost):
        """Every hex the network can reach *dest* from, with the way back and its cost."""
        tree: dict = {dest: None}
        home_cost: dict = {dest: 0.0}
        if dest not in net_adj:
            return tree, home_cost
        queue = [(0.0, dest)]
        while queue:
            cost, c = heappop(queue)
            if cost > home_cost.get(c, float("inf")):
                continue
            for n in net_adj[c]:
                if n not in hexes:
                    continue
                step = node_cost(hexes[n]) + edge_cost(hexes[c], hexes[n])
                if step == float("inf"):
                    continue
                if cost + step < home_cost.get(n, float("inf")):
                    home_cost[n] = cost + step
                    tree[n] = c
                    heappush(queue, (cost + step, n))
        return tree, home_cost

    def _guarantee_connectivity(self, hexes, places, road_edges, cfg, blocked, settled):
        """Join any settlement the traffic model left off the network, by land or by boat.

        Adjacency is the drawn network itself.  It used to be rebuilt from whichever
        canonical routes contributed a tier, which was a second, subtly different answer to
        "what counts as a road" — with edges as the stored form there is only one.
        """
        road_adj: dict = defaultdict(set)
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

        visited_global: set = set()
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

        def plain_cost(hx):
            return terrain_base_cost(hx, cfg) + river_hex_cost(hx, cfg)

        plain_edge = make_road_edge_cost(cfg, blocked, settled)

        def path_total_cost(p):
            if not p:
                return float("inf")
            total = plain_cost(hexes[p[0]])
            for i in range(1, len(p)):
                total += plain_cost(hexes[p[i]])
                total += plain_edge(hexes[p[i - 1]], hexes[p[i]])
            return total

        def adopt(path) -> None:
            """Lay *path* into the network as primary, without demoting anything."""
            for a, b in zip(path, path[1:], strict=False):
                road_adj[a].add(b)
                road_adj[b].add(a)
                road_edges.setdefault(road_edge_key(a, b), RoadTier.PRIMARY)

        ferries: list[Ferry] = []
        # Settlements the terrain puts beyond both road and ferry. Reported, not raised.
        unreachable: list = []
        max_iter = len(places) * 2
        for _ in range(max_iter):
            isolated = [s for s in places if s.coord not in main]
            if not isolated:
                break
            progressed = False
            for iso in isolated:
                best_path = None
                best_cost = float("inf")
                for target_coord in main & place_coords:
                    p = astar(hexes, iso.coord, target_coord, plain_cost, plain_edge)
                    if p:
                        cost = path_total_cost(p)
                        if cost < best_cost:
                            best_path = p
                            best_cost = cost
                if best_path:
                    adopt(best_path)
                    main |= bfs_component(iso.coord)
                    progressed = True
                    break
            if not progressed:
                # No land route to any settlement in the main component: the channel cuts
                # this one off. Join it by boat where a boat is plausible — and where it is
                # not, leave it apart. Some maps simply are in pieces: an island beyond
                # ferry range cannot be reached by road, and that is a fact about the world
                # rather than a failure of routing. Raising there would make an archipelago
                # ungenerable, which is worse than a map that honestly shows two networks.
                iso = isolated[0]
                try:
                    ferry, ferry_paths = ferry_link(
                        hexes,
                        iso.coord,
                        iso.name,
                        main,
                        cfg,
                        blocked,
                        settled,
                        plain_cost,
                        plain_edge,
                    )
                except RoutingError as exc:
                    unreachable.append((iso.coord, str(exc)))
                    # Treat its component as settled so the loop moves on to the next one
                    # rather than trying this same crossing again every pass.
                    main |= bfs_component(iso.coord)
                    main.add(iso.coord)
                    continue
                ferries.append(ferry)
                for fp in ferry_paths:
                    adopt(fp)
                main |= bfs_component(iso.coord)
                main.add(ferry.a)
                main.add(ferry.b)

        return road_edges, ferries, unreachable
