from collections import defaultdict, deque

from ..core.hex import SettlementTier, TerrainClass
from ..core.hex_grid import astar, distance, neighbors
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

        # Hexsides the rivers run along — roads may cross a river but never travel down
        # it, so the bank a road takes stays readable. Settlement hexes are exempt.
        blocked = river_edges(state.rivers)
        settled = {s.coord for s in state.settlements}

        def node_cost(hx):
            base = terrain_base_cost(hx, cfg) + river_hex_cost(hx, cfg)
            base *= 1.0 - bank_discount(hx, hexes, cfg)
            return pheromone_discount(base, hex_traffic[hx.coord], cfg)

        edge_cost = make_road_edge_cost(cfg, blocked, settled)

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
                path = astar(hexes, origin_s.coord, dest_coord, node_cost, edge_cost)
                if path is None or len(path) < 2:
                    continue
                canonical_routes[key] = path

            for c in path:
                hex_traffic[c] += 1.0
            for a, b in zip(path, path[1:], strict=False):
                edge_traffic[road_edge_key(a, b)] += 1.0

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
            road_edges, ferries = self._guarantee_connectivity(
                hexes, settlements, road_edges, cfg, blocked, settled
            )
            state.ferries.extend(ferries)

        # Again over the finished network, on tiers this time: the connectivity guarantee
        # lays trunk roads of its own, which can skirt a town like any other.
        route_through_settlements(road_edges, hexes, settled, cfg, blocked)

        prune_orphan_roads(road_edges, settled | {c for f in state.ferries for c in (f.a, f.b)})

        for a, b in road_edges:
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
        return state

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
                # No land route to any settlement in the main component: the river channel
                # cuts this one off. Join it by boat, or fail loudly if none is plausible.
                iso = isolated[0]
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
                ferries.append(ferry)
                for fp in ferry_paths:
                    adopt(fp)
                main |= bfs_component(iso.coord)
                main.add(ferry.a)
                main.add(ferry.b)

        return road_edges, ferries
