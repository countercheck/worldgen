from collections import defaultdict, deque

from ..core.hex import SettlementTier, TerrainClass
from ..core.hex_grid import astar, distance, neighbors
from ..core.pipeline import GeneratorStage
from ..core.world_state import Ferry, RoadTier, WorldState, road_edge_key
from .road_cost import (
    bank_discount,
    ferry_link,
    make_road_edge_cost,
    pheromone_discount,
    river_edges,
    river_hex_cost,
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

        tier_counts = {
            SettlementTier.CITY: cfg.road_travellers_city,
            SettlementTier.TOWN: cfg.road_travellers_town,
        }
        travellers = []
        for s in settlements:
            travellers.extend([s] * tier_counts[s.tier])
        order = self.rng.permutation(len(travellers))

        pop_arr = [float(s.population) for s in settlements]
        coords_arr = [s.coord for s in settlements]
        n_s = len(settlements)
        s_index = {s.coord: i for i, s in enumerate(settlements)}

        for idx in order:
            origin_s = travellers[idx]
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
                path = self._stitch_via_junction(
                    origin_s.coord,
                    dest_coord,
                    coords_arr,
                    canonical_routes,
                    hexes,
                    node_cost,
                    edge_cost,
                )
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
        # River edges use the lower `road_river_traffic_min` threshold so that
        # well-trafficked riverbanks become drawn roads (towpaths, river roads).
        def eligible_edge(key) -> bool:
            t = edge_traffic[key]
            if t >= cfg.road_min_traffic:
                return True
            a, b = key
            on_river = any(c in hexes and hexes[c].river_flow > 0 for c in (a, b))
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

        cities = [s for s in settlements if s.tier == SettlementTier.CITY]
        if len(cities) > 1:
            road_edges, ferries = self._guarantee_city_connectivity(
                hexes, cities, road_edges, canonical_routes, cfg, blocked, settled
            )
            state.ferries.extend(ferries)

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

    def _stitch_via_junction(
        self, origin, dest, settlement_coords, canonical_routes, hexes, node_cost, edge_cost
    ):
        def _path_cost(path):
            if not path:
                return float("inf")
            total = node_cost(hexes[path[0]])
            for i in range(1, len(path)):
                total += node_cost(hexes[path[i]])
                total += edge_cost(hexes[path[i - 1]], hexes[path[i]])
            return total

        best_path = None
        best_cost = float("inf")
        for mid in settlement_coords:
            if mid in (origin, dest):
                continue
            k1 = (min(origin, mid), max(origin, mid))
            k2 = (min(mid, dest), max(mid, dest))
            if k1 not in canonical_routes or k2 not in canonical_routes:
                continue
            seg1 = canonical_routes[k1]
            seg2 = canonical_routes[k2]
            s1 = seg1 if seg1[-1] == mid else (list(reversed(seg1)) if seg1[0] == mid else None)
            s2 = seg2 if seg2[0] == mid else (list(reversed(seg2)) if seg2[-1] == mid else None)
            if s1 is None or s2 is None:
                continue
            if s1[0] != origin or s2[-1] != dest:
                continue
            stitched = s1 + s2[1:]
            cost = _path_cost(stitched)
            if cost < best_cost:
                best_path = stitched
                best_cost = cost

        return (
            best_path if best_path is not None else astar(hexes, origin, dest, node_cost, edge_cost)
        )

    def _guarantee_city_connectivity(
        self, hexes, cities, road_edges, canonical_routes, cfg, blocked, settled
    ):
        """Join any city the traffic model left off the network, by land or by boat.

        Adjacency is the drawn network itself.  It used to be rebuilt from whichever
        canonical routes contributed a tier, which was a second, subtly different answer to
        "what counts as a road" — with edges as the stored form there is only one.
        """
        road_adj: dict = defaultdict(set)
        for a, b in road_edges:
            road_adj[a].add(b)
            road_adj[b].add(a)

        city_coords = {s.coord for s in cities}

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
        for cc in city_coords:
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
        max_iter = len(cities) * 2
        for _ in range(max_iter):
            isolated = [s for s in cities if s.coord not in main]
            if not isolated:
                break
            progressed = False
            for iso in isolated:
                best_path = None
                best_cost = float("inf")
                for target_coord in main & city_coords:
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
                # No land route to any main-component city: the river channel cuts this
                # city off. Join it by boat, or fail loudly if no ferry is plausible.
                iso = isolated[0]
                ferry, ferry_paths = ferry_link(
                    hexes,
                    iso.coord,
                    f"City {iso.name}",
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
