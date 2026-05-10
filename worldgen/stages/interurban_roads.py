from collections import defaultdict, deque

from ..core.hex import SettlementTier, TerrainClass
from ..core.hex_grid import astar, distance, neighbors
from ..core.pipeline import GeneratorStage
from ..core.world_state import Road, RoadTier, WorldState
from .road_cost import (
    river_discount,
    road_edge_cost,
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
        canonical_routes: dict = {}

        def node_cost(hx):
            base = terrain_base_cost(hx, cfg)
            base = max(0.0, base - river_discount(hx, cfg))
            pheromone = cfg.road_pheromone_factor * hex_traffic[hx.coord]
            return max(0.0, base - pheromone)

        def edge_cost(from_hx, to_hx):
            return road_edge_cost(from_hx, to_hx, cfg)

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

        # River hexes use the lower `road_river_traffic_min` threshold so that
        # well-trafficked riverbanks become drawn roads (towpaths, river roads).
        eligible = [
            c
            for c, t in hex_traffic.items()
            if t >= cfg.road_min_traffic
            or (c in hexes and hexes[c].river_flow > 0 and t >= cfg.road_river_traffic_min)
        ]
        eligible.sort(key=lambda c: hex_traffic[c], reverse=True)
        n_elig = len(eligible)
        hex_tier: dict = {}
        if n_elig > 0:
            p_cut = max(1, round(n_elig * cfg.road_primary_pct))
            s_cut = max(
                p_cut + 1,
                round(n_elig * (cfg.road_primary_pct + cfg.road_secondary_pct)),
            )
            for i, c in enumerate(eligible):
                if i < p_cut:
                    hex_tier[c] = RoadTier.PRIMARY
                elif i < s_cut:
                    hex_tier[c] = RoadTier.SECONDARY

        cities = [s for s in settlements if s.tier == SettlementTier.CITY]
        fallback_paths: list[list] = []
        if len(cities) > 1:
            hex_tier, fallback_paths = self._guarantee_city_connectivity(
                hexes, cities, hex_tier, canonical_routes, cfg
            )

        roads: list[Road] = []
        for path in canonical_routes.values():
            tier = self._path_min_tier(path, hex_tier)
            if tier is not None:
                roads.append(Road(path=path, tier=tier))
        for path in fallback_paths:
            roads.append(Road(path=path, tier=RoadTier.PRIMARY))

        for road in roads:
            for a, b in zip(road.path, road.path[1:], strict=False):
                if a in hexes and b in hexes:
                    hexes[a].road_connections.add(b)
                    hexes[b].road_connections.add(a)

        tag_river_crossings(roads, hexes)

        # Re-score habitability near roads so VillagePlacementStage benefits
        road_hex_set = set(hex_tier.keys())
        for coord, hx in hexes.items():
            if hx.settlement is not None:
                continue
            if hx.terrain_class in (TerrainClass.OCEAN, TerrainClass.LAKE):
                continue
            if any(n in road_hex_set for n in neighbors(coord)):
                hx.habitability = min(1.0, hx.habitability + 0.2)

        state.roads = roads
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

    def _path_min_tier(self, path, hex_tier) -> RoadTier | None:
        tiers = [hex_tier[c] for c in path if c in hex_tier]
        if not tiers:
            return None
        order = {RoadTier.PRIMARY: 0, RoadTier.SECONDARY: 1, RoadTier.TRACK: 2}
        return max(tiers, key=lambda t: order[t])

    def _guarantee_city_connectivity(self, hexes, cities, hex_tier, canonical_routes, cfg):
        # Build road adjacency from actual road path edges (not hex-neighbour proximity).
        # Only include paths that contribute a tier (i.e. pass the traffic threshold).
        road_adj: dict = defaultdict(set)
        for path in canonical_routes.values():
            if self._path_min_tier(path, hex_tier) is not None:
                for a, b in zip(path, path[1:], strict=False):
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
            return hex_tier, []
        main = max(components, key=len)

        def plain_cost(hx):
            return terrain_base_cost(hx, cfg)

        def plain_edge(from_hx, to_hx):
            return road_edge_cost(from_hx, to_hx, cfg)

        def path_total_cost(p):
            if not p:
                return float("inf")
            total = plain_cost(hexes[p[0]])
            for i in range(1, len(p)):
                total += plain_cost(hexes[p[i]])
                total += plain_edge(hexes[p[i - 1]], hexes[p[i]])
            return total

        fallback_paths: list[list] = []
        max_iter = len(cities) * 2
        for _ in range(max_iter):
            isolated = [s for s in cities if s.coord not in main]
            if not isolated:
                break
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
                    # Update road adjacency so the next BFS sees the new edges.
                    for a, b in zip(best_path, best_path[1:], strict=False):
                        road_adj[a].add(b)
                        road_adj[b].add(a)
                    for c in best_path:
                        if c not in hex_tier:
                            hex_tier[c] = RoadTier.PRIMARY
                    fallback_paths.append(best_path)
                    main |= bfs_component(iso.coord)
                    break

        return hex_tier, fallback_paths
