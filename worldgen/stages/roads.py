from collections import defaultdict, deque

from ..core.hex import Settlement, SettlementTier, TerrainClass
from ..core.hex_grid import astar, distance, neighbors
from ..core.pipeline import GeneratorStage
from ..core.world_state import Road, RoadTier, WorldState
from .road_cost import slope_edge_cost

_TIER_ORDER = [RoadTier.PRIMARY, RoadTier.SECONDARY, RoadTier.TRACK]


def _terrain_base_cost(hx, cfg) -> float:
    tc = hx.terrain_class
    if tc in (TerrainClass.OCEAN, TerrainClass.LAKE):
        return float("inf")
    if tc == TerrainClass.MOUNTAIN:
        return cfg.road_mountain_cost
    if tc == TerrainClass.HILL:
        return cfg.road_hill_cost
    return cfg.road_flat_cost


class RoadStage(GeneratorStage):
    def run(self, state: WorldState) -> WorldState:
        hexes = state.hexes
        cfg = self.config
        settlements = state.settlements
        if not settlements:
            return state

        hex_traffic: dict = defaultdict(float)
        canonical_routes: dict = {}  # (origin, dest) -> path

        # Node cost closure (reads live hex_traffic)
        def node_cost(hx):
            base = _terrain_base_cost(hx, cfg)
            if base == float("inf"):
                return base
            if hx.river_flow > 0:
                base = max(0.1, base - cfg.road_river_discount)
            pheromone = cfg.road_pheromone_factor * hex_traffic[hx.coord]
            return max(0.1, base - pheromone)

        # Edge cost: hyperbolic slope penalty
        def edge_cost(from_hx, to_hx):
            return slope_edge_cost(from_hx, to_hx, cfg)

        # Build traveller list
        tier_counts = {
            SettlementTier.CITY: cfg.road_travellers_city,
            SettlementTier.TOWN: cfg.road_travellers_town,
            SettlementTier.VILLAGE: cfg.road_travellers_village,
        }
        travellers = []
        for s in settlements:
            travellers.extend([s] * tier_counts[s.tier])
        order = self.rng.permutation(len(travellers))

        # Pre-compute populations as floats for gravity model
        pop_arr = [float(s.population) for s in settlements]
        coords_arr = [s.coord for s in settlements]
        n_s = len(settlements)
        s_index = {s.coord: i for i, s in enumerate(settlements)}

        for idx in order:
            origin_s: Settlement = travellers[idx]
            oi = s_index[origin_s.coord]

            # Gravity-weighted destination sampling
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

            # Route — check cache before running A* (most travellers repeat popular pairs)
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

        # Threshold traffic into tiers
        eligible = [c for c, t in hex_traffic.items() if t >= cfg.road_min_traffic]
        eligible.sort(key=lambda c: hex_traffic[c], reverse=True)
        n_elig = len(eligible)
        hex_tier: dict = {}
        if n_elig > 0:
            p_cut = max(1, round(n_elig * cfg.road_primary_pct))
            s_cut = max(p_cut + 1, round(n_elig * (cfg.road_primary_pct + cfg.road_secondary_pct)))
            t_cut = max(
                s_cut + 1,
                round(
                    n_elig * (cfg.road_primary_pct + cfg.road_secondary_pct + cfg.road_track_pct)
                ),
            )
            for i, c in enumerate(eligible):
                if i < p_cut:
                    hex_tier[c] = RoadTier.PRIMARY
                elif i < s_cut:
                    hex_tier[c] = RoadTier.SECONDARY
                elif i < t_cut:
                    hex_tier[c] = RoadTier.TRACK

        # City connectivity guarantee
        cities = [s for s in settlements if s.tier == SettlementTier.CITY]
        fallback_paths: list[list] = []
        if len(cities) > 1:
            hex_tier, fallback_paths = self._guarantee_city_connectivity(
                hexes, cities, hex_tier, cfg
            )

        # Build Road objects from canonical routes
        roads: list[Road] = []
        for path in canonical_routes.values():
            tier = self._path_min_tier(path, hex_tier)
            if tier is not None:
                roads.append(Road(path=path, tier=tier))

        # Add fallback roads for city connectivity (always PRIMARY)
        for path in fallback_paths:
            roads.append(Road(path=path, tier=RoadTier.PRIMARY))

        # Populate hex.road_connections from actual road paths
        for road in roads:
            for a, b in zip(road.path, road.path[1:], strict=False):
                if a in hexes and b in hexes:
                    hexes[a].road_connections.add(b)
                    hexes[b].road_connections.add(a)

        # Ford / bridge tagging — only where a road enters a river hex from a non-river hex
        for road in roads:
            path = road.path
            for i, c in enumerate(path):
                if c not in hexes:
                    continue
                hx = hexes[c]
                if hx.river_flow == 0:
                    continue
                # Only tag the entry point of each river-crossing run
                prev_c = path[i - 1] if i > 0 else None
                prev_hx = hexes.get(prev_c) if prev_c is not None else None
                if prev_hx is None or prev_hx.river_flow == 0:
                    if "ford" not in hx.tags:
                        hx.tags.add("ford")
                    else:
                        hx.tags.discard("ford")
                        hx.tags.add("bridge")

        # Habitability re-score for road adjacency
        road_hex_set = set(hex_tier.keys())
        for coord, hx in hexes.items():
            if hx.settlement is not None:
                continue
            if hx.terrain_class in (TerrainClass.OCEAN, TerrainClass.LAKE):
                continue
            if any(n in road_hex_set for n in neighbors(coord)):
                hx.habitability = min(1.0, hx.habitability + 0.2)

        # Promote villages near roads with high habitability
        # Respect town_min_separation for promoted villages
        town_coords_set = {s.coord for s in settlements if s.tier == SettlementTier.TOWN}
        for s in list(settlements):
            if s.tier != SettlementTier.VILLAGE:
                continue
            hx = hexes[s.coord]
            # Must be adjacent to a road hex (road adjacency re-score skips settled hexes)
            if not any(n in road_hex_set for n in neighbors(s.coord)):
                continue
            if hx.habitability <= 0.8:
                continue
            # Enforce town minimum separation among towns (including already-promoted ones)
            if not all(distance(s.coord, tc) >= cfg.town_min_separation for tc in town_coords_set):
                continue
            s.tier = SettlementTier.TOWN
            s.role = self._assign_role_simple(s.coord, hx, hexes)
            s.population = int(self.rng.integers(1_000, 10_001))
            s.name = s.name.replace("_village_", "_town_")
            town_coords_set.add(s.coord)

        state.roads = roads
        return state

    def _stitch_via_junction(
        self,
        origin: tuple,
        dest: tuple,
        settlement_coords: list,
        canonical_routes: dict,
        hexes: dict,
        node_cost,
        edge_cost,
    ) -> list | None:
        """Try to stitch origin→dest via an intermediate settlement junction.

        If canonical routes exist for both (origin, mid) and (mid, dest),
        concatenate them instead of running a full A* from origin to dest.
        Falls back to A* when no suitable junction is found.
        """
        best_path = None
        best_cost = float("inf")

        def _path_cost(path: list) -> float:
            if not path:
                return float("inf")
            total = node_cost(hexes[path[0]])
            for i in range(1, len(path)):
                total += node_cost(hexes[path[i]])
                total += edge_cost(hexes[path[i - 1]], hexes[path[i]])
            return total

        for mid in settlement_coords:
            if mid in (origin, dest):
                continue
            k1 = (min(origin, mid), max(origin, mid))
            k2 = (min(mid, dest), max(mid, dest))
            if k1 not in canonical_routes or k2 not in canonical_routes:
                continue

            seg1 = canonical_routes[k1]
            seg2 = canonical_routes[k2]

            # Orient seg1 so it ends at mid
            if seg1[-1] == mid:
                s1 = seg1
            elif seg1[0] == mid:
                s1 = list(reversed(seg1))
            else:
                continue

            # Orient seg2 so it starts at mid
            if seg2[0] == mid:
                s2 = seg2
            elif seg2[-1] == mid:
                s2 = list(reversed(seg2))
            else:
                continue

            if s1[0] != origin or s2[-1] != dest:
                continue

            stitched = s1 + s2[1:]
            cost = _path_cost(stitched)
            if cost < best_cost:
                best_path = stitched
                best_cost = cost

        if best_path is not None:
            return best_path

        return astar(hexes, origin, dest, node_cost, edge_cost)

    def _path_min_tier(self, path, hex_tier) -> RoadTier | None:
        tiers = [hex_tier[c] for c in path if c in hex_tier]
        if not tiers:
            return None
        # Return minimum quality (TRACK < SECONDARY < PRIMARY in quality order)
        order = {RoadTier.PRIMARY: 0, RoadTier.SECONDARY: 1, RoadTier.TRACK: 2}
        return max(tiers, key=lambda t: order[t])

    def _guarantee_city_connectivity(self, hexes, cities, hex_tier, cfg):
        """BFS over hex_tier to find isolated cities; connect with plain A*."""

        # Build adjacency from hex_tier
        def bfs_component(start, road_coords):
            visited = {start}
            queue = deque([start])
            while queue:
                c = queue.popleft()
                for n in neighbors(c):
                    if n in road_coords and n not in visited:
                        visited.add(n)
                        queue.append(n)
            return visited

        city_coords = {s.coord for s in cities}
        road_coords = set(hex_tier.keys()) | city_coords

        # Find largest connected component containing at least one city
        visited_global: set = set()
        components = []
        for cc in city_coords:
            if cc in visited_global:
                continue
            comp = bfs_component(cc, road_coords)
            visited_global |= comp
            components.append(comp)

        if not components:
            return hex_tier, []

        # Largest component wins
        main = max(components, key=len)

        # Plain terrain cost for gap-filling
        def plain_cost(hx):
            return _terrain_base_cost(hx, cfg)

        def slope_edge(from_hx, to_hx):
            return slope_edge_cost(from_hx, to_hx, cfg)

        def path_total_cost(p):
            """Compute total movement cost for a path using plain_cost + slope_edge."""
            if not p:
                return float("inf")
            total = plain_cost(hexes[p[0]])
            for i in range(1, len(p)):
                total += plain_cost(hexes[p[i]])
                total += slope_edge(hexes[p[i - 1]], hexes[p[i]])
            return total

        # Connect isolated cities to main component
        fallback_paths: list[list] = []
        max_iter = len(cities) * 2
        iterations = 0
        while iterations < max_iter:
            isolated = [s for s in cities if s.coord not in main]
            if not isolated:
                break
            # Find nearest main-component city by A* movement cost
            for iso in isolated:
                best_path = None
                best_cost = float("inf")
                for target_coord in main & city_coords:
                    p = astar(hexes, iso.coord, target_coord, plain_cost, slope_edge)
                    if p:
                        cost = path_total_cost(p)
                        if cost < best_cost:
                            best_path = p
                            best_cost = cost
                if best_path:
                    for c in best_path:
                        if c not in hex_tier:
                            hex_tier[c] = RoadTier.PRIMARY
                    fallback_paths.append(best_path)
                    # Expand main component
                    main |= bfs_component(iso.coord, set(hex_tier.keys()) | city_coords)
                    break
            iterations += 1

        return hex_tier, fallback_paths

    def _assign_role_simple(self, coord, hx, hexes):
        from ..core.hex import Biome, SettlementRole

        nbrs = [hexes[n] for n in neighbors(coord) if n in hexes]
        if (
            hx.river_flow > 0.5
            or hx.terrain_class == TerrainClass.COAST
            or any(n.river_flow > 0.5 for n in nbrs)
            or any(n.terrain_class == TerrainClass.COAST for n in nbrs)
        ):
            return SettlementRole.PORT
        mountain_nbrs = [n for n in nbrs if n.terrain_class == TerrainClass.MOUNTAIN]
        if mountain_nbrs:
            if any(n.elevation > 0.70 for n in mountain_nbrs):
                return SettlementRole.MINING
            return SettlementRole.FORTRESS
        fertile = sum(1 for n in nbrs if n.biome in (Biome.GRASSLAND, Biome.TEMPERATE_FOREST))
        if fertile >= 3:
            return SettlementRole.AGRICULTURAL
        return SettlementRole.MARKET
