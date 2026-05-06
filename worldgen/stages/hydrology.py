import heapq
from collections import defaultdict, deque

from ..core.hex import Hex, HexCoord, TerrainClass
from ..core.hex_grid import neighbors
from ..core.pipeline import GeneratorStage
from ..core.world_state import River, WorldState


class HydrologyStage(GeneratorStage):
    def run(self, state: WorldState) -> WorldState:
        w, h = state.width, state.height
        hexes = state.hexes

        # Build elevation array and valid coord set
        elev: dict[HexCoord, float] = {c: hx.elevation for c, hx in hexes.items()}
        ocean: set[HexCoord] = {
            c for c, hx in hexes.items() if hx.terrain_class == TerrainClass.OCEAN
        }
        lakes: set[HexCoord] = {
            c for c, hx in hexes.items() if hx.terrain_class == TerrainClass.LAKE
        }
        land: set[HexCoord] = {
            c
            for c, hx in hexes.items()
            if hx.terrain_class not in (TerrainClass.OCEAN, TerrainClass.LAKE)
        }

        # A — Priority-Flood sink filling
        filled = self._priority_flood(elev, land, ocean, w, h)
        # Epsilon tilt: hexes farther from ocean/lake get slightly higher filled elevation
        # so that flat plateau areas have a well-defined gradient toward water.
        bfs_dist = self._bfs_dist_from_ocean(filled, ocean, lakes, w, h)
        max_dist = max(bfs_dist.values()) or 1
        eps = 1e-6
        for coord in filled:
            q, r = coord
            filled[coord] += eps * bfs_dist.get(coord, max_dist) / max_dist + eps * 1e-4 * (
                q + r
            ) / (w + h)

        # B — Flow direction (steepest descent on filled surface)
        flow_dir = self._flow_direction(filled, land, ocean, lakes, w, h)

        # C — Flow accumulation (topological sort)
        acc = self._flow_accumulation(flow_dir, land)

        # D — Extract river hexes: top threshold fraction by flow accumulation count.
        # Sorting by accumulation and slicing avoids tie-boundary over-selection that
        # quantile + >= causes when many cells share the cutoff value.
        land_acc_vals = list(acc.values())
        if not land_acc_vals:
            return state
        threshold = max(0.0, min(1.0, self.config.river_flow_threshold))
        if threshold == 0.0:
            state.rivers = []
            return state
        n_river = max(1, round(len(land_acc_vals) * threshold))
        sorted_by_acc = sorted(acc.keys(), key=lambda c: acc[c], reverse=True)
        river_set: set[HexCoord] = set(sorted_by_acc[:n_river])

        max_acc = max(land_acc_vals)

        # E — Build River objects (may extend river_set via fallback for stalled rivers)
        state.rivers = self._build_rivers(
            river_set, flow_dir, hexes, land, ocean, lakes, acc, max_acc, filled, w, h
        )

        # F — Normalize river_flow and tag all river hexes (including any added by fallback)
        for coord in river_set:
            hexes[coord].river_flow = acc.get(coord, 0.0) / max_acc
        self._tag_hexes(river_set, flow_dir, hexes, ocean, lakes, w, h)

        # G — Ensure every lake has an outflow river (fill-to-spillway enforcement)
        lake_components = _get_lake_components(lakes, hexes)
        drainage_rivers = self._ensure_lake_drainage(
            lake_components, river_set, flow_dir, hexes, land, ocean, lakes, acc, filled, w, h
        )
        if drainage_rivers:
            state.rivers.extend(drainage_rivers)
            # Re-normalize and re-tag after any new river hexes were added
            max_acc = max(acc.values()) if acc else 1.0
            for coord in river_set:
                hexes[coord].river_flow = acc.get(coord, 0.0) / max_acc
            self._tag_hexes(river_set, flow_dir, hexes, ocean, lakes, w, h)

        return state

    def _bfs_dist_from_ocean(
        self,
        filled: dict[HexCoord, float],
        ocean: set[HexCoord],
        lakes: set[HexCoord],
        w: int,
        h: int,
    ) -> dict[HexCoord, int]:
        """BFS distance from ocean/lake/border for every hex in `filled`."""
        dist: dict[HexCoord, int] = {}
        queue: deque[HexCoord] = deque()
        for coord in ocean | lakes:
            dist[coord] = 0
            queue.append(coord)
        for coord in filled:
            q, r = coord
            if coord not in dist and (q == 0 or q == w - 1 or r == 0 or r == h - 1):
                dist[coord] = 0
                queue.append(coord)
        while queue:
            coord = queue.popleft()
            for nbr in neighbors(coord):
                if nbr in filled and nbr not in dist:
                    dist[nbr] = dist[coord] + 1
                    queue.append(nbr)
        return dist

    def _priority_flood(
        self,
        elev: dict[HexCoord, float],
        land: set[HexCoord],
        ocean: set[HexCoord],
        w: int,
        h: int,
    ) -> dict[HexCoord, float]:
        """Barnes et al. Priority-Flood: fill closed depressions on land."""
        filled = dict(elev)
        visited: set[HexCoord] = set()
        heap: list[tuple[float, HexCoord]] = []

        # Seed with all ocean hexes and grid-border land hexes
        for coord in ocean:
            heapq.heappush(heap, (filled[coord], coord))
            visited.add(coord)

        for coord in land:
            q, r = coord
            if q == 0 or q == w - 1 or r == 0 or r == h - 1:
                heapq.heappush(heap, (filled[coord], coord))
                visited.add(coord)

        while heap:
            e, coord = heapq.heappop(heap)
            for nbr in neighbors(coord):
                if nbr not in filled or nbr in visited:
                    continue
                visited.add(nbr)
                filled[nbr] = max(filled[nbr], e)
                heapq.heappush(heap, (filled[nbr], nbr))

        return filled

    def _flow_direction(
        self,
        filled: dict[HexCoord, float],
        land: set[HexCoord],
        ocean: set[HexCoord],
        lakes: set[HexCoord],
        w: int,
        h: int,
    ) -> dict[HexCoord, HexCoord | None]:
        """For each land hex, flow to the lowest filled neighbor.

        The caller adds an epsilon tilt before calling, so all filled elevations
        are unique — no tie-breaking needed, and the result is cycle-free.
        """
        flow_dir: dict[HexCoord, HexCoord | None] = {}
        for coord in land:
            best_coord: HexCoord | None = None
            best_elev = filled[coord]
            for nbr in neighbors(coord):
                if nbr not in filled:
                    continue
                nbr_e = filled[nbr]
                if nbr_e < best_elev:
                    best_elev = nbr_e
                    best_coord = nbr

            # A border land hex whose steepest descent leads to another border land hex
            # would produce rivers that creep along the map edge.  Terminate here instead
            # so the border acts as a drain, not a channel.
            if best_coord is not None and best_coord not in ocean and best_coord not in lakes:
                q, r = coord
                bq, br = best_coord
                if (q == 0 or q == w - 1 or r == 0 or r == h - 1) and (
                    bq == 0 or bq == w - 1 or br == 0 or br == h - 1
                ):
                    best_coord = None

            flow_dir[coord] = best_coord
        return flow_dir

    def _flow_accumulation(
        self,
        flow_dir: dict[HexCoord, HexCoord | None],
        land: set[HexCoord],
    ) -> dict[HexCoord, float]:
        """Topological sort (Kahn's) then accumulate upstream counts."""
        # Build in-degree and downstream map over land only
        in_degree: dict[HexCoord, int] = {c: 0 for c in land}
        downstream: dict[HexCoord, HexCoord | None] = {}

        for coord in land:
            ds = flow_dir.get(coord)
            downstream[coord] = ds
            if ds is not None and ds in land:
                in_degree[ds] += 1

        queue: deque[HexCoord] = deque(c for c in land if in_degree[c] == 0)
        acc: dict[HexCoord, float] = {c: 1.0 for c in land}

        while queue:
            coord = queue.popleft()
            ds = downstream[coord]
            if ds is not None and ds in land:
                acc[ds] += acc[coord]
                in_degree[ds] -= 1
                if in_degree[ds] == 0:
                    queue.append(ds)

        return acc

    def _tag_hexes(
        self,
        river_set: set[HexCoord],
        flow_dir: dict[HexCoord, HexCoord | None],
        hexes: dict[HexCoord, "Hex"],
        ocean: set[HexCoord],
        lakes: set[HexCoord],
        w: int,
        h: int,
    ) -> None:
        # upstream river neighbors count
        upstream_river_nbrs: dict[HexCoord, int] = defaultdict(int)
        for coord in river_set:
            ds = flow_dir.get(coord)
            if ds is not None and ds in river_set:
                upstream_river_nbrs[ds] += 1

        for coord in river_set:
            hx = hexes[coord]
            up_count = upstream_river_nbrs[coord]
            if up_count == 0:
                hx.tags.add("headwater")
            if up_count >= 2:
                hx.tags.add("confluence")
            q, r = coord
            on_border = q == 0 or q == w - 1 or r == 0 or r == h - 1
            if on_border or any(nbr in ocean or nbr in lakes for nbr in neighbors(coord)):
                hx.tags.add("river_mouth")

    def _build_rivers(
        self,
        river_set: set[HexCoord],
        flow_dir: dict[HexCoord, HexCoord | None],
        hexes: dict[HexCoord, Hex],
        land: set[HexCoord],
        ocean: set[HexCoord],
        lakes: set[HexCoord],
        acc: dict[HexCoord, float],
        max_acc: float,
        filled: dict[HexCoord, float],
        w: int,
        h: int,
    ) -> list[River]:
        """Trace each headwater downstream to ocean/border.

        Headwaters are derived directly from river_set and flow_dir (not from hex tags,
        since _tag_hexes runs after this method). If flow_dir stalls before reaching ocean
        (flat-area artefact), extend the path via elevation-guided search toward the nearest
        outlet; fallback hexes are added to river_set and flow_dir is updated to keep
        all downstream data consistent.
        """
        rivers: list[River] = []

        # Compute headwaters without relying on tags: any river hex with no upstream river hex
        has_upstream: set[HexCoord] = set()
        for c in river_set:
            ds = flow_dir.get(c)
            if ds is not None and ds in river_set:
                has_upstream.add(ds)
        headwaters = [c for c in river_set if c not in has_upstream]

        for start in headwaters:
            path: list[HexCoord] = [start]
            visited_path: set[HexCoord] = {start}
            current = start

            while True:
                ds = flow_dir.get(current)
                if ds is None:
                    break
                if ds in ocean or ds in lakes:
                    path.append(ds)
                    break
                if ds in visited_path:
                    break
                path.append(ds)
                visited_path.add(ds)
                current = ds

            # If the path stalled without reaching ocean or a grid border, extend via
            # elevation-guided search.  Fallback hexes are registered in river_set and
            # flow_dir is updated so that subsequent tagging is consistent.
            mouth = path[-1]
            mq, mr = mouth
            on_border = mq == 0 or mq == w - 1 or mr == 0 or mr == h - 1
            reached_water = (
                mouth in ocean
                or mouth in lakes
                or any(n in ocean or n in lakes for n in neighbors(mouth))
            )
            if not reached_water and not on_border:
                # Stage 1: valley-preferring, excluding already-visited hexes
                extension = self._guided_path_to_ocean(
                    mouth, filled, land, ocean, lakes, visited_path, w, h
                )
                if not extension:
                    # Stage 2: same elevation-guided search without the avoid constraint
                    extension = self._guided_path_to_ocean(
                        mouth, filled, land, ocean, lakes, set(), w, h
                    )
                if not extension:
                    # Stage 3: plain BFS over any hex — guaranteed to reach a border
                    extension = self._forced_exit_to_border(mouth, hexes, ocean, lakes, w, h)
                if extension:
                    # Carry the stalled mouth's accumulation through the fallback segment.
                    # Use max(natural_acc, mouth_acc) to avoid lowering the river_flow of
                    # hexes that are also traversed by natural river paths (e.g. confluences).
                    # This keeps river_flow non-decreasing along both fallback and natural paths.
                    # acc[mouth] is always present for land hexes; 1.0 (minimum accumulation)
                    # is the safe fallback in case mouth somehow isn't in acc.
                    mouth_acc = acc.get(mouth, 1.0)
                    # Update flow_dir along fallback path so _tag_hexes sees coherent state
                    prev = mouth
                    for ext_coord in extension:
                        if ext_coord in land:
                            flow_dir[prev] = ext_coord
                            river_set.add(ext_coord)
                            acc[ext_coord] = max(acc.get(ext_coord, 0.0), mouth_acc)
                            prev = ext_coord
                    path.extend(extension)

            if len(path) > 1:
                # Use the last land hex for flow_volume — path[-1] may be an ocean hex
                # which has no accumulation value.
                last_land = next((c for c in reversed(path) if c in acc), start)
                rivers.append(River(hexes=path, flow_volume=acc[last_land] / max_acc))

        return rivers

    def _guided_path_to_ocean(
        self,
        start: HexCoord,
        filled: dict[HexCoord, float],
        land: set[HexCoord],
        ocean: set[HexCoord],
        lakes: set[HexCoord],
        avoid: set[HexCoord],
        w: int,
        h: int,
    ) -> list[HexCoord]:
        """Elevation-guided Dijkstra over land hexes from *start* toward the nearest
        water-adjacent or border hex.

        Unlike a plain BFS, uphill movement is penalised heavily so the path stays in
        valleys and does not cross ridgelines or enter water tiles.
        """
        dist: dict[HexCoord, float] = {start: 0.0}
        from_map: dict[HexCoord, HexCoord | None] = {start: None}
        heap: list[tuple[float, HexCoord]] = [(0.0, start)]

        while heap:
            cost, coord = heapq.heappop(heap)
            if cost > dist[coord]:
                continue
            q, r = coord
            on_border = q == 0 or q == w - 1 or r == 0 or r == h - 1
            water_adj = any(n in ocean or n in lakes for n in neighbors(coord))
            if (on_border or water_adj) and coord != start:
                path: list[HexCoord] = []
                node: HexCoord | None = coord
                while node is not None and node != start:
                    path.append(node)
                    node = from_map[node]
                return list(reversed(path))
            for nbr in neighbors(coord):
                if nbr not in land or nbr in avoid:
                    continue
                # Penalise uphill movement to keep rivers in valleys
                elev_penalty = max(0.0, filled.get(nbr, 0.0) - filled.get(coord, 0.0)) * 1000.0
                new_cost = cost + 1.0 + elev_penalty
                if new_cost < dist.get(nbr, float("inf")):
                    dist[nbr] = new_cost
                    from_map[nbr] = coord
                    heapq.heappush(heap, (new_cost, nbr))
        return []

    def _forced_exit_to_border(
        self,
        start: HexCoord,
        hexes: dict[HexCoord, "Hex"],
        ocean: set[HexCoord],
        lakes: set[HexCoord],
        w: int,
        h: int,
    ) -> list[HexCoord]:
        """Plain BFS over all hexes (land and water) to the nearest border or water-adjacent hex.

        No elevation penalty, no avoid set — guaranteed to find a path on any finite connected grid.
        Used only when both elevation-guided passes in _guided_path_to_ocean fail.
        Uses a parent-map to reconstruct the path, avoiding O(V·L) memory cost.
        """
        came_from: dict[HexCoord, HexCoord | None] = {start: None}
        queue: deque[HexCoord] = deque([start])
        while queue:
            coord = queue.popleft()
            q, r = coord
            on_border = q == 0 or q == w - 1 or r == 0 or r == h - 1
            water_adj = any(n in ocean or n in lakes for n in neighbors(coord))
            if (on_border or water_adj) and coord != start:
                path: list[HexCoord] = []
                cur: HexCoord = coord
                while cur != start:
                    path.append(cur)
                    parent = came_from[cur]
                    assert parent is not None
                    cur = parent
                path.reverse()
                return path
            for nbr in neighbors(coord):
                if nbr in hexes and nbr not in came_from:
                    came_from[nbr] = coord
                    queue.append(nbr)
        return []

    def _ensure_lake_drainage(
        self,
        lake_components: list[set[HexCoord]],
        river_set: set[HexCoord],
        flow_dir: dict[HexCoord, HexCoord | None],
        hexes: dict[HexCoord, "Hex"],
        land: set[HexCoord],
        ocean: set[HexCoord],
        lakes: set[HexCoord],
        acc: dict[HexCoord, float],
        filled: dict[HexCoord, float],
        w: int,
        h: int,
    ) -> list[River]:
        """Raise each lake to its natural spillway, expand into submerged land, then
        route an outflow river.

        Phase 1 — fill & expand: the lake's water level rises to the elevation of the
        lowest perimeter land hex (the natural spillway).  Any land hex reachable from
        the lake whose raw elevation is below that level is submerged and converted to
        LAKE.  If the expanded body reaches the map edge it becomes OCEAN instead.

        Phase 2 — outflow: after expansion, perimeter hexes are retried in ascending
        elevation order; elevation-guided Dijkstra finds the outflow path, with plain
        BFS as a guaranteed fallback.
        """
        if not lake_components:
            return []

        def on_border(coord: HexCoord) -> bool:
            q, r = coord
            return q == 0 or q == w - 1 or r == 0 or r == h - 1

        max_acc = max(acc.values()) if acc else 1.0
        new_rivers: list[River] = []

        for component in lake_components:
            # --- Phase 1: fill lake to spillway level, expand into submerged land ---

            # Sort by raw elevation so we find the true geographic spillway, not the
            # priority-flood-adjusted one.
            border_land = sorted(
                {nbr for c in component for nbr in neighbors(c) if nbr in land},
                key=lambda c: hexes[c].elevation,
            )
            if not border_land:
                continue

            spillway_hex = border_land[0]
            water_level = hexes[spillway_hex].elevation  # lake surface rises to here
            routing_level = filled[spillway_hex]  # filled value kept for Dijkstra gradient

            # Flood-fill: find all land hexes reachable from the lake below water_level
            newly_submerged: set[HexCoord] = set()
            expand_q: deque[HexCoord] = deque(component)
            expand_seen: set[HexCoord] = set(component)
            while expand_q:
                c = expand_q.popleft()
                for nbr in neighbors(c):
                    if nbr not in hexes or nbr in expand_seen:
                        continue
                    if nbr in land and hexes[nbr].elevation < water_level:
                        newly_submerged.add(nbr)
                        expand_seen.add(nbr)
                        expand_q.append(nbr)

            # Convert submerged land hexes to lake
            for c in newly_submerged:
                hexes[c].terrain_class = TerrainClass.LAKE
                hexes[c].elevation = water_level
                hexes[c].river_flow = 0.0
                filled[c] = routing_level
                land.discard(c)
                lakes.add(c)
                river_set.discard(c)
                flow_dir.pop(c, None)
                acc.pop(c, None)

            component = component | newly_submerged

            # Raise the stored elevation of all lake hexes (including originals) to water_level
            for c in component:
                hexes[c].elevation = water_level
                filled[c] = routing_level

            # If the expanded body now touches the map edge it is ocean, not a lake
            if any(on_border(c) for c in component):
                for c in component:
                    hexes[c].terrain_class = TerrainClass.OCEAN
                    ocean.add(c)
                    lakes.discard(c)
                continue

            # --- Phase 2: route outflow river from the spillway ---

            # Recompute perimeter after expansion (newly submerged hexes change the boundary)
            border_land = sorted(
                {nbr for c in component for nbr in neighbors(c) if nbr in land},
                key=lambda c: filled.get(c, float("inf")),
            )
            if not border_land:
                continue

            # Check if a natural outflow already exists (river leaving the lake)
            outflow_exists = any(
                c in river_set and flow_dir.get(c) not in component and flow_dir.get(c) is not None
                for c in border_land
            )
            if outflow_exists:
                continue

            # Try spillways in elevation order; Dijkstra prefers valleys
            extension: list[HexCoord] = []
            spillway: HexCoord | None = None
            for candidate in border_land:
                extension = self._guided_path_to_ocean(
                    candidate, filled, land, ocean, lakes, set(), w, h
                )
                if extension:
                    spillway = candidate
                    break

            # Guaranteed fallback: plain BFS
            if not extension:
                spillway = border_land[0]
                extension = self._forced_exit_to_border(spillway, hexes, ocean, lakes, w, h)

            if not extension or spillway is None:
                continue

            path = [spillway] + extension
            mouth_acc = acc.get(spillway, 1.0)
            prev = spillway
            for coord in extension:
                if coord in land:
                    flow_dir[prev] = coord
                    river_set.add(coord)
                    acc[coord] = max(acc.get(coord, 0.0), mouth_acc)
                    prev = coord
            river_set.add(spillway)
            acc[spillway] = max(acc.get(spillway, 0.0), 1.0)
            last_land = next((c for c in reversed(path) if c in acc), spillway)
            new_rivers.append(River(hexes=path, flow_volume=acc[last_land] / max_acc))

        return new_rivers


def _get_lake_components(lakes: set[HexCoord], hexes: dict[HexCoord, "Hex"]) -> list[set[HexCoord]]:
    """Return a list of connected lake components via BFS."""
    visited: set[HexCoord] = set()
    components: list[set[HexCoord]] = []
    for seed in lakes:
        if seed in visited:
            continue
        component: set[HexCoord] = {seed}
        queue: deque[HexCoord] = deque([seed])
        while queue:
            coord = queue.popleft()
            for nbr in neighbors(coord):
                if nbr in lakes and nbr not in component:
                    component.add(nbr)
                    queue.append(nbr)
        visited |= component
        components.append(component)
    return components
