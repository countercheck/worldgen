import heapq
from collections import defaultdict, deque

import numpy as np

from ..core.hex import HexCoord, TerrainClass
from ..core.hex_grid import neighbors
from ..core.pipeline import GeneratorStage
from ..core.world_state import River, WorldState


class HydrologyStage(GeneratorStage):
    def run(self, state: WorldState) -> WorldState:
        w, h = state.width, state.height
        hexes = state.hexes

        # Build elevation array and valid coord set
        elev: dict[HexCoord, float] = {c: hx.elevation for c, hx in hexes.items()}
        land: set[HexCoord] = {
            c for c, hx in hexes.items() if hx.terrain_class != TerrainClass.OCEAN
        }
        ocean: set[HexCoord] = {
            c for c, hx in hexes.items() if hx.terrain_class == TerrainClass.OCEAN
        }

        # A — Priority-Flood sink filling
        filled = self._priority_flood(elev, land, ocean, w, h)
        # Epsilon tilt: hexes farther from ocean get slightly higher filled elevation
        # so that flat plateau areas have a well-defined gradient toward the ocean.
        bfs_dist = self._bfs_dist_from_ocean(filled, ocean, w, h)
        max_dist = max(bfs_dist.values()) or 1
        eps = 1e-6
        for coord in filled:
            q, r = coord
            filled[coord] += eps * bfs_dist.get(coord, max_dist) / max_dist + eps * 1e-4 * (
                q + r
            ) / (w + h)

        # B — Flow direction (steepest descent on filled surface)
        flow_dir = self._flow_direction(filled, land, ocean, w, h)

        # C — Flow accumulation (topological sort)
        acc = self._flow_accumulation(flow_dir, land)

        # D — Extract river hexes above threshold
        land_acc_vals = list(acc.values())
        if not land_acc_vals:
            return state
        threshold_val = np.quantile(land_acc_vals, 1.0 - self.config.river_flow_threshold)
        river_set: set[HexCoord] = {c for c, v in acc.items() if v >= threshold_val}

        max_acc = max(land_acc_vals)
        for coord in river_set:
            hexes[coord].river_flow = acc[coord] / max_acc

        # E — Tag and build River objects
        self._tag_hexes(river_set, flow_dir, hexes, ocean)
        state.rivers = self._build_rivers(river_set, flow_dir, hexes, ocean, acc, max_acc, w, h)

        return state

    def _bfs_dist_from_ocean(
        self,
        filled: dict[HexCoord, float],
        ocean: set[HexCoord],
        w: int,
        h: int,
    ) -> dict[HexCoord, int]:
        """BFS distance from ocean/border for every hex in `filled`."""
        dist: dict[HexCoord, int] = {}
        queue: deque[HexCoord] = deque()
        for coord in ocean:
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
        hexes: dict,
        ocean: set[HexCoord],
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
            if any(nbr in ocean for nbr in neighbors(coord)):
                hx.tags.add("river_mouth")

    def _build_rivers(
        self,
        river_set: set[HexCoord],
        flow_dir: dict[HexCoord, HexCoord | None],
        hexes: dict,
        ocean: set[HexCoord],
        acc: dict[HexCoord, float],
        max_acc: float,
        w: int,
        h: int,
    ) -> list[River]:
        """Trace each headwater downstream to ocean/border.

        If flow_dir stalls before reaching ocean (flat-area artefact), extend the
        path via BFS toward the nearest ocean-adjacent hex.
        """
        rivers: list[River] = []
        headwaters = [c for c in river_set if "headwater" in hexes[c].tags]

        for start in headwaters:
            path: list[HexCoord] = [start]
            visited_path: set[HexCoord] = {start}
            current = start

            while True:
                ds = flow_dir.get(current)
                if ds is None:
                    break
                if ds in ocean:
                    path.append(ds)
                    break
                if ds in visited_path:
                    break
                path.append(ds)
                visited_path.add(ds)
                current = ds

            # If the path stalled without reaching ocean or a grid border, extend via BFS.
            mouth = path[-1]
            mq, mr = mouth
            on_border = mq == 0 or mq == w - 1 or mr == 0 or mr == h - 1
            reached_ocean = any(n in ocean for n in neighbors(mouth)) or mouth in ocean
            if not reached_ocean and not on_border and len(path) > 0:
                extension = self._bfs_to_ocean(mouth, hexes, ocean, visited_path, w, h)
                path.extend(extension)

            if len(path) > 1:
                final_mouth = path[-1]
                mouth_acc = acc.get(final_mouth, acc[start])
                rivers.append(River(hexes=path, flow_volume=mouth_acc / max_acc))

        return rivers

    def _bfs_to_ocean(
        self,
        start: HexCoord,
        hexes: dict,
        ocean: set[HexCoord],
        avoid: set[HexCoord],
        w: int,
        h: int,
    ) -> list[HexCoord]:
        """BFS from start to the nearest ocean-adjacent or border hex, returning the path."""
        from_map: dict[HexCoord, HexCoord | None] = {start: None}
        queue: deque[HexCoord] = deque([start])
        while queue:
            coord = queue.popleft()
            q, r = coord
            on_border = q == 0 or q == w - 1 or r == 0 or r == h - 1
            ocean_adj = any(n in ocean for n in neighbors(coord))
            if (on_border or ocean_adj) and coord != start:
                # Reconstruct path (excluding start)
                path: list[HexCoord] = []
                node: HexCoord | None = coord
                while node is not None and node != start:
                    path.append(node)
                    node = from_map[node]
                return list(reversed(path))
            for nbr in neighbors(coord):
                if nbr not in hexes or nbr in from_map or nbr in avoid:
                    continue
                from_map[nbr] = coord
                queue.append(nbr)
        return []
