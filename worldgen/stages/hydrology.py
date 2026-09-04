import heapq
from collections import defaultdict, deque
from collections.abc import Callable

from ..core.config import CLIMATE_CONTEXTS
from ..core.hex import Hex, HexCoord, TerrainClass
from ..core.hex_grid import distance, neighbors
from ..core.pipeline import GeneratorStage
from ..core.world_state import River, WorldState

# True for hexes on the grid edge, which drain off the map.  Which coordinates those are
# depends on the grid layout, so the test travels as `WorldState.on_border` rather than
# being rebuilt from a width and a height at each site.
#
# A border hex is a valid terminal even when its own steepest descent points back inland
# (see `_flow_direction`), so path tracing must stop *on* it rather than follow it
# inward — including when a path starts there.  The one exception is an inflow inlet,
# which is a border hex water deliberately enters *through*; `_build_rivers` lets a trace
# leave the border only when it starts on one of those.
OnBorder = Callable[[HexCoord], bool]

# The edges a hex lies on — a set, because a corner hex lies on two.  Travels as a
# callable for the same reason `OnBorder` does: the mapping from coordinate to edge is
# the grid layout's business, not this stage's.  Names match
# `WorldConfig.continent_falloff_edges`: west is column 0, north is row 0.
EdgesOf = Callable[[HexCoord], frozenset[str]]


class HydrologyStage(GeneratorStage):
    def run(self, state: WorldState) -> WorldState:
        w, h = state.width, state.height
        on_border = state.on_border
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
        filled = self._priority_flood(elev, land, ocean, on_border)
        # Epsilon tilt: hexes farther from their plateau's drain point get slightly higher
        # filled elevation so that flat plateau areas have a well-defined gradient toward
        # water. Distance is scoped per-plateau (propagated only across equal-elevation
        # neighbors) rather than raw hex-distance-to-ocean — the latter can rank a cell as
        # "closer to water" than its own downhill neighbor, creating a false local minimum
        # that stalls flow_dir in the middle of a plateau.
        drain_dist = self._plateau_drain_distance(filled, land, ocean, lakes, on_border)
        max_dist = max(drain_dist.values()) or 1
        eps = 1e-6
        for coord in filled:
            q, r = coord
            filled[coord] += eps * drain_dist.get(coord, max_dist) / max_dist + eps * 1e-4 * (
                q + r
            ) / (w + h)

        # B — Flow direction (steepest descent on filled surface)
        flow_dir = self._flow_direction(filled, land, ocean, lakes, elev, on_border)

        # B2 — Rivers that arrive from beyond the border.  The map is a region, not a
        # world, so some of its water was gathered off it.  Each inlet is seeded with a
        # catchment it did not earn here, which is what makes it enter already wide.
        inlets = self._inflow_inlets(flow_dir, filled, land, on_border, self._edges_of(state))
        inflow_volume = max(1.0, self.config.river_inflow_volume * len(land))
        inflow = {c: inflow_volume for c in inlets}

        # C — Flow accumulation (topological sort)
        acc = self._flow_accumulation(flow_dir, land, inflow)

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
            river_set,
            flow_dir,
            hexes,
            land,
            ocean,
            lakes,
            acc,
            max_acc,
            filled,
            on_border,
            set(inlets),
        )

        # F — Normalize river_flow; headwater/confluence/mouth tags set from river_set
        if self.config.river_flow_continuous:
            for coord in land:
                hexes[coord].river_flow = acc.get(coord, 0.0) / max_acc
        else:
            for coord in river_set:
                hexes[coord].river_flow = acc.get(coord, 0.0) / max_acc
        self._tag_hexes(river_set, flow_dir, hexes, ocean, lakes, on_border)

        # G — Ensure every lake has an outflow river (fill-to-spillway enforcement)
        drainage_rivers, outlet_of = self._ensure_lake_drainage(
            river_set, flow_dir, hexes, land, ocean, lakes, acc, filled, on_border
        )
        if drainage_rivers:
            state.rivers.extend(drainage_rivers)
        # _ensure_lake_drainage may mutate acc/river_set even when no new rivers are
        # appended (e.g., submerging former river land into lake). Always refresh
        # normalization/tags/flow_volume before confluence splitting.
        max_acc = max(acc.values()) if acc else 1.0
        if self.config.river_flow_continuous:
            for coord in land:
                hexes[coord].river_flow = acc.get(coord, 0.0) / max_acc
        else:
            for coord in river_set:
                hexes[coord].river_flow = acc.get(coord, 0.0) / max_acc
        # Clear stale river tags from hexes submerged into lake during drainage
        # (they were removed from river_set but still carry tags from the first pass)
        _river_tags = {"river", "headwater", "confluence", "river_mouth"}
        for coord, hx in hexes.items():
            if coord not in river_set:
                hx.tags -= _river_tags
        self._tag_hexes(river_set, flow_dir, hexes, ocean, lakes, on_border)
        # An inlet is already tagged a headwater — it has no upstream hex on this map —
        # so this is what separates a river arriving from beyond the border from one
        # rising at a spring inside it.  Gated on river_set rather than on `inlets`
        # alone: _ensure_lake_drainage may have submerged an inlet into a lake, and a
        # lake hex must not be left labelled a river source.
        for coord in inlets:
            if coord in river_set:
                hexes[coord].tags.add("river_source_offmap")
        # Recompute flow_volume for all rivers now that max_acc is final;
        # _ensure_lake_drainage may remove land hexes from acc (submerged into lake)
        # which can change max_acc, making pre-drainage flow_volume values stale.
        for river in state.rivers:
            last_land = next((c for c in reversed(river.hexes) if c in acc), river.hexes[0])
            river.flow_volume = acc.get(last_land, 0.0) / max_acc

        # H — Split source-to-sea paths into source-to-confluence segments.
        # Higher-flow rivers claim their land hexes first; lower-flow tributaries are
        # trimmed at the first already-claimed hex, eliminating duplicate trunk renderings.
        # This runs after all hydrological computation (river_set, acc, flow_dir) is final
        # so that per-hex river_flow and lake drainage connectivity are unaffected.
        state.rivers = _split_at_confluences(state.rivers, land, acc, max_acc)

        # H — Tag every non-water hex in any River path with "river".
        # Done after all rivers (including drainage) are finalized, so drainage tail
        # hexes that aren't in river_set (but are genuine river-path members) are covered.
        water_classes = {TerrainClass.OCEAN, TerrainClass.LAKE}
        for river in state.rivers:
            for coord in river.hexes:
                if coord in hexes and hexes[coord].terrain_class not in water_classes:
                    hexes[coord].tags.add("river")

        # I — Mark basins that still have no way out.  Not every lake can be drained:
        # a bowl ringed by higher ground with no lower lake to spill into is a closed
        # basin, and forcing a river out of it would be a lie about the terrain.  Water
        # leaves such a basin by evaporation instead, so its shore is tagged for
        # BiomeStage to turn into wetland — that is the "percolates out into marshes"
        # outlet, and it keeps the map honest about where the water goes.
        for comp in _endorheic_components(hexes, lakes, ocean, flow_dir, outlet_of, on_border):
            for coord in comp:
                hexes[coord].tags.add("endorheic")
            shore = set(comp)
            frontier = set(comp)
            for _ in range(self.config.endorheic_marsh_radius):
                frontier = {
                    n
                    for c in frontier
                    for n in neighbors(c)
                    if n in hexes and n not in shore and n in land
                }
                if not frontier:
                    break
                shore |= frontier
                for coord in frontier:
                    hexes[coord].tags.add("endorheic_shore")

        return state

    def _plateau_drain_distance(
        self,
        filled: dict[HexCoord, float],
        land: set[HexCoord],
        ocean: set[HexCoord],
        lakes: set[HexCoord],
        on_border: OnBorder,
    ) -> dict[HexCoord, int]:
        """BFS distance from each plateau's own drain point, propagated only across
        neighbors with equal filled elevation.

        A drain point is any land hex already adjacent to water/border, or with a
        neighbor whose filled elevation is strictly lower (i.e. it already has a real
        downhill direction). Restricting propagation to equal-elevation neighbors keeps
        the distance scoped to a single flat plateau, so every interior plateau hex is
        guaranteed a same-elevation neighbor one step closer to its own drain — unlike a
        raw distance-to-ocean measure, this can never rank a hex as "closer" than a
        neighbor it cannot actually reach downhill, so it cannot create a false local
        minimum.
        """
        dist: dict[HexCoord, int] = {}
        queue: deque[HexCoord] = deque()
        for coord in ocean | lakes:
            dist[coord] = 0
            queue.append(coord)
        for coord in land:
            if coord in dist:
                continue
            has_lower_nbr = any(
                nbr in ocean
                or nbr in lakes
                or (nbr in filled and filled[nbr] < filled[coord] - 1e-12)
                for nbr in neighbors(coord)
            )
            if on_border(coord) or has_lower_nbr:
                dist[coord] = 0
                queue.append(coord)
        while queue:
            coord = queue.popleft()
            for nbr in neighbors(coord):
                if nbr not in filled or nbr in dist:
                    continue
                if abs(filled[nbr] - filled[coord]) < 1e-12:
                    dist[nbr] = dist[coord] + 1
                    queue.append(nbr)
        return dist

    def _priority_flood(
        self,
        elev: dict[HexCoord, float],
        land: set[HexCoord],
        ocean: set[HexCoord],
        on_border: OnBorder,
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
            if on_border(coord):
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
        elev: dict[HexCoord, float],
        on_border: OnBorder,
    ) -> dict[HexCoord, HexCoord | None]:
        """For each land hex, flow to the lowest filled neighbor.

        The caller adds an epsilon tilt before calling, so all filled elevations
        are unique — no tie-breaking needed, and the result is cycle-free.

        Priority-Flood does not seed lake hexes, so their filled elevation may be
        raised by the algorithm.  To guarantee that land hexes adjacent to lakes
        still drain into them, we use the raw (pre-flood) elevation for ocean and
        lake neighbors when computing steepest descent.
        """
        water = ocean | lakes
        flow_dir: dict[HexCoord, HexCoord | None] = {}
        for coord in land:
            best_coord: HexCoord | None = None
            best_elev = filled[coord]
            for nbr in neighbors(coord):
                if nbr not in filled:
                    continue
                # Use raw elevation for water hexes so PF-raised lake/ocean values
                # never appear higher than the actual landscape.
                nbr_e = elev[nbr] if nbr in water else filled[nbr]
                if nbr_e < best_elev:
                    best_elev = nbr_e
                    best_coord = nbr

            # A border land hex whose steepest descent leads to another border land hex
            # would produce rivers that creep along the map edge.  Terminate here instead
            # so the border acts as a drain, not a channel.
            if (
                best_coord is not None
                and best_coord not in ocean
                and best_coord not in lakes
                and on_border(coord)
                and on_border(best_coord)
            ):
                best_coord = None

            flow_dir[coord] = best_coord
        return flow_dir

    @staticmethod
    def _edges_of(state: WorldState) -> EdgesOf:
        """Build the coordinate-to-edge-names lookup for *state*'s grid layout."""
        w, h = state.width, state.height

        def edges_of(coord: HexCoord) -> frozenset[str]:
            col, row = state.grid_index(coord)
            found = set()
            if col == 0:
                found.add("west")
            if col == w - 1:
                found.add("east")
            if row == 0:
                found.add("north")
            if row == h - 1:
                found.add("south")
            return frozenset(found)

        return edges_of

    @staticmethod
    def _downstream_lengths(
        flow_dir: dict[HexCoord, HexCoord | None],
        land: set[HexCoord],
    ) -> dict[HexCoord, int]:
        """Land hexes from each hex to wherever its water leaves the map.

        Memoised along each chain, so the whole field costs one pass over `land` rather
        than one trace per hex.  Water hexes and the map edge terminate a chain and are
        not counted, so the number is the length of the river a hex would head.

        `flow_dir` is cycle-free by construction — the caller's epsilon tilt makes every
        filled elevation unique — but a chain that revisits a hex is still terminated
        rather than followed, so a future change to the tilt cannot hang this.
        """
        length: dict[HexCoord, int] = {}
        for start in land:
            if start in length:
                continue
            chain: list[HexCoord] = []
            seen: set[HexCoord] = set()
            current: HexCoord | None = start
            while (
                current is not None
                and current in land
                and current not in length
                and current not in seen
            ):
                seen.add(current)
                chain.append(current)
                current = flow_dir.get(current)
            tail = length.get(current, 0) if current is not None else 0
            for coord in reversed(chain):
                tail += 1
                length[coord] = tail
        return length

    def _inflow_inlets(
        self,
        flow_dir: dict[HexCoord, HexCoord | None],
        filled: dict[HexCoord, float],
        land: set[HexCoord],
        on_border: OnBorder,
        edges_of: EdgesOf,
    ) -> list[HexCoord]:
        """Border hexes where a river enters the map from a catchment beyond it.

        Eligibility is read off `flow_dir` rather than re-derived from elevations, so an
        inlet's water is guaranteed to travel inland by the very field that will route
        it.  Three conditions do the work:

        *   The hex is in `land`, which the caller builds as everything that is neither
            ocean nor lake — so a river can never rise out of open water.
        *   Its downstream hex is in `land` too.  A border hex whose steepest descent runs
            straight into a lake or the sea is a river *mouth*, not a source; admitting one
            would draw a one-hex stub from the edge into the water beside it.
        *   That downstream hex is off the border, so an inflow heads inland instead of
            creeping along the map edge.
        *   The terrain descends inland of it at all, so the hex sits in something that
            drains rather than in a rise against the edge.

        What is left is ranked by how far the water then travels.  Length has to do that
        work rather than the inland drop, which was the obvious choice and the wrong one:
        the drop is a single step's view, it ranges over orders of magnitude, and on its
        own it happily picks a hex that descends steeply inland and meets the sea three
        hexes later — which is most of what a border offers.  So the drop stays a filter,
        and the weight is the course length raised to `river_inflow_length_bias`.

        Weighting alone still leaves stubs, because `river_inflow_min_separation` can
        leave nothing but stubs to draw from once the first inlet is placed, so a course
        shorter than `river_inflow_min_length` is not eligible at all.  A map with no long
        course yields fewer inlets than asked for; importing a river that leaves again
        four hexes later would read as a mistake rather than as geography.

        The course is traced on `flow_dir`, which depends only on elevation — seeding the
        inflow does not change it — so the length weighed here is the length the river
        actually gets.
        """
        count = self.config.river_inflow_count
        wanted_edges = set(self.config.river_inflow_edges)
        if count <= 0 or not wanted_edges:
            return []

        lengths = self._downstream_lengths(flow_dir, land)
        bias = self.config.river_inflow_length_bias
        min_length = self.config.river_inflow_min_length * max(
            self.config.width, self.config.height
        )

        candidates: list[HexCoord] = []
        weights: list[float] = []
        for coord in land:
            if not on_border(coord) or not (edges_of(coord) & wanted_edges):
                continue
            downstream = flow_dir.get(coord)
            if downstream is None or downstream not in land or on_border(downstream):
                continue
            if filled[coord] - filled[downstream] <= 0.0:
                continue
            course = lengths[coord]
            if course < min_length:
                continue
            candidates.append(coord)
            weights.append(float(course) ** bias)

        if not candidates:
            return []

        # Sorted so the sampling order depends only on the seed, never on set iteration
        # order — `land` is a set, and its order is not stable across runs.
        order = sorted(range(len(candidates)), key=lambda i: candidates[i])
        candidates = [candidates[i] for i in order]
        weights = [weights[i] for i in order]

        separation = self.config.river_inflow_min_separation
        chosen: list[HexCoord] = []
        remaining = list(range(len(candidates)))
        while remaining and len(chosen) < count:
            total = sum(weights[i] for i in remaining)
            if total <= 0.0:
                break
            probs = [weights[i] / total for i in remaining]
            pick = remaining[int(self.rng.choice(len(remaining), p=probs))]
            chosen.append(candidates[pick])
            remaining = [
                i
                for i in remaining
                if distance(candidates[i], candidates[pick]) >= separation and i != pick
            ]

        return chosen

    def _flow_accumulation(
        self,
        flow_dir: dict[HexCoord, HexCoord | None],
        land: set[HexCoord],
        inflow: dict[HexCoord, float] | None = None,
    ) -> dict[HexCoord, float]:
        """Topological sort (Kahn's) then accumulate upstream counts.

        Every land hex starts with one unit of rain.  A hex in *inflow* starts with the
        off-map catchment it drains instead, which is what carries a river in over the
        border already large.
        """
        # Build in-degree and downstream map over land only
        in_degree: dict[HexCoord, int] = {c: 0 for c in land}
        downstream: dict[HexCoord, HexCoord | None] = {}

        for coord in land:
            ds = flow_dir.get(coord)
            downstream[coord] = ds
            if ds is not None and ds in land:
                in_degree[ds] += 1

        queue: deque[HexCoord] = deque(c for c in land if in_degree[c] == 0)
        inflow = inflow or {}
        acc: dict[HexCoord, float] = {c: inflow.get(c, 1.0) for c in land}

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
        on_border: OnBorder,
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
            if on_border(coord) or any(nbr in ocean or nbr in lakes for nbr in neighbors(coord)):
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
        on_border: OnBorder,
        inflow_sources: set[HexCoord] | None = None,
    ) -> list[River]:
        """Trace each headwater downstream to ocean/border.

        Headwaters are derived directly from river_set and flow_dir (not from hex tags,
        since _tag_hexes runs after this method). If flow_dir stalls before reaching ocean
        (flat-area artefact), extend the path via elevation-guided search toward the nearest
        outlet; fallback hexes are added to river_set and flow_dir is updated to keep
        all downstream data consistent.

        Paths are built as full source-to-sea traces; split into source-to-confluence
        segments by the caller after all drainage rivers are also available.
        """
        rivers: list[River] = []
        inflow_sources = inflow_sources or set()

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
                # Tested at the top of the loop so a headwater that already sits on the
                # border terminates too, instead of being traced back inland.  An inflow
                # inlet is the one border hex a trace may leave, and only as its own first
                # step: water enters the map there, so stopping would emit a one-hex
                # river.  Any border hex reached later still ends the river.
                if on_border(current) and not (current == start and start in inflow_sources):
                    break
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
            reached_water = (
                mouth in ocean
                or mouth in lakes
                or any(n in ocean or n in lakes for n in neighbors(mouth))
            )
            if not reached_water and not on_border(mouth):
                # Stage 1: valley-preferring, excluding already-visited hexes
                extension = self._guided_path_to_ocean(
                    mouth, filled, land, ocean, lakes, visited_path, on_border
                )
                if not extension:
                    # Stage 2: same elevation-guided search without the avoid constraint
                    extension = self._guided_path_to_ocean(
                        mouth, filled, land, ocean, lakes, set(), on_border
                    )
                if not extension:
                    # Stage 3: plain BFS over any hex — guaranteed to reach a border
                    extension = self._forced_exit_to_border(mouth, hexes, ocean, lakes, on_border)
                if extension:
                    mouth_acc = acc.get(mouth, 1.0)
                    prev = mouth
                    for ext_coord in extension:
                        if ext_coord in land:
                            flow_dir[prev] = ext_coord
                            if ext_coord in river_set:
                                # Merged into existing network; don't inflate its acc.
                                prev = ext_coord
                                break
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
        on_border: OnBorder,
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
            water_adj = any(n in ocean or n in lakes for n in neighbors(coord))
            if (on_border(coord) or water_adj) and coord != start:
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
        on_border: OnBorder,
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
            water_adj = any(n in ocean or n in lakes for n in neighbors(coord))
            if (on_border(coord) or water_adj) and coord != start:
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
        river_set: set[HexCoord],
        flow_dir: dict[HexCoord, HexCoord | None],
        hexes: dict[HexCoord, "Hex"],
        land: set[HexCoord],
        ocean: set[HexCoord],
        lakes: set[HexCoord],
        acc: dict[HexCoord, float],
        filled: dict[HexCoord, float],
        on_border: OnBorder,
    ) -> tuple[list[River], dict[HexCoord, HexCoord | None]]:
        """Raise each lake to its natural spillway, expand into submerged land, then
        route an outflow river.

        Returns the new rivers together with an outlet map: every lake hex maps to the
        land hex its basin drains through, or to None where no outlet could be found at
        all.  The caller uses that map to decide which basins are endorheic.

        Pass A — fill & expand: each lake's water level rises to the elevation of its
        lowest perimeter land hex (the natural spillway).  Any land hex reachable from
        the lake whose raw elevation is below that level is submerged and converted to
        LAKE.  If the expanded body reaches the map edge it becomes OCEAN instead.

        Pass B — outflow: perimeter hexes are tried in ascending elevation order and an
        elevation-guided Dijkstra finds the outflow path, with plain BFS as a fallback.

        The two passes are kept separate, rather than interleaved per basin, because
        expansion merges basins.  Routing a basin before every basin has finished
        expanding means routing against a component that is about to grow — an outflow
        aimed at what was then a lower neighbouring lake ends up pointing into the
        middle of its own basin once the two merge, which reads as a lake that drains
        into itself.  Pass B therefore runs on settled components only.

        Within Pass B basins are handled from the lowest water surface upwards: the
        terminal sink has nowhere lower to spill into and must reach the sea or the map
        edge on its own, so it is settled before anything is allowed to drain towards
        it, and every basin above it then chains into an outflow that already
        terminates.  Acyclicity comes not from that order but from the rule that a basin
        may only spill into a *strictly lower* one — an outflow only ever moves water
        downhill, so no chain of them can return to its source.
        """
        outlet_of: dict[HexCoord, HexCoord | None] = {}
        if not lakes:
            return [], outlet_of

        def reaches_terminal(
            coord: HexCoord,
            component: set[HexCoord] | None = None,
            level: float | None = None,
        ) -> bool:
            """True if water at *coord* already has somewhere to go.

            A lake counts as a terminal, not just the sea or the map edge.  Pass B gives
            every basin its own outlet, so water arriving in one is water this path no
            longer has to carry — and on a landlocked map, where there is no ocean at
            all, insisting on sea-or-border makes this return False for practically
            every river.  That is not a cosmetic difference: the caller uses this to
            decide whether to merge into an existing channel or to rewire it, so a
            false negative makes a lake outflow seize a trunk river's flow_dir and
            reverse the trunk's own course.

            *component* is the basin being drained, and is excluded: a channel that runs
            back into it is a cycle, not an outlet.  *level* is that basin's water
            surface, and a lake at or above it does not count either — Pass A raised
            every lake to its spillway, which can leave an old flow_dir pointing at what
            is now higher water, and accepting that would have a lake drain uphill into
            a puddle above it.  Requiring a strictly lower terminal is also what makes
            the basin graph acyclic, so the escape analysis always settles.
            """

            def is_open_lake(c: HexCoord) -> bool:
                if c not in lakes:
                    return False
                if component is not None and c in component:
                    return False
                return level is None or hexes[c].elevation < level - 1e-12

            seen: set[HexCoord] = set()
            cur = coord
            while cur not in seen:
                seen.add(cur)
                if cur in ocean or on_border(cur):
                    return True
                if is_open_lake(cur):
                    return True
                ds = flow_dir.get(cur)
                if ds is None:
                    return any(n in ocean or is_open_lake(n) for n in neighbors(cur))
                if ds not in land:
                    return ds in ocean or on_border(ds) or is_open_lake(ds)
                cur = ds
            return False

        def bfs_component(seeds: set[HexCoord]) -> set[HexCoord]:
            """BFS-expand *seeds* through the live `lakes` set."""
            comp: set[HexCoord] = set(seeds) & lakes
            queue: deque[HexCoord] = deque(comp)
            while queue:
                c = queue.popleft()
                for nbr in neighbors(c):
                    if nbr in lakes and nbr not in comp:
                        comp.add(nbr)
                        queue.append(nbr)
            return comp

        max_acc = max(acc.values()) if acc else 1.0
        new_rivers: list[River] = []
        processed: set[HexCoord] = set()

        # --- Pass A: fill every basin to its spillway and expand into submerged land ---
        # Sorted for determinism regardless of set iteration order.
        for seed in sorted(lakes):
            if seed in processed:
                continue

            # Derive the current connected component from the live lakes set
            component = bfs_component({seed})

            # Sort by raw elevation so we find the true geographic spillway, not the
            # priority-flood-adjusted one.
            border_land = sorted(
                {nbr for c in component for nbr in neighbors(c) if nbr in land},
                key=lambda c: hexes[c].elevation,
            )
            if not border_land:
                processed |= component
                outlet_of.update(dict.fromkeys(component))
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

            # Recompute full component from live lakes set after expansion:
            # newly submerged hexes may bridge previously separate lake components.
            component = bfs_component(component | newly_submerged)

            # Raise the stored elevation of all lake hexes (including originals) to water_level
            for c in component:
                hexes[c].elevation = water_level
                filled[c] = routing_level

            # Mark entire (post-expansion) component as processed so we never revisit
            # any hex that was merged in (e.g. via a previously separate lake component)
            processed |= component

            # If the expanded body now touches the map edge it is ocean, not a lake
            if any(on_border(c) for c in component):
                for c in component:
                    hexes[c].terrain_class = TerrainClass.OCEAN
                    ocean.add(c)
                    lakes.discard(c)
                continue

        # --- Pass B: route an outflow for every settled basin ---
        # Components are re-derived now that Pass A has finished merging them, so a
        # basin can no longer be handed an outlet that expansion later swallows.
        components = _get_lake_components(lakes, hexes)
        # Pass A left every hex of a basin at the same water level, so any one of them
        # reports it.  Lowest basin first: the terminal sink is the one basin that has
        # nowhere lower to spill into and must reach the sea or the map edge on its own,
        # so it is settled before anything is allowed to drain towards it.  Every basin
        # above it then chains into an outflow that already terminates, instead of
        # aiming at a neighbour whose own fate is still unknown.  Acyclicity comes from
        # the strictly-lower rule below, not from the order, so this is safe.
        components.sort(key=lambda comp: (hexes[min(comp)].elevation, min(comp)))
        basin_index = {c: i for i, comp in enumerate(components) for c in comp}

        for basin_id, component in enumerate(components):
            water_level = hexes[min(component)].elevation

            border_land = sorted(
                {nbr for c in component for nbr in neighbors(c) if nbr in land},
                key=lambda c: filled.get(c, float("inf")),
            )
            if not border_land:
                outlet_of.update(dict.fromkeys(component))
                continue

            # Whether this basin is closed is a water balance, not a shape.  What arrives
            # is every river mouth on its shore plus the rain falling on the open water,
            # counted in the unit `_flow_accumulation` gives one hex of land.  What leaves
            # without a river is evaporation off that same surface.  A basin taking in
            # more than it evaporates must overflow, and is given an outlet below — by
            # force, if the terrain makes it awkward.  One that evaporates everything
            # reaching it is genuinely closed, and cutting a channel out of it would
            # invent a river that should not exist.
            #
            # This is why the Caspian is closed and Baikal is not, and it replaces a test
            # the routing was making by accident: a basin used to come out closed when
            # path-finding happened to fail on it, so a dry basin with an easy saddle
            # drained while a wet one ringed by hills did not — backwards on both counts.
            basin_inflow = sum(
                acc.get(c, 0.0) for c in border_land if flow_dir.get(c) in component
            ) + float(len(component))
            evaporation = (
                self.config.endorheic_evaporation_scale
                * CLIMATE_CONTEXTS[self.config.regional_climate].evaporation
                * len(component)
            )
            if basin_inflow <= evaporation:
                outlet_of.update(dict.fromkeys(component))
                continue

            # Check if a natural outflow already exists (river leaving the lake).
            # Following flow_dir a single step is not enough: a perimeter hex belonging
            # to an *inflow* river also points at a land hex outside the component (the
            # next hex on its way to the shore), which reads as an outflow and skips
            # drainage for the basin entirely.  Walk the full flow path instead, and
            # require that it actually escapes rather than merely leaving the component.
            def drains_out_of(
                start: HexCoord,
                component: set[HexCoord] = component,
                level: float = water_level,
            ) -> bool:
                seen: set[HexCoord] = set()
                cur = start
                while cur not in seen:
                    seen.add(cur)
                    if cur in component:
                        return False  # returns to the lake: an inflow, not an outflow
                    ds = flow_dir.get(cur)
                    if ds is None:
                        break
                    cur = ds
                return reaches_terminal(start, component, level)

            natural_outlet = next(
                (c for c in border_land if c in river_set and drains_out_of(c)), None
            )
            if natural_outlet is not None:
                outlet_of.update(dict.fromkeys(component, natural_outlet))
                continue

            # Try spillways in elevation order; Dijkstra prefers valleys.
            # Use an empty lake set so drainage terminates only at ocean/border —
            # stopping at another lake adjacency would create trivial cyclic routes.
            # Exclude perimeter hexes that already flow *into* this lake (inflow mouths):
            # picking one as the "spillway" would reroute its flow_dir away from the lake,
            # silently severing the inflow without actually producing a usable new
            # outflow river (the rerouted hex gets reclaimed by the original, higher-flow
            # inflow river during confluence-splitting and the new path is dropped).
            outflow_candidates = [c for c in border_land if flow_dir.get(c) not in component]
            if not outflow_candidates:
                # Closed bowl: every rim hex drains inward, so there is no rim hex that
                # is not an inflow.  Taking the lowest one (the old fallback) picks the
                # *trunk* inflow mouth, because the biggest river carves the lowest gap
                # in the rim.  Routing an outflow from there rewires that hex's flow_dir
                # away from the lake, severing the inflow, and the resulting river is
                # then dropped by confluence-splitting when the trunk reclaims the hex —
                # leaving the basin with rivers flowing in and nothing flowing out.
                # Prefer a rim hex carrying little or no flow instead: it is nearly as
                # low, and routing through it destroys no existing channel.
                clean_rim = [c for c in border_land if c not in river_set]
                if not clean_rim:
                    # Every rim hex already carries a river into the lake.  There is no
                    # hex left that an outflow could use without taking over a channel
                    # that flows the other way, and a hex cannot carry water both in and
                    # out.  This basin is closed: record it as having no outlet and let
                    # the endorheic pass turn its shore to marsh.
                    outlet_of.update(dict.fromkeys(component))
                    continue
                outflow_candidates = sorted(
                    clean_rim,
                    key=lambda c: (acc.get(c, 0.0), filled.get(c, float("inf")), c),
                )
            # Prefer candidates that aren't *also* adjacent to a different lake: a
            # spillway sitting right on another lake's shore makes the two basins
            # topologically ambiguous (does this hex drain lake A or sit on lake B's
            # perimeter?), which can produce an outflow path that is real but looks
            # like it immediately loops back into a neighboring basin.
            clean_candidates = [
                c
                for c in outflow_candidates
                if not any(nbr in lakes and nbr not in component for nbr in neighbors(c))
            ]
            if clean_candidates:
                outflow_candidates = clean_candidates
            # Also keep the search from routing *through* any other still-active inflow
            # hex further along the path — same corruption risk as above, just not at
            # the very first step.  Only the candidate currently being tried is exempt:
            # subtracting the whole candidate list would, in the border_land fallback
            # above, clear every perimeter inflow at once and let a route from one
            # candidate rewire another.
            # Every land hex that drains into this basin, found by walking flow_dir
            # backwards from the shore.  Routing the outflow through any of them would
            # send the water straight back where it came from: one step upstream of the
            # lake is obvious, but a hex twenty steps up a tributary is just as much a
            # return path, and only avoiding the immediate shore lets the route merge
            # into a river that curls back into the same lake.
            catchment: set[HexCoord] = set()
            # Seeded from the shore rather than by scanning every land hex: flow_dir
            # points at a neighbour, so anything draining *directly* into the basin is
            # already on its perimeter.  Scanning the whole land set found the same
            # hexes at the cost of a full-map sweep for every basin.
            stack = [c for c in border_land if flow_dir.get(c) in component]
            while stack:
                c = stack.pop()
                if c in catchment:
                    continue
                catchment.add(c)
                stack.extend(
                    n
                    for n in neighbors(c)
                    if n in land and n not in catchment and flow_dir.get(n) == c
                )

            # Basins that this one may legitimately spill into: any lake whose surface
            # sits strictly below this lake's water level.  Draining into a lower basin
            # is a real drainage pattern (a chain of lakes stepping down to the sea) and
            # is the only outlet available at all on a landlocked map.  The strict
            # elevation test is what keeps the lake-to-lake graph acyclic — an outflow
            # can only ever move water downhill, so it can never route back into a basin
            # upstream of itself.
            lower_lakes: frozenset[HexCoord] = frozenset()
            if self.config.lake_chaining:
                lower_lakes = frozenset(
                    c
                    for c in lakes
                    if basin_index.get(c) != basin_id and hexes[c].elevation < water_level - 1e-12
                )

            def route_escapes(route: list[HexCoord]) -> bool:
                """True if *route* is an outflow rather than a way back into the lake.

                The builder below stops at the first hex that already carries water and
                joins it rather than stealing it, so the route a basin actually gets is
                this path only as far as that hex — and from there it is the other
                channel's course, not ours.  If that channel runs back into this basin
                the result is a lake draining into itself, which the endorheic pass then
                reports as closed.  Such a route was never an outflow, so it is rejected
                while the other candidates are still in hand, rather than three passes
                later once they have all been passed over.
                """
                merge_at = next((c for c in route if c in land and c in river_set), None)
                return merge_at is None or drains_out_of(merge_at)

            # The catchment is excluded first because an outflow that climbs its own
            # inflow valley, while not wrong, reads badly.  But excluding it means
            # excluding the basin's whole watershed — 6674 hexes on the map this was
            # found on, a tenth of the grid — and where the only way out lies through it
            # the search comes back empty and the basin falls through to the unguided
            # fallback below, which ignores elevation and will happily carry the river
            # over a mountain.  So try again without the exclusion before resorting to
            # that.  It is a preference, not a correctness rule: what actually keeps the
            # water from running back where it came from is `route_escapes`, which tests
            # the route rather than guessing at it from the terrain.
            extension: list[HexCoord] = []
            spillway: HexCoord | None = None
            for avoid_catchment in (True, False):
                for candidate in outflow_candidates:
                    avoid = (catchment - {candidate}) if avoid_catchment else set()
                    route = self._guided_path_to_ocean(
                        candidate, filled, land, ocean, lower_lakes, avoid, on_border
                    )
                    if route and route_escapes(route):
                        extension, spillway = route, candidate
                        break
                if extension:
                    break

            # Fallback: plain BFS, which ignores elevation and will carry a river over a
            # mountain to reach the border.  The balance above already established that
            # this water has to get out somehow, so the violence is warranted.
            if not extension:
                spillway = outflow_candidates[0]
                extension = self._forced_exit_to_border(
                    spillway, hexes, ocean, lower_lakes, on_border
                )

            if not extension or spillway is None:
                outlet_of.update(dict.fromkeys(component))
                continue
            outlet_of.update(dict.fromkeys(component, spillway))

            path = [spillway]
            prev = spillway
            # What leaves the basin is what arrived in it, less what evaporated on the
            # way — the same two quantities the balance above is decided on.  Seeding the
            # outflow with the spillway's own drainage instead, usually 1.0 for a single
            # hex of rain, is why a lake fed by eighteen rivers used to drain through a
            # channel carrying 0.004 of the map's flow: the exporters scale river width by
            # flow_volume, so the outlet drew as a hairline beside the torrents feeding it
            # and the basin looked stoppered even though it was, on paper, draining.
            running_acc = max(acc.get(spillway, 0.0), basin_inflow - evaporation, 1.0)
            added_land = [spillway]
            merged_into_existing = False
            for coord in extension:
                if coord not in land:
                    path.append(coord)
                    continue
                if coord in river_set:
                    # The route has reached a channel that already carries water.  Join
                    # it here and stop.  Continuing would rewire this hex's flow_dir to
                    # point along our route instead of its own, which does not add an
                    # outflow so much as reverse an existing river: everything below the
                    # stolen hex loses its upstream, and a trunk hex downstream of it is
                    # left looking like a headwater carrying the whole catchment.
                    # Whether this channel ultimately escapes is not decided here — the
                    # endorheic pass settles that once every basin has been routed.
                    merged_into_existing = True
                    flow_dir[prev] = coord
                    path.append(coord)
                    merge_acc = acc.get(coord, running_acc)
                    if running_acc > merge_acc:
                        # If the channel we joined already carries less flow than this
                        # path, clamp the newly added upstream cells down to the merge
                        # value so downstream accumulation never decreases.
                        for added in added_land:
                            acc[added] = min(acc.get(added, merge_acc), merge_acc)
                    prev = coord
                    break
                flow_dir[prev] = coord
                path.append(coord)
                river_set.add(coord)
                new_val = max(acc.get(coord, 0.0), running_acc)
                acc[coord] = new_val
                running_acc = new_val
                prev = coord
                added_land.append(coord)

            if merged_into_existing:
                seen = {prev}
                tail = prev
                while True:
                    # Tested at the top so a merge point already on the border stops
                    # here, rather than following its inland-pointing flow_dir.
                    if on_border(tail):
                        break
                    ds = flow_dir.get(tail)
                    if ds is None or ds in seen:
                        break
                    path.append(ds)
                    seen.add(ds)
                    if ds not in land:
                        break
                    acc[ds] = max(acc.get(ds, 0.0), acc.get(tail, 0.0))
                    tail = ds
            # _guided_path_to_ocean walks over land only, so a path that stopped because
            # it reached a lower lake ends on the shore hex beside it.  Append the lake
            # hex itself so the river visually enters the basin it feeds, and point
            # flow_dir at it so the chain is walkable for the escape analysis below.
            if lower_lakes and path[-1] in land:
                touching = sorted(
                    (n for n in neighbors(path[-1]) if n in lower_lakes),
                    key=lambda n: (hexes[n].elevation, n),
                )
                if touching:
                    flow_dir[path[-1]] = touching[0]
                    path.append(touching[0])

            river_set.add(spillway)
            spillway_acc = max(acc.get(spillway, 0.0), 1.0)
            if merged_into_existing:
                spillway_acc = min(spillway_acc, acc.get(prev, spillway_acc))
            acc[spillway] = spillway_acc
            last_land = next((c for c in reversed(path) if c in acc), spillway)
            if len(path) > 1:
                new_rivers.append(River(hexes=path, flow_volume=acc[last_land] / max_acc))

        return new_rivers, outlet_of


def _split_at_confluences(
    rivers: list[River],
    land: set[HexCoord],
    acc: dict[HexCoord, float],
    max_acc: float,
) -> list[River]:
    """Trim intersecting tributaries to include their confluence hex.

    Higher-flow rivers process first and claim their land hexes.  A subsequent river
    that intersects claimed land is cut at the first claimed land hex after its
    headwater, and that confluence hex is kept so the tributary visually connects to
    the trunk.  Rivers that never intersect claimed land (typically highest-flow
    trunks) keep their full source-to-mouth path.  Rivers that shrink below 2 hexes
    are dropped.  Original list order is preserved in the output.

    This runs after all hydrological computation is final so that per-hex river_flow,
    flow_dir, and lake drainage connectivity are unaffected.
    """
    # Sort by descending flow_volume; use hexes[0] as a tie-breaker for determinism.
    indexed = sorted(enumerate(rivers), key=lambda iv: (-iv[1].flow_volume, iv[1].hexes[0]))

    claimed: set[HexCoord] = set()
    result: list[tuple[int, River]] = []

    for orig_idx, river in indexed:
        path = river.hexes
        # If the headwater itself is already claimed, this river is fully subsumed by a
        # higher-flow trunk that already covers its start — drop it entirely.
        if path[0] in land and path[0] in claimed:
            continue
        # Find the first claimed land hex after the headwater (index 0 always kept).
        cut = len(path)
        intersects = False
        for i, coord in enumerate(path[1:], 1):
            if coord in land and coord in claimed:
                cut = i + 1  # for an intersecting tributary, include the confluence hex
                intersects = True
                break
        trimmed = path[:cut]
        if len(trimmed) >= 2:
            # Recalculate flow_volume from the last exclusive land hex in the trimmed path.
            # An intersecting tributary ends on the shared confluence hex, whose accumulation
            # already includes the trunk and any other branches — excluding it keeps the
            # tributary rendered at its own pre-merge discharge.
            exclusive = trimmed[:-1] if intersects else trimmed
            last_land = next((c for c in reversed(exclusive) if c in acc), trimmed[0])
            result.append(
                (orig_idx, River(hexes=trimmed, flow_volume=acc.get(last_land, 0.0) / max_acc))
            )
            claimed.update(c for c in trimmed if c in land)

    result.sort(key=lambda iv: iv[0])
    return [r for _, r in result]


def _endorheic_components(
    hexes: dict[HexCoord, "Hex"],
    lakes: set[HexCoord],
    ocean: set[HexCoord],
    flow_dir: dict[HexCoord, HexCoord | None],
    outlet_of: dict[HexCoord, HexCoord | None],
    on_border: OnBorder,
) -> list[set[HexCoord]]:
    """Return the lake components that have no outlet at all.

    A basin is endorheic when nothing drains out of it: `_ensure_lake_drainage` found no
    outlet, or the one it found leads back into the same basin.  Rivers run in and
    nothing runs out, so the water leaves by evaporation instead — the caller marks the
    shore as wetland to show where it goes.  Real basins do this (the Caspian, the Great
    Salt Lake, Lake Chad), so they are reported rather than forced open.

    Having an outlet is the whole test; where that outlet's water ends up is not.  A
    lake that drains into a closed basin still has a river flowing out of it and is not
    itself endorheic, any more than the Volga is endorheic for ending in the Caspian.
    Requiring the chain to reach the sea or the map edge would mark every lake upstream
    of a closed basin as closed too, which on a landlocked map is every lake there is.
    """
    components = _get_lake_components(lakes, hexes)
    index = {c: i for i, comp in enumerate(components) for c in comp}

    def follow(start: HexCoord) -> int | None:
        """Walk flow_dir from *start*; return -1 for escape, else the basin reached."""
        seen: set[HexCoord] = set()
        cur: HexCoord | None = start
        while cur is not None and cur not in seen:
            seen.add(cur)
            if cur in ocean or on_border(cur):
                return -1
            if cur in index:
                return index[cur]
            cur = flow_dir.get(cur)
        return None

    drains = [False] * len(components)
    for i, comp in enumerate(components):
        if any(on_border(c) for c in comp) or any(n in ocean for c in comp for n in neighbors(c)):
            drains[i] = True
            continue
        outlet = next((outlet_of.get(c) for c in sorted(comp) if outlet_of.get(c)), None)
        if outlet is None:
            continue
        reached = follow(outlet)
        if reached == -1:  # the sea or the map edge
            drains[i] = True
        elif reached is not None and reached != i:
            # Spilling into another basin only counts if that basin is genuinely lower.
            # Routing and merging both enforce this, but without the same test here an
            # outlet that ends uphill is scored as drainage, and the terminal sink — the
            # one basin that really has nowhere to go — is recorded as draining into a
            # neighbour perched above it.
            here = hexes[min(comp)].elevation
            there = hexes[min(components[reached])].elevation
            drains[i] = there < here - 1e-12

    return [comp for i, comp in enumerate(components) if not drains[i]]


def _get_lake_components(lakes: set[HexCoord], hexes: dict[HexCoord, "Hex"]) -> list[set[HexCoord]]:
    """Return a list of connected lake components via BFS.

    Seeds are sorted by coordinate for deterministic ordering across runs.
    """
    visited: set[HexCoord] = set()
    components: list[set[HexCoord]] = []
    for seed in sorted(lakes):
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
