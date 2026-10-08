"""Hydrology, part two: the rivers and the lakes.

River tracing on the hex model with its three fallback strategies, lake filling, expansion
and drainage, the endorheic-basin test, and the routing of the map's rivers along hexsides
on the corner graph.  `HydrologyStage` (in `hydrology.py`) mixes this in; the fields it
reads come from `hydrology_fill.py`.
"""

import heapq
from collections import defaultdict, deque
from typing import TYPE_CHECKING

import numpy as np

from ..core.hex import Hex, HexCoord, TerrainClass
from ..core.hex_grid import (
    Corner,
    Side,
    corner_hexes,
    neighbors,
    side_hexes,
    side_joining,
)
from ..core.world_state import River, RiverSide, WorldState
from .corner_drainage import (
    Node,
    accumulate,
    build_network,
    drain_corner,
    flow_direction,
    inlet_corner,
    is_lake_node,
    trace_streams,
)
from .hydrology_fill import OnBorder
from .precipitation import rain_per_hex

if TYPE_CHECKING:
    from ..core.config import WorldConfig


class HydrologyRivers:
    """River tracing, lake drainage and corner-graph routing for `HydrologyStage`.

    A mixin: `config` and `rng` are the stage's own, set by `GeneratorStage.__init__`.
    """

    config: "WorldConfig"
    rng: np.random.Generator

    def _route_on_corners(
        self,
        state: WorldState,
        inlets: list[HexCoord],
        closed: list[set[HexCoord]],
        min_catchment: float,
    ) -> None:
        """Route the map's rivers on the corner graph and record them.

        Everything up to here worked on hexes, to settle the ground: which hollows hold
        lakes, at what level, and which of them overflow.  The water that falls on that
        ground runs along the sides between hexes, so this drains the corner graph (see
        `corner_drainage`) over the settled ground and writes what it finds — the courses
        into `state.rivers`, each side a river runs along into `state.river_sides`, and its
        sources, ends, mouths and confluences into `state.river_corners`.
        """
        hexes = state.hexes
        water = (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER)
        land = {c for c, hx in hexes.items() if hx.terrain_class not in water}
        ocean = {c for c, hx in hexes.items() if hx.terrain_class == TerrainClass.OPEN_WATER}
        lakes = {c for c, hx in hexes.items() if hx.terrain_class == TerrainClass.INLAND_WATER}
        closed_lakes = {c for comp in closed for c in comp} & lakes
        open_lakes = _get_lake_components(lakes - closed_lakes, hexes)

        heights = {c: hx.elevation for c, hx in hexes.items()}
        net = build_network(
            heights, land, ocean, closed_lakes, open_lakes, self.config.corner_floor_blend
        )
        drainage = flow_direction(net, self.rng, self.config.river_wander_exponent)

        # Rain runs off each hex to its lowest corner, and off an open lake out of it.
        rain = rain_per_hex(state, self.config, land)
        sources: dict[Node, float] = defaultdict(float)
        drain_of: dict[HexCoord, Corner] = {}
        for coord in sorted(land):
            corner = drain_corner(coord, net, drainage)
            if corner is not None:
                drain_of[coord] = corner
                sources[corner] += rain.get(coord, 1.0)
        for node, comp in net.lake_hexes.items():
            sources[node] += sum(rain.get(c, 1.0) for c in comp)
        # A river arriving over the border carries a catchment it did not gather here.
        inflow_volume = max(1.0, self.config.river_inflow_volume * len(land))
        inflow: set[Corner] = set()
        for coord in inlets:
            corner = inlet_corner(coord, net, drainage)
            if corner is not None:
                sources[corner] += inflow_volume
                inflow.add(corner)

        drainage.acc = accumulate(drainage.flow, sources)
        acc = drainage.acc
        # A lake that overflows has a river leaving it, however little spills over.
        outlets = [drainage.flow[n] for n in sorted(net.lake_hexes) if drainage.flow.get(n)]
        streams = trace_streams(drainage, min_catchment, forced=outlets)
        max_acc = max((a for n, a in acc.items() if not is_lake_node(n)), default=1.0) or 1.0

        rivers: list[River] = []
        sides: dict[Side, RiverSide] = {}
        corner_tags: dict[Corner, set[str]] = defaultdict(set)
        lake_outlets = set(outlets)
        for path in streams.paths:
            for a, b in zip(path, path[1:], strict=False):
                sides[side_joining(a, b)] = RiverSide(
                    catchment_km2=acc[a],
                    flow=acc[a] / max_acc,
                    drop_m=max(0.0, net.elevation[a] - net.elevation[b]),
                )
            rivers.append(River(corners=path, flow_volume=acc[path[-2]] / max_acc))
            first, last = path[0], path[-1]
            if first in inflow:
                corner_tags[first].add("river_source_offmap")
            elif first not in lake_outlets:
                corner_tags[first].add("river_source")
            # A course that ends on a junction flows on as the trunk; any other end is
            # where its water leaves the river network — the sea, a lake, the map edge.
            onward = drainage.flow.get(last)
            if onward is None or is_lake_node(onward) or onward not in streams.channel:
                corner_tags[last].add("river_end")
                if onward is None or is_lake_node(onward):
                    corner_tags[last].add("river_mouth")
        # Two rivers reaching the same stretch of shore have both arrived somewhere; they
        # have not met.  A confluence is inland.
        for corner, feeders in streams.upstream.items():
            ashore = any(h in hexes and h not in land for h in corner_hexes(corner))
            if len(feeders) >= 2 and not ashore:
                corner_tags[corner].add("confluence")

        state.rivers = rivers
        state.river_sides = sides
        state.river_corners = dict(corner_tags)
        _write_hex_catchments(hexes, land, sides, drainage, drain_of, max_acc, self.config)

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
    ) -> list[list[HexCoord]]:
        """Trace each headwater downstream to ocean/border.

        Headwaters are derived directly from river_set and flow_dir (not from hex tags,
        since _tag_hexes runs after this method). If flow_dir stalls before reaching ocean
        (flat-area artefact), extend the path via elevation-guided search toward the nearest
        outlet; fallback hexes are added to river_set and flow_dir is updated to keep
        all downstream data consistent.

        Paths are built as full source-to-sea traces; split into source-to-confluence
        segments by the caller after all drainage rivers are also available.
        """
        rivers: list[list[HexCoord]] = []
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
                rivers.append(path)

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
        rain: dict[HexCoord, float] | None = None,
    ) -> tuple[list[list[HexCoord]], dict[HexCoord, HexCoord | None]]:
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
        rain = rain or {}
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

        new_rivers: list[list[HexCoord]] = []
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
                hexes[c].terrain_class = TerrainClass.INLAND_WATER
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
                    hexes[c].terrain_class = TerrainClass.OPEN_WATER
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
            ) + sum(rain.get(c, 1.0) for c in component)
            # `basin_inflow` counts hexes of the map's mean rainfall, so evaporation has
            # to be quoted in the same unit: what a year's warmth lifts off open water,
            # over what a year's rain delivers to one hex.  It is the potential rather
            # than an actual, there being no shortage of water to evaporate off a lake —
            # and it comes from the same term the runoff model takes off the top of the
            # rain, so the basins and the rivers cannot disagree about the climate.
            per_hex = self.config.pet_mm() / max(self.config.mean_precip_mm, 1e-9)
            evaporation = self.config.endorheic_evaporation_scale * per_hex * len(component)
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
            # The spillway is the outlet itself, so it carries the discharge too.  Only
            # the hexes *after* it were being given the flow, which left the one hex where
            # the river leaves the lake reading a single hex of rain — drawn hairline-thin
            # at exactly the point a reader looks to see whether the lake drains.
            acc[spillway] = running_acc
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
                        # The basin brings more water than the channel it joins already
                        # carries.  Flow must not decrease downstream, and there are two
                        # ways to hold that: clamp the basin down to the channel, or raise
                        # the channel to take what arrives.  It used to clamp — which
                        # silently poured the basin's throughput away at the junction and
                        # left the lake draining through a channel the size of whatever it
                        # happened to meet.  A tributary joining a smaller stream does not
                        # shrink to fit it; the stream below the junction grows.  So raise
                        # it, and let the tail walk below carry that downstream.
                        acc[coord] = running_acc
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
            if len(path) > 1:
                new_rivers.append(path)

        return new_rivers, outlet_of


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


def _write_hex_catchments(hexes, land, sides, drainage, drain_of, max_acc, config):
    """Record on each land hex the catchment beside it, for the viewer and for erosion.

    A hex's catchment is what drains to its own lowest corner, or, on a river bank, the
    largest river it touches.  Its `river_flow` is that catchment as a share of the
    largest on the map, written on both banks of every river — or on every draining land
    hex with `river_flow_continuous`, which is a diagnostic for the viewer.  The rivers
    themselves are `river_sides`; nothing about a river is a hex tag.
    """
    for hx in hexes.values():
        hx.river_flow = 0.0
        hx.catchment_km2 = 0.0

    touching: dict[HexCoord, float] = {}
    for side, rs in sides.items():
        for h in side_hexes(side):
            touching[h] = max(touching.get(h, 0.0), rs.catchment_km2)

    for coord in land:
        own = drainage.acc.get(drain_of.get(coord), 0.0) if coord in drain_of else 0.0
        hexes[coord].catchment_km2 = max(own, touching.get(coord, 0.0))
        if config.river_flow_continuous or coord in touching:
            hexes[coord].river_flow = hexes[coord].catchment_km2 / max_acc
