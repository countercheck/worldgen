"""Hydrology, part one: settle the surface water runs on, and which way it runs.

Priority-Flood sink filling, the epsilon tilt that gives a filled flat a gradient, flow
direction, the off-map inflow inlets, and flow accumulation.  `HydrologyStage` (in
`hydrology.py`) mixes this in; the river and lake work that reads these fields lives in
`hydrology_rivers.py`, which may import from here but not the other way round.
"""

import heapq
from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from ..core.hex import HexCoord
from ..core.hex_grid import distance, neighbors
from ..core.world_state import WorldState

if TYPE_CHECKING:
    from ..core.config import WorldConfig

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


class HydrologyFill:
    """Sink filling, flow direction and flow accumulation for `HydrologyStage`.

    A mixin: `config` and `rng` are the stage's own, set by `GeneratorStage.__init__`.
    """

    config: "WorldConfig"
    rng: np.random.Generator

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
        """For each land hex, the neighbor it drains to: a lower one, drawn at random with
        weight (drop / largest drop) ** `river_wander_exponent`.

        The caller adds an epsilon tilt before calling, so all filled elevations are
        unique and every land hex off the border has somewhere lower to go; any strictly
        lower choice keeps the result cycle-free.

        Priority-Flood does not seed lake hexes, so their filled elevation may be
        raised by the algorithm.  To guarantee that land hexes adjacent to lakes
        still drain into them, we use the raw (pre-flood) elevation for ocean and
        lake neighbors when computing steepest descent.
        """
        water = ocean | lakes
        power = self.config.river_wander_exponent
        flow_dir: dict[HexCoord, HexCoord | None] = {}
        # Sorted, so the draws land on the same hexes in the same order every run.
        for coord in sorted(land):
            here = filled[coord]
            options: list[HexCoord] = []
            drops: list[float] = []
            for nbr in neighbors(coord):
                if nbr not in filled:
                    continue
                # A border land hex that drains along the border would produce rivers
                # that creep along the map edge; it drains off the map instead.
                if on_border(coord) and nbr not in water and on_border(nbr):
                    continue
                # Use raw elevation for water hexes so PF-raised lake/ocean values
                # never appear higher than the actual landscape.
                nbr_e = elev[nbr] if nbr in water else filled[nbr]
                if nbr_e < here:
                    options.append(nbr)
                    drops.append(here - nbr_e)
            best_coord: HexCoord | None = None
            # Beside the sea or a lake, water goes into it: a coastal hex that wandered
            # along the shore instead would draw a river running parallel to the water.
            wet = [(d, n) for d, n in zip(drops, options, strict=True) if n in water]
            if wet:
                best_coord = max(wet)[1]
            elif len(options) == 1:
                best_coord = options[0]
            elif options:
                # Strictly downhill whichever is drawn, so the network stays free of
                # cycles; the weighting keeps a steep valley on its floor and lets a flat
                # one wander. Scaled by the largest drop first: on a filled flat the drops
                # are the epsilon tilt's millionths, and their powers would underflow.
                rel = np.array(drops) / max(drops)
                weights = rel**power
                pick = self.rng.random() * weights.sum()
                best_coord = options[
                    min(int(np.searchsorted(np.cumsum(weights), pick)), len(options) - 1)
                ]

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
        rain: dict[HexCoord, float] | None = None,
    ) -> dict[HexCoord, float]:
        """Topological sort (Kahn's) then accumulate upstream counts.

        Each land hex starts with the rain that fell on it — one unit on average, more on
        a windward slope and less in a shadow, or a flat unit each where *rain* is not
        given.  A hex in *inflow* starts with the off-map catchment it drains instead,
        which is what carries a river in over the border already large.
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
        rain = rain or {}
        acc: dict[HexCoord, float] = {c: inflow.get(c, rain.get(c, 1.0)) for c in land}

        while queue:
            coord = queue.popleft()
            ds = downstream[coord]
            if ds is not None and ds in land:
                acc[ds] += acc[coord]
                in_degree[ds] -= 1
                if in_degree[ds] == 0:
                    queue.append(ds)

        return acc
