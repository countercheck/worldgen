import heapq
from collections import deque

import numpy as np
from scipy.ndimage import gaussian_filter

from ..core.hex_grid import distance as hex_distance
from ..core.hex_grid import neighbors as hex_neighbors
from ..core.pipeline import GeneratorStage
from ..core.world_state import WorldState

try:
    import numba as _numba

    _jit = _numba.njit
except ImportError:  # numba optional — fall back to pure Python
    _numba = None  # type: ignore[assignment]

    def _jit(fn):  # type: ignore[misc]
        return fn


_MAX_STEPS = 64
_EVAPORATION = 0.99


@_jit
def _deposit_delta(
    arr: np.ndarray,
    ci: int,
    cj: int,
    sediment: float,
    w: int,
    h: int,
    sea_level: float,
    min_load: float,
) -> None:
    """Spread a droplet's load as a fan from where its channel meets the sea.

    A river drops most of its load right at the mouth — the plume loses competence
    within a few kilometres — so a delta is a small steep fan, not an even blanket along
    the shore.  Emptying the whole load into the single hex of entry instead built
    isolated spikes, and since droplets cross the waterline wherever they happen to
    reach it, those spikes smeared along the entire coastline: only a third of the
    infilled sea hexes were within three hexes of a river mouth and a fifth were more
    than twenty away.  Fanning each load out with a sharp radial falloff lets the many
    droplets funnelled down one channel superpose into a delta at its mouth, while a
    lone droplet arriving off a hillside leaves almost nothing.

    Nothing is lifted above the waterline: a delta progrades to sea level and then
    builds seaward, it does not pile into hills.  That is also what stops sediment from
    sealing the map edge back up into dry land far from any river.
    """
    if sediment < min_load:
        # Too little to build anything.  A droplet that trickled off a nearby hillside
        # reaches the sea carrying almost nothing, and that sediment is carried away
        # along the shore rather than settling where it entered.  Letting every such
        # arrival deposit is what smeared the shelf: with one droplet per cell of map,
        # the whole coastline silts up evenly and no delta stands out anywhere.  Only a
        # load that came down a channel is enough to build.
        return

    for radius in range(3):
        if radius == 0:
            weight = 0.6
        elif radius == 1:
            weight = 0.3
        else:
            weight = 0.1

        count = 0
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                if di > -radius and di < radius and dj > -radius and dj < radius:
                    continue  # interior of the box: belongs to a smaller ring
                i = ci + di
                j = cj + dj
                if 0 <= i < w and 0 <= j < h and arr[i, j] < sea_level:
                    count += 1
        if count == 0:
            continue

        share = sediment * weight / count
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                if di > -radius and di < radius and dj > -radius and dj < radius:
                    continue
                i = ci + di
                j = cj + dj
                if 0 <= i < w and 0 <= j < h and arr[i, j] < sea_level:
                    raised = arr[i, j] + share
                    arr[i, j] = raised if raised < sea_level else sea_level


@_jit
def _drop_particle(
    arr: np.ndarray,
    channel_affinity: np.ndarray,
    px: float,
    py: float,
    w: int,
    h: int,
    sea_level: float,
    inertia: float,
    capacity: float,
    deposition: float,
    erosion_rate: float,
    affinity_gain: float,
    delta_min_load: float,
) -> None:
    dir_x, dir_y = 0.0, 0.0
    speed = 1.0
    water = 1.0
    sediment = 0.0

    for _ in range(_MAX_STEPS):
        ci, cj = int(px), int(py)

        if ci < 0 or ci >= w or cj < 0 or cj >= h:
            break
        if arr[ci, cj] < sea_level:
            _deposit_delta(arr, ci, cj, sediment, w, h, sea_level, delta_min_load)
            break

        # Gradient from 4 neighbors (clamp at edges)
        left = arr[max(ci - 1, 0), cj]
        right = arr[min(ci + 1, w - 1), cj]
        up = arr[ci, max(cj - 1, 0)]
        down = arr[ci, min(cj + 1, h - 1)]
        gx = (right - left) * 0.5
        gy = (down - up) * 0.5

        dir_x = inertia * dir_x - (1.0 - inertia) * gx
        dir_y = inertia * dir_y - (1.0 - inertia) * gy

        length = (dir_x**2 + dir_y**2) ** 0.5
        if length < 1e-8:
            break
        dir_x /= length
        dir_y /= length

        new_px = px + dir_x
        new_py = py + dir_y
        ni, nj = int(new_px), int(new_py)

        if ni < 0 or ni >= w or nj < 0 or nj >= h:
            break

        dh = arr[ni, nj] - arr[ci, cj]
        cap = max(-dh, 0.01) * speed * water * capacity

        if sediment > cap:
            deposit = deposition * (sediment - cap)
            arr[ci, cj] += deposit
            sediment -= deposit
        else:
            erode = min(erosion_rate * (cap - sediment), abs(dh) if dh < 0 else 0.0)
            arr[ci, cj] -= erode
            sediment += erode
            if erode > 0.0:
                channel_affinity[ci, cj] += affinity_gain

        speed = max(speed + dh, 0.01)
        water *= _EVAPORATION
        px, py = new_px, new_py


def _neighbour_table(state: WorldState, w: int, h: int) -> list[list[tuple[int, int]]]:
    """Each grid cell's in-bounds hex neighbours, as flat-indexed (col, row) pairs.

    Built once and shared by the sink fill and the accumulation across every carve pass.
    Those walk every cell six ways several times over, and resolving the neighbourhood
    through `coord_at`, `neighbors` and `grid_index` each time made the widening cost more
    than the droplet simulation it follows — about four times the test suite's runtime.
    The mapping is the same on every pass, so it is worth computing once.
    """
    table: list[list[tuple[int, int]]] = []
    for i in range(w):
        for j in range(h):
            cell = []
            for n in hex_neighbors(state.coord_at(i, j)):
                if n not in state.hexes:
                    continue
                ni, nj = state.grid_index(n)
                if 0 <= ni < w and 0 <= nj < h:
                    cell.append((ni, nj))
            table.append(cell)
    return table


def _fill_sinks(
    arr: np.ndarray, sea_level: float, neighbours: list[list[tuple[int, int]]]
) -> np.ndarray:
    """A copy of *arr* with its depressions filled to their spill level (Barnes et al).

    Accumulation needs this and the droplets guarantee it will be needed: they leave the
    surface pitted, and without filling, drainage dies at the first depression it meets.
    That is not a small error — it is the difference between a map of short disconnected
    segments and one with trunk rivers, and it was most of why the channels being widened
    coincided with only a third of the rivers hydrology later found.  Hydrology fills
    sinks for exactly this reason before it routes anything; measuring the same quantity
    means filling them here too.

    Returns a copy: the fill decides where water *would* run, and is no reason to raise
    the actual ground.
    """
    filled = arr.copy()
    w, h = arr.shape
    visited = np.zeros((w, h), dtype=bool)
    heap: list[tuple[float, int, int]] = []

    # Seed from the sea and the map edge, the two places water leaves by.
    for i in range(w):
        for j in range(h):
            if arr[i, j] < sea_level or i in (0, w - 1) or j in (0, h - 1):
                heapq.heappush(heap, (float(filled[i, j]), i, j))
                visited[i, j] = True

    while heap:
        elev, i, j = heapq.heappop(heap)
        for ni, nj in neighbours[i * h + j]:
            if visited[ni, nj]:
                continue
            visited[ni, nj] = True
            filled[ni, nj] = max(filled[ni, nj], elev)
            heapq.heappush(heap, (float(filled[ni, nj]), ni, nj))
    return filled


def _grid_flow_accumulation(
    arr: np.ndarray,
    sea_level: float,
    neighbours: list[list[tuple[int, int]]],
    inflow: dict[tuple[int, int], float] | None = None,
) -> np.ndarray:
    """How much land drains through each cell, by steepest descent over the hex grid.

    Valleys have to be widened where the rivers will be, and the droplet affinity does not
    answer that — it records where droplets wandered while the terrain was still being
    cut, and only about one cell in five of the finished river network sits on it.
    Hydrology chooses its rivers by flow accumulation, so this measures the same thing.

    Cells are handled high to low, each passing its total to its lowest neighbour, so one
    sort does the work of a traversal per cell.  Routing runs over the sink-filled
    surface, not the raw one, so water crosses a depression rather than disappearing into
    it — without that the network is a scatter of short segments and no trunk river ever
    forms.
    """
    w, h = arr.shape
    land = arr >= sea_level
    acc = np.where(land, 1.0, 0.0)
    # A river entering from off the map brings a catchment this map never had.  Seeding it
    # here is what makes the valley match the water: widening scales its reach by
    # discharge, so without this an imported trunk is measured as the trickle its first
    # few on-map hexes would raise, and gets a trickle's valley.
    for cell, volume in (inflow or {}).items():
        if land[cell]:
            acc[cell] = max(acc[cell], volume)

    # Route over the filled surface, so water crosses a depression instead of vanishing
    # into it — the same thing hydrology does, and the reason the two agree on where the
    # rivers are.
    routing = _fill_sinks(arr, sea_level, neighbours)
    order = [(int(i), int(j)) for i, j in np.argwhere(land)]
    order.sort(key=lambda c: -routing[c])

    for i, j in order:
        lowest = None
        lowest_elev = routing[i, j]
        for ni, nj in neighbours[i * h + j]:
            if not land[ni, nj]:
                continue
            if routing[ni, nj] < lowest_elev:
                lowest_elev = routing[ni, nj]
                lowest = (ni, nj)
        if lowest is not None:
            acc[lowest] += acc[i, j]
    return acc


def _inflow_mouths(
    arr: np.ndarray,
    sea_level: float,
    state: WorldState,
    neighbours: list[list[tuple[int, int]]],
    edges: tuple,
    count: int,
    separation: int,
) -> list[tuple[int, int]]:
    """Border cells where a river from beyond the map would enter, best first.

    The criterion is hydrology's: land on a chosen edge whose ground falls away inland.
    Erosion cannot ask which hexes hydrology will pick — that runs three stages later and
    needs a priority-flood this stage has no reason to do — but it does not have to.
    Carving these deepens the very descent hydrology ranks on, so the mouths chosen here
    are the ones it goes on to choose, and the two agree by construction.

    Ranked by drop and thinned by `separation`, so two inlets do not land in one valley.
    """
    if count <= 0:
        return []
    w, h = arr.shape
    wanted = set(edges)
    candidates: list[tuple[float, tuple[int, int]]] = []

    for i in range(w):
        for j in range(h):
            on_edge = set()
            if i == 0:
                on_edge.add("west")
            if i == w - 1:
                on_edge.add("east")
            if j == 0:
                on_edge.add("north")
            if j == h - 1:
                on_edge.add("south")
            if not (on_edge & wanted) or arr[i, j] < sea_level:
                continue
            best = 0.0
            for ni, nj in neighbours[i * h + j]:
                if arr[ni, nj] < sea_level:
                    continue
                if ni in (0, w - 1) or nj in (0, h - 1):
                    continue  # still on the border: along the edge, not into the map
                best = max(best, arr[i, j] - arr[ni, nj])
            if best > 0.0:
                candidates.append((best, (i, j)))

    candidates.sort(key=lambda c: (-c[0], c[1]))
    chosen: list[tuple[int, int]] = []
    for _drop, cell in candidates:
        if len(chosen) >= count:
            break
        coord = state.coord_at(*cell)
        if all(hex_distance(coord, state.coord_at(*c)) >= separation for c in chosen):
            chosen.append(cell)
    return chosen


def _widen_valleys(
    arr: np.ndarray,
    discharge: np.ndarray,
    sea_level: float,
    width_max: float,
    width_exponent: float,
    floor_slope: float,
    max_relief: float,
    channel_fraction: float,
) -> None:
    """Plane a flat floor outward from each channel, in place.

    Droplet erosion only ever incises: a droplet cuts along the line it travels, so the
    field ends up carved into V-notches one cell wide however long it runs.  What widens a
    real valley is the channel wandering sideways across it for a very long time, planing
    everything down to about its own level — lateral planation, which a point process has
    no way to express.  This is that missing term.

    The result is deliberately a *flat floor between bluffs* rather than a broadened V.
    Lateral planation cuts to grade and stops at what it cannot shift, so the floor is
    level and the step up to the valley wall is abrupt — the Mississippi's bluff line, the
    escarpment walling the Nile.  Lowering cells to the channel out to a width and leaving
    everything past it alone gives that step for free; a smooth falloff would give a soft
    bowl, which reads as a dry basin rather than a valley.

    Reach scales with *discharge*, since floodplain width goes roughly with the square
    root of drainage area — hence the exponent.  Ground standing more than `max_relief`
    above the floor is valley wall: it is left alone, and the fill stops rather than
    stepping over it, which is what keeps a valley to its valley instead of planing the
    countryside, and makes `width_max` a cap rather than the usual outcome.

    Cells are never raised, only cut, and never below the channel doing the cutting — so
    this cannot invent a sink, drown land, or undo the droplets' work.

    Neighbours here are the four of the array rather than the six of the hex grid: this
    fills an area rather than tracing a line, so the shape of the neighbourhood washes out,
    and it is the same square-lattice approximation `_drop_particle` already makes.
    """
    if width_max <= 0.0:
        return
    land = arr >= sea_level
    if not land.any():
        return

    flow = np.where(land, discharge, 0.0)
    max_flow = float(flow.max())
    if max_flow <= 0.0:
        return

    threshold = float(np.quantile(flow[land], 1.0 - channel_fraction))
    channels = np.argwhere(land & (flow >= threshold))
    if len(channels) == 0:
        return

    w, h = arr.shape
    target = np.full((w, h), np.inf)
    budget = np.zeros((w, h))
    queue: deque = deque()

    for i, j in channels:
        i, j = int(i), int(j)
        reach = width_max * (flow[i, j] / max_flow) ** width_exponent
        if reach < 1.0:
            continue
        target[i, j] = arr[i, j]
        budget[i, j] = reach
        queue.append((i, j))

    while queue:
        i, j = queue.popleft()
        remaining = budget[i, j] - 1.0
        if remaining < 0.0:
            continue
        # The floor rises a little away from the channel, so a floodplain drains toward
        # its river instead of ponding.
        floor = target[i, j] + floor_slope
        for ni, nj in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if not (0 <= ni < w and 0 <= nj < h) or not land[ni, nj]:
                continue
            if arr[ni, nj] - floor > max_relief:
                continue  # valley wall: too high to plane, and the fill stops here
            # A cell reached by two valleys takes the lower floor: at a confluence the
            # larger river governs, which is what it does on the ground.
            if floor < target[ni, nj] - 1e-12:
                target[ni, nj] = floor
                budget[ni, nj] = remaining
                queue.append((ni, nj))

    cut = np.isfinite(target)
    arr[cut] = np.minimum(arr[cut], target[cut])


class ErosionStage(GeneratorStage):
    def run(self, state: WorldState) -> WorldState:
        cfg = self.config
        w, h = state.width, state.height

        # Erosion works on a normalised copy, and puts the result back in metres.
        #
        # Its constants are fractions of the map's relief rather than physical
        # quantities: erosion_capacity multiplies a height difference, the capacity floor
        # and erosion_delta_min_load are absolute heights, and all of them were tuned
        # against a 0-1 range. Fed metres directly they become centimetres — a droplet's
        # capacity collapses to nothing, so every one of them deposits, and the whole map
        # planes off to sea level within a couple of passes.
        #
        # Converted against the *known* span rather than the map's own minimum and
        # maximum, so this is a fixed change of units and not another per-map stretch of
        # the kind the rest of this work has been removing. Erosion is a shaping heuristic,
        # and its knobs being shares of the relief is the honest description of them.
        span = cfg.max_elevation_m + cfg.seabed_depth_m
        sea_shaped = cfg.seabed_depth_m / span

        # Indexed by grid column/row throughout; `state.coord_at` turns an index back
        # into a hex on the way in and out, so droplets run over the same rectangular
        # field whichever layout the grid uses.
        arr = np.zeros((w, h))
        for col in range(w):
            for row in range(h):
                elevation = state.hexes[state.coord_at(col, row)].elevation
                arr[col, row] = (elevation + cfg.seabed_depth_m) / span

        land_coords = [
            (col, row) for col in range(w) for row in range(h) if arr[col, row] >= sea_shaped
        ]

        if land_coords:
            land_arr = np.array(land_coords)
            n_land = len(land_coords)
            # Dosed per land hex rather than as a flat count, because a flat count is a
            # different amount of weather depending on how big the map is. At the old
            # default of 15000 droplets a 32x32 map got 14.6 per hex and a 128x128 got
            # 0.9 — a sixteenfold spread, and most of why small maps came out as Alpine
            # massifs while the default map stayed a barely-touched noise field.
            #
            # Per *land* hex, not per map hex: droplets are seeded on land, so a map that
            # is mostly ocean should not have its weather spread thinner over what land it
            # has.
            n_iter = max(1, int(round(cfg.erosion_droplets_per_hex * n_land)))
            affinity_interval = cfg.erosion_affinity_update_interval

            # Channel affinity: starts uniform, biases later particles toward established channels
            channel_affinity = np.ones((w, h))

            # Initial sample indices (uniform random)
            indices = self.rng.integers(0, n_land, size=n_iter)

            for step in range(n_iter):
                sq, sr = int(land_arr[indices[step], 0]), int(land_arr[indices[step], 1])
                _drop_particle(
                    arr,
                    channel_affinity,
                    float(sq),
                    float(sr),
                    w,
                    h,
                    sea_shaped,
                    cfg.erosion_inertia,
                    cfg.erosion_capacity,
                    cfg.erosion_deposition,
                    cfg.erosion_erosion_rate,
                    cfg.erosion_channel_affinity_gain,
                    cfg.erosion_delta_min_load,
                )

                # Periodically re-weight remaining indices toward established channels
                if affinity_interval > 0 and step > 0 and step % affinity_interval == 0:
                    remaining = n_iter - step - 1
                    if remaining > 0:
                        land_weights = channel_affinity[land_arr[:, 0], land_arr[:, 1]]
                        land_weights = land_weights / land_weights.sum()
                        indices[step + 1 :] = self.rng.choice(
                            n_land, size=remaining, p=land_weights
                        )

        arr = gaussian_filter(arr, sigma=0.5)

        # Back to metres below. There is deliberately no re-stretch to [0, 1] first: it
        # would undo the datum, putting the lowest point of the eroded map at the seabed
        # and the highest at the peak whatever erosion had actually done to either. Sea
        # level has to stay where it is for the word to mean anything, and a landscape
        # that has been worn down should read as worn down rather than being scaled back
        # up to fill the range it started with.
        #
        # Carve the valleys, then look again at where the water runs, and carve again.
        # One pass does not do it: widening a valley moves the drainage into it, so the
        # network measured before the first cut is not the network that exists after —
        # against hydrology's rivers a one-shot carve landed on about a quarter of them.
        # Letting terrain and drainage settle against each other is what a landscape
        # evolution model does, and it is the only way the two agree by the time anything
        # downstream reads either.
        if land_coords and cfg.valley_width_max > 0.0:
            # Rivers that enter from off the map bring a catchment this map never had, and
            # nothing here knew about it: accumulation starts every cell at one hex of
            # rain, so an imported trunk was measured as the trickle its first few on-map
            # hexes raise, and widening gave it a trickle's valley.  Seed the mouths with
            # what they actually carry — the same quantity hydrology seeds them with — and
            # the valley follows the water without any special case for it.
            neighbours = _neighbour_table(state, w, h)
            inflow_volume = max(1.0, cfg.river_inflow_volume * len(land_coords))
            inflow = {
                mouth: inflow_volume
                for mouth in _inflow_mouths(
                    arr,
                    sea_shaped,
                    state,
                    neighbours,
                    tuple(cfg.river_inflow_edges),
                    cfg.river_inflow_count,
                    cfg.river_inflow_min_separation,
                )
            }

            # The two height knobs are quoted in metres and the field is normalised, so
            # they are divided by the same span the array was built with.
            for _ in range(cfg.valley_carve_passes):
                _widen_valleys(
                    arr,
                    _grid_flow_accumulation(arr, sea_shaped, neighbours, inflow),
                    sea_shaped,
                    cfg.valley_width_max,
                    cfg.valley_width_exponent,
                    cfg.valley_floor_slope_m / span,
                    cfg.valley_max_relief_m / span,
                    cfg.valley_channel_fraction,
                )

        for col in range(w):
            for row in range(h):
                metres = float(arr[col, row]) * span - cfg.seabed_depth_m
                state.hexes[state.coord_at(col, row)].elevation = metres

        return state
