import math
from collections import defaultdict, deque

import numpy as np
from scipy.ndimage import gaussian_filter

from ..core.config import WorldConfig
from ..core.hex import HexCoord
from ..core.hex_grid import Corner, corner_hexes, hex_corner_keys, side_hexes, side_joining
from ..core.hex_grid import distance as hex_distance
from ..core.pipeline import GeneratorStage
from ..core.world_state import WorldState
from .corner_drainage import (
    CornerNetwork,
    Drainage,
    accumulate,
    build_network,
    drain_corner,
    flow_direction,
    inlet_corner,
    is_lake_node,
)
from .elevation import apply_profile

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
    alluvium: np.ndarray,
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
                    # The silt arrives whether or not the ground can rise to show it: a
                    # delta that has already prograded to sea level goes on receiving
                    # sediment and building seaward.  Crediting the clamped elevation gain
                    # instead would score the richest ground on the map — the front of an
                    # established delta — as the barest.
                    alluvium[i, j] += share


@_jit
def _drop_particle(
    arr: np.ndarray,
    alluvium: np.ndarray,
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
    overcut: float,
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
            _deposit_delta(arr, alluvium, ci, cj, sediment, w, h, sea_level, delta_min_load)
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
            # Where the water slowed and dropped its load.  This number was already being
            # computed and spent on the elevation alone, which threw away the more useful
            # half of it: how high the ground ended up is a poor proxy for what it is made
            # of.  A hillside cut down to a gentle grade and a valley floor built up to the
            # same height are the same elevation and nothing alike to plough.
            alluvium[ci, cj] += deposit
        else:
            # A droplet may not cut a cell below its own downstream neighbour (plus
            # `overcut`, normally zero).  That clamp is why droplets alone never deepen a
            # valley — which is deliberate: deepening is `_incise_channels`'s job, and a
            # droplet let loose here punches pits the sink fill then has to span.
            limit = (abs(dh) + overcut) if dh < 0 else 0.0
            erode = min(erosion_rate * (cap - sediment), limit)
            arr[ci, cj] -= erode
            sediment += erode
            # Netted, not accumulated: sediment picked back up has left.  A channel that
            # deposits on one droplet's pass and scours on the next is holding nothing, and
            # summing only the deposits would call every busy channel deep soil.
            alluvium[ci, cj] -= erode
            if erode > 0.0:
                channel_affinity[ci, cj] += affinity_gain

        speed = max(speed + dh, 0.01)
        water *= _EVAPORATION
        px, py = new_px, new_py


def _corner_draws(state: WorldState, rng: np.random.Generator) -> dict[Corner, float]:
    """One fixed uniform draw per corner of the map, for the wander rule.

    Fixed across carve passes so a corner keeps choosing the same way while its valley
    deepens, rather than scattering cuts between passes.
    """
    corners = sorted({c for coord in state.hexes for c in hex_corner_keys(coord)})
    return dict(zip(corners, rng.random(len(corners)).tolist(), strict=True))


def _drain(
    arr: np.ndarray,
    sea_level: float,
    state: WorldState,
    draws: dict[Corner, float],
    wander_exponent: float,
    rng: np.random.Generator,
    floor_blend: float = 0.0,
) -> tuple[CornerNetwork, Drainage]:
    """Drain the working surface along hexsides, as hydrology will drain the finished one.

    Valleys have to be cut where the rivers will be, and hydrology routes its rivers on
    the corner graph (`corner_drainage`), so this drains the same graph over the field
    being carved.  The fill inside `flow_direction` lets water cross a depression rather
    than vanish into it, without raising the ground itself.  How much water then passes
    each corner is `_accumulate`'s business.
    """
    w, h = arr.shape
    elevation = {state.coord_at(i, j): float(arr[i, j]) for i in range(w) for j in range(h)}
    land = {c for c, z in elevation.items() if z >= sea_level}
    net = build_network(elevation, land, set(elevation) - land, set(), [], floor_blend)
    return net, flow_direction(net, rng, wander_exponent, draws=draws)


def _accumulate(
    arr: np.ndarray,
    sea_level: float,
    state: WorldState,
    net: CornerNetwork,
    drainage: Drainage,
    inflow: dict[tuple[int, int], float] | None = None,
) -> None:
    """Fill in `drainage.acc`: every land cell's rain, and what the inlets bring.

    Every land cell's rain runs off to its lowest corner and down from there, and a river
    entering from off the map brings the catchment it gathered beyond it.
    """
    w, h = arr.shape
    land = {state.coord_at(i, j) for i in range(w) for j in range(h) if arr[i, j] >= sea_level}
    sources: dict[Corner, float] = defaultdict(float)
    for coord in sorted(land):
        corner = drain_corner(coord, net, drainage)
        if corner is not None:
            sources[corner] += 1.0
    for cell, volume in (inflow or {}).items():
        coord = state.coord_at(*cell)
        corner = inlet_corner(coord, net, drainage) if coord in land else None
        if corner is not None:
            sources[corner] += volume
    drainage.acc = accumulate(drainage.flow, sources)


def _corner_routing(
    arr: np.ndarray,
    sea_level: float,
    state: WorldState,
    draws: dict[Corner, float],
    wander_exponent: float,
    rng: np.random.Generator,
    inflow: dict[tuple[int, int], float] | None = None,
    floor_blend: float = 0.0,
) -> tuple[CornerNetwork, Drainage]:
    """`_drain` then `_accumulate`: the routing, and the water on it."""
    net, drainage = _drain(arr, sea_level, state, draws, wander_exponent, rng, floor_blend)
    _accumulate(arr, sea_level, state, net, drainage, inflow)
    return net, drainage


def _hex_discharge(state: WorldState, drainage: Drainage, w: int, h: int) -> np.ndarray:
    """Per cell, the largest discharge passing any of its corners.

    Widening fills an area outward from the channel cells, and a channel along a hexside
    runs between two of them, so both banks of a river count as the channel here.
    """
    out = np.zeros((w, h))
    for i in range(w):
        for j in range(h):
            out[i, j] = max(
                (
                    drainage.acc.get(c, 0.0)
                    for c in hex_corner_keys(state.coord_at(i, j))
                    if drainage.flow.get(c) is not None
                ),
                default=0.0,
            )
    return out


# A hexside is a kilometre-wide hex's side: 1/sqrt(3) km long.
_SIDE_KM = 1.0 / math.sqrt(3.0)


def _incise_channels(
    arr: np.ndarray,
    state: WorldState,
    drainage: Drainage,
    sea_level: float,
    span: float,
    *,
    m_per_pass: float,
    area_exponent: float,
    slope_exponent: float,
    reference_km2: float,
    reference_slope: float,
    min_gradient_m: float,
    max_cut_m: float,
) -> None:
    """Lower each channel by K * A^m * S^n, in place.

    The term the droplet model has no way to express.  A droplet carries one unit of water
    however much country it drains, so it cuts a trunk and a hillslope at the same rate and
    no valley ever gets deeper than its surroundings; with an area exponent of 0.5 a
    500 km2 channel cuts about 22 times as fast as the 1 km2 ground beside it.  That
    contrast is what makes stream capture happen: ground beside a valley that has been cut
    finds, when the next pass drains the surface again, that the valley is now the way
    down, and its own catchment jumps.  Run over a few passes, neighbouring channels stop
    running side by side and start joining.

    Water runs along hexsides, so what is cut is a side: each corner's water leaves along
    the side to its receiver, and the two hexes either side of it — both banks — are
    lowered toward the corner's new height, together with the corner's own lowest hex,
    whose height a corner's is.

    Corners are taken **outlets first**, the fill order, so a corner's receiver has already
    been lowered by the time the corner is reached.  That is what lets the floor at
    `receiver + min_gradient` permit deepening while still forbidding inversion: the whole
    trunk migrates downward together rather than being pinned to ground that has not moved.
    With `slope_exponent` of 1 the step is linear in elevation, so that floor is an exact
    stability guard and no timestep is needed.  `max_cut_m` only catches an inherited
    cliff.

    Arithmetic is in metres — `arr` is the normalised field and `span` converts — because
    a stream power law with a physical exponent means nothing in units of relief fraction.
    """
    if m_per_pass <= 0.0:
        return
    # K is derived rather than configured, so the dial stays in metres whatever the
    # exponents are: at the reference area and slope, a channel lowers by exactly
    # m_per_pass.
    k = m_per_pass / (reference_km2**area_exponent * reference_slope**slope_exponent)
    min_gap = min_gradient_m / span
    # A cell can be a bank of more than one corner, so the cap is on what it loses over the
    # whole pass, not per cut.
    deepest = arr - max_cut_m / span

    index = {coord: state.grid_index(coord) for coord in state.hexes}
    around = {
        corner: [index[h] for h in corner_hexes(corner) if h in index] for corner in drainage.order
    }

    def cell(coord):
        return index[coord]

    def land_cells(corner) -> list[tuple[int, int]]:
        return [c for c in around.get(corner, ()) if arr[c] >= sea_level]

    def height(corner) -> float:
        cells = land_cells(corner)
        return float(min(arr[c] for c in cells)) if cells else sea_level

    for corner in sorted(drainage.order, key=drainage.order.__getitem__):
        receiver = drainage.flow.get(corner)
        if receiver is None or is_lake_node(receiver):
            continue  # an outlet: base level, and nothing below it to cut toward
        here, there = height(corner), height(receiver)
        drop = here - there
        if drop <= 0.0:
            continue  # inside a filled depression; there is no gradient to cut with
        slope = drop * span / (1000.0 * _SIDE_KM)  # metres of fall per metre along the side
        acc = drainage.acc.get(corner, 0.0)
        cut_m = min(k * acc**area_exponent * slope**slope_exponent, max_cut_m)
        floor = max(here - cut_m / span, there + min_gap, sea_level)
        # The corner's own floor and both banks of the side the water leaves by.  None
        # loses more than the cap, so a bank standing on a cliff above the corner is cut
        # into rather than felled to the river in one pass.
        cells = {cell(h) for h in side_hexes(side_joining(corner, receiver)) if h in state.hexes}
        lowest = land_cells(corner)
        if lowest:
            cells.add(min(lowest, key=lambda c: (arr[c], c)))
        for c in cells:
            arr[c] = max(min(arr[c], floor), deepest[c])


def _course(drainage: Drainage, corner: Corner) -> list[Corner]:
    """The corners water starting at *corner* runs through, to where it leaves the network."""
    path: list[Corner] = []
    seen: set[Corner] = set()
    node: Corner | None = corner
    while node is not None and node not in seen:
        seen.add(node)
        path.append(node)
        node = drainage.flow.get(node)
    return path


def _choose_inlets(
    arr: np.ndarray,
    sea_level: float,
    state: WorldState,
    net: CornerNetwork,
    drainage: Drainage,
    rng: np.random.Generator,
    *,
    edges: tuple[str, ...],
    count: int,
    separation: int,
    min_length_km: float,
    length_bias: float,
) -> list[tuple[int, int]]:
    """Border cells where a river from beyond the map enters, chosen once for the world.

    This is the one place inlets are chosen.  Hydrology adopts them, with the course each
    took here (tech-debt #153).  They used to be chosen twice, here by the steepest drop
    inland and in hydrology by a weighted draw of its own, and the two agreed on none of
    ten: the imported catchment widened and cut valleys hydrology never sent the imported
    river down, while its river ran through ground carved for a trickle.

    The rule is the one hydrology used, read off the corner routing this stage carves by.
    A land cell on a chosen edge is a candidate if the water joining the network at its
    inlet corner (`inlet_corner`) then runs at least *min_length_km*.  Candidates are
    drawn at random, weighted by that length raised to *length_bias*, and thinned by
    *separation* so two inlets do not land in one valley.  Length rather than the drop
    inland, because the drop is one step's view, and on its own it happily picks a hex
    that descends steeply and meets the sea three hexes later.

    No candidate, no draw: a map whose border is all sea draws nothing.
    """
    if count <= 0 or not edges:
        return []
    w, h = arr.shape
    wanted = set(edges)
    candidates: list[tuple[int, int]] = []
    weights: list[float] = []
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
            corner = inlet_corner(state.coord_at(i, j), net, drainage)
            if corner is None:
                continue
            length = (len(_course(drainage, corner)) - 1) * _SIDE_KM
            if length <= 0.0 or length < min_length_km:
                continue
            candidates.append((i, j))
            weights.append(length**length_bias)

    chosen: list[tuple[int, int]] = []
    remaining = list(range(len(candidates)))
    while remaining and len(chosen) < count:
        total = sum(weights[k] for k in remaining)
        if total <= 0.0:
            break
        probs = [weights[k] / total for k in remaining]
        pick = remaining[int(rng.choice(len(remaining), p=probs))]
        chosen.append(candidates[pick])
        here = state.coord_at(*candidates[pick])
        remaining = [
            k
            for k in remaining
            if k != pick and hex_distance(state.coord_at(*candidates[k]), here) >= separation
        ]
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
    meander: np.ndarray | None = None,
    reference_km2: float = WorldConfig.valley_width_reference_km2,
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
    root of drainage area — hence the exponent.  It is measured against a fixed catchment,
    *reference_km2*, at which the reach is `width_max`, and is held there for anything
    bigger.  Not against the largest river on the map: a river's floodplain is a matter of
    its own discharge, and scaled by the largest, one river entering from off the map — or
    one big trunk on a bigger map — narrowed every other belt and took the smaller ones
    away entirely.  Ground standing more than `max_relief`
    above the floor is valley wall: it is left alone, and the fill stops rather than
    stepping over it, which is what keeps a valley to its valley instead of planing the
    countryside, and makes `width_max` a cap rather than the usual outcome.

    Cells are never raised, only cut, and never below the channel doing the cutting — so
    this cannot invent a sink, drown land, or undo the droplets' work.

    Neighbours here are the four of the array rather than the six of the hex grid: this
    fills an area rather than tracing a line, so the shape of the neighbourhood washes out,
    and it is the same square-lattice approximation `_drop_particle` already makes.

    When *meander* is given, each planed cell is credited there with how deep inside the
    belt it lies, and the array keeps the largest credit across calls.  The footprint of
    this fill is exactly the ground the channel has wandered over, which is the ground it
    has floored with its own sediment — the alluvium of a floodplain is laid down by the
    river moving sideways, so it cannot be read off the net vertical deposition the
    droplets record.  A pass can plane a meander belt flat and change the mean elevation
    across it hardly at all.
    """
    if width_max <= 0.0:
        return
    land = arr >= sea_level
    if not land.any():
        return

    flow = np.where(land, discharge, 0.0)
    if float(flow.max()) <= 0.0:
        return

    threshold = float(np.quantile(flow[land], 1.0 - channel_fraction))
    channels = np.argwhere(land & (flow >= threshold))
    if len(channels) == 0:
        return

    w, h = arr.shape
    target = np.full((w, h), np.inf)
    budget = np.zeros((w, h))
    # The reach of the channel each cell was claimed by, carried along with the fill so a
    # cell knows how far into its own belt it lies rather than only how far it is from a
    # channel.  Belts differ in width by an order of magnitude across one map.
    origin = np.zeros((w, h))
    queue: deque = deque()

    for i, j in channels:
        i, j = int(i), int(j)
        reach = min(width_max * (flow[i, j] / reference_km2) ** width_exponent, width_max)
        if reach < 1.0:
            continue
        target[i, j] = arr[i, j]
        budget[i, j] = reach
        origin[i, j] = reach
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
                origin[ni, nj] = origin[i, j]
                queue.append((ni, nj))

    cut = np.isfinite(target)
    arr[cut] = np.minimum(arr[cut], target[cut])

    if meander is not None:
        # How far into its *own* belt a cell lies: full at the channel, nothing at the
        # bluff.  Measured against the reach of the channel that claimed it rather than
        # against the widest on the map, because a small river's floodplain is narrow, not
        # stony — the Loire's silt and a tributary's are the same stuff, and the difference
        # between them is how much ground each covers.  Scaling every belt by the largest
        # one instead left a single bright trunk river on a map of bare valleys.
        share = np.zeros_like(meander)
        np.divide(budget, origin, out=share, where=origin > 0.0)
        np.maximum(meander, np.where(cut, np.clip(share, 0.0, 1.0), 0.0), out=meander)


def _normalise_alluvium(
    deposition: np.ndarray,
    meander: np.ndarray,
    land: np.ndarray,
    quantile: float,
    floodplain_gain: float,
    smoothing: float,
) -> np.ndarray:
    """Combine the two sediment records into a soil depth in [0, 1].

    The two arrive in incomparable units — one is a sum of elevation changes, the other a
    fraction of a reach in cells — so each is brought onto its own [0, 1] before they are
    added, rather than weighted against each other raw.  Otherwise the dial between them
    would mean something different on every map.

    Deposition is scaled against a high quantile rather than its maximum, because the
    maximum is a single cell at the front of one delta.  Dividing by that puts every
    floodplain on the map within a rounding error of bare rock, which is the one thing
    this field exists to distinguish.  Ground above the quantile simply clips.

    Erosion is netted out first and the result floored at zero: a hillside that lost more
    than it gained has no soil left, and how much more is not a depth of anything.
    """
    gained = np.where(land, np.maximum(deposition, 0.0), 0.0)
    positive = gained[gained > 0.0]
    if positive.size:
        scale = float(np.quantile(positive, quantile))
        if scale <= 0.0:
            scale = float(positive.max())
        if scale > 0.0:
            gained = gained / scale

    out = gained + floodplain_gain * np.where(land, meander, 0.0)
    if smoothing > 0.0:
        # Droplets deposit at points and soil does not exist at points.  Blurring before
        # the clamp also stops a single well-travelled cell reading as a lone rich hex in
        # the middle of a valley whose floor is all the same silt.
        out = gaussian_filter(out, sigma=smoothing)
    return np.clip(np.where(land, out, 0.0), 0.0, 1.0)


class ErosionStage(GeneratorStage):
    @staticmethod
    def _profiled(cfg) -> bool:
        """Whether the land was shaped by `elevation_profile` and should leave on it.

        Generated terrain only: an imported heightmap is a picture of somewhere, with its
        own heights, and `none` asks for the noise as erosion leaves it.
        """
        return not cfg.heightmap_path and cfg.elevation_profile != "none"

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
            # Net sediment laid down per cell, in elevation units.  Signed: a cell that is
            # scoured more than it is silted ends up negative, and is floored later.
            deposition = np.zeros((w, h))

            # Initial sample indices (uniform random)
            indices = self.rng.integers(0, n_land, size=n_iter)

            for step in range(n_iter):
                sq, sr = int(land_arr[indices[step], 0]), int(land_arr[indices[step], 1])
                _drop_particle(
                    arr,
                    deposition,
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
                    cfg.erosion_droplet_overcut_m / span,
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

        # Before the carve loop, not after: this takes the per-cell speckle off the
        # droplet field so incision cuts into a clean surface, and running it here is what
        # keeps it from damping the notches incision goes on to cut.
        if cfg.erosion_smoothing_sigma > 0.0:
            arr = gaussian_filter(arr, sigma=cfg.erosion_smoothing_sigma)

        # Put sea level back where the land share says, before anything is carved. The
        # droplets take off a share of every height rather than a depth — about two thirds
        # of it at the lowest, half at the highest — so land shaped by `elevation_profile`
        # into a low plain comes out of them mostly under the sea. The waterline is set
        # again so the same share of the map is land as went in: the coast stays roughly
        # where the noise drew it, and a delta the droplets built out into the shallows
        # stays land. Before the carving, because incision floors at sea level and would
        # otherwise read the drowned plain as sea. The heights themselves are put back on
        # the profile at the end, which keeps their order and so every valley cut here.
        if land_coords and self._profiled(cfg):
            share = len(land_coords) / (w * h)
            arr = arr - (np.quantile(arr, 1.0 - share) - sea_shaped)

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
        carving = cfg.valley_width_max > 0.0 or cfg.erosion_incision_m_per_pass > 0.0
        handoff: tuple[list[tuple[int, int]], dict[Corner, float]] | None = None
        if land_coords and carving:
            # Rivers that enter from off the map bring a catchment this map never had, and
            # nothing here knew about it: accumulation starts every cell at one hex of
            # rain, so an imported trunk was measured as the trickle its first few on-map
            # hexes raise, and widening gave it a trickle's valley.  Seed the inlets with
            # what they actually carry — the same quantity hydrology seeds them with — and
            # the valley follows the water without any special case for it.  They are
            # chosen on the first pass's routing, below, and kept for the rest.
            inflow_volume = max(1.0, cfg.river_inflow_volume * len(land_coords))
            inflow: dict[tuple[int, int], float] | None = None

            # The two height knobs are quoted in metres and the field is normalised, so
            # they are divided by the same span the array was built with.
            meander = np.zeros((w, h))
            draws = _corner_draws(state, self.rng)
            for _ in range(cfg.valley_carve_passes):
                # Drained once per pass, shared by both carving steps: they are the same
                # routing, and a second fill for the same answer is pure cost.
                net, drainage = _drain(
                    arr,
                    sea_shaped,
                    state,
                    draws,
                    cfg.river_wander_exponent,
                    self.rng,
                    cfg.corner_floor_blend,
                )
                if inflow is None:
                    inflow = {
                        cell: inflow_volume
                        for cell in _choose_inlets(
                            arr,
                            sea_shaped,
                            state,
                            net,
                            drainage,
                            self.rng,
                            edges=tuple(cfg.river_inflow_edges),
                            count=cfg.river_inflow_count,
                            separation=cfg.river_inflow_min_separation,
                            min_length_km=cfg.river_inflow_min_length * max(w, h),
                            length_bias=cfg.river_inflow_length_bias,
                        )
                    }
                _accumulate(arr, sea_shaped, state, net, drainage, inflow)
                acc = _hex_discharge(state, drainage, w, h)
                # Incise first, then widen.  Incision cuts the line; widening planes the
                # floor outward from it.  The other way round would plane a floor flat and
                # then notch the floor it had just made.
                _incise_channels(
                    arr,
                    state,
                    drainage,
                    sea_shaped,
                    span,
                    m_per_pass=cfg.erosion_incision_m_per_pass,
                    area_exponent=cfg.erosion_incision_area_exponent,
                    slope_exponent=cfg.erosion_incision_slope_exponent,
                    reference_km2=cfg.erosion_incision_reference_km2,
                    reference_slope=cfg.erosion_incision_reference_slope,
                    min_gradient_m=cfg.erosion_incision_min_gradient_m,
                    max_cut_m=cfg.erosion_incision_max_cut_m,
                )
                if cfg.valley_width_max > 0.0:
                    _widen_valleys(
                        arr,
                        acc,
                        sea_shaped,
                        cfg.valley_width_max,
                        cfg.valley_width_exponent,
                        cfg.valley_floor_slope_m / span,
                        cfg.valley_max_relief_m / span,
                        cfg.valley_channel_fraction,
                        meander,
                        cfg.valley_width_reference_km2,
                    )

            if inflow:
                handoff = (sorted(inflow), draws)

            # Soil is settled last, against the final coastline.  Deposition was recorded
            # over the pre-renormalisation field and the belts over the carved one, but
            # neither is read until here, so both are scored against the ground the rest of
            # the pipeline will actually see.
            alluvium = _normalise_alluvium(
                deposition,
                meander,
                arr >= sea_shaped,
                cfg.alluvium_quantile,
                cfg.alluvium_floodplain_gain,
                cfg.alluvium_smoothing,
            )
            for col in range(w):
                for row in range(h):
                    state.hexes[state.coord_at(col, row)].alluvium = float(alluvium[col, row])

        metres = arr * span - cfg.seabed_depth_m
        # And the land back on the profile it went in with, so the proportions asked for
        # are the proportions the rest of the pipeline sees.
        if self._profiled(cfg):
            metres = apply_profile(metres, cfg)
        for col in range(w):
            for row in range(h):
                state.hexes[state.coord_at(col, row)].elevation = float(metres[col, row])

        if handoff is not None:
            # Hand the inlets to hydrology with the course each runs, so it adopts both
            # rather than choosing and routing again (tech-debt #153).  Routed once more,
            # on the finished ground in metres: the last carve pass was routed before its
            # own cut and widening, and the profile reshapes heights after them, and a
            # course read off either one steps uphill here and there by centimetres.
            # Hydrology will not send water uphill, so it would leave the course there.
            cells, draws = handoff
            net, drainage = _drain(
                metres,
                0.0,
                state,
                draws,
                cfg.river_wander_exponent,
                self.rng,
                cfg.corner_floor_blend,
            )
            courses: list[tuple[HexCoord, list[Corner]]] = []
            for cell in cells:
                coord = state.coord_at(*cell)
                corner = inlet_corner(coord, net, drainage) if metres[cell] >= 0.0 else None
                if corner is not None:
                    courses.append((coord, _course(drainage, corner)))
            state.metadata["inflow_courses"] = courses

        return state
