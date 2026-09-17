"""What shape is the drainage network?

A real river network is a tree: headwater fingers point every which way around a divide,
join in pairs, and the trunks they build carry most of the water.  A network that fails to
do this — channels running side by side down a slope, never meeting — is not a landform,
it is a routing artefact, and the numbers here are what tell the two apart.

Everything is rebuilt from `WorldState.rivers` and the stored hex tags, so nothing new has
to be carried through the pipeline or written to `world.json`.  That works because
`HydrologyStage._split_at_confluences` trims each tributary to *end on* the trunk hex it
joins: the junction hex appears both as the tributary's terminus and as an interior hex of
the trunk, so a plain union of the per-river edge lists puts the fork back together.
"""

import cmath
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field

from ..core.hex import HexCoord, TerrainClass
from ..core.hex_grid import neighbors
from ..core.world_state import WorldState

# Axial step -> its index in `neighbors()`, so an edge can be turned into one of six
# directions without searching the list for every hex on every channel.
_DIRECTION_INDEX: dict[HexCoord, int] = {
    (1, 0): 0,
    (1, -1): 1,
    (0, -1): 2,
    (-1, 0): 3,
    (-1, 1): 4,
    (0, 1): 5,
}

# Six directions over a full turn.
_DIRECTIONS = len(_DIRECTION_INDEX)


@dataclass(frozen=True)
class DrainageNetwork:
    """The drainage graph, rebuilt from the split river paths.

    `downstream` is a forest: every hex flows to at most one other, and the outlets are
    the hexes that flow nowhere (into the sea, off the border, or into a closed basin).
    """

    downstream: dict[HexCoord, HexCoord] = field(default_factory=dict)
    upstream: dict[HexCoord, list[HexCoord]] = field(default_factory=dict)
    order: dict[HexCoord, int] = field(default_factory=dict)
    outlets: frozenset[HexCoord] = frozenset()
    # Edges refused because they contradicted an edge already recorded, or would have
    # closed a loop.  Kept as a number rather than silently dropped: water that arrives
    # at a hex by two different routes means something upstream is wrong, and a metrics
    # module that hides that is measuring a graph nobody generated.
    conflicts: int = 0

    def nodes(self) -> set[HexCoord]:
        """Every hex the network touches, sources and outlets included."""
        return set(self.downstream) | set(self.downstream.values())


def _reaches(start: HexCoord, target: HexCoord, downstream: dict[HexCoord, HexCoord]) -> bool:
    """True if following the flow from *start* arrives at *target*."""
    seen: set[HexCoord] = set()
    node: HexCoord | None = start
    while node is not None and node not in seen:
        if node == target:
            return True
        seen.add(node)
        node = downstream.get(node)
    return False


def build_network(state: WorldState) -> DrainageNetwork:
    """Rebuild the drainage graph from `state.rivers`."""
    downstream: dict[HexCoord, HexCoord] = {}
    conflicts = 0

    for river in state.rivers:
        for upper, lower in zip(river.hexes, river.hexes[1:], strict=False):
            if upper == lower:
                continue
            existing = downstream.get(upper)
            if existing is not None:
                # The same edge traced twice by two rivers is not a conflict; a second,
                # different receiver is.
                if existing != lower:
                    conflicts += 1
                continue
            if _reaches(lower, upper, downstream):
                conflicts += 1
                continue
            downstream[upper] = lower

    upstream: dict[HexCoord, list[HexCoord]] = defaultdict(list)
    for upper, lower in downstream.items():
        upstream[lower].append(upper)
    for feeders in upstream.values():
        feeders.sort()

    order = strahler_orders(downstream, upstream)
    outlets = frozenset(c for c in set(downstream.values()) if c not in downstream)

    return DrainageNetwork(
        downstream=dict(downstream),
        upstream=dict(upstream),
        order=order,
        outlets=outlets,
        conflicts=conflicts,
    )


def strahler_orders(
    downstream: dict[HexCoord, HexCoord],
    upstream: dict[HexCoord, list[HexCoord]],
) -> dict[HexCoord, int]:
    """Strahler stream order for every hex in the network.

    A hex with nothing above it is order 1.  Otherwise it takes the highest order arriving
    from above, promoted by one only when two or more streams of that same highest order
    meet — which is what makes the order a measure of *branching* rather than of length.
    """
    nodes = set(downstream) | set(downstream.values())
    pending = {c: len(upstream.get(c, ())) for c in nodes}
    order: dict[HexCoord, int] = {}
    # Sorted so the walk is reproducible; the result does not depend on the order, but a
    # metric that shuffles between runs is not one anybody can act on.
    frontier = sorted(c for c in nodes if pending[c] == 0)

    while frontier:
        node = frontier.pop()
        feeders = upstream.get(node, ())
        if not feeders:
            order[node] = 1
        else:
            highest = max(order[f] for f in feeders)
            joining = sum(1 for f in feeders if order[f] == highest)
            order[node] = highest + 1 if joining >= 2 else highest
        receiver = downstream.get(node)
        if receiver is not None:
            pending[receiver] -= 1
            if pending[receiver] == 0:
                frontier.append(receiver)

    # Anything left unresolved sat on a cycle `build_network` failed to refuse.  Give it a
    # floor rather than a KeyError, and let `conflicts` be the thing that reports it.
    for node in nodes:
        order.setdefault(node, 1)
    return order


def links_by_order(net: DrainageNetwork) -> dict[int, int]:
    """How many *links* the network has of each Strahler order.

    A link is a maximal run of hexes of one order flowing into each other, and the ratio
    between successive counts is what Horton's law is about.  Counting hexes instead would
    measure how long the channels are and return a ratio near one however the network
    branches, which is the opposite of the question.
    """
    counts: dict[int, int] = defaultdict(int)
    for node, node_order in net.order.items():
        feeders = net.upstream.get(node, ())
        # A hex begins a link when no hex of its own order flows into it.
        if not any(net.order[f] == node_order for f in feeders):
            counts[node_order] += 1
    return dict(counts)


def bifurcation_ratio(link_counts: dict[int, int]) -> float:
    """Horton's bifurcation ratio: the geometric mean of N(w) / N(w+1).

    Real networks sit between about 3 and 5.  Returns NaN when the network never reaches a
    third order, because a ratio taken over a single step is not evidence of anything — and
    a network with no third order is itself the finding.
    """
    if not link_counts:
        return math.nan
    highest = max(link_counts)
    if highest < 3:
        return math.nan
    ratios = [
        math.log(link_counts[w] / link_counts[w + 1])
        for w in range(1, highest)
        if link_counts.get(w) and link_counts.get(w + 1)
    ]
    if not ratios:
        return math.nan
    return math.exp(statistics.fmean(ratios))


def _direction(a: HexCoord, b: HexCoord) -> int | None:
    """Which of the six ways *a* flows to reach *b*, or None if they are not neighbours."""
    return _DIRECTION_INDEX.get((b[0] - a[0], b[1] - a[1]))


def azimuth_concentration(net: DrainageNetwork, orders: set[int] | None = None) -> float:
    """How much the channels agree on a direction, from 0 (isotropic) to 1 (collimated).

    The circular mean resultant length over the flow azimuths.  Read it over order-1
    channels: trunks on a coastal map legitimately share a heading because they all reach
    for the same sea, but the first-order fingers around a divide should point every which
    way.  Headwaters that agree on a direction mean the terrain is a tilted plane.
    """
    total = 0j
    count = 0
    for upper, lower in net.downstream.items():
        if orders is not None and net.order.get(upper) not in orders:
            continue
        idx = _direction(upper, lower)
        if idx is None:
            continue
        total += cmath.exp(2j * math.pi * idx / _DIRECTIONS)
        count += 1
    if count == 0:
        return 0.0
    return abs(total) / count


def river_azimuth_concentration(state: WorldState) -> float:
    """How much whole rivers agree on a heading, from 0 (isotropic) to 1 (collimated).

    Each river is reduced to one mean direction, and the concentration is taken over those
    — not over individual steps.  That is the difference that matters at this resolution:
    a map's channels are sparse enough that two rivers running side by side sit several
    kilometres apart and never touch, so a metric over *touching* hexes cannot see them,
    while a metric over headings can.

    A caution on reading it: with N rivers, an isotropic set still scores about 1/sqrt(N)
    by chance, so a dozen rivers give roughly 0.3 for free.  It is evidence at 0.6 and
    above, not at 0.4.
    """
    headings = []
    for river in state.rivers:
        steps = 0j
        for upper, lower in zip(river.hexes, river.hexes[1:], strict=False):
            idx = _direction(upper, lower)
            if idx is not None:
                steps += cmath.exp(2j * math.pi * idx / _DIRECTIONS)
        if steps != 0:
            headings.append(steps / abs(steps))
    if not headings:
        return 0.0
    return abs(sum(headings)) / len(headings)


def first_order_link_ratio(link_counts: dict[int, int]) -> float:
    """N(1) / N(2) — the bifurcation ratio's first step, defined where the ratio is not.

    Real networks put this between about 3 and 5.  A much larger number means a crowd of
    unbranched first-order threads that never pair up into anything, which is what a
    network with no third order looks like from below.  Infinite when nothing reaches
    second order at all.
    """
    first, second = link_counts.get(1, 0), link_counts.get(2, 0)
    if not first:
        return math.nan
    if not second:
        return math.inf
    return first / second


def pair_statistics(net: DrainageNetwork, channel: set[HexCoord]) -> tuple[float, float]:
    """The fraction of touching channel hexes that run side by side, and that join.

    Two channel hexes that are neighbours either join — one flows into the other — or they
    do not.  When they do not and they are heading the same way, they are two rivers a
    kilometre apart running in parallel and never meeting, which is exactly the artefact
    this module exists to name.  Returns `(parallel_fraction, joined_fraction)`.
    """
    parallel = 0
    joined = 0
    total = 0
    for a in channel:
        for b in neighbors(a):
            if b not in channel or b <= a:  # each unordered pair once
                continue
            total += 1
            if net.downstream.get(a) == b or net.downstream.get(b) == a:
                joined += 1
                continue
            da, db = net.downstream.get(a), net.downstream.get(b)
            if da is not None and db is not None and _direction(a, da) == _direction(b, db):
                parallel += 1
    if total == 0:
        return 0.0, 0.0
    return parallel / total, joined / total


@dataclass(frozen=True)
class DrainageMetrics:
    """What the drainage network of one world measures."""

    land_hexes: int
    channel_hexes: int
    river_count: int
    confluence_count: int
    confluence_rate: float
    strahler_max: int
    link_counts: dict[int, int]
    bifurcation_ratio: float
    first_order_link_ratio: float
    drainage_density: float
    mean_channel_length: float
    median_channel_length: float
    azimuth_concentration: float
    headwater_azimuth_concentration: float
    river_azimuth_concentration: float
    parallel_pair_fraction: float
    joined_pair_fraction: float
    conflicts: int

    def summary(self) -> str:
        """One line, for a debug plate or a commit message."""
        return (
            f"orders {self.strahler_max} · "
            f"confluences {self.confluence_count} ({self.confluence_rate:.3f}/hex) · "
            f"N1/N2 {self.first_order_link_ratio:.1f} · "
            f"R_b {self.bifurcation_ratio:.1f} · "
            f"R(rivers) {self.river_azimuth_concentration:.2f} · "
            f"density {self.drainage_density:.3f}"
        )


def drainage_metrics(state: WorldState) -> DrainageMetrics:
    """Measure the drainage network of a finished world.

    Drainage density is channel hexes over land hexes.  At the project's one hex to the
    kilometre that *is* the usual km/km² figure — a channel hex is a kilometre of channel
    covering a square kilometre of ground — so there is no second number to report.  Note
    that a 1 km sample only resolves channels draining tens of square kilometres, so the
    plausible band here (roughly 0.02 to 0.25) sits well below the 0.5–5 km/km² quoted for
    field surveys of the same landscapes.
    """
    water = (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER)
    land = {c for c, hx in state.hexes.items() if hx.terrain_class not in water}

    net = build_network(state)
    channel = net.nodes() & land

    confluences = [c for c in channel if len(net.upstream.get(c, ())) >= 2]
    lengths = [len(r.hexes) for r in state.rivers]
    counts = links_by_order(net)
    parallel_fraction, joined_fraction = pair_statistics(net, channel)

    return DrainageMetrics(
        land_hexes=len(land),
        channel_hexes=len(channel),
        river_count=len(state.rivers),
        confluence_count=len(confluences),
        confluence_rate=len(confluences) / len(channel) if channel else 0.0,
        strahler_max=max(net.order.values()) if net.order else 0,
        link_counts=counts,
        bifurcation_ratio=bifurcation_ratio(counts),
        first_order_link_ratio=first_order_link_ratio(counts),
        drainage_density=len(channel) / len(land) if land else 0.0,
        mean_channel_length=statistics.fmean(lengths) if lengths else 0.0,
        median_channel_length=statistics.median(lengths) if lengths else 0.0,
        azimuth_concentration=azimuth_concentration(net),
        headwater_azimuth_concentration=azimuth_concentration(net, orders={1}),
        river_azimuth_concentration=river_azimuth_concentration(state),
        parallel_pair_fraction=parallel_fraction,
        joined_pair_fraction=joined_fraction,
        conflicts=net.conflicts,
    )
