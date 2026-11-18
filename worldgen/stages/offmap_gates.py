"""Where trade from beyond the map comes in: gates on its edges (tech-debt #149).

The map is a region cut out of a larger world, and the roads and sea lanes that cross it
run on off it. A road leaves by the easiest way out — a valley mouth, a low pass — and a
ship comes in where the open sea runs nearest a coast. So each edge in
`offmap_trade_edges` gets gates in proportion to its length:

*   **land gates**, one per `offmap_land_gate_spacing` hexes of land border, on the border
    hexes from which getting `offmap_gate_depth` hexes into the map costs least over
    `make_travel_cost` — one search per edge, seeded from the line that deep in;
*   **sea gates**, one per `offmap_sea_gate_spacing` hexes of open-water border, on the
    border water nearest a coast.

Gates crowd each other out softly rather than by a hard exclusion. Each candidate is
ranked on its edge, 0 for the cheapest way in and 1 for the dearest, and every chosen
gate within `offmap_gate_crowding_radius` adds `offmap_gate_crowding_penalty × (1 − d /
radius)` to that, so the next one goes elsewhere unless elsewhere is much harder going. The choice is greedy and deterministic —
cheapest penalised score first, ties by coordinate — and reads the terrain only, so it
runs before any settlement is placed and draws nothing from the stage's generator.

What the gates carry is `offmap_trade.offmap_trade_flows`.
"""

from collections import deque

import numpy as np

from ..core import routing
from ..core.hex import HexCoord, TerrainClass
from ..core.hex_grid import distance, neighbors
from ..core.pipeline import GeneratorStage
from ..core.world_state import Gate, WorldState
from .haulage import travel_field
from .riverside import river_index

LAND, SEA = "land", "sea"
_WATER = (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER)


def _depth_line(state: WorldState, edge: str, depth: int) -> list[HexCoord]:
    """The hexes *depth* in from *edge*, clamped to the far side of a small map."""
    w, h = state.width, state.height
    if edge in ("west", "east"):
        col = min(depth, w - 1) if edge == "west" else max(0, w - 1 - depth)
        return [state.coord_at(col, row) for row in range(h)]
    row = min(depth, h - 1) if edge == "north" else max(0, h - 1 - depth)
    return [state.coord_at(col, row) for col in range(w)]


def _land_scores(state, cfg, field, edge: str, border: list[HexCoord]) -> dict[HexCoord, float]:
    """What it costs to get `offmap_gate_depth` hexes inland from each border hex of *edge*.

    One multi-source search from the land on the depth line, priced in the direction a
    traveller coming in walks (`routing.from_seats`). A border hex that cannot reach the
    line at all — a headland cut off by water — has no score and is no gate.
    """
    hexes = state.hexes
    grid = field.grid
    line = [c for c in _depth_line(state, edge, cfg.offmap_gate_depth) if c in grid.index]
    seeds = np.array(
        sorted(grid.index[c] for c in line if hexes[c].terrain_class not in _WATER), np.int64
    )
    if seeds.size == 0:
        return {}
    cost, _, _ = routing.from_seats(
        grid.nbr, field.node, field.edge, np.zeros((0, 6), np.bool_), False, seeds, np.inf
    )
    out: dict[HexCoord, float] = {}
    for c in border:
        v = float(cost[grid.index[c]])
        if np.isfinite(v):
            out[c] = v
    return out


def _sea_scores(state, border: list[HexCoord]) -> dict[HexCoord, float]:
    """How many hexes of water lie between each border sea hex and the nearest land."""
    hexes = state.hexes
    land = [c for c, hx in hexes.items() if hx.terrain_class not in _WATER]
    dist = {c: 0 for c in land}
    queue = deque(sorted(land))
    while queue:
        c = queue.popleft()
        for n in neighbors(c):
            hx = hexes.get(n)
            if hx is not None and n not in dist and hx.terrain_class is TerrainClass.OPEN_WATER:
                dist[n] = dist[c] + 1
                queue.append(n)
    return {c: float(dist[c]) for c in border if c in dist}


def _pick(scores: dict[HexCoord, float], count: int, chosen: list[HexCoord], cfg) -> list[HexCoord]:
    """The *count* best of *scores*, each one pushing the next away from it.

    A score is a cost, and costs on one edge can differ a hundredfold — a valley floor
    against a cliff — so a penalty multiplying them could never push a gate off the
    valley onto its next-door hex. So the costs are read as ranks instead, 0 for the
    cheapest candidate on the edge and 1 for the dearest, and crowding adds
    `offmap_gate_crowding_penalty × (1 − d / radius)` for every gate within the radius:
    at 1, the hex beside a gate is as bad a site as the worst ground on the edge.

    *chosen* holds the gates already placed, on this edge or another, and is extended: a
    gate in a corner crowds the next edge's candidates as much as its own.
    """
    radius, penalty = cfg.offmap_gate_crowding_radius, cfg.offmap_gate_crowding_penalty
    ranked = sorted(scores, key=lambda c: (scores[c], c))
    top = max(1, len(ranked) - 1)
    left = {c: i / top for i, c in enumerate(ranked)}
    picked: list[HexCoord] = []
    for _ in range(count):
        if not left:
            break

        def crowded(c: HexCoord) -> float:
            if radius <= 0 or penalty <= 0.0:
                return 0.0
            return penalty * sum(max(0.0, 1.0 - distance(c, g) / radius) for g in chosen)

        best = min(left, key=lambda c: (left[c] + crowded(c), c))
        picked.append(best)
        chosen.append(best)
        del left[best]
    return picked


def choose_gates(state: WorldState, cfg) -> list[Gate]:
    """The gates on every edge in `offmap_trade_edges`, in the order they were chosen."""
    if cfg.offmap_trade_share <= 0.0 or not cfg.offmap_trade_edges:
        return []
    hexes = state.hexes
    field = travel_field(hexes, cfg, river_index(state, cfg))
    taken: set[HexCoord] = set()
    chosen: list[HexCoord] = []
    gates: list[Gate] = []
    for edge in cfg.offmap_trade_edges:
        border = sorted(c for c in hexes if edge in state.edges_of(c) and c not in taken)
        land = [c for c in border if hexes[c].terrain_class not in _WATER]
        sea = [c for c in border if hexes[c].terrain_class is TerrainClass.OPEN_WATER]
        sea_dist = _sea_scores(state, sea) if sea else {}
        for kind, cells, spacing in (
            (LAND, land, cfg.offmap_land_gate_spacing),
            (SEA, sea, cfg.offmap_sea_gate_spacing),
        ):
            if not cells:
                continue
            scores = _land_scores(state, cfg, field, edge, cells) if kind == LAND else sea_dist
            count = max(1, round(len(cells) / spacing))
            for c in _pick(scores, count, chosen, cfg):
                taken.add(c)
                gates.append(Gate(coord=c, edge=edge, kind=kind))
    return gates


class OffmapGateStage(GeneratorStage):
    """Places the gates off-map trade comes in by (`choose_gates`)."""

    def run(self, state: WorldState) -> WorldState:
        state.gates = choose_gates(state, self.config)
        return state


__all__ = ["LAND", "SEA", "OffmapGateStage", "choose_gates"]
