"""Places that live on the people going past: tolls on the roads, and crossroads (#142).

The bulk trade `CityPromotionStage` charges is cargo — grain, manufactures, the river
trade — and it goes by water wherever it can, so it barely touches a bridge. The traffic a
bridge town, a pass town or a caravan city lived on was the other kind: travellers, droves
and carts on the road. Pontage was charged by the cart and the head of cattle; the Great
St Bernard hospice and the towns either side of the Alps lived on the mule trains; Palmyra
and the caravanserais of the Silk Road on the camel trains that had to water there.

`InterurbanRoadStage` routes that traffic and keeps every journey (`WorldState.journeys`):
how many a year between each pair of places, and the path they take. Here those journeys
pay their way:

*   **Tolls** — at a bridge over water too big to wade, at a pass (`chokepoints.is_pass`),
    and at a watering stop on a desert crossing longer than a day's march, which is the
    oasis the road comes nearest if there is one and the best-watered roadside hex if not.
    Each journey crossing pays `toll_per_journey` food to the settlement within
    `toll_radius` of the site, unless that settlement is one of the journey's own ends.
*   **Crossroads** — a road hex where three or more busy roads meet, away from any
    settlement. The passing trade there is worth `toll_per_journey` per journey through
    it, and it scores that times the number of roads meeting; a hex scoring
    `crossroads_min_draw` founds a crossroads town.

Conserved, as every other flow is: what a toll or a crossroads keeps comes off the two
ends of the journeys that paid it, half from each. A toll nobody collects founds a
settlement where it brings `toll_min_draw` food, through the same rule `ResourceStage`
uses for the bulk tolls.

Pure: this reads the world and returns what is owed; `ChokepointStage` applies it.
"""

from collections import defaultdict

from ..core.hex import SOIL_RANK, Biome, HexCoord
from ..core.hex_grid import distance, hex_range
from .riverside import Rivers

BRIDGE, PASS, DESERT, CROSSROADS = "bridge", "pass", "desert", "crossroads"
OASIS = "oasis"

# A journey's two ends, in `road_edge_key` order.
Pair = tuple[HexCoord, HexCoord]


def _bridge_site(a, b, hexes) -> HexCoord:
    """The end of a bridge a town grows at: the bank with the better soil."""
    return max((a, b), key=lambda c: (SOIL_RANK.get(hexes[c].soil, 0), c == a))


def _watering_stop(run, hexes, cfg) -> HexCoord | None:
    """Where a caravan crossing *run* of desert waters: by the oasis the road comes nearest.

    The roadside hex closest to an `oasis` spring within `toll_radius` of the road, if one
    is; otherwise the roadside hex with the most groundwater, then the most rain. Only on
    the road, since a caravanserai stood on the way and not off it.
    """
    on_road = [c for c in run if hexes[c].road_connections]
    if not on_road:
        return None
    springs = {
        n
        for c in on_road
        for n in hex_range(c, cfg.toll_radius)
        if n in hexes and OASIS in hexes[n].tags
    }
    if springs:
        return min(on_road, key=lambda c: (min(distance(c, s) for s in springs), c))
    return max(on_road, key=lambda c: (hexes[c].groundwater_mm, hexes[c].moisture, c))


def toll_sites(
    state, cfg, rivers: Rivers, is_pass
) -> dict[tuple[HexCoord, str], dict[Pair, float]]:
    """Every road toll point and the journeys crossing it: `(site, kind) -> {pair: n}`.

    *is_pass* says whether a hex is a pass; it is `chokepoints.is_pass`, passed in rather
    than imported so this module need not depend on the stage that applies it.
    """
    hexes = state.hexes
    day = int(cfg.market_day_radius)
    sites: dict[tuple[HexCoord, str], dict[Pair, float]] = defaultdict(dict)
    for pair, (n, path) in sorted(state.journeys.items()):
        if n <= 0.0:
            continue
        crossed: set[tuple[HexCoord, str]] = set()
        for a, b in zip(path, path[1:], strict=False):
            if frozenset((a, b)) in rivers.bridged and b in hexes[a].road_connections:
                crossed.add((_bridge_site(a, b, hexes), BRIDGE))
        for c in path[1:-1]:
            if hexes[c].road_connections and is_pass(c):
                crossed.add((c, PASS))
        run: list[HexCoord] = []
        for c in (*path, None):
            if c is not None and hexes[c].biome is Biome.DESERT:
                run.append(c)
                continue
            if len(run) >= day:
                stop = _watering_stop(run, hexes, cfg)
                if stop is not None:
                    crossed.add((stop, DESERT))
            run = []
        for key in crossed:
            sites[key][pair] = sites[key].get(pair, 0.0) + n
    return dict(sites)


def crossroads(state, cfg) -> list[tuple[float, HexCoord, dict[Pair, float]]]:
    """Road hexes where three or more busy roads meet, best first: `(score, hex, journeys)`.

    A road is busy where it carries `road_min_traffic` journeys, the same bar that makes it
    a road at all; a junction of three is a crossroads, of two only a bend. The score is the
    passing trade in food (`toll_per_journey` per journey through the hex) times the roads
    meeting there, so a hub of five quiet roads can rival a junction of three busy ones.
    """
    hexes = state.hexes
    busy: dict[HexCoord, int] = defaultdict(int)
    for (a, b), edge in state.road_edges.items():
        if edge.traffic >= cfg.road_min_traffic:
            busy[a] += 1
            busy[b] += 1
    meeting = {c: k for c, k in busy.items() if k >= 3 and hexes[c].settlement is None}
    if not meeting:
        return []
    through: dict[HexCoord, dict[Pair, float]] = defaultdict(dict)
    for pair, (n, path) in sorted(state.journeys.items()):
        for c in path[1:-1]:
            if c in meeting:
                through[c][pair] = through[c].get(pair, 0.0) + n
    out = []
    for c, pairs in through.items():
        score = sum(pairs.values()) * cfg.toll_per_journey * meeting[c]
        out.append((score, c, pairs))
    out.sort(key=lambda x: (-x[0], x[1]))
    return out


__all__ = ["BRIDGE", "CROSSROADS", "DESERT", "PASS", "crossroads", "toll_sites"]
