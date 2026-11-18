"""Trade from beyond the map, through the gates on its edges (tech-debt #149).

Every other flow in the model begins and ends on the map, so a road across a desert or
over a range carries only the traffic of the towns either side of it — and on an arid map
that is two to five towns' worth. The places that lived on the long trade — Palmyra and
the caravanserais of the Silk Road, the towns either side of the Alpine passes, an
entrepôt on a coast — lived on traffic that began and ended far beyond them. That traffic
comes in through `WorldState.gates` (`OffmapGateStage`), and is of two kinds:

*   **imports** — a gate's trade with the towns and cities it reaches, split between them
    by size times the share of a cargo that survives the haul, as `CityPromotionStage`
    splits the manufactures between its cities;
*   **transit** — trade only passing through, from a gate to a gate on another edge, split
    between them by what survives the haul. Two sea gates are left out: a ship sailing
    past the map never touches it.

Each gate brings `offmap_trade_share` of the people in the map's towns and cities, split
evenly over the gates, `offmap_transit_share` of it as transit. It is routed by
`bulk_routes` within `haulage_range_land × offmap_trade_range_mult` — and transit
`offmap_transit_range_mult` times further — so it goes by water where it can, and a sea
gate's trade lands at a coastal town.

It is worth from outside, so unlike every other flow it is not conserved: what it brings
is new people. The routing is pure; `CityPromotionStage` applies it and charges each flow
the quays and tolls on its path, and the roads carry it as freight of kind `offmap`.
"""

from dataclasses import dataclass

from ..core.hex import HexCoord, SettlementTier
from ..core.world_state import WorldState
from .haulage import bulk_routes, usable_fraction
from .offmap_gates import SEA
from .riverside import Rivers

IMPORT, TRANSIT = "import", "transit"


@dataclass(frozen=True)
class OffmapFlow:
    """One gate's trade with a town, or with another gate: from *origin* to *dest* along *path*.

    *origin* is always a gate. *path* runs origin to destination inclusive, hex by hex, so
    anything charged where the cargo passes can be read off it. *people* is the worth of
    the cargo in the people it feeds.
    """

    origin: HexCoord
    dest: HexCoord
    people: float
    path: tuple[HexCoord, ...]
    kind: str


def _walk(toward: dict[HexCoord, HexCoord], dest: HexCoord, gate: HexCoord) -> tuple[HexCoord, ...]:
    """The path from *gate* to *dest*, read back off the search seeded at the gate."""
    walk = [dest]
    while walk[-1] != gate:
        walk.append(toward[walk[-1]])
    return tuple(reversed(walk))


def offmap_trade_flows(state: WorldState, cfg, rivers: Rivers) -> list[OffmapFlow]:
    """Every gate's imports and through-trade, in gate order and then by destination.

    One search per gate, seeded at the gate: the route from any hex to it, walked back,
    is the route a cargo from the gate takes there. The cost is priced toward the gate
    rather than away from it, which differs only on slopes, and only in what share of
    the cargo is reckoned to survive.
    """
    gates = state.gates
    if cfg.offmap_trade_share <= 0.0 or not gates:
        return []
    hexes = state.hexes
    live = [s for s in state.settlements if s.tier in (SettlementTier.TOWN, SettlementTier.CITY)]
    total = sum(s.population for s in live)
    if total <= 0:
        return []
    per_gate = cfg.offmap_trade_share * total / len(gates)
    pop = {s.coord: s.population for s in live}
    at_gate = {g.coord for g in gates}
    budget = cfg.haulage_range_land * cfg.offmap_trade_range_mult
    far = budget * cfg.offmap_transit_range_mult

    flows: list[OffmapFlow] = []
    for gate in gates:
        cost, toward = bulk_routes(hexes, [gate.coord], cfg, budget=far, rivers=rivers)

        imports = {
            c: pop[c] * usable_fraction(cost[c], budget)
            for c in sorted(pop)
            if c in cost and c not in at_gate
        }
        through = {
            g.coord: usable_fraction(cost[g.coord], far)
            for g in gates
            if g.edge != gate.edge and not (g.kind == gate.kind == SEA) and g.coord in cost
        }
        for kind, weights, out in (
            (IMPORT, imports, per_gate * (1.0 - cfg.offmap_transit_share)),
            (TRANSIT, through, per_gate * cfg.offmap_transit_share),
        ):
            weights = {c: w for c, w in weights.items() if w > 0.0}
            norm = sum(weights.values())
            if norm <= 0.0 or out <= 0.0:
                continue
            for dest, w in weights.items():
                flows.append(
                    OffmapFlow(
                        gate.coord, dest, out * w / norm, _walk(toward, dest, gate.coord), kind
                    )
                )
    return flows


__all__ = ["IMPORT", "TRANSIT", "OffmapFlow", "offmap_trade_flows"]
