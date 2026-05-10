from ..core.hex import SettlementTier
from ..core.hex_grid import astar
from ..core.pipeline import GeneratorStage
from ..core.world_state import Road, RoadTier, WorldState
from .road_cost import (
    river_discount,
    road_edge_cost,
    tag_river_crossings,
    terrain_base_cost,
)


class VillageTrackStage(GeneratorStage):
    """Connects villages to the nearest existing road hex via TRACK roads."""

    def run(self, state: WorldState) -> WorldState:
        hexes = state.hexes
        cfg = self.config

        villages = [s for s in state.settlements if s.tier == SettlementTier.VILLAGE]
        if not villages:
            return state

        # Road hexes already placed by InterurbanRoadStage
        road_hex_set = {c for c, hx in hexes.items() if hx.road_connections}
        # Also include city/town coords as valid targets
        settled_major = {
            s.coord
            for s in state.settlements
            if s.tier in (SettlementTier.CITY, SettlementTier.TOWN)
        }
        targets = road_hex_set | settled_major

        def node_cost(hx):
            base = terrain_base_cost(hx, cfg)
            return max(0.1, base - river_discount(hx, cfg))

        def edge_cost(from_hx, to_hx):
            return road_edge_cost(from_hx, to_hx, cfg)

        new_roads: list[Road] = []

        for village in villages:
            if not targets:
                break

            # Sort targets by coordinate-distance heuristic; try each until one is reachable.
            sorted_targets = sorted(
                targets,
                key=lambda t: abs(t[0] - village.coord[0]) + abs(t[1] - village.coord[1]),
            )
            path = None
            for candidate in sorted_targets:
                path = astar(hexes, village.coord, candidate, node_cost, edge_cost)
                if path and len(path) >= 2:
                    break
            if path and len(path) >= 2:
                new_roads.append(Road(path=path, tier=RoadTier.TRACK))
                for a, b in zip(path, path[1:], strict=False):
                    if a in hexes and b in hexes:
                        hexes[a].road_connections.add(b)
                        hexes[b].road_connections.add(a)
                # Village's hex is now a road endpoint — add to targets for later villages
                targets.add(village.coord)

        tag_river_crossings(new_roads, hexes)

        state.roads.extend(new_roads)
        return state
