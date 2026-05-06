from ..core.hex import SettlementTier, TerrainClass
from ..core.hex_grid import astar
from ..core.pipeline import GeneratorStage
from ..core.world_state import Road, RoadTier, WorldState


def _terrain_base_cost(hx, cfg) -> float:
    tc = hx.terrain_class
    if tc == TerrainClass.OCEAN:
        return float("inf")
    if tc == TerrainClass.MOUNTAIN:
        return cfg.road_mountain_cost
    if tc == TerrainClass.HILL:
        return cfg.road_hill_cost
    return cfg.road_flat_cost


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
            base = _terrain_base_cost(hx, cfg)
            if base == float("inf"):
                return base
            if hx.river_flow > 0:
                base = max(0.1, base - cfg.road_river_discount)
            return base

        def edge_cost(from_hx, to_hx):
            return abs(to_hx.elevation - from_hx.elevation) * cfg.road_slope_cost

        new_roads: list[Road] = []

        for village in villages:
            if not targets:
                break

            # Find nearest target by hex distance heuristic, then A*
            nearest = min(
                targets,
                key=lambda t: abs(t[0] - village.coord[0]) + abs(t[1] - village.coord[1]),
            )
            path = astar(hexes, village.coord, nearest, node_cost, edge_cost)
            if path and len(path) >= 2:
                new_roads.append(Road(path=path, tier=RoadTier.TRACK))
                for a, b in zip(path, path[1:], strict=False):
                    if a in hexes and b in hexes:
                        hexes[a].road_connections.add(b)
                        hexes[b].road_connections.add(a)
                # Village's hex is now a road endpoint — add to targets for later villages
                targets.add(village.coord)

        # Tag fords/bridges on new track roads
        for road in new_roads:
            path = road.path
            for i, c in enumerate(path):
                if c not in hexes:
                    continue
                hx = hexes[c]
                if hx.river_flow == 0:
                    continue
                prev_c = path[i - 1] if i > 0 else None
                prev_hx = hexes.get(prev_c) if prev_c is not None else None
                if prev_hx is None or prev_hx.river_flow == 0:
                    if "ford" not in hx.tags:
                        hx.tags.add("ford")
                    else:
                        hx.tags.discard("ford")
                        hx.tags.add("bridge")

        state.roads.extend(new_roads)
        return state
