from ..core.hex import SettlementTier
from ..core.hex_grid import astar
from ..core.pipeline import GeneratorStage
from ..core.world_state import RoadTier, WorldState, road_edge_key
from .road_cost import (
    WATER,
    bank_discount,
    make_road_edge_cost,
    river_edges,
    river_hex_cost,
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

        # Road hexes already placed by InterurbanRoadStage.  Roads traverse ocean and
        # lake hexes, so a mid-crossing water hex carries road connections and would
        # otherwise be a legal target — leaving a track that runs into the water and
        # stops there.  A track has to join the network on dry land; the route it takes
        # to get there may still cross water, which stays bracketed by its own endpoints.
        road_hex_set = {
            c for c, hx in hexes.items() if hx.road_connections and hx.terrain_class not in WATER
        }
        # Also include city/town coords as valid targets
        settled_major = {
            s.coord
            for s in state.settlements
            if s.tier in (SettlementTier.CITY, SettlementTier.TOWN)
        }
        targets = road_hex_set | settled_major

        # Roads may cross a river but never travel down the channel, so which bank a
        # track runs on stays readable. Settlement hexes are exempt.
        blocked = river_edges(state.rivers)
        settled = {s.coord for s in state.settlements}

        def node_cost(hx):
            base = terrain_base_cost(hx, cfg) + river_hex_cost(hx, cfg)
            return base * (1.0 - bank_discount(hx, hexes, cfg))

        edge_cost = make_road_edge_cost(cfg, blocked, settled)

        new_edges: dict = {}

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
                for a, b in zip(path, path[1:], strict=False):
                    if a in hexes and b in hexes:
                        # A track never demotes a road already laid: where a village lane
                        # joins the highway it is the highway that gets drawn.
                        new_edges.setdefault(road_edge_key(a, b), RoadTier.TRACK)
                        hexes[a].road_connections.add(b)
                        hexes[b].road_connections.add(a)
                # Village's hex is now a road endpoint — add to targets for later villages
                targets.add(village.coord)

        for key, tier in new_edges.items():
            state.road_edges.setdefault(key, tier)
        tag_river_crossings(new_edges, hexes)
        return state
