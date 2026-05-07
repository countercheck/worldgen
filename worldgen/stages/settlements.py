from ..core.hex import Biome, Settlement, SettlementRole, SettlementTier, TerrainClass
from ..core.hex_grid import distance, grade_reachable_count, hex_range, neighbors
from ..core.pipeline import GeneratorStage
from ..core.world_state import WorldState


def _assign_role(coord, hx, hexes) -> SettlementRole:
    nbrs = [hexes[n] for n in neighbors(coord) if n in hexes]

    if (
        hx.river_flow > 0.5
        or hx.terrain_class == TerrainClass.COAST
        or any(n.river_flow > 0.5 for n in nbrs)
        or any(n.terrain_class == TerrainClass.COAST for n in nbrs)
    ):
        return SettlementRole.PORT

    mountain_nbrs = [n for n in nbrs if n.terrain_class == TerrainClass.MOUNTAIN]
    if mountain_nbrs:
        if any(n.elevation > 0.70 for n in mountain_nbrs):
            return SettlementRole.MINING
        return SettlementRole.FORTRESS

    fertile = sum(1 for n in nbrs if n.biome in (Biome.GRASSLAND, Biome.TEMPERATE_FOREST))
    if fertile >= 3:
        return SettlementRole.AGRICULTURAL

    return SettlementRole.MARKET


class SettlementStage(GeneratorStage):
    def run(self, state: WorldState) -> WorldState:
        hexes = state.hexes
        cfg = self.config

        def grade_ok(a_hx, b_hx):
            delta = abs(b_hx.elevation - a_hx.elevation)
            grade_pct = delta * cfg.road_elev_range_m * 100.0 / cfg.hex_size_m
            return grade_pct < cfg.road_slope_cap_pct

        reachable_cache: dict[tuple[int, int], int] = {}

        def reachable(coord):
            if coord not in reachable_cache:
                reachable_cache[coord] = grade_reachable_count(
                    coord, hexes, grade_ok, cfg.settlement_min_reachable
                )
            return reachable_cache[coord]

        # Collect eligible land hexes sorted by habitability
        land = [
            (coord, hx)
            for coord, hx in hexes.items()
            if hx.habitability > 0
            and hx.terrain_class not in (TerrainClass.OCEAN, TerrainClass.LAKE)
        ]
        land.sort(key=lambda x: x[1].habitability, reverse=True)

        settlements: list[Settlement] = []
        city_coords: list = []
        town_coords: list = []
        village_coords: list = []

        # --- Cities ---
        city_idx = 0
        for coord, hx in land:
            if len(city_coords) >= cfg.target_city_count:
                break
            if reachable(coord) < cfg.settlement_min_reachable:
                continue
            if all(distance(coord, c) >= cfg.city_min_separation for c in city_coords):
                pop = int(self.rng.integers(10_000, 50_001))
                role = _assign_role(coord, hx, hexes)
                name = f"{hx.biome.name.lower()}_city_{city_idx}"
                s = Settlement(
                    coord=coord,
                    tier=SettlementTier.CITY,
                    role=role,
                    population=pop,
                    name=name,
                )
                hx.settlement = s
                city_coords.append(coord)
                settlements.append(s)
                city_idx += 1

        # --- Towns ---
        # Adjust scores: halve within 30-hex shadow of any city
        adjusted: dict = {coord: hx.habitability for coord, hx in hexes.items()}
        for city_coord in city_coords:
            for nearby in hex_range(city_coord, 30):
                if nearby in adjusted:
                    adjusted[nearby] *= 0.5

        # Find local maxima in adjusted scores (among non-city land hexes)
        local_maxima = []
        for coord, _hx in land:
            if hexes[coord].settlement is not None:
                continue
            score = adjusted[coord]
            if score <= 0:
                continue
            nbr_scores = [adjusted.get(n, 0.0) for n in neighbors(coord) if n in hexes]
            if score > max(nbr_scores, default=0.0):
                local_maxima.append((score, coord))
        local_maxima.sort(reverse=True)

        town_idx = 0
        for _, coord in local_maxima:
            if len(town_coords) >= cfg.target_town_count:
                break
            hx = hexes[coord]
            if hx.settlement is not None:
                continue
            if reachable(coord) < cfg.settlement_min_reachable:
                continue
            if all(distance(coord, c) >= cfg.town_min_separation for c in town_coords):
                pop = int(self.rng.integers(1_000, 10_001))
                role = _assign_role(coord, hx, hexes)
                name = f"{hx.biome.name.lower()}_town_{town_idx}"
                s = Settlement(
                    coord=coord,
                    tier=SettlementTier.TOWN,
                    role=role,
                    population=pop,
                    name=name,
                )
                hx.settlement = s
                if "confluence" in hx.tags:
                    hx.tags.add("confluence_town")
                town_coords.append(coord)
                settlements.append(s)
                town_idx += 1

        # --- Villages (weighted random with min-distance rejection) ---
        candidates = [
            coord
            for coord, hx in hexes.items()
            if hx.habitability > 0.3
            and hx.terrain_class not in (TerrainClass.OCEAN, TerrainClass.LAKE)
            and hx.settlement is None
        ]
        weights = [
            2.0 if hexes[c].river_flow > 0 or hexes[c].terrain_class == TerrainClass.COAST else 1.0
            for c in candidates
        ]
        total_w = sum(weights)
        probs = [w / total_w for w in weights]

        # Weighted shuffle via argsort of uniform samples divided by probs (Gumbel trick)
        u = self.rng.random(len(candidates))
        order = sorted(range(len(candidates)), key=lambda i: -(u[i] ** (1.0 / probs[i])))

        placed_all = city_coords + town_coords
        village_idx = 0
        for i in order:
            coord = candidates[i]
            hx = hexes[coord]
            if hx.settlement is not None:
                continue
            if reachable(coord) < cfg.settlement_min_reachable:
                continue
            if all(distance(coord, c) >= 3 for c in placed_all):
                pop = int(self.rng.integers(100, 1_001))
                role = _assign_role(coord, hx, hexes)
                name = f"{hx.biome.name.lower()}_village_{village_idx}"
                s = Settlement(
                    coord=coord,
                    tier=SettlementTier.VILLAGE,
                    role=role,
                    population=pop,
                    name=name,
                )
                hx.settlement = s
                village_coords.append(coord)
                placed_all.append(coord)
                settlements.append(s)
                village_idx += 1

        # --- Special pass tags ---
        all_settlement_coords = set(city_coords + town_coords + village_coords)
        for coord, hx in hexes.items():
            if hx.terrain_class != TerrainClass.HILL:
                continue
            if coord in all_settlement_coords:
                continue
            nearby = [c for c in hex_range(coord, 3) if c in hexes and c != coord]
            if hx.habitability == max(
                (hexes[c].habitability for c in nearby if c in hexes),
                default=hx.habitability,
            ):
                hx.tags.add("pass")

        state.settlements = settlements
        return state
