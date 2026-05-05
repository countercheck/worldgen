import math

from ..core.hex import TerrainClass
from ..core.hex_grid import neighbors
from ..core.pipeline import GeneratorStage
from ..core.world_state import WorldState


class ClimateStage(GeneratorStage):
    def run(self, state: WorldState) -> WorldState:
        self._compute_temperature(state)
        self._compute_moisture(state)
        return state

    def _compute_temperature(self, state: WorldState) -> None:
        height = state.height
        lat_range = self.config.latitude_temp_range
        lapse = self.config.altitude_lapse_rate

        for (_, r), h in state.hexes.items():
            row_frac = r / max(height - 1, 1)
            lat_temp = math.sin(row_frac * math.pi)
            temp = lat_temp * lat_range + (1.0 - lat_range) / 2.0
            temp -= h.elevation * lapse
            h.temperature = max(0.0, min(1.0, temp))

        # Smooth temperature so it varies gradually over space rather than
        # jumping sharply between adjacent hexes that differ in elevation.
        for _ in range(5):
            smoothed = {}
            for coord, h in state.hexes.items():
                nbr_temps = [h.temperature]
                for n in neighbors(coord):
                    if n in state.hexes:
                        nbr_temps.append(state.hexes[n].temperature)
                smoothed[coord] = sum(nbr_temps) / len(nbr_temps)
            for coord, t in smoothed.items():
                state.hexes[coord].temperature = t

    def _compute_moisture(self, state: WorldState) -> None:
        wind = self.config.wind_direction
        wlen = math.hypot(wind[0], wind[1])
        if wlen == 0.0:
            wlen = 1.0
        wd = (wind[0] / wlen, wind[1] / wlen)

        def pos(coord):
            q, r = coord
            return (q + r * 0.5, float(r))

        def dot_wind(coord):
            x, y = pos(coord)
            return wd[0] * x + wd[1] * y

        sorted_coords = sorted(state.hexes.keys(), key=dot_wind)

        # Atmospheric moisture carried by wind (depleted by orographic precipitation)
        atm: dict = {}
        for coord, h in state.hexes.items():
            if h.terrain_class == TerrainClass.OCEAN:
                atm[coord] = 1.0

        orographic = self.config.orographic_strength
        sea_level = self.config.sea_level

        for coord in sorted_coords:
            h = state.hexes[coord]
            if h.terrain_class == TerrainClass.OCEAN:
                h.moisture = 1.0
                atm[coord] = 1.0
                continue

            hx, hy = pos(coord)
            upwind_vals = []
            for n in neighbors(coord):
                if n not in state.hexes:
                    continue
                nx, ny = pos(n)
                # Neighbor is upwind if it lies opposite the wind direction
                if wd[0] * (nx - hx) + wd[1] * (ny - hy) < 0 and n in atm:
                    upwind_vals.append(atm[n])

            incoming = sum(upwind_vals) / len(upwind_vals) if upwind_vals else 1.0

            lift = max(0.0, h.elevation - sea_level)
            fraction = min(1.0, lift * orographic)
            precip = incoming * fraction
            h.moisture = precip
            atm[coord] = max(0.0, incoming - precip)

        # River-adjacency and coastal moisture bonuses
        for coord, h in state.hexes.items():
            if h.terrain_class == TerrainClass.OCEAN:
                continue
            for n in neighbors(coord):
                if n in state.hexes and state.hexes[n].river_flow > 0:
                    h.moisture += 0.15
                    break
            for n in neighbors(coord):
                if n in state.hexes and state.hexes[n].terrain_class == TerrainClass.OCEAN:
                    h.moisture += 0.1
                    break

        # Normalize land moisture to [0, 1]
        land_vals = [
            h.moisture for h in state.hexes.values() if h.terrain_class != TerrainClass.OCEAN
        ]
        if land_vals:
            lo = min(land_vals)
            hi = max(land_vals)
            span = hi - lo if hi > lo else 1.0
            for h in state.hexes.values():
                if h.terrain_class != TerrainClass.OCEAN:
                    h.moisture = (h.moisture - lo) / span
