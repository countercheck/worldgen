import math

import numpy as np
from scipy.ndimage import gaussian_filter

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
        w, h = state.width, state.height
        height = state.height
        base = self.config.base_temperature
        lat_range = self.config.latitude_temp_range
        lapse = self.config.altitude_lapse_rate

        for (_, r), hx in state.hexes.items():
            row_frac = r / max(height - 1, 1)
            lat_temp = math.sin(row_frac * math.pi)
            # Subtract the mean of sin over [0, π] (= 2/π ≈ 0.637) so that
            # base_temperature is the true map mean temperature.
            temp = base + (lat_temp - 2.0 / math.pi) * lat_range
            temp -= hx.elevation * lapse
            hx.temperature = max(0.0, min(1.0, temp))

        # Smooth temperature with gaussian_filter (replaces 5 manual neighbor-average passes)
        temp_arr = np.array([[state.hexes[(q, r)].temperature for r in range(h)] for q in range(w)])
        temp_arr = gaussian_filter(temp_arr, sigma=1.0)
        for q in range(w):
            for r in range(h):
                state.hexes[(q, r)].temperature = float(temp_arr[q, r])

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
            elif h.terrain_class == TerrainClass.LAKE:
                h.moisture = 1.0

        orographic = self.config.orographic_strength
        sea_level = self.config.sea_level

        for coord in sorted_coords:
            h = state.hexes[coord]
            if h.terrain_class in (TerrainClass.OCEAN, TerrainClass.LAKE):
                h.moisture = 1.0
                if h.terrain_class == TerrainClass.OCEAN:
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
        water = (TerrainClass.OCEAN, TerrainClass.LAKE)
        for coord, h in state.hexes.items():
            if h.terrain_class in water:
                continue
            if self.config.moisture_bleed_passes == 0:
                for n in neighbors(coord):
                    if n in state.hexes and "river" in state.hexes[n].tags:
                        h.moisture += 0.15
                        break
            for n in neighbors(coord):
                if n in state.hexes and state.hexes[n].terrain_class in water:
                    h.moisture += 0.1
                    break

        # Normalize land moisture to [0, 1]
        land_vals = [h.moisture for h in state.hexes.values() if h.terrain_class not in water]
        if land_vals:
            lo = min(land_vals)
            hi = max(land_vals)
            span = hi - lo if hi > lo else 1.0
            for h in state.hexes.values():
                if h.terrain_class not in water:
                    h.moisture = (h.moisture - lo) / span

        # Elevation-gated bleed: river moisture spreads to adjacent lower-or-equal hexes
        if self.config.moisture_bleed_passes > 0:
            for _ in range(self.config.moisture_bleed_passes):
                additions: dict = {}
                for coord, h in state.hexes.items():
                    if h.terrain_class in water:
                        continue
                    best = 0.0
                    for n in neighbors(coord):
                        if n not in state.hexes:
                            continue
                        nh = state.hexes[n]
                        if nh.terrain_class in water:
                            continue
                        if nh.elevation <= h.elevation + 1e-6 and nh.river_flow > best:
                            best = nh.river_flow
                    additions[coord] = best
                for coord, h in state.hexes.items():
                    if h.terrain_class not in water:
                        h.moisture = min(
                            1.0, h.moisture + self.config.moisture_bleed_strength * additions[coord]
                        )
            # Re-normalize after bleed
            land_vals = [h.moisture for h in state.hexes.values() if h.terrain_class not in water]
            if land_vals:
                lo = min(land_vals)
                hi = max(land_vals)
                span = hi - lo if hi > lo else 1.0
                for h in state.hexes.values():
                    if h.terrain_class not in water:
                        h.moisture = (h.moisture - lo) / span

        base = self.config.base_moisture
        if base != 0.0:
            for h in state.hexes.values():
                if h.terrain_class not in water:
                    h.moisture = max(0.0, min(1.0, h.moisture + base))
