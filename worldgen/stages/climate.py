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
        cfg = self.config
        base = cfg.mean_temperature_c
        lat_range = cfg.latitude_temp_range_c
        # The environmental lapse rate is quoted per kilometre of ascent, so the drop is
        # the hex's height in kilometres times that. Elevation is still a fraction of the
        # range here, hence the conversion through road_elev_range_m; when elevation
        # itself becomes metres this term simplifies.
        lapse_per_elev = cfg.lapse_rate_c_per_km * (cfg.road_elev_range_m / 1000.0)

        for (_, r), hx in state.hexes.items():
            row_frac = r / max(height - 1, 1)
            lat_temp = math.sin(row_frac * math.pi)
            # Subtract the mean of sin over [0, π] (= 2/π ≈ 0.637) so that
            # mean_temperature_c is the true map mean.
            temp = base + (lat_temp - 2.0 / math.pi) * lat_range
            # Measured from sea level: a hex at the waterline gets no lapse, which is what
            # makes the figure a real temperature rather than one relative to the map's
            # own lowest point.
            temp -= max(0.0, hx.elevation - cfg.sea_level) * lapse_per_elev
            hx.temperature = temp

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

            # Anchor the pattern to the region's climate.  What the orographic pass
            # produces is a *relative* map — which slopes catch the rain and which sit in
            # a shadow — and stretching it to [0, 1] says nothing about whether the
            # region is wet or dry.  Its mean lands near 0.15 because precipitation falls
            # off sharply inland, so against a dry threshold of 0.2 almost every hex read
            # as arid whatever climate was asked for, and every region came out
            # shrubland.  A gamma keeps the ordering and the [0, 1] bounds intact while
            # moving the mean onto the value the region's climate calls for, so rain
            # shadow still decides which parts are wetter — just around the right centre.
            target = self.config.regional_moisture
            vals = np.array(
                [h.moisture for h in state.hexes.values() if h.terrain_class not in water]
            )
            if len(vals) and 0.0 < target < 1.0 and vals.max() > 0.0:
                lo_g, hi_g = 0.01, 25.0
                for _ in range(40):
                    mid = (lo_g + hi_g) / 2.0
                    if float((vals**mid).mean()) > target:
                        lo_g = mid
                    else:
                        hi_g = mid
                gamma = (lo_g + hi_g) / 2.0
                for h in state.hexes.values():
                    if h.terrain_class not in water:
                        h.moisture = float(h.moisture**gamma)

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
                        if "river" not in nh.tags:
                            continue
                        if nh.elevation < h.elevation - 1e-6:
                            continue
                        if nh.river_flow > best:
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
