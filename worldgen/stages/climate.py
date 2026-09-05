import math

import numpy as np
from scipy.ndimage import gaussian_filter

from ..core.hex import TerrainClass
from ..core.hex_grid import neighbors
from ..core.pipeline import GeneratorStage
from ..core.world_state import WorldState
from .precipitation import orographic_pattern


class ClimateStage(GeneratorStage):
    def run(self, state: WorldState) -> WorldState:
        self._compute_temperature(state)
        self._compute_moisture(state)
        return state

    def _compute_temperature(self, state: WorldState) -> None:
        w, h = state.width, state.height
        cfg = self.config
        base = cfg.mean_temperature_c
        lat_range = cfg.latitude_temp_range_c
        # The lapse rate is quoted per kilometre of ascent and elevation is metres above
        # sea level, so this is a division by a thousand and nothing more.

        # Latitude is the grid *row*, which on an offset grid is the true north-south
        # axis; on an axial grid it is r, as before.
        for coord, hx in state.hexes.items():
            row = state.grid_index(coord)[1]
            row_frac = row / max(h - 1, 1)
            lat_temp = math.sin(row_frac * math.pi)
            # Subtract the mean of sin over [0, π] (= 2/π ≈ 0.637) so that
            # mean_temperature_c is the true map mean.
            temp = base + (lat_temp - 2.0 / math.pi) * lat_range
            # A hex at the waterline gets no lapse, which is what makes the figure a real
            # temperature rather than one relative to the map's own lowest point.
            temp -= max(0.0, hx.elevation) / 1000.0 * cfg.lapse_rate_c_per_km
            hx.temperature = temp

        # Smooth temperature with gaussian_filter (replaces 5 manual neighbor-average passes)
        coords = [[state.coord_at(col, row) for row in range(h)] for col in range(w)]
        temp_arr = np.array([[state.hexes[c].temperature for c in column] for column in coords])
        temp_arr = gaussian_filter(temp_arr, sigma=1.0)
        for col in range(w):
            for row in range(h):
                state.hexes[coords[col][row]].temperature = float(temp_arr[col, row])

    def _compute_moisture(self, state: WorldState) -> None:
        # The wind-and-lift sweep lives in `precipitation` because HydrologyStage needs
        # it too, and needs it *before* this stage runs: a catchment in a rain shadow
        # should raise a smaller river, not merely grow a drier biome.  Sharing the one
        # function is what keeps the two from disagreeing — a map whose biomes say desert
        # while its rivers say floodplain is worse than either being wrong alone.
        for coord, precip in orographic_pattern(state, self.config).items():
            state.hexes[coord].moisture = precip

        # River-adjacency and coastal moisture bonuses
        water = (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER)
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

        # Smear the pattern, the way the temperature field is smeared. Weather systems
        # are wide, and rain falls either side of the ridge that lifted it rather than
        # only on the hex that did the lifting.
        # Indexed by grid column/row, like the temperature smear above: `state.coord_at`
        # is what makes the field the same rectangle whichever layout the grid uses.
        width, height = state.width, state.height
        coords = [[state.coord_at(col, row) for row in range(height)] for col in range(width)]
        arr = np.array([[state.hexes[c].moisture for c in column] for column in coords])
        arr = gaussian_filter(arr, sigma=2.0)
        for col in range(width):
            for row in range(height):
                state.hexes[coords[col][row]].moisture = float(arr[col, row])

        # Then put the pattern into millimetres a year.
        #
        # What the orographic pass produces is *relative* — which slopes catch the rain
        # and which sit in a shadow — and says nothing about whether the region is wet or
        # dry. Scaling so its mean lands on the climate's rainfall supplies that, and a
        # linear scale is the honest way: if a leeward valley receives a third of what the
        # windward slope does, that ratio is a fact about the terrain and should survive
        # being told how wet the region is overall.
        #
        # It used to stretch to [0, 1] and then fit a gamma to move the mean onto a
        # target. The gamma held the bounds while shifting the centre, but it warped the
        # distribution to do it, so the leeward-to-windward ratio came out different for a
        # wet region than a dry one. In millimetres there are no bounds to hold.
        land_vals = [h.moisture for h in state.hexes.values() if h.terrain_class not in water]
        if land_vals:
            mean = sum(land_vals) / len(land_vals)
            scale = (self.config.mean_precip_mm / mean) if mean > 0 else 0.0
            for h in state.hexes.values():
                if h.terrain_class not in water:
                    h.moisture = max(0.0, h.moisture * scale + self.config.base_precip_mm)

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
                        # No ceiling: moisture is millimetres a year now, and a river
                        # valley in a wet region genuinely does receive more than the
                        # old normalised 1.0 would have allowed.
                        h.moisture = (
                            h.moisture
                            + self.config.moisture_bleed_strength
                            * self.config.mean_precip_mm
                            * additions[coord]
                        )
            # No renormalising after the bleed. It used to stretch back to [0, 1], which
            # would now throw away the rainfall the scaling above established — the bleed
            # adds water to river valleys, and that extra water is the point.

        # Standing water is not short of it. Left at the raw carrier value of 1.0 these
        # would read as the driest hexes on the map wherever moisture is drawn or scored.
        for h in state.hexes.values():
            if h.terrain_class in water:
                h.moisture = self.config.mean_precip_mm
