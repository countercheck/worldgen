"""Hydrology: settle the water on the eroded heightmap, then route the rivers.

`HydrologyStage.run` is the whole sequence in order.  The steps themselves live in two
mixins: `hydrology_fill.py` (sink filling, flow direction, accumulation) and
`hydrology_rivers.py` (river tracing, lake drainage, the endorheic test, hexside routing).
"""

from ..core.hex import HexCoord, TerrainClass
from ..core.hex_grid import neighbors
from ..core.pipeline import GeneratorStage
from ..core.world_state import WorldState
from .hydrology_fill import EdgesOf, HydrologyFill, OnBorder
from .hydrology_rivers import (
    HydrologyRivers,
    _endorheic_components,
    _get_lake_components,
    _write_hex_catchments,
)
from .precipitation import rain_per_hex

# Re-exported so `hydrology.<name>` keeps resolving for anything that reached in before
# the split.  Patching one of these here does not reach the copy the mixins call.
__all__ = [
    "EdgesOf",
    "HydrologyStage",
    "OnBorder",
    "_endorheic_components",
    "_get_lake_components",
    "_write_hex_catchments",
]


class HydrologyStage(HydrologyFill, HydrologyRivers, GeneratorStage):
    def run(self, state: WorldState) -> WorldState:
        w, h = state.width, state.height
        on_border = state.on_border
        hexes = state.hexes

        # Build elevation array and valid coord set
        elev: dict[HexCoord, float] = {c: hx.elevation for c, hx in hexes.items()}
        ocean: set[HexCoord] = {
            c for c, hx in hexes.items() if hx.terrain_class == TerrainClass.OPEN_WATER
        }
        lakes: set[HexCoord] = {
            c for c, hx in hexes.items() if hx.terrain_class == TerrainClass.INLAND_WATER
        }
        land: set[HexCoord] = {
            c
            for c, hx in hexes.items()
            if hx.terrain_class not in (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER)
        }

        # A — Priority-Flood sink filling
        filled = self._priority_flood(elev, land, ocean, on_border)
        # Epsilon tilt: hexes farther from their plateau's drain point get slightly higher
        # filled elevation so that flat plateau areas have a well-defined gradient toward
        # water. Distance is scoped per-plateau (propagated only across equal-elevation
        # neighbors) rather than raw hex-distance-to-ocean — the latter can rank a cell as
        # "closer to water" than its own downhill neighbor, creating a false local minimum
        # that stalls flow_dir in the middle of a plateau.
        drain_dist = self._plateau_drain_distance(filled, land, ocean, lakes, on_border)
        max_dist = max(drain_dist.values()) or 1
        eps = 1e-6
        for coord in filled:
            q, r = coord
            filled[coord] += eps * drain_dist.get(coord, max_dist) / max_dist + eps * 1e-4 * (
                q + r
            ) / (w + h)

        # B — Flow direction (downhill on the filled surface, weighted towards the steepest)
        flow_dir = self._flow_direction(filled, land, ocean, lakes, elev, on_border)

        # B2 — Rivers that arrive from beyond the border.  The map is a region, not a
        # world, so some of its water was gathered off it.  Each inlet is seeded with a
        # catchment it did not earn here, which is what makes it enter already wide.
        inlets = self._inflow_inlets(flow_dir, filled, land, on_border, self._edges_of(state))
        inflow_volume = max(1.0, self.config.river_inflow_volume * len(land))
        inflow = {c: inflow_volume for c in inlets}

        # C — Flow accumulation.  Rain is not uniform: the wind drops it climbing a range
        # and arrives dry beyond, so a catchment in a rain shadow raises a smaller river.
        # The field averages 1.0 per land hex whatever `rain_shadow_strength` is set to,
        # so this redistributes the map's water without changing how much there is.
        rain = rain_per_hex(state, self.config, land)
        acc = self._flow_accumulation(flow_dir, land, inflow, rain)

        # D — Extract river hexes: everywhere enough water passes to cut a channel.
        #
        # This used to take the top 5% of land by accumulation, which is a rank and not a
        # threshold at all despite the name: every map came out with exactly 5.6% of its
        # land under channel, desert and rainforest alike, and the smallest stream on both
        # drained a single square kilometre. Rainfall had no bearing on how much of a
        # region drained.
        #
        # A channel forms where discharge is enough to keep one open, and discharge is
        # catchment area times runoff depth. Runoff is what is left of the rain after the
        # ground and its plants have taken their share, which is why the relationship with
        # climate is so much sharper than rainfall alone: a region getting 200 mm loses
        # nearly all of it to evapotranspiration and supports almost no perennial channel,
        # while one getting 2000 mm sheds most of it.
        land_acc_vals = list(acc.values())
        if not land_acc_vals:
            return state
        runoff_mm = self.config.runoff_mm(self.config.mean_precip_mm)
        if runoff_mm <= 0.0 or self.config.channel_min_discharge <= 0.0:
            state.rivers = []
            return state
        min_catchment = self.config.channel_min_discharge / runoff_mm
        # What a square kilometre of this region sheds in a year, recorded so that anything
        # reading the world — the campaign's Major/Minor river line — turns a catchment
        # into discharge exactly as the generator does, without reimplementing runoff.
        state.metadata["runoff_mm"] = runoff_mm
        river_set: set[HexCoord] = {c for c, a in acc.items() if a >= min_catchment}
        if not river_set:
            # However dry, the map keeps its single largest drainage line so that lakes
            # still have somewhere to spill and the coast still has a river mouth.
            river_set = {max(acc, key=lambda c: acc[c])}

        max_acc = max(land_acc_vals)

        # E — The hex model's own courses.  They are not the map's rivers any more — those
        # run along hexsides, below — but lake drainage is decided on this model: which
        # basins overflow, and through which shore, is a question about whole basins and
        # their water balance, and the routing fallbacks here feed it.
        self._build_rivers(
            river_set,
            flow_dir,
            hexes,
            land,
            ocean,
            lakes,
            acc,
            max_acc,
            filled,
            on_border,
            set(inlets),
        )

        # G — Raise every lake to its spillway and decide which basins drain.
        _, outlet_of = self._ensure_lake_drainage(
            river_set, flow_dir, hexes, land, ocean, lakes, acc, filled, on_border, rain
        )

        # I — Mark basins that still have no way out.  Not every lake can be drained:
        # a bowl ringed by higher ground with no lower lake to spill into is a closed
        # basin, and forcing a river out of it would be a lie about the terrain.  Water
        # leaves such a basin by evaporation instead, so its shore is tagged for
        # BiomeStage to turn into wetland — that is the "percolates out into marshes"
        # outlet, and it keeps the map honest about where the water goes.
        closed = _endorheic_components(hexes, lakes, ocean, flow_dir, outlet_of, on_border)
        for comp in closed:
            for coord in comp:
                hexes[coord].tags.add("endorheic")
            shore = set(comp)
            frontier = set(comp)
            for _ in range(self.config.endorheic_marsh_radius):
                frontier = {
                    n
                    for c in frontier
                    for n in neighbors(c)
                    if n in hexes and n not in shore and n in land
                }
                if not frontier:
                    break
                shore |= frontier
                for coord in frontier:
                    hexes[coord].tags.add("endorheic_shore")

        # R — The rivers themselves, along the sides between hexes.
        self._route_on_corners(state, inlets, closed, min_catchment)
        return state
