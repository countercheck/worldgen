"""Where the rain falls, as a pattern over the terrain.

Wind carries moisture in off the sea and drops it where the ground lifts it.  What is
left downwind of a range is a rain shadow, and the point of computing this before the
rivers rather than after is that a shadow should show in the water: a catchment behind
mountains ought to raise smaller rivers, and a lake in one ought to be likelier to close.

Shared by `ClimateStage`, which turns the pattern into per-hex moisture for the biomes,
and by `HydrologyStage`, which uses it to decide how much rain each hex contributes to
flow accumulation.  Kept in one place because the two must agree: a map whose biomes say
desert while its rivers say floodplain is worse than either being wrong alone.

Depends only on elevation, terrain class and the wind — never on rivers.  That is what
lets hydrology run it, and it is the whole reason this is a separate function rather than
a method on `ClimateStage`, which cannot run first because its later moisture bonuses read
the river tags hydrology produces.
"""

import math

from ..core.hex import HexCoord, TerrainClass
from ..core.hex_grid import neighbors
from ..core.world_state import WorldState


def orographic_pattern(state: WorldState, config) -> dict[HexCoord, float]:
    """Relative precipitation per hex: wind swept across the map, rained out by lift.

    Every hex of open water reads 1.0 — the air above it is saturated.  On land the value
    is what the parcel arriving from upwind gives up climbing to that hex, so it is a
    *pattern* and not a depth: flat ground lifts nothing and so reads near zero, which is
    why callers scale it against its own mean rather than using it raw.
    """
    wind = config.wind_direction
    wlen = math.hypot(wind[0], wind[1])
    if wlen == 0.0:
        wlen = 1.0
    wd = (wind[0] / wlen, wind[1] / wlen)

    # A linear function of the axial coord, as is the pixel transform, so the direction
    # this sweeps is the same one whatever layout the grid uses.
    def pos(coord: HexCoord) -> tuple[float, float]:
        q, r = coord
        return (q + r * 0.5, float(r))

    def dot_wind(coord: HexCoord) -> float:
        x, y = pos(coord)
        return wd[0] * x + wd[1] * y

    sorted_coords = sorted(state.hexes.keys(), key=dot_wind)

    orographic = config.orographic_strength
    sea_level = config.sea_level

    # Atmospheric moisture still aloft, depleted as it rains out.
    atm: dict[HexCoord, float] = {}
    precip: dict[HexCoord, float] = {}
    for coord, h in state.hexes.items():
        if h.terrain_class == TerrainClass.OCEAN:
            atm[coord] = 1.0
        elif h.terrain_class == TerrainClass.LAKE:
            precip[coord] = 1.0

    for coord in sorted_coords:
        h = state.hexes[coord]
        if h.terrain_class in (TerrainClass.OCEAN, TerrainClass.LAKE):
            precip[coord] = 1.0
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
        rained_out = incoming * fraction
        precip[coord] = rained_out
        atm[coord] = max(0.0, incoming - rained_out)

    return precip


def rain_per_hex(state: WorldState, config, land: set[HexCoord]) -> dict[HexCoord, float]:
    """Rain on each land hex, in the unit flow accumulation counts in — one hex's worth.

    `orographic_pattern` is a pattern, not a depth, and taken literally it would leave
    every plain a desert: flat ground lifts no air and so rains out almost nothing, while
    the model has no notion of the frontal rain that actually waters lowlands.  So it is
    scaled to average 1.0 across the land and then blended toward flat rain by
    `rain_shadow_strength`, which reads as: 0 gives every hex the same rain, and 1 takes
    the orographic pattern at its word.

    The mean is held at 1.0 whatever the strength, so the map's total water does not
    change with this setting — only where it falls.  That keeps `river_flow_threshold`
    and `river_inflow_volume`, both of which are fractions, meaning what they did.

    Lake hexes are covered too, since a basin's own surface collects rain and the water
    balance counts it.  Their value is the mean of the land around them rather than their
    own pattern entry, which is 1.0 for every water hex — that 1.0 says the air above
    open water is saturated, which is what makes a sea a moisture *source*, and reading it
    as rainfall would hand a lake in the driest shadow on the map an average soaking.
    """
    strength = config.rain_shadow_strength
    if strength <= 0.0 or not land:
        return dict.fromkeys(land, 1.0)

    pattern = orographic_pattern(state, config)
    total = sum(pattern.get(c, 0.0) for c in land)
    if total <= 0.0:
        return dict.fromkeys(land, 1.0)
    mean = total / len(land)

    rain = {c: max(0.0, 1.0 + strength * (pattern.get(c, 0.0) / mean - 1.0)) for c in land}

    for coord, hx in state.hexes.items():
        if hx.terrain_class != TerrainClass.LAKE:
            continue
        nearby = [rain[n] for n in neighbors(coord) if n in rain]
        rain[coord] = sum(nearby) / len(nearby) if nearby else 1.0
    return rain
