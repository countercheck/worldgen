"""Oases: where groundwater surfaces in the desert.

An oasis is a soil fact, not a climate one. The rain is the desert's; what differs is the
water under it, surfacing as a spring or in reach of a well — Kharga, Dakhla and Siwa in
the Egyptian desert, the Fezzan, Tafilalt under the Atlas. It works through the same lever
as `soil_water_carryover`: water added to what soil's dry arm reads for the growing season,
here from an aquifer on a few hexes rather than from last winter's rain everywhere.

Sited where groundwater does surface. A closed depression comes first — a `hollow` or the
rim of an endorheic basin, the Siwa and Qattara kind — then ground lying below its
surroundings, which takes in the foot of an upland where the water table meets the slope.
Candidates are drawn at random with a weight that rises with that rank, so the plausible
sites are the likely ones without every map putting its oases in the same place.

Only on DESERT, so a region whose palette has no desert has none; and SoilStage runs this
on its own child generator, which no other stage shares, so no other stage's draws move.

The spring hex is tagged `oasis`. Desert watering stops and caravanserais (#142) should
read that tag: these are the places a caravan route across the desert would have to pass.
"""

import numpy as np

from ..core.hex import Biome, HexCoord, TerrainClass
from ..core.hex_grid import distance, hex_range, neighbors
from ..core.world_state import WorldState

OASIS_TAG = "oasis"
_DEPRESSION_TAGS = ("hollow", "endorheic_shore")


def place_oases(state: WorldState, cfg, rng: np.random.Generator) -> list[HexCoord]:
    """Water a few desert hexes from below; return the springs, in the order drawn."""
    hexes = state.hexes
    desert = sorted(
        c
        for c, h in hexes.items()
        if h.biome is Biome.DESERT and h.terrain_class is not TerrainClass.OPEN_WATER
    )
    count = int(round(cfg.oasis_per_1000_km2 * len(desert) / 1000.0))
    if count == 0 or cfg.oasis_groundwater_mm <= 0.0:
        return []

    def lie(c):
        """How far the hex lies below the ground around it: positive in a hollow or at the
        foot of a rise."""
        around = [hexes[n].elevation for n in neighbors(c) if n in hexes]
        return (sum(around) / len(around) - hexes[c].elevation) if around else 0.0

    def in_depression(c):
        return any(t in hexes[c].tags for t in _DEPRESSION_TAGS)

    # Only ground the water table can reach: a depression, or ground lying below what
    # surrounds it. Open sand on a rise has its water too far down for any spring.
    sites = [c for c in desert if in_depression(c) or lie(c) > 0.0]
    if not sites:
        return []
    # Weighted by rank rather than by the raw figure, so the weighting has no scale to
    # tune: depressions first, then the lower the ground lies the likelier, and the best
    # site is n times as likely as the least.
    ranked = sorted(sites, key=lambda c: (in_depression(c), lie(c), c))
    weight = np.arange(1, len(ranked) + 1, dtype=float)

    # Far enough apart that two oases' watered ground never touches.
    spacing = 2 * cfg.oasis_radius + 2
    springs: list[HexCoord] = []
    open_ = np.ones(len(ranked), dtype=bool)
    while len(springs) < count and open_.any():
        p = np.where(open_, weight, 0.0)
        pick = ranked[int(rng.choice(len(ranked), p=p / p.sum()))]
        springs.append(pick)
        for i, c in enumerate(ranked):
            if open_[i] and distance(c, pick) < spacing:
                open_[i] = False

    for spring in springs:
        hexes[spring].tags.add(OASIS_TAG)
        for c in hex_range(spring, cfg.oasis_radius):
            h = hexes.get(c)
            if h is not None and h.biome is Biome.DESERT:
                h.groundwater_mm = cfg.oasis_groundwater_mm
    return springs


__all__ = ["OASIS_TAG", "place_oases"]
