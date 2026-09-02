from ..core.hex import TerrainClass
from ..core.hex_grid import neighbors
from ..core.pipeline import GeneratorStage
from ..core.world_state import WorldState


def gradient_m_per_km(coord, hx, hexes, cfg) -> float:
    """How steeply the ground lies at this hex, in metres of rise per kilometre.

    Measured as *tilt*: the steepest fall across the hex, taken over the three pairs of
    opposite neighbours.  Opposite neighbours are two kilometres apart, hence the halving.

    The obvious alternative — mean absolute difference to all six neighbours — answers a
    different question, and the wrong one.  It reports how rough the *surroundings* are
    rather than how the ground underfoot lies, so it calls a valley floor steep: the
    valley sides stand above it on both flanks, and their height goes into the mean
    whatever the floor itself is doing.  Rivers run along valley floors, which is how that
    version came to price river corridors as mountain and drove roads *away* from the
    banks they should follow.

    Tilt cancels symmetric surroundings, which is what makes it right.  A valley floor
    reads level because both flanks rise equally; so does a ridge crest, because both
    fall equally — and a crest is walkable along, whatever the drop either side.  A
    hillside reads steep, because uphill and downhill neighbours genuinely differ.

    In metres deliberately.  The thresholds used to be fractions of the elevation range,
    so what counted as a mountain moved with `road_elev_range_m` — 120 m/km at the 3000 m
    default, but 20 m/km on a 500 m map, where it called rolling downland a peak.  At
    1 hex = 1 km this is a gradient in the ordinary sense and means the same thing on
    every map.
    """
    ring = neighbors(coord)
    tilt = 0.0
    # neighbors() returns the six in ring order, so i and i+3 are opposite.
    for i in range(3):
        a, b = hexes.get(ring[i]), hexes.get(ring[i + 3])
        if a is None or b is None:
            # On the map edge, fall back to the drop to whichever neighbour is present.
            present = a or b
            if present is not None:
                tilt = max(tilt, abs(hx.elevation - present.elevation))
            continue
        tilt = max(tilt, abs(a.elevation - b.elevation) / 2.0)
    return tilt * cfg.road_elev_range_m


def classify(gradient: float, cfg) -> TerrainClass:
    """Which slope band a gradient falls in."""
    if gradient >= cfg.terrain_escarpment_gradient_m:
        return TerrainClass.ESCARPMENT
    if gradient >= cfg.terrain_steep_gradient_m:
        return TerrainClass.STEEP
    if gradient >= cfg.terrain_rolling_gradient_m:
        return TerrainClass.ROLLING
    return TerrainClass.FLAT


class TerrainClassificationStage(GeneratorStage):
    def run(self, state: WorldState) -> WorldState:
        cfg = self.config
        sea = cfg.sea_level
        coast_threshold = sea + 0.05

        # Pass 1: assign OCEAN
        for h in state.hexes.values():
            if h.elevation < sea:
                h.terrain_class = TerrainClass.OCEAN

        # Pass 2: classify land hexes by how steeply they lie.
        #
        # There is deliberately no altitude term. The old rule made anything above 0.8 of
        # the elevation range a mountain regardless of slope, which put nearly a third of
        # a 128x128 map's "mountain" hexes on ground gentler than 75 m/km — high plateaus
        # and upland basins that are perfectly walkable and farmable, but were priced at
        # ten times flat ground for roads and refused settlement outright. Where altitude
        # genuinely matters it is read directly: the treeline from `biome_alpine_elev`,
        # and mine workings from a settlement's own elevation test.
        for coord, h in state.hexes.items():
            if h.terrain_class == TerrainClass.OCEAN:
                continue

            nbrs = [state.hexes[n] for n in neighbors(coord) if n in state.hexes]
            if h.elevation < coast_threshold and any(
                n.terrain_class == TerrainClass.OCEAN for n in nbrs
            ):
                h.terrain_class = TerrainClass.COAST
                continue

            h.terrain_class = classify(gradient_m_per_km(coord, h, state.hexes, cfg), cfg)

        return state
