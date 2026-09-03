import numpy as np
from opensimplex import OpenSimplex

from ..core.pipeline import GeneratorStage
from ..core.world_state import WorldState


def _axes(cfg, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """The base coordinate axes the noise and the coast variance both sample on."""
    return np.arange(w) / w * cfg.noise_scale, np.arange(h) / h * cfg.noise_scale


def noise_field(cfg, rng, w: int, h: int) -> tuple[np.ndarray, OpenSimplex]:
    """The raw terrain field: FBM over warped coordinates, normalised to [0, 1].

    Split out from `ElevationStage` so that an imported coastline can be filled with the
    same terrain the generator would have produced on its own, rather than a second
    implementation of it.

    Returns the field and the coast generator, which `apply_continent_falloff` needs.
    Returning it rather than letting the caller derive its own is what keeps the three
    seed draws below in one place and in one order — reordering them, or adding a fourth,
    changes the terrain of every existing seed.
    """
    seed_a = int(rng.integers(0, 2**32))
    seed_b = int(rng.integers(0, 2**32))
    seed_c = int(rng.integers(0, 2**32))
    gen = OpenSimplex(seed_a)
    warp_gen = OpenSimplex(seed_b)
    coast_gen = OpenSimplex(seed_c)

    # Everything here works in grid column/row — the (w, h) array indices — and only the
    # final write-back goes through `state.coord_at` to find the hex, so the field is the
    # same shape whichever layout the grid uses.
    q_1d, r_1d = _axes(cfg, w, h)

    # Domain warp: batch evaluate on the cartesian grid (2 calls instead of 2*w*h)
    warp_x = warp_gen.noise2array(q_1d, r_1d).T * cfg.domain_warp_strength  # (w, h)
    warp_y = warp_gen.noise2array(q_1d + 100.0, r_1d + 100.0).T * cfg.domain_warp_strength

    # Warped coordinates as flat arrays for single-loop FBM
    nx_flat = (q_1d[:, np.newaxis] + warp_x).ravel()  # (w*h,)
    ny_flat = (r_1d[np.newaxis, :] + warp_y).ravel()

    # FBM accumulation via single flat loop (avoids Python double-loop overhead)
    octaves = cfg.noise_octaves
    amps = np.array([cfg.noise_persistence**i for i in range(octaves)])
    freqs = np.array([cfg.noise_lacunarity**i for i in range(octaves)])
    max_val = float(amps.sum())

    noise2 = gen.noise2
    result = np.empty(w * h)
    for i in range(w * h):
        nx, ny = nx_flat[i], ny_flat[i]
        v = 0.0
        for j in range(octaves):
            v += noise2(nx * freqs[j], ny * freqs[j]) * amps[j]
        result[i] = v / max_val
    arr = result.reshape(w, h)

    # The regional tilt is deliberately *not* applied here.  It used to be, and it was
    # competing with the normalisation immediately below, which promptly stretched the
    # result back out — so asking for half a range of tilt got you rather less than that.
    # It now goes on in metres after the conversion, where it does what it says.

    # Normalise to [0, 1] *before* the falloff.  The falloff multiplies towards zero, so
    # it only pulls the map edge underwater if zero is the bottom of the elevation range.
    # Applied to raw noise, which straddles zero, it instead pulls the edge towards the
    # middle of the range: the edge came out around 0.46–0.59 against a 0.45 sea level, so
    # whether the sea reached the border at all was a coin flip per seed, and on a map
    # where it did not every drop of water was trapped inland.
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo)

    return arr, coast_gen


def falloff_ramp(cfg, coast_gen: OpenSimplex, w: int, h: int) -> np.ndarray:
    """The continent falloff's blend factor: 0 at the participating edges, 1 inland.

    Separate from `apply_continent_falloff` because an imported coastline blends towards
    its own seabed rather than the configured one, but wants the same edge ramp.
    """
    edges = set(cfg.continent_falloff_edges)

    # Distance to the nearest *participating* border, in hexes.  Edges left out
    # contribute nothing, so the land runs off the map on that side instead of ending in
    # a coast — a map whose world continues past the border.
    qi = np.arange(w, dtype=float)[:, np.newaxis]
    ri = np.arange(h, dtype=float)[np.newaxis, :]
    far = float(max(w, h)) * 4.0
    dq = np.full((w, 1), far)
    if "west" in edges:
        dq = np.minimum(dq, qi)
    if "east" in edges:
        dq = np.minimum(dq, (w - 1) - qi)
    dr = np.full((1, h), far)
    if "north" in edges:
        dr = np.minimum(dr, ri)
    if "south" in edges:
        dr = np.minimum(dr, (h - 1) - ri)

    # Combine the two axes with a p-norm rather than a plain minimum.  min() is what
    # makes the continent read as a rectangle: it holds the shelf at a constant width
    # right up to where two edges meet, giving a square corner.  The p-norm pulls the
    # corner inward — at 45 degrees the two distances are equal and the result is the
    # smaller of them — so headlands round off.  With one axis unused its term vanishes
    # and this reduces to the other.
    dqs = np.maximum(dq, 1e-9)
    drs = np.maximum(dr, 1e-9)
    edge_dist = (dqs**-2.0 + drs**-2.0) ** -0.5

    # Capped at a quarter of the shorter side: on a map smaller than the shelf itself
    # there is no interior left to be a continent, and the whole thing sinks.  Real maps
    # are far larger than the shelf and never hit this.
    shelf = max(1.0, float(min(cfg.continent_shelf_hexes, min(w, h) // 4)))
    t = np.clip(edge_dist / shelf, 0.0, 1.0)

    if cfg.continent_shelf_variance > 0.0:
        # Let the shelf's inner boundary wander, so the coast is not a band of even
        # width.  The perturbation is multiplicative: it scales t, which is already zero
        # on the border, so however far the coastline swings inland the outermost ring of
        # hexes stays under water and the sea still reaches the map edge.  Low frequency —
        # this shapes bays and headlands, not the metre-by-metre wiggle the terrain noise
        # already provides.
        q_1d, r_1d = _axes(cfg, w, h)
        coast = coast_gen.noise2array(q_1d * 0.6, r_1d * 0.6).T
        t = np.clip(t * (1.0 + cfg.continent_shelf_variance * coast), 0.0, 1.0)

    # Smoothstep, not a straight ramp: easing both ends flattens the shelf where it meets
    # the sea and where it meets the interior, so the coast varies with the noise behind
    # it instead of being a uniform wall.  Where that noise is high the drop is still
    # abrupt, which is what a sea cliff is.
    return t * t * (3.0 - 2.0 * t)


def apply_continent_falloff(
    arr: np.ndarray, cfg, coast_gen: OpenSimplex, w: int, h: int
) -> np.ndarray:
    """Sink the map edge, so the sea rings the continent and rivers have a coast."""
    if not cfg.continent_falloff or not set(cfg.continent_falloff_edges):
        return arr

    t = falloff_ramp(cfg, coast_gen, w, h)

    # Blend towards the bottom of the shaped range, which the metres conversion then puts
    # at `-seabed_depth_m`.  In the old [0, 1] model this blended towards a configured
    # `continent_seabed` fraction; now the depth of the sea floor is a real figure and
    # lives in that conversion, so what the shaped field needs here is simply its floor.
    # Dropping the border further than that puts an abyss against the shore, and the
    # whole descent then has to happen across the shelf — which is what made the coast a
    # cliff whatever width it was given.
    return arr * t


def to_metres(arr: np.ndarray, cfg) -> np.ndarray:
    """A shaped [0, 1] field into metres above sea level.

    Sea level is the datum: land is positive, the sea floor negative, and zero means sea
    level by definition rather than by a configured threshold.  Every test downstream —
    is this ocean, how far above the water does the ground stand, is it above the
    treeline — is then a statement about the world rather than a position on a per-map
    axis.
    """
    span = cfg.max_elevation_m + cfg.seabed_depth_m
    return arr * span - cfg.seabed_depth_m


def apply_tilt(arr: np.ndarray, cfg, w: int, h: int) -> np.ndarray:
    """The regional tilt, in metres: raise one edge of the map by the figure given.

    Applied after the metres conversion, not before the shaping.  In the old model it
    went on first and was immediately undone by a normalisation, so asking for half a
    range of tilt got you rather less.
    """
    gx, gy = cfg.elevation_gradient_m
    if gx == 0.0 and gy == 0.0:
        return arr
    qf = np.linspace(-0.5, 0.5, w)[:, np.newaxis]
    rf = np.linspace(-0.5, 0.5, h)[np.newaxis, :]
    return arr + gx * qf + gy * rf


def write_elevations(state: WorldState, arr: np.ndarray) -> None:
    """Copy a (width, height) field onto the hexes.

    Going through `state.coord_at` is what makes the field layout-agnostic: axial and
    offset grids both index it by the same column/row.
    """
    for col in range(state.width):
        for row in range(state.height):
            state.hexes[state.coord_at(col, row)].elevation = float(arr[col, row])


class ElevationStage(GeneratorStage):
    def run(self, state: WorldState) -> WorldState:
        cfg = self.config
        w, h = state.width, state.height

        arr, coast_gen = noise_field(cfg, self.rng, w, h)
        arr = apply_continent_falloff(arr, cfg, coast_gen, w, h)
        arr = apply_tilt(to_metres(arr, cfg), cfg, w, h)
        write_elevations(state, arr)

        return state
