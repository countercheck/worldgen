"""Elevation taken from an image rather than from noise.

Two readings of the same picture, selected by `WorldConfig.heightmap_mode`:

``elevation``
    The image is a greyscale heightmap.  Luminance maps linearly onto [0, 1], black
    lowest and white highest, and that is the terrain.

``coastline``
    The image is only a land/sea stencil.  Heights still come from the generator's own
    noise, shaped so that everything the stencil calls land sits above `sea_level` and
    everything it calls sea sits below.  The coastline you drew is the coastline you get.

The conversion functions are pure and take arrays, so the maths is testable without
touching the filesystem.  The stage delegates the actual file read to
`worldgen.export.heightmap_import`, which is the layer allowed to do I/O.
"""

import warnings

import numpy as np

from ..core.pipeline import GeneratorStage
from ..core.world_state import WorldState
from .elevation import falloff_ramp, noise_field, write_elevations


def _box_axis(a: np.ndarray, m: int, axis: int) -> np.ndarray:
    """Area-average *a* down (or up) to *m* cells along *axis*.

    Exact box filtering by way of the prefix sum.  A pixel row is a piecewise-constant
    function, so its integral is piecewise-linear, so interpolating the cumulative sum at
    fractional cell boundaries is exact rather than approximate — and differencing those
    gives the mean over each cell.  O(n + m) instead of the O(n * m) of building an
    overlap matrix, and it handles the finer-grid-than-image direction too.
    """
    a = np.moveaxis(a, axis, 0)
    n = a.shape[0]
    cs = np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(a, axis=0)], axis=0)

    edges = np.linspace(0.0, n, m + 1)
    # Clamp the index first and take the fraction against the clamped value.  The last
    # edge sits exactly on n, which would otherwise index one past the end; this way it
    # lands on idx = n-1 with frac = 1.0 and reads cs[n], with no special case.
    idx = np.minimum(np.floor(edges).astype(np.intp), n - 1)
    frac = (edges - idx).reshape((-1,) + (1,) * (a.ndim - 1))

    vals = cs[idx] + frac * (cs[idx + 1] - cs[idx])
    out = np.diff(vals, axis=0) / np.diff(edges).reshape((-1,) + (1,) * (a.ndim - 1))
    return np.moveaxis(out, 0, axis)


def resample_to_grid(pixels: np.ndarray, width: int, height: int) -> np.ndarray:
    """Area-average `(px_h, px_w)` pixels onto a `(width, height)` grid.

    The image is stretched to fill the grid on each axis independently, so an aspect
    mismatch distorts rather than cropping or letterboxing.

    Every hex takes the mean of the pixels its footprint covers, not the one pixel under
    its centre, so downsampling a large image does not alias a coastline into a staircase.

    Output is indexed `[col, row]` to match what the elevation stages write back, with
    image row 0 landing on grid row 0.  Both are north, so there is no vertical flip.
    """
    if width < 1 or height < 1:
        raise ValueError(f"grid must be at least 1x1, got {width}x{height}")
    if pixels.ndim != 2 or pixels.size == 0:
        raise ValueError(f"expected a 2-D image array, got shape {pixels.shape}")

    out = _box_axis(_box_axis(pixels.astype(np.float64), height, axis=0), width, axis=1)
    return out.T  # (px_h, px_w) -> (row, col) -> (col, row)


def land_mask(
    lum: np.ndarray, alpha: np.ndarray | None, threshold: float, invert: bool
) -> np.ndarray:
    """Which pixels the stencil calls land.

    Alpha wins where the image has a meaningful one — drawing a continent on a
    transparent background is the natural way to make a stencil, and its brightness then
    means nothing.  `load_luminance` already reports a uniformly opaque band as absent,
    so this only defers to alpha when the alpha actually varies.
    """
    if alpha is not None:
        return alpha >= 0.5
    mask = lum >= threshold
    return ~mask if invert else mask


def _smoothstep(t: np.ndarray) -> np.ndarray:
    return t * t * (3.0 - 2.0 * t)


def shape_to_mask(noise: np.ndarray, mask: np.ndarray, cfg) -> np.ndarray:
    """Fit the noise field to a land/sea stencil.

    Land is lifted into `[sea_level, 1]` and sea pushed into `[continent_seabed,
    sea_level)`, so the classification that follows reproduces the stencil exactly rather
    than approximately.

    Both sides ramp over `continent_shelf_hexes` measured from the coast, which is what
    keeps the shore shallow: a coastal hex sits just off sea level and the terrain only
    reaches its full range once it is a shelf's width inland.  Dropping straight to the
    noise's own values at the water's edge would put a cliff along every coast.

    The result is then stretched to fill [0, 1] with sea level pinned in place.  That is
    not cosmetic.  `ErosionStage` finishes by rescaling whatever range it is handed onto
    [0, 1], which moves sea level relative to the terrain — and this field is built
    against a fixed sea level, so the rescale would drag the whole shelf under.  Measured
    on a 96x83 import, it took the continent from 32% of the map to 2%: a coastline
    dissolved into specks.  Filling the range here makes that rescale a no-op, so erosion
    carves the terrain exactly as it does for a generated world and leaves the coast where
    the stencil put it.
    """
    from scipy.ndimage import distance_transform_edt

    sea_level = float(cfg.sea_level)
    shelf = max(1.0, float(cfg.continent_shelf_hexes))

    # A hair either side of sea level, so a coastal hex classifies as the stencil drew it
    # rather than landing exactly on the threshold and going whichever way the comparison
    # happens to fall.
    margin = min(1e-3, sea_level * 0.01)

    out = np.empty(mask.shape, dtype=np.float64)

    if mask.any():
        # Distance from each land cell to the nearest sea cell, and vice versa.  With no
        # sea at all the transform has nothing to measure from and returns the array
        # diagonal, which saturates the ramp — the whole map reads as deep interior,
        # which is the right answer for a stencil with no coast.
        d_land = distance_transform_edt(mask)
        t = _smoothstep(np.clip(d_land / shelf, 0.0, 1.0))
        floor = sea_level + margin
        out[mask] = (floor + (1.0 - floor) * noise * t)[mask]

    sea = ~mask
    if sea.any():
        d_sea = distance_transform_edt(sea)
        s = _smoothstep(np.clip(d_sea / shelf, 0.0, 1.0))
        ceiling = sea_level - margin
        out[sea] = (ceiling * (1.0 - s))[sea]

    return _fill_range(out, mask, sea_level)


def _fill_range(arr: np.ndarray, mask: np.ndarray, sea_level: float) -> np.ndarray:
    """Stretch *arr* to span [0, 1] with `sea_level` held where it is.

    Each side of the waterline is scaled independently, so no hex crosses it: the deepest
    sea lands on 0, the highest land on 1, and the coastline is exactly where it was.
    """
    out = arr.copy()

    sea = ~mask
    if sea.any():
        depth = sea_level - out[sea].min()
        if depth > 0.0:
            out[sea] = sea_level - (sea_level - out[sea]) * (sea_level / depth)

    if mask.any():
        relief = out[mask].max() - sea_level
        if relief > 0.0:
            out[mask] = sea_level + (out[mask] - sea_level) * ((1.0 - sea_level) / relief)

    return out


class ImageElevationStage(GeneratorStage):
    """`ElevationStage`'s counterpart for a world traced from a picture.

    Substituted for it by `worldgen.stages.stages_for` when `heightmap_path` is set.  The
    swap keeps the stage count identical, so every later stage still draws the same child
    RNG and an imported world differs from a generated one only in its terrain.
    """

    def run(self, state: WorldState) -> WorldState:
        cfg = self.config
        w, h = state.width, state.height

        if not cfg.heightmap_path:
            raise ValueError(
                "ImageElevationStage needs heightmap_path set; it is the picture the "
                "terrain is read from"
            )

        # Deferred so that importing the stage does not drag in the export layer, which
        # pulls matplotlib in through its sibling renderers.
        from ..export.heightmap_import import load_luminance

        lum, alpha = load_luminance(cfg.heightmap_path)

        px_h, px_w = lum.shape
        if px_w < w or px_h < h:
            # A box filter upsamples by replication, so a grid finer than the image gets
            # blocks of identical elevation.  TerrainClassificationStage reads slope, and
            # slope inside a flat block is zero, so most of the map comes out FLAT.
            warnings.warn(
                f"heightmap is {px_w}x{px_h} but the grid is {w}x{h}; the image will be "
                "upsampled into blocks of equal elevation, which flattens the terrain "
                "classification. Use a larger image or a smaller grid.",
                stacklevel=2,
            )

        if cfg.heightmap_mode == "coastline":
            mask = land_mask(lum, alpha, cfg.heightmap_land_threshold, cfg.heightmap_invert)
            # Resample the mask as floats and re-threshold: averaging the coverage of each
            # hex and asking whether it is mostly land is an antialiased downsample, where
            # sampling the stencil at hex centres would drop islands and pinch straits.
            grid_mask = resample_to_grid(mask.astype(np.float64), w, h) >= 0.5
            noise, coast_gen = noise_field(cfg, self.rng, w, h)
            arr = shape_to_mask(noise, grid_mask, cfg)
            if cfg.heightmap_coast_falloff:
                # Opt-in, because the stencil is normally the authority: this rings the
                # map with sea whatever was drawn, which guarantees rivers a coast to
                # reach at the cost of eating a landmass that ran off the edge.
                t = falloff_ramp(cfg, coast_gen, w, h)
                arr = arr * t + cfg.continent_seabed * (1.0 - t)
        else:
            arr = resample_to_grid(lum, w, h)

        write_elevations(state, np.clip(arr, 0.0, 1.0))

        return state
