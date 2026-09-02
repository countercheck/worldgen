"""Importing an image as terrain.

Three things are being pinned down here, and they fail in different ways:

* the loader, where the interesting bug is silent — `convert("L")` clamps a 16-bit image
  at 255 instead of rescaling it, so a DEM turns into a two-tone mess with no error;
* the resampler, which is exact box averaging and so can be checked against a slow
  reference rather than merely asserted to be plausible;
* the axis convention, where a transpose or a flip produces a perfectly valid-looking map
  of the wrong world.
"""

import numpy as np
import pytest
from PIL import Image

from worldgen.core.config import WorldConfig
from worldgen.core.pipeline import GeneratorPipeline
from worldgen.export.heightmap_import import load_luminance
from worldgen.stages import default_stages, stages_for
from worldgen.stages.elevation import ElevationStage
from worldgen.stages.image_elevation import (
    ImageElevationStage,
    _box_axis,
    land_mask,
    resample_to_grid,
    shape_to_mask,
)


def _save(tmp_path, arr, name="hm.png", mode=None):
    """Write *arr* as an image. Rows are image rows, so arr is (height, width)."""
    path = tmp_path / name
    im = Image.fromarray(arr) if mode is None else Image.fromarray(arr, mode)
    im.save(path)
    return str(path)


def _gradient(h, w, axis=0):
    """A ramp from black to white along *axis*, as uint8."""
    ramp = np.linspace(0, 255, h if axis == 0 else w, dtype=np.uint8)
    return np.repeat(ramp[:, None], w, 1) if axis == 0 else np.repeat(ramp[None, :], h, 0)


# --- the loader --------------------------------------------------------------


def test_eight_bit_maps_to_unit_range(tmp_path):
    lum, alpha = load_luminance(_save(tmp_path, np.array([[0, 128, 255]], dtype=np.uint8)))
    assert lum[0, 0] == 0.0
    assert lum[0, 2] == 1.0
    assert lum[0, 1] == pytest.approx(128 / 255)
    assert alpha is None


def test_sixteen_bit_is_scaled_not_clipped(tmp_path):
    """The regression this mode branch exists for.

    Pillow reopens a 16-bit PNG as "I;16", and `convert("L")` clamps at 255 rather than
    rescaling — so every value above 1/257 of the range collapses to pure white and a
    real DEM imports as a silhouette. Nothing else in the suite would notice.
    """
    values = [0, 10000, 20000, 30000]
    lum, _ = load_luminance(_save(tmp_path, np.array([values], dtype=np.uint16), "16.png"))

    assert len(set(lum.ravel().tolist())) == len(values), (
        f"16-bit values {values} collapsed to {sorted(set(lum.ravel().tolist()))}"
    )
    assert lum.max() < 1.0, "30000/65535 should not reach full white"
    assert lum[0, 1] == pytest.approx(10000 / 65535)


def test_colour_images_use_luminance(tmp_path):
    red = np.zeros((1, 1, 3), np.uint8)
    red[0, 0] = (255, 0, 0)
    lum, _ = load_luminance(_save(tmp_path, red, "rgb.png"))
    assert lum[0, 0] == pytest.approx(76 / 255, abs=0.01)


def test_alpha_read_only_when_it_varies(tmp_path):
    varied = np.zeros((1, 2, 4), np.uint8)
    varied[0, 0] = (255, 255, 255, 0)
    varied[0, 1] = (255, 255, 255, 255)
    _, alpha = load_luminance(_save(tmp_path, varied, "rgba.png"))
    assert alpha is not None
    assert alpha.tolist() == [[0.0, 1.0]]

    opaque = np.zeros((1, 2, 4), np.uint8)
    opaque[..., 3] = 255
    _, alpha = load_luminance(_save(tmp_path, opaque, "opaque.png"))
    assert alpha is None, "a uniformly opaque band says nothing about where the land is"


def test_palette_images_load(tmp_path):
    src = Image.fromarray(_gradient(4, 4)).convert("P")
    path = tmp_path / "pal.png"
    src.save(path)
    lum, _ = load_luminance(str(path))
    assert lum.shape == (4, 4)
    assert lum.min() >= 0.0 and lum.max() <= 1.0


def test_shape_is_rows_by_columns(tmp_path):
    lum, _ = load_luminance(_save(tmp_path, np.zeros((5, 3), np.uint8), "ns.png"))
    assert lum.shape == (5, 3), "a 3-wide, 5-tall image should load as (5, 3)"


@pytest.mark.parametrize("kind", ["missing", "garbage"])
def test_unreadable_files_raise_value_error(tmp_path, kind):
    """The CLI converts ValueError to a clean message; anything else is a traceback."""
    path = tmp_path / f"{kind}.png"
    if kind == "garbage":
        path.write_bytes(b"this is definitely not a png")

    with pytest.raises(ValueError) as exc:
        load_luminance(str(path))
    assert str(path) in str(exc.value)


# --- the resampler -----------------------------------------------------------


def _dense_reference(a, m, axis=0):
    """Area-average by explicitly building the pixel/cell overlap matrix."""
    a = np.moveaxis(a, axis, 0)
    n = a.shape[0]
    weights = np.zeros((m, n))
    for cell in range(m):
        lo, hi = cell * n / m, (cell + 1) * n / m
        for px in range(n):
            weights[cell, px] = max(0.0, min(px + 1, hi) - max(px, lo))
        weights[cell] /= weights[cell].sum()
    return np.moveaxis((weights @ a.reshape(n, -1)).reshape((m,) + a.shape[1:]), 0, axis)


@pytest.mark.parametrize("n,m", [(10, 3), (3, 10), (7, 7), (1, 5), (5, 1), (100, 37)])
def test_box_average_matches_a_dense_reference(n, m):
    """Exactness, not plausibility: the fast prefix-sum form must equal the slow one."""
    a = np.random.default_rng(0).random((n, 4))
    assert np.allclose(_box_axis(a, m, axis=0), _dense_reference(a, m, axis=0))


def test_downsample_averages_rather_than_samples():
    """A checkerboard is the discriminating case: any point sample gives 0 or 1."""
    board = (np.indices((4, 4)).sum(0) % 2).astype(float)
    assert np.allclose(resample_to_grid(board, 2, 2), 0.5)


def test_upsample_replicates_exactly():
    out = resample_to_grid(np.array([[0.0, 1.0], [2.0, 3.0]]), 4, 4)
    assert np.allclose(out[:2, :2], 0.0)
    assert np.allclose(out[2:, 2:], 3.0)


@pytest.mark.parametrize("px_h,px_w,w,h", [(100, 73, 7, 10), (2, 100, 16, 16), (8, 8, 8, 8)])
def test_resample_shape_mean_and_range(px_h, px_w, w, h):
    px = np.random.default_rng(1).random((px_h, px_w))
    out = resample_to_grid(px, w, h)

    assert out.shape == (w, h), "the grid is indexed [col, row]"
    assert out.mean() == pytest.approx(px.mean()), "a box average preserves the total"
    assert out.min() >= px.min() - 1e-12 and out.max() <= px.max() + 1e-12, (
        f"averaging cannot leave the input range [{px.min():.3f}, {px.max():.3f}]"
    )


def test_constant_image_stays_constant():
    assert np.allclose(resample_to_grid(np.full((7, 5), 0.3), 9, 3), 0.3)


def test_north_stays_north():
    """Image row 0 is the top and grid row 0 is north, so there is no vertical flip."""
    dark_top = np.vstack([np.zeros((10, 6)), np.ones((10, 6))])
    out = resample_to_grid(dark_top, 5, 8)
    assert out[:, 0].mean() < out[:, -1].mean(), "the dark half should land in the north"


def test_west_stays_west():
    dark_left = np.hstack([np.zeros((6, 10)), np.ones((6, 10))])
    out = resample_to_grid(dark_left, 8, 5)
    assert out[0, :].mean() < out[-1, :].mean(), "the dark half should land in the west"


# --- the coastline stencil ---------------------------------------------------


def test_threshold_and_invert():
    lum = np.array([[0.0, 1.0]])
    assert land_mask(lum, None, 0.5, False).tolist() == [[False, True]]
    assert land_mask(lum, None, 0.5, True).tolist() == [[True, False]]


def test_alpha_overrides_brightness():
    """A stencil drawn on transparency is white-on-nothing; brightness means nothing."""
    lum = np.array([[0.0, 1.0]])
    assert land_mask(lum, np.array([[1.0, 0.0]]), 0.5, False).tolist() == [[True, False]]


def test_stencil_is_honoured_exactly():
    """The mode's whole promise: what you drew as land classifies as land."""
    cfg = WorldConfig(width=40, height=40)
    noise = np.random.default_rng(2).random((40, 40))
    yy, xx = np.mgrid[0:40, 0:40]
    mask = ((xx - 20) ** 2 + (yy - 20) ** 2) < 12**2

    out = shape_to_mask(noise, mask, cfg)
    assert (out[mask] >= cfg.sea_level).all(), "drawn land dipped below sea level"
    assert (out[~mask] < cfg.sea_level).all(), "drawn sea rose above sea level"
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_shaped_field_fills_the_unit_range():
    """The property that keeps the coast alive through erosion.

    `ErosionStage` finishes by rescaling whatever range it is handed onto [0, 1]. A field
    that already spans it is left alone; one that does not has sea level dragged through
    its terrain. Before this was enforced, a 32%-land stencil came out of the full
    pipeline at 2% — the continent broke up into specks.
    """
    cfg = WorldConfig(width=48, height=48)
    noise = np.random.default_rng(4).random((48, 48))
    yy, xx = np.mgrid[0:48, 0:48]
    mask = ((xx - 24) ** 2 + (yy - 24) ** 2) < 14**2

    out = shape_to_mask(noise, mask, cfg)
    assert out.min() == pytest.approx(0.0), f"deepest sea is {out.min():.3f}, not 0"
    assert out.max() == pytest.approx(1.0), f"highest land is {out.max():.3f}, not 1"


def test_coastline_survives_erosion():
    """End to end on the stage that used to destroy it.

    Not an exact-preservation test: erosion's `gaussian_filter` softens the shore and
    moves the waterline a couple of points either way, and its deposition genuinely
    raises land, as it does for a generated world. What is being guarded against is the
    collapse — the same stencil used to come out of the pipeline at a fourteenth of its
    area.
    """
    from worldgen.core.world_state import WorldState
    from worldgen.stages.elevation import write_elevations
    from worldgen.stages.erosion import ErosionStage

    n = 64
    cfg = WorldConfig(width=n, height=n, erosion_iterations=0)
    noise = np.random.default_rng(5).random((n, n))
    yy, xx = np.mgrid[0:n, 0:n]
    mask = ((xx - n // 2) ** 2 + (yy - n // 2) ** 2) < 24**2

    ws = WorldState.empty(1, n, n)
    write_elevations(ws, shape_to_mask(noise, mask, cfg))
    before = np.mean([h.elevation >= cfg.sea_level for h in ws.hexes.values()])

    ErosionStage(cfg, np.random.default_rng(0)).run(ws)
    after = np.mean([h.elevation >= cfg.sea_level for h in ws.hexes.values()])

    assert after == pytest.approx(before, abs=0.05), (
        f"erosion moved the waterline from {before:.1%} land to {after:.1%}"
    )


def test_coast_is_shallower_than_the_interior():
    """Land ramps up over the shelf, so the shore is not a wall."""
    cfg = WorldConfig(width=40, height=40)
    noise = np.ones((40, 40))
    yy, xx = np.mgrid[0:40, 0:40]
    mask = ((xx - 20) ** 2 + (yy - 20) ** 2) < 15**2

    out = shape_to_mask(noise, mask, cfg)
    rim = mask & ~np.roll(mask, 1, axis=0)
    assert out[rim].mean() < out[20, 20], "the coast should sit below the interior"


@pytest.mark.parametrize("fill", [True, False])
def test_degenerate_stencils_still_produce_a_field(fill):
    """An all-land or all-sea image is legal input, not a crash."""
    cfg = WorldConfig(width=16, height=16)
    noise = np.random.default_rng(3).random((16, 16))
    out = shape_to_mask(noise, np.full((16, 16), fill), cfg)
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0


# --- the stage ---------------------------------------------------------------


def _run_stage(path, width=16, height=16, seed=42, **over):
    cfg = WorldConfig(width=width, height=height, heightmap_path=path, **over)
    pipeline = GeneratorPipeline(seed, cfg)
    pipeline.add_stage(ImageElevationStage)
    return pipeline.run()


@pytest.mark.parametrize("layout", ["axial", "offset"])
@pytest.mark.parametrize("mode", ["elevation", "coastline"])
def test_stage_fills_every_hex(tmp_path, layout, mode):
    path = _save(tmp_path, _gradient(64, 64), f"{layout}-{mode}.png")
    state = _run_stage(path, grid_layout=layout, heightmap_mode=mode)

    assert len(state.hexes) == 16 * 16
    elevations = [h.elevation for h in state.hexes.values()]
    assert all(0.0 <= e <= 1.0 for e in elevations)
    assert len(set(elevations)) > 1, "every hex kept its default — nothing was written"


def test_stage_preserves_image_orientation(tmp_path):
    """A dark-north image must produce a low-north world, in either layout."""
    path = _save(tmp_path, _gradient(64, 64), "ns-grad.png")
    state = _run_stage(path)

    north = [state.hexes[state.coord_at(c, 0)].elevation for c in range(16)]
    south = [state.hexes[state.coord_at(c, 15)].elevation for c in range(16)]
    assert np.mean(north) < np.mean(south), "the dark top of the image should be the north"


def test_stage_is_reproducible(tmp_path):
    path = _save(tmp_path, _gradient(64, 64, axis=1), "repro.png")
    a = _run_stage(path, heightmap_mode="coastline")
    b = _run_stage(path, heightmap_mode="coastline")
    assert {c: h.elevation for c, h in a.hexes.items()} == {
        c: h.elevation for c, h in b.hexes.items()
    }


def test_stage_without_a_path_is_rejected():
    cfg = WorldConfig(width=8, height=8)
    pipeline = GeneratorPipeline(1, cfg)
    pipeline.add_stage(ImageElevationStage)
    with pytest.raises(ValueError, match="heightmap_path"):
        pipeline.run()


def test_upsampling_warns(tmp_path):
    """Blocky elevation flattens terrain classification, so it should not pass quietly."""
    path = _save(tmp_path, _gradient(8, 8), "tiny.png")
    with pytest.warns(UserWarning, match="upsampled"):
        _run_stage(path, width=64, height=64)


def test_coast_falloff_sinks_the_border(tmp_path):
    """Opt-in, and when opted into it must actually ring the map with sea."""
    path = _save(tmp_path, np.full((64, 64), 255, np.uint8), "allland.png")
    cfg_kw = dict(heightmap_mode="coastline", width=32, height=32)
    plain = _run_stage(path, **cfg_kw)
    ringed = _run_stage(path, heightmap_coast_falloff=True, **cfg_kw)

    corner = plain.coord_at(0, 0)
    assert plain.hexes[corner].elevation >= plain.metadata["config"]["sea_level"]
    assert ringed.hexes[corner].elevation < ringed.metadata["config"]["sea_level"], (
        "heightmap_coast_falloff should pull the map edge under water"
    )


# --- the stage swap ----------------------------------------------------------


def test_swap_is_a_no_op_without_a_heightmap():
    assert stages_for(WorldConfig()) == default_stages()


def test_swap_replaces_only_the_elevation_stage(tmp_path):
    """Positional, and the same length.

    `GeneratorPipeline.run` draws one child seed per stage from the parent stream before
    constructing it, so an equal-length substitution is what guarantees hydrology, climate
    and settlement see exactly the seeds they would have in a generated world.
    """
    path = _save(tmp_path, _gradient(8, 8), "swap.png")
    base = default_stages()
    swapped = stages_for(WorldConfig(heightmap_path=path))

    assert len(swapped) == len(base)
    assert ImageElevationStage in swapped and ElevationStage not in swapped
    for got, expected in zip(swapped, base, strict=True):
        assert got is expected or (expected is ElevationStage and got is ImageElevationStage)
