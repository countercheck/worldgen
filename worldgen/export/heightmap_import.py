"""Reading an image in as terrain.

The only `Image.open` in the feature lives here, because `export/` is the layer that is
allowed to touch the filesystem.  `ImageElevationStage` calls `load_luminance` rather than
opening the file itself.

Everything that can go wrong with a file is normalised to `ValueError`, which is what the
CLI catches and turns into a `ClickException`.  Pillow raises `OSError`,
`UnidentifiedImageError` and `DecompressionBombError` for the various ways an image can be
unreadable, and none of those are `ValueError`, so without this they would each surface as
a raw traceback.
"""

import numpy as np
from PIL import Image, UnidentifiedImageError

# 16-bit greyscale, in the byte orders Pillow distinguishes.
_UINT16_MODES = frozenset({"I;16", "I;16B", "I;16L", "I;16N"})


def load_luminance(path) -> tuple[np.ndarray, np.ndarray | None]:
    """Read *path* as brightness, and alpha where the image carries any.

    Returns `(luminance, alpha)`, both float64 in [0, 1] with shape `(px_h, px_w)`.
    `alpha` is None when the image has no alpha band, or when the band is uniformly
    opaque and so says nothing about which pixels were meant to be there.
    """
    path = str(path)
    try:
        with Image.open(path) as im:
            # `Image.open` only reads the header.  A truncated or corrupt file does not
            # fail until the pixels are actually pulled in.
            im.load()
            lum = _luminance(im, path)
            alpha = _alpha(im)
    except (FileNotFoundError, IsADirectoryError) as exc:
        raise ValueError(f"heightmap {path!r} could not be opened: {exc}") from exc
    except UnidentifiedImageError as exc:
        raise ValueError(f"heightmap {path!r} is not an image Pillow can read") from exc
    except Image.DecompressionBombError as exc:
        raise ValueError(f"heightmap {path!r} is too large to decode safely: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"heightmap {path!r} could not be read: {exc}") from exc

    if lum.size == 0:
        raise ValueError(f"heightmap {path!r} has no pixels")
    return lum, alpha


def _luminance(im: Image.Image, path: str) -> np.ndarray:
    """Brightness as float64 in [0, 1].

    The mode is inspected *before* any conversion.  `convert("L")` on a 16-bit image
    clamps at 255 instead of rescaling, so a 16-bit DEM would come out as a two-tone
    mess with everything above 1/257 of the range reading as pure white.
    """
    if im.mode in _UINT16_MODES:
        return np.asarray(im).astype(np.float64) / 65535.0

    if im.mode == "I":
        # 32-bit integer.  Some 16-bit TIFFs land here rather than in I;16.
        arr = np.asarray(im).astype(np.float64)
        peak = float(arr.max()) if arr.size else 0.0
        if peak > 65535.0:
            raise ValueError(
                f"heightmap {path!r} holds values up to {peak:.0f}, beyond the 16-bit "
                "range this importer can scale; convert it to 8- or 16-bit greyscale"
            )
        return arr / 65535.0

    if im.mode == "F":
        # Floating point is already in whatever units the author chose.  Taking it as
        # [0, 1] is the only reading that does not amount to a histogram stretch.
        arr = np.asarray(im).astype(np.float64)
        if arr.size and (arr.min() < 0.0 or arr.max() > 1.0):
            raise ValueError(
                f"heightmap {path!r} is floating point with values outside [0, 1] "
                f"({arr.min():.3f} to {arr.max():.3f}); rescale it before importing"
            )
        return arr

    # L, LA, P, 1, RGB, RGBA, CMYK.  `convert` handles the palette lookup and the
    # RGB-to-luminance weighting, and drops alpha, which `_alpha` reads separately.
    return np.asarray(im.convert("L")).astype(np.float64) / 255.0


def _alpha(im: Image.Image) -> np.ndarray | None:
    """The alpha band as float64 in [0, 1], or None if it carries no information.

    A fully opaque band is treated as absent: every JPEG-turned-PNG has one, and reading
    it as "all land" would silently override the brightness threshold.
    """
    if "A" not in im.getbands():
        return None
    arr = np.asarray(im.getchannel("A")).astype(np.float64) / 255.0
    if arr.size == 0 or float(arr.min()) >= 1.0:
        return None
    return arr
