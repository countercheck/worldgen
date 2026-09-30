"""Icons for settlements — a pickaxe for a mine, an axe for a camp, a church among roofs
for a city — and marks on rivers: where one rises, where it ends, and its white water.

Drawn as coordinates in a unit square — x right, y down, -1 to 1 either way — so the SVG
exporter, the PNG exporter and the debug viewer all draw one shape and cannot come to
disagree about it. Each icon is a handle, drawn as a thick line, and a head, drawn as a
filled polygon; the exporters scale both into a white disc the size of a settlement marker.
"""

import math
from dataclasses import dataclass

from ..core.hex import SettlementRole


@dataclass(frozen=True)
class Glyph:
    handle: tuple[tuple[float, float], tuple[float, float]]
    head: tuple[tuple[float, float], ...]
    # Handle thickness, as a share of the disc's radius.
    handle_width: float = 0.2


# A felling axe: a straight haft from bottom left, and a blade on one side of its head —
# narrow where it grips the haft, flaring to a wide, slightly curved cutting edge.
AXE = Glyph(
    handle=((-0.60, 0.82), (0.38, -0.46)),
    head=((0.41, -0.47), (1.08, -0.24), (0.97, 0.12), (0.64, 0.40), (0.19, -0.19)),
)

# A miner's pick: the same haft, and a head crossing its end square — a curved bar,
# thickest over the haft and pointed at both ends, the points curving back towards the
# grip.
PICKAXE = Glyph(
    handle=((-0.60, 0.82), (0.25, -0.27)),
    # A crescent traced along the head: bowed away from the grip, tapering to both points.
    head=(
        (-0.45, -0.92),
        (-0.13, -0.87),
        (0.2, -0.76),
        (0.49, -0.59),
        (0.73, -0.36),
        (0.93, -0.07),
        (1.07, 0.22),
        (0.85, 0.04),
        (0.59, -0.16),
        (0.32, -0.37),
        (0.05, -0.56),
        (-0.22, -0.76),
    ),
)

ROLE_GLYPH = {
    SettlementRole.MINING: PICKAXE,
    SettlementRole.LUMBER: AXE,
}


@dataclass(frozen=True)
class Emblem:
    """A silhouette: filled polygons, drawn dark on a coloured disc."""

    shapes: tuple[tuple[tuple[float, float], ...], ...]


# A city: a church tower and spire between two houses, the right one set a little lower.
# Three separate shapes with a sliver between them, so the roofs still read as roofs
# when the whole thing is a dozen pixels across.
CITY = Emblem(
    shapes=(
        # The house on the left: walls and a pitched roof.
        ((-0.85, 0.75), (-0.85, 0.2), (-0.55, -0.15), (-0.25, 0.2), (-0.25, 0.75)),
        # The tower, and its spire.
        ((-0.18, 0.75), (-0.18, -0.35), (0.0, -1.0), (0.18, -0.35), (0.18, 0.75)),
        # The house on the right.
        ((0.25, 0.75), (0.25, 0.3), (0.55, 0.0), (0.85, 0.3), (0.85, 0.75)),
    )
)

# A city's disc: larger than a trade's, and gold, the colour cities had as a star.
CITY_RADIUS = 7.5
CITY_DISC = "#e8c547"
CITY_DISC_RGB = (232, 197, 71)
CITY_INK = "#2b2b2b"
CITY_INK_RGB = (43, 43, 43)

# Radius of the disc an icon sits in, in the same pixel units as the other settlement
# markers at scale 1, about a city's star. Larger than a plain village's 2.5: a pick
# drawn smaller than this reads as a smudge.
DISC_RADIUS = 6.5


def placed(glyph: Glyph, cx: float, cy: float, r: float):
    """The glyph's handle ends and head polygon in pixels, for a disc of radius *r* at (cx, cy).

    The icon fills 80% of the disc, so the disc's outline stays clear of it.
    """
    s = r * 0.8

    def at(p):
        return (cx + p[0] * s, cy + p[1] * s)

    return (
        (at(glyph.handle[0]), at(glyph.handle[1])),
        tuple(at(p) for p in glyph.head),
        glyph.handle_width * r,
    )


# Marks on a river, drawn in the river's own frame: x runs downstream, y across the
# water, and one unit is RIVER_MARK_SIZE pixels at scale 1. A renderer turns them to the
# river's bearing with `along`.
#
# A source is a spring: a ring with a dot, which needs no bearing.
RIVER_SOURCE_RING = 0.7
RIVER_SOURCE_DOT = 0.3
# An end is an arrowhead pointing where the water goes: out to sea, into a lake, off the
# map or into the ground.
RIVER_END_ARROW = ((1.1, 0.0), (-0.6, 0.85), (-0.2, 0.0), (-0.6, -0.85))
# White water — a cataract or rapids — is three bars laid across the river, white on a
# dark casing, the broken water a surveyor draws.
RAPIDS_BARS = (
    ((-0.6, -0.9), (-0.6, 0.9)),
    ((0.0, -0.9), (0.0, 0.9)),
    ((0.6, -0.9), (0.6, 0.9)),
)
RIVER_MARK_SIZE = 4.0
RAPIDS_INK = "#ffffff"
RAPIDS_CASING = "#1f3b57"


def along(points, cx: float, cy: float, size: float, bearing_deg: float):
    """*points* from a river mark's frame into pixels, at (cx, cy), *size* pixels to the
    unit, turned so x runs along *bearing_deg* (screen degrees, y down)."""
    a = math.radians(bearing_deg)
    ca, sa = math.cos(a), math.sin(a)
    return tuple((cx + (x * ca - y * sa) * size, cy + (x * sa + y * ca) * size) for x, y in points)


def placed_emblem(emblem: Emblem, cx: float, cy: float, r: float):
    """The emblem's polygons in pixels, for a disc of radius *r* at (cx, cy).

    Filling 80% of the disc, as `placed` does, so the two kinds of icon sit alike.
    """
    s = r * 0.8
    return tuple(tuple((cx + x * s, cy + y * s) for x, y in shape) for shape in emblem.shapes)


__all__ = [
    "AXE",
    "CITY",
    "CITY_DISC",
    "CITY_DISC_RGB",
    "CITY_INK",
    "CITY_INK_RGB",
    "CITY_RADIUS",
    "DISC_RADIUS",
    "PICKAXE",
    "RAPIDS_BARS",
    "RAPIDS_CASING",
    "RAPIDS_INK",
    "RIVER_END_ARROW",
    "RIVER_MARK_SIZE",
    "RIVER_SOURCE_DOT",
    "RIVER_SOURCE_RING",
    "ROLE_GLYPH",
    "Emblem",
    "Glyph",
    "along",
    "placed",
    "placed_emblem",
]
