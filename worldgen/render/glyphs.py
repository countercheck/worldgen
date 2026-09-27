"""Icons for the settlements that exist for a trade: a pickaxe for a mine, an axe for a camp.

Drawn as coordinates in a unit square — x right, y down, -1 to 1 either way — so the SVG
exporter, the PNG exporter and the debug viewer all draw one shape and cannot come to
disagree about it. Each icon is a handle, drawn as a thick line, and a head, drawn as a
filled polygon; the exporters scale both into a white disc the size of a settlement marker.
"""

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


__all__ = ["AXE", "DISC_RADIUS", "PICKAXE", "ROLE_GLYPH", "Glyph", "placed"]
