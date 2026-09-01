"""Format-independent legend model, shared by the SVG and PNG exporters.

Holds everything about a legend that does not depend on the output format: which rows
appear, in what order, under what label, and where the panel sits on the canvas.  Each
exporter draws the rows with its own primitives and its own colour representation, so
the two can gain a legend row or change placement without drifting apart.
"""

import math
from dataclasses import dataclass

from ..core.hex import SettlementTier
from ..core.world_state import RoadTier, WorldState

# Stand-in steps for the continuous greyscale used by color_mode="elevation".
ELEVATION_RAMP = (0.1, 0.3, 0.5, 0.7, 0.9)

# Settlement symbols are drawn at absolute sizes tuned for the default 12px hex, so a
# legend glyph box of this side reproduces them at exactly their on-map size.  Scaling a
# glyph by `box / SYMBOL_BOX` keeps the legend in proportion at other hex sizes.
SYMBOL_BOX = 12.0

CORNERS = ("top-right", "bottom-left")


@dataclass(frozen=True)
class LegendRow:
    """One legend entry.

    *kind* selects the glyph the exporter draws; *sample* carries whatever that glyph
    needs — a representative `Hex` for "fill" (so each exporter can run it through its own
    fill lookup), a `RoadTier` for "road", a `SettlementTier` for "settlement", and nothing
    for "ramp" or "river".
    """

    kind: str  # "fill" | "ramp" | "river" | "road" | "settlement"
    label: str
    sample: object = None


def validate(corner: str, scale: float) -> None:
    """Raise ValueError on legend settings an exporter cannot honour."""
    if corner not in CORNERS:
        raise ValueError(f"legend_corner must be one of {CORNERS}, got {corner!r}")
    if scale <= 0:
        raise ValueError(f"legend_scale must be positive, got {scale!r}")


def _enum_sort_key(member) -> tuple[str, int]:
    """Order categories by enum class name, then by declaration order."""
    cls = type(member)
    return (cls.__name__, list(cls).index(member))


def _fill_category(h, color_mode: str):
    """The enum member that decides a hex's fill — mirrors each exporter's `_get_hex_fill`."""
    if color_mode == "terrain":
        return h.terrain_class
    if color_mode == "land_cover":
        return h.land_cover
    if color_mode == "biome":
        return h.biome if h.biome is not None else h.terrain_class
    return None  # "elevation" is continuous, not categorical


def _label(member) -> str:
    return str(member.value).replace("_", " ").title()


def rows(ws: WorldState, color_mode: str, layers: set[str]) -> list[LegendRow]:
    """Legend rows for *ws*, covering only what the given layers actually draw."""
    out: list[LegendRow] = []

    if "terrain" in layers:
        if color_mode == "elevation":
            out.append(LegendRow("ramp", "Low → high elevation"))
        else:
            # One representative hex per category, so exporters can reuse their fill lookup.
            samples: dict = {}
            for hex_item in ws.hexes.values():
                category = _fill_category(hex_item, color_mode)
                if category is not None:
                    samples.setdefault(category, hex_item)
            for category in sorted(samples, key=_enum_sort_key):
                out.append(LegendRow("fill", _label(category), samples[category]))

    if "rivers" in layers and ws.rivers:
        out.append(LegendRow("river", "River"))

    if "roads" in layers and ws.roads:
        present = {road.tier for road in ws.roads}
        for tier in RoadTier:
            if tier in present:
                out.append(LegendRow("road", f"{_label(tier)} road", tier))

    if "settlements" in layers and ws.settlements:
        present_tiers = {s.tier for s in ws.settlements}
        for tier in SettlementTier:
            if tier in present_tiers:
                out.append(LegendRow("settlement", _label(tier), tier))

    return out


def placement(
    ws: WorldState,
    hex_size: float,
    padding: float,
    corner: str,
    ox: float,
    oy: float,
    canvas_w: float,
    canvas_h: float,
    panel_w: float,
    panel_h: float,
    margin: float,
    axial_to_pixel,
) -> tuple[float, float]:
    """Top-left corner for a panel of *panel_w* x *panel_h*, tucked into an empty corner.

    The axial-to-pixel transform shears the grid into a parallelogram, so a rectangular hex
    map leaves large empty triangles at the top-right and bottom-left of the canvas.  Every
    hex centre satisfies `d = py - px/sqrt(3)` within a fixed band, so the drawn area is
    bounded by two parallel diagonals.  The panel is placed flush against whichever diagonal
    bounds the chosen corner — not jammed into the canvas corner, which on a wide map can
    sit thousands of pixels from the nearest hex.

    A map with few columns has little shear and so little room; there the panel is clamped
    into the corner and the exporter's opaque backing keeps it readable over the map.
    """
    root3 = math.sqrt(3)
    offsets = [py - px / root3 for px, py in (axial_to_pixel(c, hex_size) for c in ws.hexes)]
    half_hex = root3 / 2 * hex_size

    if corner == "bottom-left":
        x = float(padding)
        # Binding corner is the panel's top-right: it must clear the lower diagonal.
        y = oy + (x + panel_w - ox) / root3 + max(offsets) + half_hex + margin
    else:  # "top-right"
        x = float(max(padding, canvas_w - padding - panel_w))
        # Binding corner is the panel's bottom-left: it must clear the upper diagonal.
        y = oy + (x - ox) / root3 + min(offsets) - half_hex - margin - panel_h

    x = min(max(x, padding), max(padding, canvas_w - padding - panel_w))
    y = min(max(y, padding), max(padding, canvas_h - padding - panel_h))
    return x, y
