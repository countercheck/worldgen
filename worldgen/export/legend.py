"""Format-independent legend model, shared by the SVG and PNG exporters.

Holds everything about a legend that does not depend on the output format: which rows
appear, in what order, under what label, and where the panel sits on the canvas.  Each
exporter draws the rows with its own primitives and its own colour representation, so
the two can gain a legend row or change placement without drifting apart.
"""

import math
from dataclasses import dataclass

from ..core.hex import (
    DEFAULT_TERRAIN_BANDS,
    SettlementRole,
    SettlementTier,
    terrain_bands,
    terrain_label,
)
from ..core.hex_grid import (
    corner_to_pixel,
    road_polylines,
    road_water_transitions,
    side_corners,
    side_joining,
)
from ..core.world_state import RoadTier, WorldState
from ..render.debug_viewer import is_fog
from ..render.glyphs import ROLE_GLYPH

# Stand-in steps for the continuous greyscale used by color_mode="elevation".
ELEVATION_RAMP = (0.1, 0.3, 0.5, 0.7, 0.9)

# Settlement symbols are drawn at absolute sizes tuned for the default 12px hex, so a
# legend glyph box of this side reproduces them at exactly their on-map size.  Scaling a
# glyph by `box / SYMBOL_BOX` keeps the legend in proportion at other hex sizes.
SYMBOL_BOX = 12.0

CORNERS = ("top-right", "bottom-left")


def stroke_scale(hex_size: float) -> float:
    """Multiplier taking a line width tuned for the reference hex to *hex_size*.

    Road and river widths are written for a 12px hex, the default — the same reference
    the symbols use.  Without this a map exported at `hex_size=30` draws hairline rivers
    between huge hexes, and one at `hex_size=6` drowns in roads.
    """
    return hex_size / SYMBOL_BOX


@dataclass(frozen=True)
class Metrics:
    """Panel geometry, derived from the row list and the exporter's text measurement.

    Computed before the canvas is sized so an exporter can grow a small canvas to fit the
    panel; a clamp alone cannot rescue a panel larger than the image it sits in.
    """

    glyph: float  # side of a row's square glyph box
    font: float
    row_h: float
    inner: float  # padding between the panel border and its content
    gap: float  # between a glyph and its label
    title_h: float
    width: float
    height: float


def metrics(hex_size: float, scale: float, n_rows: int, label_w: float, title_w: float) -> Metrics:
    """Panel geometry for *n_rows* rows whose widest label measures *label_w*."""
    g = hex_size * scale
    font = g * 0.8
    inner = g * 0.7
    gap = g * 0.6
    title_h = font * 2.0
    return Metrics(
        glyph=g,
        font=font,
        row_h=g * 1.5,
        inner=inner,
        gap=gap,
        title_h=title_h,
        width=max(inner * 2 + g + gap + label_w, inner * 2 + title_w),
        height=inner * 2 + title_h + n_rows * g * 1.5,
    )


@dataclass(frozen=True)
class LegendRow:
    """One legend entry.

    *kind* selects the glyph the exporter draws; *sample* carries whatever that glyph
    needs — a representative `Hex` for "fill" (so each exporter can run it through its own
    fill lookup), a `RoadTier` for "road", a `SettlementTier` for "settlement", a
    `SettlementRole` for "resource" (a mining or lumber village, drawn as its tool), and
    nothing for "ramp" or "river".
    """

    kind: str  # "fill" | "ramp" | "river" | "road" | "anchorage"
    #       | "ford" | "bridge" | "settlement" | "resource"
    #       | "river_source" | "river_end" | "rapids"
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


def _fill_category(h, color_mode: str, bands=DEFAULT_TERRAIN_BANDS):
    """The enum member that decides a hex's fill — mirrors each exporter's `_get_hex_fill`."""
    # Unseen ground on a partial map is not a terrain category. Its stored values are
    # defaults, so left to the branches below it would enter the key as whatever those
    # defaults happen to label — "Flat", on a map that is mostly unexplored.
    if is_fog(h):
        return None
    if color_mode == "terrain":
        return terrain_label(h, *bands)
    if color_mode == "land_cover":
        return h.land_cover
    if color_mode == "biome":
        return h.biome if h.biome is not None else terrain_label(h, *bands)
    if color_mode == "soil":
        return h.soil
    if color_mode == "land_use":
        return h.land_use
    return None  # "elevation" is continuous, not categorical


def _label(member) -> str:
    return str(member.value).replace("_", " ").title()


def anchorage_points(ws: WorldState) -> list:
    """Every land hex where a *drawn* route takes to the water, in stable order.

    Two sources, drawn with the same symbol because they mean the same thing to a
    reader: a road whose leg across an ocean or lake is not drawn, and a ferry standing
    in for a road where a river channel cuts the network in two.

    Shore points are filtered to hexes that a drawn road leg actually reaches.  A land
    leg of a single hex cannot be drawn as a polyline (`road_polylines` discards it), and
    marking its shore would leave an anchor sitting on the coast with no road attached to
    it.  Ferry landings are never filtered — the ferry is the connection, whether or not a
    road leg happens to be drawable at either end.
    """
    drawn = {c for _, leg in road_polylines(ws.road_edges, ws.hexes) for c in leg}
    points = {c for c in road_water_transitions(ws.sea_edges, ws.hexes) if c in drawn}
    points |= {c for ferry in ws.ferries for c in (ferry.a, ferry.b)}
    return sorted(points)


def _bearings(ws: WorldState, hex_size: float) -> dict:
    """The screen bearing, in degrees, of the river along each side it runs: downstream."""
    bearing: dict = {}
    for river in ws.rivers:
        for a, b in zip(river.corners, river.corners[1:], strict=False):
            (ax, ay), (bx, by) = corner_to_pixel(a, hex_size), corner_to_pixel(b, hex_size)
            bearing[side_joining(a, b)] = math.degrees(math.atan2(by - ay, bx - ax))
    return bearing


def _midpoint(side, hex_size: float) -> tuple[float, float]:
    (ax, ay), (bx, by) = (corner_to_pixel(c, hex_size) for c in side_corners(side))
    return (ax + bx) / 2.0, (ay + by) / 2.0


def river_marks(ws: WorldState, hex_size: float) -> list:
    """Where rivers rise, where they end and where they run white, as `(x, y, kind,
    bearing)` in stable order, in pixels before the exporter's offset. *kind* is "source",
    "end" or "rapids" (a cataract or rapids); *bearing* is the river's, downstream, in
    degrees.

    A river runs along hexsides, so a source or an end is a corner — read off the corner
    tags `HydrologyStage` sets, at the course's first or last corner — and white water is a
    side, marked at its middle from the tags `CataractStage` sets.  Every renderer marks the
    same places.
    """
    bearing = _bearings(ws, hex_size)
    out = []
    for river in ws.rivers:
        sides = river.sides()
        if not sides:
            continue
        first, last = river.corners[0], river.corners[-1]
        if "river_source" in ws.river_corners.get(first, ()):
            out.append((*corner_to_pixel(first, hex_size), "source", bearing[sides[0]]))
        if "river_end" in ws.river_corners.get(last, ()):
            out.append((*corner_to_pixel(last, hex_size), "end", bearing[sides[-1]]))
    for side, rs in ws.river_sides.items():
        if rs.tags & {"cataract", "rapids"}:
            out.append((*_midpoint(side, hex_size), "rapids", bearing.get(side, 0.0)))
    return sorted(out)


def crossings(ws: WorldState, hex_size: float) -> list:
    """Every ford and bridge, as `(x, y, kind, angle)` in stable order, at the middle of
    the river side it crosses, in pixels before the exporter's offset.

    *angle* is the bearing in degrees of the river along that side.  Both symbols are laid
    across the water rather than along it, so an exporter draws them rotated a further
    90°; a bridge aligned with the current would read as a second river.
    """
    bearing = _bearings(ws, hex_size)
    out = []
    for side, rs in ws.river_sides.items():
        kind = "bridge" if "bridge" in rs.tags else "ford" if "ford" in rs.tags else None
        if kind is not None:
            out.append((*_midpoint(side, hex_size), kind, bearing.get(side, 0.0)))
    return sorted(out)


def rows(ws: WorldState, color_mode: str, layers: set[str]) -> list[LegendRow]:
    """Legend rows for *ws*, covering only what the given layers actually draw."""
    out: list[LegendRow] = []

    if "terrain" in layers:
        if color_mode == "elevation":
            out.append(LegendRow("ramp", "Low → high elevation"))
        else:
            # One representative hex per category, so exporters can reuse their fill lookup.
            samples: dict = {}
            bands = terrain_bands(ws)
            for hex_item in ws.hexes.values():
                category = _fill_category(hex_item, color_mode, bands)
                if category is not None:
                    samples.setdefault(category, hex_item)
            for category in sorted(samples, key=_enum_sort_key):
                out.append(LegendRow("fill", _label(category), samples[category]))

            # Named last, because it is the absence of the categories above rather than
            # another one of them.
            fog = next((h for h in ws.hexes.values() if is_fog(h)), None)
            if fog is not None:
                out.append(LegendRow("fill", "Unseen", fog))

    if "rivers" in layers and ws.rivers:
        out.append(LegendRow("river", "River"))
        tagged = {t for tags in ws.river_corners.values() for t in tags}
        tagged |= {t for rs in ws.river_sides.values() for t in rs.tags}
        if "river_source" in tagged:
            out.append(LegendRow("river_source", "River source"))
        if "river_end" in tagged:
            out.append(LegendRow("river_end", "River end"))
        if tagged & {"cataract", "rapids"}:
            out.append(LegendRow("rapids", "Rapids"))

    if "roads" in layers and ws.road_edges:
        present = {edge.tier for edge in ws.road_edges.values()}
        for tier in RoadTier:
            if tier in present:
                out.append(LegendRow("road", f"{_label(tier)} road", tier))

    if "anchorages" in layers and anchorage_points(ws):
        out.append(LegendRow("anchorage", "Anchorage"))

    if "crossings" in layers:
        # Read straight off the side tags rather than via `crossings()`, which works out
        # positions and angles the legend does not use.
        tagged = {t for rs in ws.river_sides.values() for t in rs.tags}
        if "ford" in tagged:
            out.append(LegendRow("ford", "Ford"))
        if "bridge" in tagged:
            out.append(LegendRow("bridge", "Bridge"))

    if "settlements" in layers and ws.settlements:
        present_tiers = {s.tier for s in ws.settlements}
        for tier in SettlementTier:
            if tier in present_tiers:
                out.append(LegendRow("settlement", _label(tier), tier))
        present_roles = {s.role for s in ws.settlements if s.tier is SettlementTier.VILLAGE} & set(
            ROLE_GLYPH
        )
        for role, label in (
            (SettlementRole.MINING, "Mine"),
            (SettlementRole.LUMBER, "Lumber camp"),
        ):
            if role in present_roles:
                out.append(LegendRow("resource", label, role))

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
    # How far a hex's own polygon reaches past its centre along d, i.e. the support of a
    # flat-top hexagon in that direction.  Maximising `sin θ - cos θ / sqrt(3)` over the
    # vertices at 0°..300° gives 2/sqrt(3), not the sqrt(3)/2 half-height — using the
    # half-height under-reserves by 0.29 * hex_size and lets small panels clip terrain.
    hex_support = 2 / root3 * hex_size

    if corner == "bottom-left":
        x = float(padding)
        # Binding corner is the panel's top-right: it must clear the lower diagonal.
        y = oy + (x + panel_w - ox) / root3 + max(offsets) + hex_support + margin
    else:  # "top-right"
        x = float(max(padding, canvas_w - padding - panel_w))
        # Binding corner is the panel's bottom-left: it must clear the upper diagonal.
        y = oy + (x - ox) / root3 + min(offsets) - hex_support - margin - panel_h

    x = min(max(x, padding), max(padding, canvas_w - padding - panel_w))
    y = min(max(y, padding), max(padding, canvas_h - padding - panel_h))
    return x, y
