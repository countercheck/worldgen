"""Where the names go on a map.

Format-independent, like `rivers.width_bands`: the SVG and PNG exporters place labels by
the same rules and differ only in how they measure and draw text, so a name cannot sit
above its town in one and beside it in the other.

Three rules, all from how a printed atlas is set:

**Size says rank.** A city's name is set larger and bolder than a town's, a town's than a
village's, so a reader finds the big places first without reading anything.

**Rivers are italic and follow the water.** A river's name is set along its middle reach,
turned to the river's heading, in the river's colour — the convention that tells a river
name from a place name at a glance.

**Nothing overlaps.** Each label tries four places round its town (above, below, right,
left) and takes the first one clear of every marker and every label already set; if none
is clear it is left off. Labels are set in order of rank, so a crowded map drops village
names before it drops a city's.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

from ..core.hex import HexCoord, Settlement, SettlementTier, TerrainClass
from ..core.hex_grid import Corner, corner_to_pixel
from ..core.world_state import River, WorldState

# Font size as a fraction of the hex size.
_TIER_SCALE = {SettlementTier.CITY: 0.95, SettlementTier.TOWN: 0.75, SettlementTier.VILLAGE: 0.58}
_RIVER_SCALE = 0.62
_TIER_ORDER = {SettlementTier.CITY: 0, SettlementTier.TOWN: 1, SettlementTier.VILLAGE: 3}
_RIVER_ORDER = 2
# How far from a settlement's centre its label stands, and how much room its marker takes,
# as fractions of the hex size.
_GAP = 0.55
_MARKER = 0.45
_WATER = (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER)

Box = tuple[float, float, float, float]
# (text, font size, bold) -> (width, height) in pixels.
Measure = Callable[[str, float, bool], tuple[float, float]]


@dataclass(frozen=True)
class PlacedLabel:
    text: str
    # Centre of the text, in canvas pixels.
    x: float
    y: float
    size: float
    bold: bool
    italic: bool
    # Degrees clockwise from horizontal; only rivers turn.
    angle: float
    river: bool


def estimate_width(text: str, size: float, bold: bool) -> tuple[float, float]:
    """A width for text that cannot be measured, as in an SVG the browser lays out later.

    Average glyph advance in a sans-serif face is a little over half the font size; bold
    runs about a tenth wider.
    """
    return len(text) * size * (0.62 if bold else 0.56), size


def _rotated_box(cx: float, cy: float, w: float, h: float, angle: float) -> Box:
    """The axis-aligned box round a *w* x *h* rectangle centred on (cx, cy), turned."""
    a = math.radians(angle)
    bw = abs(w * math.cos(a)) + abs(h * math.sin(a))
    bh = abs(w * math.sin(a)) + abs(h * math.cos(a))
    return (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)


def _clear(box: Box, taken: list[Box]) -> bool:
    x0, y0, x1, y1 = box
    return all(x1 <= a0 or x0 >= a1 or y1 <= b0 or y0 >= b1 for a0, b0, a1, b1 in taken)


def _river_reach(ws: WorldState, river) -> list[Corner]:
    """The corners the river runs through: its course, which never crosses water."""
    return list(river.corners)


def place_labels(
    ws: WorldState,
    to_pixel: Callable[[HexCoord], tuple[float, float]],
    hex_size: float,
    measure: Measure = estimate_width,
    settlements: bool = True,
    rivers: bool = True,
) -> list[PlacedLabel]:
    """Every label that fits, positioned, in the order they should be drawn.

    *to_pixel* maps a hex to its centre on the canvas, offsets included. *settlements*
    says whether settlement markers are drawn — they are obstacles if so — and *rivers*
    whether rivers are, since a river's name means nothing without the river under it.
    """
    taken: list[Box] = []
    if settlements:
        m = _MARKER * hex_size
        for s in ws.settlements:
            x, y = to_pixel(s.coord)
            taken.append((x - m, y - m, x + m, y + m))

    jobs: list[tuple[tuple[int, int, HexCoord], str, Settlement | River]] = []
    for s in ws.settlements:
        jobs.append(((_TIER_ORDER[s.tier], -s.population, s.coord), "settlement", s))
    if rivers:
        for river in ws.rivers:
            reach = _river_reach(ws, river)
            if river.name and len(reach) >= 3:
                jobs.append(((_RIVER_ORDER, -len(reach), reach[0]), "river", river))
    jobs.sort(key=lambda j: j[0])

    placed: list[PlacedLabel] = []
    for _, kind, item in jobs:
        label = (
            _place_settlement(item, to_pixel, hex_size, measure, taken)
            if kind == "settlement"
            else _place_river(ws, item, to_pixel, hex_size, measure, taken)
        )
        if label is not None:
            placed.append(label)
    return placed


def _place_settlement(s, to_pixel, hex_size, measure, taken) -> PlacedLabel | None:
    size = _TIER_SCALE[s.tier] * hex_size
    bold = s.tier is SettlementTier.CITY
    w, h = measure(s.name, size, bold)
    x, y = to_pixel(s.coord)
    gap = _GAP * hex_size
    for cx, cy in (
        (x, y - gap - h / 2),
        (x, y + gap + h / 2),
        (x + gap + w / 2, y),
        (x - gap - w / 2, y),
    ):
        box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        if _clear(box, taken):
            taken.append(box)
            return PlacedLabel(s.name, cx, cy, size, bold, False, 0.0, False)
    return None


def _place_river(ws, river, to_pixel, hex_size, measure, taken) -> PlacedLabel | None:
    size = _RIVER_SCALE * hex_size
    w, h = measure(river.name, size, False)
    reach = _river_reach(ws, river)
    n = len(reach)
    # Corners are placed from the same origin `to_pixel` puts the hexes on.
    ox, oy = to_pixel((0, 0))

    def at(corner: Corner) -> tuple[float, float]:
        px, py = corner_to_pixel(corner, hex_size)
        return px + ox, py + oy

    # The middle of the reach first, then either third: the middle is where a reader's
    # eye expects it, and the thirds are where there is often more room.  Each angle is
    # read across two sides either way, which smooths the zigzag of a hexside course.
    for i in dict.fromkeys((n // 2, n // 3, (2 * n) // 3)):
        ax, ay = at(reach[max(0, i - 2)])
        bx, by = at(reach[min(n - 1, i + 2)])
        angle = math.degrees(math.atan2(by - ay, bx - ax))
        # Never upside down: a name reads left to right whichever way the water runs.
        if angle > 90:
            angle -= 180
        elif angle <= -90:
            angle += 180
        x, y = at(reach[i])
        # Beside the line rather than on it, on either bank.
        nx, ny = -math.sin(math.radians(angle)), math.cos(math.radians(angle))
        off = h * 0.5 + hex_size * 0.2
        for side in (-1, 1):
            cx, cy = x + side * nx * off, y + side * ny * off
            box = _rotated_box(cx, cy, w, h, angle)
            if _clear(box, taken):
                taken.append(box)
                return PlacedLabel(river.name, cx, cy, size, False, True, angle, True)
    return None


__all__ = ["PlacedLabel", "estimate_width", "place_labels"]
