"""What the browser can look at: the finished map drawn several ways, and one hex in detail.

A view is either an export style (the maps a user would hand out) or a debug plate (one
attribute over the whole grid). The two renderers place the grid at different origins and
sizes, so turning a click back into a hex has to know which one drew the picture; both are
recorded here, beside the call that draws with them.
"""

import io
from dataclasses import dataclass, fields
from enum import Enum

from ..core.hex import Hex
from ..core.hex_grid import axial_to_pixel, pixel_to_axial
from ..core.world_state import WorldState
from ..export import png_export, svg_export
from ..render import debug_viewer

EXPORT_STYLES = ("atlas", "topographic", "wargame")
_DEBUG_HEX_SIZE = 20.0


@dataclass(frozen=True)
class _Frame:
    """Where a renderer put hex (0, 0): its hex size and the padding round the grid."""

    hex_size: float
    padding: float


def views_for(model: str) -> list[str]:
    layers = list(debug_viewer.DEBUG_LAYERS)
    if model == "organic":
        layers += debug_viewer.HAULAGE_LAYERS
    return [*EXPORT_STYLES, *layers]


def resolve(model: str, requested: str) -> str:
    """The view named *requested*, as this module spells it, or KeyError.

    What comes back is the module's own string, never the caller's, so a view name that
    reaches a response header or a filename is one the server wrote, not one a request did.
    """
    for view in views_for(model):
        if view == requested:
            return view
    raise KeyError(requested)


def _frame(view: str) -> _Frame:
    if view in EXPORT_STYLES:
        default = svg_export.SVGConfig()
        return _Frame(default.hex_size, default.padding)
    # The debug viewer pads by one hex and nothing more.
    return _Frame(_DEBUG_HEX_SIZE, 0.0)


def render_svg(state: WorldState, view: str) -> str:
    if view in EXPORT_STYLES:
        return svg_export.render(state, svg_export.SVGConfig(style=view))
    if view in views_for(state.metadata.get("config", {}).get("model", "organic")):
        return debug_viewer.render_svg(state, view, hex_size=_DEBUG_HEX_SIZE)
    raise KeyError(view)


def render_png(state: WorldState, style: str) -> bytes:
    if style not in EXPORT_STYLES:
        raise KeyError(style)
    image = png_export.render(state, png_export.PNGConfig(style=style))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _origin(state: WorldState, frame: _Frame) -> tuple[float, float]:
    """Where the renderer drawing *frame* put axial (0, 0).

    Both renderers shift the grid so its top-left hex sits one hex size plus the padding
    in from the corner.
    """
    size = frame.hex_size
    pixels = [axial_to_pixel(coord, size) for coord in state.hexes]
    ox = -(min(p[0] for p in pixels) - size) + frame.padding
    oy = -(min(p[1] for p in pixels) - size) + frame.padding
    return ox, oy


def hex_at(state: WorldState, view: str, x: float, y: float) -> Hex | None:
    """The hex under point (x, y) of *view*'s picture, or None off the grid."""
    if not state.hexes:
        return None
    frame = _frame(view)
    ox, oy = _origin(state, frame)
    return state.hexes.get(pixel_to_axial(x - ox, y - oy, frame.hex_size))


def hex_outline(state: WorldState, view: str, h: Hex) -> dict:
    """Where *h* sits in *view*'s picture: its centre and size, to draw a highlight."""
    frame = _frame(view)
    ox, oy = _origin(state, frame)
    px, py = axial_to_pixel(h.coord, frame.hex_size)
    return {"x": px + ox, "y": py + oy, "size": frame.hex_size}


def _plain(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, set | frozenset):
        return sorted(_plain(v) for v in value)
    if isinstance(value, tuple | list):
        return [_plain(v) for v in value]
    return value


def describe(state: WorldState, h: Hex) -> dict:
    """Everything a hex carries, plus the names that make it readable: its settlement,
    the rivers through it, and the settlement whose territory it is."""
    out = {}
    for f in fields(h):
        if f.name in ("settlement", "road_connections"):
            continue
        out[f.name] = _plain(getattr(h, f.name))
    out["roads"] = len(h.road_connections)
    if h.settlement is not None:
        s = h.settlement
        out["settlement"] = {
            "name": s.name,
            "tier": s.tier.value,
            "role": s.role.value,
            "population": s.population,
            "culture": s.culture,
            "etymology": s.etymology,
        }
    out["rivers"] = [r.name or "(unnamed)" for r in state.rivers if h.coord in r.banks()]
    if h.territory is not None:
        owner = state.hexes.get(h.territory)
        if owner is not None and owner.settlement is not None:
            out["territory_of"] = owner.settlement.name
    return out
