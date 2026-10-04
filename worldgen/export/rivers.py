"""Shared river drawing model: how wide to draw each stretch of a river.

Format-independent, so the SVG and PNG exporters band a river identically and cannot
drift apart on what a given width means.
"""

from ..core.config import WorldConfig
from ..core.hex_grid import Corner, Side, side_joining
from ..core.world_state import River, RiverSide, WorldState


def major_sides(ws: WorldState) -> set[Side]:
    """The river sides carrying enough water to float a barge: the rivers that matter.

    The same line the campaign rules draw between a major and a minor river.  A minor
    river can be waded almost anywhere, so a ford on one marks nothing a reader needs;
    only a major river's fords are worth putting on the map.  The world's own config and
    runoff decide it, falling back to the defaults on a world that recorded neither.
    """
    cfg = WorldConfig.from_dict(ws.metadata.get("config") or {})
    runoff = ws.metadata.get("runoff_mm") or cfg.runoff_mm(cfg.mean_precip_mm)
    return {
        side
        for side, rs in ws.river_sides.items()
        if rs.catchment_km2 * runoff >= cfg.navigable_min_discharge
    }


def width_bands(
    river: River,
    river_sides: dict[Side, RiverSide],
    min_width: float,
    max_width: float,
    steps: int,
    exponent: float = 0.5,
) -> list[tuple[list[Corner], float]]:
    """Contiguous runs of *river* that share a drawn width, as `(corners, width)`.

    A river carries more water the further down it you go.  Drawing the whole of one at a
    single width — taken from `River.flow_volume`, which is measured at the mouth — makes
    a headwater trickle look exactly like the trunk it feeds.  Each side is sized instead
    by its own `RiverSide.flow`, so a river visibly grows downstream and a glance at the
    map ranks two rivers against each other.

    Width follows a curve over that flow rather than tracking it linearly (see
    *exponent* on `width_for_flow`); set *steps* to quantise it into discrete widths
    instead.

    Adjacent sides of equal width merge into one run — which is most of them under
    banding, and few under continuous scaling, where a river costs roughly one polyline
    per side.  Consecutive runs share their joining corner, so the drawn line stays
    continuous across a change of width.
    """
    path = river.corners
    if len(path) < 2:
        return []

    steps = max(0, int(steps))

    def side_width(a: Corner, b: Corner) -> float:
        rs = river_sides.get(side_joining(a, b))
        flow = rs.flow if rs is not None else 0.0
        if flow <= 0:
            # A side with no record of its own still belongs to a river that has a size.
            flow = river.flow_volume
        return width_for_flow(flow, min_width, max_width, steps, exponent)

    out: list[tuple[list[Corner], float]] = []
    run = [path[0]]
    current: float | None = None
    for a, b in zip(path, path[1:], strict=False):
        this = side_width(a, b)
        if current is None:
            current = this
        if this != current:
            out.append((run, current))
            run = [a]
            current = this
        run.append(b)
    if len(run) >= 2 and current is not None:
        out.append((run, current))
    return out


def width_for_flow(
    flow: float, min_width: float, max_width: float, steps: int, exponent: float = 0.5
) -> float:
    """Drawn width for a normalised *flow* in [0, 1].

    *flow* is raised to *exponent* before it is mapped onto the width range.  It has to
    be: `river_flow` is drainage accumulation divided by the basin maximum, and
    accumulation is roughly power-law distributed, so on a typical map the median river
    hex sits near 0.02 and the 90th percentile near 0.13.  Mapped linearly, that spends
    the entire width range on the last few hexes of the single largest trunk and draws
    everything else — nine tenths of the river network — as one indistinguishable
    hairline at `min_width`.  The default 0.5 is the square root, which is also what
    hydraulic geometry gives for channel width against discharge; 1.0 restores the raw
    linear mapping, and values below 0.5 widen the small streams further.

    With `steps == 0` the width then tracks that curve continuously, so a river tapers
    smoothly from headwater to mouth.  A positive *steps* quantises it into that many
    discrete widths instead, which merges neighbouring segments into far fewer polylines
    and gives the stepped look of a stream-order map; `steps == 1` draws every river at
    `max_width`.
    """
    flow = min(1.0, max(0.0, flow)) ** exponent
    if steps <= 0:
        return min_width + (max_width - min_width) * flow
    if steps == 1:
        return max_width
    band = min(steps - 1, max(0, int(flow * steps)))
    return min_width + (max_width - min_width) * band / (steps - 1)


def validate(min_width: float, max_width: float, steps: int, exponent: float = 0.5) -> None:
    """Raise ValueError on river width settings an exporter cannot honour.

    `steps == 0` is valid and means continuous scaling, not "no widths".
    """
    if steps < 0:
        raise ValueError(f"river_width_steps must be >= 0, got {steps!r}")
    if min_width <= 0:
        raise ValueError(f"river_min_width must be positive, got {min_width!r}")
    if max_width < min_width:
        raise ValueError(
            f"river_max_width must be >= river_min_width, got {max_width!r} < {min_width!r}"
        )
    if exponent <= 0:
        raise ValueError(f"river_width_exponent must be positive, got {exponent!r}")
