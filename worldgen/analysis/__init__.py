"""Measurements over a finished world.

A measuring instrument, not a generator: `analysis` reads a complete `WorldState` and
reports numbers about it.  It imports `core` and nothing else, and — the rule that matters
— **no stage may import it**.  A stage that could read its own report card would start
tuning itself against a metric, and the pipeline would stop being a straight line.

`render`, `export`, `cli` and the tests may import it freely.
"""

from .drainage import (
    DrainageMetrics,
    DrainageNetwork,
    bifurcation_ratio,
    build_network,
    drainage_metrics,
    first_order_link_ratio,
    links_by_order,
    river_azimuth_concentration,
    strahler_orders,
)

__all__ = [
    "DrainageMetrics",
    "DrainageNetwork",
    "bifurcation_ratio",
    "build_network",
    "drainage_metrics",
    "first_order_link_ratio",
    "links_by_order",
    "river_azimuth_concentration",
    "strahler_orders",
]
