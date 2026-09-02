class RoutingError(Exception):
    """A road connection the network requires has no legal route.

    Raised rather than silently degrading: roads may not run down a river channel, and
    where that seals a component off entirely the generator joins it by ferry.  If even a
    ferry is implausible — the gap is wider than `road_ferry_max_hop` — the world is
    geometrically broken and the seed is worth reproducing, so it fails loudly.
    """
