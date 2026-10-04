class RoutingError(Exception):
    """A road connection the network requires has no legal route.

    Raised rather than silently degrading, where a route is required and the ground offers
    none: the world is geometrically broken and the seed is worth reproducing, so it fails
    loudly.
    """
