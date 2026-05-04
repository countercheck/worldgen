from dataclasses import dataclass, field
from enum import Enum

from .hex import Hex, HexCoord, Settlement


class RoadTier(Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TRACK = "track"


@dataclass
class River:
    hexes: list[HexCoord]
    flow_volume: float


@dataclass
class Road:
    path: list[HexCoord]
    tier: RoadTier


@dataclass
class WorldState:
    seed: int
    width: int
    height: int
    hexes: dict[HexCoord, Hex] = field(default_factory=dict)
    rivers: list[River] = field(default_factory=list)
    settlements: list[Settlement] = field(default_factory=list)
    roads: list[Road] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @classmethod
    def empty(cls, seed: int, width: int, height: int) -> "WorldState":
        """Create an empty world state."""
        state = cls(seed=seed, width=width, height=height)
        for q in range(width):
            for r in range(height):
                state.hexes[(q, r)] = Hex(coord=(q, r))
        return state

    def get(self, coord: HexCoord) -> Hex | None:
        """Get hex at coordinate, or None if out of bounds."""
        return self.hexes.get(coord)

    def all_land(self) -> list[Hex]:
        """All non-ocean hexes."""
        from .hex import TerrainClass

        return [h for h in self.hexes.values() if h.terrain_class != TerrainClass.OCEAN]

    def all_water(self) -> list[Hex]:
        """All ocean hexes."""
        from .hex import TerrainClass

        return [h for h in self.hexes.values() if h.terrain_class == TerrainClass.OCEAN]
