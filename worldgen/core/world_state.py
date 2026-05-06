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

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "version": "1.0",
            "seed": self.seed,
            "width": self.width,
            "height": self.height,
            "metadata": self.metadata,
            "hexes": [
                {
                    "q": h.coord[0],
                    "r": h.coord[1],
                    "elevation": h.elevation,
                    "moisture": h.moisture,
                    "temperature": h.temperature,
                    "biome": h.biome.value if h.biome is not None else None,
                    "terrain_class": h.terrain_class.value,
                    "land_cover": h.land_cover.value if h.land_cover is not None else None,
                    "river_flow": h.river_flow,
                    "habitability": h.habitability,
                    "cultivated": h.cultivated,
                    "tags": sorted(h.tags),
                    "road_connections": sorted([list(c) for c in h.road_connections]),
                }
                for h in self.hexes.values()
            ],
            "rivers": [
                {"hexes": [list(c) for c in r.hexes], "flow_volume": r.flow_volume}
                for r in self.rivers
            ],
            "settlements": [
                {
                    "coord": list(s.coord),
                    "tier": s.tier.value,
                    "role": s.role.value,
                    "population": s.population,
                    "name": s.name,
                }
                for s in self.settlements
            ],
            "roads": [
                {"path": [list(c) for c in r.path], "tier": r.tier.value} for r in self.roads
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WorldState":
        """Reconstruct WorldState from a dict produced by to_dict()."""
        from .hex import (
            Biome,
            Hex,
            LandCover,
            Settlement,
            SettlementRole,
            SettlementTier,
            TerrainClass,
        )

        version = data.get("version")
        if version is not None and version != "1.0":
            raise ValueError(
                f"Unsupported WorldState version '{version}'. Expected '1.0'."
            )

        ws = cls(
            seed=data["seed"],
            width=data["width"],
            height=data["height"],
            metadata=data.get("metadata", {}),
        )

        settlements = [
            Settlement(
                coord=tuple(sd["coord"]),
                tier=SettlementTier(sd["tier"]),
                role=SettlementRole(sd["role"]),
                population=sd["population"],
                name=sd["name"],
            )
            for sd in data.get("settlements", [])
        ]
        ws.settlements = settlements
        settlement_by_coord = {s.coord: s for s in settlements}

        for hd in data.get("hexes", []):
            coord = (hd["q"], hd["r"])
            h = Hex(
                coord=coord,
                elevation=hd["elevation"],
                moisture=hd["moisture"],
                temperature=hd["temperature"],
                biome=Biome(hd["biome"]) if hd.get("biome") is not None else None,
                terrain_class=TerrainClass(hd["terrain_class"]),
                land_cover=LandCover(hd["land_cover"])
                if hd.get("land_cover") is not None
                else None,
                river_flow=hd["river_flow"],
                habitability=hd["habitability"],
                cultivated=hd["cultivated"],
                tags=set(hd.get("tags", [])),
                road_connections={tuple(c) for c in hd.get("road_connections", [])},
            )
            h.settlement = settlement_by_coord.get(coord)
            ws.hexes[coord] = h

        ws.rivers = [
            River(hexes=[tuple(c) for c in rd["hexes"]], flow_volume=rd["flow_volume"])
            for rd in data.get("rivers", [])
        ]
        ws.roads = [
            Road(path=[tuple(c) for c in rd["path"]], tier=RoadTier(rd["tier"]))
            for rd in data.get("roads", [])
        ]

        return ws

    @classmethod
    def from_json(cls, path: str) -> "WorldState":
        from worldgen.export.json_export import load

        return load(path)
