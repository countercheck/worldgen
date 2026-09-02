from dataclasses import dataclass, field
from enum import Enum

from .hex import Hex, HexCoord, Settlement


class RoadTier(Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TRACK = "track"


# Draw precedence where routes of different tiers share an edge: higher wins, and
# renderers paint in ascending order so a primary road is never overdrawn by a track.
ROAD_TIER_RANK = {RoadTier.TRACK: 0, RoadTier.SECONDARY: 1, RoadTier.PRIMARY: 2}


# Serialised schema version.  1.1 replaced the single "habitability" key with one score
# per settlement tier; a 1.0 file still loads, its lone value read into all three.  The
# bump exists so the schema cannot change shape under a fixed version string — an old
# reader handed a 1.1 file fails with a clear message instead of silently missing fields.
SCHEMA_VERSION = "1.1"
SUPPORTED_SCHEMA_VERSIONS = frozenset({"1.0", "1.1"})


@dataclass
class River:
    hexes: list[HexCoord]
    flow_volume: float


@dataclass
class Road:
    path: list[HexCoord]
    tier: RoadTier


@dataclass
class Ferry:
    """A boat link between two land hexes the road network cannot join by land.

    Roads may not run down a river channel (it would hide which bank they are on), so a
    component sealed off by a river mesh — a delta island, a braided confluence — is
    joined by water instead.  Both endpoints are land hexes carrying road ends; the
    crossing itself is drawn as a pair of anchorages rather than a line.
    """

    a: HexCoord
    b: HexCoord


@dataclass
class WorldState:
    seed: int
    width: int
    height: int
    hexes: dict[HexCoord, Hex] = field(default_factory=dict)
    rivers: list[River] = field(default_factory=list)
    settlements: list[Settlement] = field(default_factory=list)
    roads: list[Road] = field(default_factory=list)
    ferries: list[Ferry] = field(default_factory=list)
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
        """All non-water hexes."""
        from .hex import TerrainClass

        return [
            h
            for h in self.hexes.values()
            if h.terrain_class not in (TerrainClass.OCEAN, TerrainClass.LAKE)
        ]

    def all_ocean(self) -> list[Hex]:
        """All ocean hexes (map-edge-connected water bodies)."""
        from .hex import TerrainClass

        return [h for h in self.hexes.values() if h.terrain_class == TerrainClass.OCEAN]

    def all_lakes(self) -> list[Hex]:
        """All lake hexes (inland water bodies)."""
        from .hex import TerrainClass

        return [h for h in self.hexes.values() if h.terrain_class == TerrainClass.LAKE]

    def all_water(self) -> list[Hex]:
        """All water hexes (ocean and lakes)."""
        from .hex import TerrainClass

        return [
            h
            for h in self.hexes.values()
            if h.terrain_class in (TerrainClass.OCEAN, TerrainClass.LAKE)
        ]

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "version": SCHEMA_VERSION,
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
                    "habitability_city": h.habitability_city,
                    "habitability_town": h.habitability_town,
                    "habitability_village": h.habitability_village,
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
            "ferries": [{"a": list(f.a), "b": list(f.b)} for f in self.ferries],
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
        if version is not None and version not in SUPPORTED_SCHEMA_VERSIONS:
            supported = ", ".join(sorted(SUPPORTED_SCHEMA_VERSIONS))
            raise ValueError(f"Unsupported WorldState version '{version}'. Supported: {supported}.")

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
                # Files written before habitability was split per tier carry a single
                # "habitability"; read it into all three rather than rejecting them.
                habitability_city=hd.get("habitability_city", hd.get("habitability", 0.0)),
                habitability_town=hd.get("habitability_town", hd.get("habitability", 0.0)),
                habitability_village=hd.get("habitability_village", hd.get("habitability", 0.0)),
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
        ws.ferries = [Ferry(a=tuple(fd["a"]), b=tuple(fd["b"])) for fd in data.get("ferries", [])]

        return ws

    @classmethod
    def from_json(cls, path: str) -> "WorldState":
        from worldgen.export.json_export import load

        return load(path)
