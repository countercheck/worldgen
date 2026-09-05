from dataclasses import dataclass, field
from enum import Enum

from .hex import Hex, HexCoord, Settlement
from .hex_grid import AXIAL, GRID_LAYOUTS, grid_coord, grid_index


class RoadTier(Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TRACK = "track"


# Draw precedence where routes of different tiers share an edge: higher wins, and
# renderers paint in ascending order so a primary road is never overdrawn by a track.
ROAD_TIER_RANK = {RoadTier.TRACK: 0, RoadTier.SECONDARY: 1, RoadTier.PRIMARY: 2}


# Serialised schema version.  1.1 replaced the single "habitability" key with one score
# per settlement tier; a 1.0 file still loads, its lone value read into all three.  1.2
# added "layout", which says how width/height map onto hex coordinates; an earlier file
# predates offset grids, so it loads as "axial".  The bump exists so the schema cannot
# change shape under a fixed version string — an old reader handed a 1.2 file fails with
# a clear message instead of silently missing fields.  1.3 records "slope" and "relief"
# and stops recording a steepness class: "flat", "hill" and "mountain" were thresholds on
# the slope now stored, and an earlier file's band reads back as plain land, with the
# slope recomputed by whoever needs it.
SCHEMA_VERSION = "1.3"
SUPPORTED_SCHEMA_VERSIONS = frozenset({"1.0", "1.1", "1.2", "1.3"})

# Steepness bands a pre-1.3 file may carry in "terrain_class".  Land is land; how steep it
# was is a number that file never wrote down, so it comes back as 0.0 rather than being
# guessed from the band it fell in.
_LEGACY_TERRAIN_BANDS = frozenset({"flat", "hill", "mountain"})


def _terrain_class_from(value: str):
    """Read a hex's class, translating the steepness bands a pre-1.3 file may carry."""
    from .hex import TerrainClass

    if value in _LEGACY_TERRAIN_BANDS:
        return TerrainClass.LAND
    return TerrainClass(value)


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
    # How `width` x `height` maps onto hex coordinates — "axial" (a rhombus, drawn as a
    # leaning parallelogram) or "offset" (a rectangle with ragged north and south edges).
    # See `core.hex_grid`; hexes are keyed by axial coordinates either way.
    layout: str = AXIAL
    hexes: dict[HexCoord, Hex] = field(default_factory=dict)
    rivers: list[River] = field(default_factory=list)
    settlements: list[Settlement] = field(default_factory=list)
    roads: list[Road] = field(default_factory=list)
    ferries: list[Ferry] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @classmethod
    def empty(cls, seed: int, width: int, height: int, layout: str = AXIAL) -> "WorldState":
        """Create an empty world state of `width` x `height` hexes in the given layout."""
        state = cls(seed=seed, width=width, height=height, layout=layout)
        for col in range(width):
            for row in range(height):
                coord = state.coord_at(col, row)
                state.hexes[coord] = Hex(coord=coord)
        return state

    def get(self, coord: HexCoord) -> Hex | None:
        """Get hex at coordinate, or None if out of bounds."""
        return self.hexes.get(coord)

    def coord_at(self, col: int, row: int) -> HexCoord:
        """The hex coordinate at grid column *col*, row *row*.

        Stages that work on a (width, height) array — the noise, erosion and climate
        fields — index it by column and row and come back through here for the hex,
        which is what keeps them layout-agnostic.
        """
        return grid_coord(self.layout, col, row)

    def grid_index(self, coord: HexCoord) -> tuple[int, int]:
        """The grid column and row of *coord* — the inverse of `coord_at`."""
        return grid_index(self.layout, coord)

    def on_border(self, coord: HexCoord) -> bool:
        """True for hexes on the outermost ring of the grid, which drain off the map."""
        col, row = self.grid_index(coord)
        return col == 0 or col == self.width - 1 or row == 0 or row == self.height - 1

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
            "layout": self.layout,
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
                    "slope": h.slope,
                    "relief": h.relief,
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
        )

        version = data.get("version")
        if version is not None and version not in SUPPORTED_SCHEMA_VERSIONS:
            supported = ", ".join(sorted(SUPPORTED_SCHEMA_VERSIONS))
            raise ValueError(f"Unsupported WorldState version '{version}'. Supported: {supported}.")

        layout = data.get("layout", AXIAL)
        if layout not in GRID_LAYOUTS:
            supported = ", ".join(GRID_LAYOUTS)
            raise ValueError(f"Unknown WorldState layout '{layout}'. Supported: {supported}.")

        ws = cls(
            seed=data["seed"],
            width=data["width"],
            height=data["height"],
            layout=layout,
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
                terrain_class=_terrain_class_from(hd["terrain_class"]),
                slope=hd.get("slope", 0.0),
                relief=hd.get("relief", 0.0),
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
