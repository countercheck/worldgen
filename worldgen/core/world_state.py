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
# added four keys, from two lines of work that landed together:
#
#   "layout"          how width/height map onto hex coordinates; a file predating offset
#                     grids loads as "axial".
#   "territory" and   which settlement works a hex, and what it costs that settlement to
#   "territory_cost"  reach it.
#   "catchment_km2"   the upstream area draining through a hex, kept alongside the
#                     normalised "river_flow" because that one is a rank and this a
#                     quantity.
#
# All four default when absent, so a 1.0 or 1.1 file still loads — and so does a 1.2 file
# written before the other half existed, since neither half is required.  The bump exists
# so the schema cannot change shape under a fixed version string: an old reader handed a
# newer file fails with a clear message instead of silently missing fields.
# 1.3 replaced "roads" — a list of whole journeys, each with its own path and tier — with
# "road_edges", one tier per undirected edge.  A 1.2 file still loads: its paths are walked
# into edges, the higher tier winning where journeys overlap, which is the rule the renderer
# already applied when drawing them.  The reverse is not true, so 1.3 is a real bump.
#
# 1.4 split "sea_edges" out of "road_edges": an edge with a foot in the water is a sea leg,
# not a road, and while the two were mixed there was no way to ask whether two places were
# joined *by land*.  A 1.3 file loads with no sea edges, which reads its water legs as
# roads — the shape it was written with.
#
# 1.5 put "delta_elevation_m" on each edge, signed in the direction of the key.  It is the
# quantity that decides how slow a segment is, and nothing reading a world should have to
# reconstruct the cost model to find it.  An older file loads with zeroes.
SCHEMA_VERSION = "1.6"
SUPPORTED_SCHEMA_VERSIONS = frozenset({"1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6"})


# Terrain classes before they were reframed as bands of gradient. "hill" and "mountain"
# named landforms and were partly decided on altitude; the bands that replaced them are
# named for slope and measured in metres per kilometre. The mapping is the closest
# equivalent, so a world saved earlier still loads.
_TERRAIN_ALIASES = {"hill": "rolling", "mountain": "steep"}


@dataclass
class River:
    hexes: list[HexCoord]
    flow_volume: float


@dataclass
class Road:
    """A drawable run of road: consecutive hexes, all of one tier.

    Not what the model stores.  The network lives in `WorldState.road_edges` as one tier
    per edge; this is what `road_polylines` hands a renderer, split at junctions, at tier
    changes and at water.  Building it is a view over the graph, so nothing needs to keep
    a list of these in sync with the edges they came from.
    """

    path: list[HexCoord]
    tier: RoadTier


@dataclass(frozen=True)
class RoadEdge:
    """One step of road or sea, with what it costs a traveller to be told rather than
    worked out again.

    `delta_elevation_m` is signed and read in the direction of the canonical key: positive
    means the road climbs from `a` to `b`, negative that it falls.  A consumer wanting the
    effort takes the absolute value, which is what the cost model does — a road is
    cut-and-fill and pays for the descent as well as the climb — but the sign is kept
    because a map that cannot say which way is uphill is missing something a reader wants.

    Derivable from the two hexes, and stored anyway: it is the quantity that decides how
    slow the segment is, and anything reading `world.json` should not have to reconstruct
    the cost model to find out.
    """

    tier: RoadTier
    delta_elevation_m: float = 0.0


def road_edge_key(a: HexCoord, b: HexCoord) -> tuple[HexCoord, HexCoord]:
    """The canonical key for the edge between two hexes.

    A road edge is undirected, so both endpoints order into one key and `(a, b)` and
    `(b, a)` cannot become two half-roads that disagree about their tier.
    """
    return (a, b) if a <= b else (b, a)


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
    # The road network, as one tier per undirected edge — keyed by `road_edge_key`.
    #
    # It used to be a list of `Road` objects, one per journey between a pair of
    # settlements, each holding the whole path end to end.  Those overlapped almost
    # completely: on a 128x128 map, 1,941 of them stored 322,730 hex entries covering
    # 3,645 distinct edges, so every edge was written about ninety times and the drawn
    # network existed only as a transient the renderer rebuilt each time.  Tier was worse
    # than redundant — it belonged to a whole journey, so one quiet hex demoted a trunk
    # route end to end, and a map came out 1,935 secondary against 6 primary.
    #
    # An edge is the thing a tier is actually a property of.  `road_polylines` walks this
    # for anything that needs lines to draw, and `hex.road_connections` stays as the
    # adjacency index into it.
    road_edges: dict[tuple[HexCoord, HexCoord], RoadEdge] = field(default_factory=dict)
    # The water legs of the same network, kept apart from the roads rather than mixed in.
    #
    # Routes cross open water because water is cheap to cross — rightly, since sea carriage
    # ran at a fraction of land carriage before the railway. But an edge with a foot in the
    # sea is not a road, and while both lived in `road_edges` the distinction could not be
    # drawn: half the network by hex count was water, "road coverage" counted the sea in,
    # and the map's single connected network was single only *through* the sea. By land
    # alone that map is forty networks tied together by eight crossings.
    #
    # Same shape as `road_edges`, so connectivity by land is the components of one and
    # connectivity by any means is the components of both.
    sea_edges: dict[tuple[HexCoord, HexCoord], RoadEdge] = field(default_factory=dict)
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
                    "land_cover": h.land_cover.value if h.land_cover is not None else None,
                    "soil": h.soil.value if h.soil is not None else None,
                    "land_use": h.land_use.value if h.land_use is not None else None,
                    "rural_population": h.rural_population,
                    "river_flow": h.river_flow,
                    "catchment_km2": h.catchment_km2,
                    "habitability_city": h.habitability_city,
                    "habitability_town": h.habitability_town,
                    "habitability_village": h.habitability_village,
                    "cultivated": h.cultivated,
                    "territory": list(h.territory) if h.territory is not None else None,
                    "territory_cost": h.territory_cost,
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
            "road_edges": [
                {
                    "a": list(a),
                    "b": list(b),
                    "tier": edge.tier.value,
                    "delta_elevation_m": edge.delta_elevation_m,
                }
                for (a, b), edge in sorted(self.road_edges.items())
            ],
            "sea_edges": [
                {
                    "a": list(a),
                    "b": list(b),
                    "tier": edge.tier.value,
                    "delta_elevation_m": edge.delta_elevation_m,
                }
                for (a, b), edge in sorted(self.sea_edges.items())
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
            LandUse,
            Settlement,
            SettlementRole,
            SettlementTier,
            SoilQuality,
            TerrainClass,
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
            # Terrain classes were renamed when they stopped meaning landform and started
            # meaning gradient. The stored strings carry over so older worlds still open.
            terrain = _TERRAIN_ALIASES.get(hd["terrain_class"], hd["terrain_class"])
            h = Hex(
                coord=coord,
                elevation=hd["elevation"],
                moisture=hd["moisture"],
                temperature=hd["temperature"],
                biome=Biome(hd["biome"]) if hd.get("biome") is not None else None,
                terrain_class=TerrainClass(terrain),
                land_cover=LandCover(hd["land_cover"])
                if hd.get("land_cover") is not None
                else None,
                river_flow=hd["river_flow"],
                catchment_km2=hd.get("catchment_km2", 0.0),
                # Files written before habitability was split per tier carry a single
                # "habitability"; read it into all three rather than rejecting them.
                habitability_city=hd.get("habitability_city", hd.get("habitability", 0.0)),
                habitability_town=hd.get("habitability_town", hd.get("habitability", 0.0)),
                habitability_village=hd.get("habitability_village", hd.get("habitability", 0.0)),
                cultivated=hd["cultivated"],
                # Added at schema 1.6. Absent in anything older, and a world written before
                # soil existed has no answer to give — reading one back leaves these None
                # rather than inventing a class the generator never assigned.
                soil=SoilQuality(hd["soil"]) if hd.get("soil") else None,
                land_use=LandUse(hd["land_use"]) if hd.get("land_use") else None,
                rural_population=hd.get("rural_population", 0.0),
                # Absent before 1.2: such a file simply records no catchments.
                territory=tuple(hd["territory"]) if hd.get("territory") is not None else None,
                territory_cost=hd.get("territory_cost", 0.0),
                tags=set(hd.get("tags", [])),
                road_connections={tuple(c) for c in hd.get("road_connections", [])},
            )
            h.settlement = settlement_by_coord.get(coord)
            ws.hexes[coord] = h

        ws.rivers = [
            River(hexes=[tuple(c) for c in rd["hexes"]], flow_volume=rd["flow_volume"])
            for rd in data.get("rivers", [])
        ]

        def read_edges(rows):
            return {
                road_edge_key(tuple(ed["a"]), tuple(ed["b"])): RoadEdge(
                    RoadTier(ed["tier"]), ed.get("delta_elevation_m", 0.0)
                )
                for ed in rows
            }

        ws.sea_edges = read_edges(data.get("sea_edges", []))
        if "road_edges" in data:
            ws.road_edges = read_edges(data["road_edges"])
        else:
            # Schema 1.2 and earlier stored whole journeys, which overlap. Where two of
            # them share an edge the higher tier wins, which is the rule the renderer
            # applied at draw time anyway — so an old file loads as the network it drew.
            for rd in data.get("roads", []):
                tier = RoadTier(rd["tier"])
                path = [tuple(c) for c in rd["path"]]
                for a, b in zip(path, path[1:], strict=False):
                    key = road_edge_key(a, b)
                    have = ws.road_edges.get(key)
                    if have is None or ROAD_TIER_RANK[tier] > ROAD_TIER_RANK[have.tier]:
                        ws.road_edges[key] = RoadEdge(tier)
        ws.ferries = [Ferry(a=tuple(fd["a"]), b=tuple(fd["b"])) for fd in data.get("ferries", [])]

        return ws

    @classmethod
    def from_json(cls, path: str) -> "WorldState":
        from worldgen.export.json_export import load

        return load(path)
