from dataclasses import dataclass, field
from enum import Enum

from .hex import Hex, HexCoord, Settlement
from .hex_grid import (
    AXIAL,
    GRID_LAYOUTS,
    Corner,
    Side,
    grid_coord,
    grid_index,
    side_hexes,
    side_joining,
)


class RoadTier(Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TRACK = "track"


# Draw precedence where routes of different tiers share an edge: higher wins, and
# renderers paint in ascending order so a primary road is never overdrawn by a track.
ROAD_TIER_RANK = {RoadTier.TRACK: 0, RoadTier.SECONDARY: 1, RoadTier.PRIMARY: 2}


# Serialised schema version.
#
# 2.0 moved rivers off the hexes and onto the sides between them: a river is a chain of
# corners, and what is true of a stretch of it — its catchment, its fall, a ford or a
# bridge — belongs to the side it runs along (`river_sides`), and what is true of a point
# — a source, a mouth, a confluence — to a corner (`river_corners`).  That changes what
# every reader of a world means by "on a river", so nothing older is loaded or migrated:
# a world is regenerated from its seed.
SCHEMA_VERSION = "2.0"
SUPPORTED_SCHEMA_VERSIONS = frozenset({"2.0"})


@dataclass
class River:
    """One course of the river network, from where it starts to where it ends.

    A course runs along hexsides: *corners* is its path from upstream to downstream, each
    corner one side from the next.  It starts at a spring, a lake's outlet or the map edge,
    and ends where its water leaves the network or on the corner where it joins a larger
    river, which carries on — so a tributary's last corner is a corner of its trunk.
    """

    corners: list[Corner]
    # Catchment at the last side, as a fraction of the largest on the map.
    flow_volume: float
    # Empty for a river too small to have been named.
    name: str = ""

    def sides(self) -> list[Side]:
        """The sides the course runs along, in order."""
        return [side_joining(a, b) for a, b in zip(self.corners, self.corners[1:], strict=False)]

    def banks(self) -> list[HexCoord]:
        """Every hex beside the course, in order downstream, each once."""
        out: list[HexCoord] = []
        seen: set[HexCoord] = set()
        for side in self.sides():
            for h in side_hexes(side):
                if h not in seen:
                    seen.add(h)
                    out.append(h)
        return out


@dataclass
class RiverSide:
    """One hexside a river runs along, and what is true of that stretch of it.

    `catchment_km2` is the area draining past it — the physical size of the river, and
    what fording, bridging and navigation are decided on.  `flow` is the same thing as a
    rank against the largest river on the map, for drawing.  `drop_m` is how far the
    river falls along the side.  `tags` hold what later stages put there: ford, bridge,
    cataract, rapids.
    """

    catchment_km2: float
    flow: float
    drop_m: float = 0.0
    tags: set[str] = field(default_factory=set)


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
    # Every hexside some river runs along, keyed by `hex_grid.side_of` names.
    river_sides: dict[Side, RiverSide] = field(default_factory=dict)
    # Tags on the corners of the river network: source, end, confluence, mouth.
    river_corners: dict[Corner, set[str]] = field(default_factory=dict)
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
            if h.terrain_class not in (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER)
        ]

    def all_open_water(self) -> list[Hex]:
        """Water that reaches the map edge, and so carries what enters it away."""
        from .hex import TerrainClass

        return [h for h in self.hexes.values() if h.terrain_class == TerrainClass.OPEN_WATER]

    def all_inland_water(self) -> list[Hex]:
        """Water with no way off the map: a basin, fresh or salt."""
        from .hex import TerrainClass

        return [h for h in self.hexes.values() if h.terrain_class == TerrainClass.INLAND_WATER]

    def all_water(self) -> list[Hex]:
        """All water hexes (ocean and lakes)."""
        from .hex import TerrainClass

        return [
            h
            for h in self.hexes.values()
            if h.terrain_class in (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER)
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
                    "alluvium": h.alluvium,
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
                {
                    "corners": [list(c) for c in r.corners],
                    "flow_volume": r.flow_volume,
                    "name": r.name,
                }
                for r in self.rivers
            ],
            "river_sides": [
                {
                    "side": list(side),
                    "catchment_km2": rs.catchment_km2,
                    "flow": rs.flow,
                    "drop_m": rs.drop_m,
                    "tags": sorted(rs.tags),
                }
                for side, rs in sorted(self.river_sides.items())
            ],
            "river_corners": [
                {"corner": list(corner), "tags": sorted(tags)}
                for corner, tags in sorted(self.river_corners.items())
            ],
            "settlements": [
                {
                    "coord": list(s.coord),
                    "tier": s.tier.value,
                    "role": s.role.value,
                    "population": s.population,
                    "name": s.name,
                    "culture": s.culture,
                    "etymology": s.etymology,
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
        if version not in SUPPORTED_SCHEMA_VERSIONS:
            supported = ", ".join(sorted(SUPPORTED_SCHEMA_VERSIONS))
            raise ValueError(
                f"Unsupported WorldState version '{version}'. Supported: {supported}. "
                "Worlds from before 2.0 put rivers on hexes rather than between them and "
                "cannot be converted; regenerate this one from its seed."
            )

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
                culture=sd["culture"],
                etymology=sd["etymology"],
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
                slope=hd["slope"],
                relief=hd["relief"],
                alluvium=hd["alluvium"],
                land_cover=LandCover(hd["land_cover"])
                if hd.get("land_cover") is not None
                else None,
                river_flow=hd["river_flow"],
                catchment_km2=hd["catchment_km2"],
                habitability_city=hd["habitability_city"],
                habitability_town=hd["habitability_town"],
                habitability_village=hd["habitability_village"],
                cultivated=hd["cultivated"],
                soil=SoilQuality(hd["soil"]) if hd.get("soil") else None,
                land_use=LandUse(hd["land_use"]) if hd.get("land_use") else None,
                rural_population=hd["rural_population"],
                territory=tuple(hd["territory"]) if hd.get("territory") is not None else None,
                territory_cost=hd["territory_cost"],
                tags=set(hd.get("tags", [])),
                road_connections={tuple(c) for c in hd.get("road_connections", [])},
            )
            h.settlement = settlement_by_coord.get(coord)
            ws.hexes[coord] = h

        ws.rivers = [
            River(
                corners=[tuple(c) for c in rd["corners"]],
                flow_volume=rd["flow_volume"],
                name=rd["name"],
            )
            for rd in data["rivers"]
        ]
        ws.river_sides = {
            tuple(sd["side"]): RiverSide(
                catchment_km2=sd["catchment_km2"],
                flow=sd["flow"],
                drop_m=sd["drop_m"],
                tags=set(sd["tags"]),
            )
            for sd in data["river_sides"]
        }
        ws.river_corners = {tuple(cd["corner"]): set(cd["tags"]) for cd in data["river_corners"]}

        def read_edges(rows):
            return {
                road_edge_key(tuple(ed["a"]), tuple(ed["b"])): RoadEdge(
                    RoadTier(ed["tier"]), ed["delta_elevation_m"]
                )
                for ed in rows
            }

        ws.sea_edges = read_edges(data["sea_edges"])
        ws.road_edges = read_edges(data["road_edges"])
        ws.ferries = [Ferry(a=tuple(fd["a"]), b=tuple(fd["b"])) for fd in data["ferries"]]

        return ws
