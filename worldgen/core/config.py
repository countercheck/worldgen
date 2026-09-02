import json
from dataclasses import asdict, dataclass, fields
from typing import Any

from .hex_grid import GRID_LAYOUTS


@dataclass(frozen=True)
class ClimateContext:
    """The climate of the region as a whole.

    `base_temperature` and `moisture_target` set where the region sits on the temperature
    and moisture axes; `palette` is the set of biomes that can occur there.  The palette
    is what keeps a region coherent: an arid region varies from desert to steppe to
    alpine with altitude, but never produces jungle three valleys over.
    """

    base_temperature: float
    moisture_target: float
    palette: frozenset


def _palette(*names: str) -> frozenset:
    from .hex import Biome

    return frozenset(getattr(Biome, n) for n in names)


# Biomes every region can produce regardless of climate, because they are made by
# terrain rather than by climate: bare peaks, and waterlogged ground beside rivers.
_ALWAYS = ("ALPINE", "WETLAND", "OCEAN")

CLIMATE_CONTEXTS: dict[str, ClimateContext] = {
    "boreal": ClimateContext(0.22, 0.55, _palette("TUNDRA", "BOREAL", "GRASSLAND", *_ALWAYS)),
    "temperate": ClimateContext(
        0.50, 0.55, _palette("TEMPERATE_FOREST", "GRASSLAND", "BOREAL", "SHRUBLAND", *_ALWAYS)
    ),
    "mediterranean": ClimateContext(
        0.62, 0.35, _palette("SHRUBLAND", "GRASSLAND", "TEMPERATE_FOREST", *_ALWAYS)
    ),
    "arid": ClimateContext(0.68, 0.15, _palette("DESERT", "SHRUBLAND", "GRASSLAND", *_ALWAYS)),
    "tropical": ClimateContext(
        0.85, 0.75, _palette("TROPICAL", "GRASSLAND", "SHRUBLAND", *_ALWAYS)
    ),
}


@dataclass
class WorldConfig:
    """All tunable parameters for world generation."""

    width: int = 128
    height: int = 128
    # How width x height are laid out on the hex grid:
    #   "axial"   q in [0, width), r in [0, height).  The flat-top pixel transform
    #             shears that rhombus, so the drawn map is a leaning parallelogram.
    #   "offset"  odd-q offset column/row.  The drawn map is a rectangle; odd columns
    #             sit half a hex lower than even ones, so the north and south edges are
    #             ragged.  A square needs height about 0.87 * width, since hex columns
    #             are spaced 1.5 hex-sizes apart and rows sqrt(3).
    grid_layout: str = "axial"
    sea_level: float = 0.25

    # Elevation
    noise_octaves: int = 6
    noise_persistence: float = 0.5
    noise_lacunarity: float = 2.0
    noise_scale: float = 3.0
    domain_warp_strength: float = 0.3
    continent_falloff: bool = True
    continent_shelf_hexes: int = 10
    # Which map edges the sea comes in from.  An edge left out of this list gets no
    # shelf, so the land simply runs off the map there — the world continues past the
    # border rather than ending in a coast.
    continent_falloff_edges: tuple[str, ...] = ("north", "south", "east", "west")
    continent_shelf_variance: float = 0.35
    continent_seabed: float = 0.15
    elevation_gradient: tuple[float, float] = (0.0, 0.0)

    # Terrain classification
    terrain_hill_gradient: float = 0.02
    terrain_mountain_gradient: float = 0.04

    # Erosion
    erosion_iterations: int = 15000
    erosion_inertia: float = 0.05
    erosion_capacity: float = 4.0
    erosion_deposition: float = 0.3
    erosion_erosion_rate: float = 0.3
    erosion_channel_affinity_gain: float = 0.5
    erosion_affinity_update_interval: int = 500
    erosion_delta_min_load: float = 0.15

    # Hydrology
    river_flow_threshold: float = 0.05
    river_flow_continuous: bool = False  # True: river_flow on all draining land hexes
    moisture_bleed_passes: int = 0  # 0 = flat river bonus (default); >0 = elevation-gated bleed
    moisture_bleed_strength: float = 0.3

    # Lake drainage
    lake_chaining: bool = True  # Let a lake spill into a strictly lower lake, not only the sea
    endorheic_marsh_radius: int = 1  # Shore band (hexes) turned to wetland around a closed basin
    endorheic_marsh_min_moisture: float = 0.40  # Below this a closed basin is arid, not marshy

    def __post_init__(self) -> None:
        if self.grid_layout not in GRID_LAYOUTS:
            raise ValueError(
                f"unknown grid_layout {self.grid_layout!r}; choose from {', '.join(GRID_LAYOUTS)}"
            )
        self.wind_direction = _coerce_pair("wind_direction", self.wind_direction)
        self.elevation_gradient = _coerce_pair("elevation_gradient", self.elevation_gradient)
        if not (0.0 <= self.river_flow_threshold <= 1.0):
            raise ValueError(
                f"river_flow_threshold must be in [0, 1], got {self.river_flow_threshold}"
            )
        if self.moisture_bleed_passes < 0:
            raise ValueError(
                f"moisture_bleed_passes must be >= 0, got {self.moisture_bleed_passes}"
            )
        self.continent_falloff_edges = _coerce_edges(self.continent_falloff_edges)
        if not (0.0 <= self.continent_shelf_variance <= 1.0):
            raise ValueError(
                f"continent_shelf_variance must be in [0, 1], got {self.continent_shelf_variance}"
            )
        if self.continent_shelf_hexes < 1:
            raise ValueError(
                f"continent_shelf_hexes must be >= 1, got {self.continent_shelf_hexes}"
            )
        if not (0.0 <= self.continent_seabed < self.sea_level):
            raise ValueError(
                "continent_seabed must be in [0, sea_level) so the map edge is under "
                f"water, got seabed={self.continent_seabed}, sea_level={self.sea_level}"
            )
        if self.endorheic_marsh_radius < 0:
            raise ValueError(
                f"endorheic_marsh_radius must be >= 0, got {self.endorheic_marsh_radius}"
            )
        if not (0.0 <= self.endorheic_marsh_min_moisture <= 1.0):
            raise ValueError(
                "endorheic_marsh_min_moisture must be in [0, 1], "
                f"got {self.endorheic_marsh_min_moisture}"
            )
        if not (0.0 <= self.moisture_bleed_strength <= 1.0):
            raise ValueError(
                f"moisture_bleed_strength must be in [0, 1], got {self.moisture_bleed_strength}"
            )
        if self.regional_climate not in CLIMATE_CONTEXTS:
            raise ValueError(
                f"unknown regional_climate {self.regional_climate!r}; "
                f"choose from {', '.join(sorted(CLIMATE_CONTEXTS))}"
            )
        context = CLIMATE_CONTEXTS[self.regional_climate]
        if self.base_temperature is None:
            self.base_temperature = context.base_temperature
        if self.regional_moisture is None:
            self.regional_moisture = context.moisture_target
        if not (0.0 < self.regional_moisture < 1.0):
            raise ValueError(f"regional_moisture must be in (0, 1), got {self.regional_moisture}")
        if not (0.0 <= self.base_temperature <= 1.0):
            raise ValueError(f"base_temperature must be in [0, 1], got {self.base_temperature}")
        if not (0.0 <= self.latitude_temp_range <= 1.0):
            raise ValueError(
                f"latitude_temp_range must be in [0, 1], got {self.latitude_temp_range}"
            )
        if self.erosion_delta_min_load < 0:
            raise ValueError(
                f"erosion_delta_min_load must be >= 0, got {self.erosion_delta_min_load}"
            )
        if self.erosion_affinity_update_interval < 1:
            raise ValueError(
                "erosion_affinity_update_interval must be >= 1, "
                f"got {self.erosion_affinity_update_interval}"
            )
        if self.erosion_channel_affinity_gain < 0:
            raise ValueError(
                f"erosion_channel_affinity_gain must be >= 0, "
                f"got {self.erosion_channel_affinity_gain}"
            )
        if self.hex_size_m <= 0:
            raise ValueError(f"hex_size_m must be > 0, got {self.hex_size_m}")
        if self.road_elev_range_m <= 0:
            raise ValueError(f"road_elev_range_m must be > 0, got {self.road_elev_range_m}")
        if self.road_slope_free_pct < 0:
            raise ValueError(f"road_slope_free_pct must be >= 0, got {self.road_slope_free_pct}")
        if self.road_slope_cap_pct <= self.road_slope_free_pct:
            raise ValueError(
                "road_slope_cap_pct must be greater than road_slope_free_pct, "
                f"got cap={self.road_slope_cap_pct}, free={self.road_slope_free_pct}"
            )
        if self.road_slope_cap_mult <= 0:
            raise ValueError(f"road_slope_cap_mult must be > 0, got {self.road_slope_cap_mult}")
        if self.settlement_min_reachable < 1:
            raise ValueError(
                f"settlement_min_reachable must be >= 1, got {self.settlement_min_reachable}"
            )
        for name in (
            "cultivation_city_radius",
            "cultivation_town_radius",
            "cultivation_village_radius",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        for name in (
            "food_fertile_value",
            "food_marginal_value",
            "food_wetland_value",
            "food_water_value",
            "habitability_agri_weight",
            "habitability_river_bonus",
            "habitability_coast_bonus",
            "habitability_hill_bonus",
            "habitability_confluence_bonus",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        if self.biome_dry_moist > self.biome_wet_moist:
            raise ValueError(
                "biome_dry_moist must be <= biome_wet_moist, got "
                f"{self.biome_dry_moist} > {self.biome_wet_moist}"
            )
        if not (0.0 <= self.road_bank_discount_min_flow <= 1.0):
            raise ValueError(
                "road_bank_discount_min_flow must be in [0, 1], "
                f"got {self.road_bank_discount_min_flow}"
            )
        if self.road_river_hex_cost < 0:
            raise ValueError(f"road_river_hex_cost must be >= 0, got {self.road_river_hex_cost}")
        if self.road_ferry_max_hop < 1:
            raise ValueError(f"road_ferry_max_hop must be >= 1, got {self.road_ferry_max_hop}")
        if self.road_water_cost < 0:
            raise ValueError(f"road_water_cost must be >= 0, got {self.road_water_cost}")
        if self.road_embark_cost < 0:
            raise ValueError(f"road_embark_cost must be >= 0, got {self.road_embark_cost}")
        if self.road_disembark_cost < 0:
            raise ValueError(f"road_disembark_cost must be >= 0, got {self.road_disembark_cost}")
        if self.road_river_crossing_base < 0:
            raise ValueError(
                f"road_river_crossing_base must be >= 0, got {self.road_river_crossing_base}"
            )
        if self.road_river_crossing_flow < 0:
            raise ValueError(
                f"road_river_crossing_flow must be >= 0, got {self.road_river_crossing_flow}"
            )
        if self.road_river_traffic_min < 0:
            raise ValueError(
                f"road_river_traffic_min must be >= 0, got {self.road_river_traffic_min}"
            )

    # Climate
    # The map is a region, not a world: 500 km at 1 hex = 1 km is about 4.5 degrees of
    # latitude, some 3 C of mean annual temperature.  Altitude does far more than that
    # over the same distance — 3000 m of relief is nearly 20 C — and rain shadow more
    # again.  So the region has one climate, named here, and the variety inside it comes
    # from terrain: elevation via the lapse rate, and moisture via the orographic term.
    # This is what stops the biome mix from being an accident of the average elevation.
    regional_climate: str = "temperate"
    wind_direction: tuple[float, float] = (1.0, 0.0)
    base_temperature: float | None = None  # None = take it from regional_climate
    latitude_temp_range: float = 0.0  # negligible across a region; raise only for a continent
    altitude_lapse_rate: float = 0.4
    orographic_strength: float = 2.0
    base_moisture: float = 0.0  # flat bias added to land moisture after anchoring
    regional_moisture: float | None = None  # mean land moisture; None = from regional_climate

    # Biome thresholds
    biome_alpine_elev: float = 0.85
    biome_cold_temp: float = 0.25
    biome_warm_temp: float = 0.6
    biome_dry_moist: float = 0.2
    biome_wet_moist: float = 0.5

    # Settlements
    city_min_separation: int = 20
    town_min_separation: int = 8
    target_city_count: int = 6
    target_town_count: int = 24

    # Cultivation radii — also the catchment each tier is scored on by HabitabilityStage
    cultivation_city_radius: int = 8
    cultivation_town_radius: int = 4
    cultivation_village_radius: int = 2

    # Habitability — food value of one hex, by land cover band.  Water is deliberately
    # non-zero: a coastal site fishes, and scoring the sea at nothing penalised coastal
    # sites twice. Tundra, desert, alpine and bare rock are always zero.
    food_fertile_value: float = 1.0
    food_marginal_value: float = 0.4
    food_wetland_value: float = 0.15
    food_water_value: float = 0.4

    # Habitability — weight on the catchment mean, plus flat site bonuses
    habitability_agri_weight: float = 0.40
    habitability_river_bonus: float = 0.25
    habitability_coast_bonus: float = 0.25
    habitability_hill_bonus: float = 0.15
    habitability_confluence_bonus: float = 0.10

    # World scale
    hex_size_m: float = 1000.0  # metres per hex
    road_elev_range_m: float = 3000.0  # metres for full 0→1 elevation span

    # Roads — base terrain costs
    road_mountain_cost: float = 10.0
    road_hill_cost: float = 3.0
    road_flat_cost: float = 1.0

    # Roads — traveller simulation
    road_travellers_city: int = 500
    road_travellers_town: int = 100
    road_travellers_village: int = 20
    road_gravity_exponent: float = 1.5
    # Roads follow river valleys along the *bank*, never down the channel itself, so the
    # side of the river a road runs on stays readable. Discount applies to land hexes
    # adjacent to a river, scaled by the largest adjacent river's flow.
    road_bank_discount: float = 0.5
    road_bank_discount_min_flow: float = 0.2
    road_pheromone_factor: float = 0.1

    # Roads — water bodies (oceans + lakes treated as traversable)
    road_water_cost: float = 0.05
    road_embark_cost: float = 8.0
    road_disembark_cost: float = 8.0

    # Roads — river crossings (perpendicular to flow, charged on each land↔river edge)
    road_river_crossing_base: float = 4.0
    road_river_crossing_flow: float = 12.0

    # Cost of standing a road *on* a river hex. The channel exclusion only covers the
    # hexsides a river is drawn along; a meander or braid puts two river hexes side by
    # side without one, and a road threading those is still in the water. This makes
    # occupying the channel uneconomic while leaving a genuine crossing affordable.
    road_river_hex_cost: float = 12.0

    # Roads — ferries. A component cut off by a river mesh (a delta island, a braided
    # confluence) is joined by boat rather than by a road running down the channel.
    # Longer than this and no plausible ferry exists, so routing raises instead.
    road_ferry_max_hop: int = 4
    road_slope_cost: float = 2.0
    road_slope_free_pct: float = 3.0  # grade % below which slope costs nothing
    road_slope_cap_pct: float = 25.0  # grade % at which cost saturates
    road_slope_cap_mult: float = 10.0  # saturation multiplier at cap grade
    road_min_traffic: int = 3
    road_river_traffic_min: int = 1
    road_primary_pct: float = 0.10
    road_secondary_pct: float = 0.30
    road_track_pct: float = 0.60

    # Settlement placement
    settlement_min_reachable: int = 100  # min hexes reachable below cap grade

    @classmethod
    def from_json(cls, path: str) -> "WorldConfig":
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)
        _coerce_tuples(data)
        return _construct(cls, data)

    def to_json(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_yaml(cls, path: str) -> "WorldConfig":
        """Load config from YAML file. An 'export:' section is ignored."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError("YAML config root must be a mapping/object.")
        data.pop("export", None)
        _coerce_tuples(data)
        return _construct(cls, data)

    def to_yaml(self, path: str) -> None:
        """Save config to YAML file."""
        import yaml

        d = asdict(self)
        # Every tuple field, not just the numeric pairs: yaml.dump writes a bare tuple
        # as !!python/tuple, which safe_load then refuses to read back.
        for key, value in d.items():
            if isinstance(value, tuple):
                d[key] = list(value)
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False)


_TUPLE_FIELDS = ("wind_direction", "elevation_gradient")
_EDGES = ("north", "south", "east", "west")

# Settings that used to exist. A key here is dropped with a warning naming what replaced
# it, so a config written against an older version still loads instead of crashing.
_RETIRED_FIELDS: dict[str, str] = {}


def _construct(cls: type, data: dict) -> "WorldConfig":
    """Build a config from a loaded mapping, with a readable error for a bad key.

    Without this an unknown key reaches the dataclass constructor and raises `TypeError`,
    which the CLI's `except ValueError` handlers do not catch — so a single typo in a YAML
    file surfaced as a raw traceback rather than a message. Every other config error is a
    `ValueError`; this makes an unrecognised key one too.
    """
    import warnings

    for name in sorted(set(data) & set(_RETIRED_FIELDS)):
        warnings.warn(
            f"Config setting {name!r} has been retired: {_RETIRED_FIELDS[name]}",
            DeprecationWarning,
            stacklevel=3,
        )
        data.pop(name)

    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        plural = "s" if len(unknown) > 1 else ""
        listed = ", ".join(repr(name) for name in unknown)
        suggestions = "; ".join(
            f"{name!r}: did you mean {close!r}?"
            for name in unknown
            if (close := _closest(name, known)) is not None
        )
        message = f"Unknown config setting{plural}: {listed}."
        if suggestions:
            message += f" {suggestions}"
        raise ValueError(message)

    return cls(**data)


def _closest(name: str, known: set[str]) -> str | None:
    """The nearest known setting name, if one is close enough to be worth suggesting."""
    import difflib

    matches = difflib.get_close_matches(name, sorted(known), n=1, cutoff=0.7)
    return matches[0] if matches else None


def _coerce_edges(value: Any) -> tuple[str, ...]:
    """Normalise the falloff-edge list, keeping a stable order and rejecting typos."""
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",") if part.strip()]
    try:
        given = list(value)
    except TypeError as exc:
        raise ValueError(
            f"continent_falloff_edges must be a list of edge names, got {value!r}"
        ) from exc
    lowered = []
    for edge in given:
        if not isinstance(edge, str):
            raise ValueError(f"continent_falloff_edges entries must be strings, got {edge!r}")
        name = edge.strip().lower()
        if name not in _EDGES:
            raise ValueError(
                f"unknown edge {edge!r} in continent_falloff_edges; choose from {', '.join(_EDGES)}"
            )
        lowered.append(name)
    # Deduplicate but keep _EDGES order so the value is canonical and comparable.
    return tuple(e for e in _EDGES if e in set(lowered))


def _coerce_pair(key: str, value: Any) -> tuple[float, float]:
    """Normalize a 2D vector-like value into a 2-float tuple."""
    if isinstance(value, str) or value is None:
        raise ValueError(f"{key} must be an iterable of two numbers, got {value!r}")
    try:
        pair = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{key} must be an iterable of two numbers, got {value!r}") from exc
    if len(pair) != 2:
        raise ValueError(f"{key} must have exactly two values, got {len(pair)}")
    # bool is a subclass of int; reject True/False so accidental flags do not
    # silently become numeric vector components (1.0/0.0).
    if not all(isinstance(v, int | float) and not isinstance(v, bool) for v in pair):
        raise ValueError(f"{key} must contain only numbers, got {pair!r}")
    return (float(pair[0]), float(pair[1]))


def _coerce_tuples(data: dict[str, Any]) -> None:
    for key in _TUPLE_FIELDS:
        if key in data:
            data[key] = _coerce_pair(key, data[key])
