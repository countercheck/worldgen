import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class WorldConfig:
    """All tunable parameters for world generation."""

    width: int = 128
    height: int = 128
    sea_level: float = 0.45

    # Elevation
    noise_octaves: int = 6
    noise_persistence: float = 0.5
    noise_lacunarity: float = 2.0
    noise_scale: float = 3.0
    domain_warp_strength: float = 0.3
    continent_falloff: bool = True
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

    # Hydrology
    river_flow_threshold: float = 0.05
    river_flow_continuous: bool = False  # True: river_flow on all draining land hexes
    moisture_bleed_passes: int = 0       # 0 = flat river bonus (default); >0 = elevation-gated bleed
    moisture_bleed_strength: float = 0.3

    def __post_init__(self) -> None:
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
        if not (0.0 <= self.moisture_bleed_strength <= 1.0):
            raise ValueError(
                "moisture_bleed_strength must be in [0, 1], "
                f"got {self.moisture_bleed_strength}"
            )
        if not (0.0 <= self.base_temperature <= 1.0):
            raise ValueError(f"base_temperature must be in [0, 1], got {self.base_temperature}")
        if not (0.0 <= self.latitude_temp_range <= 1.0):
            raise ValueError(
                f"latitude_temp_range must be in [0, 1], got {self.latitude_temp_range}"
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
        if not (0.0 <= self.road_river_discount_min_flow <= 1.0):
            raise ValueError(
                "road_river_discount_min_flow must be in [0, 1], "
                f"got {self.road_river_discount_min_flow}"
            )
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
    wind_direction: tuple[float, float] = (1.0, 0.0)
    base_temperature: float = 0.5  # map's central temperature (0=arctic, 1=tropical)
    latitude_temp_range: float = 0.1  # pole-to-equator spread (was 0.6; tiny at 1 hex=1 km)
    altitude_lapse_rate: float = 0.4
    orographic_strength: float = 2.0
    base_moisture: float = 0.0

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

    # Cultivation radii
    cultivation_city_radius: int = 8
    cultivation_town_radius: int = 4
    cultivation_village_radius: int = 2

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
    road_river_discount: float = 0.5
    road_river_discount_min_flow: float = 0.2
    road_pheromone_factor: float = 0.1

    # Roads — water bodies (oceans + lakes treated as traversable)
    road_water_cost: float = 0.05
    road_embark_cost: float = 8.0
    road_disembark_cost: float = 8.0

    # Roads — river crossings (perpendicular to flow, charged on each land↔river edge)
    road_river_crossing_base: float = 4.0
    road_river_crossing_flow: float = 12.0
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
        return cls(**data)

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
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        """Save config to YAML file."""
        import yaml

        d = asdict(self)
        for key in _TUPLE_FIELDS:
            d[key] = list(d[key])
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False)


_TUPLE_FIELDS = ("wind_direction", "elevation_gradient")


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
