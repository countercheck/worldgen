import json
from dataclasses import asdict, dataclass


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

    def __post_init__(self) -> None:
        if not (0.0 <= self.river_flow_threshold <= 1.0):
            raise ValueError(
                f"river_flow_threshold must be in [0, 1], got {self.river_flow_threshold}"
            )
        if not (0.0 <= self.base_temperature <= 1.0):
            raise ValueError(f"base_temperature must be in [0, 1], got {self.base_temperature}")
        if not (0.0 <= self.latitude_temp_range <= 1.0):
            raise ValueError(
                f"latitude_temp_range must be in [0, 1], got {self.latitude_temp_range}"
            )

    # Climate
    wind_direction: tuple[float, float] = (1.0, 0.0)
    base_temperature: float = 0.5  # map's central temperature (0=arctic, 1=tropical)
    latitude_temp_range: float = 0.1  # pole-to-equator spread (was 0.6; tiny at 1 hex=1 km)
    altitude_lapse_rate: float = 0.4
    orographic_strength: float = 2.0

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
    road_pheromone_factor: float = 0.1
    road_slope_cost: float = 5.0
    road_min_traffic: int = 3
    road_primary_pct: float = 0.10
    road_secondary_pct: float = 0.30
    road_track_pct: float = 0.60

    @classmethod
    def from_json(cls, path: str) -> "WorldConfig":
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
