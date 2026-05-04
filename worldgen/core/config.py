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

    # Climate
    wind_direction: tuple[float, float] = (1.0, 0.0)
    latitude_temp_range: float = 0.6
    altitude_lapse_rate: float = 0.4

    # Settlements
    city_min_separation: int = 20
    town_min_separation: int = 8
    target_city_count: int = 6
    target_town_count: int = 24

    # Roads
    road_mountain_cost: float = 10.0
    road_hill_cost: float = 3.0
    road_flat_cost: float = 1.0
    road_river_crossing_cost: float = 5.0

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
