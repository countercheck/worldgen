import json
import os
from dataclasses import asdict, dataclass, fields
from typing import Any

from .hex_grid import GRID_LAYOUTS

# How an imported image is read.  See `WorldConfig.heightmap_mode`.
HEIGHTMAP_MODES = ("elevation", "coastline")


@dataclass(frozen=True)
class ClimateContext:
    """The climate of the region as a whole.

    `base_temperature` and `moisture_target` set where the region sits on the temperature
    and moisture axes; `palette` is the set of biomes that can occur there.  The palette
    is what keeps a region coherent: an arid region varies from desert to steppe to
    alpine with altitude, but never produces jungle three valleys over.
    """

    mean_temperature_c: float
    mean_precip_mm: float
    palette: frozenset


def _palette(*names: str) -> frozenset:
    from .hex import Biome

    return frozenset(getattr(Biome, n) for n in names)


# Biomes every region can produce regardless of climate, because they are made by
# terrain rather than by climate: bare peaks, and waterlogged ground beside rivers.
_ALWAYS = ("ALPINE", "WETLAND", "OCEAN")

CLIMATE_CONTEXTS: dict[str, ClimateContext] = {
    # Mean annual temperature at sea level, in degrees Celsius — real figures for the
    # regions these name, rather than positions on an abstract 0-1 axis.
    "boreal": ClimateContext(1.0, 450.0, _palette("TUNDRA", "BOREAL", "GRASSLAND", *_ALWAYS)),
    "temperate": ClimateContext(
        10.0, 800.0, _palette("TEMPERATE_FOREST", "GRASSLAND", "BOREAL", "SHRUBLAND", *_ALWAYS)
    ),
    "mediterranean": ClimateContext(
        16.0, 550.0, _palette("SHRUBLAND", "GRASSLAND", "TEMPERATE_FOREST", *_ALWAYS)
    ),
    "arid": ClimateContext(21.0, 200.0, _palette("DESERT", "SHRUBLAND", "GRASSLAND", *_ALWAYS)),
    "tropical": ClimateContext(
        26.0, 2000.0, _palette("TROPICAL", "GRASSLAND", "SHRUBLAND", *_ALWAYS)
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
    # Sea level is the datum: elevation is metres above it, so it is zero by
    # definition and is no longer a setting. What the map looks like is set by how
    # high the land stands and how deep the sea lies, below.

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
    # The highest ground on the map, in metres above sea level. This is the map's
    # vertical scale and the single most consequential setting for what kind of
    # country it is: 2400 gives real uplands with a little ground above the treeline,
    # 1200 gives downland and hills, 4000 gives an Alpine massif. How much of the map
    # ends up under water follows from this and seabed_depth_m rather than being set
    # directly, which is the honest way round.
    max_elevation_m: float = 1500.0
    # How deep the sea floor lies at the map edge, in metres below sea level. A
    # continental shelf, not an abyss — the falloff blends the border down to this,
    # and a shallow shelf makes a gentler coast than a plunge would.
    seabed_depth_m: float = 200.0
    # Land no higher than this beside the sea is coast: the beach, the marsh, the harbour
    # flat. Metres above sea level, so it means the same on any map.
    coast_max_elevation_m: float = 100.0
    # A directional tilt across the whole map, in metres: [east, south]. Positive east
    # raises the eastern edge, positive south the southern one.
    elevation_gradient_m: tuple[float, float] = (0.0, 0.0)

    # Elevation from an image.  Setting a path swaps ImageElevationStage in for
    # ElevationStage, so the terrain comes from the picture rather than from noise.
    # A relative path resolves against the working directory, as `--config` does.
    #   "elevation"  the image is a greyscale heightmap: luminance maps linearly onto
    #                the elevation range, black at the sea floor and white at
    #                max_elevation_m.
    #   "coastline"  the image is only a land/sea stencil.  Heights still come from the
    #                noise, shaped so that land sits above sea level and sea below it,
    #                which makes the drawn coastline the coastline.
    heightmap_path: str | None = None
    heightmap_mode: str = "elevation"
    # Coastline mode only.  Alpha decides land where the image has a meaningful alpha
    # channel; otherwise luminance is compared against this threshold.
    heightmap_land_threshold: float = 0.5
    heightmap_invert: bool = False  # True: the darker side of the threshold is the land
    # Coastline mode only.  The stencil is authoritative by default, so land drawn
    # running off the edge stays land.  Set true to also apply the rectangular edge
    # falloff, which rings the map with sea and guarantees rivers a coast to reach.
    heightmap_coast_falloff: bool = False

    # Terrain classification — bands of gradient, in metres of rise per kilometre.
    # Absolute, not a fraction of the elevation range: the old fractional thresholds moved
    # with the map's vertical scale, so what counted as a mountain was 120 m/km at a
    # 3000 m span and 20 m/km on a 500 m map, where it called rolling downland a peak.
    #   under 30    FLAT        level going: plough it, cart across it
    #   30 to 100   ROLLING     undulating; farmed, and a laden cart manages
    #   100 to 250  STEEP       pack animals, terraces, no wheels
    #   over 250    ESCARPMENT  a break of slope; on foot and with effort
    terrain_rolling_gradient_m: float = 30.0
    terrain_steep_gradient_m: float = 100.0
    terrain_escarpment_gradient_m: float = 250.0

    # Erosion
    # Droplets run per land hex, not per map. A flat count is a different amount of
    # weather depending on map size — at the old 15000 a 32x32 map got 14.6 per hex and a
    # 128x128 got 0.9, which is most of why small maps came out mountainous and the
    # default map came out as barely-touched noise.
    #
    # The dose is what decides whether the map has valleys. Below about one per hex the
    # rivers only scratch a line into the noise: there is no floodplain, and the ground
    # beside a trunk river climbs at a constant 12 m per km with no break of slope. It is
    # also non-monotonic at the low end, because erosion incises before it fills — half a
    # droplet per hex leaves the map *rougher* than none at all.
    # 3.0 is chosen against two things that pull opposite ways. Valleys want erosion:
    # the floodplain appears between 1 and 2 droplets per hex, where the rise beside a
    # trunk river falls from 27 m per km to 4 m. Rain shadow wants relief: the orographic
    # term lifts on elevation above sea level, so wearing the high ground down flattens
    # the moisture contrast — measured windward-to-leeward it falls from 0.37 at the old
    # dose to 0.13 at 15 per hex, taking the dry biomes with it. 3.0 has the floodplains
    # and keeps about three quarters of the shadow; going higher buys level ground at the
    # cost of the map having any dry country at all.
    erosion_droplets_per_hex: float = 3.0
    erosion_inertia: float = 0.05
    erosion_capacity: float = 4.0
    erosion_deposition: float = 0.3
    erosion_erosion_rate: float = 0.3
    erosion_channel_affinity_gain: float = 0.5
    erosion_affinity_update_interval: int = 500
    erosion_delta_min_load: float = 0.15

    # Hydrology
    # A channel forms where enough water passes to keep one open. Discharge is
    # catchment area times runoff depth, so this is in km2 x mm — the product of the
    # two. It replaced a threshold that was really a rank: the top 5% of land by
    # accumulation, which gave every map the same 5.6% of its land under channel
    # whether it was desert or rainforest.
    channel_min_discharge: float = 20000.0
    # Rain the ground and its plants take before anything runs off. Evapotranspiration
    # rises with temperature — that is most of what it is — so it is expressed as a base
    # plus a rate per degree rather than a flat figure. A flat one gave a boreal region
    # the same runoff as a desert, when cold country in fact sheds nearly all its rain;
    # that is why the taiga is full of rivers on modest rainfall.
    #   boreal      1 C   ->  80 mm taken of 450  ->  370 mm runs off
    #   temperate  10 C   -> 350 mm taken of 800  ->  450 mm
    #   arid       21 C   -> 680 mm taken of 200  ->  nothing but the floor
    evapotranspiration_base_mm: float = 50.0
    evapotranspiration_per_c_mm: float = 30.0
    # A floor on runoff, so even a desert drains its largest valleys after a storm
    # rather than having no watercourses at all.
    min_runoff_mm: float = 25.0
    # Runoff above which flat ground beside a river waterlogs into bog or marsh. A
    # runoff test rather than a rainfall one: peat forms in cold country on modest rain,
    # because so little of it evaporates away again.
    wetland_min_runoff_mm: float = 300.0
    river_flow_continuous: bool = False  # True: river_flow on all draining land hexes
    moisture_bleed_passes: int = 0  # 0 = flat river bonus (default); >0 = elevation-gated bleed
    moisture_bleed_strength: float = 0.3

    # Lake drainage
    lake_chaining: bool = True  # Let a lake spill into a strictly lower lake, not only the sea
    endorheic_marsh_radius: int = 1  # Shore band (hexes) turned to wetland around a closed basin
    # Below this much annual rainfall a closed basin evaporates to a salt pan rather
    # than holding a marshy shore.
    endorheic_marsh_min_precip_mm: float = 300.0

    def __post_init__(self) -> None:
        if self.grid_layout not in GRID_LAYOUTS:
            raise ValueError(
                f"unknown grid_layout {self.grid_layout!r}; choose from {', '.join(GRID_LAYOUTS)}"
            )
        self.wind_direction = _coerce_pair("wind_direction", self.wind_direction)
        self.elevation_gradient_m = _coerce_pair("elevation_gradient_m", self.elevation_gradient_m)
        if self.channel_min_discharge < 0:
            raise ValueError(
                f"channel_min_discharge must be >= 0, got {self.channel_min_discharge}"
            )
        if self.evapotranspiration_base_mm < 0 or self.evapotranspiration_per_c_mm < 0:
            raise ValueError(
                "evapotranspiration terms must be >= 0, got "
                f"{self.evapotranspiration_base_mm} and {self.evapotranspiration_per_c_mm}"
            )
        if self.wetland_min_runoff_mm < 0:
            raise ValueError(
                f"wetland_min_runoff_mm must be >= 0, got {self.wetland_min_runoff_mm}"
            )
        if self.min_runoff_mm < 0:
            raise ValueError(f"min_runoff_mm must be >= 0, got {self.min_runoff_mm}")
        if self.navigable_min_discharge < self.channel_min_discharge:
            raise ValueError(
                "navigable_min_discharge must be at least channel_min_discharge — a river "
                "cannot float a boat where there is not enough water for a channel, got "
                f"{self.navigable_min_discharge} and {self.channel_min_discharge}"
            )
        if not (0.0 <= self.moisture_resupply_per_hex <= 1.0):
            raise ValueError(
                f"moisture_resupply_per_hex must be in [0, 1], got {self.moisture_resupply_per_hex}"
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
        if self.max_elevation_m <= 0.0:
            raise ValueError(f"max_elevation_m must be above sea level, got {self.max_elevation_m}")
        if self.coast_max_elevation_m < 0.0:
            raise ValueError(
                f"coast_max_elevation_m must be >= 0, got {self.coast_max_elevation_m}"
            )
        if self.seabed_depth_m <= 0.0:
            raise ValueError(
                "seabed_depth_m is a depth below sea level and must be positive, got "
                f"{self.seabed_depth_m}"
            )
        if self.heightmap_path is not None:
            # A programmatic caller reaches for a Path; everything downstream, including
            # the JSON and YAML dumps, wants a plain string.
            if isinstance(self.heightmap_path, os.PathLike):
                self.heightmap_path = os.fspath(self.heightmap_path)
            if not isinstance(self.heightmap_path, str):
                raise ValueError(
                    f"heightmap_path must be a path or None, got {type(self.heightmap_path).__name__}"
                )
            if not self.heightmap_path:
                raise ValueError("heightmap_path must not be empty; use null to disable it")
        if self.heightmap_mode not in HEIGHTMAP_MODES:
            raise ValueError(
                f"unknown heightmap_mode {self.heightmap_mode!r}; "
                f"choose from {', '.join(HEIGHTMAP_MODES)}"
            )
        if not (0.0 <= self.heightmap_land_threshold <= 1.0):
            raise ValueError(
                f"heightmap_land_threshold must be in [0, 1], got {self.heightmap_land_threshold}"
            )
        if self.endorheic_marsh_radius < 0:
            raise ValueError(
                f"endorheic_marsh_radius must be >= 0, got {self.endorheic_marsh_radius}"
            )
        if self.endorheic_marsh_min_precip_mm < 0.0:
            raise ValueError(
                "endorheic_marsh_min_precip_mm must be >= 0, "
                f"got {self.endorheic_marsh_min_precip_mm}"
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
        if self.mean_temperature_c is None:
            self.mean_temperature_c = context.mean_temperature_c
        if self.mean_precip_mm is None:
            self.mean_precip_mm = context.mean_precip_mm
        if not (0.0 < self.mean_precip_mm <= 12000.0):
            raise ValueError(
                "mean_precip_mm must be a plausible annual rainfall in millimetres, "
                f"got {self.mean_precip_mm}"
            )
        if self.food_drowned_precip_mm <= self.biome_wet_precip_mm:
            raise ValueError(
                "food_drowned_precip_mm must be above biome_wet_precip_mm, got "
                f"{self.food_drowned_precip_mm} and {self.biome_wet_precip_mm}"
            )
        if self.biome_dry_precip_mm >= self.biome_wet_precip_mm:
            raise ValueError(
                "biome_dry_precip_mm must be below biome_wet_precip_mm, got "
                f"{self.biome_dry_precip_mm} and {self.biome_wet_precip_mm}"
            )
        if not (-60.0 <= self.mean_temperature_c <= 50.0):
            raise ValueError(
                "mean_temperature_c must be a plausible mean annual temperature in "
                f"Celsius, got {self.mean_temperature_c}"
            )
        if self.latitude_temp_range_c < 0.0:
            raise ValueError(
                f"latitude_temp_range_c must be >= 0, got {self.latitude_temp_range_c}"
            )
        if self.lapse_rate_c_per_km < 0.0:
            raise ValueError(f"lapse_rate_c_per_km must be >= 0, got {self.lapse_rate_c_per_km}")
        if self.biome_treeline_temp_c > self.biome_cold_temp_c:
            raise ValueError(
                "biome_treeline_temp_c must be at or below biome_cold_temp_c — trees stop "
                "above the treeline, so it cannot be warmer than the cold band, got "
                f"{self.biome_treeline_temp_c} and {self.biome_cold_temp_c}"
            )
        if self.biome_cold_temp_c >= self.biome_warm_temp_c:
            raise ValueError(
                "biome_cold_temp_c must be below biome_warm_temp_c, got "
                f"{self.biome_cold_temp_c} and {self.biome_warm_temp_c}"
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

        if self.road_slope_free_pct < 0:
            raise ValueError(f"road_slope_free_pct must be >= 0, got {self.road_slope_free_pct}")
        if self.road_slope_cap_pct <= self.road_slope_free_pct:
            raise ValueError(
                "road_slope_cap_pct must be greater than road_slope_free_pct, "
                f"got cap={self.road_slope_cap_pct}, free={self.road_slope_free_pct}"
            )
        if self.road_travellers_per_pop <= 0:
            raise ValueError(
                f"road_travellers_per_pop must be > 0, got {self.road_travellers_per_pop}"
            )
        if self.road_travellers_max < 1:
            raise ValueError(f"road_travellers_max must be >= 1, got {self.road_travellers_max}")
        if self.road_slope_cap_mult <= 0:
            raise ValueError(f"road_slope_cap_mult must be > 0, got {self.road_slope_cap_mult}")
        # Below 2.0 the rule can never fire: a detour is two legs where there was one.
        if self.road_settlement_detour_max_mult < 2.0:
            raise ValueError(
                "road_settlement_detour_max_mult must be >= 2.0 (a detour is two legs "
                f"where there was one), got {self.road_settlement_detour_max_mult}"
            )
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
        for name in (
            "terrain_rolling_gradient_m",
            "terrain_steep_gradient_m",
            "terrain_escarpment_gradient_m",
            "haulage_range_land",
            "rural_field_radius",
            "market_day_radius",
            "travel_ascent_per_hex",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        # The regime ordering is the model's central claim: a farmer's daily walk is
        # shorter than a day's return to market, which is shorter than the distance bulk
        # grain survives overland. Invert any of these and the hierarchy stops meaning
        # anything, so it is checked rather than assumed.
        if not (self.rural_field_radius < self.market_day_radius < self.haulage_range_land):
            raise ValueError(
                "haulage ranges must increase: rural_field_radius < market_day_radius < "
                f"haulage_range_land, got {self.rural_field_radius} < "
                f"{self.market_day_radius} < {self.haulage_range_land}"
            )
        if self.haulage_range_water_mult < 1.0:
            raise ValueError(
                "haulage_range_water_mult must be >= 1 (water cannot carry bulk less far "
                f"than land), got {self.haulage_range_water_mult}"
            )
        if not (0.0 < self.marketable_surplus_fraction <= 1.0):
            raise ValueError(
                "marketable_surplus_fraction must be in (0, 1], got "
                f"{self.marketable_surplus_fraction}"
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
    # Mean annual temperature at sea level, in degrees Celsius. None takes it from
    # regional_climate. Real degrees rather than a 0-1 axis, so a threshold set against it
    # means what it says and cannot shift when the axis is reinterpreted.
    mean_temperature_c: float | None = None
    # Degrees between the pole-ward and equator-ward edges. Negligible across a region:
    # 128 km is about a degree of latitude, worth a few tenths of a degree.
    latitude_temp_range_c: float = 0.0
    # The environmental lapse rate: how fast air cools with height. 6.5 C/km is the
    # standard figure, and being a real rate it stays right whatever the map's relief.
    lapse_rate_c_per_km: float = 6.5
    orographic_strength: float = 2.0
    # How much of its moisture deficit the air makes back over each kilometre it travels,
    # by evaporation from the ground it crosses. Without it the orographic sweep is a
    # one-way drying and the far side of the map gets no rain at all: a rain shadow would
    # extend from the first hill to the border rather than being the local feature it is.
    # At 0.08 the air is most of the way recovered after twenty kilometres or so.
    moisture_resupply_per_hex: float = 0.08
    # A flat bias in millimetres a year, added after the pattern is anchored. Positive
    # wets the whole region, negative dries it.
    base_precip_mm: float = 0.0
    # Mean annual precipitation over land, in millimetres. None takes it from
    # regional_climate. The orographic pass gives a relative wet/dry pattern — which
    # slopes catch the rain and which sit in a shadow — and this says what that pattern
    # is worth in real rainfall.
    mean_precip_mm: float | None = None

    # ---- Haulage economics -------------------------------------------------
    # The ranges the settlement hierarchy is built on.  Each is a *travel-cost* budget,
    # not a distance, so terrain shortens it: at 1 hex = 1 km these are calibrated to give
    # the historical figure on flat ground, and less across hills.
    #
    #   rural_field_radius  ~2 km   the daily walk out to the fields.  Chisholm: cropping
    #                               intensity falls off past ~1 km, land past 3-4 km is
    #                               grazing or waste, and by ~5 km founding a daughter
    #                               settlement beats continuing to walk.
    #   market_day_radius   ~10 km  out to market, business done, and home inside a day.
    #                               Bracton held markets should stand 6 2/3 miles apart,
    #                               being a third of a twenty-mile day out and a third
    #                               back; English market towns do cluster at 10-15 km.
    #   haulage_range_land  ~40 km  bulk grain overland before the team has eaten the
    #                               cargo.  The softest number here — what is actually
    #                               well attested is the ratio below, not this absolute.
    haulage_range_land: float = 40.0
    # Diocletian's Price Edict prices land carriage at roughly 55x sea and 11x river for
    # the same tonne-kilometre.  This multiplier is why large pre-industrial cities sit on
    # navigable water and inland ones stay small: nothing gates a city, water simply
    # extends what can feed it.
    haulage_range_water_mult: float = 15.0
    # river_flow at or above which a river floats a boat. Below it a river is something
    # you ford, not something you ship grain down.
    # Discharge at which a river will float a boat, in the same km2 x mm as
    # channel_min_discharge. Well above it: most of a drainage net is wadeable
    # headwater, and only the trunk carries cargo.
    navigable_min_discharge: float = 150000.0
    # A farming household eats most of what it grows; only this share can leave for a
    # market. Sizing markets off the *surplus* rather than the production is why the tier
    # ratios come out right without target counts anywhere.
    marketable_surplus_fraction: float = 0.20
    # Naismith's rule: this many metres of ascent cost as much as one hex of level
    # ground. Catchments are walked, not engineered, so they use this rather than
    # road_slope_cost — that curve prices grading a road and saturates at ten times base,
    # which over eroded terrain shrinks a catchment to a third of its proper reach.
    travel_ascent_per_hex: float = 125.0

    # ---- River crossings ---------------------------------------------------
    # A river is not uniformly crossable. Most of its length is an obstacle; a few places
    # are not, and those places are why towns sit where they do. Crossings are settled
    # before anything is built, so a bridging point can be the reason a market grows there
    # rather than a consequence of one.
    #
    # Fords are physical and free: shallow braided water anyone can wade. What makes a
    # reach shallow is small discharge over a slack bed — a steep reach of the same river
    # is a gorge, and a large one is deep whatever its bed is doing.
    # Catchment area, in km2, at or below which the water can be waded. A physical
    # figure and comparable between maps, unlike river_flow, which is normalised against
    # the largest accumulation present and so is a rank rather than a quantity. A stream
    # draining a few tens of km2 is ankle deep and a step across; one draining thousands
    # is not.
    ford_max_catchment_km2: float = 60.0
    # Local relief, in metres, that doubles how hard a reach is to get across. Fast water
    # takes your feet from under you whatever its depth, and at a kilometre to the hex it
    # is the approaches rather than the span that defeat a bridge — both scale with how
    # steep the ground is, so relief makes a reach behave like a bigger river for fording
    # and for building alike. A floodplain has a few metres of it; a gorge has hundreds.
    crossing_relief_m: float = 60.0
    # A bridge is capital, so it appears only where enough traffic will use it: the
    # surplus a site needs within reach, per multiple of the widest wadeable span. A river
    # twice that width needs twice the traffic to be worth the structure. This is
    # road_river_crossing_base's idea doing its proper work — a threshold a particular
    # site either clears or does not, rather than a toll charged along every watercourse.
    # Nobody bridges to nowhere.
    bridge_pressure_per_span: float = 3.0
    crossing_pressure_radius: int = 6  # how far either bank is searched for that surplus
    crossing_min_separation: int = 4  # nobody builds two bridges within sight of each other
    # Getting across away from a crossing, per multiple of the wadeable span, charged on
    # each land-river edge. Deliberately has no fixed term, unlike road_river_crossing_base:
    # that base is the capital of building a bridge, and somebody walking to market pays no
    # capital — what stops them is how much water is in the way.
    travel_ford_cost: float = 8.0
    crossing_use_cost: float = 0.5  # using an existing ford or bridge

    # Biome thresholds
    # The mean annual temperature at which trees stop, in Celsius. The treeline is not a
    # height — it is a temperature, and the height it happens to fall at follows from how
    # warm the region is and how fast air cools with altitude. That is why it stands at
    # about 1800 m in temperate country, some 500 m in the subarctic, and above 4000 m in
    # the tropics. A fixed altitude could not say any of that, and a fraction of the
    # elevation range said the opposite: it gave every map the same share of alpine ground
    # however low its hills.
    #
    # -2 C rather than the 1 C first set here, for two reasons. It is the better figure:
    # treelines sit near -1 to -2 C mean annual in the Alps and in Scandinavia, and
    # Siberian larch grows a great deal colder than that, so 1 C put the line too low
    # everywhere. And 1 C collided exactly with the boreal region's own mean of 1 C, which
    # put its treeline at sea level — the alpine test runs before every temperature rule,
    # so a boreal map came out as bare rock from shore to shore with no boreal forest
    # anywhere in it, and supported five settlements on 16 000 hexes.
    biome_treeline_temp_c: float = -2.0
    # Mean annual temperature bounding the cold and warm biome bands, in Celsius. Taiga
    # gives way to broadleaf woodland around 5 C; the warm band begins where subtropical
    # vegetation takes over, around 18 C.
    biome_cold_temp_c: float = 5.0
    biome_warm_temp_c: float = 18.0
    # Annual rainfall bounding the dry and wet biome bands, in millimetres. Below about
    # 400 mm you get steppe and desert; above about 1000 mm, closed wet forest.
    biome_dry_precip_mm: float = 400.0
    biome_wet_precip_mm: float = 1000.0
    # Annual rainfall at which ground is drowned for farming — leached, waterlogged, and
    # worth nothing. The wet arm of the agricultural curve falls to zero here. It used to
    # fall away to a normalised 1.0, which was the ceiling of the old moisture scale
    # rather than anything about growing food.
    food_drowned_precip_mm: float = 3000.0

    # Settlements — the classic model (ranked placement against fixed counts)
    city_min_separation: int = 20
    town_min_separation: int = 8
    target_city_count: int = 6
    target_town_count: int = 24

    # ---- Settlements: the haulage model ------------------------------------
    # Counts here are emergent, not configured.  What is set is how far people travel and
    # how poor a site may be before nobody bothers; the map decides how many result.

    # The rural surface. The countryside is a continuous productive surface rather than a
    # list of hamlets: a market's draw is the surplus of its catchment, and integrating
    # over the food field gives the same number as enumerating ~900 hamlets on a 128x128
    # map, almost none of which would carry military or administrative weight.
    rural_field_radius: float = 2.5  # the daily walk to the fields; sets cultivated extent
    # Calibrated so market towns land in their historical 500-2500 band: across five
    # seeds at 128x128 this gives medians of 1260-1700 and a largest of 4200-5450.
    people_per_food: float = 400.0

    # Market centres. A market goes where it can gather the most surplus inside a day's
    # return — central-place logic with a real transport cost rather than an abstract one.
    market_day_radius: float = 10.0
    market_kernel_decay: float = 4.0  # d0 in the 1/(1 + d/d0) depletion share
    # A suppression disc only, to stop two markets sharing a hexside. Real spacing comes
    # from competition for surplus, which is what makes it dense on rich ground and sparse
    # on poor — a fixed separation cannot express that.
    market_min_separation: int = 5
    # The one density knob, replacing target_city_count and target_town_count both: stop
    # planting once the best remaining site scores below this. Calibrated to ~70-85 markets
    # at 128x128 (England had ~700 markets in ~130,000 km2; this map is about an eighth of
    # that): on a temperate map with continent_falloff_edges = ("south",), seeds 42/7/3/11/19
    # give 74-81 markets — one per ~205 km2 of land, which is a 15 km lattice, with each
    # market about 10 km from its nearest neighbour. Those two figures bracket Bracton's
    # 6 2/3-mile rule read as a third of a day out and back, and observed English market
    # clustering at 10-15 km. They differ because markets cluster on good ground rather
    # than tiling evenly, so quote whichever one the question is actually about.
    #
    # It is an absolute threshold on gathered surplus, not a target, so density follows the
    # land with no further machinery: the same 14.0 yields 9 markets on an arid map and 74 on
    # a temperate one, monotone in mean food per land hex, while median market population
    # stays flat across every climate.  Fertility decides how *many* markets a region carries,
    # not how big each one grows.
    market_viability_floor: float = 14.0

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

    # Roads — base terrain costs
    road_escarpment_cost: float = 20.0
    road_steep_cost: float = 10.0
    road_rolling_cost: float = 3.0
    road_flat_cost: float = 1.0

    # Roads — traveller simulation
    #
    # Travellers come from population, not from tier. A flat count per tier meant a town of
    # 6,200 and one of 900 each sent the same hundred people, so population entered the
    # model only on the destination side of the gravity term and a large market wore no
    # deeper a road out of its own gates than a small one did.
    #
    # 0.04 keeps the total near what the tier counts produced on a 128x128 map (about 8,000
    # travellers over 74 markets), so it is a redistribution rather than a change of dose.
    road_travellers_per_pop: float = 0.04
    # A cap, so one large city cannot drown the map. Reached only by a settlement above
    # 12,500 people, which on these maps means a real city rather than a market town.
    road_travellers_max: int = 500
    # How sharply a destination's appeal falls with distance: weight is pop / d**this.
    #
    # 2.5 rather than the 1.5 a modern gravity model would use, because a laden cart is not
    # a lorry. At 1.5 a traveller was nearly as likely to make for a town forty kilometres
    # off as the one ten kilometres away, so 72% of every possible pair of settlements
    # ended up with a road of its own and the network came out a mat rather than a
    # hierarchy. Pre-industrial traffic is overwhelmingly local; the exponent is what says
    # so.
    road_gravity_exponent: float = 2.5
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
    # A road passing a settlement at one hex is bent through it instead, because a road
    # that skirts a town at the width of a field is a motor-age idea. This is what the
    # detour may cost, as a multiple of the edge it replaces.
    #
    # 2.0 is free: the detour is two legs where there was one, so on even ground it costs
    # exactly double by construction. What this bounds is the ground *beyond* that — the
    # town on the far bank of a river, up an escarpment, or above a cliff. Those are
    # caught together because they are all simply dear, which a grade cap would miss: the
    # worst case measured on a 128x128 map cost 31 times its bypass at a grade of 4%,
    # having been hauled onto a river channel rather than up anything.
    #
    # The slope cap still refuses outright what a laden cart cannot climb at all; this
    # decides what is merely not worth it.
    road_settlement_detour_max_mult: float = 4.0
    road_min_traffic: int = 3
    road_river_traffic_min: int = 1
    road_primary_pct: float = 0.10
    road_secondary_pct: float = 0.30
    road_track_pct: float = 0.60

    # Settlement placement
    settlement_min_reachable: int = 100  # min hexes reachable below cap grade

    def treeline_m(self) -> float:
        """The altitude the treeline falls at, in metres above sea level.

        Derived rather than set: it is where the lapse rate has cooled the region's mean
        annual temperature down to the point trees stop. About 1400 m in temperate
        country, at sea level in the subarctic, near 4000 m in the tropics — all from one
        temperature and one rate, none of it configured per map.
        """
        if self.lapse_rate_c_per_km <= 0.0:
            return float("inf")
        drop = self.mean_temperature_c - self.biome_treeline_temp_c
        return max(0.0, drop / self.lapse_rate_c_per_km * 1000.0)

    def runoff_mm(self, precip_mm: float, temp_c: float | None = None) -> float:
        """How much of a year's rain runs off, rather than returning to the air.

        What is left after evapotranspiration, floored so even a desert drains its
        largest valleys after a storm. This is the term that makes drainage far more
        climate-sensitive than rainfall alone, in both directions: the ground takes its
        share off the top, and how big that share is depends on how warm it is. A cold
        region sheds nearly everything that falls on it, which is why the taiga is full of
        rivers on rainfall a Mediterranean hillside would call meagre.
        """
        if temp_c is None:
            temp_c = self.mean_temperature_c
        et = self.evapotranspiration_base_mm + self.evapotranspiration_per_c_mm * max(0.0, temp_c)
        return max(self.min_runoff_mm, precip_mm - et)

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


_TUPLE_FIELDS = ("wind_direction", "elevation_gradient_m")
_EDGES = ("north", "south", "east", "west")

# Settings that used to exist. A key here is dropped with a warning naming what replaced
# it, so a config written against an older version still loads instead of crashing.
_RETIRED_FIELDS: dict[str, str] = {
    "base_temperature": (
        "temperature is in degrees Celsius now, not on a 0-1 axis; use mean_temperature_c "
        "(temperate is 10.0)"
    ),
    "latitude_temp_range": (
        "temperature is in degrees Celsius now; use latitude_temp_range_c, the spread in "
        "degrees between the map's pole-ward and equator-ward edges"
    ),
    "altitude_lapse_rate": (
        "the lapse rate is a real rate now; use lapse_rate_c_per_km (6.5 is standard)"
    ),
    "biome_cold_temp": (
        "biome temperature bands are in Celsius now; use biome_cold_temp_c (try 5.0)"
    ),
    "biome_warm_temp": (
        "biome temperature bands are in Celsius now; use biome_warm_temp_c (try 18.0)"
    ),
    "base_moisture": (
        "moisture is measured in millimetres of annual rainfall now; use base_precip_mm"
    ),
    "regional_moisture": (
        "moisture is measured in millimetres of annual rainfall now; use mean_precip_mm "
        "(temperate is 800)"
    ),
    "biome_dry_moist": (
        "biome moisture bands are in millimetres of annual rainfall now; use "
        "biome_dry_precip_mm (try 400)"
    ),
    "biome_wet_moist": (
        "biome moisture bands are in millimetres of annual rainfall now; use "
        "biome_wet_precip_mm (try 1000)"
    ),
    "endorheic_marsh_min_moisture": (
        "moisture is measured in millimetres of annual rainfall now; use "
        "endorheic_marsh_min_precip_mm (try 300)"
    ),
    "river_flow_threshold": (
        "channels are decided by discharge now, not by taking the top fraction of land by "
        "accumulation; use channel_min_discharge (catchment km2 times runoff mm)"
    ),
    "navigable_river_flow": (
        "navigability is decided by discharge now; use navigable_min_discharge"
    ),
    "sea_level": (
        "elevation is metres above sea level now, so sea level is zero by definition; "
        "set max_elevation_m and seabed_depth_m instead"
    ),
    "continent_seabed": (
        "elevation is metres above sea level now; use seabed_depth_m, a depth in metres"
    ),
    "elevation_gradient": (
        "elevation is metres now; use elevation_gradient_m, a tilt in metres across the map"
    ),
    "biome_alpine_elev": (
        "the treeline is a temperature now, not a height: trees stop where it is too cold "
        "for them, and the altitude that happens at follows from the region's warmth and "
        "the lapse rate. Use biome_treeline_temp_c (try 1.0)"
    ),
    "road_elev_range_m": (
        "elevation is metres throughout, so nothing needs converting from a 0-1 range any "
        "more; max_elevation_m sets the map's vertical scale"
    ),
    "erosion_iterations": (
        "erosion is now dosed per land hex so it means the same thing at any map size; "
        "use erosion_droplets_per_hex (try 8.0, which is 15000 droplets on a 48x48 map)"
    ),
    "terrain_hill_gradient": (
        "terrain classes are now bands of absolute gradient; use "
        "terrain_rolling_gradient_m, in metres of rise per kilometre"
    ),
    "terrain_mountain_gradient": (
        "terrain classes are now bands of absolute gradient; use "
        "terrain_steep_gradient_m, in metres of rise per kilometre"
    ),
}

# Settings that were renamed rather than dropped. The value carries over, so a config
# written against an older version keeps working and keeps meaning what it meant.
_RENAMED_FIELDS: dict[str, str] = {
    "road_hill_cost": "road_rolling_cost",
    "road_mountain_cost": "road_steep_cost",
}


def _construct(cls: type, data: dict) -> "WorldConfig":
    """Build a config from a loaded mapping, with a readable error for a bad key.

    Without this an unknown key reaches the dataclass constructor and raises `TypeError`,
    which the CLI's `except ValueError` handlers do not catch — so a single typo in a YAML
    file surfaced as a raw traceback rather than a message. Every other config error is a
    `ValueError`; this makes an unrecognised key one too.
    """
    import warnings

    for old in sorted(set(data) & set(_RENAMED_FIELDS)):
        new = _RENAMED_FIELDS[old]
        warnings.warn(
            f"Config setting {old!r} has been renamed to {new!r}; carrying the value over.",
            DeprecationWarning,
            stacklevel=3,
        )
        data.setdefault(new, data.pop(old))
        data.pop(old, None)

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
