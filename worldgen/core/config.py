import json
from dataclasses import asdict, dataclass, fields
from typing import Any


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

    # Terrain classification — bands of gradient, in metres of rise per kilometre.
    # Absolute, not a fraction of the elevation range: the old fractional thresholds moved
    # with road_elev_range_m, so what counted as a mountain was 120 m/km at the 3000 m
    # default and 20 m/km on a 500 m map, where it called rolling downland a peak.
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
    river_flow_threshold: float = 0.05
    river_flow_continuous: bool = False  # True: river_flow on all draining land hexes
    moisture_bleed_passes: int = 0  # 0 = flat river bonus (default); >0 = elevation-gated bleed
    moisture_bleed_strength: float = 0.3

    # Lake drainage
    lake_chaining: bool = True  # Let a lake spill into a strictly lower lake, not only the sea
    endorheic_marsh_radius: int = 1  # Shore band (hexes) turned to wetland around a closed basin
    endorheic_marsh_min_moisture: float = 0.40  # Below this a closed basin is arid, not marshy

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
        if not (0.0 <= self.navigable_river_flow <= 1.0):
            raise ValueError(
                f"navigable_river_flow must be in [0, 1], got {self.navigable_river_flow}"
            )
        if not (0.0 < self.marketable_surplus_fraction <= 1.0):
            raise ValueError(
                "marketable_surplus_fraction must be in (0, 1], got "
                f"{self.marketable_surplus_fraction}"
            )
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
    navigable_river_flow: float = 0.35
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
    biome_alpine_elev: float = 0.85
    biome_cold_temp: float = 0.25
    biome_warm_temp: float = 0.6
    biome_dry_moist: float = 0.2
    biome_wet_moist: float = 0.5

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
    # planting once the best remaining site scores below this. Calibrated against ~70-85
    # markets at 128x128 (England had ~700 markets in ~130,000 km2; this map is about an
    # eighth of that).
    market_viability_floor: float = 5.0

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
    road_escarpment_cost: float = 20.0
    road_steep_cost: float = 10.0
    road_rolling_cost: float = 3.0
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
_RETIRED_FIELDS: dict[str, str] = {
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
