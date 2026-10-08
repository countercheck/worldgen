# Worldgen Reference

How every calculation works, and what every value does. The code is the
ground truth — every non-trivial formula and number cited here links to its
source line.

This doc is split for two audiences:

- **§3 Pipeline & Algorithms** — for extending or debugging stages. Each
  stage is described with reads/writes, the algorithm in plain English, and
  every key formula tagged with `file:line`.
- **§4 Configuration Reference** — for tuning a world without reading code.
  Tables of `WorldConfig` parameters with defaults, ranges, and effects.

§5 catalogues the magic numbers that sit *outside* `WorldConfig` but still
shape every map.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Data Model](#2-data-model)
3. [Pipeline & Algorithms](#3-pipeline--algorithms)
   - [3.1 Elevation](#31-elevation)
   - [3.1a Elevation from an Image](#31a-elevation-from-an-image)
   - [3.2 Erosion](#32-erosion)
   - [3.3 Terrain Classification](#33-terrain-classification)
   - [3.4 Water Bodies](#34-water-bodies)
   - [3.5 Hydrology](#35-hydrology)
   - [3.6 Climate](#36-climate)
   - [3.7 Biomes](#37-biomes)
   - [3.8 Land Cover](#38-land-cover)
   - [3.9 Habitability](#39-habitability)
   - [3.10 City & Town Placement](#310-city--town-placement) — `classic`
   - [3.10a River Crossings](#310a-river-crossings--organic) — `organic`
   - [3.10b Market Centres](#310b-market-centres--organic) — `organic`
   - [3.11 Interurban Roads](#311-interurban-roads)
   - [3.12 Cultivation (Cities & Towns)](#312-cultivation-cities--towns)
   - [3.13 Village Placement](#313-village-placement)
   - [3.14 Village Tracks](#314-village-tracks)
   - [3.15 Village Cultivation](#315-village-cultivation)
   - [3.16 Naming](#316-naming)
4. [Configuration Reference](#4-configuration-reference)
5. [In-Code Constants](#5-in-code-constants)
6. [Outputs](#6-outputs)
7. [Glossary](#7-glossary)

---

## 1. Overview

### Scale

- **1 hex = 1 km**, set by `hex_size_m`. The pipeline assumes the kilometre-scale
  interpretation throughout — settlement separation, cultivation radii, and every
  haulage range are quoted in kilometres because a hex is one.
- **Every physical field is in real units.** Elevation is metres above sea level,
  temperature degrees Celsius, rainfall millimetres a year, catchment area square
  kilometres, gradient metres per kilometre. None of them are normalised, so a threshold
  written against one means the same thing on every map. See § [4](#4-configuration-reference).
- **Default grid:** 200 × 200 on the `offset` layout (≈40,000 km², a large kingdom or a small country).
- **Coordinates:** axial `(q, r)`, flat-top hexagons.
  Neighbours, distance, ranges, and pixel conversion live in
  [worldgen/core/hex_grid.py](../worldgen/core/hex_grid.py).
  Hex distance uses the standard axial/cube hex-distance formula:
  `(|Δq| + |Δr| + |Δq+Δr|) // 2` ([hex_grid.py:26](../worldgen/core/hex_grid.py#L26)).

### Reproducibility

A single integer seed reproduces every world bit-for-bit. Mechanism:

1. The pipeline holds a parent `numpy.random.Generator` seeded from the CLI
   `--seed` ([pipeline.py:31](../worldgen/core/pipeline.py#L31)).
2. Before each stage runs, the parent generator draws a fresh 32-bit integer
   and seeds a **child** `Generator` for that stage
   ([pipeline.py:50](../worldgen/core/pipeline.py#L50)).
3. Stages only ever use their child RNG — never global `numpy.random` or
   Python's `random`. Container iteration (`set`, `dict.items()`) is
   sorted before any random choice that depends on order, so insertion-order
   nondeterminism cannot leak into output.

The seed and full config are written into `WorldState.metadata` at the
start of the run ([pipeline.py:46–47](../worldgen/core/pipeline.py#L46))
and round-trip through `world.json`.

### Pipeline

The stage list is defined once, in
[worldgen/stages/\_\_init\_\_.py](../worldgen/stages/__init__.py), and both the CLI and
the test fixtures read it from there. Each stage is a pure transformer: `state → state`
([pipeline.py:20](../worldgen/core/pipeline.py#L20)). Stages never write files; that's
`worldgen/export/`'s job.

**There are two settlement models**, selected with `generate --model`. They share the nine
physical stages and diverge after them:

- **`classic`** ranks hexes on habitability and places a *configured number* of
  cities and towns at a fixed minimum separation, then sprinkles villages. Population is
  drawn at random from a per-tier band and nothing about the site enters the number.
- **`organic`** (default) derives the hierarchy from pre-industrial haulage economics, and each of
  its three tiers is a different question about reach. Markets go where the most surplus
  can reach them inside a **day's return**; cities are markets that other *markets* can
  reach over `haulage_range_land` with water counting fifteen times; villages stand where
  the road has no way round — a bridgehead or a pass — and are sized from a **morning's
  walk** out to the fields. None of the three is a target count.

  Under all three sits a **soil** model (§ [3.7a](#37a-soil)): what the ground could
  support, kept separate from what grows on it and from what is done with it. Good soil
  carries wildwood until somebody clears it, cleared ground yields more than wood, and
  where clearing stops is set by scarcity rather than by a radius
  (§ [3.10c](#310c-land-use-clearing-and-rural-density--organic)) — so the map shows
  assarted country round the markets and wood standing in the gaps.

  It runs none of `classic`'s village stages: the dispersed peasantry is a productive
  surface rather than a list of hamlets (§ [3.10b](#310b-market-centres--organic)), and it
  now carries an explicit population — about 60 people per km² on a temperate map at the
  c. 1800 defaults, against England and Wales's 59 in the 1801 census. A temperate map carries ~80 settlements where `classic` carries
  ~1,100. See § [3.10a](#310a-river-crossings--organic),
  § [3.10b](#310b-market-centres--organic),
  § [3.10c](#310c-land-use-clearing-and-rural-density--organic),
  § [3.10d](#310d-resource-settlements--organic) and
  § [3.11a](#311a-chokepoints--organic).

The difference is not cosmetic. On one landlocked desert map at 128×128, `classic` places
a city of 48,000 — and its five largest populations are identical, figure for figure, to
those it produces on a fertile temperate coast, because they come from `rng.integers` and
the land tells them nothing. `organic` caps the same map at 2,706 and rings its
settlements around the shore of the inland sea.

The diagram below shows the `classic` pipeline. `SoilStage` runs before `LandCoverStage` in both models. `organic`
replaces `CityTownStage` with `CrossingStage → MarketStage → LandUseStage →
CityPromotionStage → ResourceStage`, inserts `ChokepointStage` after `InterurbanRoadStage`, and drops
`CultivationStage` — `LandUseStage` does that job and more.

```mermaid
flowchart TD
    Start([seed + WorldConfig]) --> Elev

    subgraph Terrain["Terrain &amp; Hydrology"]
        Elev[ElevationStage<br/><i>noise → hex.elevation</i>]
        Eros[ErosionStage<br/><i>carves channels in hex.elevation</i>]
        Tcls[TerrainClassificationStage<br/><i>hex.terrain_class</i>]
        Wbod[WaterBodiesStage<br/><i>splits OCEAN vs LAKE; fixes COAST</i>]
        Hydr[HydrologyStage<br/><i>state.rivers, river_sides, river_corners</i>]
        Cata[CataractStage<br/><i>cataract / rapids side tags</i>]
        Elev --> Eros --> Tcls --> Wbod --> Hydr --> Cata
    end

    subgraph Climate["Climate &amp; Cover"]
        Clim[ClimateStage<br/><i>hex.temperature, hex.moisture</i>]
        Biom[BiomeStage<br/><i>hex.biome + WETLAND override</i>]
        Lcov[LandCoverStage<br/><i>hex.land_cover</i>]
        Clim --> Biom --> Lcov
    end

    subgraph Settle["Settlements &amp; Roads"]
        Hab[HabitabilityStage<br/><i>habitability_city/town/village ∈ [0,1]</i>]
        Cit[CityTownStage<br/><i>cities + towns; pass tags</i>]
        Iur[InterurbanRoadStage<br/><i>PRIMARY/SECONDARY + habitability_village +0.2</i>]
        Cul[CultivationStage<br/><i>hex.cultivated near cities/towns</i>]
        Vil[VillagePlacementStage<br/><i>villages on frontier / near roads</i>]
        Vtk[VillageTrackStage<br/><i>TRACK roads to network</i>]
        Vcul[VillageCultivationStage<br/><i>hex.cultivated near villages</i>]
        Hab --> Cit --> Iur --> Cul --> Vil --> Vtk --> Vcul
    end

    Cata --> Clim
    Lcov --> Hab
    Vcul --> End([WorldState])

    classDef terrain fill:#e8d5b7,stroke:#8b6f47,color:#3a2e1c
    classDef climate fill:#cfe8d4,stroke:#5a8a6f,color:#1c3a2e
    classDef settle fill:#d5d8e8,stroke:#5a6f8a,color:#1c2a3a
    classDef io fill:#f5f5f5,stroke:#666,color:#222

    class Elev,Eros,Tcls,Wbod,Hydr terrain
    class Clim,Biom,Lcov climate
    class Hab,Cit,Iur,Cul,Vil,Vtk,Vcul settle
    class Start,End io
```

---

## 2. Data Model

### `WorldState` — [worldgen/core/world_state.py](../worldgen/core/world_state.py)

Schema version 2.0. A `world.json` from before rivers moved onto hexsides is refused with
a message to regenerate it from its seed; there is no migration.

| Field | Type | Notes |
|---|---|---|
| `seed` | `int` | The RNG seed for this run |
| `width`, `height` | `int` | Grid dimensions in hexes |
| `layout` | `str` | `"axial"` or `"offset"` — how `width`/`height` map onto hex coordinates |
| `hexes` | `dict[HexCoord, Hex]` | Every cell, keyed by axial `(q, r)` in either layout |
| `rivers` | `list[River]` | Courses along hexsides: `River.corners` from upstream to downstream, plus `flow_volume` and `name`. A tributary's last corner is a corner of its trunk |
| `river_sides` | `dict[Side, RiverSide]` | Every hexside a river runs along: `catchment_km2`, `flow` (a 0–1 rank against the largest river), `drop_m`, and `tags` ⊂ {`ford`, `bridge`, `cataract`, `rapids`} |
| `river_corners` | `dict[Corner, set[str]]` | Tags on the points of the network: `river_source`, `river_source_offmap`, `river_end`, `river_mouth`, `confluence` |
| `settlements` | `list[Settlement]` | Cities, towns, and villages combined |
| `road_edges` | `dict[(HexCoord, HexCoord), RoadEdge]` | The road network, one tier (PRIMARY / SECONDARY / TRACK) per undirected land edge |
| `sea_edges` | `dict[(HexCoord, HexCoord), RoadEdge]` | The water legs of the same network, kept apart from the roads |
| `ferries` | `list[Ferry]` | Kept in the schema, but nothing fills it since rivers moved onto hexsides: no land is sealed off by a river any more |
| `metadata` | `dict` | `{"seed": ..., "config": ...}` snapshot |

Convenience accessors: `all_land()`, `all_open_water()`, `all_inland_water()`,
`all_water()` ([world_state.py:49–79](../worldgen/core/world_state.py#L49)).

### `Hex` — [worldgen/core/hex.py:66–80](../worldgen/core/hex.py#L66)

| Field | Type | Range | Written by |
|---|---|---|---|
| `coord` | `HexCoord` (`(q, r)`) | — | construction |
| `elevation` | `float` | `[0.0, 1.0]` after normalization | Elevation, Erosion, Hydrology (lake fill) |
| `moisture` | `float` | `[0.0, 1.0]` | Climate |
| `wet_season_precip_mm`, `dry_season_precip_mm` | `float` | mm; sum to `moisture` | Climate (`wet_season_share`) |
| `temperature` | `float` | `[0.0, 1.0]` (clamped) | Climate |
| `terrain_class` | `TerrainClass` | enum | Terrain Class, Water Bodies, Hydrology |
| `biome` | `Biome \| None` | enum | Biome |
| `land_cover` | `LandCover \| None` | enum | Land Cover |
| `river_flow` | `float` | `[0.0, 1.0]`, normalized to map max; written on *both* banks of every river side | Hydrology |
| `habitability_city` | `float` | `[0.0, 1.0]` | Habitability (catchment radius 8) |
| `habitability_town` | `float` | `[0.0, 1.0]` | Habitability (catchment radius 4) |
| `habitability_village` | `float` | `[0.0, 1.0]` | Habitability (catchment radius 2), Roads (+0.2 near roads) |
| `settlement` | `Settlement \| None` | — | City/Town, Village |
| `road_connections` | `set[HexCoord]` | adjacent cells with roads | Interurban Roads, Village Tracks |
| `cultivated` | `bool` | — | Cultivation, Village Cultivation |
| `tags` | `set[str]` | — | many stages; vocabulary below |

### Enums

- **`TerrainClass`** — `OCEAN, LAKE, COAST, FLAT, HILL, MOUNTAIN`
  ([hex.py:7–13](../worldgen/core/hex.py#L7)).
- **`Biome`** — `TUNDRA, BOREAL, TEMPERATE_FOREST, GRASSLAND, SHRUBLAND,
  DESERT, TROPICAL, WETLAND, OCEAN, ALPINE`
  ([hex.py:30–40](../worldgen/core/hex.py#L30)).
- **`LandCover`** — `OPEN_WATER, BOG, MARSH, DENSE_FOREST, WOODLAND, SCRUB,
  OPEN, TUNDRA, DESERT, ALPINE, BARE_ROCK`
  ([hex.py:16–27](../worldgen/core/hex.py#L16)).
- **`SettlementTier`** — `CITY, TOWN, VILLAGE`
  ([hex.py:52–55](../worldgen/core/hex.py#L52)).
- **`SettlementRole`** — `AGRICULTURAL, PORT, MARKET` (`MINING` and `FORTRESS` retired)
  ([hex.py:205–216](../worldgen/core/hex.py#L205)).
- **`RoadTier`** — `PRIMARY, SECONDARY, TRACK`
  ([world_state.py:7–10](../worldgen/core/world_state.py#L7)).

### Tags Vocabulary (`Hex.tags`)

| Tag | Meaning | Set by |
|---|---|---|
| `"prominent_site"` | ROLLING hex that is a local-max `habitability_town` within 3-hex range, no settlement | City/Town (`classic`) |
| `"pass"` | A col: a saddle whose flanks both rise at least `terrain_steep_gradient_m` | Chokepoints (`organic`) |
| `"confluence_town"` | TOWN settled on a hex at a `confluence` corner | City/Town |
| `"hollow"` | Land in a closed hollow too small or shallow for a lake, or a small island in a lake; waterlogs to wetland | Water bodies |

No river feature is a hex tag. Rivers run along hexsides (see
[ARCHITECTURE.md](ARCHITECTURE.md#river-model)), so what is true of a stretch of river is
tagged on its side and what is true of a point on its corner.

**Side tags** (`RiverSide.tags`, in `state.river_sides`):

| Tag | Meaning | Set by |
|---|---|---|
| `"ford"` | The river can be waded here: every side with span ≤ 1, plus rare fords on barge rivers (§3.10a) | Crossing (`organic`), `tag_river_crossings` |
| `"bridge"` | A bridge, and always one a road crosses: Crossing marks candidate sites, and `tag_river_crossings` keeps those a road uses and adds one wherever a road crosses water too big to wade | Crossing, `tag_river_crossings` |
| `"cataract"` | A barge river falling too steeply to navigate; breaks a reach, so cargo portages | Cataracts |
| `"rapids"` | White water on a river too small for a cataract: see `rapids_min_drop_m` | Cataracts |

**Corner tags** (`state.river_corners`):

| Tag | Meaning |
|---|---|
| `"river_source"` | First corner of a drawn river that rises on the map — a spring, not water arriving from off it |
| `"river_source_offmap"` | First corner of a river that enters across the map edge |
| `"river_end"` | Last corner of a river that stops without joining another: into the sea or a lake, off the map, or into the ground |
| `"river_mouth"` | A `river_end` where the water runs into the sea or a lake |
| `"confluence"` | Where a tributary joins its trunk. Inland only: two rivers reaching the sea at one corner have not met |

All are set by Hydrology. A hex is "at" a corner feature if it is one of the corner's
three land hexes, which is how `riverside.river_index` hands them to habitability, city
roles and naming.

A road never stands in a river: it runs beside one on a bank, and crosses it by stepping
across a side, which costs `road_river_crossing_base + road_river_crossing_flow × flow`
once. `tag_river_crossings` then tags each side a road crosses (§3.11).

---

## 3. Pipeline & Algorithms

### 3.1 Elevation

[stages/elevation.py](../worldgen/stages/elevation.py)

**Purpose:** Generate the base heightmap from layered noise.

**Reads:** nothing (works from an empty `WorldState`).
**Writes:** `hex.elevation` for every hex.

**Config:** `noise_octaves`, `noise_persistence`, `noise_lacunarity`,
`noise_scale`, `domain_warp_strength`, `continent_falloff`,
`continent_falloff_edges`, `continent_shelf_hexes`, `continent_shelf_variance`,
`max_elevation_m`, `seabed_depth_m`, `elevation_gradient_m`.

**Algorithm**

1. Two independent OpenSimplex generators are seeded from the stage's RNG —
   one for the base height field, one for **domain warping**
   ([elevation.py:13–16](../worldgen/stages/elevation.py#L13)).
2. Each grid coordinate is offset by the warp generator before sampling
   the base field. This breaks up the visible "noise grain" and produces
   more organic coastlines:
   ```
   warp_x = warp.noise(q, r)         * domain_warp_strength
   warp_y = warp.noise(q+100, r+100) * domain_warp_strength
   nx = q + warp_x;  ny = r + warp_y
   ```
   ([elevation.py:23–28](../worldgen/stages/elevation.py#L23)). The
   `+100` offset on the y warp ensures the two channels are independent
   samples of the same generator.
3. **Fractal Brownian motion (fBm)** sums `noise_octaves` octaves of the
   base noise:
   ```
   for j in range(octaves):
       v += noise(nx * lacunarity^j, ny * lacunarity^j) * persistence^j
   elevation = v / sum(persistence^j)   # normalize to [-1, 1]
   ```
   ([elevation.py:31–43](../worldgen/stages/elevation.py#L31)). Higher
   `persistence` keeps more detail in late octaves (rougher); higher
   `lacunarity` increases frequency between octaves (more high-frequency
   detail).
4. **Linear stretch to `[0, 1]`.** This shaping stays, but only as scaffolding for the
   falloff: the falloff blends *towards* the seabed and needs a known floor to blend
   from. Applied to raw noise, which straddles zero, the map edge came out mid-range
   instead of underwater, so whether the sea reached the border at all was a coin flip
   per seed — and on a map where it did not, every drop of water was trapped inland.
5. **Continent falloff** (optional) sinks the map's edges to the seabed over a shelf
   `continent_shelf_hexes` wide. Only the edges named in `continent_falloff_edges`
   participate, so land can run off the map on the others — a world that continues past
   the border. Three details matter:
   - The two axes combine with a **p-norm, not a minimum**. A minimum holds the shelf at
     constant width right up to where two edges meet, giving a square corner; the p-norm
     pulls the corner inward so headlands round off.
   - The shelf's inner boundary wanders by `continent_shelf_variance`, applied
     *multiplicatively* to a value already zero at the border — so however far the
     coastline swings inland, the outermost ring stays underwater and the sea still
     reaches the edge.
   - The ramp is a **smoothstep**, easing at both ends, so the coast varies with the
     noise behind it rather than being a uniform wall. Where that noise is high the drop
     is still abrupt, which is what a sea cliff is.
6. **Into metres above sea level.** `elevation = shaped × (max_elevation_m +
   seabed_depth_m) − seabed_depth_m`. Sea level is the datum: land is positive, the sea
   floor negative, and zero means sea level by definition rather than by a threshold.
7. **Regional tilt** (optional) adds `elevation_gradient_m` as `[east, south]` metres
   across the map, centred on `[-0.5, +0.5]`. It goes on in metres, after the conversion. It used to
   run before the shaping, where a normalisation promptly stretched the result back out,
   so asking for half a range of tilt got you rather less than that.

8. **Elevation profile** (`apply_profile`, unless `elevation_profile: none`). The land
   hexes are ranked by height and each is given the height at the same rank of a
   log-normal with the profile's mode and median, cut off at `max_elevation_m`. Ranks are
   kept, so ridges, valleys and the coastline stay where the noise put them; only the
   share of land at each height changes. Droplet erosion then takes off a share of every
   height, so `ErosionStage` resets sea level after the droplets to keep the land share
   that went in, and applies the profile again at the end: the final heights match it
   exactly.

**Gotchas**

- Elevations are absolute. `450` means 450 m, not a percentile — every downstream test
  ("is this ocean", "how far above the water does this stand", "is it above the
  treeline") is a statement about the world rather than a position on a per-map axis.
  There is no `sea_level` setting to raise; set `max_elevation_m` and `seabed_depth_m`.
- Without `continent_falloff` (or with `continent_falloff_edges: []`), expect a
  near-total land map whose interior basin is the terminal sink — endorheic by geometry.
- The shelf width is capped at a quarter of the shorter side. On a map smaller than the
  shelf there is no interior left to be a continent and the whole thing sinks; real maps
  never hit this.

---

### 3.1a Elevation from an Image

[stages/image_elevation.py](../worldgen/stages/image_elevation.py),
[export/heightmap_import.py](../worldgen/export/heightmap_import.py)

**Purpose:** Take the terrain from a picture instead of generating it.

**Reads:** nothing from `WorldState`; the image named by `heightmap_path`.
**Writes:** `hex.elevation` for every hex — the same contract as `ElevationStage`.

**Config:** `heightmap_path`, `heightmap_mode`, `heightmap_land_threshold`,
`heightmap_invert`, `heightmap_coast_falloff`, plus `sea_level` and
`continent_shelf_hexes` in coastline mode.

Selected by `stages.stages_for(config)`, which substitutes this stage for
`ElevationStage` when `heightmap_path` is set. The substitution is positional and
keeps the stage count, so every later stage draws the same child RNG it would
have in a generated world.

**Algorithm**

1. `export/heightmap_import.load_luminance` reads the file — the only file I/O in
   the feature, kept in the layer that is allowed to do it. It branches on the
   Pillow mode *before* converting: `convert("L")` clamps a 16-bit image at 255
   rather than rescaling it, which would silently reduce a real DEM to a
   silhouette. Alpha comes back only when the band actually varies.
2. The pixels are **area-averaged** onto the grid, stretched to fill it on each
   axis independently. Each hex takes the mean of the pixels its footprint covers,
   computed exactly via prefix sums in `O(n + m)`, so downsampling a large image
   does not alias a coastline into a staircase. Image row 0 maps to grid row 0 —
   both are north, so there is no flip.
3. In `elevation` mode that resampled luminance *is* the terrain, mapped linearly
   from `0..255` (or `0..65535`) onto `[0, 1]`.
4. In `coastline` mode the image is reduced to a land/sea mask instead — by alpha
   where it is meaningful, otherwise by `heightmap_land_threshold`. The mask is
   resampled as floats and re-thresholded at 0.5, which antialiases it rather than
   dropping islands and pinching straits. `noise_field` from `ElevationStage` then
   supplies the heights, and `shape_to_mask` fits them to the stencil: land ramps
   from just above `sea_level` up into the noise's full range over
   `continent_shelf_hexes` inland, and sea ramps from just below `sea_level` down
   to the deep floor, both eased with the same smoothstep the continent falloff
   uses. The result is stretched to fill `[0, 1]` with sea level pinned in place.

**Why the range is filled.** `ErosionStage` ends by renormalising whatever range it
is handed onto `[0, 1]`, which moves `sea_level` relative to the terrain. A
coastline field is built against a *fixed* sea level, so without this it would have
its whole shelf dragged under: measured on a 96×83 import, a stencil covering 32% of
the map came out of the full pipeline at 2%, the continent broken into specks.
Filling the range makes that renormalisation a no-op, and erosion then carves the
imported terrain exactly as it does a generated one.

**Notes**

- `elevation` mode gets no such protection, by design — it has no fixed reference to
  preserve. Erosion will stretch a low-contrast heightmap's contrast and move its
  coast. Use `worldgen import-heightmap`, which skips erosion, when the image must be
  reproduced exactly.
- The range is re-anchored again after `heightmap_coast_falloff`, since that blend moves
  the waterline and takes the field back off `[0, 1]`. Without it the opt-in path walks
  straight back into the renormalisation the anchoring exists to defuse.
- Nothing forces an imported field below `sea_level`, so a stencil with no sea in it
  produces a world with no ocean. "Rivers reach the ocean" stops being an invariant
  there, and the range cannot be anchored at both ends — no all-land field both stays
  above `sea_level` and reaches zero — so erosion will flood part of the map. The stage
  warns.
- Alpha is only taken as the stencil when a real share of the image (≥1%) is transparent.
  A single antialiased or lossily-round-tripped pixel would otherwise outrank the
  brightness threshold and carry the whole map with it.
- Mode `"I"` (32-bit integer) is the one ambiguous case: a DEM stored in metres is
  indistinguishable from 16-bit-scaled data, and `0–3000 / 65535` is a map entirely below
  sea level. Scaling by the observed peak would be the histogram stretch this importer
  refuses to do, so a suspiciously low peak warns instead.
- An image smaller than the grid is upsampled by replication — a box filter has
  nothing else to do — giving blocks of equal elevation that read as `FLAT` to terrain
  classification. The stage warns.

---

### 3.2 Erosion

[stages/erosion.py](../worldgen/stages/erosion.py)

**Purpose:** Sculpt valleys by simulating water particles flowing downhill,
removing high-frequency noise, and producing natural-looking channels.

**Reads:** `hex.elevation`.
**Writes:** `hex.elevation`, still in metres above sea level.

**Config:** `erosion_droplets_per_hex`, `erosion_inertia`, `erosion_capacity`,
`erosion_deposition`, `erosion_erosion_rate`, `erosion_channel_affinity_gain`,
`erosion_affinity_update_interval`, `erosion_delta_min_load`, `max_elevation_m`,
`seabed_depth_m`.

**Units.** The erosion constants are *shares of the map's relief* rather than physical
quantities: `erosion_capacity` multiplies a height difference, and the capacity floor and
`erosion_delta_min_load` are absolute heights, all tuned against a `[0, 1]` range. Fed
metres directly they become centimetres, a droplet's capacity collapses to nothing, every
droplet deposits, and the whole map planes off to sea level within a couple of passes. So
the stage converts to a normalised copy at its boundary and back on the way out —
against the **known** span `max_elevation_m + seabed_depth_m`, so it is a fixed change of
units, not a per-map stretch.

**Dose.** Droplets run **per land hex**, not per map. A flat count is a different amount
of weather depending on map size: at the old default of 15,000, a 32×32 map got 14.6
droplets per hex and a 128×128 map 0.9 — a sixteenfold spread, and most of why small maps
came out as Alpine massifs while the default map stayed a barely-touched noise field. Per
*land* hex specifically, so a mostly-ocean map does not have its weather spread thinner
over what land it has.

**Algorithm** (particle-based hydraulic erosion, JIT-compiled with numba
when available — falls back to pure Python).

For each of `round(erosion_droplets_per_hex × land_hexes)` particles, drop one at a
randomly chosen land hex and simulate up to `_MAX_STEPS = 64` steps of flow
([erosion.py:18, 42](../worldgen/stages/erosion.py#L18)):

1. **Compute local gradient** from 4 neighbours (clamped at edges):
   ```
   gx = (right - left) * 0.5
   gy = (down  - up)   * 0.5
   ```
   ([erosion.py:52–57](../worldgen/stages/erosion.py#L52)).
2. **Update direction** with momentum:
   ```
   dir = inertia * dir_prev - (1 - inertia) * gradient
   ```
   ([erosion.py:59–60](../worldgen/stages/erosion.py#L59)). `inertia` near
   0 = pure gradient descent; near 1 = particle ignores terrain. Default
   `0.05` keeps channels mostly aligned with steepest descent but
   smooths sharp turns.
3. **Move one cell** along the normalised direction
   ([erosion.py:65–69](../worldgen/stages/erosion.py#L65)).
4. **Sediment transport**:
   ```
   dh       = elev[next] - elev[here]              # negative = downhill
   capacity = max(-dh, 0.01) * speed * water * erosion_capacity
   if sediment > capacity:
       deposit  = erosion_deposition * (sediment - capacity)
       arr[here] += deposit;  sediment -= deposit
   else:
       erode = min(erosion_erosion_rate * (capacity - sediment), |dh| if dh<0 else 0)
       arr[here] -= erode;    sediment += erode
       channel_affinity[here] += erosion_channel_affinity_gain
   ```
   ([erosion.py:75–87](../worldgen/stages/erosion.py#L75)).
   The `0.01` floor on `-dh` prevents capacity from collapsing to 0 on
   flats, which would freeze sediment in place.
5. **Update speed/water** between steps:
   ```
   speed = max(speed + dh, 0.01)
   water *= 0.99   # _EVAPORATION
   ```
   ([erosion.py:89–90](../worldgen/stages/erosion.py#L89)). Particles
   accelerate downhill, decelerate uphill, and gradually evaporate so
   they cannot dig forever.
6. **Termination** when the particle leaves the grid, drops below sea
   level (deposits remaining sediment), or stalls below `1e-8` direction
   magnitude
   ([erosion.py:42–66](../worldgen/stages/erosion.py#L42)).

**Channel affinity** — a self-reinforcing trick. Every
`erosion_affinity_update_interval` particles, the spawn distribution is
re-weighted by `channel_affinity` so later particles tend to start in
already-eroded channels, deepening them
([erosion.py:136–143](../worldgen/stages/erosion.py#L136)). With the
default `affinity_update_interval=500`, the first 500 particles spawn
uniformly to discover channels, then later batches reinforce them.

**Deltas.** A droplet reaching the sea with at least `erosion_delta_min_load` still
aboard spreads it as a fan with a sharp radial falloff (weights 0.6 / 0.3 / 0.1 over three
rings), never lifting anything above the waterline. Emptying the whole load into the
single hex of entry built isolated spikes, and since droplets cross the waterline wherever
they happen to reach it, those spikes smeared along the entire coastline — only a third of
infilled sea hexes were within three hexes of a river mouth and a fifth were more than
twenty away. Fanning lets the many droplets funnelled down one channel superpose into a
delta at its mouth, while a lone droplet off a hillside leaves almost nothing.

**Post-process**

1. Gaussian blur with `sigma=0.5` to remove single-cell artefacts.
2. Convert back to metres against the known span.

There is deliberately **no re-stretch to `[0, 1]`** here. It would undo the datum, putting
the lowest point of the eroded map at the seabed and the highest at the peak whatever
erosion had actually done to either. Sea level has to stay where it is for the word to
mean anything, and a landscape that has been worn down should read as worn down rather
than being scaled back up to fill the range it started with.

**Gotchas**

- Erosion is the bottleneck of the pipeline. Halving `erosion_droplets_per_hex` ≈ halves
  total runtime, but it is not a cosmetic knob: it decides whether the map has valleys at
  all. Below about one droplet per hex the rivers only scratch a line into the noise and
  there is no floodplain. It is also a **climate** setting, because the orographic term
  lifts on height above sea level — wearing the high ground down flattens the rain
  shadow. `3.0` has floodplains and keeps most of the shadow.
- Without numba, this stage is roughly 10× slower; install numba
  (`pip install numba`) for full speed.

---

### 3.3 Terrain Classification

[stages/terrain_class.py](../worldgen/stages/terrain_class.py)

**Purpose:** Bucket every hex into `OCEAN / COAST / FLAT / ROLLING / STEEP / ESCARPMENT`.

**The classes are bands of gradient, not of altitude.** They describe how the ground
*lies*, so a high plateau is level ground and is classed as such. The old rule made
anything above 0.8 of the elevation range a mountain regardless of slope, which put nearly
a third of a 128×128 map's "mountain" hexes on ground gentler than 75 m/km — upland basins
that are perfectly walkable and farmable, but were priced at ten times flat ground for
roads and refused settlement outright. There is now no altitude term at all. Where
altitude genuinely matters it is read directly: the treeline from `biome_treeline_temp_c`,
and mine workings from a settlement's own elevation.

**Reads:** `hex.elevation`, neighbours.
**Writes:** `hex.terrain_class`.

**Config:** `coast_max_elevation_m`, `terrain_rolling_gradient_m`,
`terrain_steep_gradient_m`, `terrain_escarpment_gradient_m`.

**Algorithm** ([terrain_class.py](../worldgen/stages/terrain_class.py)):

```
# Pass 1 — sea level is the datum, so this is not a configured threshold
for hex in all hexes:
    if hex.elevation < 0.0:
        hex.terrain_class = OCEAN

# Pass 2
for hex in non-ocean hexes:
    if hex.elevation < coast_max_elevation_m and any neighbour is OCEAN:
        hex.terrain_class = COAST
        continue

    gradient = tilt(hex)                              # m/km, see below
    if   gradient >= terrain_escarpment_gradient_m:  ESCARPMENT
    elif gradient >= terrain_steep_gradient_m:       STEEP
    elif gradient >= terrain_rolling_gradient_m:     ROLLING
    else:                                            FLAT
```

**Gradient is measured as tilt**, the steepest fall across the hex taken over the three
pairs of *opposite* neighbours (which are two kilometres apart, hence a halving). At
1 hex = 1 km with elevation in metres this is a gradient in the ordinary sense, with
nothing to convert.

The obvious alternative — mean absolute difference to all six neighbours — answers a
different question, and the wrong one. It reports how rough the *surroundings* are rather
than how the ground underfoot lies, so it calls a valley floor steep: the valley sides
stand above it on both flanks, and their height enters the mean whatever the floor is
doing. Rivers run along valley floors, which is how that version came to price river
corridors as mountain and drove roads *away* from the banks they should follow.

Tilt cancels symmetric surroundings, which is what makes it right. A valley floor reads
level because both flanks rise equally; so does a ridge crest, because both fall equally
— and a crest is walkable along, whatever the drop either side. A hillside reads steep,
because uphill and downhill neighbours genuinely differ.

**Notes**

- Inland water created by Erosion lows but never connecting to a map edge is classified
  OCEAN here; the next stage corrects that.

---

### 3.4 Water Bodies

[stages/water_bodies.py](../worldgen/stages/water_bodies.py)

**Purpose:** Distinguish OCEAN (map-edge-connected) from inland LAKE, fill the
closed hollows on land, and fix COAST hexes that ended up adjacent only to a lake.

**Reads:** `hex.terrain_class` (OCEAN from previous stage).
**Writes:** `hex.terrain_class` (some OCEAN → LAKE; some COAST →
`HILL/FLAT/MOUNTAIN`).

**Algorithm** ([water_bodies.py:21–39](../worldgen/stages/water_bodies.py#L21)):

1. Collect every hex marked OCEAN. BFS over OCEAN-OCEAN adjacency to
   discover connected components ([water_bodies.py:42–52](../worldgen/stages/water_bodies.py#L42)).
2. For each component: if any hex sits on the map border, leave it as
   OCEAN. Otherwise, reclassify every hex in the component to LAKE
   ([water_bodies.py:33–36](../worldgen/stages/water_bodies.py#L33)).
3. **`_fill_hollows`**: a Priority-Flood from the sea, the lakes and the map edge finds the
   level water would have to rise to before it could run out of each hex. A connected
   run of land under that level is a hollow. One at least `lake_min_hexes` across and
   `lake_min_depth_m` deep, in a region shedding `lake_min_runoff_mm`, becomes LAKE — it
   stands at its rim, so it overflows there and hydrology gives it an outlet. A smaller
   one at least `hollow_wetland_min_depth_m` deep, and any island smaller than
   `lake_min_hexes` inside a lake, is tagged `hollow` for `BiomeStage` to waterlog.
4. **`_fix_coast_hexes`** ([water_bodies.py:60–101](../worldgen/stages/water_bodies.py#L60)):
   COAST was assigned earlier based on adjacency to OCEAN, but some of
   those neighbours are now LAKE. For each COAST hex that has no actual
   ocean neighbour:
   - If it sits beside a lake at low elevation, *keep* COAST (acts as a
     lake shore for downstream stages).
   - Otherwise re-run the gradient classification
     (`FLAT/ROLLING/STEEP/ESCARPMENT`).

   This pass reads the terrain gradient bands from `state.metadata["config"]` — that's
   why `pipeline.run()` snapshots config into metadata at startup.

---

### 3.5 Hydrology

[stages/hydrology.py](../worldgen/stages/hydrology.py),
[stages/hydrology_fill.py](../worldgen/stages/hydrology_fill.py),
[stages/hydrology_rivers.py](../worldgen/stages/hydrology_rivers.py),
[stages/corner_drainage.py](../worldgen/stages/corner_drainage.py)

The biggest stage in the pipeline. It settles the water on the eroded heightmap — which
hollows hold lakes, at what level, which of them overflow — and then routes the rivers
along the sides between hexes.

`HydrologyStage.run` in `hydrology.py` is the whole sequence in order. The steps live in
two mixins: `hydrology_fill.py` holds sink filling, the plateau tilt, flow direction, the
off-map inlets and flow accumulation; `hydrology_rivers.py` holds river tracing and its
fallbacks, lake filling and drainage, the endorheic test, and the routing on corners.

**Reads:** `hex.elevation`, `hex.terrain_class`.
**Writes:** `state.rivers`, `state.river_sides`, `state.river_corners`,
`state.metadata["runoff_mm"]`, `hex.catchment_km2` and `hex.river_flow` (on both banks of
every river), `hex.tags` (`endorheic`, `endorheic_shore`). May also raise lake
water-levels and convert land hexes to LAKE/OCEAN if a basin needs to expand to its
spillway.

**Config:** `channel_min_discharge`, `navigable_min_discharge`,
`evapotranspiration_base_mm`, `evapotranspiration_per_c_mm`, `min_runoff_mm`,
`river_wander_exponent`, `corner_floor_blend`, `river_flow_continuous`, `lake_chaining`,
`endorheic_marsh_radius`, `endorheic_marsh_min_precip_mm`, and the `river_inflow_*` keys.

**Why two models.** Rivers run along hexsides, so the network the water finally runs
through is the hexes' *corners*. But whether a lake overflows, and through which shore, is
a question about whole basins and their water balance, and the hex model already answered
it well. So the stage keeps the hex model for the ground (parts A–C below) and drains the
corner graph over the ground it has settled (part D). The hex model's own courses are
never drawn.

**A — Settling the ground on hexes**

1. **Priority-Flood** sink-fills closed depressions on land, using a min-heap seeded with
   ocean and border land hexes (Barnes et al. 2014). After this pass every land hex has a
   non-decreasing path of `filled[...]` values to the sea.

2. **Epsilon tilt** breaks ties on plateaus:
   ```
   filled[c] += 1e-6 * drain_dist[c]/max_dist
              + 1e-6 * 1e-4 * (q + r) / (w + h)
   ```
   The first term gives a plateau a gradient toward the point it drains from, measured
   across that plateau only; the second is a coordinate tie-break that makes the result
   independent of dict iteration order.

3. **Flow direction** — each land hex points at a lower neighbour on the filled surface,
   drawn at random with weight (drop / largest drop)^`river_wander_exponent`, so rivers on
   level ground wander into each other rather than running as parallel straight lines.

4. **Flow accumulation** — Kahn's topological sort, then
   `acc[c] = rain[c] + sum(acc[upstream])`. Rain is the shared precipitation field
   (`precipitation.py`), averaging 1.0 per land hex, so at 1 hex = 1 km `acc` is
   catchment in square kilometres, and a rain shadow raises smaller rivers.

5. **The channel threshold** — a channel forms where enough water passes to keep one open:
   ```
   PET          = evapotranspiration_base_mm
                  + evapotranspiration_per_c_mm * max(0, temp_c)
   runoff_mm    = max(min_runoff_mm,
                      precip_mm - precip_mm / sqrt(1 + (precip_mm / PET)^2))
   min_catchment = channel_min_discharge / runoff_mm
   ```
   `runoff_mm` is recorded in `state.metadata["runoff_mm"]`, so anything reading the world
   — the campaign's major/minor river line — turns a catchment into discharge exactly as
   the generator does.

   **Discharge, not rank.** The old `river_flow_threshold` was documented as a flow
   minimum and implemented as "take the top 5% of land by accumulation", so every
   climate — desert and rainforest alike — got 5.6% of its land under channel. Runoff
   uses Pike's curve rather than subtracting the evaporative demand outright: ground
   cannot evaporate rain it never received, so actual evaporation approaches the demand
   where rain is plentiful and approaches the rain where it is not. A plain subtraction
   zeroed every climate whose demand exceeds its rain — mediterranean (550 mm) and arid
   (200 mm) both fell to the floor and drew identical rivers. Because the demand rises
   with temperature, cold country still sheds nearly everything it receives.

**B — Lakes**

6. **Hex courses** — the hexes over the threshold are traced to the sea, with
   elevation-guided Dijkstra and then BFS as fallbacks where a trace stalls. These courses
   exist only to feed lake drainage.

7. **Lake drainage** — each lake finds its natural **spillway** (the lowest land hex on
   its perimeter), rises to it, and floods any land below the new level. A lake that
   reaches the map edge becomes OCEAN. Otherwise its outflow is routed toward the sea or
   the border. With `lake_chaining`, a lake may spill into another lake — including a
   closed one, as the Jordan runs into the Dead Sea.

8. **Endorheic basins** — a bowl ringed by higher ground with no lower lake to spill into
   keeps no outlet: forcing a river out of it would be a lie about the terrain. Its water
   leaves by evaporation, so its hexes are tagged `endorheic` and the land within
   `endorheic_marsh_radius` `endorheic_shore`, for `BiomeStage` to turn into wetland.

**C — Rivers from off the map.** Inlets on the border, chosen by the `river_inflow_*`
keys (§ [4.5](#rivers-from-off-the-map)), are seeded with a catchment they did not gather
here, so they enter already wide.

**D — The rivers, on corners** (`_route_on_corners`, using `corner_drainage.py`)

9. **The corner graph.** Every corner touches three hexes and three other corners.
   - A corner is a node if any of its hexes is land. Its height is the **lowest** of its
     land hexes, lifted `corner_floor_blend` of the way toward their **mean**. A corner is
     where valley sides meet, so it belongs at the floor; but at a blend of 0 all six
     corners of a valley-floor hex stand at the same height, which leaves a level lane a
     hex out from every river, and tributaries ran down it beside their trunk for miles
     instead of joining it — 20% ran more than 5 km within 2 km of their trunk. At 0.25 a
     corner against the valley side stands above one in the middle of the floor, the floor
     tilts toward its river, and that falls to about 12%, mostly the parallel drainage of
     broad plains, which is real.
   - Water runs only along a side with **land on both hands**. A side with water on one
     hand is a shoreline, and one with the map edge on one hand is the frame; a river
     along either would be a line drawn beside the water, not a watercourse.
   - A corner touching the sea, a closed lake or the map edge is a **terminal**: water
     reaching it leaves the network.
   - Each open lake is a single **super-node**, named `(q, r, 2)` after its least hex so it
     sorts with the corners without colliding with one. It sits at its water level, joins
     every corner of its shore, and spills wherever the fill first reached it from below.

10. **Fill and direction.** A priority flood from the terminals fills every depression;
    the heap is broken by insertion order, so on a flat the fill spreads outward from the
    drain one step at a time and the order nodes leave the heap is downhill everywhere.
    Each corner then picks one earlier (lower) neighbour, with the same wander weighting
    as the hex model, so the network never loops. A corner beside the water runs into it.

11. **Rain and accumulation.** Each land hex drains whole into its lowest corner (the
    earliest in the fill), each open lake gathers the rain on its own hexes, and inflow
    inlets add their imported catchment. Accumulating down the flow gives every corner its
    catchment in km².

12. **Courses** (`trace_streams`). A corner is channel where at least `min_catchment`
    passes it; everything downstream of a lake's outlet is channel however little
    spills. Each course runs from where a channel begins — a spring, a lake outlet, an
    inflow — until it leaves the network or meets a bigger river. At a confluence the
    larger branch carries on and the smaller ends on the junction corner, so a
    tributary's last corner is a corner of its trunk and nothing is drawn twice. A course
    reaching an open lake ends on the shore; what leaves the lake is a course of its own.
    If nothing reaches the threshold, the map keeps its single largest drainage line.

13. **Recording.** Each course becomes a `River` (`corners`, `flow_volume`). Each side
    it runs along becomes a `RiverSide` with `catchment_km2`, `flow` (the catchment as a
    share of the largest) and `drop_m`. Its corners are tagged in `river_corners`:
    `river_source` (rises on the map), `river_source_offmap` (an inflow), `river_end`
    (stops without joining another river), `river_mouth` (a `river_end` into the sea or a
    lake), and `confluence` where two or more courses meet — inland only, since two
    rivers reaching one stretch of shore have arrived, not met. Every exporter and the
    campaign map draw a source as a ring, an end as an arrowhead pointing downstream, and
    a cataract or rapids as white bars across the river.

14. **Hex catchments.** For the viewer and for erosion, each land hex records in
    `catchment_km2` what drains to its own lowest corner or, on a bank, the largest river
    it touches; `river_flow` is that as a share of the largest on the map, written on
    *both* banks of every river side and zero elsewhere. With
    `river_flow_continuous=True`, every draining land hex gets a value — a diagnostic for
    inspecting the drainage field. Nothing deciding anything about a river reads
    `river_flow`; it reads the river sides.

---

### 3.5a Cataracts

[cataracts.py](../worldgen/stages/cataracts.py), both models, straight after hydrology. A
river side whose catchment carries a barge (`navigable_min_discharge`) and whose water
falls at least `cataract_min_drop_m` per km is tagged `cataract`. The fall is measured
along the river's own course, over the side and the sides either side of it
(`riverside.side_gradients`), because a corner stands at its lowest hex and the fall over
any single side comes in lumps. Three things follow:

- **A barrier.** `navigable` is false on a cataract, so a cargo afloat must land above it
  and load again below. The tag goes out in the world file for anything else — a campaign
  movement rule — that cares whether a boat can pass.
- **A portage.** Landing and reloading is a change of mode, so `CityPromotionStage` pays
  the quays either side (§ [3.10d](#310d-resource-settlements--organic)). Provisioning
  through a portage is small: two quay charges and a land hop are a large share of
  `haulage_range_land`, so the markets above a cataract mostly stop sending to the city
  below it. What the great portage towns lived on was the long trade down the river, and
  that is the **river trade**: every inland river town ships `river_trade_share` of its
  worth downstream to the coast, and pays at each portage on the way (§ [3.10e](#310e-city-provisioning-trade-and-freight--organic)).
- **A mill site.** `site_bonus` pays a site on or beside a cataract
  `habitability_mill_bonus`.

A cataract side breaks a navigable **reach**: a barge goes from one bank hex to the next
only along a run of navigable sides joined corner to corner, and the hexes beside a
cataract are its **portage**. A smaller river falling at least `rapids_min_drop_m` per km
over a catchment of `rapids_min_catchment_km2` or more is tagged `rapids` instead; that is
drawn as white water and, in the campaign, has to be bridged or pontooned rather than
waded. It changes nothing else in the generator.

It has to run before `HabitabilityStage`, which reads `navigable` to score harbours.

---

### 3.6 Climate

[stages/climate.py](../worldgen/stages/climate.py)

**Purpose:** Compute temperature and moisture fields. Two independent
sub-passes, run sequentially.

**Reads:** `hex.elevation`, `hex.terrain_class`, `state.river_sides`.
**Writes:** `hex.temperature`, `hex.moisture`, `hex.wet_season_precip_mm`,
`hex.dry_season_precip_mm`.

**Config:** `regional_climate`, `mean_temperature_c`, `latitude_temp_range_c`,
`lapse_rate_c_per_km`, `wind_direction`, `orographic_strength`,
`moisture_resupply_per_hex`, `mean_precip_mm`, `base_precip_mm`, `wet_season_share`,
`moisture_bleed_passes`, `moisture_bleed_strength`, `max_elevation_m`.

**The map is a region, not a world.** 500 km at 1 hex = 1 km is about 4.5° of latitude,
some 3 °C; altitude does far more over the same distance and rain shadow more again. So
the region has one climate, named by `regional_climate`, and the variety within it comes
from terrain. Each climate also carries a **palette** of biomes it can produce, which is
what stops an arid region growing a jungle three valleys over.

#### Temperature — degrees Celsius

```
row_frac    = row / max(height - 1, 1)        # 0 at top, 1 at bottom; `row` is the grid
                                              # row, which is `r` on an axial grid and
                                              # the true north-south axis on an offset one
lat_temp    = sin(row_frac * π)               # 0 at poles, 1 at equator
temperature = mean_temperature_c
            + (lat_temp - 2/π) * latitude_temp_range_c
            - max(0, elevation_m) / 1000 * lapse_rate_c_per_km
```

The `2/π` subtraction is the mean of `sin` over `[0, π]`, so `mean_temperature_c` is the
true map-average rather than the equator value. The output is Gaussian-blurred with
`sigma=1.0`.

Two things follow from the units. `lapse_rate_c_per_km` is the **real environmental lapse
rate**, 6.5 °C/km, not a tuning constant. And the lapse is applied to height above the
**waterline** — `max(0, elevation)` — so a hex at sea level gets none, which is what makes
the result a real temperature rather than one relative to the map's own lowest point.

`latitude_temp_range_c` defaults to `0.0` because across 128 km it is genuinely
negligible; raise it only for a continent-scale map.

#### Moisture — millimetres a year

1. **Orographic precipitation.** Sort all hexes by their dot product with the wind
   direction, so upwind hexes process first. The wind carries atmospheric moisture;
   lifting it over higher terrain condenses it as rain:
   ```
   incoming = mean(atm[upwind neighbours]) or 1.0 if none upwind
   lift     = max(0, elevation_m) / max_elevation_m
   fraction = min(1, lift * orographic_strength)
   precip   = incoming * fraction
   left     = max(0, incoming - precip)
   atm[hex] = left + moisture_resupply_per_hex * (1 - left)      # air picks moisture back up
   ```
   Windward slopes are wet, lee slopes dry.

   **The resupply term matters.** Without it the sweep is a one-way drying: whatever the
   first barrier takes is gone for good, so the far side of a 128 km map receives nothing
   at all, rainfall spans a factor of eight from coast to interior, and a temperate map
   reads 60% shrubland. Real air is resupplied continuously by evaporation, which is why
   a rain shadow is a local feature tens of kilometres deep rather than everything
   downwind of the first hill.

2. **River and coastal bonuses.** For every land hex: `+0.15` if it is *near a river*
   (only when `moisture_bleed_passes == 0`), and `+0.1` if any neighbour is OCEAN or
   LAKE. Cumulative, so a coastal hex near a river gets `+0.25`. A river runs along a
   hexside, and the ground it waters is the four hexes at that side's two corners: the
   two banks and the hex at either end.

3. **Gaussian smear** at `sigma=2.0`, over land only. Weather systems are wide, and rain
   falls either side of the ridge that lifted it rather than only on the hex that did the
   lifting. It is a normalised convolution — the land's values smeared, divided by the
   smeared land mask — so each land hex is a weighted mean of the land around it:
   ```
   moisture = gaussian(moisture * land) / gaussian(land)        # on land hexes
   ```
   Water holds the sweep's carrier value of 1.0 here — saturated air, not rain — and the
   plain smear used to blend it into every shore, lee and windward alike, so a lee coast
   drew rain from the sea behind it (tech-debt #123). On a 64×64 mediterranean map that put
   the lee coast at 1420 mm against an interior of 411; over land only it is 616 against
   510, and the windward coast stays the wetter at 951.

4. **Into millimetres a year.** The orographic pass produces a *relative* pattern —
   which slopes catch the rain and which sit in a shadow — and says nothing about whether
   the region is wet or dry. A **linear** scale putting the land mean on
   `mean_precip_mm`, plus `base_precip_mm`, supplies that:
   ```
   moisture = moisture * (mean_precip_mm / land_mean) + base_precip_mm
   ```
   Linear is the honest choice: if a leeward valley receives a third of what the windward
   slope does, that ratio is a fact about the terrain and should survive being told how
   wet the region is overall. The previous version stretched to `[0, 1]` and fitted a
   **gamma** to move the mean onto a target — which held the bounds but warped the
   distribution to do it, so the leeward-to-windward ratio came out different for a wet
   region than for a dry one. In millimetres there are no bounds to hold.

5. **Optional moisture bleed.** When `moisture_bleed_passes > 0`, the flat `+0.15` river
   bonus is replaced by an iterative diffusion: each pass a hex gains
   `moisture_bleed_strength × mean_precip_mm × flow` from the largest river near it whose
   water — the lower of its two banks — stands at or above the hex. This builds a wider moisture corridor along big rivers,
   especially in valleys, but never uphill. There is no ceiling on the result — moisture
   is millimetres now, and a valley that receives more rain than the ridge above it is
   simply a wetter valley.

6. **Wet and dry seasons.** Last, so the halves always sum to the year:
   ```
   wet_season_precip_mm = moisture * wet_season_share
   dry_season_precip_mm = moisture - wet_season_precip_mm
   ```
   The orographic pattern says *where* rain falls; the region's climate says *when*, and
   every hex divides the same way. `wet_season_share` is the wetter six months' share of
   the year in the long-run normals of the places each climate is named after:

   | climate | share | crops grow in | the year |
   |---|---|---|---|
   | temperate | 0.55 | dry half | Atlantic Europe: rain every month, a mild autumn-winter peak |
   | mediterranean | 0.75 | dry half | winter wet, summer dry (Athens, Seville ~0.8; Rome ~0.65) |
   | arid | 0.8 | dry half | the winter-rain desert of the Maghreb, Egypt and the Levant; set `crops_grow_in_wet_season` for a Sahel |
   | tropical | 0.8 | wet half | savanna and monsoon: the kharif crop is sown into the rains |
   | boreal | 0.65 | wet half | summer rain, and the winter's snow lies until it melts into the spring sowing |

   `crops_grow_in_wet_season` says which half the crops grow in. Soil reads the two halves (§ [3.7a](#37a-soil)). Nothing else does: biomes and land
   cover still read the year's rain.

**Ocean and lake hexes** are set to `mean_precip_mm` at the end, so they do not read as the
driest ground where moisture is drawn; nothing downstream reads rainfall on water.

---

### 3.7 Biomes

[stages/biomes.py](../worldgen/stages/biomes.py)

**Purpose:** Whittaker-style biome assignment based on temperature,
moisture, elevation, and water/river adjacency.

**Reads:** `hex.terrain_class`, `hex.elevation`, `hex.temperature`,
`hex.moisture`, `hex.tags`.
**Writes:** `hex.biome`.

**Config:** `regional_climate`, `biome_treeline_temp_c`, `biome_snowline_temp_c`,
`biome_cold_temp_c`,
`biome_warm_temp_c`, `biome_dry_precip_mm`, `biome_wet_precip_mm`,
`wetland_min_runoff_mm`, `endorheic_marsh_min_precip_mm`.

**Algorithm** ([biomes.py](../worldgen/stages/biomes.py)):

```
if terrain_class in (OCEAN, LAKE):
    biome = OCEAN
elif temperature_c < biome_snowline_temp_c:                 # default -8 C
    biome = ALPINE
elif temperature_c < biome_treeline_temp_c:                 # default -2 C
    biome = TUNDRA
elif temperature_c < biome_cold_temp_c:                     # default 5 C
    biome = pick(BOREAL, ...)
elif temperature_c >= biome_warm_temp_c:                    # default 18 C
    biome = pick(DESERT, ...)    if precip_mm < biome_dry_precip_mm
            pick(GRASSLAND, ...) if dry <= precip_mm < biome_wet_precip_mm  # 1000 mm
            pick(TROPICAL, ...)  otherwise
else:  # temperate band
    biome = pick(DESERT, ...)            if precip_mm < biome_dry_precip_mm
            pick(GRASSLAND, ...)         if dry <= precip_mm < wet
            pick(TEMPERATE_FOREST, ...)  otherwise
```

**`pick` draws from the region's palette**, taking the first candidate the climate can
actually produce and falling back towards its staple. So a hex that would have been
tropical in a boreal region becomes the closest thing that region has, rather than
importing a biome from three climate zones away. Desert is offered before shrubland in
the cool-and-dry branch because a dry region does not stop being a desert for being cool
— the Gobi and the Great Basin are cold deserts.

**The treeline is a temperature, not a height.** The altitude it falls at follows from the
region's warmth and the lapse rate: ~1850 m temperate, ~500 m boreal, above 4300 m
tropical. A fixed altitude could not say any of that, and the fraction-of-range version it
replaced said the opposite — it gave every map the same share of alpine ground however low
its hills. `WorldConfig.treeline_m()` reports that altitude, but the stage tests each
hex's own temperature, so the line bends with latitude instead of being computed once at
the map's mean.

**Two lines divide cold country, not one.** Above the treeline nothing grows tall; above
the *snowline* nothing grows at all. Between them is tundra — treeless but vegetated, and
that is most of what stands above a subarctic treeline. Only above `biome_snowline_temp_c`
is ground genuinely barren, and that is ALPINE. At the default -8 °C no map with 1500 m of
relief has any permanent snow on it, which is correct: a temperate range that high has no
glaciers either. Raise `max_elevation_m` towards 2400 and bare peaks appear on their own.

**Below the treeline and cold is taiga, whatever the rainfall.** The cold band used to
split TUNDRA from BOREAL on `biome_dry_precip_mm` — the same 400 mm that separates desert
from steppe — which made rain the thing that stops trees in the subarctic. It is not; cold
is, and the treeline already says so. Siberian larch grows on 200–400 mm, and the boreal
region's own mean is 450, so any rain shadow tipped land into tundra at zero food value.
Between the elevation-keyed ALPINE test and this one, a boreal map came out 41% bare rock
and 23% tundra, and supported five settlements on fifteen thousand land hexes. It is now
49% taiga and 41% tundra, with no bare rock at all, and supports fourteen. No other
climate's map changes by a single hex — none of them has land below -2 °C.

**Wetland overrides**, applied afterwards:

```
# Riverside waterlogging — on the lower bank of each river side, since the higher
# bank sheds its water back into the river
if terrain_class in (FLAT, COAST) and hex is the lower bank of a river side
   and runoff_mm(precip, temp) > wetland_min_runoff_mm
   and temperature_c >= biome_treeline_temp_c:
    biome = WETLAND

# The shore of a closed basin
if "endorheic_shore" in tags and terrain_class in (FLAT, COAST)
   and precip_mm >= endorheic_marsh_min_precip_mm
   and temperature_c >= biome_treeline_temp_c:
    biome = WETLAND
```

Waterlogging is tested on **runoff, not rainfall**. It is not a question of how much rain
arrives but of whether the ground can get rid of it: flat land beside a river, where what
the sky delivers exceeds what the air takes back, holds a water table at the surface. The
rainfall version asked for more than the wet biome band, which on a temperate map at
800 mm almost nothing reaches, so bogs vanished from the map entirely — and it would have
called a cold region dry when cold country is exactly where peat forms, because so little
of its rain evaporates away.

The endorheic rule's moisture floor keeps arid basins as salt pans: a closed basin in a
desert is a playa, not a swamp.

**Notes**

- Both wetland rules depend on a tag, so they fire only on or beside the feature — not as
  a thick buffer. For wider wetlands raise `moisture_bleed_passes`.
- Keep `biome_treeline_temp_c` clear of every climate's own mean temperature. The cold
  tests run ahead of every other temperature rule, so a treeline landing at sea level puts
  a whole region above it: at `1.0`, which is the boreal region's mean, a boreal map grew
  no taiga at all and supported five settlements on sixteen thousand hexes.
- ALPINE and TUNDRA do not go through `pick`. They are what happens when ground is too
  cold to grow anything, which every region has somewhere above it, so `_ALWAYS` lists
  them both and any palette can produce them.

---

### 3.7a Soil

[stages/soil.py](../worldgen/stages/soil.py)

**Purpose:** Say what the ground could support, before anything is done with it.

**Reads:** `hex.terrain_class`, `hex.elevation` (through the gradient),
`hex.wet_season_precip_mm`, `hex.dry_season_precip_mm`, `hex.temperature`, `hex.biome`, `hex.tags`, `hex.catchment_km2`.
**Writes:** `hex.soil`.

**Config:** § [4.8](#48-soil-food-and-habitability--37a-soil-39-habitability).

#### Why it exists

`food_value` used to key on `land_cover`, which said a hex was fertile **because grass grew
on it**. That is backwards, and it is why the map could not tell a floodplain from a chalk
down. Grass on temperate lowland is what you get after clearing or on thin soil; the best
ground in northern Europe carried wildwood until somebody assarted it.

Soil is now asked about directly, and the answer is a ladder:
`UNUSABLE < GRAZING < MARGINAL < ARABLE < PRIME`.

```
water, wetland, above the treeline          → UNUSABLE

alluvium: gentle ground on or beside a
river with catchment >= ford_max_catchment  → PRIME

otherwise the worse of two arms:
  slope    >= terrain_escarpment_gradient_m → UNUSABLE
           >= terrain_steep_gradient_m      → GRAZING
           otherwise                        → ARABLE
  rainfall, a season at a time (below):
    2 × growing <  soil_dry_farming_min_precip_mm → UNUSABLE
                <  biome_dry_precip_mm            → GRAZING
    2 × wet     <= biome_wet_precip_mm            → ARABLE
            <  food_drowned_precip_mm         → MARGINAL
            otherwise                         → UNUSABLE

then, last of all:
  temperature < biome_cold_temp_c            → capped at MARGINAL
```

**Each failure has its season.** A crop fails for want of water in the season it grows in,
and ground is leached and waterlogged by the wet half of the year, so the dry arm reads the
growing season and the wet arm the wet one, each against half the annual bands (`2 ×` the
season is the same comparison). In a year without seasons both halves are half the year's
rain, and the rule reads exactly the annual bands — which is what keeps soil and biomes on
the one pair of figures. Before the halves are read, the ground carries
`soil_water_carryover` of the wet season's surplus over the dry across into the drought:

```
carried = soil_water_carryover × max(0, wet − dry)
growing = wet                  if crops_grow_in_wet_season
          dry + carried        otherwise
wet    −= carried
```

Where crops grow in the rains (monsoon, taiga) the dry season is a fallow winter — in the
taiga a snowpack that melts into the spring sowing — and its drought costs no crop. Reading
the dry half there took a 128×128 subarctic map from 6 markets to 2; reading the growing
season gives it 15.

A loam holds 100–200 mm a crop can draw; the winter rains fill it and the crop ripens on it
after they stop, which is what the Mediterranean's bare fallow was for. At the default 0.3 a
mediterranean year (0.75) reads as 0.6, a temperate one (0.55) as 0.52.

So the same annual rain farms worse the more it bunches. 480 mm falling evenly ploughs; 480
mm with three quarters of it in winter is grazing. That is the Mediterranean's summer
drought, and it is what makes the region pastoral — winter cereal on a thin fallow, the
flocks walked to the hills for summer grass — on rainfall that would plough in Kent. Until
tech-debt #123 the region came out pastoral for the wrong reason: the climate smear blended
the sea into the coasts, which widened every climate's rainfall spread, and the bands had
been calibrated against the widened spread. Remove that and a mediterranean map's rain sits
inside the arable band: 50% arable to temperate's 50% on the 96×96 test map.

**Rainfall fails differently at each end.** Under the dry-farming limit nothing is grown at
all. Between that and the arable band you get steppe, where grass grows and a crop will not,
which is grazing. Above the band the ground is leached and waterlogged, which is poor
*arable*, not pasture — the first version of this rule was symmetric, and calling a
rainforest "grazing" was the tell that it was wrong.

**Alluvium skips the rainfall arm, because the Nile does not need rain.** In a desert the
floodplain is not merely the best land, it is the only land — which is why an arid map's
settlements string along its rivers instead of being absent altogether.

**The cold cap is applied last, so it binds alluvium too.** A flood meadow on the Lena is
the best ground in the taiga and still will not grow wheat. Capping before that branch let a
boreal floodplain out at PRIME, which would have said a subarctic river bottom is worth what
Kent is; and without the cap at all, a boreal map comes out 13% arable and 42% richer.

**One new threshold, and only one.** `soil_dry_farming_min_precip_mm` is the dry-farming
limit; every other figure is a setting that already existed and already meant the right
thing. `ford_max_catchment_km2` is the neatest of the reuses — it is the same question asked
from the other side, since a river you cannot wade is one that floods and lays down silt.
The seasons add no threshold: `soil_water_carryover` is a share, not a rainfall figure, and
each season is read against the bands that were already there.

**Result at 128×128, seed 42:**

| climate | unusable | grazing | marginal | arable | prime |
|---|---|---|---|---|---|
| temperate | 18.1% | 17.7% | 11.2% | **48.3%** | 4.6% |
| mediterranean | 9.5% | **50.4%** | 3.6% | 25.8% | 10.7% |
| arid | **89.6%** | 5.2% | 0.0% | 0.1% | 5.1% |
| boreal | 28.0% | 18.7% | 53.3% | **0.0%** | 0.0% |
| tropical | 35.9% | 13.7% | **47.6%** | 0.0% | 2.7% |

Mediterranean comes out pastoral, arid is desert with its life on the rivers, the taiga
grows no wheat at all, and the tropics are leached. One rule set; the region decides.

#### Land cover follows soil

`LandCoverStage` now takes the region's own woodland on PRIME and ARABLE ground — oak here,
taiga there, gallery forest along a desert river — so **good soil is under trees until
somebody clears it**. A temperate map went from 50% open grass to 41% woodland and 27% dense
forest, which is what northern Europe looked like before the assarting centuries.

---

### 3.8 Land Cover

[stages/land_cover.py](../worldgen/stages/land_cover.py)

**Purpose:** Pure derivation of `land_cover` from `terrain_class`, `biome`,
and `moisture`. Adds visual texture without rolling new dice.

**Reads:** `hex.terrain_class`, `hex.biome`, `hex.moisture`.
**Writes:** `hex.land_cover`.

**Config:** `biome_wet_precip_mm` (sets the dense-forest threshold).

**Algorithm** ([land_cover.py:16–44](../worldgen/stages/land_cover.py#L16)):

```
if terrain_class in (OCEAN, LAKE):  OPEN_WATER
if terrain_class == MOUNTAIN:       BARE_ROCK
if biome == ALPINE:                 ALPINE
if biome == TUNDRA:                 TUNDRA
if biome == DESERT:                 DESERT
if biome == WETLAND:
    if terrain_class == COAST:      MARSH
    else (FLAT):                    BOG
if biome == BOREAL:                 DENSE_FOREST
if biome == TEMPERATE_FOREST and moisture > (wet_moist + 1) / 2:
                                    DENSE_FOREST
if biome in (TEMPERATE_FOREST, TROPICAL):
                                    WOODLAND
if biome == SHRUBLAND:              SCRUB
otherwise (GRASSLAND):              OPEN
```

**Why `(wet_moist + 1) / 2`?** Every TEMPERATE_FOREST hex already passes
`moisture >= wet_moist`, so the dense-forest threshold needs to be
higher than that to ensure both DENSE_FOREST and WOODLAND actually
appear. Splitting the surviving range in half — i.e. `(wet_moist + 1) / 2`
— gives a roughly even partition.

---

### 3.9 Habitability

[stages/habitability.py](../worldgen/stages/habitability.py)

**Purpose:** Three `[0, 1]` scores per hex — one per settlement tier — used
as the input to all settlement placement.

A site is scored on the land it can actually feed itself from, not on the
biome of the single hex it stands on. The **catchment** is the mean food
value of every hex within reach, so a town ringed by grassland beats one on
an identical hex ringed by desert. Reach depends on tier: a city draws on a
far wider hinterland than a village, so the same hex is scored three times,
at each tier's cultivation radius (8 / 4 / 2).

**Reads:** `hex.land_cover`, `hex.moisture`, `hex.terrain_class`,
`hex.biome`, `hex.tags`, neighbours.
**Writes:** `hex.habitability_city`, `hex.habitability_town`,
`hex.habitability_village`.

**Config:** `food_prime_value`, `food_arable_value`, `food_marginal_value`,
`food_grazing_value`, `food_wetland_value`, `food_water_value`, `habitability_agri_weight`,
`habitability_river_bonus`, `habitability_coast_bonus`,
`habitability_hill_bonus`, `habitability_confluence_bonus`,
`cultivation_city_radius`, `cultivation_town_radius`,
`cultivation_village_radius`, `biome_dry_precip_mm`, `biome_wet_precip_mm`,
`food_drowned_precip_mm`.

**Algorithm** ([habitability.py](../worldgen/stages/habitability.py)):

```
# Per-hex food value, by land cover band
PRIME soil                → food_prime_value
ARABLE soil               → food_arable_value
MARGINAL soil             → food_marginal_value
GRAZING soil              → food_grazing_value
UNUSABLE soil             → 0
BOG, MARSH                → food_wetland_value    (cover, not soil: a fen is not ploughland)
OPEN_WATER                → food_water_value      (cover, not soil: the sea is a fishery)
TUNDRA/DESERT/ALPINE/ROCK → 0.0

# Rainfall is not monotonic for farming — a tent, not a ramp.
# dry = biome_dry_precip_mm, wet = biome_wet_precip_mm, drowned = food_drowned_precip_mm
moisture_factor(p) = 0.0                          if p <= 0 or p >= drowned
                   = p / dry                      if p < dry
                   = 1.0                          if dry <= p <= wet
                   = (drowned - p)/(drowned - wet) if p > wet

# Hard zeros — you cannot found a settlement here
if terrain_class in (OCEAN, LAKE, STEEP, ESCARPMENT) or biome == WETLAND:
    every score = 0.0

# Site bonuses, identical across tiers
bonus  = habitability_river_bonus      if a river runs along one of the hex's sides
       + habitability_coast_bonus      if hex or neighbour is COAST
       + habitability_hill_bonus       if a rise with a level neighbour
       + habitability_confluence_bonus if the hex touches a `confluence` corner

# One score per tier, each normalised against its own map-max
raw[tier] = habitability_agri_weight * mean(food value within radius[tier]) + bonus
hex.habitability_<tier> = raw[tier] / max(raw[tier] across map)
```

**Notes**

- **Water is not worth zero.** A coastal site fishes. Scoring the sea at
  nothing penalised coastal sites twice — half their catchment counted as
  waste ground, and the coastal bonus existed largely to repair the damage.
  Wetland sits *below* open water, being neither good fishing nor good
  ploughing, which matches bog and marsh resisting cultivation outright.
- **Land cover, not biome, is the key.** Cover already folds in terrain and
  moisture (the dense-forest/woodland split is a moisture threshold) and is
  what Cultivation tests against, so the two cannot disagree about what is
  farmable. The moisture curve then discriminates *within* a band rather
  than re-deciding what the cover already settled.
- Off-map neighbours are excluded from the mean rather than counted as
  zero, so a hex on the map border is not scored as though the edge were
  desert.
- Each tier normalises against its own best site: the scores are only
  compared within a tier, and a shared divisor would let the widest
  catchment squash the other two.
- Roads add `+0.2` to neighbour `habitability_village` *after* this stage,
  in Interurban Roads. Only the village score — cities and towns are
  already sited by then, and a road they caused should not retroactively
  flatter the ground it runs over.
- Costs one walk of the largest radius per hex (217 hexes at r=8), with
  per-hex food values computed once into a lookup table. A 128×128 world
  generates end to end in under 4s.

---

### 3.10 City & Town Placement

[stages/city_town.py](../worldgen/stages/city_town.py)

**Purpose:** Place up to `target_city_count` cities and `target_town_count`
towns by greedy selection on their own tier's habitability score, with minimum-separation
constraints.

**Reads:** `hex.habitability_city`, `hex.habitability_town`,
`hex.terrain_class`, `hex.biome`,
`hex.elevation`, neighbours.
**Writes:** `hex.settlement`, `state.settlements`, `hex.tags` (`"prominent_site"`,
`"confluence_town"`).

**Model:** `classic` only. The `organic` model replaces this stage with
§ [3.10b](#310b-market-centres--organic).

**Config:** `target_city_count`, `target_town_count`, `city_min_separation`,
`town_min_separation`, `settlement_min_reachable`, plus the road-grade
parameters used for reachability (`hex_size_m`, `road_slope_cap_pct`).

**Reachability filter.** For each candidate hex, BFS over land
neighbours where the connecting edge satisfies
`grade_pct < road_slope_cap_pct` (default 25 %)
([city_town.py:37–47](../worldgen/stages/city_town.py#L37),
[hex_grid.py:142–171](../worldgen/core/hex_grid.py#L142)). If fewer than
`settlement_min_reachable` (default 100) hexes are reachable, the
candidate is rejected. This keeps cities off geographically isolated
peaks and tiny islands.

**Cities** ([city_town.py:70–91](../worldgen/stages/city_town.py#L70)):
sort all land hexes by `habitability_city` descending — the widest catchment,
because a capital is chosen for the hinterland it can draw on; greedily accept each
one whose distance from every prior city is `>= city_min_separation`.
Each city gets a uniform-random population in `[10_000, 50_000]` and a
role from `_assign_role` (below).

**Towns** ([city_town.py:93–138](../worldgen/stages/city_town.py#L93)):

1. Sort on `habitability_town` — a different surface from the city score,
   with its own peaks, because a market town lives off the fields in
   walking distance rather than off a province. This replaces the old
   blanket `× 0.5 within 30 hexes of a city` damp, which existed only to
   push towns off the capitals' sites; `town_min_separation` now does that.
2. Find local maxima of that score (hexes whose score beats all 6
   neighbours).
3. Greedy placement with `town_min_separation` (default 8). Population
   uniform random in `[1_000, 10_000]`. Towns on a hex touching a
   `confluence` corner also get the `"confluence_town"` tag.

**Role assignment** ([city_town.py:8–33](../worldgen/stages/city_town.py#L8)):

```
PORT         if waterside(hx): a bank of a navigable reach, next to one, or a shore
AGRICULTURAL elif >= 3 neighbours are GRASSLAND or TEMPERATE_FOREST
MARKET       otherwise
```

`navigable` is `haulage.navigable` — open water, or a river whose discharge clears
`navigable_min_discharge` — the same predicate `habitability.site_bonus` uses to award
`habitability_harbour_bonus`. Sharing it is the point: the site scored as a harbour is the
settlement labelled a port.

`_assign_role` is shared by all three placement stages — classic cities and towns,
villages, and organic markets — so the roles mean the same thing whichever model ran.

> **`PORT` means water a boat can use, not water in sight.** Almost every settlement this
> generator places stands on water — 36 of 36 cities and 143 of 144 towns across six 64×64
> reference maps — because that is what pre-industrial siting does and `habitability`
> rewards it. So a port test that fired on any water would be nearly constant. The rule
> used to read "any `river` tag or COAST hex within one step", which **43%** of all land
> satisfies against **22%** for navigable water; on the seed-42 reference only 20 of 105
> river hexes floated a boat (measured when rivers still occupied hexes). It now asks
> `riverside.waterside`: the hex is a bank of a navigable reach, or next to one, or on a
> shore. A hamlet on a headwater brook was labelled the same as a
> coastal harbour. On that map the change takes 127 ports down to 86, and the classic
> temperate maps now spread across all three roles instead of piling into one.
>
> Two distributions are worth knowing before reading a map. **Arid maps stay near-total
> ports** (74 of 75 on seed 42) because settlement in a dry country clusters on the few
> watercourses that exist — which is the Nile answer, and correct. **The organic model is
> 100% port at every seed tried**, because it places 3–18 settlements at the very best
> sites and every one of those is on navigable water. Both are now true statements about
> the map rather than artifacts of a loose test.

> **`MINING` and `FORTRESS` were retired.** There used to be a branch between `PORT` and
> `AGRICULTURAL` splitting settlements with a steep neighbour on `elevation > 0.70`. That
> threshold dated from the retired [0, 1] elevation axis and read as seventy centimetres
> once the axis became metres, so every settlement beside steep ground came out `MINING`
> and `FORTRESS` was unreachable — a 96×96 temperate map yielded 71 mining settlements and
> no fortresses. Rather than invent an altitude for two roles nothing consumed, both were
> removed from `SettlementRole`. Ground that used to be classed on its steep neighbours
> now falls through to the fertility test, so those settlements come out `AGRICULTURAL` or
> `MARKET`. Nothing downstream reads `Settlement.role` at all — not the stages, not the
> exporters, not the renderers — so the change is invisible outside `world.json`.
>
> A `world.json` written before this change records `"mining"` or `"fortress"` and will
> not load; regenerate it from its seed.

**Pass tagging:** after settlements are placed, every empty ROLLING hex that is the
local-max `habitability_town` within 3-hex range gets the `"prominent_site"` tag. Nothing reads it. Used for rendering
mountain passes and as a convenient query for module authors.

---

### 3.10a River Crossings — `organic`

[stages/crossings.py](../worldgen/stages/crossings.py)

**Purpose:** Decide where a river can be got across, before anything is built on the map.

A river is not uniformly crossable. Most of its length is an obstacle; a few places are
not, and those places are why towns sit where they do. This stage runs **before** market
placement, deliberately: a bridging point is the cheapest ground in a district to reach
from both banks, so it should be a *reason* a market grows there rather than something
noticed afterwards.

**Reads:** `state.river_sides` (catchment, drop), the food surface.
**Writes:** `RiverSide.tags` (`"ford"`, `"bridge"`) — a crossing is a river side, the step
from one bank to the other.

**Config:** `ford_max_catchment_km2`, `crossing_relief_m`, `bridge_pressure_per_span`,
`crossing_pressure_radius`, `crossing_min_separation`, `ford_serves_bridge_fraction`,
`rare_ford_max_span`, `rare_ford_separation`.

**The distinction the stage rests on:** a **ford is terrain and is free** — shallow
braided water anyone can wade, needing nobody's permission. A **bridge is capital**, and
appears only where enough traffic will use it. Nobody bridges to nowhere.

```
span      = effective width, in multiples of the wadeable catchment,
            inflated by local relief / crossing_relief_m
ford      if span <= 1                      # you can wade it
ford      if the river floats a barge and span <= rare_ford_max_span,
             easiest first, none within rare_ford_separation of another
                                            # the rare slack reach of a big river
bridge    if surplus within crossing_pressure_radius
             >= bridge_pressure_per_span * span
           and no bridge within crossing_min_separation
           and no ford there over water >= ford_serves_bridge_fraction of its own
```

**A big river's fords are rare.** Nothing that floats a barge is wadeable by size alone,
so without the second rule a major river is forded only where a road later crosses it.
The slackest reaches a little past the wading span are fords as well, spaced far apart:
on a 200×200 map, about one major-river side in eighty. They are the places a column is
marched to.

**A bridge here is a candidate site, not yet a bridge.** The roads decide which are built:
`tag_river_crossings` (§ [3.11](#311-interurban-roads)) keeps a bridge only where a road crosses
it, and adds one wherever a road crosses water too big to wade. So every bridge on a
finished map carries a road. Fords stay whether or not a road uses them, because a ford is
terrain.

**A brook's ford is no reason not to bridge the trunk.** A ford stands in for a bridge
only over water of its own size; otherwise a map with a ford on every brook would bridge
almost nothing.

**Relief, not just discharge.** Fast water takes your feet from under you whatever its
depth, and at a kilometre to the hex it is the approaches rather than the span that defeat
a bridge — both scale with how steep the ground is, so relief makes a reach behave like a
bigger river for fording and for building alike. A floodplain has a few metres of relief;
a gorge has hundreds.

**Relief is measured as channel drop**, along the watercourse, not as the spread of the
surrounding ground. The latter reports how deep the *valley* is — median 255 m on a test
map — which killed all but 2 of 126 fords. A river running along a flat valley floor
between high sides is easy to cross; the valley's depth is not the crossing's problem.

---

### 3.10b Market Centres — `organic`

[stages/markets.py](../worldgen/stages/markets.py), over
[stages/haulage.py](../worldgen/stages/haulage.py)

**Purpose:** Place market towns where the most surplus can reach them inside a day's
return, and size them from what they actually gather. Replaces `CityTownStage`.

**Reads:** the food surface, `hex.terrain_class`, `hex.elevation`, `hex.tags`,
the river sides and their crossings.
**Writes:** `state.settlements`, `hex.settlement`, `hex.territory`,
`hex.territory_cost`.

**Config:** § [4.10](#410-haulage-and-markets--the-organic-model).

#### The countryside is a surface, not a list of settlements

The decisive simplification. A market's draw is the surplus of its catchment, and whether
you discretise that catchment into forty hamlets or integrate over the food field gives
the same number — so the dispersed peasantry is modelled as a **continuous productive
surface and never enumerated**. Historically faithful village density would be ~900
objects on a 128×128 map (Domesday: ~13,000 vills over ~130,000 km²), almost none of which
carry military or administrative weight.

**This is why `organic` omits `VillagePlacementStage`, `VillageTrackStage` and
`VillageCultivationStage`.** They site a hamlet on every hex clearing a habitability bar,
which buried 74 markets under 835 settlements on a 128×128 temperate map — so most of what
the viewer showed was not the haulage model. `classic` keeps all three and is unchanged. A
settlement tier *below* the market will return, but gated on holding something — a bridge,
a pass — rather than sprinkled across the countryside.

The win is legibility rather than speed. The three stages cost 0.8 s of a 15.2 s pipeline;
`InterurbanRoadStage` is 12.3 s of the 14.4 s that remain, and SVG export of a map with 835
settlements dominated the wall clock of a full `generate` run either way.

**Surplus, not production, is what travels.** A farming household eats most of what it
grows; `marketable_surplus_fraction` (~20%) is what can leave. Sizing markets off the
surplus is why the tier ratios come out right with no target counts anywhere.

#### Planting — lazy greedy, which is exact here

```
heap = [(-bound(c), c) for c in sorted(settleable_land)]   # ring-disc upper bound
while heap:
    _, c = heappop(heap)
    if c in suppressed:            continue    # 1. suppression
    s = score(c)                               # 2. day-reach score against what remains
    if s < -heap[0][0] - EPS:                  # 3. stale → re-push
        heappush(heap, (-s, c)); continue
    if s < market_viability_floor: break       # 4. true max below floor → done
    plant(c)
    for n, w in reach(c):
        remaining[n] *= (1 - w)                # deplete exactly what was scored
    suppressed |= hex_range(c, market_min_separation)
```

```
reach(c) = the hexes a catchment rooted at c would walk — a single-source Dijkstra over
           the travel-cost field, budget market_day_radius, plus the fishery rim on the
           terms fishery_rim grants it — each weighted usable_fraction(cost, market_day_radius)
score(c) = (1 + site_bonus(c)) * Σ remaining[n] * w   over reach(c)
```

**Siting and the catchment read the same ground.** A candidate is scored on exactly the
walk its catchment will make once it is planted, at exactly the weight the market will
later be sized on, so a ridge or an estuary beside a candidate lowers its score. Until
#117 the score summed a plain hex disc of rings with a separate `1/(1 + d/4)` kernel: the
radius meant a *ring count* when siting and a *cost budget* when gathering, off-map and
across-the-water counted as if it could be walked, and the planting score ran about four
times the real gather. `market_kernel_decay` went with it.

Depletion only ever *reduces* other sites' scores, so the score function is monotone
non-increasing and a popped entry that is still fresh is provably the true maximum. The
heap is seeded with the plain ring disc at the same weights as an **upper bound**: every
step costs at least one unit, so a hex at ring *d* costs at least *d* and weighs no more
than at cost *d*, and a rim water hex sits one ring beyond its donor — so the Dijkstra
only runs on the candidates that pop, and exactness survives.

**That predicate order is load-bearing** — suppression, then recompute, then staleness,
then the floor. Testing the floor before the staleness check ends the loop on the first
suppressed hex popped.

**Partial depletion, not hard claiming.** A market takes a distance-decayed *share* of the
surplus it reaches — the haulage weight, 1 at the seat to 0 at the day's edge — so rich
country supports another market 8 km away while poor country supports none for 30. Hard
claiming would collapse spacing to one number and reinvent `city_min_separation` with
extra steps.

#### Catchments and population

One multi-source Dijkstra from every market seat over the travel-cost field, budget
`market_day_radius`, ties broken on `(cost, coord, owner)`. Each claimed land hex then
donates its adjacent unclaimed water hexes to its owner — a **fishery rim**, granted
rather than traversed, so a coastal market gets its `food_water_value` without claiming a
strait.

```
draw(m)       = Σ over catchment of surplus[h] * usable_fraction(cost_h, market_day_radius)
population(m) = max(1, round(draw(m) * people_per_food))
```

`usable_fraction` reaches **exactly zero** at the range limit. It is not a soft decay: it
is the distance at which the team has eaten the load. One constant sets reach and falloff
together.

**The travel-cost field is not the road-cost field**, and this was got wrong once. Reusing
the road model's river charge (then `river_hex_cost`, 12.0 per river hex) exceeded the 10.0 day budget outright, and `terrain_base_cost`'s
3×/10× bands made the median step 3.0, so markets reached 3 hexes instead of 10. Both are
excluded. Ascent uses **Naismith's rule** (`travel_ascent_per_hex`) rather than
`road_slope_cost`, because a catchment is walked, not engineered — the road curve prices
*grading* a slope and saturates at ten times base, which over eroded terrain shrinks a
catchment to a third of its proper reach.

Catchments are terrain-shaped, not round: measured disc-fill is 0.43 at the median, and
they visibly stop at ridges and stretch down valleys.

**Rivers in the travel-cost field** are read through `riverside.river_index`, once, so
every haulage question asks them the same way:

- **Crossing on foot** is charged once per river side crossed (`ford_cost`):
  `travel_ford_cost × span` away from a crossing, or `crossing_use_cost` at a ford or
  bridge. A trunk river therefore bounds a market's catchment instead of being invisible
  to it.
- **By barge**, cargo moves between riverside hexes only along a navigable **reach** — a
  run of sides whose discharge clears `navigable_min_discharge`, joined corner to corner —
  so two hexes are joined by water when both are banks of the same reach, or a bank and
  the open water its reach runs into. A cataract side breaks a reach, and the hexes beside
  it are a **portage**: the cargo lands above the falls and loads again below.
- **Timber** floats from any bank of a river past `timber_float_min_discharge`.

---

### 3.10c Land Use, Clearing and Rural Density — `organic`

[stages/land_use.py](../worldgen/stages/land_use.py)

**Purpose:** Decide what is actually done with each hex, size the markets on it, and count
who lives on the land. Replaces `CultivationStage` in the `organic` model.

**Reads:** `hex.soil`, `hex.territory`, `hex.territory_cost`, `state.metadata["market_seats"]`.
**Writes:** `hex.land_use`, `hex.cultivated`, `hex.rural_population`, `state.settlements`.

**Config:** § [4.8](#48-soil-food-and-habitability--37a-soil-39-habitability), § [4.10](#410-haulage-and-markets--the-organic-model).

#### Clearing is priced, and the margin is set by scarcity

The old rule drew a disc — eight hexes round a city, four round a town — so a city on thin
ground cleared exactly as far as one on a floodplain. What decides it now is rent:

```
rent(hex) = potential_food * usable_fraction(territory_cost, market_day_radius)
cleared   ⟺  rent >= clearing_margin * max(rent over that catchment)
```

Transport cost stands in for the effort of working land that far out, and **the bar is
relative to the best land in reach**. A market with a floodplain has a high bar and leaves
its hillsides to sheep; a market on uniformly thin ground has a low bar and ploughs the
scrub. The worse the land, the more pressure to use bad land — the extensive margin set
against the best alternative available, which is what rent theory actually says.

Measured on a temperate 128×128: markets whose best ground is only ARABLE plough down to a
mean soil rank of 2.29, markets with PRIME in reach stop at 2.69. An absolute threshold
cannot produce that — under one, a poor market simply clears less.

Von Thünen's rings still fall out, because rent falls with cost. What is new is that a
ring's *width* varies with what its catchment holds.

It needs one knob and one pass: no per-soil clearing costs, and no fixed point to iterate,
because rent depends on the catchment and the soil and both are settled before this runs.

```
water                                   → WATER
UNUSABLE soil                           → WASTE
outside every catchment, ploughable     → WOOD     (the wildwood: reached by nobody)
outside every catchment, otherwise      → WASTE
GRAZING soil in a catchment             → PASTURE  (grazing needs no clearing)
ploughable, rent clears clearing_margin → ARABLE
ploughable, rent clears pasture_margin  → PASTURE  (too poor to plough, near enough to graze)
ploughable, rent clears neither         → WOOD     (the parish's woods: its worst ground)
```

Ploughland and pasture are open ground: clearing turns forest, woodland and scrub
`land_cover` to `open`, so a map coloured by cover shows the fields. Food is unaffected;
`potential_food` reads the cover only for water and wetland. `SettledGroundStage` clears
the hex under every town and village, and the ring round a city, once every settlement is
founded and before the names are given.

`hex.cultivated` survives as a derived boolean (`land_use is ARABLE`) because the classic
village stages and the JSON schema both read it.

#### Cleared ground yields more, which is what makes clearing worth doing

```
actual_food = potential_food * yield_of[land_use]
```

Wood on prime soil feeds a third of what the same soil feeds under the plough. A settlement
therefore grows by assarting its hinterland rather than merely by sitting in it, and the map
shows cleared country round the markets with wood standing in the gaps.

#### Sizing lives here, so population has one owner

`MarketStage` plants and allocates on **potential** food — a settler picks land for what it
will yield once worked, not for the wildwood standing on it. This stage clears, then founds
and sizes each market from **actual** food. Nothing writes a provisional population that a
later stage overwrites.

#### Rural density is the same sum read the other way round

```
rural_population = actual_food * (1 - marketable_surplus_fraction) * people_per_food
```

The peasantry has been in the arithmetic since markets were built: a market draws
`marketable_surplus_fraction` of what its catchment yields, so the other two thirds feeds the
people who grew it. Defining it this way means the two figures reconcile by construction
rather than by calibration.

**Result at 128×128, seed 42:**

| climate | arable | pasture | wood | waste | people/km² | rural share |
|---|---|---|---|---|---|---|
| temperate | 10% | 8% | 47% | 27% | 38.5 | 87% |
| mediterranean | 4% | 7% | 15% | 66% | 17.6 | 85% |
| arid | 2% | 2% | 7% | 81% | 10.8 | 88% |
| boreal | 4% | 4% | 16% | 67% | 12.1 | 89% |
| tropical | 6% | 9% | 47% | 30% | 24.0 | 90% |

"Waste" is the medieval word and the right one: unenclosed ground nobody is working.

---

### 3.10d Resource Settlements — `organic`

[resources.py](../worldgen/stages/resources.py). Three kinds of place that no fertility puts
where they belong, founded after city promotion and before roads so the network reaches
them. Each is fed from beyond itself, like a city, so none conjures people: every workforce
is drawn from the settlements that can haul food to it, weighted by `usable_fraction` over
the bulk cost, none giving more than `resource_draw_share` of its weighted population. The
map's total population is unchanged by this stage.

- **Ports** (TOWN, role `port`). `CityPromotionStage` records the quays on its provisioning
  routes that no settlement stands within `transship_radius` of, with the trade share the
  city kept for them. Quays within `transship_radius` of each other are pooled onto the
  busiest; where that share reaches `port_min_population` a port is founded and paid it.
  A port past `city_min_population` is a city.
- **Mines** (VILLAGE, role `mining`). Deposits are drawn at random over ground with at least
  `ore_min_relief_m` of relief, weighted towards the higher, at `ore_deposits_per_1000_km2`
  of that ground and `ore_min_separation` apart. A deposit is worked only if its metal can
  reach an outlet (a city, or a port-role settlement) by bulk haulage within
  `haulage_range_land` × `ore_haul_range_mult`; smelted ore is worth more per ton than grain,
  so it goes further, but not without limit. On temperate 128x128 maps every deposit has an
  outlet in reach — port-role markets stand every 10-15 km — so the rule binds only where
  water is scarce. A workforce drawn lognormally around `mine_workforce` goes to the town
  within `resource_attach_radius` if there is one, and otherwise founds a village.
- **Lumber camps** (VILLAGE, role `lumber`). On a WOOD hex on or beside water that floats
  timber — open water, or a bank of a river past `timber_float_min_discharge`, a lower bar
  than a barge needs (when rivers occupied hexes, about half of them against 15%
  navigable); worth the woodland within `lumber_radius`, discounted by the bulk haul to the
  nearest city. Placed best-first, `lumber_min_separation` apart, while that worth reaches
  `lumber_min_score`, employing `lumber_people_per_wood_hex` per wooded hex.

A village that the food within reach cannot bring to `resource_min_population` is not
founded. `InterurbanRoadStage` routes to mining and lumber villages as well as to cities
and towns. On seeds 42/7/3 at 128x128 this gives 5-9 ports, 2-6 mines and a handful of
camps.

---

### 3.10e City Provisioning, Trade and Freight — `organic`

[cities.py](../worldgen/stages/cities.py), [river_trade.py](../worldgen/stages/river_trade.py). Promotion picks which markets become cities
(§ [3.10b](#310b-market-centres--organic) for how markets are planted); what each city
grows to is decided by where the countryside's surplus goes, and by the trade between the
cities.

- **Provisioning goes where it fetches most.** Each market ships `city_draw_share` of its
  surplus, split between the cities in reach by *pull* — a city's size times the share of
  the cargo that survives the haul — raised to `city_pull_sharpness`. A big city outbids a
  near small one, as London drew grain from Norfolk and Yorkshire past nearer towns, while
  carriage still gives the nearest city most. The split runs `city_pull_rounds` times, each
  pulling with the sizes the last produced, so a capital can come to dominate. People
  follow the food, so population is conserved: urban share is set by the surplus fraction,
  and pull only decides how it divides between cities and towns.
- **Manufactures run between the cities.** Each city puts `manufactured_trade_share` of its
  people's worth into trade with the cities within `haulage_range_land` ×
  `manufactured_range_mult`, split by size and haul. Every shipment pays `transship_share`
  at each change of mode on its route to whoever stands at the quay, drawn half from each
  city; quays nobody stands at are left for a port (§ [3.10d](#310d-resource-settlements--organic)).
- **The interior trades down its rivers.** Every town or city on a navigable reach, or a
  step from one, and off the coast ships `river_trade_share` of its people's worth to the
  coastal town or city it reaches most cheaply within `haulage_range_land` ×
  `river_trade_range_mult` — downstream only: afloat on one water, round a cataract on its
  portage, across a lake to its outlet, along the sea to the quay, and never to a bank
  draining less than the one before (`river_trade.downstream_rank`). It pays
  `transship_share` at every quay on the way, the landing and loading again at each portage
  among them, drawn half from each end; quays nobody stands at are left for a port, which
  is how a cataract with no town at it gets one. This is the trade Aswan, Louisville and
  the fall-line towns lived on, which provisioning is too short-haul to carry. Each flow is
  kept with its whole path in `metadata["river_trade"]` as `[origin q, r, destination q,
  r, people it feeds, [[q, r], …]]`, so anything charged where a cargo passes — a quay, a
  portage, a toll — can read which flows pass it. A river that leaves the map carries no
  trade: its coast is off the map.
- **Freight wears the roads.** Every flow — provisioning, manufactures, river trade, and ore
  from the mines, which goes to the best-paying city by the same pull — is recorded in
  `metadata["freight"]` as `[origin q, r, destination q, r, people it feeds, kind]`.
  `InterurbanRoadStage` puts journeys on each route: `road_raw_freight_per_person` for
  provisioning and ore, `road_goods_freight_per_person` for manufactures. Timber floats and
  river trade goes by boat, so neither wears a road. On seeds 42 and 7, 54-67% of city-to-city goods by weight goes by sea, and
  the land legs of those routes are all secondary or better.

---

### 3.11 Interurban Roads

[stages/interurban_roads.py](../worldgen/stages/interurban_roads.py)

**Purpose:** Build the inter-city road network (PRIMARY and SECONDARY
tiers). Uses gravity-model traveller simulation over A*-pathed routes,
with self-reinforcing pheromone trails.

**Reads:** `hex.terrain_class`, `hex.elevation`, `state.river_sides`,
`hex.coord`, `state.settlements` (CITY and TOWN tiers only).
**Writes:** `state.road_edges`, `state.sea_edges`, `hex.road_connections`,
`RiverSide.tags` (`"ford"` / `"bridge"`), `hex.tags` (`"switchback"`),
`hex.habitability_village` (+0.2 boost).

**Config:** `road_travellers_per_pop`, `road_travellers_max`,
`road_gravity_exponent`, `road_pheromone_factor`,
`road_flat_cost`, `road_water_cost`, `road_embark_cost`, `road_disembark_cost`,
`road_river_crossing_base`, `road_river_crossing_flow`,
`road_slope_cost`, `road_slope_free_pct`, `road_slope_cap_pct`,
`road_slope_cap_mult`, `road_min_traffic`, `road_river_traffic_min`,
`road_primary_pct`, `road_secondary_pct`, `hex_size_m`.

#### Cost model — [stages/road_cost.py](../worldgen/stages/road_cost.py)

The A* used by every road stage is in
[hex_grid.py:86–139](../worldgen/core/hex_grid.py#L86); it takes a
**node-cost** function (cost to *enter* a hex) and an **edge-cost**
function (cost of the transition between two hexes).

**Node cost** ([road_cost.py:32–46](../worldgen/stages/road_cost.py#L32),
combined in [interurban_roads.py:35–39](../worldgen/stages/interurban_roads.py#L35)):
```
base_cost = match terrain_class:
    OPEN_WATER | INLAND_WATER → road_water_cost  (default 0.05)
    COAST | LAND              → road_flat_cost   (1.0)

# No steepness term. The climb is priced on the *edge* by
# road_delta_elevation_per_hex, from the actual metres of rise between the
# two hexes; a per-hex surcharge on top billed the same ascent twice, and
# billed it wrongly wherever the band and the grade disagreed.

pheromone = road_pheromone_factor * traffic_so_far[hex]

node_cost = max(0, base - pheromone)
```

Roads follow river valleys — Roman "river roads" — but **nothing pays them to**.
There was a `road_bank_discount` here, a fraction knocked off the cost of any hex beside
a river. It was deleted, because the pull it claimed was almost entirely geometry: a
valley floor is the low, level, well-watered ground that already leads somewhere, and the
cost model rewards all three without being told about rivers at all.

Measured on a 128×128 temperate map, roads run on a riverbank **2.78×** as often as bank
occurs in the dry land. With the discount removed that falls to **2.52×** — nine tenths of
the effect survives the term that was supposed to cause it. What the discount bought was
not worth two config knobs and a paragraph of tuning.

Rivers run along hexsides, so a road beside one is simply on one bank and which side of
the river it — and anything standing on it — is on is always readable; nothing has to keep
it off the water.

**Crossing a river is an edge cost, charged once** (`river_crossing_edge_cost`). A step
from one hex to the next across a river side costs

```
road_river_crossing_base + road_river_crossing_flow * flow      # flow: the side's 0–1 rank
```

on top of the ordinary edge. The base is the crossing itself — a ford's approaches or a
bridge's abutments — and the flow term the span. When rivers occupied hexes a road paid a
per-hex charge (`road_river_hex_cost`) to enter one and was banned from running along the
channel, so the bank it was on stayed readable; a delta or a braid could then seal land
off, and ferries (`road_ferry_max_hop`) had to join it. All of that is retired: there is no
channel to run down, a crossing is one step, and no land is cut off by a river.

#### Traveller simulation — [interurban_roads.py:44–91](../worldgen/stages/interurban_roads.py#L44)

For each settlement, emit `population × road_travellers_per_pop` travellers,
capped at `road_travellers_max`. Process them busiest-origin-first — the
pheromone makes order decide which route is worn first and which then snap
onto it, so trunk routes are laid before the journeys that tributary into
them; a random order had minor journeys laying track for trunk routes to
follow. Ties keep settlement order, so it stays deterministic.

Each traveller picks a destination via gravity:
```
dist[d]   = max(1, hex_distance(origin, d))
weight[d] = population[d] / dist[d] ^ road_gravity_exponent      (default 2.5)
prob[d]   = weight[d] / sum(weight)        # excluding origin
```
Then routes to that destination — but **to the network, not to the hex**. A
traveller bound for a town does not need a road of their own all the way there;
they need to reach the road that already goes there. So the search
(`astar_to_any`) runs against every hex from which the destination is already
reachable along roads that exist, and stops at whichever it touches first. The
rest of the journey is that road. The first traveller finds nothing and paths
the whole way, becoming the road everyone after them joins.

This is what stops the network being a mat. Pathing all the way to the seat
had each route find its own line, and A* — whose heuristic assumes 1.0 per
step, and so misprices anything cheaper — cannot reliably find the same line
twice, so routes ran *beside* one another rather than joining. Aiming at the
network means a route joins it by construction rather than by the
pathfinder's good luck.

It is also far cheaper, because the search ends at the first road it meets
rather than at the far side of the map. Measured at 128×128 against pathing
to the hex: **217k node expansions against 1,329k, a road stage of 1.6s
against 7.4s, and 11.8% of the land covered against 19.3%** — with braiding
at 9.7% against 32.7% and clean degree-2 corridor at 70% against 43%. Plain
Dijkstra, which is optimal and therefore the ceiling, gives 12.3% coverage
and 4.1% braiding for 39.7s: routing to the network gets a *better* network
than optimal point-to-point routing, ten times faster, because it is
answering a better question.

Traffic on every hex of the path is incremented.

To save A* calls, each (origin, destination) pair is cached as a
**canonical route**. The first traveller does the pathing; everyone after
re-uses it.

There used to be a second cache in front of that: `_stitch_via_junction`
welded two existing legs together at an intermediate settlement rather than
pathing directly. It never compared the stitch against the direct route — A*
ran only when *no* stitch candidate existed at all — so once a handful of
legs existed almost everything after was a concatenation, and concatenations
became legs for the next stitch. Measured at 128×128: **1,690 of 1,944 routes
(87%) were never pathfound**, the median stitched route ran 168 hexes against
66 for a routed one, and the worst was 1,047 hexes between endpoints 24 km
apart. It is deleted.

After tiering the network is **split into road and sea**. An edge with a foot in
open water is a sea leg, not a road, and goes to `WorldState.sea_edges`; only dry
edges stay in `road_edges`. Routes cross water because water is cheap to cross,
rightly so — sea carriage ran at a fraction of land carriage before the railway —
but while the two were mixed the distinction could not be drawn. On the 128×128
reference map **half the network by hex count was water**, "road coverage" of
9.4% was really 6.0%, and the single connected network was single only *through*
the sea: by land alone it was forty networks tied together by eight crossings
averaging 62 water hexes apiece.

`_join_by_land` then adds what the traffic model declined to, on a cost function
that refuses water outright: **one road network per landmass**. A network in which
neighbouring markets can only be reached by boat is not a road network and a
wargame cannot march down it. `VillageTrackStage` is land-only for the same reason.

It joins **road components, not settlements**, and the difference is load-bearing.
A spur that lands a sea leg survives `prune_orphan_roads` — a road to a harbour is
a road to somewhere — but it holds no settlement, so a settlement-only rule had
nothing to join it to; `ChokepointStage` then founded a village on such a spur and
the village came out cut off by land from the 37 settlements it shared ground with.
Joining components means anything founded later is on the one network by
construction, whatever founds it. A settlement standing on no road at all counts as
a component of one, so the component rule subsumes the seat rule it replaced.

It also builds *less* road than the seat rule did. That version added only the
routed path to its joined set rather than the whole component the seat belonged to,
so a second seat in the same stranded piece was routed again and got a road of its
own to the trunk. Merging whole components connects each piece once: on the 128×128
temperate reference map the network fell from 1,560 hexes to 1,338 with the same
connectivity.

After that, two passes tidy the network. `route_through_settlements` bends
any road skirting a settlement so it passes through instead (§ 4.19), and
`prune_orphan_roads` drops any component reaching no settlement —
`road_river_traffic_min` admits a riverbank edge on a single traveller, so a
stretch of towpath can qualify while joining nothing. The
connectivity guarantee then runs over **every** settlement, not just the
cities: it used to require two or more cities, so the organic model had
nothing watching it, and the map stayed connected only because stitching made
most routes concatenations of the same few legs.

#### Tier classification — [interurban_roads.py:93–114](../worldgen/stages/interurban_roads.py#L93)

After all travellers are processed, hexes are filtered:
```
eligible = edges where traffic >= road_min_traffic                     (default 3)
        OR the edge runs along a riverbank and traffic >= road_river_traffic_min (default 1)
# along a bank: both hexes beside the same river, and not across it — a step across
# is a crossing, and earns no towpath
```
Sort by traffic descending, then:
- top `road_primary_pct` (10 %) → PRIMARY
- next `road_secondary_pct` (30 %) → SECONDARY
- rest → no tier (TRACK is reserved for village connectors)

A canonical route's tier is the **highest** tier any hex on it earned
(`_path_min_tier`, [interurban_roads.py:191–196](../worldgen/stages/interurban_roads.py#L191)).
Routes whose hexes are all below the traffic threshold are dropped
entirely.

#### Tier gaps — [road_cost.py `fill_tier_gaps`](../worldgen/stages/road_cost.py)

Tiers are cut per edge, so where two routes take neighbouring hexes for a step the traffic
splits and a trunk road dips a class and comes back. After the network is built, a
lower-class stretch of at most `road_tier_gap_max_edges` edges running from the end of a
PRIMARY (then SECONDARY) road to another, unconnected piece of that class is promoted to
it. A connector leaving the middle of one trunk for another keeps its class.

The land join that reconnects a road split by a sea leg takes the lower of the best tiers
within two hexes of each of its ends, rather than always TRACK.

#### Connectivity guarantee — [interurban_roads.py:198–276](../worldgen/stages/interurban_roads.py#L198)

If the traffic-driven graph leaves any city in a separate component,
`_guarantee_city_connectivity` runs A* (using only the plain terrain
costs, *no* pheromone) from each isolated city to the largest connected
component, and inserts those paths as PRIMARY roads. Bounded to
`2 * len(cities)` iterations to prevent runaway cases.

#### Side effects

- **River-crossing tags**: `tag_river_crossings(road_edges, state, cfg)` tags each
  river side a road crosses, from what the water is rather than how busy the road is,
  so the result does not depend on the order routes were built in:
  - over water too big to wade (`side_span` > 1) → `"bridge"`, whatever the road's
    tier: a track over a navigable river is not a track through it;
  - over wadeable water → the ford (`"ford"`), unless the road is PRIMARY, which is
    worth a bridge anyway;
  - a side already bridged is left alone, and a bridge site no road crosses is
    untagged — `CrossingStage` marks where traffic would justify one, and a site the
    network never reached was never built. So **every bridge on a finished map carries
    a road.** Fords stay either way: they are terrain.
- **Habitability boost** (+0.2, capped at 1.0) applied to every land hex
  adjacent to a road
  ([interurban_roads.py:140–147](../worldgen/stages/interurban_roads.py#L140)).
  This feeds VillagePlacementStage so that road corridors attract
  villages.

---

### 3.11a Chokepoints — `organic`

[stages/chokepoints.py](../worldgen/stages/chokepoints.py)

**Purpose:** Found the settlement tier below the market, on bridgeheads and passes that
carry real traffic. This is the tier `organic` withholds from `VillagePlacementStage` —
gated on holding something rather than sprinkled across the countryside.

**Reads:** `state.road_edges`, `state.river_sides`, `hex.soil`, `hex.tags`, `hex.elevation`, `hex.territory`,
`hex.territory_cost`, the food surface.
**Writes:** `state.settlements`, `hex.settlement`, and a `pass` tag on every saddle that
qualifies.

**Config:** § [4.10](#410-haulage-and-markets--the-organic-model).

#### It runs after the roads, and that is the whole idea

A chokepoint is not a good site that happens to have traffic; it is a bad site that has
traffic anyway. Only the built network can say which crossings carry any, so this cannot
run before `InterurbanRoadStage` — and because every candidate already sits on a road, the
settlements it founds perturb nothing. No route is recut, so the traffic that justified a
village is the same traffic after it exists.

#### Both halves of the gate bind

```
candidate  =  carries a road of at least chokepoint_min_road_tier
              AND (is a bridgehead, or is a pass)
              AND is settleable, and unoccupied
```

A **bridgehead** is an end of a bridged river side that a road actually crosses — a river
runs along a hexside, so the two hexes either side of it are the bridge's two ends. Of the
two, the town grows on the better-soiled one: a bridge over a desert river has its village
on the irrigated bank, not on the sand opposite; both count where the ground is as good on
either. Without the road test the tier founded villages beside phantom crossings — on one
96×96 fixture, six of seven stood at bridges no road touched.

A bridge on a farm track is a plank, not a town; a busy road over open country is passing
through nowhere in particular. On a 128×128 temperate map about twenty features clear
both, which is what actually sets the size of this tier — `chokepoint_min_draw` only says
which of those are worth a glyph.

#### A pass is a saddle the ground either side forbids going round

Read off the neighbour ring: walk the six in order and count the *runs* of ground standing
above you. None is a summit or a pit, one is a hillside, two or more is a col. The relief
reported is the **lesser** of the two flanks, because a pass is only as walled as its
weaker side — one cliff and one gentle rise is somewhere you simply walk over.

The threshold is `terrain_steep_gradient_m` rather than a setting of its own. A hex is
1 km across, so that figure is already the gradient at which the map says "pack animals,
terraces, no wheels", and a separate knob could only disagree with the terrain bands about
what a road cannot climb.

**Passes are rare, and that is the model working.** `slope_edge_cost` charges every metre
of climb, so a router offered a way round a ridge takes it. On 1500 m of relief a temperate
map has 19 qualifying saddles and roads use 2 of them; at 2400 m there are 70 and roads use
none. A pass settlement appears only where the ground leaves no way round.

#### Sized from what the markets left behind

```
residual(hex)  =  surplus * (1 - usable_fraction(territory_cost, market_day_radius))
                  if the hex is in a market's catchment, else surplus
```

`gather` weights every hex by `usable_fraction` of the distance to its market, falling to
exactly zero at `market_day_radius` — so the complement is precisely the fraction the
market could not haul, and ground outside every catchment was never drawn on at all.

This is what stops the tier double-counting the one above it: **a village on a market's
doorstep finds nothing left and is not founded**, with no rule anywhere saying villages may
not stand near markets. Market populations are unchanged by this stage.

Planting is greedy over the residual, claiming a village's fields **outright** rather than
taking the decaying share markets use. Markets compete for the same countryside, which is
what makes their spacing follow the land; the fields around a village are walked out to and
back from daily and are not shared with anybody.

The floor is then applied to the **real** catchment draw, not to the estimate planting ranks
on, and the survivors are re-partitioned — so `chokepoint_min_draw × people_per_food` is
exactly the smallest village in people, and no village is left holding the smaller catchment
it had while a rejected neighbour was still in.

**Result at 128×128, seed 42, `continent_falloff_edges: [south]`:**

| climate | cities | towns | villages | of which on passes | median village |
|---|---|---|---|---|---|
| temperate | 3 | 73 | 6 | 2 | 277 |
| mediterranean | 1 | 38 | 1 | 0 | 120 |
| arid | 0 | 14 | 0 | — | — |
| boreal | 0 | 14 | 0 | — | — |

Against a median market town of 967 on the temperate map, and a largest city of 39,805 —
three tiers, each an order apart, and none of them a target count.

---

### 3.12 Cultivation (Cities & Towns)

[stages/cultivation.py](../worldgen/stages/cultivation.py) — `CultivationStage`

**Purpose:** Mark hexes as cleared/cultivated within a radius of cities and
towns.

**Reads:** `state.settlements`, `hex.land_cover`.
**Writes:** `hex.cultivated`.

**Config:** `cultivation_city_radius` (default 8), `cultivation_town_radius`
(default 4).

**Algorithm** ([cultivation.py:19–37](../worldgen/stages/cultivation.py#L19)):
for each CITY/TOWN settlement, walk every hex within its tier's radius
(via `hex_range`) and set `cultivated = True` unless the hex is in the
**RESISTANT** land cover set:

```
RESISTANT = {BOG, MARSH, BARE_ROCK, ALPINE, TUNDRA, DESERT, OPEN_WATER}
```
([cultivation.py:6–16](../worldgen/stages/cultivation.py#L6)).

The cultivation field is read by VillagePlacementStage to detect the
"frontier" — hexes that are cultivated but border uncultivated land,
ideal for new villages.

---

### 3.13 Village Placement

[stages/village_placement.py](../worldgen/stages/village_placement.py)

**Purpose:** Place villages by stochastic weighted sampling, biased toward
either the cultivation frontier or road corridors.

**Reads:** `hex.habitability_village` (already road-boosted), `hex.land_cover`,
`hex.terrain_class`, `hex.cultivated`, `hex.road_connections`,
`state.settlements`.
**Writes:** `hex.settlement`, `state.settlements`.

**Config:** `settlement_min_reachable`, plus the road-grade parameters for
the same reachability filter cities/towns use.

**Candidacy** ([village_placement.py:38–73](../worldgen/stages/village_placement.py#L38)):
a hex is a candidate if **all** of these hold:
- not OCEAN/LAKE
- no existing settlement
- `habitability_village > 0`
- `land_cover not in RESISTANT` (same set as cultivation)
- `grade_reachable_count(...) >= settlement_min_reachable`
- **and** at least one of: on the cultivation frontier, OR road-adjacent

Each candidate's weight starts at `habitability_village` and is multiplied:
- `× 2.0` if on the frontier
  ([village_placement.py:67](../worldgen/stages/village_placement.py#L67))
- `× 1.5` if road-adjacent

Both can stack (×3.0).

**Stochastic placement** uses the **Efraimidis–Spirakis weighted-sampling
without-replacement key**
([village_placement.py:80–81](../worldgen/stages/village_placement.py#L80)):
```
u = uniform_random per candidate
order = sort by  -u^(1/weight)   descending
```
This is equivalent to drawing weighted samples without replacement. Then
the stage walks `order` and accepts a candidate iff it is `>= 3` hexes
from every already-placed settlement (cities, towns, or villages)
([village_placement.py:89](../worldgen/stages/village_placement.py#L89)).

Population is uniform random in `[100, 1_000]`; role uses the same
`_assign_role` as cities/towns. There is no target count — placement
runs until candidates are exhausted.

---

### 3.14 Village Tracks

[stages/village_tracks.py](../worldgen/stages/village_tracks.py)

**Purpose:** Connect each village to the existing road network via a TRACK
road.

**Reads:** `hex.road_connections`, `state.settlements`,
`hex.terrain_class`, `hex.elevation`, `state.river_sides`.
**Writes:** `state.road_edges` (new TRACK edges), `hex.road_connections`,
`RiverSide.tags` (ford/bridge, by `tag_river_crossings`).

**Algorithm** ([village_tracks.py:19–66](../worldgen/stages/village_tracks.py#L19)):

```
targets = all road hexes  ∪  all city/town coords
for village in villages:
    sort targets by Manhattan-ish (q,r) distance from the village
    for candidate in sorted_targets:
        path = astar(village -> candidate, node_cost, edge_cost)
        if path with len >= 2: break
    add path as a TRACK Road, update road_connections
    add the village's hex to targets   # later villages can re-use it
```

Cost functions are identical to the interurban stage's, *minus the
pheromone term* — village tracks don't compete for shared traffic, they
just want the cheapest viable route. River discount is still applied.

---

### 3.15 Village Cultivation

[stages/cultivation.py:40–54](../worldgen/stages/cultivation.py#L40) — `VillageCultivationStage`

Mirror of CultivationStage but using `cultivation_village_radius`
(default 2) and only iterating VILLAGE-tier settlements. RESISTANT land
cover types are skipped, same as before. Runs last so that it doesn't
interfere with the cultivation frontier signal used by VillagePlacement.

---

### 3.16 Naming

[stages/naming.py](../worldgen/stages/naming.py) — `NamingStage`, last in both models;
vocabulary in [naming/](../worldgen/naming/)

Replaces the placeholder names settlements are founded with (`grassland_market_3`) and
names the major rivers. Last because tier, role and population are only final after every
settlement stage, and because a stage appended at the end draws the last child seed:
`naming_cultures: 0` gives the same world with placeholder names.

1. **Languages.** `naming_cultures` languages are invented from the stage's RNG
   (`naming.conlang.Language`): a sound inventory, a syllable shape, a spelling table,
   head-first or head-last compounding, and how often names are phrases or wear down where
   their parts meet. A language makes one word per meaning, seeded from the meaning, so
   "ford" is the same syllable in every ford town. With `naming_substrate`, one more
   language is invented for the rivers.
2. **Culture regions.** Homelands are spread by farthest-point sampling over the land, then
   every hex goes to the homeland that reaches it cheapest. A step costs one hex, plus
   climb over `naming_region_climb_m`, plus `naming_region_river_cost` to step across a river side
   draining `naming_great_river_km2` or more, plus `naming_region_water_cost` per water hex — so
   frontiers fall on ridges and great rivers.
3. **Rivers**, largest catchment first, down to `naming_river_min_catchment_km2`: named in
   the substrate language, or by whoever holds the mouth.
4. **Settlements**, cities first. `naming.site.read_site` reads the hex and its ring — with
   the river features beside it, from the sides it touches (ford, bridge, cataract,
   rapids) and the corners it stands at (mouth, confluence, spring) — into
   weighted *heads* (ford, bridge, mouth, falls, pass, harbour, mine, clearing, hill,
   shore, and per-tier heads such as farm, hamlet, town, stronghold) and *qualifiers*
   (land cover, trees and beasts of the biome, colour, rich soil, high ground, salt, great
   or little, and the compass point from the nearest bigger place within
   `naming_direction_radius`). A head is drawn by squared weight; a qualifier from the
   site's, a founder's name (`naming_founder_weight`), or a nearby named river
   (`naming_river_weight`). A name longer than `naming_max_letters`, or within
   `naming_min_edit_distance` edits of any name already given, is redrawn.

**Culture packs.** A region can speak a pack instead of an invented language
(`naming_packs`, `naming_substrate_pack`). A pack is a YAML file — the built-in ones in
[naming/packs/](../worldgen/naming/packs/), your own in the folders `naming_pack_dirs`
lists — giving, for every head and qualifier above, the words a naming tradition used, and
templates for how they combine. [export/culture_packs.py](../worldgen/export/culture_packs.py)
reads the files, `naming.packs.schema.parse_pack` checks each one (every error names the
file and field), and one processor, `naming.packs.processor.PackCulture`, runs them all.
[CULTURE_PACKS.md](CULTURE_PACKS.md) is the authoring guide. Languages are invented for
every region either way and then replaced, so choosing packs never changes what the other
regions are called; each pack culture in `metadata["cultures"]` records its `pack`,
`pack_source` (builtin or user) and a `pack_hash` of its content.

**Renaming.** `worldgen rename` runs this stage alone over a saved world.json
(`stages.rename`), with the world's recorded config and new naming settings. It gives the
stage the child seed a full run would have drawn for it from the naming seed (the world's
own by default), so a world renamed with its own seed and config keeps every name. River
names are cleared before naming, and the naming seed is recorded as
`metadata["naming_seed"]`.

Each settlement records its `culture` and an English `etymology` ("ford on the Vassa");
`metadata["cultures"]` lists the languages. Labels on the SVG and PNG exports are placed by
[export/labels.py](../worldgen/export/labels.py): sized by tier, rivers italic along their
middle reach, and none overlapping.

---

## 4. Configuration Reference

All defaults live in [worldgen/core/config.py](../worldgen/core/config.py).
Validation rules are in `__post_init__` and are noted inline below.

**Everything here is a physical quantity in real units** — metres, degrees Celsius,
millimetres of rain a year, kilometres, square kilometres. That was not always true: the
generator used to carry elevation, temperature and moisture on normalised `[0, 1]` axes,
which meant a threshold written against one of them silently meant something different on
every map, because each axis was re-stretched to the range that map happened to occupy.
Six such normalisations were removed. Where a setting replaced one, the row says so.

Two settings are shipped in [worldgen/default_config.yaml](../worldgen/default_config.yaml)
with commentary; `worldgen init-config` writes that file out. `tests/test_docs.py` checks
that every field below exists and that no field is missing, so this table cannot drift
from the dataclass again without the suite failing.

### 4.1 Grid

| Param | Type | Default | Range | Effect |
|---|---|---|---|---|
| `width` | `int` | `200` | ≥ 1 | Map width in hexes (`1 hex = 1 km` by convention) |
| `height` | `int` | `200` | ≥ 1 | Map height in hexes |
| `grid_layout` | `str` | `"offset"` | `axial` \| `offset` | Grid shape — see below |
| `model` | `str` | `"organic"` | `classic` \| `organic` | Which settlement and road model the pipeline runs. In the config rather than only on the CLI so `world.json` records which model made the map — seed plus config is the reproduction record. The `--model` flag overrides it |

`grid_layout` decides which hexes a world is built from:

- **`axial`** — `q` runs `[0, width)`, `r` runs `[0, height)`. That rhombus is sheared
  by the flat-top pixel transform, so the drawn map is a leaning parallelogram with a
  straight edge on all four sides.
- **`offset`** — odd-q offset column/row, stored as the axial coordinate each
  column/row names. The drawn map is a **rectangle**: odd columns sit half a hex lower
  than even ones, so the north and south edges are **ragged** while east and west stay
  straight.

Hexes are keyed by axial coordinates in both layouts, so adjacency, distance and
pathfinding are identical; only the set of hexes differs. Stages that work on a
`(width, height)` array go through `WorldState.coord_at(col, row)` and
`WorldState.grid_index(coord)` to cross between array indices and hex coordinates, and
`WorldState.on_border(coord)` is the layout-aware map-edge test the hydrology and
water-body stages drain to.

Columns are spaced `1.5 * hex_size` apart and rows `sqrt(3) * hex_size`, so an offset
map comes out square at `height ≈ 0.87 * width` — `128 x 111`, for instance.

### 4.2 Elevation — § [3.1](#31-elevation)

Elevation is **metres above sea level**. Sea level is the datum, so it is zero by
definition and is not a setting — the old `sea_level` fraction is retired. What kind of
country the map is comes from the two vertical-scale settings below.


| Param | Type | Default | Range | Effect |
|---|---|---|---|---|
| `max_elevation_m` | `float` | `1500.0` | `> 0` | Highest ground above sea level. The single most consequential setting for what country this is: `800` gives downland, `1500` mixed uplands, `3000` an Alpine massif |
| `elevation_profile` | `str` | `"lowland"` | `lowland`, `rolling`, `upland`, `custom`, `none` | How much of the land lies at each height. Before erosion (`ElevationStage`) the land hexes are ranked by height and each is given the height at its rank in a log-normal — a bell skewed low with a long tail of high ground — so every map has the same proportions whatever its noise did. Presets (most common / median height): lowland 50 / 150 m, rolling 100 / 250 m, upland 200 / 450 m. `custom` uses the two settings below; `none` keeps the noise. `ErosionStage` resets sea level after the droplets so the land share going in is kept, and puts the land back on the profile at the end, so the final heights match it exactly. Generated terrain only. Replaces `elevation_hypsometry_exponent`. |
| `elevation_profile_mode_m` | `float` | `50.0` | `> 0`, `< median` | The most common land height for `elevation_profile: custom`. |
| `elevation_profile_median_m` | `float` | `150.0` | `< max_elevation_m` | The median land height for `elevation_profile: custom`, before the log-normal is cut off at `max_elevation_m` — a median near the ceiling comes out lower. |
| `seabed_depth_m` | `float` | `200.0` | `> 0` | How deep the sea floor lies at the map edge. A continental shelf, not an abyss. How much of the map ends up underwater follows from this and `max_elevation_m` |
| `coast_max_elevation_m` | `float` | `100.0` | `≥ 0` | Land no higher than this beside the sea is classed COAST |
| `noise_octaves` | `int` | `6` | ≥ 1 | Number of fBm octaves. Higher = more detail at the cost of speed |
| `noise_persistence` | `float` | `0.5` | `(0, 1]` typ. | Amplitude multiplier per octave (`amp *= persistence^i`). Higher = rougher terrain |
| `noise_lacunarity` | `float` | `2.0` | `> 1` typ. | Frequency multiplier per octave (`freq *= lacunarity^i`) |
| `noise_scale` | `float` | `3.0` | `> 0` | Coordinate scale: domain spans `[0, noise_scale]`. Higher = more variation per hex |
| `domain_warp_strength` | `float` | `0.3` | `≥ 0` | Magnitude of the domain-warp offset. `0` disables warping; higher gives more organic coastlines |
| `continent_falloff` | `bool` | `True` | — | Apply edge falloff so the sea rings the map and every river has a coast to reach. `False` gives a landlocked map whose interior basin is the terminal sink — endorheic by geometry |
| `continent_falloff_edges` | `tuple[str, ...]` | `('north', 'south', 'east', 'west')` | subset of the four | Which edges the sea comes in from. Drop an edge to let the land run off the map there instead of ending in a coast. `()` is the same as `continent_falloff: false`. Rivers can still drain off any border, ocean or not, so a partly-open map is not a trapped one |
| `continent_shelf_hexes` | `int` | `10` | `≥ 1` | Width in hexes (km) of the shelf over which land drops to the sea. In hexes rather than a fraction of the map, so the coastal gradient is the same per km at any size. Capped at a quarter of the shorter side |
| `continent_shelf_variance` | `float` | `0.35` | `[0, 1]` | How much the shelf's inner edge wanders. `0` gives a coast of even width; higher makes bays and headlands. The terrain noise already moves the shoreline a good deal, so this is a nudge |
| `elevation_gradient_m` | `(float, float)` | `(0.0, 0.0)` | — | Directional tilt `[east, south]` **in metres**, applied after shaping. `(0, -600)` stands the north edge 600 m higher and runs the map downhill to the south. Replaces `elevation_gradient`, which was a fraction of an abstract range |

### 4.2a Elevation from an Image — § [3.1a](#31a-elevation-from-an-image)

| Param | Type | Default | Range | Effect |
|---|---|---|---|---|
| `heightmap_path` | `str \| None` | `None` | — | Path to an image to read the terrain from, resolved against the working directory. Setting it swaps `ImageElevationStage` in for `ElevationStage` |
| `heightmap_mode` | `str` | `"elevation"` | `elevation`, `coastline` | `elevation` reads the image as a greyscale heightmap; `coastline` reads it as a land/sea stencil and fills it with generated terrain |
| `heightmap_land_threshold` | `float` | `0.5` | `[0, 1]` | Coastline mode. Brightness at or above which a pixel is land. Ignored where the image has a meaningful alpha channel |
| `heightmap_invert` | `bool` | `False` | — | Coastline mode. Treat the darker side of the threshold as the land instead; with an alpha stencil, treat the transparent side as the land |
| `heightmap_coast_falloff` | `bool` | `False` | — | Coastline mode. Also apply the rectangular edge falloff on every edge (regardless of `continent_falloff_edges`), ringing the map with sea. Off by default, so the stencil is authoritative |

### 4.3 Terrain Classification — § [3.3](#33-terrain-classification)

Terrain classes are **bands of gradient, in metres of rise per kilometre**. They describe
how the ground lies, not how high it is, so a high plateau is level ground and is classed
as such. Absolute rather than a fraction of the elevation range, so a band means the same
thing whatever the map's vertical scale — the old `terrain_hill_gradient` made a mountain
120 m/km on one map and 20 m/km on another.

| Param | Type | Default | Range | Effect |
|---|---|---|---|---|
| `terrain_rolling_gradient_m` | `float` | `30.0` | `≥ 0` | m/km above which ground stops being FLAT. Below it: level going — plough it, cart across it |
| `terrain_steep_gradient_m` | `float` | `100.0` | `> rolling` | m/km above which wheels stop working. ROLLING below, STEEP above |
| `terrain_escarpment_gradient_m` | `float` | `250.0` | `> steep` | m/km above which it is a break of slope: on foot and with effort |

### 4.4 Erosion — § [3.2](#32-erosion)

| Param | Type | Default | Range | Effect |
|---|---|---|---|---|
| `erosion_droplets_per_hex` | `float` | `3.0` | `≥ 0` | Droplets run **per land hex**, not per map, so the dose means the same at any size. Replaces `erosion_iterations`, a flat count that gave a 32×32 map 14.6 droplets per hex and a 128×128 map 0.9 — a sixteenfold spread, and most of why small maps came out as Alpine massifs. It decides whether the map has valleys: below about one per hex the rivers only scratch a line into the noise and there is no floodplain. It is also a climate setting, since the orographic term lifts on height above sea level and wearing the high ground down flattens the rain shadow |
| `erosion_inertia` | `float` | `0.05` | `[0, 1]` | Direction smoothing. `0` = pure gradient descent; near `1` = the droplet ignores terrain |
| `erosion_capacity` | `float` | `4.0` | `> 0` | Sediment-carrying capacity multiplier. Higher = droplets erode more aggressively |
| `erosion_deposition` | `float` | `0.3` | `[0, 1]` typ. | Fraction of excess sediment deposited each step when over capacity |
| `erosion_erosion_rate` | `float` | `0.3` | `[0, 1]` typ. | Fraction of the capacity deficit eroded each step |
| `erosion_channel_affinity_gain` | `float` | `0.5` | `≥ 0` (validated) | Affinity bump per erosion event. Higher = stronger channel reinforcement |
| `erosion_affinity_update_interval` | `int` | `500` | `≥ 1` (validated) | Droplets between channel-affinity re-weighting passes |
| `erosion_delta_min_load` | `float` | `0.15` | `≥ 0` (validated) | Sediment a droplet must still carry on reaching the sea for it to build anything. Below this the load is treated as carried away along the shore. Without it every droplet trickling off a nearby hillside deposited where it entered the water, silting the shelf evenly instead of building deltas at the river mouths |

The erosion constants are shares of the map's relief rather than physical quantities, so
the stage converts to a normalised copy at its boundary and back to metres on the way out
— against the **known** span, so it is a fixed change of units and not a per-map stretch.

#### Valley widening

Droplets only incise: each cuts along its own path, so the model carves narrow V-notches
and nothing ever widens them. Real valleys get their width from the channel migrating
sideways over geological time, planing the floor flat between bluffs — the Mississippi's
floor is tens of kilometres across, the Nile's ten to twenty, and neither was cut by water
going straight down. Carving and drainage decide each other, so this runs as a short
convergence loop rather than a single pass. Floodplains come out 79% wider.

Each pass drains the working surface on the same corner graph hydrology uses
(`corner_drainage.py`, with `corner_floor_blend`), so the valleys are cut where the rivers
will run. Water runs along hexsides, so incision lowers a side: both banks, toward the new
height of the corner upstream, together with that corner's own lowest hex. Valley widening still works outward
from channel hexes, which is the shape a floodplain has.

| Param | Type | Default | Range | Effect |
|---|---|---|---|---|
| `erosion_incision_m_per_pass` | `float` | `12.0` | `≥ 0` (validated) | Metres a reference channel lowers per carve pass — the dial for how hard rivers cut. `K` in `K·A^m·S^n` is derived from this rather than set directly, so the tunable stays in metres whatever the exponents are. `0` disables incision |
| `erosion_incision_area_exponent` | `float` | `0.5` | `≥ 0` (validated) | `m`. The exponent that creates the effect: it is the contrast between what a trunk cuts and what the hillslope beside it cuts, and hence whether one channel ever captures another. At the default a 500 km² channel lowers about 22× as fast as 1 km² of ground. `0` reverts to eroding by slope alone, which is the behaviour that gave unbranched parallel threads |
| `erosion_incision_slope_exponent` | `float` | `1.0` | `≥ 0` (validated) | `n`. `m/n = 0.5` is the textbook concavity. `1.0` also makes the step linear in elevation, which is what lets the floor against the receiver act as an exact stability guard; other values can overshoot |
| `erosion_incision_reference_km2` | `float` | `500.0` | `> 0` (validated) | `A_ref`, the catchment in km² at which `erosion_incision_m_per_pass` is the lowering. A mid-sized trunk on a 64×64 map |
| `erosion_incision_reference_slope` | `float` | `0.01` | `> 0` (validated) | `S_ref`, the slope at which `erosion_incision_m_per_pass` is the lowering. `0.01` is 10 m/km |
| `erosion_incision_min_gradient_m` | `float` | `0.01` | `≥ 0` (validated) | The gap kept above the receiver, in metres. Incision walks outlets first, so the receiver has already dropped when a cell is reached: this floor never prevents deepening, it only prevents flow being turned back on itself, which is what keeps the surface free of new sinks for hydrology to refill |
| `erosion_incision_max_cut_m` | `float` | `40.0` | `≥ 0` (validated) | Cap on one cell's lowering in one pass, in metres. A guard against an inherited cliff, not a parameter of normal operation |
| `erosion_droplet_overcut_m` | `float` | `0.0` | `≥ 0` (validated) | How far a droplet may cut past its own receiver, in metres. `0` because the droplets' job is transport and roughening while incision does the deepening; non-zero makes droplets punch pits the sink fill must span, which buys no capture — capture needs a sustained trunk-to-hillslope contrast a point process cannot deliver |
| `erosion_smoothing_sigma` | `float` | `0.5` | `≥ 0` (validated) | Width of the blur applied after the droplets and before the carve loop, in cells. Running before rather than after is what keeps it from damping the notches incision cuts. `0` skips it |
| `valley_carve_passes` | `int` | `3` | `≥ 0` (validated) | Cut, re-measure the drainage, cut again. One pass does not do it: widening a valley moves the water into it, so the network measured before the first cut is not the one that exists after |
| `valley_width_max` | `float` | `6.0` | `≥ 0` (validated) | Cap on valley half-width, in hexes. `0` disables widening entirely |
| `valley_width_exponent` | `float` | `0.6` | `≥ 0` (validated) | Discharge → width. `0.5` is the textbook root |
| `valley_floor_slope_m` | `float` | `2.5` | `≥ 0` (validated) | Rise per hex away from the channel, in metres. Small but not zero, so a floodplain drains toward its river rather than ponding |
| `valley_max_relief_m` | `float` | `100.0` | `≥ 0` (validated) | How much height a river may plane away per pass, in metres. What stops a valley eating the landscape: a channel cuts a floodplain out of ground near its own level but cannot take down a bluff standing well above it, so anything higher is valley wall and stays. This is what makes valleys self-limiting — pinched in a gorge, broad where the ground is already low |
| `valley_channel_fraction` | `float` | `0.02` | `(0, 1]` (validated) | What fraction of the land counts as channel, by discharge — the cells valleys are planed outward from |

#### Alluvium

`_drop_particle` works out how much sediment each droplet lays down and spends the number
on the elevation alone, which throws away the more useful half of it: how high the ground
ended up is a poor proxy for what it is made of. A hillside cut down to a gentle grade and
a valley floor built up to the same height are the same elevation, the same slope, and
nothing alike to plough. `Hex.alluvium` records the sediment itself, as a depth in
`[0, 1]` against the map's own richest ground.

Two sources, and **the second is the larger**. *Net droplet deposition*, with erosion
subtracted — sediment picked back up has left, so a channel that deposits on one pass and
scours on the next is holding nothing, and summing only the deposits would call every busy
channel deep soil. This finds deltas and valley bottoms. Then *the meander belts*
`_widen_valleys` already computes, which is most of the floodplain on a map: a valley floor
is laid down by the channel moving *sideways*, so a planation pass can floor a whole valley
with silt and change the mean elevation across it hardly at all. Vertical deposition cannot
see it.

The two arrive in incomparable units — a sum of elevation changes, and a fraction of a
reach in cells — so each is brought onto its own `[0, 1]` before they are added, rather
than weighted against each other raw. Otherwise the dial between them means something
different on every map. Belt depth is scaled against the reach of its **own** channel, not
the widest on the map: scaling globally made `(flow/max_flow)**0.6` tiny for anything but
the trunk river, so every other valley read as bare and the map showed one bright ribbon.
A small river's floodplain is narrow, not stony.

The field is read off the erosion model and never fed back into it, so the elevations are
bit-identical whatever these are set to.

| Param | Type | Default | Range | Effect |
|---|---|---|---|---|
| `alluvium_floodplain_gain` | `float` | `1.0` | `≥ 0` (validated) | How much a floodplain belt counts as alluvial, at its channel. `0` leaves only droplet deposition, which finds deltas and valley bottoms but not meander belts |
| `alluvium_smoothing` | `float` | `1.0` | `≥ 0` (validated) | Blur in cells before normalising. Droplets deposit at points and soil does not. `0` disables |
| `alluvium_quantile` | `float` | `0.98` | `(0, 1]` (validated) | Depth treated as full, for the droplet term only. A quantile rather than the maximum, which is one cell at the front of one delta and would flatten every floodplain on the map against it |

### 4.5 Hydrology — § [3.5](#35-hydrology)

A channel forms where enough water passes to keep one open: **discharge = catchment area
× runoff depth**. This replaced `river_flow_threshold`, which was documented as a flow
minimum and implemented as a rank — the top 5% of land by accumulation — so desert and
rainforest alike got 5.6% of their land under channel. Now arid country drains 1.2% with
no navigable river at all, and tropical 12.6%.

| Param | Type | Default | Range | Effect |
|---|---|---|---|---|
| `channel_min_discharge` | `float` | `6000.0` | `> 0` | Catchment km² × runoff mm needed to cut a channel. Also the dial that decides whether the drainage *network* branches: a basin shows only as many Strahler orders as its area divides into channel-sized pieces, so basin ÷ threshold sets the branching. At the old `20000` (41.7 km²) that ratio was about five on a 64 km map — no seed reached third order and first-order streams outnumbered second by nine to one, against Horton's three to five. `6000` is 12.5 km², by the humid-temperate regional curve `W = 2.5·A^0.4` a channel ~7 m across and ~0.6 m deep — a watercourse a cart must ford — and it leaves 88% of the land dry |
| `river_wander_exponent` | `float` | `1.0` | `≥ 0` | How water picks its way downhill, in hydrology and in erosion's valley carving alike: each hex drains to a lower neighbour drawn at random with weight (drop / largest drop)^k. 0 is any downhill neighbour alike, 1 in proportion to the drop, 8+ all but always the steepest. Pure steepest descent drew rivers on level ground as ranks of straight parallel lines. |
| `corner_floor_blend` | `float` | `0.25` | `0–1` | How high a corner stands between its three hexes, in hydrology and in erosion: the lowest of them plus this share of the way to their mean. 0 leaves a level lane a hex out from every river, down which tributaries run beside the trunk for miles; 0.25 tilts the floor toward the river and halves that. |
| `navigable_min_discharge` | `float` | `60000.0` | `> channel` | ...and to float a boat. Consumed by the haulage model: a navigable hex multiplies a city's supply reach |
| `cataract_min_drop_m` | `float` | `20.0` | `>= 0` | Metres per km a barge-sized river falls, measured along its course over a side and its neighbours, before that side is a cataract: no boat passes, cargo portages round it, and the fall drives mills (§ [3.5a](#35a-cataracts)). 20 m/km is a 2% gradient, strong rapids at a kilometre to the hex. 0 turns cataracts off |
| `rapids_min_drop_m` | `float` | `10.0` | `>= 0` | White water on a smaller river: a river side draining at least `rapids_min_catchment_km2` that falls this far per km (and is not already a cataract) is tagged `rapids` and drawn as white water on every map. Boats and the generator's crossings are unaffected; the campaign makes it a bridge-or-pontoon crossing. 0 turns rapids off. |
| `rapids_min_catchment_km2` | `float` | `200.0` | `>= 0` | The smallest river `rapids_min_drop_m` marks. |
| `evapotranspiration_base_mm` | `float` | `50.0` | `≥ 0` | Rain the ground and its plants take before anything runs off, even at freezing |
| `evapotranspiration_per_c_mm` | `float` | `30.0` | `≥ 0` | ...plus this much per degree of mean temperature. Why cold country sheds nearly all its rain and the taiga is full of rivers |
| `min_runoff_mm` | `float` | `25.0` | `≥ 0` | Floor, so even a desert drains its largest valleys |
| `wetland_min_runoff_mm` | `float` | `300.0` | `≥ 0` | Runoff above which flat riverside ground waterlogs. Tested on runoff, not rainfall: waterlogging is not about how much rain arrives but whether the ground can shed it |
| `river_flow_continuous` | `bool` | `False` | — | Record `hex.river_flow` on every draining land hex rather than only on river banks. A diagnostic for inspecting the drainage field; it does not add rivers to the map |
| `lake_chaining` | `bool` | `True` | — | Allow a lake to spill into a strictly lower lake, not just the sea. Chains of lakes stepping down to the coast are the only outlet on a landlocked map |
| `lake_min_hexes` | `int` | `20` | `≥ 1` | A closed hollow on land (ground water can only leave by filling it to its rim) at least this many hexes across, `lake_min_depth_m` deep at the rim, in a region shedding `lake_min_runoff_mm` a year, becomes a lake standing at its rim (`WaterBodiesStage`). Islands smaller than this inside a lake are tagged `hollow`. |
| `lake_min_depth_m` | `float` | `5.0` | `≥ 0` | See `lake_min_hexes`. |
| `lake_min_runoff_mm` | `float` | `50.0` | `≥ 0` | Regional runoff (at `mean_precip_mm`) below which no hollow fills: a dry basin stays a basin. |
| `hollow_wetland_min_depth_m` | `float` | `1.0` | `≥ 0` | A hollow too small or shallow for a lake but at least this deep is tagged `hollow`, and waterlogs to `WETLAND` under the closed-basin marsh rule (`endorheic_marsh_min_precip_mm`, level, below the treeline). |
| `endorheic_marsh_radius` | `int` | `1` | `≥ 0` | Where a basin genuinely has no outlet, water leaves by evaporation; this many hexes of its shore become wetland. `0` disables |
| `endorheic_marsh_min_precip_mm` | `float` | `300.0` | `≥ 0` | A closed basin drier than this is a salt pan, not a marsh, and gets no wetland shore |
| `endorheic_evaporation_scale` | `float` | `1.0` | `≥ 0` (validated) | Multiplier on the potential evapotranspiration a basin loses its water to. Whether a basin is closed is a **water balance**, not a shape: it overflows when the rivers reaching it plus the rain on its surface exceed what evaporates off that surface. Above `1` closes more basins; `0` makes every basin with any inflow overflow. It replaced a test the routing was making by accident — a basin came out closed when path-finding happened to fail on it, so a dry basin with an easy saddle drained while a wet one ringed by hills did not, backwards on both counts. On one map 5,234 of 7,089 lake hexes came out endorheic |
| `rain_shadow_strength` | `float` | `0.5` | `[0, 1]` (validated) | How much the rain shadow shapes the **rivers**, not only the biomes. `0` rains evenly on every hex; `1` takes the orographic pattern at its word, so a catchment behind mountains raises a smaller river and a lake there is likelier to evaporate away than to overflow. The map's total rainfall does not change with this — only where it falls. Below `1` because the pattern models only rain wrung out by lift, and taken literally it leaves every plain a desert |

#### Rivers from off the map

The grid is a region, not a world, so a river may well have gathered its water beyond the
border. An **inlet** is a border land hex whose own terrain already descends inland; it is
seeded with a catchment it did not earn on this map, which makes it arrive already wide
instead of starting as a trickle at the edge. Off-map inlet *erosion* by droplets was
tried twice, measured worse than not doing it, and reverted — a droplet is one raindrop
wherever it starts, so seeding them at a mouth digs a pit that inverts the inland fall and
disqualifies the very cell it was meant to serve. Discharge seeding does the job instead.

| Param | Type | Default | Range | Effect |
|---|---|---|---|---|
| `river_inflow_count` | `int` | `2` | `≥ 0` (validated) | How many rivers enter from beyond the border. `0` disables, and the map drains outward only |
| `river_inflow_volume` | `float` | `0.15` | `≥ 0` (validated) | Size of that off-map catchment, as a fraction of this map's own land area. Relative rather than absolute so the same value means the same thing on any grid size: `0.15` is a river carrying half again what the largest wholly-on-map river would |
| `river_inflow_min_separation` | `int` | `6` | `≥ 0` (validated) | Minimum spacing between inlets, in hexes. Without it the best few candidates are usually the same valley mouth, and the map gets one river drawn three times |
| `river_inflow_edges` | `tuple[str, ...]` | all four | edge names (validated) | Which edges water may arrive from. Narrowing it puts the off-map highlands on a chosen side: `["west"]` means every river from beyond the border enters in the west. Normalised like `continent_falloff_edges` — any case, any order, deduplicated |
| `river_inflow_length_bias` | `float` | `2.0` | `≥ 0` (validated) | Exponent on the length of the course an inlet would take, preferring water that crosses the map over water that ducks back off it. `0` ignores length entirely |
| `river_inflow_min_length` | `float` | `0.1` | `≥ 0` (validated) | Shortest course worth importing a river for, as a fraction of the longer map dimension. Weighting alone still leaves stubs, because the separation rule can leave nothing but stubs to choose from once the first inlet is placed — and a river that enters the map and leaves it four hexes later reads as a mistake rather than as geography. A map offering nothing longer gets fewer rivers than `river_inflow_count` asks for, which is the honest outcome; `0` disables the floor |

### 4.6 Climate — § [3.6](#36-climate)

The map is a **region, not a world**: 500 km at 1 hex = 1 km is about 4.5° of latitude,
some 3 °C. Altitude does far more over the same distance and rain shadow more again. So
the region has one named climate, and the variety within it comes from terrain.

Temperature is in **degrees Celsius** and rainfall in **millimetres a year**. Both used to
be `[0, 1]` axes; three defects were hiding in the moisture one alone, including a
`moisture_factor` that returned zero above `1.0` and so, read in millimetres, zeroed every
hex's food on the map.

| Param | Type | Default | Range | Effect |
|---|---|---|---|---|
| `regional_climate` | `str` | `'temperate'` | `boreal`, `temperate`, `mediterranean`, `arid`, `tropical` | Sets the region's mean temperature and rainfall, and the palette of biomes it can produce — so an arid region runs desert to steppe to alpine with altitude but never grows a jungle three valleys over |
| `mean_temperature_c` | `float \| None` | `None` → from climate | `-30..40` | Mean annual temperature at sea level. Blank takes it from `regional_climate` (boreal 1, temperate 10, mediterranean 16, arid 21, tropical 26). Pinning it while also naming a climate is rarely what you want |
| `mean_precip_mm` | `float \| None` | `None` → from climate | `(0, 12000]` | Mean annual rainfall over land. Blank takes it from `regional_climate` (boreal 450, temperate 800, mediterranean 550, arid 200, tropical 2000) |
| `wet_season_share` | `float \| None` | `None` → from climate | `[0.5, 1]` | The wetter half-year's share of the year's rain. Blank takes it from `regional_climate` (boreal 0.65, temperate 0.55, mediterranean 0.75, arid 0.8, tropical 0.8). Soil reads the dry half for drought and the wet half for leaching, so the same rain farms worse the more it bunches; 0.5 reads exactly as annual rain. See § [3.6](#36-climate) step 6 for the figures behind each climate |
| `crops_grow_in_wet_season` | `bool \| None` | `None` → from climate | — | Whether the crops grow in the wet half-year or the dry one; soil's dry arm reads that season. Blank takes it from `regional_climate` (true for boreal and tropical; false for temperate, mediterranean and arid). The Mediterranean grows through its summer drought; the monsoon sows into the rains; the taiga's winter is snow stored to spring. Set true for a Sahel-type desert |
| `lapse_rate_c_per_km` | `float` | `6.5` | `≥ 0` | How fast air cools with height. 6.5 is the standard environmental lapse rate — a real rate, applied to height above the waterline |
| `latitude_temp_range_c` | `float` | `0.0` | `≥ 0` | Degrees between the map's pole-ward and equator-ward edges. Negligible across a region; raise only for a continental map |
| `wind_direction` | `(float, float)` | `(1.0, 0.0)` | — | Prevailing wind vector driving orographic precipitation and moisture transport. Magnitude is normalised; only direction matters |
| `orographic_strength` | `float` | `2.0` | `> 0` | Wind-driven precipitation intensity. Higher = wetter windward slopes and drier rain shadows |
| `moisture_resupply_per_hex` | `float` | `0.08` | `[0, 1]` | Share of its moisture deficit the air makes back each km by evaporation. Without it a rain shadow runs from the first hill to the map edge: rainfall spanned 8× coast to interior and a temperate map read 60% shrubland |
| `base_precip_mm` | `float` | `0.0` | — | Flat rainfall bias in mm/year added to every land hex after the orographic pass. Shifts a whole map wetter or drier without changing the pattern |
| `moisture_bleed_passes` | `int` | `0` | `≥ 0` (validated) | Moisture carried inland along a river, so a valley is greener than the ground above it. `0` uses the flat river bonus only |
| `moisture_bleed_strength` | `float` | `0.3` | `[0, 1]` (validated) | Share of the difference moved per pass. Only used when `moisture_bleed_passes > 0` |

### 4.7 Biome Thresholds — § [3.7](#37-biomes)

Real units throughout: Celsius for temperature bands, millimetres a year for rainfall.

| Param | Type | Default | Range | Effect |
|---|---|---|---|---|
| `biome_treeline_temp_c` | `float` | `-2.0` | `≤ biome_cold_temp_c` | Mean annual temperature at which trees stop. **The treeline is a temperature, not a height** — the altitude it falls at follows from the region's warmth and the lapse rate: ~1850 m temperate, ~500 m boreal, above 4300 m tropical. Replaces `biome_alpine_elev`, a fixed fraction that gave every map the same share of alpine ground however low its hills. Keep it clear of every climate's own mean: the alpine test runs ahead of every temperature rule, so a treeline landing at sea level makes a whole region bare rock |
| `biome_snowline_temp_c` | `float` | `-8.0` | `< biome_treeline_temp_c` | Mean annual temperature at which continuous plant cover stops — the second of the two lines that divide cold country. Between snowline and treeline is tundra: treeless but vegetated, and most of what stands above a subarctic treeline. Only above this is ground barren, and that is ALPINE. Deliberately colder than 1500 m of relief can reach, so a default map has no permanent snow — a temperate range that high has no glaciers either. Raise `max_elevation_m` towards 2400 and bare peaks appear |
| `biome_cold_temp_c` | `float` | `5.0` | `< warm` | Below this the cold biomes take over — taiga gives way to broadleaf woodland around here |
| `biome_warm_temp_c` | `float` | `18.0` | `> cold` | Above this the warm biomes take over, where subtropical vegetation begins |
| `biome_dry_precip_mm` | `float` | `400.0` | `< wet` | Below about this you get steppe and desert |
| `biome_wet_precip_mm` | `float` | `1000.0` | `> dry` | Above about this, closed wet forest. Also gates DENSE_FOREST in Land Cover |
| `food_drowned_precip_mm` | `float` | `3000.0` | `> biome_wet_precip_mm` | Annual rainfall at which ground is leached, waterlogged and worth nothing for farming. The wet arm of the agricultural curve falls to zero here |

### 4.8 Soil, food and habitability — § [3.7a](#37a-soil), [3.9](#39-habitability)

Food value of one hex, by land cover band. `TUNDRA`, `DESERT`, `ALPINE` and `BARE_ROCK`
are always `0`.

| Param | Type | Default | Range | Effect |
|---|---|---|---|---|
| `food_prime_value` | `float` | `1.4` | ≥ 0 | `PRIME` — alluvium: the floodplain of a river too big to wade |
| `food_alluvium_bonus` | `float` | `0.5` | ≥ 0 | What deep alluvium adds, as a multiple of the soil's base value: `potential_food` returns `soil_value * (1 + food_alluvium_bonus * alluvium)`. **Multiplicative on purpose.** Silt is a soil, not a climate — it renews what cropping strips, which is why the great river valleys carry the people they do, but it does not water a desert or hold a crop on bare rock. A flat bonus would have made a silted dune farmland, and the deltas are exactly where the alluvium runs deepest. `soil_value` is already zero for `UNUSABLE`, and no multiplier lifts a zero |
| `food_arable_value` | `float` | `1.0` | ≥ 0 | `ARABLE` — ordinary farmland |
| `food_marginal_value` | `float` | `0.55` | ≥ 0 | `MARGINAL` — ploughable and poor: leached, waterlogged, or podzol |
| `food_grazing_value` | `float` | `0.35` | ≥ 0 | `GRAZING` — too steep to plough or too dry to crop; run stock on it |
| `food_wetland_value` | `float` | `0.15` | ≥ 0 | `BOG`, `MARSH` — valued on cover, not soil, because a fen is not ploughland. Deliberately below water: neither good fishing nor good ploughing |
| `food_water_value` | `float` | `0.4` | ≥ 0 | `OPEN_WATER` — fishing, and valued on cover for the same reason. Non-zero so a coastal site is not penalised for having sea in its catchment |
| `soil_dry_farming_min_precip_mm` | `float` | `250.0` | ≥ 0 | The dry-farming limit: annual rainfall below which no crop is grown without irrigation, whatever the ground is like. **The only new threshold the soil rules need** — everything else reuses `terrain_rolling_gradient_m`, `terrain_steep_gradient_m`, `terrain_escarpment_gradient_m`, `biome_dry_precip_mm`, `biome_wet_precip_mm`, `food_drowned_precip_mm`, `biome_cold_temp_c` and `ford_max_catchment_km2`, each of which already means the right thing. At `250` an arid map is 85% unusable with its life on the rivers; at `400` it is 87% and mediterranean loses a fifth of its grazing |
| `soil_water_carryover` | `float` | `0.3` | `[0, 0.5]` | Share of the wet season's surplus over the dry that the ground banks and gives back in the drought, before soil reads each season against half the annual bands. 0 reads each season bare — a mediterranean map then has no arable at all, since a crop would see only the summer's rain; 0.5 evens every year out. On the 96×96 test map 0.2 / 0.3 / 0.4 put mediterranean farmland at 27% / 40% / 52% against temperate's 51–54% |
| `yield_arable` | `float` | `1.0` | ≥ 0 | What cleared ground under the plough yields, as a fraction of its soil's potential |
| `yield_pasture` | `float` | `0.55` | ≥ 0 | What grazed ground yields |
| `yield_wood` | `float` | `0.30` | ≥ 0 | What ground still under trees yields. The gap between this and `yield_arable` is what gives clearing economic weight — a settlement grows by assarting its hinterland, not merely by sitting in it |
| `yield_multiplier` | `float` | `1.7` | > 0 | Farming technology, as one multiplier on `yield_arable` and `yield_pasture`. 1.7 is England c. 1800, 1.0 England c. 1300; wood and fishery are left alone, and `potential_food` (which siting and land use read) is unchanged, so the market network stays put while more people live on it. `worldgen.yaml` carries a table of settings measured against historical densities from Roman Britain to 1830s Belgium |
| `clearing_margin` | `float` | `0.35` | ≥ 0 | Where clearing stops, as a fraction of the **best rent the settlement can reach**. Relative rather than absolute, and that is the substance: a market with a floodplain has a high bar and leaves its hillsides to sheep, while a market on uniformly thin ground has a low bar and ploughs the scrub — the worse the land, the more pressure to use bad land. The extensive margin set against the best alternative available, which is what rent theory actually says |
| `pasture_margin` | `float` | `0.15` | `[0, 1]` | Ploughable ground within a catchment whose rent falls short of `clearing_margin` but reaches this fraction of the best is cleared to **pasture**; below it, the parish keeps it as **wood**. At or above `clearing_margin` it does nothing. With 0.35 / 0.15 a temperate map is about 30% arable, 28% pasture, 20% waste and 23% wood, most of that wood beyond every market's reach — England c. 1800 was about 30 / 45 / 20 / 5. |
| `settled_clear_radius_city` | `int` | `1` | `≥ 0` | Hexes round a city cleared of forest and scrub for gardens, orchards and paddocks (`SettledGroundStage`). Every town and village clears its own hex; a lumber camp keeps its trees. |

Fertile and marginal hexes are additionally scaled by a rainfall curve peaking across
`[biome_dry_precip_mm, biome_wet_precip_mm]` and falling to `0` at both ends — too dry is
desert, too wet is `food_drowned_precip_mm`. Water and wetland ignore it.

| Param | Type | Default | Range | Effect |
|---|---|---|---|---|
| `habitability_agri_weight` | `float` | `0.40` | ≥ 0 | Weight on the catchment mean |
| `habitability_river_bonus` | `float` | `0.25` | ≥ 0 | Flat, if a river runs along one of the hex's sides — either bank |
| `habitability_harbour_bonus` | `float` | `0.60` | Site bonus for access to water that will float a barge — sea, lake, or a river above `navigable_min_discharge`. Distinct from `habitability_coast_bonus`, which is amenity: a beach to land a boat on. This one is about **bulk**, and it is the largest site bonus there is because it is the largest thing about a site — a town on navigable water can be provisioned from fifteen times the distance, which is the whole reason cities exist in this model. It has to be this big to be visible: a harbour site loses about a fifth of its day-range catchment to sea, which scores `food_water_value` (0.4) against arable's 1.0, so on agriculture alone it is worth ~22% less than an inland site and inland sites outnumber it nine to one. Before this term **not one of 74 markets stood on navigable water**, and no city could be maritime |
| `habitability_coast_bonus` | `float` | `0.25` | ≥ 0 | Flat, if the hex or a neighbour is `COAST` |
| `habitability_hill_bonus` | `float` | `0.15` | ≥ 0 | For a site overlooking the ground beside it, scaled by `habitability_hill_relief_m` |
| `habitability_hill_relief_m` | `float` | `75.0` | `> 0` (validated) | Metres of drop a site must command to be paid that bonus in full; below it the bonus scales down. It used to be paid flat to any ROLLING hex with a FLAT neighbour, which asked two band questions and got the wrong answer to both — a knoll and a bluff were worth the same, and a level floodplain beside a bluff collected the bonus for standing *under* the drop that commands it |
| `habitability_confluence_bonus` | `float` | `0.10` | ≥ 0 | Flat, on a hex at a `confluence` corner (this hex only) |
| `habitability_mill_bonus` | `float` | `0.15` | `>= 0` | Site bonus for standing on or beside a cataract: water power for a mill |

Bonuses are binary within each term: a hex beside one river side scores the same as one
with a river along three, and coastal adjacency is radius 1 only.

### 4.9 Settlements — the classic model — § [3.10](#310-city--town-placement), [3.13](#313-village-placement)

Used by `generate --model classic`. Counts and spacing are **inputs**: the map is told how
many cities to have, so it produces six whether it is one fertile plain or landlocked
desert. See § 4.10 for the model that derives them instead.

| Param | Type | Default | Range | Effect |
|---|---|---|---|---|
| `target_city_count` | `int` | `6` | ≥ 0 | Maximum cities placed. The actual count may be lower if fewer candidates pass separation |
| `target_town_count` | `int` | `24` | ≥ 0 | Maximum towns placed |
| `city_min_separation` | `int` | `20` | ≥ 1 | Minimum hex distance between cities |
| `town_min_separation` | `int` | `8` | ≥ 1 | Minimum hex distance between towns |
| `settlement_min_reachable` | `int` | `100` | `≥ 1` (validated) | Minimum hexes reachable below the slope cap. Filters out unreachable peaks and tiny islands. Used by **both** models |

### 4.10 Haulage and markets — the organic model

Used by `generate --model organic`. Before rail and the motor lorry, the binding
constraint on where people live and how large a place can grow is what it costs to move
bulk grain, and that one constraint generates the whole hierarchy — so there is no target
count anywhere here.

Each range is a **travel-cost budget rather than a distance**, so terrain shortens it: at
1 hex = 1 km they are calibrated to give the historical figure on flat ground and less
across hills. The ordering `rural_field_radius < market_day_radius < haulage_range_land`
is the model's core claim and is enforced in `__post_init__`.

| Param | Type | Default | Range | Effect |
|---|---|---|---|---|
| `rural_field_radius` | `float` | `2.5` | `> 0` | The daily walk out to the fields; sets cultivated extent. Chisholm: cropping intensity falls off past ~1 km, and land past 3–4 km is grazing or waste |
| `market_day_radius` | `float` | `10.0` | `> rural_field_radius` | Out to market, business done, and home inside a day. Bracton held markets should stand 6⅔ miles apart, being a third of a twenty-mile day out and a third back; English market towns do cluster at 10–15 km |
| `haulage_range_land` | `float` | `40.0` | `> market_day_radius` | Travel cost at which bulk food is worth nothing overland — the team has eaten the cargo. The softest figure here; what is well attested is the ratio below |
| `haulage_transship_cost` | `float` | `4.0` | What it costs to get a cargo onto the water and off again, charged once at each land-water transition, in the same units as `haulage_range_land`. Without it the sea is a teleport. 4.0, half the old 8.0: at 8 the portage round a cataract cost so much of the 40-unit budget that the markets above a fall stopped supplying the city below it; at 4 portage trade rises about fivefold and the city tier holds (at 2 the largest port swallows its neighbours). |
| `haulage_river_transship_cost` | `float` | `1.0` | The transship charge where the water is a navigable river or a lake rather than the open sea: a barge ties up at a bank, where a sea-going ship wants a harbour. A cataract portage is two of these. At 1.0 portage trade runs 2-6 times what it does at the sea charge on seeds 42/7/3, and the city tier holds; much lower thins it |
| `haulage_range_water_mult` | `float` | `15.0` | `≥ 1` | How much further the same cargo goes by water. Diocletian's Price Edict prices land carriage at 28–56× sea and 6–11× river per tonne-kilometre (Duncan-Jones: sea 1, river 4.9, wagon 28, pack animal 56). **This is why large pre-industrial cities sit on navigable water and inland ones stay small**: nothing gates a city, water simply extends what can feed it |
| `marketable_surplus_fraction` | `float` | `0.40` | `(0, 1]` | Share of what a farming household grows that can leave for a market; it eats the rest. Sets the share of people in towns: 0.40 with `yield_multiplier` 1.7 is England c. 1800 (25% in market towns on the map, 27.5% over 5,000 in 1801); 0.32 is England c. 1300. Sizing markets off the *surplus* rather than the production is why the tier ratios come out right without target counts |
| `people_per_food` | `float` | `80.0` | `> 0` | People fed per unit of food, and the one scale factor for the whole population of the map, settlements and countryside alike. **It is the density control:** it moves every population in proportion and nothing else — not the market count, the clearing or the rural share. 80 puts a temperate 128x128 map at 58–59 people per km² at the c. 1800 defaults, against 59 in England and Wales's 1801 census, and the 96×96 test mainland at 54. It was 145 until siting moved onto the day-reach (#117), which left markets clearing more of the land, and 180 before `elevation_hypsometry_exponent` laid most land low, since flat ground farms better. Density is linear in it, so it is also the knob for a country thinner than the era table's rows |
| `travel_ascent_per_hex` | `float` | `125.0` | `> 0` | Naismith's rule: metres of ascent costing as much as one hex of level ground. Catchments are *walked*, not engineered, so they use this rather than `road_slope_cost` — that curve prices grading a road and saturates at ten times base, which over eroded terrain shrinks a catchment to a third of its proper reach |
| `travel_ford_cost` | `float` | `16.0` | `≥ 0` | Getting across away from a crossing, per multiple of the wadeable span, charged once per river side crossed. Deliberately has no fixed term, unlike `road_river_crossing_base`: that base is the capital of *building* a bridge, and somebody walking to market pays no capital |

Those set what a market can reach. The three below decide where markets are planted: a
site is scored on the surplus it can gather inside a day's return, the best site is taken,
the surplus it draws on is depleted, and the scan repeats until nothing clears the floor.

| Param | Type | Default | Range | Effect |
|---|---|---|---|---|
| `city_min_draw` | `float` | `40.0` | How much *other markets'* surplus must be able to reach a town before it is a city. The one density knob for the tier above the market, and the same shape as `market_viability_floor` below it: an absolute threshold on what can be gathered rather than a target count, so a rich coast grows several cities and a landlocked desert grows none. Not comparable to the floor — a market gathers a countryside inside a day's cart, a city gathers *markets* over `haulage_range_land` with water counting fifteen times. At `40.0`: 2 cities on a temperate coast (20,781 / 12,397 against a median town of 445), 1 on a mediterranean map, **0 on an arid one whether coastal or landlocked**. It survived the soil recalibration unchanged, which is worth recording rather than assuming — the surplus fraction rose by 1.6× while actual food fell, and the two shifts cancelled. Validated `> 0` |
| `city_draw_share` | `float` | `0.7` | Share of its marketable surplus a market ships to the cities, split between them by pull (`city_pull_sharpness`). Also caps how much a candidate counts towards promotion, nearest markets first, which is what stops the first city promoted from claiming half the map. Validated in (0, 1] |
| `city_pull_sharpness` | `float` | `1.25` | How hard the best-paying city pulls a market's grain from the others. A city's pull is its size times the share of the cargo that survives the haul; a market splits its shipment in proportion to pull to this power. Very high sends everything to the best-paying city; 0 splits it evenly. 1.25 puts the largest city at 1.8-2.3x the second on seeds 42 and 7 |
| `city_pull_rounds` | `int` | `4` | Rounds of that split, each pulling with the sizes the last produced: pull follows size and size follows pull, which is how a capital comes to dominate. 1 pulls with founding sizes, when every candidate is a market of a few hundred and distance decides everything |
| `manufactured_trade_share` | `float` | `0.1` | Share of each city's people's worth put into manufactured trade with the other cities, split by size and haul. The shipments pay `transship_share` at every quay and portage on the way, drawn half from each city. 0 turns city-to-city trade off |
| `manufactured_range_mult` | `float` | `3.0` | How many times `haulage_range_land` manufactures travel: worth more per ton than grain, so carried further |
| `river_trade_share` | `float` | `0.2` | Long-haul trade down the rivers: each town or city on a navigable reach and off the coast ships this share of its people's worth downstream to the coastal town it reaches most cheaply, paying `transship_share` at every quay and portage on the way, drawn half from each end. The trade the portage towns lived on — Aswan, Louisville, the fall-line towns. On eight test worlds 0.2 grows the towns at portages by about 8% (matched town for town) while other towns lose about 1%; 0.1 gives +4.5% and 0.3 +12.5%. 0 turns it off and leaves the world exactly as it was. Validated `>= 0` |
| `river_trade_range_mult` | `float` | `6.0` | How many times `haulage_range_land` the river trade reaches: a boat going down rides the current, and the cargo is a region's staple export — Ohio flatboats ran 2,000 km to New Orleans. Twice manufactures' reach. Validated `>= 0` |
| `transship_share` | `float` | `0.2` | Transshipment: every change in how a city's cargo travels on its way — cart to boat, boat to cart, barge to ship at a river mouth — leaves this share of the people it feeds with the settlement handling that quay (the nearest within `transship_radius`), off the city's gain. Conserved; only the places between the source market and the city are paid, since loading and unloading are already part of those two. 0.2 turns a quay town with next to no hinterland into a trade town of 3-5k on seeds 42/7/3. Handled cargo per settlement is written to `metadata["transshipment"]` as `[q, r, food]`. Validated in [0, 0.5] |
| `transship_radius` | `int` | `2` | How far from a quay, in hexes, its handling settlement may stand. Validated `>= 0` |
| `city_min_population` | `int` | `5000` | A town that grows past this becomes a city whatever it draws — the entrepôt, which handles a hinterland's trade rather than eating its food. 0 turns it off. Validated `>= 0` |
| `port_min_population` | `int` | `250` | A port is founded at an unattended quay (or waterfront of quays within `transship_radius`) whose trade share reaches this many people. 0 turns ports off. § [3.10d](#310d-resource-settlements--organic) |
| `port_city_min_separation` | `int` | `12` | A port whose trade alone reaches `city_min_population` is a city only if no other city stands within this many hexes, and … |
| `port_city_min_farmland` | `int` | `20` | … at least this many ploughed hexes lie within `market_day_radius` of it. Otherwise it is a town: a portage in a bog beside the city it serves is a landing, not a second city. |
| `ore_min_relief_m` | `float` | `200.0` | Relief a hex needs to carry ore. About the top 8% of land on a temperate map |
| `ore_deposits_per_1000_km2` | `float` | `5.0` | Expected deposits per 1000 km² of ground above `ore_min_relief_m`, drawn as a Poisson count. 0 turns mines off |
| `ore_min_separation` | `int` | `8` | Hexes between deposits |
| `ore_haul_range_mult` | `float` | `2.0` | How many times `haulage_range_land` smelted ore is carried to an outlet (a city or a port-role settlement); a deposit with none in reach goes unworked. About 1 suits iron and coal, 3 or more lead, tin and silver |
| `mine_workforce` | `float` | `400.0` | Mean people a seam employs |
| `mine_workforce_sigma` | `float` | `0.5` | Lognormal spread of a seam's workforce around the mean |
| `lumber_radius` | `int` | `4` | Radius over which a camp counts the woodland it works |
| `lumber_min_score` | `float` | `35.0` | Woodland within `lumber_radius`, discounted by the bulk haul to the nearest city, that a camp must reach. 0 turns camps off |
| `lumber_min_separation` | `int` | `10` | Hexes between camps |
| `timber_float_min_discharge` | `float` | `15000.0` | Discharge (km² × mm) a river needs to float logs: below `navigable_min_discharge`, since timber was driven down rivers far too small for a barge |
| `lumber_people_per_wood_hex` | `float` | `3.0` | People a camp employs per wooded hex it works |
| `resource_attach_radius` | `int` | `1` | A workforce joins a settlement this close instead of founding a village beside it |
| `resource_draw_share` | `float` | `0.25` | The most of its haulage-weighted population any settlement sends to feed a new workforce. Validated in (0, 1] |
| `resource_min_population` | `int` | `30` | A village the food within reach cannot bring to this size is not founded |
| `market_viability_floor` | `float` | `20.8` | `> 0` | The one density knob, replacing `target_city_count` and `target_town_count` both: stop planting once the best remaining site scores below this. A site scores the surplus its day-reach delivers — the walk its catchment will make, each hex at its `usable_fraction` — so the floor is in the units of the real gather (the ring disc it replaced ran about four times that, which is why it fell from 24.0 in #117). At 20.8 a temperate 128×128 map with `continent_falloff_edges: [south]` plants 115–125 markets across seeds 42/7/3/11/19, against 30 on an arid one — an absolute threshold on gathered surplus rather than a target, so density follows the land. **The window:** the acceptance tests pass from 19.9 to 21.3, bounded below by the wood left on the 96×96 test mainland (9.8% at 19.8 against a floor of 10%) and above by the single arid village on the thin-country chokepoint map. 20.8 is the lowest value at which the five-climate reference counts markets strictly in order of food; below it boreal counts more than tropical, inside the test's tie tolerance. Rural density bounded the bottom until `people_per_food` was recalibrated to 80. **Lowering it raises the rural population too and thins the wood**, and that is a real feedback rather than rounding: more markets mean more catchments, more catchments mean more ground cleared, and cleared ground feeds more people than the wood it replaced |
| `chokepoint_min_road_tier` | `str` | `secondary` | `primary` \| `secondary` \| `track` | Least road tier a crossing must carry before it is worth a settlement. A bridge on a farm track is a plank, not a town. **This is what actually sets the size of the village tier**: on a 128×128 temperate map `secondary` admits about twenty candidate features, `track` admits a hundred and twenty |
| `chokepoint_min_separation` | `int` | `2` | `>= 0` | Suppression disc, and how far a village must stand off an existing settlement. The economics would mostly do this anyway — there is no residual surplus close to a market — but a bridge on a town's own doorstep is the town's bridge whatever the arithmetic says |
| `chokepoint_min_draw` | `float` | `0.30` | `>= 0` | The smallest village worth founding, in food units; multiply by `people_per_food` to read it as people, so `0.30` is 24 — hamlet scale, which is what a bridgehead settlement was. It also has to be: this tier lives on *residual* surplus, and the residual thinned when the soil model raised the market count from 65 to 76 on the same map, so at the hundred-person figure this used to mean, a temperate map grows exactly one village. Applied to the real catchment draw rather than to the estimate planting ranks on, which is what makes that relation exact. What is gathered is *residual* surplus — what the markets could not haul — over `rural_field_radius`, so it is not comparable with `market_viability_floor` |
| `market_min_separation` | `int` | `5` | `≥ 1` | A suppression disc only, to stop two markets sharing a hexside. Real spacing comes from competition for surplus, which is what makes markets dense on rich ground and sparse on poor — a fixed separation cannot express that |

### 4.11 River crossings — fords and bridges

A river is not uniformly crossable. Most of its length is an obstacle; a few places are
not, and those places are why towns sit where they do. Crossings are settled **before**
anything is built, so a bridging point can be the reason a market grows there rather than
something noticed afterwards.

A **ford is terrain and is free** — shallow braided water anyone can wade, needing nobody's
permission. A **bridge is capital** and appears only where enough traffic will use it.

| Param | Type | Default | Range | Effect |
|---|---|---|---|---|
| `ford_max_catchment_km2` | `float` | `60.0` | `> 0` | Catchment area at or below which the water can be waded. A physical figure comparable between maps, unlike `river_flow`, which is normalised against the largest accumulation present and so is a rank rather than a quantity. A stream draining a few tens of km² is ankle deep and a step across; one draining thousands is not |
| `crossing_relief_m` | `float` | `60.0` | `> 0` | Local relief, in metres, that doubles how hard a reach is to get across. Fast water takes your feet from under you whatever its depth, and at a kilometre to the hex it is the approaches rather than the span that defeat a bridge — both scale with how steep the ground is. A floodplain has a few metres of this; a gorge has hundreds |
| `bridge_pressure_per_span` | `float` | `3.0` | `> 0` | Surplus needed within reach per multiple of the widest wadeable span before a bridge is worth building. A river twice that width needs twice the traffic. Nobody bridges to nowhere |
| `crossing_pressure_radius` | `int` | `6` | `≥ 1` | How far either bank is searched for that surplus |
| `crossing_min_separation` | `int` | `4` | `≥ 1` | Nobody builds two bridges within sight of each other |
| `ford_serves_bridge_fraction` | `float` | `0.5` | `≥ 0` | A ford within `crossing_min_separation` makes a bridge needless only over water at least this fraction of the bridge's catchment. 0 lets any ford stand in for any bridge |
| `rare_ford_max_span` | `float` | `1.6` | `≥ 1` | A river that floats a barge is forded at its slackest reaches up to this span (1.0 is the wading limit). 1.0 turns these fords off |
| `rare_ford_separation` | `int` | `15` | `≥ 1` | Hexes kept between two such fords, which is what keeps them rare |
| `crossing_use_cost` | `float` | `1.0` | `≥ 0` | Cost of using an existing ford or bridge, charged once per crossing |

### 4.12 Cultivation Radii — § [3.12](#312-cultivation-cities--towns), [3.15](#315-village-cultivation)

| Param | Type | Default | Range | Effect |
|---|---|---|---|---|
| `cultivation_city_radius` | `int` | `8` | ≥ 0 | Hex range marked cultivated around each city |
| `cultivation_town_radius` | `int` | `4` | ≥ 0 | Around each town |
| `cultivation_village_radius` | `int` | `2` | ≥ 0 | Around each village (runs after villages place) |

These radii do double duty: each is also the catchment Habitability scores that tier on
(§ [3.9](#39-habitability)).

### 4.13 World Scale — § [3.11](#311-interurban-roads)

| Param | Type | Default | Range | Effect |
|---|---|---|---|---|
| `hex_size_m` | `float` | `1000.0` | `> 0` (validated) | Metres per hex. With the default, `1 hex = 1 km` |

`road_elev_range_m` is retired. Elevation is metres throughout, so nothing needs
converting from a `[0, 1]` range: a grade is `(Δelevation_m / hex_size_m) × 100` directly.

### 4.14 Roads — Terrain Costs — § [3.11](#311-interurban-roads)

Node cost (cost to *enter* a hex) by `terrain_class`. See
[road_cost.py](../worldgen/stages/road_cost.py).

There is no steepness surcharge, and the settings that were one — `road_mountain_cost`,
`road_hill_cost`, and the `road_rolling_cost` / `road_steep_cost` / `road_escarpment_cost`
that briefly replaced them — are all retired. Grade is priced per edge from the metres of
rise between two hexes, so a per-hex band charged the same climb a second time, and
charged it wrongly wherever the band and the grade disagreed: a level valley floor was
taxed at escarpment rates because the bluff above it fell inside the averaging window.

| Param | Type | Default | Effect |
|---|---|---|---|
| `road_flat_cost` | `float` | `1.0` | Node cost for any land hex, coast included |
| `road_water_cost` | `float` | `0.05` | Node cost for open or inland water. Validated `≥ 0`. Low to allow short over-water hops; the heavy lifting is in the embark/disembark edge cost |

### 4.15 Roads — Traveller Simulation

| Param | Type | Default | Range | Effect |
|---|---|---|---|---|
| `road_travellers_per_pop` | `float` | `0.04` | `> 0` | Travellers emitted per head of population. Replaces the three per-tier counts, which made a market of 6,200 and one of 900 each send the same hundred people — population entered only on the *destination* side of the gravity term, so every origin wore the same road out of its gates. `0.04` keeps the total near what the tier counts gave (about 8,000 over 74 markets at 128×128), so it redistributes rather than changes the dose |
| `road_travellers_max` | `int` | `500` | `≥ 1` | Cap per settlement, so one large city cannot drown the map. Reached only above 12,500 people |
| `road_raw_freight_per_person` | `float` | `0.01` | `≥ 0` | Road journeys a raw-goods flow (provisioning into a city, ore from a mine) puts on its route per person it feeds. Bulk went short distances or by water, so this is low: it wears the spokes into each city without taking the primary tier from the roads between cities. Timber floats and wears no road |
| `road_goods_freight_per_person` | `float` | `0.2` | `≥ 0` | Road journeys a manufactured-goods flow between two cities puts on its route per person it feeds: the carrier and wagon trade the main roads carried. At 0.2 the land legs of city-to-city trade routes are all secondary or better and their primary share rises by 5-8 points; much higher spreads the fixed primary tier thinner and lowers it |
| `road_gravity_exponent` | `float` | `2.5` | `≥ 0` | Distance exponent in the gravity model: a destination's appeal is `pop / distance ** this`. `2.5` rather than the `1.5` a modern gravity model would use, because a laden cart is not a lorry — at `1.5` a traveller was nearly as likely to make for a town 40 km off as one 10 km away, so 72% of every possible pair of settlements ended up with a road of its own and the network came out a mat rather than a hierarchy |
| `road_pheromone_factor` | `float` | `0.1` | `≥ 0` | Cost reduction per unit traffic. Higher = stronger highway-reinforcement effect |

### 4.16 Roads — Water Transitions

Edge cost, charged once on the land↔water transition.

| Param | Type | Default | Effect |
|---|---|---|---|
| `road_embark_cost` | `float` | `8.0` | Land → water (validated `≥ 0`) |
| `road_disembark_cost` | `float` | `8.0` | Water → land (validated `≥ 0`) |

### 4.17 Roads — River Crossings

Edge cost charged once, on a step from one hex to the next across a river side
(§ [3.11](#311-interurban-roads)). `road_river_hex_cost`
and `road_ferry_max_hop` are retired: a road no longer stands in a river, so there is no
per-hex charge and no land a river can seal off.

| Param | Type | Default | Effect |
|---|---|---|---|
| `road_river_crossing_base` | `float` | `8.0` | Constant component (validated `≥ 0`), charged once per crossing of a river side — the capital of building a bridge |
| `road_river_crossing_flow` | `float` | `24.0` | Multiplied by the river's flow at the side crossed. Big rivers are dramatically more expensive to bridge |

### 4.18 Roads — Slope Penalty

Slope cost is a rational function of grade percent — zero below `free_pct`, saturating at
`cost × cap_mult` near `cap_pct`. See [road_cost.py](../worldgen/stages/road_cost.py).

| Param | Type | Default | Effect |
|---|---|---|---|
| `road_delta_elevation_per_hex` | `float` | `25.0` | Metres of elevation change costing as much as one hex of level going — **the switchback, priced**. At 1 hex = 1 km a road climbing 200 m is not a straight ramp but several kilometres of zigzag folded inside that hex, and this is the exchange rate that says so. Anchored on `travel_ascent_per_hex` (125, Naismith's rule for a walker) divided by about five, a laden cart being far more sensitive to gradient than a commander on foot. Symmetric in up and down, unlike the walker's: a road is cut-and-fill, and a steep descent needs braking and washes out. Validated `> 0` |
| `road_switchback_grade_pct` | `float` | `10.0` | A road edge at or above this grade tags both its hexes `"switchback"`. The zigzag is priced but cannot be drawn at this scale — a switchback is a hundred-metre feature and a hex is a kilometre — so the tag is how a reader, or a wargame counting movement, knows the segment is slow. Validated in `(0, road_slope_cap_pct]` |
| `road_slope_cap_pct` | `float` | `25.0` | Grade % at which the penalty saturates. Validated `> road_slope_free_pct` |

`road_slope_cap_pct` is now a **refusal**, not a saturation. The curve it replaced was free
below 3% and levelled off at ten times base above 25%, so a road met a 65% face, paid a flat
twenty for it, and went straight up: on a 4000 m map the steepest road grade was **64.8%**,
and with the refusal it is **24.4%**. The free band was no better — 3% is exactly
`terrain_rolling_gradient_m`, the FLAT boundary, so every flat edge cost nothing and every
flat route was a tie.

The same threshold is also the **`grade_is_under_cap`** check used by
`settlement_min_reachable`: if no road would cross a 25%+ grade, no settlement should be
placed where its only escape requires one.

### 4.19 Roads — Network Classification

| Param | Type | Default | Effect |
|---|---|---|---|
| `road_settlement_skirt_cost` | `float` | `4.0` | What a road pays to pass a settlement at one hex without entering it — an edge whose two ends both neighbour the same seat. The cost-model half of the rule `route_through_settlements` applies afterwards, and the half that can actually shift a route at one hex: a *discount* on the town cannot, because the direct route and the detour both pay for the same two ring hexes, so the detour's extra cost is exactly what the town costs. Drive that to zero and the detour ties; it never wins, and ties go to heap order. Modest at `4.0`, about four hexes of level going — enough to shift a road that was indifferent, not enough to drag one over a mountain to call at a village. Validated `≥ 0` |
| `road_settlement_detour_max_mult` | `float` | `4.0` | A road passing a settlement at one hex is bent through it instead — a road skirting a town at the width of a field is a motor-age idea. This caps what the detour may cost, as a multiple of the edge it replaces; validated `≥ 2.0`, since a detour is two legs where there was one and so costs double on even ground by construction. What it bounds is the ground *beyond* that: the town on the far bank of a river, or up an escarpment. It catches a dear crossing and a steep bank together, which a grade cap would not — the worst case measured cost 31× its bypass at a grade of 4%, having been hauled onto a river channel (when rivers still occupied hexes) rather than up anything |
| `road_min_traffic` | `int` | `3` | Minimum traffic for a hex to count as a road at all |
| `road_river_traffic_min` | `int` | `1` | Lower threshold for an edge along a riverbank — both hexes beside the same river, not across it (validated `≥ 0`). Lets towpaths and river roads become roads on light traffic |
| `road_primary_pct` | `float` | `0.10` | Top fraction of eligible hexes, by traffic, that become PRIMARY |
| `road_secondary_pct` | `float` | `0.30` | Next fraction, which become SECONDARY |
| `road_track_pct` | `float` | `0.60` | Currently unused by InterurbanRoadStage — TRACK is reserved for village connectors. Kept so the three percentages sum to 1.0 |
| `road_tier_gap_max_edges` | `int` | `12` | Longest lower-class stretch promoted where a road of one class stops and resumes across it (`fill_tier_gaps`). 0 turns it off |

### 4.20 Naming — § [3.16](#316-naming)

| Param | Type | Default | Effect |
|---|---|---|---|
| `naming_cultures` | `int` | `3` | Culture regions, each with an invented language. 0 turns naming off and keeps the placeholder names. Validated `≥ 0` |
| `naming_substrate` | `bool` | `true` | An older people named the rivers and the present ones kept the names. `false`: each river is named by whoever holds its mouth |
| `naming_packs` | `list[str]` | `[]` | Hand-written culture packs for the regions, in order; regions past the end get invented languages. Any of `arabic`, `dutch`, `english`, `french`, `german`, `italian`, `latin`, `norse`, `slavic`, `spanish`, `welsh`, or Tolkien's `sindarin`, `quenya`, `khuzdul`, `rohirric`, `hobbitish`, or `hive` (an insect people). Validated: known, no repeats, at most `naming_cultures` |
| `naming_substrate_pack` | `str` | `""` | A pack for the people who named the rivers in place of an invented language — `welsh` under `english` is England. Empty for an invented one |
| `naming_pack_dirs` | `list[str]` | `[]` | Folders of your own culture pack YAML files ([CULTURE_PACKS.md](CULTURE_PACKS.md)), read after the built-in packs in order; a pack whose key is already taken replaces the earlier one. Relative to the working directory. An unknown pack key is refused when the naming stage runs, with the available packs listed |
| `naming_region_climb_m` | `float` | `150.0` | Metres of climb costing as much as one hex of level going when culture regions spread |
| `naming_region_river_cost` | `float` | `8.0` | Added for crossing a great river, in hexes of level going |
| `naming_region_water_cost` | `float` | `2.0` | Added per hex of open or inland water crossed |
| `naming_great_river_km2` | `float` | `1000.0` | Catchment above which a river is great: a frontier between peoples, and named in two syllables. About the top 3% of rivers on a 128x128 map |
| `naming_river_min_catchment_km2` | `float` | `200.0` | Rivers draining at least this much are named. About a quarter of rivers on a 128x128 map |
| `naming_min_edit_distance` | `int` | `2` | Single-letter edits any two names must be apart; capped at a quarter of the shorter name. Validated `≥ 1` |
| `naming_max_letters` | `int` | `12` | Longest word in a settlement's name before another is drawn; the whole name may run to twice this, so phrase names like Villeneuve-sur-Lot pass. Validated `≥ 4` |
| `naming_hill_relief_m` | `float` | `100.0` | Relief at which a site can be named for its hill |
| `naming_high_elevation_m` | `float` | `800.0` | Elevation above which a site can be called high |
| `naming_direction_radius` | `int` | `12` | How near a bigger place must be for a settlement to be named for the side of it it lies on |
| `naming_founder_weight` | `float` | `0.4` | Weight of a founder's name as a qualifier, against about 1–2 for a site's strongest feature |
| `naming_river_weight` | `float` | `3.0` | Weight of a nearby named river as a qualifier (Avonmouth, Stratford-on-Avon) |

---

## 5. In-Code Constants

Magic numbers and weights that live outside `WorldConfig` but materially
shape map output. Change these by editing the source file.

| Name | Value | Location | Effect |
|---|---|---|---|
| `_MAX_STEPS` | `64` | [erosion.py:18](../worldgen/stages/erosion.py#L18) | Max steps per erosion particle. Larger = longer-running particles, deeper channels |
| `_EVAPORATION` | `0.99` | [erosion.py:19](../worldgen/stages/erosion.py#L19) | Per-step water evaporation. Lower = particles die faster, less erosion downstream |
| Erosion delta fan weights | `0.6 / 0.3 / 0.1` | [erosion.py](../worldgen/stages/erosion.py) | Radial falloff over three rings when a droplet unloads at the sea |
| Hydrology epsilon (BFS) | `1e-6` | [hydrology.py:63](../worldgen/stages/hydrology.py#L63) | Per-step plateau tilt magnitude |
| Hydrology epsilon (coord) | `1e-4 * eps` | [hydrology.py:66](../worldgen/stages/hydrology.py#L66) | Coordinate-based tiebreak (≈`1e-10`) |
| Elevation Dijkstra penalty | `× 1000` | [hydrology_rivers.py:283](../worldgen/stages/hydrology_rivers.py#L283) | Cost multiplier for uphill movement during stalled-river extension |
| Erosion Gaussian sigma | `0.5` | [erosion.py:145](../worldgen/stages/erosion.py#L145) | Final smoothing pass after erosion |
| Temperature Gaussian sigma | `1.0` | [climate.py:36](../worldgen/stages/climate.py#L36) | Smoothing pass on temperature field |
| Flat river moisture bonus | `+0.15` | [climate.py:103](../worldgen/stages/climate.py#L103) | Used when `moisture_bleed_passes == 0` |
| Coastal moisture bonus | `+0.10` | [climate.py:107](../worldgen/stages/climate.py#L107) | Always applied to land hexes adjacent to OCEAN/LAKE |
| Moisture Gaussian sigma | `2.0` | [climate.py](../worldgen/stages/climate.py) | Smear on the rainfall field. Weather systems are wide; rain falls either side of the ridge that lifted it |
| LandCover dense-forest threshold | `wet_precip_mm * 1.5` | [land_cover.py](../worldgen/stages/land_cover.py) | Splits TEMPERATE_FOREST into DENSE_FOREST vs WOODLAND |
| Habitability land-cover bands | sets | [habitability.py:21–23](../worldgen/stages/habitability.py#L21) | Which covers count as fertile / marginal / wetland. The *values* are config (§4) |
| City population range | `[10_000, 50_000]` | [city_town.py:69](../worldgen/stages/city_town.py#L69) | Uniform random per city |
| Town population range | `[1_000, 10_000]` | [city_town.py:113](../worldgen/stages/city_town.py#L113) | |
| Town placement role: AGRICULTURAL fertile-neighbour count | `>= 3` | [city_town.py:26](../worldgen/stages/city_town.py#L26) | GRASSLAND or TEMPERATE_FOREST neighbours required |
| Prominent-site tag radius | `3` hexes | [city_town.py:137](../worldgen/stages/city_town.py#L137) | Local-max `habitability_town` neighbourhood for `"prominent_site"` tag |
| Village population range | `[100, 1_000]` | [village_placement.py:90](../worldgen/stages/village_placement.py#L90) | |
| Village minimum separation | `3` hexes | [village_placement.py:89](../worldgen/stages/village_placement.py#L89) | Hardcoded — not a `WorldConfig` parameter |
| Village frontier weight bonus | `× 2.0` | [village_placement.py:67](../worldgen/stages/village_placement.py#L67) | |
| Village road-adjacent bonus | `× 1.5` | [village_placement.py:69](../worldgen/stages/village_placement.py#L69) | |
| Road-adjacent habitability boost | `+0.2` (cap 1.0) | [interurban_roads.py:147](../worldgen/stages/interurban_roads.py#L147) | Applied to `habitability_village` only, after road tiers are decided; feeds VillagePlacement |
| Cultivation `RESISTANT` set | `{BOG, MARSH, BARE_ROCK, ALPINE, TUNDRA, DESERT, OPEN_WATER}` | [cultivation.py:6–16](../worldgen/stages/cultivation.py#L6) | Land covers immune to cultivation, used by both Cultivation and VillagePlacement |
| WorldState JSON schema version | `"2.0"` | [world_state.py](../worldgen/core/world_state.py) | Written by `to_dict`. `from_dict` accepts `2.0` only: 2.0 moved rivers onto hexsides, which changes what every reader means by "on a river", so an older file is rejected with a message to regenerate it from its seed rather than migrated |

---

## 6. Outputs

`worldgen generate` writes everything to the output directory (default
`./output/`).

| File | What it is |
|---|---|
| `config.json` | The `WorldConfig` used for this run. Reload with `--config config.json` to repro. |
| `world.json` | Full `WorldState` dump (lossless round-trip via `WorldState.to_dict / from_dict`) |
| `elevation.png` | Greyscale heightmap (post-erosion) |
| `terrain_class.png` | Categorical: ocean/lake/coast/flat/hill/mountain |
| `river_flow.png` | Normalised flow accumulation (blue intensity = flow) |
| `temperature.png` | Greyscale temperature field |
| `moisture.png` | Greyscale moisture field |
| `biome.png` | Categorical biome map |
| `habitability_city.svg` | Settlement suitability at the city catchment (radius 8) |
| `habitability_town.svg` | Settlement suitability at the town catchment (radius 4) |
| `habitability_village.svg` | Settlement suitability at the village catchment (radius 2), post-road boost |
| `settlements.png` | City / town / village markers |
| `roads.png` | PRIMARY / SECONDARY / TRACK lines |
| `land_cover.png` | Categorical land-cover map |
| `cultivation.png` | Cultivated-vs-wild overlay |

Re-render any attribute later without re-running the pipeline:

```bash
worldgen render --input output/world.json --attribute biome --output biome.png
```

For SVG output (atlas / topographic / wargame styles, layer toggles, custom
hex sizes), see the **SVG export** section of [README.md](../README.md).

---

## 7. Glossary

- **Axial coordinates** — A 2-axis hex coordinate system `(q, r)` covering
  the same set of hexes as 3-axis cube coords; the third axis
  `s = -q - r` is implicit. Used throughout the codebase.
- **Corner, side** — The points and edges between hexes, where rivers run. A corner is
  `(q, r, k)` with `k` in {0, 1} and touches three hexes; a side is `(q, r, s)` with `s` in
  {0, 1, 2} and lies between two. Each hex owns two corners and three sides, so every one
  has a single name (`core/hex_grid.py`).
- **Span** — How hard a river side is to get across, in multiples of the widest wadeable
  stream: width from catchment, inflated by how fast the water falls
  (`riverside.side_span`). A span of 1 or less can be waded.
- **fBm (fractional Brownian motion)** — Sum of multiple noise octaves
  with decreasing amplitude and increasing frequency. Produces
  multi-scale terrain in one pass.
- **Domain warp** — Sampling a noise field at coordinates that are
  themselves perturbed by another noise field. Breaks up grid-aligned
  artefacts and produces curvier coastlines.
- **Lapse rate** — Rate at which temperature decreases with altitude.
- **Orographic precipitation** — Rain caused by air being lifted as it
  flows over higher terrain. Creates the wet-windward / dry-lee pattern.
- **Priority-Flood** — A heap-based algorithm (Barnes et al., 2014) for
  raising closed depressions in a heightmap up to the elevation of their
  lowest outlet, ensuring every land cell can drain to the boundary.
- **Flow accumulation** — The number of upstream cells whose drainage
  passes through each cell. The "river-iness" of a hex.
- **Whittaker diagram** — Classic 2-axis biome chart (temperature vs
  precipitation) used to assign biomes from climate inputs.
- **Gravity model** — Discrete-choice probability proportional to
  `population[d] / distance[d]^k`; used here to pick traveller
  destinations.
- **Pheromone trail** — Self-reinforcing cost reduction along already-
  used paths, modelled on ant-colony optimisation. Concentrates
  random travellers onto a small number of recognisable highways.
- **Efraimidis–Spirakis key sampling** — Weighted sampling without
  replacement: draw `u ~ Uniform(0,1)` per item, compute `u^(1/weight)`,
  and sort descending. Avoids repeated cumulative-distribution builds.
- **Cultivation frontier** — Hexes that are cultivated but border
  uncultivated land. Used as the natural location for new villages.
