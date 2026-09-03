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
- **Default grid:** 128 × 128 (≈16,000 km², the size of a small kingdom).
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

- **`classic`** (default) ranks hexes on habitability and places a *configured number* of
  cities and towns at a fixed minimum separation, then sprinkles villages. Population is
  drawn at random from a per-tier band and nothing about the site enters the number.
- **`organic`** derives the hierarchy from pre-industrial haulage economics. Markets go
  where the most surplus can reach them inside a day's return, and their number follows
  the land rather than a target. It runs no village stages at all: the countryside is a
  productive surface rather than a list of hamlets (§ [3.10b](#310b-market-centres--organic)),
  so a temperate map carries ~80 settlements where `classic` carries ~1,100. See
  § [3.10a](#310a-river-crossings--organic) and § [3.10b](#310b-market-centres--organic).

The difference is not cosmetic. On one landlocked desert map at 128×128, `classic` places
a city of 48,000 — and its five largest populations are identical, figure for figure, to
those it produces on a fertile temperate coast, because they come from `rng.integers` and
the land tells them nothing. `organic` caps the same map at 2,706 and rings its
settlements around the shore of the inland sea.

The diagram below shows the `classic` pipeline.

```mermaid
flowchart TD
    Start([seed + WorldConfig]) --> Elev

    subgraph Terrain["Terrain &amp; Hydrology"]
        Elev[ElevationStage<br/><i>noise → hex.elevation</i>]
        Eros[ErosionStage<br/><i>carves channels in hex.elevation</i>]
        Tcls[TerrainClassificationStage<br/><i>hex.terrain_class</i>]
        Wbod[WaterBodiesStage<br/><i>splits OCEAN vs LAKE; fixes COAST</i>]
        Hydr[HydrologyStage<br/><i>state.rivers, hex.river_flow, river tags</i>]
        Elev --> Eros --> Tcls --> Wbod --> Hydr
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

    Hydr --> Clim
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

### `WorldState` — [worldgen/core/world_state.py:25–34](../worldgen/core/world_state.py#L25)

| Field | Type | Notes |
|---|---|---|
| `seed` | `int` | The RNG seed for this run |
| `width`, `height` | `int` | Grid dimensions in hexes |
| `layout` | `str` | `"axial"` or `"offset"` — how `width`/`height` map onto hex coordinates |
| `hexes` | `dict[HexCoord, Hex]` | Every cell, keyed by axial `(q, r)` in either layout |
| `rivers` | `list[River]` | Source-to-confluence (or source-to-sea) paths |
| `settlements` | `list[Settlement]` | Cities, towns, and villages combined |
| `roads` | `list[Road]` | PRIMARY / SECONDARY / TRACK paths |
| `metadata` | `dict` | `{"seed": ..., "config": ...}` snapshot |

Convenience accessors: `all_land()`, `all_ocean()`, `all_lakes()`,
`all_water()` ([world_state.py:49–79](../worldgen/core/world_state.py#L49)).

### `Hex` — [worldgen/core/hex.py:66–80](../worldgen/core/hex.py#L66)

| Field | Type | Range | Written by |
|---|---|---|---|
| `coord` | `HexCoord` (`(q, r)`) | — | construction |
| `elevation` | `float` | `[0.0, 1.0]` after normalization | Elevation, Erosion, Hydrology (lake fill) |
| `moisture` | `float` | `[0.0, 1.0]` | Climate |
| `temperature` | `float` | `[0.0, 1.0]` (clamped) | Climate |
| `terrain_class` | `TerrainClass` | enum | Terrain Class, Water Bodies, Hydrology |
| `biome` | `Biome \| None` | enum | Biome |
| `land_cover` | `LandCover \| None` | enum | Land Cover |
| `river_flow` | `float` | `[0.0, 1.0]`, normalized to map max | Hydrology |
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
- **`SettlementRole`** — `AGRICULTURAL, PORT, MINING, FORTRESS, MARKET`
  ([hex.py:58–63](../worldgen/core/hex.py#L58)).
- **`RoadTier`** — `PRIMARY, SECONDARY, TRACK`
  ([world_state.py:7–10](../worldgen/core/world_state.py#L7)).

### Tags Vocabulary (`Hex.tags`)

| Tag | Meaning | Set by |
|---|---|---|
| `"river"` | Hex carries a river path | Hydrology |
| `"headwater"` | River hex with no upstream river neighbour | Hydrology `_tag_hexes` |
| `"confluence"` | River hex with ≥2 upstream river neighbours | Hydrology |
| `"river_mouth"` | River hex on map border or adjacent to ocean/lake | Hydrology |
| `"ford"` | First road crossing of a river hex | `tag_river_crossings` |
| `"bridge"` | Second road to cross the same river hex (upgrades a ford) | `tag_river_crossings` |
| `"pass"` | HILL hex that's a local-max `habitability_town` within 3-hex range, no settlement | City/Town |
| `"confluence_town"` | TOWN settled on a hex already tagged `"confluence"` | City/Town |

Roads may cross a river but never travel along one: the hexsides a river is drawn
along are excluded from road pathfinding outright (`make_road_edge_cost`), and
`road_river_hex_cost` prices out the meander and braid cases the edge rule cannot
see. Settlement hexes are exempt only far enough to be *reached*: the hexside opens
when the town's counterpart is dry land, never when it is another river hex, so a
town on the water cannot be used to carry on down the channel. Where a river mesh
seals a component off entirely, the network is joined by a `Ferry` (drawn as a pair
of anchorages) rather than a road in the channel; if the gap is wider than
`road_ferry_max_hop`, routing raises `RoutingError` instead of degrading quietly.

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
   across the map, centred on `[-0.5, +0.5]`. It goes on **last**, in metres. It used to
   run before the shaping, where a normalisation promptly stretched the result back out,
   so asking for half a range of tilt got you rather less than that.

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

**Purpose:** Distinguish OCEAN (map-edge-connected) from inland LAKE, and
fix COAST hexes that ended up adjacent only to a lake.

**Reads:** `hex.terrain_class` (OCEAN from previous stage).
**Writes:** `hex.terrain_class` (some OCEAN → LAKE; some COAST →
`HILL/FLAT/MOUNTAIN`).

**Algorithm** ([water_bodies.py:21–39](../worldgen/stages/water_bodies.py#L21)):

1. Collect every hex marked OCEAN. BFS over OCEAN-OCEAN adjacency to
   discover connected components ([water_bodies.py:42–52](../worldgen/stages/water_bodies.py#L42)).
2. For each component: if any hex sits on the map border, leave it as
   OCEAN. Otherwise, reclassify every hex in the component to LAKE
   ([water_bodies.py:33–36](../worldgen/stages/water_bodies.py#L33)).
3. **`_fix_coast_hexes`** ([water_bodies.py:60–101](../worldgen/stages/water_bodies.py#L60)):
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

[stages/hydrology.py](../worldgen/stages/hydrology.py)

The biggest stage in the pipeline (~780 lines). It builds the river
network from the eroded heightmap.

**Reads:** `hex.elevation`, `hex.terrain_class`.
**Writes:** `hex.river_flow`, `hex.tags` (river/headwater/confluence/
river_mouth), `state.rivers`. May also raise lake water-levels and
convert land hexes to LAKE/OCEAN if a basin needs to expand to its
spillway.

**Config:** `channel_min_discharge`, `navigable_min_discharge`,
`evapotranspiration_base_mm`, `evapotranspiration_per_c_mm`, `min_runoff_mm`,
`river_flow_continuous`, `lake_chaining`, `endorheic_marsh_radius`,
`endorheic_marsh_min_precip_mm`.

**Algorithm**

Nine steps, top to bottom in
[hydrology.py:11–124](../worldgen/stages/hydrology.py#L11):

1. **Priority-Flood** sink-fills closed depressions on land, using a
   min-heap seeded with ocean and border land hexes (Barnes et al. 2014,
   [hydrology.py:153–186](../worldgen/stages/hydrology.py#L153)). After
   this pass, every land hex has a non-decreasing path of `filled[...]`
   values to the sea.

2. **Epsilon tilt** adds tiny perturbations to break ties on plateaus
   ([hydrology.py:35–40](../worldgen/stages/hydrology.py#L35)):
   ```
   filled[c] += 1e-6 * dist_from_water[c]/max_dist
              + 1e-6 * 1e-4 * (q + r) / (w + h)
   ```
   The first term gives plateaus a gradient *away from* water (so they
   drain consistently); the second is a coordinate-based tiebreaker that
   makes the result reproducible regardless of dict iteration order.

3. **Flow direction** — for each land hex, point at the lowest neighbour
   on the filled surface ([hydrology.py:188–235](../worldgen/stages/hydrology.py#L188)).
   Two subtleties:
   - For ocean/lake neighbours, use **raw** elevation, not filled, so a
     priority-flood-raised lake never appears higher than the actual
     terrain around it.
   - If both `from` and `to` sit on the map border, terminate flow at
     `from` rather than letting the river creep along the edge.

4. **Flow accumulation** — Kahn's topological sort, then accumulate
   ([hydrology.py:237–265](../worldgen/stages/hydrology.py#L237)):
   ```
   acc[c] = 1 + sum(acc[upstream])
   ```
   Each hex contributes 1 unit; downstream hexes accumulate the sum of
   their upstream tributaries. **At 1 hex = 1 km, `acc` is catchment area in square
   kilometres** — a physical quantity, comparable between one map and another. It is
   written to `hex.catchment_km2` and read by the crossing and haulage models.

5. **River extraction** — a channel forms where enough water passes to keep one open:
   ```
   runoff_mm    = max(min_runoff_mm,
                      precip_mm - (evapotranspiration_base_mm
                                   + evapotranspiration_per_c_mm * max(0, temp_c)))
   min_catchment = channel_min_discharge / runoff_mm
   river_set     = {c for c, area in acc.items() if area >= min_catchment}
   ```
   **Discharge, not rank.** The old `river_flow_threshold` was documented as a flow
   minimum and implemented as "take the top 5% of land by accumulation", so every
   climate — desert and rainforest alike — got 5.6% of its land under channel. Because
   runoff is rainfall less evapotranspiration, and evapotranspiration rises with
   temperature, cold country now sheds nearly everything it receives: arid land drains
   1.2% with **no navigable river at all**, tropical 12.6%.

   `hex.river_flow` remains a normalised `[0, 1]` rank, retained for river *rendering*
   width. Anything making a decision about a river reads `catchment_km2` instead.

6. **Build River objects** — for each headwater (river hex with no
   upstream river hex), trace `flow_dir` to its mouth
   ([hydrology.py:296–394](../worldgen/stages/hydrology.py#L296)). If a
   trace stalls before reaching water, three fallbacks try in order:
   - **Stage 1:** elevation-guided Dijkstra avoiding already-traced hexes.
     Uphill cost = `1.0 + 1000 * Δelev`
     ([hydrology.py:435–436](../worldgen/stages/hydrology.py#L435)) so the
     path stays in valleys.
   - **Stage 2:** same Dijkstra without the avoid set.
   - **Stage 3:** plain BFS (always succeeds on a finite grid).

   Fallback hexes are inserted into `river_set` and `flow_dir` is updated
   so subsequent tagging stays consistent.

7. **Tagging** ([hydrology.py:267–294](../worldgen/stages/hydrology.py#L267)):
   - `headwater`: river hex with 0 upstream river neighbours.
   - `confluence`: river hex with ≥2 upstream river neighbours.
   - `river_mouth`: river hex on the map border, or with an ocean/lake
     neighbour.

8. **Lake drainage** ([hydrology.py:481–711](../worldgen/stages/hydrology.py#L481))
   guarantees every lake has a visible outflow river:
   - Find each lake's natural **spillway** (lowest land hex on its
     perimeter, by raw elevation).
   - Raise the lake's surface to that elevation; flood-fill any land
     below the new water-level, converting it to LAKE.
   - If the expanded lake reaches the map edge, promote the whole
     component to OCEAN (it has touched the sea).
   - Otherwise, run elevation-guided Dijkstra from the spillway toward
     the nearest border or ocean neighbour. Append the result as a new
     `River` and merge it with any existing river network on the way out.

9. **Confluence splitting** ([hydrology.py:714–758](../worldgen/stages/hydrology.py#L714)):
   the source-to-sea paths produced in step 6 overlap whenever
   tributaries merge. Rivers are sorted by descending `flow_volume` and
   each one claims its hexes; later (lower-flow) rivers are trimmed at
   the first already-claimed hex. The result: every `River` object in
   `state.rivers` is a single source-to-confluence (or source-to-sea)
   segment with no duplicate trunk drawing.

**Output**

- If `river_flow_continuous=False` (default): `hex.river_flow = acc[c] / max_acc`
  for every river hex, and `0.0` for everything else.
- If `True`: every draining land hex gets a normalised flow value (handy
  if you want to render the underlying drainage gradient).

---

### 3.6 Climate

[stages/climate.py](../worldgen/stages/climate.py)

**Purpose:** Compute temperature and moisture fields. Two independent
sub-passes, run sequentially.

**Reads:** `hex.elevation`, `hex.terrain_class`, `hex.tags` (`"river"`),
`hex.river_flow`.
**Writes:** `hex.temperature`, `hex.moisture`.

**Config:** `regional_climate`, `mean_temperature_c`, `latitude_temp_range_c`,
`lapse_rate_c_per_km`, `wind_direction`, `orographic_strength`,
`moisture_resupply_per_hex`, `mean_precip_mm`, `base_precip_mm`,
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

2. **River and coastal bonuses.** For every land hex: `+0.15` if any neighbour carries
   the `"river"` tag (only when `moisture_bleed_passes == 0`), and `+0.1` if any
   neighbour is OCEAN or LAKE. Cumulative, so a coastal river-adjacent hex gets `+0.25`.

3. **Gaussian smear** at `sigma=2.0`. Weather systems are wide, and rain falls either
   side of the ridge that lifted it rather than only on the hex that did the lifting.

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
   `moisture_bleed_strength × max(neighbour.river_flow)` from any river-tagged neighbour
   at or above its own elevation. This builds a wider moisture corridor along big rivers,
   especially in valleys, but never uphill. There is no ceiling on the result — moisture
   is millimetres now, and a valley that receives more rain than the ridge above it is
   simply a wetter valley.

**Ocean and lake hexes** keep `moisture = 1.0`, a sentinel rather than a rainfall figure;
nothing downstream reads rainfall on water.

---

### 3.7 Biomes

[stages/biomes.py](../worldgen/stages/biomes.py)

**Purpose:** Whittaker-style biome assignment based on temperature,
moisture, elevation, and water/river adjacency.

**Reads:** `hex.terrain_class`, `hex.elevation`, `hex.temperature`,
`hex.moisture`, `hex.tags`.
**Writes:** `hex.biome`.

**Config:** `regional_climate`, `biome_treeline_temp_c`, `biome_cold_temp_c`,
`biome_warm_temp_c`, `biome_dry_precip_mm`, `biome_wet_precip_mm`,
`wetland_min_runoff_mm`, `endorheic_marsh_min_precip_mm`.

**Algorithm** ([biomes.py](../worldgen/stages/biomes.py)):

```
treeline_m = max(0, (mean_temperature_c - biome_treeline_temp_c)
                    / lapse_rate_c_per_km * 1000)

if terrain_class in (OCEAN, LAKE):
    biome = OCEAN
elif elevation_m > treeline_m:
    biome = ALPINE
elif temperature_c < biome_cold_temp_c:                     # default 5 C
    biome = pick(TUNDRA, ...)  if precip_mm < biome_dry_precip_mm     # default 400 mm
            pick(BOREAL, ...)  otherwise
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
its hills.

**Wetland overrides**, applied afterwards:

```
# Riverside waterlogging
if terrain_class in (FLAT, COAST) and "river" in tags
   and runoff_mm(precip, temp) > wetland_min_runoff_mm
   and elevation_m <= treeline_m:
    biome = WETLAND

# The shore of a closed basin
if "endorheic_shore" in tags and terrain_class in (FLAT, COAST)
   and precip_mm >= endorheic_marsh_min_precip_mm
   and elevation_m <= treeline_m:
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
- Keep `biome_treeline_temp_c` clear of every climate's own mean temperature. The alpine
  test runs ahead of every temperature rule, so a treeline landing at sea level makes a
  whole region bare rock: at `1.0`, which is the boreal region's mean, a boreal map grew
  no taiga at all and supported five settlements on sixteen thousand hexes.

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

**Config:** `food_fertile_value`, `food_marginal_value`,
`food_wetland_value`, `food_water_value`, `habitability_agri_weight`,
`habitability_river_bonus`, `habitability_coast_bonus`,
`habitability_hill_bonus`, `habitability_confluence_bonus`,
`cultivation_city_radius`, `cultivation_town_radius`,
`cultivation_village_radius`, `biome_dry_precip_mm`, `biome_wet_precip_mm`,
`food_drowned_precip_mm`.

**Algorithm** ([habitability.py](../worldgen/stages/habitability.py)):

```
# Per-hex food value, by land cover band
OPEN, WOODLAND            → food_fertile_value  × moisture_factor
SCRUB, DENSE_FOREST       → food_marginal_value × moisture_factor
BOG, MARSH                → food_wetland_value
OPEN_WATER                → food_water_value
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
bonus  = habitability_river_bonus      if hex or neighbour has "river"
       + habitability_coast_bonus      if hex or neighbour is COAST
       + habitability_hill_bonus       if a rise with a level neighbour
       + habitability_confluence_bonus if "confluence" in hex.tags

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
**Writes:** `hex.settlement`, `state.settlements`, `hex.tags` (`"pass"`,
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

**Cities** ([city_town.py:62–82](../worldgen/stages/city_town.py#L62)):
sort all land hexes by `habitability_city` descending — the widest catchment,
because a capital is chosen for the hinterland it can draw on; greedily accept each
one whose distance from every prior city is `>= city_min_separation`.
Each city gets a uniform-random population in `[10_000, 50_000]` and a
role from `_assign_role` (below).

**Towns** ([city_town.py:84–128](../worldgen/stages/city_town.py#L84)):

1. Sort on `habitability_town` — a different surface from the city score,
   with its own peaks, because a market town lives off the fields in
   walking distance rather than off a province. This replaces the old
   blanket `× 0.5 within 30 hexes of a city` damp, which existed only to
   push towns off the capitals' sites; `town_min_separation` now does that.
2. Find local maxima of that score (hexes whose score beats all 6
   neighbours).
3. Greedy placement with `town_min_separation` (default 8). Population
   uniform random in `[1_000, 10_000]`. Towns on `"confluence"` hexes
   also get the `"confluence_town"` tag.

**Role assignment** ([city_town.py:8–29](../worldgen/stages/city_town.py#L8)):

```
PORT         if "river" tag, COAST terrain, or any neighbour matches either
MINING       elif any STEEP/ESCARPMENT neighbour has elevation > 0.70
FORTRESS     elif any STEEP/ESCARPMENT neighbour (but lower)
AGRICULTURAL elif >= 3 neighbours are GRASSLAND or TEMPERATE_FOREST
MARKET       otherwise
```

`_assign_role` is shared by all three placement stages — classic cities and towns,
villages, and organic markets — so the roles mean the same thing whichever model ran.

> **Known stale: the `0.70` cutoff.** It dates from when elevation ran `0` to `1`. In
> metres it is seventy centimetres, so in practice every settlement with a steep
> neighbour is classed `MINING` and `FORTRESS` is unreachable — a 96×96 temperate map
> yields 71 mining settlements and no fortresses. Left as-is deliberately: what a fortress
> or a mine *is* has not been defined, so there is no basis yet for choosing a real
> altitude. Both roles are currently decorative — nothing downstream reads
> `Settlement.role` except the renderer's glyph choice.

**Pass tagging:** after settlements are placed, every empty ROLLING hex that is the
local-max `habitability_town` within 3-hex range gets the `"pass"` tag. Used for rendering
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

**Reads:** `hex.catchment_km2`, `hex.elevation`, `hex.tags`.
**Writes:** `hex.tags` (`"ford"`, `"bridge"`).

**Config:** `ford_max_catchment_km2`, `crossing_relief_m`, `bridge_pressure_per_span`,
`crossing_pressure_radius`, `crossing_min_separation`.

**The distinction the stage rests on:** a **ford is terrain and is free** — shallow
braided water anyone can wade, needing nobody's permission. A **bridge is capital**, and
appears only where enough traffic will use it. Nobody bridges to nowhere.

```
span      = effective width, in multiples of the wadeable catchment,
            inflated by local relief / crossing_relief_m
ford      if span <= 1                      # you can wade it
bridge    if surplus within crossing_pressure_radius
             >= bridge_pressure_per_span * span
           and no other crossing within crossing_min_separation
```

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
crossings.
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
heap = [(-score(c), c) for c in sorted(settleable_land)]
while heap:
    _, c = heappop(heap)
    if c in suppressed:            continue    # 1. suppression
    s = score(c)                               # 2. recompute against current remaining
    if s < -heap[0][0] - EPS:                  # 3. stale → re-push
        heappush(heap, (-s, c)); continue
    if s < market_viability_floor: break       # 4. true max below floor → done
    plant(c)
    for d, n in kernel_of(c):
        remaining[n] *= (1 - share(d))         # partial depletion
    suppressed |= hex_range(c, market_min_separation)
```

Depletion only ever *reduces* other sites' scores, so the score function is monotone
non-increasing and a popped entry that is still fresh is provably the true maximum.

**That predicate order is load-bearing** — suppression, then recompute, then staleness,
then the floor. Testing the floor before the staleness check ends the loop on the first
suppressed hex popped.

**Partial depletion, not hard claiming.** A market takes a distance-decayed *share* of the
surplus it reaches, so rich country supports another market 8 km away while poor country
supports none for 30. Hard claiming would collapse spacing to one number and reinvent
`city_min_separation` with extra steps.

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
`river_hex_cost` (12.0) exceeded the 10.0 day budget outright, and `terrain_base_cost`'s
3×/10× bands made the median step 3.0, so markets reached 3 hexes instead of 10. Both are
excluded. Ascent uses **Naismith's rule** (`travel_ascent_per_hex`) rather than
`road_slope_cost`, because a catchment is walked, not engineered — the road curve prices
*grading* a slope and saturates at ten times base, which over eroded terrain shrinks a
catchment to a third of its proper reach.

Catchments are terrain-shaped, not round: measured disc-fill is 0.43 at the median, and
they visibly stop at ridges and stretch down valleys.

---

### 3.11 Interurban Roads

[stages/interurban_roads.py](../worldgen/stages/interurban_roads.py)

**Purpose:** Build the inter-city road network (PRIMARY and SECONDARY
tiers). Uses gravity-model traveller simulation over A*-pathed routes,
with self-reinforcing pheromone trails.

**Reads:** `hex.terrain_class`, `hex.elevation`, `hex.river_flow`,
`hex.coord`, `state.settlements` (CITY and TOWN tiers only).
**Writes:** `state.roads`, `hex.road_connections`, `hex.tags`
(`"ford"` / `"bridge"`), `hex.habitability_village` (+0.2 boost).

**Config:** `road_travellers_per_pop`, `road_travellers_max`,
`road_gravity_exponent`, `road_bank_discount`,
`road_bank_discount_min_flow`, `road_pheromone_factor`,
`road_escarpment_cost`, `road_steep_cost`, `road_rolling_cost`, `road_flat_cost`,
`road_water_cost`, `road_embark_cost`, `road_disembark_cost`,
`road_river_crossing_base`, `road_river_crossing_flow`,
`road_river_hex_cost`, `road_ferry_max_hop`,
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
    OCEAN | LAKE → road_water_cost          (default 0.05)
    ESCARPMENT   → road_escarpment_cost     (20.0)
    STEEP        → road_steep_cost          (10.0)
    ROLLING      → road_rolling_cost        (3.0)
    COAST | FLAT → road_flat_cost           (1.0)

river_hex_cost = road_river_hex_cost if on a river else 0

# A *fraction* of the hex's cost, not a flat subtraction
discount  = road_bank_discount * max(adjacent_flow, road_bank_discount_min_flow)
            if beside a river and not on one else 0
base     *= (1 - discount)

pheromone = road_pheromone_factor * traffic_so_far[hex]

node_cost = max(0, base + river_hex_cost - pheromone)
```

The bank discount makes routes prefer to follow river valleys (Roman "river roads")
**along the bank rather than down the channel**, so which side of a river a road — and
anything standing on it — is on stays readable.

**It is a fraction, and that matters.** Subtracting a fixed 0.5 is half the cost of level
ground but a twentieth of an escarpment, so the pull quietly evaporated exactly where a
valley route is worth most: on rough country, where the alternative is going over the top.
When the terrain bands were re-cut as gradients and the average land cost rose from 2.9 to
4.6, that dilution was enough to invert the preference outright — roads began *under*-using
river corridors relative to how much of the map they cover. As a fraction the pull holds
its meaning at any cost scale, and on flat ground at the default it is arithmetically what
it always was.

The matching `river_hex_cost` prices out threading a meander or a braid, where two river
hexes sit side by side without a drawn hexside between them for the channel exclusion to
catch. The pheromone term makes the *order* of traveller processing matter — once enough
travellers have used a path it becomes cheap and subsequent travellers reinforce it, which
is what concentrates random travellers onto a small number of recognisable highways.

**Edge cost** ([road_cost.py:62–92](../worldgen/stages/road_cost.py#L62)):
```
edge_cost = slope_edge_cost + water_edge_cost + river_crossing_edge_cost

# slope_edge_cost — elevation is already metres, so nothing needs converting
grade_pct = |Δelevation_m| * 100 / hex_size_m                # percent
if grade_pct <= road_slope_free_pct:                # default 3 %
    slope = 0
elif grade_pct >= road_slope_cap_pct:               # default 25 %
    slope = road_slope_cost * road_slope_cap_mult   # = 2 * 10 = 20
else:
    raw = road_slope_cost * (grade_pct - free) / (cap - grade_pct)
    slope = min(raw, road_slope_cost * road_slope_cap_mult)

# water_edge_cost (charged on land↔water transitions)
embark    = road_embark_cost     (8.0)   if to_water and not from_water
disembark = road_disembark_cost  (8.0)   if from_water and not to_water

# river_crossing_edge_cost (charged on land↔river transitions)
flow = max(from.river_flow, to.river_flow)
crossing = road_river_crossing_base + road_river_crossing_flow * flow
                                    (default 4 + 12*flow)
```

A perpendicular crossing of a 1-hex-wide river hits `river_crossing_edge_cost`
twice (entering, then leaving the river hex), so the base+flow values
represent **half** the total perpendicular-crossing cost. Travelling
*along* a river never triggers it, since both hexes are river hexes
([road_cost.py:78–83](../worldgen/stages/road_cost.py#L78)).

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
traveller bound for a town does not need a road of his own all the way there;
he needs to reach the road that already goes there. So the search
(`astar_to_any`) runs against every hex from which the destination is already
reachable along roads that exist, and stops at whichever it touches first. The
rest of the journey is that road. The first traveller finds nothing and paths
the whole way, becoming the road everyone after him joins.

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

After tiering, two passes tidy the network. `route_through_settlements` bends
any road skirting a settlement so it passes through instead (§ 4.19), and
`prune_orphan_roads` drops any component reaching neither a settlement nor a
ferry landing — `road_river_traffic_min` admits a riverbank edge on a single
traveller, so a stretch of towpath can qualify while joining nothing. The
connectivity guarantee then runs over **every** settlement, not just the
cities: it used to require two or more cities, so the organic model had
nothing watching it, and the map stayed connected only because stitching made
most routes concatenations of the same few legs.

#### Tier classification — [interurban_roads.py:93–114](../worldgen/stages/interurban_roads.py#L93)

After all travellers are processed, hexes are filtered:
```
eligible = edges where traffic >= road_min_traffic                   (default 3)
        OR "river" in hex.tags and traffic >= road_river_traffic_min (default 1)
```
Sort by traffic descending, then:
- top `road_primary_pct` (10 %) → PRIMARY
- next `road_secondary_pct` (30 %) → SECONDARY
- rest → no tier (TRACK is reserved for village connectors)

A canonical route's tier is the **highest** tier any hex on it earned
(`_path_min_tier`, [interurban_roads.py:191–196](../worldgen/stages/interurban_roads.py#L191)).
Routes whose hexes are all below the traffic threshold are dropped
entirely.

#### Connectivity guarantee — [interurban_roads.py:198–276](../worldgen/stages/interurban_roads.py#L198)

If the traffic-driven graph leaves any city in a separate component,
`_guarantee_city_connectivity` runs A* (using only the plain terrain
costs, *no* pheromone) from each isolated city to the largest connected
component, and inserts those paths as PRIMARY roads. Bounded to
`2 * len(cities)` iterations to prevent runaway cases.

#### Side effects

- **River-crossing tags**: `tag_river_crossings` walks each road; the
  first time it enters a river hex (from a non-river hex) it adds
  `"ford"`. A second visit upgrades that tag to `"bridge"`
  ([road_cost.py:95–116](../worldgen/stages/road_cost.py#L95)).
- **Habitability boost** (+0.2, capped at 1.0) applied to every land hex
  adjacent to a road
  ([interurban_roads.py:140–147](../worldgen/stages/interurban_roads.py#L140)).
  This feeds VillagePlacementStage so that road corridors attract
  villages.

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
`hex.terrain_class`, `hex.elevation`, `hex.river_flow`.
**Writes:** `state.roads` (new TRACK Roads), `hex.road_connections`,
`hex.tags` (ford/bridge).

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
| `width` | `int` | `128` | ≥ 1 | Map width in hexes (`1 hex = 1 km` by convention) |
| `height` | `int` | `128` | ≥ 1 | Map height in hexes |
| `grid_layout` | `str` | `"axial"` | `axial` \| `offset` | Grid shape — see below |

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
| `heightmap_invert` | `bool` | `False` | — | Coastline mode. Treat the darker side of the threshold as the land instead |
| `heightmap_coast_falloff` | `bool` | `False` | — | Coastline mode. Also apply the rectangular edge falloff, ringing the map with sea. Off by default, so the stencil is authoritative |

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

### 4.5 Hydrology — § [3.5](#35-hydrology)

A channel forms where enough water passes to keep one open: **discharge = catchment area
× runoff depth**. This replaced `river_flow_threshold`, which was documented as a flow
minimum and implemented as a rank — the top 5% of land by accumulation — so desert and
rainforest alike got 5.6% of their land under channel. Now arid country drains 1.2% with
no navigable river at all, and tropical 12.6%.

| Param | Type | Default | Range | Effect |
|---|---|---|---|---|
| `channel_min_discharge` | `float` | `20000.0` | `> 0` | Catchment km² × runoff mm needed to cut a channel |
| `navigable_min_discharge` | `float` | `150000.0` | `> channel` | ...and to float a boat. Consumed by the haulage model: a navigable hex multiplies a city's supply reach |
| `evapotranspiration_base_mm` | `float` | `50.0` | `≥ 0` | Rain the ground and its plants take before anything runs off, even at freezing |
| `evapotranspiration_per_c_mm` | `float` | `30.0` | `≥ 0` | ...plus this much per degree of mean temperature. Why cold country sheds nearly all its rain and the taiga is full of rivers |
| `min_runoff_mm` | `float` | `25.0` | `≥ 0` | Floor, so even a desert drains its largest valleys |
| `wetland_min_runoff_mm` | `float` | `300.0` | `≥ 0` | Runoff above which flat riverside ground waterlogs. Tested on runoff, not rainfall: waterlogging is not about how much rain arrives but whether the ground can shed it |
| `river_flow_continuous` | `bool` | `False` | — | Record `hex.river_flow` on every draining land hex rather than only on channel hexes. A diagnostic for inspecting the drainage field; it does not add rivers to the map |
| `lake_chaining` | `bool` | `True` | — | Allow a lake to spill into a strictly lower lake, not just the sea. Chains of lakes stepping down to the coast are the only outlet on a landlocked map |
| `endorheic_marsh_radius` | `int` | `1` | `≥ 0` | Where a basin genuinely has no outlet, water leaves by evaporation; this many hexes of its shore become wetland. `0` disables |
| `endorheic_marsh_min_precip_mm` | `float` | `300.0` | `≥ 0` | A closed basin drier than this is a salt pan, not a marsh, and gets no wetland shore |

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
| `biome_cold_temp_c` | `float` | `5.0` | `< warm` | Below this the cold biomes take over — taiga gives way to broadleaf woodland around here |
| `biome_warm_temp_c` | `float` | `18.0` | `> cold` | Above this the warm biomes take over, where subtropical vegetation begins |
| `biome_dry_precip_mm` | `float` | `400.0` | `< wet` | Below about this you get steppe and desert |
| `biome_wet_precip_mm` | `float` | `1000.0` | `> dry` | Above about this, closed wet forest. Also gates DENSE_FOREST in Land Cover |
| `food_drowned_precip_mm` | `float` | `3000.0` | `> biome_wet_precip_mm` | Annual rainfall at which ground is leached, waterlogged and worth nothing for farming. The wet arm of the agricultural curve falls to zero here |

### 4.8 Habitability — § [3.9](#39-habitability)

Food value of one hex, by land cover band. `TUNDRA`, `DESERT`, `ALPINE` and `BARE_ROCK`
are always `0`.

| Param | Type | Default | Range | Effect |
|---|---|---|---|---|
| `food_fertile_value` | `float` | `1.0` | ≥ 0 | `OPEN`, `WOODLAND` — prime arable |
| `food_marginal_value` | `float` | `0.4` | ≥ 0 | `SCRUB`, `DENSE_FOREST` — grazing and hard-to-clear forest |
| `food_wetland_value` | `float` | `0.15` | ≥ 0 | `BOG`, `MARSH` — deliberately below water; neither good fishing nor good ploughing |
| `food_water_value` | `float` | `0.4` | ≥ 0 | `OPEN_WATER` — fishing. Non-zero so a coastal site is not penalised for having sea in its catchment |

Fertile and marginal hexes are additionally scaled by a rainfall curve peaking across
`[biome_dry_precip_mm, biome_wet_precip_mm]` and falling to `0` at both ends — too dry is
desert, too wet is `food_drowned_precip_mm`. Water and wetland ignore it.

| Param | Type | Default | Range | Effect |
|---|---|---|---|---|
| `habitability_agri_weight` | `float` | `0.40` | ≥ 0 | Weight on the catchment mean |
| `habitability_river_bonus` | `float` | `0.25` | ≥ 0 | Flat, if the hex or a neighbour carries a river |
| `habitability_coast_bonus` | `float` | `0.25` | ≥ 0 | Flat, if the hex or a neighbour is `COAST` |
| `habitability_hill_bonus` | `float` | `0.15` | ≥ 0 | Flat, for a rise overlooking a plain |
| `habitability_confluence_bonus` | `float` | `0.10` | ≥ 0 | Flat, on a river junction (this hex only) |

Bonuses are binary within each term: a hex with one river neighbour scores the same as one
ringed by six, and adjacency is radius 1 only.

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
| `haulage_range_water_mult` | `float` | `15.0` | `≥ 1` | How much further the same cargo goes by water. Diocletian's Price Edict prices land carriage at roughly 55× sea and 11× river per tonne-kilometre. **This is why large pre-industrial cities sit on navigable water and inland ones stay small**: nothing gates a city, water simply extends what can feed it |
| `marketable_surplus_fraction` | `float` | `0.20` | `(0, 1]` | Share of what a farming household grows that can leave for a market; it eats the rest. Sizing markets off the *surplus* rather than the production is why the tier ratios come out right without target counts |
| `people_per_food` | `float` | `400.0` | `> 0` | People supported per unit of haulage-weighted food. Calibrated so market towns land in their historical 500–2500 band: across five seeds at 128×128 this gives medians of 1260–1700 and a largest of 4200–5450 |
| `travel_ascent_per_hex` | `float` | `125.0` | `> 0` | Naismith's rule: metres of ascent costing as much as one hex of level ground. Catchments are *walked*, not engineered, so they use this rather than `road_slope_cost` — that curve prices grading a road and saturates at ten times base, which over eroded terrain shrinks a catchment to a third of its proper reach |
| `travel_ford_cost` | `float` | `8.0` | `≥ 0` | Getting across away from a crossing, per multiple of the wadeable span, charged on each land–river edge. Deliberately has no fixed term, unlike `road_river_crossing_base`: that base is the capital of *building* a bridge, and somebody walking to market pays no capital |

Those set what a market can reach. The three below decide where markets are planted: a
site is scored on the surplus it can gather inside a day's return, the best site is taken,
the surplus it draws on is depleted, and the scan repeats until nothing clears the floor.

| Param | Type | Default | Range | Effect |
|---|---|---|---|---|
| `market_viability_floor` | `float` | `14.0` | `> 0` | The one density knob, replacing `target_city_count` and `target_town_count` both: stop planting once the best remaining site scores below this. Calibrated to ~70–85 markets at 128×128 (England had ~700 markets in ~130,000 km²; this map is about an eighth of that): a temperate map with `continent_falloff_edges: [south]` gives 74–81 across seeds 42/7/3/11/19 — one per ~205 km² of land, a 15 km lattice, each about 10 km from its nearest neighbour. An absolute threshold on gathered surplus rather than a target, so density follows the land — the same value yields 9 markets on an arid map and 74 on a temperate one |
| `market_min_separation` | `int` | `5` | `≥ 1` | A suppression disc only, to stop two markets sharing a hexside. Real spacing comes from competition for surplus, which is what makes markets dense on rich ground and sparse on poor — a fixed separation cannot express that |
| `market_kernel_decay` | `float` | `4.0` | `> 0` | `d₀` in the `1/(1 + d/d₀)` share a market takes from each hex it reaches |

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
| `crossing_use_cost` | `float` | `0.5` | `≥ 0` | Cost of using an existing ford or bridge |

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
[road_cost.py](../worldgen/stages/road_cost.py). Renamed with the gradient bands: what was
`road_mountain_cost` is now `road_steep_cost`, and `road_hill_cost` is `road_rolling_cost`.

| Param | Type | Default | Effect |
|---|---|---|---|
| `road_escarpment_cost` | `float` | `20.0` | Node cost for ESCARPMENT — a break of slope |
| `road_steep_cost` | `float` | `10.0` | Node cost for STEEP — pack animals, no wheels |
| `road_rolling_cost` | `float` | `3.0` | Node cost for ROLLING |
| `road_flat_cost` | `float` | `1.0` | Node cost for FLAT and COAST |
| `road_water_cost` | `float` | `0.05` | Node cost for OCEAN/LAKE. Validated `≥ 0`. Low to allow short over-water hops; the heavy lifting is in the embark/disembark edge cost |

### 4.15 Roads — Traveller Simulation

| Param | Type | Default | Range | Effect |
|---|---|---|---|---|
| `road_travellers_per_pop` | `float` | `0.04` | `> 0` | Travellers emitted per head of population. Replaces the three per-tier counts, which made a market of 6,200 and one of 900 each send the same hundred people — population entered only on the *destination* side of the gravity term, so every origin wore the same road out of its gates. `0.04` keeps the total near what the tier counts gave (about 8,000 over 74 markets at 128×128), so it redistributes rather than changes the dose |
| `road_travellers_max` | `int` | `500` | `≥ 1` | Cap per settlement, so one large city cannot drown the map. Reached only above 12,500 people |
| `road_gravity_exponent` | `float` | `2.5` | `≥ 0` | Distance exponent in the gravity model: a destination's appeal is `pop / distance ** this`. `2.5` rather than the `1.5` a modern gravity model would use, because a laden cart is not a lorry — at `1.5` a traveller was nearly as likely to make for a town 40 km off as one 10 km away, so 72% of every possible pair of settlements ended up with a road of its own and the network came out a mat rather than a hierarchy |
| `road_bank_discount` | `float` | `0.5` | `[0, 1]` typ. | Maximum node-cost reduction on a hex *beside* a river, as a fraction of the hex's base cost, scaled by the largest adjacent river's flow. Proportional rather than absolute so it is not diluted when the cost scale changes. River hexes themselves get nothing |
| `road_bank_discount_min_flow` | `float` | `0.2` | `[0, 1]` (validated) | Floor on `river_flow` used in the discount. Prevents tiny headwaters from losing their corridor pull |
| `road_pheromone_factor` | `float` | `0.1` | `≥ 0` | Cost reduction per unit traffic. Higher = stronger highway-reinforcement effect |

### 4.16 Roads — Water Transitions

Edge cost, charged once on the land↔water transition.

| Param | Type | Default | Effect |
|---|---|---|---|
| `road_embark_cost` | `float` | `8.0` | Land → water (validated `≥ 0`) |
| `road_disembark_cost` | `float` | `8.0` | Water → land (validated `≥ 0`) |

### 4.17 Roads — River Crossings

Edge cost charged on each land↔river transition. A perpendicular crossing of a 1-hex-wide
river hits this twice, entering and leaving the river hex.

| Param | Type | Default | Effect |
|---|---|---|---|
| `road_river_crossing_base` | `float` | `4.0` | Constant component (validated `≥ 0`) — the capital of building a bridge |
| `road_river_crossing_flow` | `float` | `12.0` | Multiplied by `max(from.river_flow, to.river_flow)`. Big rivers are dramatically more expensive to bridge |
| `road_river_hex_cost` | `float` | `12.0` | Node cost for standing a road *on* a river hex (validated `≥ 0`). Prices out threading a meander or braid while leaving a genuine crossing affordable |
| `road_ferry_max_hop` | `int` | `4` | Longest boat hop used to join a component a river mesh cuts off (validated `≥ 1`). Beyond it, routing raises `RoutingError` |

### 4.18 Roads — Slope Penalty

Slope cost is a rational function of grade percent — zero below `free_pct`, saturating at
`cost × cap_mult` near `cap_pct`. See [road_cost.py](../worldgen/stages/road_cost.py).

| Param | Type | Default | Effect |
|---|---|---|---|
| `road_slope_cost` | `float` | `2.0` | Base slope penalty multiplier |
| `road_slope_free_pct` | `float` | `3.0` | Grade % below which slope is free (validated `≥ 0`) |
| `road_slope_cap_pct` | `float` | `25.0` | Grade % at which the penalty saturates. Validated `> road_slope_free_pct` |
| `road_slope_cap_mult` | `float` | `10.0` | Multiplier applied at saturation. Validated `> 0`, so maximum slope cost is `2.0 × 10 = 20` per edge |

The same `road_slope_cap_pct` is also the threshold for the **`grade_is_under_cap`** check
used by `settlement_min_reachable`: if no road would willingly cross a 25%+ grade, no
settlement should be placed where its only escape requires one.

### 4.19 Roads — Network Classification

| Param | Type | Default | Effect |
|---|---|---|---|
| `road_settlement_skirt_cost` | `float` | `4.0` | What a road pays to pass a settlement at one hex without entering it — an edge whose two ends both neighbour the same seat. The cost-model half of the rule `route_through_settlements` applies afterwards, and the half that can actually shift a route at one hex: a *discount* on the town cannot, because the direct route and the detour both pay for the same two ring hexes, so the detour's extra cost is exactly what the town costs. Drive that to zero and the detour ties; it never wins, and ties go to heap order. Modest at `4.0`, about four hexes of level going — enough to shift a road that was indifferent, not enough to drag one over a mountain to call at a village. Validated `≥ 0` |
| `road_settlement_detour_max_mult` | `float` | `4.0` | A road passing a settlement at one hex is bent through it instead — a road skirting a town at the width of a field is a motor-age idea. This caps what the detour may cost, as a multiple of the edge it replaces; validated `≥ 2.0`, since a detour is two legs where there was one and so costs double on even ground by construction. What it bounds is the ground *beyond* that: the town on the far bank of a river, or up an escarpment. It catches a dear crossing and a steep bank together, which a grade cap would not — the worst case measured cost 31× its bypass at a grade of 4%, having been hauled onto a river channel rather than up anything |
| `road_min_traffic` | `int` | `3` | Minimum traffic for a hex to count as a road at all |
| `road_river_traffic_min` | `int` | `1` | Lower threshold for river hexes (validated `≥ 0`). Lets riverbanks become roads on light traffic |
| `road_primary_pct` | `float` | `0.10` | Top fraction of eligible hexes, by traffic, that become PRIMARY |
| `road_secondary_pct` | `float` | `0.30` | Next fraction, which become SECONDARY |
| `road_track_pct` | `float` | `0.60` | Currently unused by InterurbanRoadStage — TRACK is reserved for village connectors. Kept so the three percentages sum to 1.0 |

---

## 5. In-Code Constants

Magic numbers and weights that live outside `WorldConfig` but materially
shape map output. Change these by editing the source file.

| Name | Value | Location | Effect |
|---|---|---|---|
| `_MAX_STEPS` | `64` | [erosion.py:18](../worldgen/stages/erosion.py#L18) | Max steps per erosion particle. Larger = longer-running particles, deeper channels |
| `_EVAPORATION` | `0.99` | [erosion.py:19](../worldgen/stages/erosion.py#L19) | Per-step water evaporation. Lower = particles die faster, less erosion downstream |
| Erosion delta fan weights | `0.6 / 0.3 / 0.1` | [erosion.py](../worldgen/stages/erosion.py) | Radial falloff over three rings when a droplet unloads at the sea |
| Role: MINING elevation cutoff | `> 0.70` | [city_town.py:21](../worldgen/stages/city_town.py#L21) | **Stale.** A leftover from normalised elevation; 0.70 m in the current units, so `FORTRESS` is never assigned. See § [3.10](#310-city--town-placement) |
| Hydrology epsilon (BFS) | `1e-6` | [hydrology.py:35](../worldgen/stages/hydrology.py#L35) | Per-step plateau tilt magnitude |
| Hydrology epsilon (coord) | `1e-4 * eps` | [hydrology.py:38](../worldgen/stages/hydrology.py#L38) | Coordinate-based tiebreak (≈`1e-10`) |
| Elevation Dijkstra penalty | `× 1000` | [hydrology.py:435](../worldgen/stages/hydrology.py#L435) | Cost multiplier for uphill movement during stalled-river extension |
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
| Pass tag radius | `3` hexes | [city_town.py:137](../worldgen/stages/city_town.py#L137) | Local-max `habitability_town` neighbourhood for `"pass"` tag |
| Village population range | `[100, 1_000]` | [village_placement.py:90](../worldgen/stages/village_placement.py#L90) | |
| Village minimum separation | `3` hexes | [village_placement.py:89](../worldgen/stages/village_placement.py#L89) | Hardcoded — not a `WorldConfig` parameter |
| Village frontier weight bonus | `× 2.0` | [village_placement.py:67](../worldgen/stages/village_placement.py#L67) | |
| Village road-adjacent bonus | `× 1.5` | [village_placement.py:69](../worldgen/stages/village_placement.py#L69) | |
| Road-adjacent habitability boost | `+0.2` (cap 1.0) | [interurban_roads.py:147](../worldgen/stages/interurban_roads.py#L147) | Applied to `habitability_village` only, after road tiers are decided; feeds VillagePlacement |
| Cultivation `RESISTANT` set | `{BOG, MARSH, BARE_ROCK, ALPINE, TUNDRA, DESERT, OPEN_WATER}` | [cultivation.py:6–16](../worldgen/stages/cultivation.py#L6) | Land covers immune to cultivation, used by both Cultivation and VillagePlacement |
| WorldState JSON schema version | `"1.2"` | [world_state.py:27–28](../worldgen/core/world_state.py#L27) | Written by `to_dict`. `from_dict` accepts `1.0`, `1.1` and `1.2`; a `1.0` file's single `habitability` is read into all three tier scores, and `1.0`/`1.1` files get defaults for `territory` and `catchment_km2`. Anything else is rejected by name |

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
