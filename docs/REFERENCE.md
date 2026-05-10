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
   - [3.2 Erosion](#32-erosion)
   - [3.3 Terrain Classification](#33-terrain-classification)
   - [3.4 Water Bodies](#34-water-bodies)
   - [3.5 Hydrology](#35-hydrology)
   - [3.6 Climate](#36-climate)
   - [3.7 Biomes](#37-biomes)
   - [3.8 Land Cover](#38-land-cover)
   - [3.9 Habitability](#39-habitability)
   - [3.10 City & Town Placement](#310-city--town-placement)
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

- **1 hex = 1 km.** Hex size and elevation span can be tuned via
  `hex_size_m` and `road_elev_range_m` (used for grade-percent math in
  road costs), but the pipeline assumes the kilometre-scale interpretation
  for things like settlement separation and cultivation radii.
- **Default grid:** 128 × 128 (≈16,000 km², the size of a small kingdom).
- **Coordinates:** axial `(q, r)`, flat-top hexagons.
  Neighbours, distance, ranges, and pixel conversion live in
  [worldgen/core/hex_grid.py](../worldgen/core/hex_grid.py).
  Hex distance is the standard cube-distance halved:
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

15 stages, run in this exact order
([cli.py:58–74](../worldgen/cli.py#L58)). Each stage is a pure
transformer: `state → state`
([pipeline.py:20](../worldgen/core/pipeline.py#L20)). Stages never write
files; that's `worldgen/export/`'s job.

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
        Hab[HabitabilityStage<br/><i>hex.habitability ∈ [0,1]</i>]
        Cit[CityTownStage<br/><i>cities + towns; pass tags</i>]
        Iur[InterurbanRoadStage<br/><i>PRIMARY/SECONDARY + habitability +0.2</i>]
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
| `hexes` | `dict[HexCoord, Hex]` | Every cell, keyed by `(q, r)` |
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
| `habitability` | `float` | `[0.0, 1.0]` | Habitability, Roads (+0.2 near roads) |
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
| `"pass"` | HILL hex that's a local-max habitability within 3-hex range, no settlement | City/Town |
| `"confluence_town"` | TOWN settled on a hex already tagged `"confluence"` | City/Town |

---

## 3. Pipeline & Algorithms

### 3.1 Elevation

[stages/elevation.py](../worldgen/stages/elevation.py)

**Purpose:** Generate the base heightmap from layered noise.

**Reads:** nothing (works from an empty `WorldState`).
**Writes:** `hex.elevation` for every hex.

**Config:** `noise_octaves`, `noise_persistence`, `noise_lacunarity`,
`noise_scale`, `domain_warp_strength`, `continent_falloff`,
`elevation_gradient`.

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
4. **Linear gradient** (optional) tilts the map by `(gx, gy)` using
   coordinates centred at `[-0.5, +0.5]`
   ([elevation.py:46–50](../worldgen/stages/elevation.py#L46)). Useful
   for biasing one edge upward (e.g., a continental ridge).
5. **Continent falloff** (optional) multiplies by
   `max(0, 1 - sqrt(qf² + rf²))` where `qf`, `rf` are normalised to
   `[-1, +1]` — a circular cone that drops to 0 at the corners
   ([elevation.py:52–56](../worldgen/stages/elevation.py#L52)). This
   is what gives default worlds their island-on-ocean look.
6. **Linear stretch** to `[0, 1]`
   ([elevation.py:58–60](../worldgen/stages/elevation.py#L58)).

**Gotchas**

- The output range is always `[0, 1]` *after* stretch, but
  `sea_level=0.45` is then used as a fixed cutoff. So elevation values
  are not absolute — `0.45` means "the 45th percentile of the noise
  distribution after falloff," not "450 m."
- Without `continent_falloff`, expect a nearly-full-coverage land map
  unless `sea_level` is raised.

---

### 3.2 Erosion

[stages/erosion.py](../worldgen/stages/erosion.py)

**Purpose:** Sculpt valleys by simulating water particles flowing downhill,
removing high-frequency noise, and producing natural-looking channels.

**Reads:** `hex.elevation` (and `sea_level` to bound the simulation to
land).
**Writes:** `hex.elevation` (modified, then re-normalised to `[0, 1]`).

**Config:** `erosion_iterations`, `erosion_inertia`, `erosion_capacity`,
`erosion_deposition`, `erosion_erosion_rate`,
`erosion_channel_affinity_gain`, `erosion_affinity_update_interval`,
`sea_level`.

**Algorithm** (particle-based hydraulic erosion, JIT-compiled with numba
when available — falls back to pure Python).

For each of `erosion_iterations` particles, drop one at a randomly chosen
land hex and simulate up to `_MAX_STEPS = 64` steps of flow
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

**Post-process**

1. Gaussian blur with `sigma=0.5` to remove single-cell artefacts
   ([erosion.py:145](../worldgen/stages/erosion.py#L145)).
2. Linear stretch to `[0, 1]`
   ([erosion.py:147–149](../worldgen/stages/erosion.py#L147)).

**Gotchas**

- Erosion is the bottleneck of the pipeline at default settings (15,000
  particles × up to 64 steps). Halving `erosion_iterations` ≈ halves
  total runtime with mostly cosmetic loss of channel detail.
- Without numba, this stage is roughly 10× slower; install numba
  (`pip install numba`) for full speed.

---

### 3.3 Terrain Classification

[stages/terrain_class.py](../worldgen/stages/terrain_class.py)

**Purpose:** Bucket every hex into `OCEAN / COAST / FLAT / HILL / MOUNTAIN`
based on absolute elevation and local steepness.

**Reads:** `hex.elevation`, neighbours.
**Writes:** `hex.terrain_class`.

**Config:** `sea_level`, `terrain_hill_gradient`,
`terrain_mountain_gradient`.

**Algorithm** ([terrain_class.py:8–41](../worldgen/stages/terrain_class.py#L8)):

```
coast_threshold = sea_level + 0.05            # hardcoded offset

# Pass 1
for hex in all hexes:
    if hex.elevation < sea_level:
        hex.terrain_class = OCEAN

# Pass 2
for hex in non-ocean hexes:
    if hex.elevation < coast_threshold and any neighbour is OCEAN:
        hex.terrain_class = COAST
        continue

    gradient = mean(|hex.elev - n.elev| for n in neighbours)
    if gradient > terrain_mountain_gradient or hex.elevation > 0.8:
        hex.terrain_class = MOUNTAIN
    elif gradient >= terrain_hill_gradient:
        hex.terrain_class = HILL
    else:
        hex.terrain_class = FLAT
```

**Notes**

- The `+0.05` coast offset and the hardcoded `> 0.8` mountain ceiling are
  not in `WorldConfig` — see §5.
- Inland water created by Erosion lows but never connecting to a map
  edge is classified OCEAN here; the next stage corrects that.

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
   - Otherwise re-run the gradient classification (`HILL/FLAT/MOUNTAIN`).

   This pass uses `sea_level`, `terrain_hill_gradient`, and
   `terrain_mountain_gradient` from `state.metadata["config"]` — that's
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

**Config:** `river_flow_threshold`, `river_flow_continuous`.

**Algorithm**

Eight phases, top to bottom in
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
   their upstream tributaries. Map maximum is the largest river-mouth
   value.

5. **River extraction** ([hydrology.py:51–60](../worldgen/stages/hydrology.py#L51)):
   ```
   n_river = max(1, round(n_land * river_flow_threshold))
   river_set = top-n_river hexes by acc
   ```
   Sorting + slicing (rather than a quantile threshold) avoids over-
   selection at tie boundaries. With default `river_flow_threshold=0.05`,
   the top 5 % of land hexes by drainage become river hexes.

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

**Config:** `base_temperature`, `latitude_temp_range`,
`altitude_lapse_rate`, `wind_direction`, `orographic_strength`,
`base_moisture`, `moisture_bleed_passes`, `moisture_bleed_strength`,
`sea_level`.

#### Temperature ([climate.py:18–39](../worldgen/stages/climate.py#L18))

For each hex:
```
row_frac = r / max(height - 1, 1)             # 0 at top, 1 at bottom
lat_temp = sin(row_frac * π)                   # 0 at poles, 1 at equator
temperature = base_temperature
            + (lat_temp - 2/π) * latitude_temp_range
            - elevation * altitude_lapse_rate
temperature = clamp(temperature, 0, 1)
```

The `2/π` subtraction is the mean of `sin` over `[0, π]`, so
`base_temperature` represents the *true map-average* temperature instead
of the equator value. The output is then Gaussian-blurred with
`sigma=1.0` to smooth pixel-by-pixel jaggedness
([climate.py:36](../worldgen/stages/climate.py#L36)).

**At default settings** (`latitude_temp_range=0.1`, `altitude_lapse_rate=0.4`):
elevation dominates latitude. A peak at elevation `0.8` is `0.32` colder
than sea level — equivalent to crossing more than the entire pole-to-
equator gradient. This was tuned for `1 hex = 1 km`; a continent-scale
map (1 hex = 100 km) would need a much larger `latitude_temp_range`.

#### Moisture ([climate.py:41–161](../worldgen/stages/climate.py#L41))

Three sub-steps:

1. **Orographic precipitation.** Sort all hexes by their dot product with
   the wind direction (so upwind hexes process first). The wind carries
   atmospheric moisture; lifting it over higher terrain causes it to
   condense as rain ([climate.py:52–93](../worldgen/stages/climate.py#L52)):
   ```
   incoming = mean(atm[upwind neighbours]) or 1.0 if none upwind
   lift     = max(0, hex.elevation - sea_level)
   fraction = min(1, lift * orographic_strength)
   hex.moisture = incoming * fraction
   atm[hex]     = incoming - precip          # depleted moving downwind
   ```
   Result: windward slopes are wet, lee slopes (rain shadows) dry.

2. **River + coastal bonuses** ([climate.py:96–108](../worldgen/stages/climate.py#L96)).
   For every land hex:
   - If `moisture_bleed_passes == 0` (default), add `+0.15` if any
     neighbour has `"river"` tag.
   - Add `+0.1` if any neighbour is OCEAN or LAKE.

   These are cumulative — a coastal river-adjacent hex gets `+0.25`.

3. **Normalisation** to land-only `[0, 1]` ([climate.py:110–118](../worldgen/stages/climate.py#L110)).

4. **Optional moisture bleed** ([climate.py:120–154](../worldgen/stages/climate.py#L120)).
   When `moisture_bleed_passes > 0`, the flat `+0.15` river bonus is
   replaced by an iterative diffusion: each pass, a hex gains
   `moisture_bleed_strength * max(neighbour.river_flow)` from any
   river-tagged neighbour at `≤` its own elevation. This builds a wider
   moisture corridor along big rivers, especially in valleys, but never
   uphill. Re-normalised after the passes.

5. **Base moisture floor** ([climate.py:156–160](../worldgen/stages/climate.py#L156)):
   add `base_moisture` to every land hex (then clamp to `[0, 1]`). Use
   to dial the whole world wetter or drier.

**Ocean and lake hexes** keep `moisture = 1.0`
([climate.py:60–74](../worldgen/stages/climate.py#L60)).

---

### 3.7 Biomes

[stages/biomes.py](../worldgen/stages/biomes.py)

**Purpose:** Whittaker-style biome assignment based on temperature,
moisture, elevation, and water/river adjacency.

**Reads:** `hex.terrain_class`, `hex.elevation`, `hex.temperature`,
`hex.moisture`, `hex.tags`.
**Writes:** `hex.biome`.

**Config:** `biome_alpine_elev`, `biome_cold_temp`, `biome_warm_temp`,
`biome_dry_moist`, `biome_wet_moist`.

**Algorithm** ([biomes.py:7–46](../worldgen/stages/biomes.py#L7)):

```
if terrain_class in (OCEAN, LAKE):
    biome = OCEAN
elif elevation > alpine_elev:                         # default 0.85
    biome = ALPINE
elif temperature < cold_temp:                         # default 0.25
    biome = TUNDRA   if moisture < dry_moist          # default 0.20
            BOREAL   otherwise
elif temperature >= warm_temp:                        # default 0.60
    biome = DESERT     if moisture < dry_moist
            GRASSLAND  if dry_moist <= moisture < wet_moist   # default 0.50
            TROPICAL   if moisture >= wet_moist
else:  # temperate
    biome = SHRUBLAND        if moisture < dry_moist
            GRASSLAND        if dry_moist <= moisture < wet_moist
            TEMPERATE_FOREST if moisture >= wet_moist

# WETLAND override (post-pass)
if terrain_class in (FLAT, COAST)
   and moisture > wet_moist
   and "river" in tags
   and elevation <= alpine_elev:
    biome = WETLAND
```

**Notes**

- Thresholds are in normalised `[0, 1]` units, matched to
  `temperature` and `moisture`.
- The WETLAND override depends on the `"river"` tag, so it only fires
  *immediately* on river hexes — not a thick wetland buffer. To get
  wider wetlands, raise `moisture_bleed_passes` so more hexes pass the
  `moisture > wet_moist` test.

---

### 3.8 Land Cover

[stages/land_cover.py](../worldgen/stages/land_cover.py)

**Purpose:** Pure derivation of `land_cover` from `terrain_class`, `biome`,
and `moisture`. Adds visual texture without rolling new dice.

**Reads:** `hex.terrain_class`, `hex.biome`, `hex.moisture`.
**Writes:** `hex.land_cover`.

**Config:** `biome_wet_moist` (used to set the dense-forest threshold).

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

**Purpose:** Composite `[0, 1]` score per hex, used as the input to all
settlement placement.

**Reads:** `hex.terrain_class`, `hex.biome`, `hex.tags`, neighbours.
**Writes:** `hex.habitability`.

**Config:** none — all weights are hardcoded (see §5).

**Algorithm** ([habitability.py:20–69](../worldgen/stages/habitability.py#L20)):

```
# Hard zeros
if terrain_class in (OCEAN, LAKE, MOUNTAIN) or biome == WETLAND:
    raw = 0.0
    continue

raw = 0.0
if "river" in hex.tags or any neighbour has "river":
    raw += 0.35                       # river adjacency
raw += 0.25 * AGRI_SCORE[biome]       # agricultural potential
if terrain_class == HILL and any neighbour is FLAT:
    raw += 0.15                       # defensible hill
if terrain_class == COAST or any neighbour is COAST:
    raw += 0.15                       # coastal access
if "confluence" in hex.tags:
    raw += 0.10                       # river junction

# Final normalisation
hex.habitability = raw / max(raw across map)
```

`AGRI_SCORE` ([habitability.py:6–17](../worldgen/stages/habitability.py#L6)):
GRASSLAND `1.0`, TROPICAL `0.8`, TEMPERATE_FOREST `0.7`, SHRUBLAND `0.5`,
BOREAL `0.3`, everything else `0.0`.

**Notes**

- The score *before* normalisation can exceed 1 (max sum is
  `0.35 + 0.25 + 0.15 + 0.15 + 0.10 = 1.00` if every bonus fires *and*
  the biome is GRASSLAND), so normalisation is by the actual map-max,
  not a fixed cap.
- Roads add `+0.2` to neighbour habitability *after* this stage, in
  Interurban Roads — so village placement sees a different (boosted)
  habitability than city/town placement did.

---

### 3.10 City & Town Placement

[stages/city_town.py](../worldgen/stages/city_town.py)

**Purpose:** Place up to `target_city_count` cities and `target_town_count`
towns by greedy selection on habitability with minimum-separation
constraints.

**Reads:** `hex.habitability`, `hex.terrain_class`, `hex.biome`,
`hex.elevation`, neighbours.
**Writes:** `hex.settlement`, `state.settlements`, `hex.tags` (`"pass"`,
`"confluence_town"`).

**Config:** `target_city_count`, `target_town_count`, `city_min_separation`,
`town_min_separation`, `settlement_min_reachable`, plus the road-grade
parameters used for reachability (`hex_size_m`, `road_elev_range_m`,
`road_slope_cap_pct`).

**Reachability filter.** For each candidate hex, BFS over land
neighbours where the connecting edge satisfies
`grade_pct < road_slope_cap_pct` (default 25 %)
([city_town.py:37–47](../worldgen/stages/city_town.py#L37),
[hex_grid.py:142–171](../worldgen/core/hex_grid.py#L142)). If fewer than
`settlement_min_reachable` (default 100) hexes are reachable, the
candidate is rejected. This keeps cities off geographically isolated
peaks and tiny islands.

**Cities** ([city_town.py:62–82](../worldgen/stages/city_town.py#L62)):
sort all land hexes by `habitability` descending; greedily accept each
one whose distance from every prior city is `>= city_min_separation`.
Each city gets a uniform-random population in `[10_000, 50_000]` and a
role from `_assign_role` (below).

**Towns** ([city_town.py:84–128](../worldgen/stages/city_town.py#L84)):

1. Build an *adjusted* habitability map: multiply by `0.5` everywhere
   within 30 hexes of any city ([city_town.py:85–89](../worldgen/stages/city_town.py#L85)).
   This stops towns from clustering in the cities' best-spots.
2. Find local maxima of the adjusted score (hexes whose adjusted score
   beats all 6 neighbours).
3. Greedy placement with `town_min_separation` (default 8). Population
   uniform random in `[1_000, 10_000]`. Towns on `"confluence"` hexes
   also get the `"confluence_town"` tag.

**Role assignment** ([city_town.py:8–29](../worldgen/stages/city_town.py#L8)):

```
PORT         if "river" tag, COAST terrain, or any neighbour matches either
MINING       elif any mountain neighbour has elevation > 0.70
FORTRESS     elif any mountain neighbour (but lower)
AGRICULTURAL elif >= 3 neighbours are GRASSLAND or TEMPERATE_FOREST
MARKET       otherwise
```

**Pass tagging** ([city_town.py:130–142](../worldgen/stages/city_town.py#L130)):
After settlements are placed, every empty HILL hex that is the local-max
habitability within 3-hex range gets the `"pass"` tag. Used for rendering
mountain passes and as a convenient query for module authors.

---

### 3.11 Interurban Roads

[stages/interurban_roads.py](../worldgen/stages/interurban_roads.py)

**Purpose:** Build the inter-city road network (PRIMARY and SECONDARY
tiers). Uses gravity-model traveller simulation over A*-pathed routes,
with self-reinforcing pheromone trails.

**Reads:** `hex.terrain_class`, `hex.elevation`, `hex.river_flow`,
`hex.coord`, `state.settlements` (CITY and TOWN tiers only).
**Writes:** `state.roads`, `hex.road_connections`, `hex.tags`
(`"ford"` / `"bridge"`), `hex.habitability` (+0.2 boost).

**Config:** `road_travellers_city`, `road_travellers_town`,
`road_gravity_exponent`, `road_river_discount`,
`road_river_discount_min_flow`, `road_pheromone_factor`,
`road_mountain_cost`, `road_hill_cost`, `road_flat_cost`,
`road_water_cost`, `road_embark_cost`, `road_disembark_cost`,
`road_river_crossing_base`, `road_river_crossing_flow`,
`road_slope_cost`, `road_slope_free_pct`, `road_slope_cap_pct`,
`road_slope_cap_mult`, `road_min_traffic`, `road_river_traffic_min`,
`road_primary_pct`, `road_secondary_pct`, `hex_size_m`,
`road_elev_range_m`.

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
    MOUNTAIN     → road_mountain_cost       (10.0)
    HILL         → road_hill_cost           (3.0)
    COAST | FLAT → road_flat_cost           (1.0)

river_discount = road_river_discount * max(river_flow, road_river_discount_min_flow)
                                     if river_flow > 0 else 0
pheromone      = road_pheromone_factor * traffic_so_far[hex]

node_cost = max(0, base_cost - river_discount - pheromone)
```

The river discount makes routes naturally prefer to follow rivers (Roman
"river roads"). The pheromone term makes the *order* of traveller
processing matter — once enough travellers have used a path, it becomes
cheap and subsequent travellers reinforce it. This is what concentrates
random travellers onto a small number of recognisable highways.

**Edge cost** ([road_cost.py:62–92](../worldgen/stages/road_cost.py#L62)):
```
edge_cost = slope_edge_cost + water_edge_cost + river_crossing_edge_cost

# slope_edge_cost
grade_pct = |Δelev| * road_elev_range_m * 100 / hex_size_m   # percent
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

For each settlement, emit `road_travellers_city` (default 500) or
`road_travellers_town` (default 100) travellers. Process them in random
order. Each traveller picks a destination via gravity:
```
dist[d]   = max(1, hex_distance(origin, d))
weight[d] = population[d] / dist[d] ^ road_gravity_exponent      (default 1.5)
prob[d]   = weight[d] / sum(weight)        # excluding origin
```
Then A*-paths from origin to chosen destination using the (live, pheromone-
updated) cost functions. Traffic on every hex of the path is incremented.

To save A* calls, each (origin, destination) pair is cached as a
**canonical route** ([interurban_roads.py:73–75](../worldgen/stages/interurban_roads.py#L73)).
The first traveller does the pathing; everyone after re-uses it. Before
falling back to a fresh A*, the stage tries to **stitch** through an
intermediate settlement (`origin → mid → dest` if both legs are already
canonical) and uses whichever is cheaper
([interurban_roads.py:152–189](../worldgen/stages/interurban_roads.py#L152)).

#### Tier classification — [interurban_roads.py:93–114](../worldgen/stages/interurban_roads.py#L93)

After all travellers are processed, hexes are filtered:
```
eligible = hexes where traffic >= road_min_traffic                   (default 3)
        OR hex.river_flow > 0 and traffic >= road_river_traffic_min  (default 1)
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

**Reads:** `hex.habitability` (already road-boosted), `hex.land_cover`,
`hex.terrain_class`, `hex.cultivated`, `hex.road_connections`,
`state.settlements`.
**Writes:** `hex.settlement`, `state.settlements`.

**Config:** `settlement_min_reachable`, plus the road-grade parameters for
the same reachability filter cities/towns use.

**Candidacy** ([village_placement.py:38–73](../worldgen/stages/village_placement.py#L38)):
a hex is a candidate if **all** of these hold:
- not OCEAN/LAKE
- no existing settlement
- `habitability > 0`
- `land_cover not in RESISTANT` (same set as cultivation)
- `grade_reachable_count(...) >= settlement_min_reachable`
- **and** at least one of: on the cultivation frontier, OR road-adjacent

Each candidate's weight starts at `habitability` and is multiplied:
- `× 2.0` if on the frontier
  ([village_placement.py:67](../worldgen/stages/village_placement.py#L67))
- `× 1.5` if road-adjacent

Both can stack (×3.0).

**Stochastic placement** uses the **Gumbel-max trick**
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
Validation rules are in `__post_init__`
([config.py:42–113](../worldgen/core/config.py#L42)) and are noted inline
below.

### 4.1 Grid

| Param | Type | Default | Range | Effect |
|---|---|---|---|---|
| `width` | `int` | `128` | ≥ 1 | Map width in hexes (`1 hex = 1 km` by convention) |
| `height` | `int` | `128` | ≥ 1 | Map height in hexes |

### 4.2 Elevation Noise — § [3.1](#31-elevation)

| Param | Type | Default | Range | Effect |
|---|---|---|---|---|
| `sea_level` | `float` | `0.45` | `[0, 1]` | Below this, hexes become OCEAN. Used by Erosion (skips ocean), TerrainClass, WaterBodies, Climate (orographic floor) |
| `noise_octaves` | `int` | `6` | ≥ 1 | Number of fBm octaves. Higher = more detail at the cost of speed |
| `noise_persistence` | `float` | `0.5` | `(0, 1]` typ. | Amplitude multiplier per octave (`amp *= persistence^i`). Higher = rougher terrain |
| `noise_lacunarity` | `float` | `2.0` | `> 1` typ. | Frequency multiplier per octave (`freq *= lacunarity^i`). Higher = more high-frequency detail |
| `noise_scale` | `float` | `3.0` | `> 0` | Coordinate scale: domain spans `[0, noise_scale]`. Higher = "zoomed in" (more land variation per hex) |
| `domain_warp_strength` | `float` | `0.3` | `≥ 0` | Magnitude of the domain warp offset. `0` disables warping; higher values produce more organic coastlines |
| `continent_falloff` | `bool` | `True` | — | Apply circular cone falloff to bias water at edges. Turn off for full-coverage land maps |
| `elevation_gradient` | `(float, float)` | `(0.0, 0.0)` | — | Linear tilt added post-noise. `(0.1, 0)` raises the right edge by 0.1 |

### 4.3 Terrain Classification — § [3.3](#33-terrain-classification)

| Param | Type | Default | Range | Effect |
|---|---|---|---|---|
| `terrain_hill_gradient` | `float` | `0.02` | `≥ 0` | Mean-neighbour-Δelev threshold for HILL classification |
| `terrain_mountain_gradient` | `float` | `0.04` | `≥ 0` | Threshold for MOUNTAIN. Hexes with elevation > 0.8 are also MOUNTAIN regardless of gradient |

### 4.4 Erosion — § [3.2](#32-erosion)

| Param | Type | Default | Range | Effect |
|---|---|---|---|---|
| `erosion_iterations` | `int` | `15000` | ≥ 0 | Total particles dropped. The dominant cost in the pipeline. Halve to halve runtime; double for finer channels |
| `erosion_inertia` | `float` | `0.05` | `[0, 1]` | Direction smoothing. `0` = pure gradient descent; near `1` = particle ignores terrain |
| `erosion_capacity` | `float` | `4.0` | `> 0` | Sediment-carrying capacity multiplier. Higher = particles erode more aggressively |
| `erosion_deposition` | `float` | `0.3` | `[0, 1]` typ. | Fraction of excess sediment deposited each step when over-capacity |
| `erosion_erosion_rate` | `float` | `0.3` | `[0, 1]` typ. | Fraction of capacity deficit eroded each step |
| `erosion_channel_affinity_gain` | `float` | `0.5` | `≥ 0` (validated) | Affinity bump per erosion event. Higher = stronger channel-reinforcement bias |
| `erosion_affinity_update_interval` | `int` | `500` | `≥ 1` (validated) | Particles between channel-affinity re-weighting passes |

### 4.5 Hydrology — § [3.5](#35-hydrology)

| Param | Type | Default | Range | Effect |
|---|---|---|---|---|
| `river_flow_threshold` | `float` | `0.05` | `[0, 1]` (validated) | Top fraction of land hexes by flow accumulation that become rivers. `0` disables rivers |
| `river_flow_continuous` | `bool` | `False` | — | If true, `hex.river_flow` is set on every draining land hex (good for flow heatmaps); else only on river hexes |

### 4.6 Climate — § [3.6](#36-climate)

| Param | Type | Default | Range | Effect |
|---|---|---|---|---|
| `wind_direction` | `(float, float)` | `(1.0, 0.0)` | — | Prevailing wind vector (orographic precip & atmospheric moisture transport). Magnitude is normalised; only direction matters |
| `base_temperature` | `float` | `0.5` | `[0, 1]` (validated) | Map-mean temperature. `0`=arctic, `1`=tropical |
| `latitude_temp_range` | `float` | `0.1` | `[0, 1]` (validated) | Pole-to-equator temperature spread. Tiny default tuned for `1 hex = 1 km`; raise for continent-scale maps |
| `altitude_lapse_rate` | `float` | `0.4` | `≥ 0` | Temperature drop per unit elevation. With default settings, elevation dominates latitude — high mountains are reliably alpine |
| `orographic_strength` | `float` | `2.0` | `> 0` | Orographic precipitation aggressiveness. Higher = wetter windward slopes, drier rain shadows |
| `base_moisture` | `float` | `0.0` | typically `[0, 1]` | Flat moisture floor added to every land hex after normalisation. Use to dial the world wetter or drier |
| `moisture_bleed_passes` | `int` | `0` | `≥ 0` (validated) | If `0`, use the simple `+0.15` flat river bonus. If `> 0`, run that many elevation-gated diffusion passes from rivers (creates wider riparian zones) |
| `moisture_bleed_strength` | `float` | `0.3` | `[0, 1]` (validated) | Per-pass bleed strength. Only used when `moisture_bleed_passes > 0` |

### 4.7 Biome Thresholds — § [3.7](#37-biomes)

All thresholds are in normalised `[0, 1]` units, matched to `hex.elevation`,
`hex.temperature`, and `hex.moisture`.

| Param | Type | Default | Range | Effect |
|---|---|---|---|---|
| `biome_alpine_elev` | `float` | `0.85` | `[0, 1]` | Above this, biome is ALPINE regardless of climate |
| `biome_cold_temp` | `float` | `0.25` | `[0, 1]` | Below this, biomes go TUNDRA / BOREAL |
| `biome_warm_temp` | `float` | `0.6` | `[0, 1]` | Above this, biomes go DESERT / GRASSLAND / TROPICAL. Between cold and warm, temperate biomes apply |
| `biome_dry_moist` | `float` | `0.2` | `[0, 1]` | Below this in any temp band, the dry biome (DESERT/SHRUBLAND/TUNDRA) is chosen |
| `biome_wet_moist` | `float` | `0.5` | `[0, 1]` | Above this, the wettest biome of the band (TROPICAL / TEMPERATE_FOREST / BOREAL) is chosen. Also gates the WETLAND override and DENSE_FOREST in LandCover |

### 4.8 Settlements — § [3.10](#310-city--town-placement), [3.13](#313-village-placement)

| Param | Type | Default | Range | Effect |
|---|---|---|---|---|
| `target_city_count` | `int` | `6` | ≥ 0 | Maximum cities placed. Actual count may be lower if fewer candidates pass separation |
| `target_town_count` | `int` | `24` | ≥ 0 | Maximum towns placed |
| `city_min_separation` | `int` | `20` | ≥ 1 | Minimum hex distance between cities |
| `town_min_separation` | `int` | `8` | ≥ 1 | Minimum hex distance between towns. (Villages always use a hardcoded `3`.) |
| `settlement_min_reachable` | `int` | `100` | `≥ 1` (validated) | Minimum hexes reachable below the slope cap. Filters out unreachable peaks and tiny islands |

### 4.9 Cultivation Radii — § [3.12](#312-cultivation-cities--towns), [3.15](#315-village-cultivation)

| Param | Type | Default | Range | Effect |
|---|---|---|---|---|
| `cultivation_city_radius` | `int` | `8` | ≥ 0 | Hex range marked cultivated around each city |
| `cultivation_town_radius` | `int` | `4` | ≥ 0 | Around each town |
| `cultivation_village_radius` | `int` | `2` | ≥ 0 | Around each village (runs after villages place) |

### 4.10 World Scale — § [3.11](#311-interurban-roads)

These exist so road-grade calculations can express slopes in real-world
percent terms, not normalised elevation deltas.

| Param | Type | Default | Range | Effect |
|---|---|---|---|---|
| `hex_size_m` | `float` | `1000.0` | `> 0` (validated) | Metres per hex. With the default, `1 hex = 1 km` |
| `road_elev_range_m` | `float` | `3000.0` | `> 0` (validated) | Real-world elevation span of the `[0, 1]` range. With defaults, an elevation Δ of `0.01` between adjacent hexes equals 30 m over 1 km = 3 % grade |

### 4.11 Roads — Terrain Costs — § [3.11](#311-interurban-roads)

Node-cost (cost to *enter* a hex) by `terrain_class`. See [road_cost.py](../worldgen/stages/road_cost.py).

| Param | Type | Default | Effect |
|---|---|---|---|
| `road_mountain_cost` | `float` | `10.0` | Node cost for MOUNTAIN |
| `road_hill_cost` | `float` | `3.0` | Node cost for HILL |
| `road_flat_cost` | `float` | `1.0` | Node cost for FLAT and COAST |
| `road_water_cost` | `float` | `0.05` | Node cost for OCEAN/LAKE. Validated `≥ 0`. Low to allow short over-water hops; the heavy lifting is in the embark/disembark edge cost |

### 4.12 Roads — Traveller Simulation

| Param | Type | Default | Range | Effect |
|---|---|---|---|---|
| `road_travellers_city` | `int` | `500` | ≥ 0 | Travellers emitted per city |
| `road_travellers_town` | `int` | `100` | ≥ 0 | Per town |
| `road_travellers_village` | `int` | `20` | ≥ 0 | Reserved; not currently consumed by InterurbanRoadStage (which only pathes between cities and towns) |
| `road_gravity_exponent` | `float` | `1.5` | `≥ 0` | Distance exponent in gravity model. Higher = travellers strongly prefer nearby destinations |
| `road_river_discount` | `float` | `0.5` | `[0, 1]` typ. | Maximum node-cost reduction (multiplied by river flow) on river hexes |
| `road_river_discount_min_flow` | `float` | `0.2` | `[0, 1]` (validated) | Floor on `river_flow` used in the discount. Prevents tiny headwaters from losing the discount entirely |
| `road_pheromone_factor` | `float` | `0.1` | `≥ 0` | Cost reduction per unit traffic. Higher = stronger highway-reinforcement effect |

### 4.13 Roads — Water Transitions

Edge cost (charged once on the land↔water transition).

| Param | Type | Default | Effect |
|---|---|---|---|
| `road_embark_cost` | `float` | `8.0` | Land → water (validated `≥ 0`) |
| `road_disembark_cost` | `float` | `8.0` | Water → land (validated `≥ 0`) |

### 4.14 Roads — River Crossings

Edge cost charged on each land↔river transition. A perpendicular crossing
of a 1-hex-wide river hits this twice (entering and leaving the river hex).

| Param | Type | Default | Effect |
|---|---|---|---|
| `road_river_crossing_base` | `float` | `4.0` | Constant component (validated `≥ 0`) |
| `road_river_crossing_flow` | `float` | `12.0` | Multiplied by `max(from.river_flow, to.river_flow)`. Big rivers are dramatically more expensive to bridge |

### 4.15 Roads — Slope Penalty

Slope cost is a **rational** function of grade percent — zero below `free_pct`,
saturating at `cost * cap_mult` near `cap_pct`. See
[road_cost.py:17–29](../worldgen/stages/road_cost.py#L17).

| Param | Type | Default | Effect |
|---|---|---|---|
| `road_slope_cost` | `float` | `2.0` | Base slope penalty multiplier |
| `road_slope_free_pct` | `float` | `3.0` | Grade % below which slope is free (validated `≥ 0`) |
| `road_slope_cap_pct` | `float` | `25.0` | Grade % at which penalty saturates. Validated `> road_slope_free_pct` |
| `road_slope_cap_mult` | `float` | `10.0` | Multiplier applied at saturation. Validated `> 0`. So max slope cost = `2.0 * 10 = 20` per edge |

The same `road_slope_cap_pct` is also the threshold for the
**`grade_is_under_cap`** check used by `settlement_min_reachable` —
i.e., if no road would willingly cross a 25 %+ grade, no settlement
should be placed where its only escape requires one.

### 4.16 Roads — Network Classification

| Param | Type | Default | Effect |
|---|---|---|---|
| `road_min_traffic` | `int` | `3` | Minimum traffic for a hex to count as a road at all |
| `road_river_traffic_min` | `int` | `1` | Lower threshold for river hexes (validated `≥ 0`). Lets riverbanks become roads even with light traffic |
| `road_primary_pct` | `float` | `0.10` | Top fraction of eligible hexes (by traffic) that become PRIMARY |
| `road_secondary_pct` | `float` | `0.30` | Next fraction that become SECONDARY. Remaining eligible hexes become unranked (no road drawn) |
| `road_track_pct` | `float` | `0.60` | Currently unused by InterurbanRoadStage (TRACK is reserved for village connectors). Kept in config so the three percentages sum to 1.0 |

---

## 5. In-Code Constants

Magic numbers and weights that live outside `WorldConfig` but materially
shape map output. Change these by editing the source file.

| Name | Value | Location | Effect |
|---|---|---|---|
| `_MAX_STEPS` | `64` | [erosion.py:18](../worldgen/stages/erosion.py#L18) | Max steps per erosion particle. Larger = longer-running particles, deeper channels |
| `_EVAPORATION` | `0.99` | [erosion.py:19](../worldgen/stages/erosion.py#L19) | Per-step water evaporation. Lower = particles die faster, less erosion downstream |
| Coast threshold offset | `+0.05` | [terrain_class.py:10](../worldgen/stages/terrain_class.py#L10), [water_bodies.py:71](../worldgen/stages/water_bodies.py#L71) | `coast_threshold = sea_level + 0.05`. Land hex below this with an ocean neighbour → COAST |
| Mountain elevation cap | `> 0.8` | [terrain_class.py:36](../worldgen/stages/terrain_class.py#L36), [water_bodies.py:96](../worldgen/stages/water_bodies.py#L96) | Forces MOUNTAIN regardless of gradient |
| Hydrology epsilon (BFS) | `1e-6` | [hydrology.py:35](../worldgen/stages/hydrology.py#L35) | Per-step plateau tilt magnitude |
| Hydrology epsilon (coord) | `1e-4 * eps` | [hydrology.py:38](../worldgen/stages/hydrology.py#L38) | Coordinate-based tiebreak (≈`1e-10`) |
| Elevation Dijkstra penalty | `× 1000` | [hydrology.py:435](../worldgen/stages/hydrology.py#L435) | Cost multiplier for uphill movement during stalled-river extension |
| Erosion Gaussian sigma | `0.5` | [erosion.py:145](../worldgen/stages/erosion.py#L145) | Final smoothing pass after erosion |
| Temperature Gaussian sigma | `1.0` | [climate.py:36](../worldgen/stages/climate.py#L36) | Smoothing pass on temperature field |
| Flat river moisture bonus | `+0.15` | [climate.py:103](../worldgen/stages/climate.py#L103) | Used when `moisture_bleed_passes == 0` |
| Coastal moisture bonus | `+0.10` | [climate.py:107](../worldgen/stages/climate.py#L107) | Always applied to land hexes adjacent to OCEAN/LAKE |
| LandCover dense-forest threshold | `(wet_moist + 1) / 2` | [land_cover.py:37](../worldgen/stages/land_cover.py#L37) | Splits TEMPERATE_FOREST into DENSE_FOREST vs WOODLAND |
| Habitability — river adjacency | `+0.35` | [habitability.py:39](../worldgen/stages/habitability.py#L39) | |
| Habitability — agricultural | `+0.25 × AGRI_SCORE` | [habitability.py:42](../worldgen/stages/habitability.py#L42) | |
| Habitability — defensible hill | `+0.15` | [habitability.py:48](../worldgen/stages/habitability.py#L48) | HILL with FLAT neighbour |
| Habitability — coastal | `+0.15` | [habitability.py:54](../worldgen/stages/habitability.py#L54) | |
| Habitability — confluence | `+0.10` | [habitability.py:58](../worldgen/stages/habitability.py#L58) | |
| `AGRI_SCORE` table | dict | [habitability.py:6–17](../worldgen/stages/habitability.py#L6) | Per-biome agricultural multiplier (GRASSLAND=1.0, TROPICAL=0.8, TEMPERATE_FOREST=0.7, SHRUBLAND=0.5, BOREAL=0.3, others 0) |
| City population range | `[10_000, 50_000]` | [city_town.py:69](../worldgen/stages/city_town.py#L69) | Uniform random per city |
| Town population range | `[1_000, 10_000]` | [city_town.py:113](../worldgen/stages/city_town.py#L113) | |
| Town habitability damp | `× 0.5 within 30 hexes of any city` | [city_town.py:87–89](../worldgen/stages/city_town.py#L87) | |
| Town placement role: MINING elev cutoff | `> 0.70` | [city_town.py:21](../worldgen/stages/city_town.py#L21) | Mountain neighbour elevation needed for MINING role |
| Town placement role: AGRICULTURAL fertile-neighbour count | `>= 3` | [city_town.py:26](../worldgen/stages/city_town.py#L26) | GRASSLAND or TEMPERATE_FOREST neighbours required |
| Pass tag radius | `3` hexes | [city_town.py:137](../worldgen/stages/city_town.py#L137) | Local-max habitability neighbourhood for `"pass"` tag |
| Village population range | `[100, 1_000]` | [village_placement.py:90](../worldgen/stages/village_placement.py#L90) | |
| Village minimum separation | `3` hexes | [village_placement.py:89](../worldgen/stages/village_placement.py#L89) | Hardcoded — not a `WorldConfig` parameter |
| Village frontier weight bonus | `× 2.0` | [village_placement.py:67](../worldgen/stages/village_placement.py#L67) | |
| Village road-adjacent bonus | `× 1.5` | [village_placement.py:69](../worldgen/stages/village_placement.py#L69) | |
| Road-adjacent habitability boost | `+0.2` (cap 1.0) | [interurban_roads.py:147](../worldgen/stages/interurban_roads.py#L147) | Applied after road tiers are decided; feeds VillagePlacement |
| Cultivation `RESISTANT` set | `{BOG, MARSH, BARE_ROCK, ALPINE, TUNDRA, DESERT, OPEN_WATER}` | [cultivation.py:6–16](../worldgen/stages/cultivation.py#L6) | Land covers immune to cultivation, used by both Cultivation and VillagePlacement |
| WorldState JSON schema version | `"1.0"` | [world_state.py:84, 140](../worldgen/core/world_state.py#L84) | Round-trip compatibility check |

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
| `habitability.png` | Composite settlement-suitability score (post-road boost) |
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
- **Gumbel-max trick** — Equivalent to weighted-without-replacement
  sampling: draw `u ~ Uniform(0,1)` per item and sort by
  `-u^(1/weight)`. Avoids the cost of repeated cumulative-distribution
  builds.
- **Cultivation frontier** — Hexes that are cultivated but border
  uncultivated land. Used as the natural location for new villages.
