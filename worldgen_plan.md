# Worldgen — Procedural Map Generation System
## Implementation Plan for Claude Code

**Project:** Hex-based procedural world generator for TTRPGs, worldbuilding, and wargaming  
**Scale:** 1 hex = 1 km  
**Language:** Python 3.11+  
**Seed:** Every run must be fully reproducible from a single integer seed

---

## Architecture Constraints (Never Violate These)

- Data, logic, and rendering are always separate layers. Generators never import from render/.
- Every random call uses a seeded `numpy.random.Generator` passed explicitly — no global random state.
- Each pipeline stage is a pure-ish transformer: `stage.run(WorldState) -> WorldState`.
- All tunable parameters live in `WorldConfig`. Nothing is hardcoded in stage logic.
- Export is first-class: JSON, SVG, and PNG are supported outputs from day one.

---

## Project Structure

```
worldgen/
├── core/
│   ├── hex.py              # Hex dataclass, HexCoord, TerrainClass, Biome enums
│   ├── world_state.py      # WorldState, River, Settlement, Road dataclasses
│   ├── pipeline.py         # GeneratorPipeline, GeneratorStage base class
│   ├── config.py           # WorldConfig dataclass + named presets
│   └── hex_grid.py         # Axial math, neighbor lookup, A*, ring/range queries
├── stages/
│   ├── elevation.py        # Fractal Brownian Motion + domain warping
│   ├── erosion.py          # Particle-based hydraulic erosion
│   ├── terrain_class.py    # Classify hexes: ocean/coast/flat/hill/mountain
│   ├── hydrology.py        # Sink fill, flow accumulation, river extraction, lakes
│   ├── climate.py          # Temperature gradient, orographic moisture, rain shadow
│   ├── biomes.py           # Whittaker-style temperature × moisture → biome
│   ├── habitability.py     # Composite hex score for settlement placement
│   ├── settlements.py      # City → town → village placement by tier
│   └── roads.py            # Cost-weighted A* road network generation
├── export/
│   ├── json_export.py      # Full WorldState → JSON
│   ├── svg_export.py       # Hex map → SVG with configurable layers
│   └── png_export.py       # Rasterized map export via Pillow
├── render/
│   └── debug_viewer.py     # matplotlib hex renderer for development
├── tests/
│   ├── test_hex_grid.py
│   ├── test_elevation.py
│   ├── test_hydrology.py
│   ├── test_settlements.py
│   └── test_pipeline.py
├── presets/
│   ├── temperate_continent.json
│   ├── arid_archipelago.json
│   └── river_delta.json
├── cli.py                  # Entry point: worldgen generate --seed 42 --config preset.json
└── requirements.txt
```

---

## Phase 0 — Foundation

**Goal:** Pipeline runs end-to-end with empty stages. Hex grid math is correct. Debug viewer shows output.

### Task 0.1 — Dependencies

```
requirements.txt:
numpy>=1.26
scipy>=1.12
noise>=1.2.2
opensimplex>=0.4
networkx>=3.2
matplotlib>=3.8
Pillow>=10.0
click>=8.1
```

### Task 0.2 — `core/hex.py`

Implement:
- `HexCoord = tuple[int, int]` (axial q, r)
- `TerrainClass` enum: `OCEAN, COAST, FLAT, HILL, MOUNTAIN`
- `Biome` enum: `TUNDRA, BOREAL, TEMPERATE_FOREST, GRASSLAND, SHRUBLAND, DESERT, TROPICAL, WETLAND, OCEAN, ALPINE`
- `@dataclass Hex` with fields:
  - `coord: HexCoord`
  - `elevation: float` (0.0–1.0)
  - `moisture: float` (0.0–1.0)
  - `temperature: float` (0.0–1.0)
  - `biome: Biome | None`
  - `river_flow: float` (0.0 = none)
  - `terrain_class: TerrainClass`
  - `settlement: Settlement | None`
  - `road_connections: set[HexCoord]`
  - `tags: set[str]`  ← freeform: "ford", "pass", "confluence", "harbor"

### Task 0.3 — `core/hex_grid.py`

Implement axial coordinate operations. Reference: https://www.redblobgames.com/grids/hexes/

- `neighbors(coord) -> list[HexCoord]` — 6 neighbors in axial coords
- `distance(a, b) -> int` — hex distance
- `ring(center, radius) -> list[HexCoord]`
- `hex_range(center, radius) -> list[HexCoord]`
- `axial_to_pixel(coord, hex_size) -> tuple[float, float]` — flat-top layout
- `pixel_to_axial(x, y, hex_size) -> HexCoord`
- `astar(grid, start, goal, cost_fn) -> list[HexCoord] | None`
  - `cost_fn(hex: Hex) -> float` is passed in externally
  - Returns None if no path exists

**Test in `tests/test_hex_grid.py`:**
- `distance(origin, neighbor) == 1` for all 6 neighbors
- `len(ring(origin, 2)) == 12`
- A* finds shortest path on flat grid
- A* returns None when blocked

### Task 0.4 — `core/world_state.py`

```python
@dataclass
class WorldState:
    seed: int
    width: int                      # hex columns
    height: int                     # hex rows
    hexes: dict[HexCoord, Hex]
    rivers: list[River]
    settlements: list[Settlement]
    roads: list[Road]
    metadata: dict                  # stores all generator params

    @classmethod
    def empty(cls, seed: int, config: WorldConfig) -> WorldState: ...

    def get(self, coord: HexCoord) -> Hex | None: ...
    def all_land(self) -> list[Hex]: ...
    def all_water(self) -> list[Hex]: ...

@dataclass
class River:
    hexes: list[HexCoord]           # ordered source → mouth
    flow_volume: float

@dataclass
class Settlement:
    coord: HexCoord
    tier: SettlementTier            # CITY, TOWN, VILLAGE
    role: SettlementRole            # AGRICULTURAL, PORT, MINING, FORTRESS, MARKET
    population: int
    name: str

@dataclass
class Road:
    path: list[HexCoord]
    tier: RoadTier                  # PRIMARY, SECONDARY, TRACK
```

### Task 0.5 — `core/pipeline.py`

```python
class GeneratorStage(ABC):
    def __init__(self, config: WorldConfig, rng: np.random.Generator): ...

    @abstractmethod
    def run(self, state: WorldState) -> WorldState: ...

class GeneratorPipeline:
    def __init__(self, seed: int, config: WorldConfig):
        self.rng = np.random.default_rng(seed)

    def add_stage(self, stage_cls: type[GeneratorStage]) -> Self: ...
    def run(self) -> WorldState: ...
```

Each stage receives its own child RNG derived from the pipeline RNG so stage order changes don't affect other stages' outputs.

### Task 0.6 — `core/config.py`

```python
@dataclass
class WorldConfig:
    width: int = 128
    height: int = 128
    sea_level: float = 0.45         # elevation threshold for ocean
    # Elevation
    noise_octaves: int = 6
    noise_persistence: float = 0.5
    noise_lacunarity: float = 2.0
    noise_scale: float = 3.0
    domain_warp_strength: float = 0.3
    continent_falloff: bool = True
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
    city_min_separation: int = 20   # hex distance
    town_min_separation: int = 8
    target_city_count: int = 6
    target_town_count: int = 24
    # Roads
    road_mountain_cost: float = 10.0
    road_hill_cost: float = 3.0
    road_flat_cost: float = 1.0
    road_river_crossing_cost: float = 5.0
```

Load from JSON. Support named presets in `presets/`.

### Task 0.7 — `render/debug_viewer.py`

matplotlib renderer that colorizes hexes by a given attribute and saves PNG.

```python
def render(state: WorldState, attribute: str, output_path: str): ...
# attribute: "elevation" | "moisture" | "temperature" | "biome" | "river_flow" | "habitability"
```

Use flat-top hex layout. Draw hex outlines. Color by attribute using sensible colormaps (terrain = terrain colormap, moisture = Blues, etc.).

**This viewer must be runnable after every phase to inspect intermediate output.**

### Task 0.8 — `cli.py` skeleton

```
worldgen generate --seed INT --config PATH --output-dir PATH --width INT --height INT
worldgen render --input PATH --attribute STR --output PATH
```

---

## Phase 1 — Terrain & Elevation

**Goal:** Generates a plausible heightmap with mountain ranges, valleys, and coastal regions.

### Task 1.1 — `stages/elevation.py`

`ElevationStage.run(state) -> WorldState`

Steps:
1. Generate a 2D numpy float array of shape `(width, height)` using layered Simplex noise (use `opensimplex`).
2. Apply **domain warping**: offset each sample coordinate by another noise function scaled by `config.domain_warp_strength`. This breaks up the regular look.
3. If `config.continent_falloff`: multiply by a radial falloff mask (elliptical, peaks at center, approaches 0 at edges). This produces island/continent shapes.
4. Normalize to [0, 1].
5. Sample heightmap at each hex center and assign to `hex.elevation`.

### Task 1.2 — `stages/erosion.py`

`ErosionStage.run(state) -> WorldState`

Work on a numpy heightmap array (convert hex dict → array, erode, convert back).

Implement particle-based hydraulic erosion:
1. Drop a particle at a random land position.
2. Particle flows downhill (biased by inertia: mix of previous direction and steepest descent).
3. Each step: pick up sediment proportional to gradient × speed × capacity; deposit sediment if carrying more than capacity.
4. Particle evaporates after N steps or when it reaches water.
5. Repeat for `config.erosion_iterations`.

After erosion: smooth the result with a light Gaussian blur (sigma=0.5) to remove single-pixel artifacts, then re-normalize and write back to hexes.

### Task 1.3 — `stages/terrain_class.py`

`TerrainClassificationStage.run(state) -> WorldState`

Assign `hex.terrain_class` based on:
- `elevation < sea_level` → `OCEAN`
- `sea_level <= elevation < sea_level + 0.05` → `COAST`
- Gradient < 0.08 → `FLAT`
- Gradient 0.08–0.2 → `HILL`
- Gradient > 0.2 or elevation > 0.8 → `MOUNTAIN`

Gradient = average elevation difference from 6 neighbors.

**Test:** After running all three stages, at least 40% of hexes should be land. Mountain hexes should form contiguous clusters, not isolated spikes.

---

## Phase 2 — Hydrology

**Goal:** Plausible river network flows from mountains to sea. Lakes form naturally. Confluences are tagged.

### Task 2.1 — `stages/hydrology.py`

Implement as a single stage with internal sub-steps, or split into `SinkFillStage` + `FlowStage` + `RiverStage` — your choice. The WorldState output must have:
- `hex.river_flow` set for all river hexes (0.0 = no river)
- `state.rivers` populated with ordered `River` objects (source → mouth)
- River hexes tagged: `"headwater"`, `"confluence"`, `"river_mouth"`

**Sub-step A — Sink Filling:**  
Use the Priority-Flood algorithm on the elevation array to fill closed depressions. Without this, rivers terminate in random inland pits.

Reference: Barnes et al. (2014) "Priority-flood: An optimal depression-filling and watershed-labeling algorithm."

**Sub-step B — Flow Direction:**  
For each land hex, flow direction = lowest neighbor. If tied, break ties by index for determinism.

**Sub-step C — Flow Accumulation:**  
Topologically sort hexes (upstream first). For each hex, `flow[hex] += 1 + sum(flow[uphill_neighbors])`. Store result normalized to [0, 1] as `hex.river_flow`.

**Sub-step D — River Extraction:**  
Hexes where `flow_accumulation > threshold` (configurable, default top 5%) are rivers. Threshold should produce connected river paths, not isolated high-flow dots — validate this.

**Sub-step E — Tag special hexes:**  
- Confluence: river hex with 2+ upstream river neighbors
- Headwater: river hex with 0 upstream river neighbors  
- River mouth: river hex adjacent to ocean

**Test:** Every river path must be connected (no gaps). Every river must terminate at ocean or a lake. No river should flow uphill.

---

## Phase 3 — Climate & Biomes

**Goal:** Temperature and moisture vary realistically. Biomes match climate. Rain shadows appear on mountain leeward sides.

### Task 3.1 — `stages/climate.py`

`ClimateStage.run(state) -> WorldState`

**Temperature:**
- Latitude component: hex row normalized to [0,1]; apply sine curve so mid-latitudes are warm, edges cool. Scale by `config.latitude_temp_range`.
- Altitude component: subtract `hex.elevation * config.altitude_lapse_rate`.
- Normalize result to [0, 1] and store as `hex.temperature`.

**Moisture:**
- Start with base moisture = 0.5 for all land hexes.
- Wind vector (from config) determines prevailing direction.
- For each hex, accumulate moisture lost by upwind hexes hitting terrain. Windward side of mountains = wet; leeward = dry.
- Add moisture bonus for river adjacency and coastal proximity.
- Ocean hexes = 1.0. Normalize and store as `hex.moisture`.

### Task 3.2 — `stages/biomes.py`

`BiomeStage.run(state) -> WorldState`

Whittaker lookup — assign `hex.biome` based on temperature × moisture thresholds:

| | Low moisture | Medium moisture | High moisture |
|---|---|---|---|
| **Cold** | TUNDRA | TUNDRA | BOREAL |
| **Temperate** | SHRUBLAND | GRASSLAND | TEMPERATE_FOREST |
| **Warm** | DESERT | GRASSLAND | TROPICAL |

Ocean hexes → `Biome.OCEAN`. Mountain hexes above elevation 0.85 → `ALPINE`.

Expose threshold values in WorldConfig so users can tune biome distributions.

---

## Phase 4 — Settlements & Roads

**Goal:** Cities, towns, and villages emerge from geography. Road network connects them with realistic routing.

### Task 4.1 — `stages/habitability.py`

`HabitabilityStage.run(state) -> WorldState`

Compute a composite habitability score [0, 1] per land hex. Store on hex (add `habitability: float` field).

Weighted sum of:
- River adjacency: +0.35 if adjacent to river hex
- Agricultural potential: GRASSLAND=1.0, TEMPERATE_FOREST=0.7, SHRUBLAND=0.5, BOREAL=0.3, DESERT/TUNDRA=0.0 — weighted 0.25
- Defensibility: HILL adjacent to FLAT = +0.15
- Coastal access: adjacent to COAST hex = +0.15
- Penalty: OCEAN, MOUNTAIN, WETLAND = 0.0 (override entire score)

Normalize final scores to [0, 1] across all land hexes.

### Task 4.2 — `stages/settlements.py`

`SettlementPlacementStage.run(state) -> WorldState`

**Cities:** Select top-N habitability hexes subject to minimum separation (`config.city_min_separation`). Use a greedy approach: sort by habitability desc, place city if no existing city within min_separation hexes. Target `config.target_city_count`.

**Towns:** Re-score hexes excluding city influence zones (within 10 hexes of a city, halve the score). Place towns at remaining local maxima with `config.town_min_separation`. Target `config.target_town_count`.

**Villages:** Poisson disk sampling over habitable hexes (habitability > 0.3) with min separation of 3 hexes. River and coast hexes weighted 2×.

**Special sites — tag these hexes:**
- `"pass"`: HILL or MOUNTAIN hex between two lower regions that is highest-habitability in its local area
- `"confluence_town"`: confluence-tagged hex with high flow and high habitability
- `"ford"`: river hex on a road crossing (added in road stage)

**Assign roles** from geography:
- Adjacent to major river → `PORT`
- Adjacent to coast → `PORT`
- On or adjacent to MOUNTAIN → `FORTRESS` or `MINING`
- Surrounded by GRASSLAND/TEMPERATE_FOREST → `AGRICULTURAL`
- Otherwise → `MARKET`

**Names:** Generate placeholder names as `{biome}_{tier}_{index}` — e.g. `grassland_city_0`. Real name generation is a future enhancement.

### Task 4.3 — `stages/roads.py`

`RoadNetworkStage.run(state) -> WorldState`

**Traversal cost function:**
```python
def traversal_cost(hex: Hex) -> float:
    base = {
        TerrainClass.FLAT: 1.0,
        TerrainClass.COAST: 1.2,
        TerrainClass.HILL: 3.0,
        TerrainClass.MOUNTAIN: 10.0,
        TerrainClass.OCEAN: float('inf'),
    }[hex.terrain_class]
    if hex.river_flow > 0:
        base += config.road_river_crossing_cost  # river crossing penalty
    return base
```

**Connection order:**
1. Connect each city to every other city (primary roads) — use A*.
2. Connect each town to its nearest city (secondary roads).
3. Connect each village to its nearest town or road hex (tracks).

**Road merging:** Track how many times each hex appears in a road path. Hexes appearing in 3+ routes are upgraded to primary. 2 routes = secondary. 1 route = track.

**Ford tagging:** Any river hex that a road passes through → tag `"ford"`. If two or more roads cross the same river hex → tag `"bridge"`.

**Iterate once:** After roads exist, re-score habitability with road adjacency bonus (+0.2 if adjacent to PRIMARY road). Promote any village with score > 0.8 to town. This loop runs once only.

---

## Phase 5 — Export

**Goal:** Full map exportable as JSON, SVG, and PNG.

### Task 5.1 — `export/json_export.py`

Serialize full `WorldState` to JSON. Include:
- All hex data (coord, elevation, moisture, temperature, biome, river_flow, terrain_class, tags, settlement ref, road connections)
- All rivers (ordered hex lists)
- All settlements (coord, tier, role, population, name)
- All roads (path, tier)
- Metadata (seed, config params, generation timestamp)

Must be deserializable back to `WorldState` (implement `WorldState.from_json()`).

### Task 5.2 — `export/svg_export.py`

Generate SVG hex map with configurable layer visibility.

Layers (each togglable via config):
- `terrain` — hex fill colored by terrain class or biome
- `elevation` — contour-style shading
- `rivers` — blue strokes along river hex paths, width scaled by flow
- `roads` — brown/grey strokes, line weight by road tier
- `settlements` — icons by tier (circle=village, square=town, star=city)
- `labels` — settlement names
- `grid` — hex outlines

Output: valid SVG file openable in Inkscape for manual GM annotation.

### Task 5.3 — `export/png_export.py`

Rasterize the SVG using Pillow at configurable DPI. Support style presets:
- `atlas` — colored terrain + rivers + settlements + labels
- `topographic` — elevation contours, greyscale
- `wargame` — high contrast, bold hex outlines, minimal color

---

## Phase 6 — CLI & Presets

**Goal:** Single command generates and exports a complete world.

### Task 6.1 — `cli.py`

```bash
# Generate a world
worldgen generate --seed 42 --config presets/temperate_continent.json --output-dir ./output/

# Generate with overrides
worldgen generate --seed 42 --width 256 --height 256 --output-dir ./output/

# Render an existing world
worldgen render --input ./output/world.json --style atlas --output ./output/map.png

# List available presets
worldgen presets
```

On generate, output:
- `world.json` — full serialized WorldState
- `map_terrain.png` — atlas style
- `map.svg` — full SVG with all layers

### Task 6.2 — Presets

Create three JSON preset files:

**`temperate_continent.json`** — Default. Balanced temperature, high moisture, multiple river systems, 6 cities.

**`arid_archipelago.json`** — Low moisture, high sea_level (more ocean), desert/shrubland dominant, coastal settlements, fewer rivers.

**`river_delta.json`** — Flat terrain (low noise scale), very high flow accumulation, dense river network, wetland biome prominent, many small settlements.

---

## Testing Strategy

- Each stage has a corresponding test file.
- Tests run each stage in isolation with a fixed seed and assert structural properties (not exact values, which are seed-dependent).
- Key invariants to test:
  - All rivers reach ocean or a lake (no stranded rivers)
  - No river flows uphill
  - City separation constraints are respected
  - All road paths are connected (no gaps)
  - JSON export/import round-trips losslessly
  - Same seed always produces same output (reproducibility)

Run tests with: `pytest tests/ -v`

---

## Implementation Order

```
Phase 0 (foundation) — must be complete before any stage work
  0.1 dependencies
  0.2 hex.py
  0.3 hex_grid.py  ← test thoroughly here
  0.4 world_state.py
  0.5 pipeline.py
  0.6 config.py
  0.7 debug_viewer.py  ← get this working, use it constantly
  0.8 cli.py skeleton

Phase 1 (terrain) — do not proceed until output looks geographically plausible
  1.1 elevation
  1.2 erosion
  1.3 terrain classification

Phase 2 (hydrology) — do not proceed until rivers reach the sea
  2.1 sink fill + flow accumulation + river extraction

Phase 3 (climate + biomes) — do not proceed until rain shadows are visible
  3.1 climate
  3.2 biomes

Phase 4 (civilization) — do not proceed until habitability scores make intuitive sense
  4.1 habitability
  4.2 settlements
  4.3 roads

Phase 5 (export)
  5.1 JSON
  5.2 SVG
  5.3 PNG

Phase 6 (CLI + presets)
  6.1 CLI
  6.2 presets
```

**At each phase boundary:** Run the debug viewer. Inspect output. Confirm the results make geographic sense before proceeding. A bad elevation stage will propagate problems through every subsequent stage.

---

## Notes for Claude Code

- When implementing noise in `elevation.py`, prefer `opensimplex` over the `noise` library — it has no patent concerns and has a cleaner Python API.
- The erosion stage is the slowest. Implement it on a numpy array (not the hex dict) and only convert back at the end. For a 128×128 map with 15k iterations, target under 30 seconds on CPU.
- When implementing A* in `hex_grid.py`, use a priority queue (`heapq`). The heuristic is hex distance × minimum possible edge cost.
- The SVG export should use `<polygon>` elements for hexes (6 vertices computed from axial coords) and `<polyline>` for rivers and roads.
- All file I/O goes through the export/ layer. Stages never write files.
- If a stage produces unexpected output (e.g. all hexes classified as MOUNTAIN), add a `validate()` method to `WorldState` that checks basic invariants and call it at the end of each stage during development.
