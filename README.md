# Worldgen

Hex-based procedural world generator for TTRPGs, worldbuilding, and wargaming.

- **Scale:** 1 hex = 1 km
- **Reproducible:** any world can be regenerated from a single integer seed
- **Pipelined:** each generation stage is a pure transformer — swap or extend without touching others

## Status

| Phase | Description | Status |
|---|---|---|
| 0 | Foundation — hex grid, pipeline, config | ✓ Complete |
| 1 | Terrain & Elevation — fBm noise, erosion, terrain classification | ✓ Complete |
| 2 | Hydrology — sink filling, flow accumulation, river networks | ✓ Complete |
| 3 | Climate & Biomes — temperature gradient, orographic moisture, Whittaker biomes | ✓ Complete |
| 4 | Settlements & Roads — cities, towns, villages, inter-urban & track roads | ✓ Complete |
| 5 | Export — JSON round-trip, SVG hex map, PNG raster | ✓ Complete |
| 6 | CLI — generate / render / presets commands | ✓ Complete |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
worldgen generate --seed 42
worldgen generate --seed 42 --width 256 --height 256 --output-dir ./my_world
worldgen generate --seed 42 --config path/to/config.json
```

Outputs go to `./output/` by default:

```
output/
├── config.json          # WorldConfig used for this run
├── world.json           # full WorldState (JSON round-trip)
├── elevation.png
├── terrain_class.png
├── river_flow.png
├── temperature.png
├── moisture.png
├── biome.png
├── habitability.png
├── settlements.png
├── roads.png
├── land_cover.png
└── cultivation.png
```

Re-render any attribute from a saved world without re-running the pipeline:

```bash
worldgen render --input output/world.json --attribute biome --output biome.png
```

Available attributes: `elevation`, `terrain_class`, `river_flow`, `temperature`, `moisture`,
`biome`, `habitability`, `settlements`, `roads`, `land_cover`, `cultivation`.

List available presets:

```bash
worldgen presets
```

## Architecture

```
worldgen/
├── core/           # data types and pipeline only — no rendering, no file I/O
│   ├── hex.py          # Hex dataclass, enums (TerrainClass, Biome, LandCover, ...)
│   ├── world_state.py  # WorldState, River, Settlement, Road
│   ├── hex_grid.py     # axial math, neighbors, A*, ring/range queries
│   ├── pipeline.py     # GeneratorPipeline, GeneratorStage base class
│   └── config.py       # WorldConfig — all tunable parameters
├── stages/         # pure transformers: stage.run(WorldState) -> WorldState
│   ├── elevation.py         # fractal Brownian motion + domain warping
│   ├── erosion.py           # particle-based hydraulic erosion
│   ├── terrain_class.py     # ocean / coast / flat / hill / mountain
│   ├── hydrology.py         # Priority-Flood, flow accumulation, river extraction
│   ├── climate.py           # temperature gradient, orographic moisture
│   ├── biomes.py            # Whittaker-style temp × moisture → biome
│   ├── land_cover.py        # land cover classification
│   ├── habitability.py      # composite score for settlement placement
│   ├── city_town.py         # city & town placement
│   ├── interurban_roads.py  # inter-settlement road network
│   ├── cultivation.py       # city/town cultivation rings
│   ├── village_placement.py # village placement
│   ├── village_tracks.py    # village-scale track roads
│   └── village_cultivation.py
├── export/         # all file I/O lives here
│   ├── json_export.py  # WorldState ↔ JSON
│   ├── svg_export.py   # hex map → SVG
│   └── png_export.py   # rasterised map via Pillow
├── render/         # matplotlib debug viewer (never imported by stages)
│   └── debug_viewer.py
└── cli.py
```

**Rules that are never violated:**

- `core/` has no rendering or I/O
- Stages are pure transformers — they receive and return `WorldState`
- All random calls use a seeded `numpy.random.Generator` passed explicitly — no global state
- All tunable parameters live in `WorldConfig` — nothing hardcoded in stage logic

## Generation pipeline

```
ElevationStage → ErosionStage → TerrainClassificationStage → HydrologyStage → …
```

The full pipeline runs 14 stages, continuing through climate, biomes, land cover, habitability,
city/town placement, inter-urban roads, cultivation, village placement, village tracks, and
village cultivation. Each stage receives the full `WorldState` and returns it with new fields
populated. Stages are composed in `GeneratorPipeline`:

```python
from worldgen.core.config import WorldConfig
from worldgen.core.pipeline import GeneratorPipeline
from worldgen.stages.elevation import ElevationStage
from worldgen.stages.erosion import ErosionStage
from worldgen.stages.terrain_class import TerrainClassificationStage
from worldgen.stages.hydrology import HydrologyStage

cfg = WorldConfig(width=128, height=128)
pipeline = GeneratorPipeline(seed=42, config=cfg)
pipeline.add_stage(ElevationStage).add_stage(ErosionStage) \
        .add_stage(TerrainClassificationStage).add_stage(HydrologyStage)
state = pipeline.run()
```

## Configuration

All parameters are in `WorldConfig`. Key knobs:

```python
WorldConfig(
    width=128, height=128,
    sea_level=0.45,              # fraction of hexes below sea
    noise_octaves=6,             # fBm detail levels
    erosion_iterations=15000,    # more = sharper valleys
    river_flow_threshold=0.05,   # top N% of flow accumulation becomes rivers
    base_temperature=0.5,        # 0 = arctic, 1 = tropical
    target_city_count=6,
    target_town_count=24,
    road_mountain_cost=10.0,     # cost multiplier for mountain hexes
)
```

Save / load a config:

```bash
# save the config used for a run
worldgen generate --seed 42 --output-dir ./my_world
# reload it
worldgen generate --seed 42 --config ./my_world/config.json
```

## Development

```bash
python3 -m pytest          # run tests
python3 -m ruff check .    # lint
python3 -m ruff format .   # format
```

Tests assert structural invariants rather than exact values (outputs are seed-dependent). Key invariants checked: rivers reach ocean, no accumulation decrease downstream, river paths are connected, same seed → same output.

Requires Python 3.11+.
