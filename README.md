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
| 3 | Climate & Biomes | Planned |
| 4 | Settlements & Roads | Planned |
| 5 | Export (JSON, SVG, PNG) | Planned |
| 6 | CLI & Presets | Planned |

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
├── config.json       # full WorldConfig used for this run
├── elevation.png     # debug view: raw elevation
├── terrain_class.png # debug view: ocean / coast / flat / hill / mountain
└── river_flow.png    # debug view: river network and flow volume
```

## Architecture

```
worldgen/
├── core/           # data types and pipeline only — no rendering, no file I/O
│   ├── hex.py          # Hex dataclass, enums (TerrainClass, Biome, ...)
│   ├── world_state.py  # WorldState, River, Settlement, Road
│   ├── hex_grid.py     # axial math, neighbors, A*, ring/range queries
│   ├── pipeline.py     # GeneratorPipeline, GeneratorStage base class
│   └── config.py       # WorldConfig — all tunable parameters
├── stages/         # pure transformers: stage.run(WorldState) -> WorldState
│   ├── elevation.py    # fractal Brownian motion + domain warping
│   ├── erosion.py      # particle-based hydraulic erosion
│   ├── terrain_class.py
│   └── hydrology.py    # Priority-Flood, flow accumulation, river extraction
├── export/         # all file I/O lives here
├── render/         # matplotlib debug viewer (never imported by stages)
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

Each stage receives the full `WorldState` and returns it with new fields populated. Stages are composed in `GeneratorPipeline`:

```python
from worldgen.core.config import WorldConfig
from worldgen.core.pipeline import GeneratorPipeline
from worldgen.stages.elevation import ElevationStage
from worldgen.stages.erosion import ErosionStage
from worldgen.stages.terrain_class import TerrainClassificationStage
from worldgen.stages.hydrology import HydrologyStage

cfg = WorldConfig(width=128, height=128, seed=42)
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
