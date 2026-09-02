# Worldgen

Hex-based procedural world generator for TTRPGs, worldbuilding, and wargaming.

- **Scale:** 1 hex = 1 km
- **Reproducible:** any world can be regenerated from a single integer seed
- **Pipelined:** each generation stage is a pure transformer — swap or extend without touching others

For a full reference of every calculation, formula, and config value, see
[docs/REFERENCE.md](docs/REFERENCE.md).

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

Always activate `.venv` before running `worldgen` — the CLI depends on the package being
installed in the active interpreter's environment. If you see
`ModuleNotFoundError: No module named 'worldgen'`, it means `worldgen` resolved to a
different Python (e.g. a global/pyenv interpreter) than the one you installed into. Run
`which worldgen` to check, then either activate `.venv` or re-run `pip install -e ".[dev]"`
in the interpreter you intend to use.

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
├── elevation.svg
├── terrain_class.svg
├── river_flow.svg
├── temperature.svg
├── moisture.svg
├── biome.svg
├── habitability_city.svg
├── habitability_town.svg
├── habitability_village.svg
├── settlements.svg
├── roads.svg
├── land_cover.svg
└── cultivation.svg
```

Re-render any attribute from a saved world without re-running the pipeline:

```bash
worldgen render --input output/world.json --attribute biome --output biome.svg
```

Available attributes: `elevation`, `terrain_class`, `river_flow`, `temperature`, `moisture`,
`biome`, `habitability_city`, `habitability_town`, `habitability_village`,
`settlements`, `roads`, `land_cover`, `cultivation`. `habitability` is accepted as a
shorthand for `habitability_city` — the widest catchment, and the one that decides where
the map's anchor settlements go.

## SVG export

```bash
worldgen export --input output/world.json --output world.svg
worldgen export --input output/world.json --output topo.svg --style topographic
worldgen export --input output/world.json --output wargame.svg --style wargame --hex-size 8
worldgen export --input output/world.json --output custom.svg \
    --color-mode land_cover --layers terrain,rivers,settlements,labels
```

Or via the Python API using `SVGConfig`:

```python
from worldgen.core.pipeline import GeneratorPipeline
from worldgen.core.config import WorldConfig
from worldgen.export.svg_export import save, SVGConfig

state = GeneratorPipeline(seed=42, config=WorldConfig()).run()

# default atlas style — biome colors, all layers
save(state, "world.svg", SVGConfig())

# topographic — elevation colors, terrain + rivers + grid only
save(state, "topo.svg", SVGConfig(style="topographic"))

# wargame — terrain colors, roads + settlements + grid
save(state, "wargame.svg", SVGConfig(style="wargame", hex_size=8.0))

# fully custom
save(
    state,
    "custom.svg",
    SVGConfig(
        color_mode="land_cover",
        layers={"terrain", "rivers", "settlements", "labels"},
        hex_size=16.0,
        padding=30,
    ),
)
```

`SVGConfig` options:

| Option | Default | Values |
|---|---|---|
| `style` | `"atlas"` | `"atlas"`, `"topographic"`, `"wargame"` |
| `color_mode` | `"biome"` | `"biome"`, `"terrain"`, `"land_cover"`, `"elevation"` |
| `layers` | all | any subset of `{"terrain", "rivers", "roads", "settlements", "labels", "grid", "anchorages", "crossings", "legend"}` |
| `hex_size` | `12.0` | pixels per hex |
| `padding` | `20` | border padding in pixels |
| `legend_corner` | `"top-right"` | `"top-right"`, `"bottom-left"` |
| `legend_scale` | `1.0` | legend size as a multiple of `hex_size` |
| `river_min_width` | `0.5` (PNG `1.0`) | width of the thinnest headwater, at the reference 12px hex |
| `river_max_width` | `4.0` | width at full flow, at the reference 12px hex |
| `river_width_steps` | `0` | `0` scales continuously with each hex's flow; a positive value quantises into that many discrete widths |
| `river_width_exponent` | `0.5` | curve applied to flow before it is mapped onto the width range; `1.0` is linear |

`style` is a shortcut that sets `color_mode` and `layers` together: `"topographic"` forces elevation coloring with terrain + rivers + grid; `"wargame"` forces terrain coloring with roads + rivers + settlements + grid + anchorages + crossings — a wargame map is read while moving units, so the features that gate movement matter as much as the roads. Both include the legend. For `"topographic"` and `"wargame"`, the `color_mode` and `layers` values are fixed by the style and any explicitly provided values are ignored. Only `"atlas"` (the default) uses the `color_mode` and `layers` you provide.

### Where settlements go

A site is scored on the land it can actually feed itself from, not on the biome of the
single hex it stands on. The **catchment** is the mean food value of every hex within
reach, so a town ringed by grassland beats one on an identical hex ringed by desert.

Reach depends on tier, so every hex is scored three times — at the city, town and village
cultivation radii (8 / 4 / 2 hexes) — and each tier is placed on its own surface. A
capital is chosen for the province it can draw on; a market town for the fields within
walking distance.

Food value comes from land cover, in four configurable bands: fertile (`OPEN`,
`WOODLAND`), marginal (`SCRUB`, `DENSE_FOREST`), wetland (`BOG`, `MARSH`), and water
(`OPEN_WATER`). Tundra, desert, alpine and bare rock are worth nothing.

**Water is deliberately not zero** — a coastal site fishes. Scoring the sea at nothing
penalised coastal sites twice over: half their catchment counted as waste ground, and the
coastal bonus existed largely to repair the damage. Wetland sits *below* open water,
being neither good fishing nor good ploughing, which matches bog and marsh resisting
cultivation outright.

Fertile and marginal hexes are then scaled by a moisture curve. This is a tent, not a
ramp — too dry is desert, too wet is waterlogged — peaking across the same
`[biome_dry_moist, biome_wet_moist]` band the biome classifier uses, so the two cannot
disagree. Land cover already buckets moisture coarsely, so the curve discriminates
*within* a band: the wet end of grassland is better farmland than the dry end.

On top of the catchment sit four flat site bonuses — river adjacency, coastal access, a
hill overlooking a plain, and a river confluence. These are binary within each term: a hex
with one river neighbour scores the same as one ringed by six, and adjacency is radius 1
only. All nine values live in `WorldConfig`.

### Roads and anchorages

Routes between different settlement pairs share trunk segments, so each map edge is
awarded to the highest road tier that uses it and drawn exactly once, in ascending tier
order — a track branching off a primary road can never paint over the trunk it leaves.

Roads may cross water, but the water leg is not drawn, so a route would otherwise appear
to stop dead at the shore. The `"anchorages"` layer marks the land hex on each side of a
crossing with an anchor symbol, so a route that continues by boat reads as one.

### Roads and rivers

Roads follow river valleys along the **bank**, never down the channel. A road drawn on
the river would hide which side of it the road — and anything standing on that hex — is
on, which matters as soon as you are moving units around the map. Three things enforce
it: the hexsides a river is drawn along are excluded from pathfinding outright, a node
cost (`road_river_hex_cost`) prices out threading a meander or a braid where no drawn
hexside exists to exclude, and the bank discount (`road_bank_discount`) gives routes a
reason to hug the valley from beside it.

Crossing a river stays legal and stays expensive, scaled by flow — a big river costs
roughly a 30-hex detour to cross, a headwater stream far less. Every crossing is tagged
`"ford"`, upgraded to `"bridge"` where a second road crosses the same hex, and the
`"crossings"` layer draws both: a bridge as a twin span with abutments, a ford as the
same span broken, each laid square across the current.

Settlement hexes are exempt from the exclusion, but only far enough to be *reached* — the
hexside opens when the town's counterpart is dry land, never when it is another river
hex. A town on the water must be reachable; it is not a licence to carry on down the
channel a hex at a time.

Where a river mesh seals a city off completely — a delta island, a braided confluence —
there is no bank route out, so the network is joined by a **ferry** instead of a road in
the channel. A ferry is a real link (it counts for connectivity) drawn as a pair of
anchorages, the same symbol used for sea legs. If the gap is wider than
`road_ferry_max_hop`, no plausible ferry exists and generation raises `RoutingError`
rather than quietly producing a compromised map — so a seed that fails is worth
reporting.

### River widths

Rivers are drawn at several discrete widths so flow is readable at a glance and two
rivers can be ranked against each other. Width comes from each hex's own `river_flow`,
not from the river's total, so a river visibly **grows downstream** rather than being
drawn end-to-end at the volume measured at its mouth.

Flow is not mapped onto width linearly. `river_flow` is drainage accumulation over the
basin maximum, and accumulation is roughly power-law distributed — on a typical map the
median river hex sits near `0.02` and the 90th percentile near `0.13`. Mapped straight
onto the width range that spends the whole range on the last few hexes of the single
largest trunk and draws the other nine tenths of the network as one indistinguishable
hairline. `river_width_exponent` (default `0.5`, the square root — also what hydraulic
geometry gives for channel width against discharge) shapes the curve; `1.0` restores the
raw linear mapping and lower values widen small streams further.

By default (`river_width_steps=0`) width then tracks that curve continuously, so a river
tapers smoothly from headwater to mouth. That costs roughly one polyline per hex.

Set a positive `river_width_steps` to quantise into that many discrete widths instead.
Neighbouring segments sharing a width merge into one polyline, so a river costs a handful
of elements rather than one per hex, and the result has the stepped look of a stream-order
map. Banding buckets the curved flow, so it spreads across the bands rather than piling
into the thinnest one. `river_width_steps=1` draws every river uniformly at
`river_max_width`.

### Line widths and hex size

River and road widths — and a track's dash pattern — are written for the default **12px
hex** and scale linearly from there, the same reference the settlement, anchorage and
crossing symbols use. Exporting at `hex_size=28` doubles-and-then-some every line rather
than leaving hairline rivers between huge hexes; exporting at 6 thins them to match.
Legend glyphs scale with `legend_scale` on the same basis, so the key always matches the
map it describes.

### Legend

The `"legend"` layer draws a key for whatever the map actually contains — fill categories
for the active `color_mode`, plus rows for rivers, each road tier present, anchorages
where any road meets water, fords and bridges where they exist, and each settlement tier
present. Symbols are drawn with the
same code as the map itself, so they always match.

Because the axial-to-pixel transform shears the grid into a parallelogram, a rectangular
hex map leaves large empty triangles at the top-right and bottom-left of the canvas. The
legend is placed flush against the map's edge inside one of them (`legend_corner`), so it
never covers terrain. On maps too narrow for it to fit, the panel is clamped into the
corner and its opaque backing keeps it readable. The legend scales with `hex_size`; bump
`legend_scale` to enlarge it on big exports.

`PNGConfig` takes the same `legend_corner` and `legend_scale` options, and also has
`"legend"` in its default layers. Row selection and panel placement are shared between the
two exporters in `worldgen/export/legend.py`; each one only draws the rows.

## Presets

Presets are JSON files that override any subset of `WorldConfig` fields. Place them in a `presets/` directory and load with `--config`:

```bash
worldgen presets                                      # list available presets
worldgen generate --seed 42 --config presets/island.json
```

No built-in presets ship with the project — create your own. Any field omitted in the JSON falls back to its `WorldConfig` default:

```json
{
    "width": 96,
    "height": 96,
    "continent_falloff": 0.8,
    "sea_level": 0.60,
    "base_temperature": 0.75,
    "target_city_count": 3,
    "target_town_count": 12
}
```

Key fields to customize per world type:

| Field | Effect |
|---|---|
| `sea_level` | fraction of hexes below sea (0.3 = lots of land, 0.7 = archipelago) |
| `continent_falloff` | edge-falloff strength — higher = more island-shaped |
| `base_temperature` | 0 = arctic, 1 = tropical |
| `noise_octaves` | fBm detail levels — more = rougher terrain |
| `erosion_iterations` | more = sharper valleys |
| `target_city_count` / `target_town_count` | settlement density |

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
│   ├── habitability.py      # per-tier catchment score for settlement placement
│   ├── city_town.py         # city & town placement
│   ├── interurban_roads.py  # inter-settlement road network
│   ├── cultivation.py       # city/town and village cultivation rings
│   ├── village_placement.py # village placement
│   └── village_tracks.py    # village-scale track roads
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
pipeline.add_stage(ElevationStage).add_stage(ErosionStage).add_stage(
    TerrainClassificationStage
).add_stage(HydrologyStage)
state = pipeline.run()
```

## Configuration

All parameters are in `WorldConfig`. Key knobs:

```python
WorldConfig(
    width=128,
    height=128,
    sea_level=0.45,  # fraction of hexes below sea
    noise_octaves=6,  # fBm detail levels
    erosion_iterations=15000,  # more = sharper valleys
    river_flow_threshold=0.05,  # top N% of flow accumulation becomes rivers
    base_temperature=0.5,  # 0 = arctic, 1 = tropical
    target_city_count=6,
    target_town_count=24,
    road_mountain_cost=10.0,  # cost multiplier for mountain hexes
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
