# Worldgen — Technical Debt Audit

**Date:** 2026-05-13  
**Scope:** All source under `worldgen/`, tests, CI, and dependencies  
**Scoring:** Priority = (Impact + Risk) × (6 − Effort), 1–5 scale each

---

## Summary

| # | Item | Category | Priority | Effort |
|---|------|----------|----------|--------|
| 1 | Missing tests for 6 stage files | Test | **36** | S |
| 2 | `settlements.py` is dead code | Code | **35** | S |
| 3 | CI coverage check silently passes | Test | **30** | XS |
| 4 | `RoadStage` ≈ `InterurbanRoadStage` (mass duplication) | Code | **21** | M |
| 5 | `networkx` listed as dependency but never imported | Dependency | **20** | XS |
| 6 | `_assign_role` defined in three places | Code | **20** | S |
| 7 | Magic numbers in `settlements.py` not in `WorldConfig` | Code | **16** | S |
| 8 | `WorldState.from_json` violates layer rule | Architecture | **16** | S |
| 9 | `WorldConfig.__post_init__` references field before declaration | Code | **15** | XS |
| 10 | Pipeline assembly buried inside CLI command | Architecture | **15** | M |
| 11 | Missing `presets/` directory — documented feature is broken | Docs | **12** | S |
| 12 | `_get_lake_components` dead code in `hydrology.py` | Code | **10** | XS |
| 13 | `hydrology.py` is 781 lines — prime split candidate | Code | **9** | M |
| 14 | Bare `dict`/`list` type annotations throughout stages | Code | **8** | S |
| 15 | Rivers stay a list of paths while roads became a graph | Architecture | **8** | S |
| 16 | Moisture smear blends the ocean's carrier value into coastal rainfall | Model | **12** | M |
| 17 | `_assign_role` compares metre elevation against 0.70 — FORTRESS unreachable | Code | **12** | S |
| 18 | Market siting scores a plain hex disc, not the day-reach the catchment walks | Model | **20** | L |
| 19 | One off-map river strips the floodplain off half the map's channels | Model | **28** | M |
| 20 | An off-map river imports discharge but no sediment, so it cuts where it should build | Model | **18** | M |

---

## Item Details

### 1 — Missing tests for 6 stage files
**Category:** Test debt | **Priority: 36** | **Effort: S (1–2 days)**

Six stages have no corresponding test file at all:

- `stages/city_town.py` — city and town placement
- `stages/interurban_roads.py` — primary/secondary road network
- `stages/village_placement.py` — village placement
- `stages/village_tracks.py` — village-to-road tracks
- `stages/terrain_class.py` — terrain classification
- `stages/erosion.py` — hydraulic erosion

These are not trivial helpers — they form the backbone of the settlement and road subsystem. Per `CLAUDE.md`, every stage must have a corresponding test file asserting structural invariants (not exact values). The CI `cov-fail-under=50` floor is also silently bypassed (see item 3), so this gap is currently invisible to CI.

**Fix:** Write structural invariant tests for each stage following the existing pattern in `test_hydrology.py` and `test_roads.py`. Minimum invariants: stage output has correct type, no exceptions on a small (32×32) world, key properties (e.g., city separation respected, all tracks reach a road node) hold.

---

### 2 — `settlements.py` is dead code
**Category:** Code debt | **Priority: 35** | **Effort: S (half-day)**

`worldgen/stages/settlements.py` contains `SettlementStage`, a monolithic stage that places cities, towns, and villages in one pass. It is never imported by the CLI or any other module. The current pipeline uses `CityTownStage` (from `city_town.py`) for cities and towns, and `VillagePlacementStage` (from `village_placement.py`) for villages.

The old file still has its own copy of `_assign_role` (identical to the one in `city_town.py`) and its own village placement logic that diverges from `VillagePlacementStage`. Any contributor reading the codebase would be confused about which stage to modify when changing settlement behavior.

**Fix:** Delete `worldgen/stages/settlements.py`. Verify no test or import references it (a quick grep confirms none do outside the file's own tests, if any exist). Add a note to `CLAUDE.md` clarifying that `CityTownStage` + `VillagePlacementStage` is the current split.

---

### 3 — CI coverage check silently passes regardless of result
**Category:** Test debt | **Priority: 30** | **Effort: XS (5 minutes)**

In `.github/workflows/ci.yml`:

```yaml
- name: Check test coverage
  run: python -m pytest --cov=worldgen --cov-fail-under=50 --quiet || true
```

The `|| true` makes this step always succeed. The 50% coverage floor is effectively unenforced — coverage can drop to 0% and CI stays green. Combined with item 1 (six untested stages), the actual coverage is likely well below 50%.

**Fix:** Remove `|| true`. Raise the floor to 70% once items 1 and 2 are addressed. Until then, at minimum remove the `|| true` so coverage regressions are visible.

---

### 4 — `RoadStage` and `InterurbanRoadStage` are near-identical
**Category:** Code debt | **Priority: 21** | **Effort: M (2–3 days)**

`roads.py` (367 lines) and `interurban_roads.py` (276 lines) share the following logic verbatim or with only trivial differences:

- `node_cost` / `edge_cost` closures (identical)
- Traffic accumulation loop structure (identical)
- Eligible-hex filtering with `road_river_traffic_min` (identical)
- Tier cut-point computation (identical except `RoadStage` adds TRACK tier)
- `_stitch_via_junction` helper (structurally identical, minor signature difference)
- `_path_min_tier` helper (identical)
- Road-connections population loop (identical)
- `tag_river_crossings` call (identical)
- Habitability re-score loop (identical)
- `_guarantee_city_connectivity` (nearly identical — `InterurbanRoadStage` fixes a subtle BFS adjacency bug that `RoadStage` doesn't have, meaning bugs fixed in one won't be fixed in the other)

The only structural differences are: `InterurbanRoadStage` excludes villages from the traveller pool and does not emit TRACK-tier roads; `RoadStage` additionally promotes villages near high-habitability roads and has a village promotion block.

**Fix:** Extract a `_RoadBase` mixin or a standalone `run_road_simulation(settlements, hexes, cfg, rng, *, include_tracks, include_villages)` function in a new `stages/_road_shared.py` module. Both stages call this. This eliminates ~180 lines of duplication and ensures bug fixes reach both stages.

---

### 5 — `networkx` listed as a hard dependency but never imported
**Category:** Dependency debt | **Priority: 20** | **Effort: XS (5 minutes)**

`pyproject.toml` lists `networkx>=3.2` as a runtime dependency. Searching all Python files under `worldgen/` finds zero imports of `networkx`. It was likely a placeholder from the initial plan (the `worldgen_plan.md` mentions it as a candidate for road pathfinding). All graph algorithms are implemented directly in `hex_grid.py` (A*, BFS, topological sort).

A dead dependency adds install weight, increases the attack surface for supply-chain issues, and creates confusion about what the library actually uses.

**Fix:** Remove `networkx>=3.2` from `[project] dependencies` in `pyproject.toml`. If networkx support is planned for a future phase, document it as a future optional dependency in a comment.

---

### 6 — `_assign_role` exists in three incompatible versions
**Category:** Code debt | **Priority: 20** | **Effort: S (half-day)**

Settlement role assignment logic lives in three places:

1. `stages/settlements.py` — `_assign_role(coord, hx, hexes)` (dead code per item 2, but still in the repo)
2. `stages/city_town.py` — `_assign_role(coord, hx, hexes)` (identical to #1; used by `CityTownStage` and imported by `VillagePlacementStage`)
3. `stages/roads.py` — `RoadStage._assign_role_simple(self, coord, hx, hexes)` (simplified private method used only for village promotions during road building; differs from #2 in that it skips the `hex_range` fertile biome check)

The `_assign_role_simple` vs `_assign_role` split means promoted villages can receive different roles than villages placed initially, producing inconsistent results depending on the code path.

**Fix:** Consolidate into a single `assign_settlement_role(coord, hx, hexes)` function in `road_cost.py` or a new `stages/_settlement_shared.py`. Delete the duplicate in `settlements.py` (as part of item 2). Have `RoadStage` call the shared function.

---

### 7 — Magic numbers in `settlements.py` not in `WorldConfig`
**Category:** Code debt | **Priority: 16** | **Effort: S (1 day)**

Several tunable thresholds are hardcoded in `stages/city_town.py` and `stages/settlements.py` rather than exposed through `WorldConfig`:

| Value | Location | Meaning |
|-------|----------|---------|
| `0.70` | `city_town.py:21` | Mining role elevation threshold |
| `30` | `city_town.py:89` | City influence shadow radius (hexes) |
| `0.5` | `city_town.py:92` | Shadow multiplier |
| `0.3` | `settlements.py:138` | Village habitability minimum |
| `3` | `settlements.py/city_town.py` | Village minimum separation |

`CLAUDE.md` explicitly states: "All tunable parameters live in `WorldConfig`; nothing hardcoded in stage logic." These violate that rule.

**Fix:** Add fields `city_shadow_radius`, `city_shadow_mult`, `mining_elevation_threshold`, `village_min_habitability`, and `village_min_separation` to `WorldConfig` with the current values as defaults. Update `default_config.yaml` accordingly.

---

### 8 — `WorldState.from_json` imports from `export/` in violation of layer rules
**Category:** Architecture debt | **Priority: 16** | **Effort: S (half-day)**

```python
# worldgen/core/world_state.py
@classmethod
def from_json(cls, path: str) -> "WorldState":
    from worldgen.export.json_export import load

    return load(path)
```

`core/` must not import from `export/` — that's an explicit architecture constraint in `CLAUDE.md` ("stages never write files"; by extension, `core/` data types should not depend on `export/`). This creates a latent circular import risk: if `json_export.py` ever needs to import any utility from `core/world_state.py` beyond `WorldState` itself, a circular dependency forms.

**Fix:** Remove `WorldState.from_json`. Callers (there is one in `cli.py`) should import `load` from `worldgen.export.json_export` directly. Update `cli.py` and any tests that call `WorldState.from_json`.

---

### 8a — `ImageElevationStage` imports a reader from `export/`
**Category:** Architecture debt | **Priority: 8** | **Effort: S (half-day)**

```python
# worldgen/stages/image_elevation.py, inside run()
from ..export.heightmap_import import load_luminance
```

The same tension as item 8, from the other side. `CLAUDE.md` says `stages/` are pure transformers and that all file I/O lives in `export/`. The `Image.open` does live in `export/`, which is the half of the rule that matters, but a stage now triggers a file read where every other stage is a pure function of `WorldState`.

The alternative was rejected on balance, and the reasoning is worth keeping: `GeneratorPipeline` registers stage *classes* and discards the `stage_config` dict `add_stage` accepts, so injecting a pre-loaded array would mean reviving that parameter, giving two sources of truth (config names the file, the injected array holds the pixels) and forcing every pipeline assembler — CLI, `import-heightmap`, `tests/worlds.py`, any notebook — to remember to pre-load. Stashing the array on `state.metadata` is worse still, since that dict is serialised verbatim into `world.json`.

One concrete cost today: `worldgen/export/__init__.py` eagerly imports `png_export` and `svg_export`, so this pulls matplotlib into any programmatic pipeline that uses the stage. The import is inside `run()`, so `worldgen --help` is unaffected.

**Fix (if it becomes a problem):** make `export/__init__.py` lazy, which removes the matplotlib cost and leaves only the layering question. Only revive `stage_config` if a second stage needs injected data too; otherwise the honest cleanup is to *delete* the dead parameter.

---

### 9 — `WorldConfig.__post_init__` references `wind_direction` before it is declared
**Category:** Code debt | **Priority: 15** | **Effort: XS (5 minutes)**

`config.py` line 43 reads `self.wind_direction = _coerce_pair(...)` inside `__post_init__`, but `wind_direction` is not declared as a dataclass field until line 115 (after the `# Climate` comment block). This works at runtime because Python's dataclass machinery sets all fields before calling `__post_init__`, but it is semantically misleading and fragile: any developer moving `wind_direction` above `__post_init__` in a refactor would break the validation ordering silently.

The same pattern exists for `elevation_gradient` (declared at line 19, used at line 44 — this one is fine; only `wind_direction` comes after `__post_init__`).

**Fix:** Move `wind_direction` (and its related climate fields) above `__post_init__`. Group fields logically: dimensions → elevation → terrain → erosion → hydrology → **climate** (including `wind_direction`) → biome → settlements → roads.

---

### 10 — Pipeline assembly is buried inside the CLI `generate` command
**Category:** Architecture debt | **Priority: 15** | **Effort: M (1–2 days)**

The canonical stage sequence is defined only inside the `generate()` function in `cli.py`, using 15 deferred imports:

```python
from .stages.biomes import BiomeStage
from .stages.city_town import CityTownStage

# ... 13 more
pipeline.add_stage(ElevationStage)
# ... 14 more add_stage calls
```

This means the default pipeline cannot be instantiated without going through the CLI. There is no public `build_default_pipeline(seed, config)` function that other consumers (notebooks, scripts, other CLIs) can call. It also makes it harder to write integration tests that exercise the full pipeline without invoking Click.

**Fix:** Extract a `build_pipeline(seed: int, config: WorldConfig) -> GeneratorPipeline` factory function, either in `cli.py` or a new `worldgen/pipeline_factory.py`. The `generate` command delegates to it. Tests can call it directly.

---

### 11 — `presets/` directory does not exist; `worldgen presets` silently shows nothing
**Category:** Documentation debt | **Priority: 12** | **Effort: S (half-day)**

The `worldgen_plan.md` specifies three preset files (`temperate_continent.json`, `arid_archipelago.json`, `river_delta.json`). The `cli.py presets` command exists and searches for them. The README documents their use. But the `presets/` directory was never created, so running `worldgen presets` outputs `No presets found` rather than an error, silently misleading users.

**Fix:** Create `presets/` and add the three planned presets with representative `WorldConfig` values. Alternatively, if presets are not yet ready, update the README and CLI help to note they are forthcoming, and make the command return a non-zero exit code when the directory is missing.

---

### 12 — `_get_lake_components` is dead code in `hydrology.py`
**Category:** Code debt | **Priority: 10** | **Effort: XS (2 minutes)**

`hydrology.py` defines a module-level function `_get_lake_components(lakes, hexes)` at line ~750. It is never called anywhere in the codebase. The `_ensure_lake_drainage` method implements its own inline `bfs_component` closure instead. The function is a leftover from an earlier iteration.

**Fix:** Delete `_get_lake_components`. Add a test to catch future accidental re-introduction of dead module-level functions if desired.

---

### 13 — `hydrology.py` is 781 lines and a candidate for splitting
**Category:** Code debt | **Priority: 9** | **Effort: M (2 days)**

`hydrology.py` handles five distinct algorithms in one file: Priority-Flood sink filling, epsilon-tilt BFS, flow direction, flow accumulation (Kahn's topological sort), river tracing (including three fallback strategies), lake drainage/expansion, and confluence splitting. It is the largest file in the project by a factor of 2×.

This is lower priority than the items above because the code is correct and well-commented. However, it is difficult to navigate and unit-test individual algorithms in isolation.

**Fix (optional / Phase 7):** Split into `hydrology_fill.py` (Priority-Flood + flow direction + accumulation) and `hydrology_rivers.py` (river tracing + lake drainage + confluence split). The public `HydrologyStage` can import from both.

---

### 14 — Bare `dict`/`list` type annotations throughout stage code
**Category:** Code debt | **Priority: 8** | **Effort: S (1 day)**

Multiple stage files use unparameterised type annotations:

```python
hex_traffic: dict = defaultdict(float)  # should be dict[HexCoord, float]
canonical_routes: dict = {}  # should be dict[tuple, list[HexCoord]]
hex_tier: dict = {}  # should be dict[HexCoord, RoadTier]
metadata: dict = field(default_factory=dict)  # should be dict[str, Any]
```

These are legal Python 3.11 but suppress type-checker warnings that would catch bugs (e.g., passing a `HexCoord` key where a `str` is expected).

**Fix:** Annotate all unparameterised `dict` and `list` fields with their full generic types. Enable `ruff`'s `ANN` rules at warning level to catch new occurrences.

---

## Phased Remediation Plan

These items are sized to run alongside feature work, not as a separate rewrite sprint.

### Phase A — Quick wins (1 day total, zero risk)
Items that are safe one-liner or near-one-liner changes with immediate CI benefit:

1. Remove `|| true` from CI coverage check (**item 3**)
2. Remove `networkx` from `pyproject.toml` (**item 5**)
3. Delete `_get_lake_components` from `hydrology.py` (**item 12**)
4. Move `wind_direction` field above `__post_init__` in `config.py` (**item 9**)

### Phase B — Test gap closure (2–3 days)
Address the most critical correctness risk before any feature work:

5. Write tests for `city_town`, `terrain_class`, `erosion` (**item 1**, first half)
6. Write tests for `interurban_roads`, `village_placement`, `village_tracks` (**item 1**, second half)
7. Raise CI `cov-fail-under` to 65%

### Phase C — Dead code and duplication cleanup (2–3 days)
Eliminate confusion and prevent divergence:

8. Delete `settlements.py` (**item 2**)
9. Consolidate `_assign_role` into a shared module (**item 6**)
10. Remove `WorldState.from_json` layer violation (**item 8**)

### Phase D — Config and architecture hygiene (2 days)
Harden the system for future extension:

11. Move magic numbers into `WorldConfig` (**item 7**)
12. Extract `build_pipeline()` factory from CLI (**item 10**)
13. Create three preset files in `presets/` (**item 11**)
14. Parameterise `dict`/`list` annotations (**item 14**)

### Phase E — Road stage refactor (3 days, warrants a branch)
Highest-effort structural change — do last when Phase B tests provide a safety net:

15. Extract `_road_shared.py` and unify `RoadStage` + `InterurbanRoadStage` (**item 4**)

### Phase F — Optional, low priority
16. Split `hydrology.py` into two modules (**item 13**) — only if hydrology needs active development

---

### 15 — Rivers stay a list of paths while roads became a graph
**Category:** Architecture | **Priority: 8** | **Effort: S**

`WorldState.road_edges` is now `{edge: RoadEdge}` — one tier and one delta elevation per
undirected edge. `WorldState.rivers` is still `list[River]`, each a whole path with a
`flow_volume`. The asymmetry is deliberate and this entry exists so nobody "fixes" it by
mistake, but two things in it are worth revisiting.

**Why rivers were left alone.** The redundancy that made the road conversion worth doing
is not there. Measured at 128×128: 98 rivers, 970 hex entries over 872 distinct edges —
**1.11 entries per edge, against 89 for roads before the change** — and *no* edge shared
between two rivers, because a tributary terminates at its confluence rather than
continuing down the trunk. A drainage network is a tree; road journeys shared trunks
almost entirely. A river is also genuinely a path — source, mouth, direction, one day a
name — where a `Road` was only a journey someone happened to make.

**The actual debt is two representations of flow.** `River.flow_volume` sits alongside
`Hex.river_flow`, at different granularities, with nothing keeping them in step. The
stages all read the per-hex value; `flow_volume` is written by hydrology and read by the
renderers only. They have not drifted, and there is no test that would notice if they did.

**And one asymmetry that has been checked and is fine.** `river_edges()` derives the
hexsides a road may not travel from the *paths*, while `is_river()` and every road cost
term read the `"river"` *tag*. If those disagreed a road could be blocked from a hexside
whose hexes it does not consider river at all. Measured: 46 hexes sit on a river path
untagged, and every one is the discharge — 33 LAKE, 13 OCEAN. No hex carries the tag
without being on a path. The two agree exactly on land.

**Do this if** rivers gain per-segment attributes the way roads did (navigability by
tonnage, a ford's difficulty, a named reach), at which point an edge map earns its place.
Until then the path form carries information the graph would lose.


### 16 — Moisture smear blends the ocean's carrier value into coastal rainfall
**Category:** Model | **Priority: 12** | **Effort: M**

Ocean and lake hexes hold `moisture = 1.0` when the Gaussian smear runs — the
*carrier* value the orographic sweep transports, not rainfall — and the smear blends
it into every coastal land hex, windward and leeward alike. A lee shore draws rain
from the sea behind it, which is the effect the orographic pass exists to prevent.

**Fixed once, and reverted deliberately.** A land-only normalised convolution
(smear `arr * land_mask` against a smeared mask) removes the artifact cleanly — and
`test_each_climate_comes_out_as_the_country_it_is_named_after` immediately fails,
because the artifact is quietly load-bearing: blending 1.0 into the coasts and then
rescaling the land mean *widens* each climate's rainfall distribution, and the soil
rainfall bands were calibrated against those widened distributions. Remove it and a
mediterranean map's rainfall clusters inside the [`biome_dry_precip_mm`,
`biome_wet_precip_mm`] arable band: 46% arable against temperate's 34%, and the
"mediterranean comes out pastoral" acceptance claim inverts.

**The real fix is a decision, not a patch:** what should make a 550 mm climate
pastoral once the artifact is gone? Historically it is summer drought — seasonality
the model does not represent — so either the rainfall bands get per-climate
recalibration, or soil gains a seasonality term, and both change the acceptance
table. The one-line convolution fix is in the review record, ready once that call
is made.

### 17 — `_assign_role` compares metre elevation against 0.70
**Category:** Code | **Priority: 12** | **Effort: S**

`city_town.py` still reads the retired [0, 1] elevation axis: any steep neighbour
above 0.70 *metres* makes a settlement MINING, so FORTRESS is unreachable and — with
the water tests generous — 73 of 74 organic settlements come out `port`. Deferred in
the PR that introduced the metres axis ("needs a decision about what a fortress and
a mine are"); recorded here so the deferral has an address.


### 18 — Market siting scores a plain hex disc, not the day-reach the catchment walks
**Category:** Model | **Priority: 20** | **Effort: L**

`MarketStage._plant` ranks candidates by integrating surplus over plain hex rings out to
`market_day_radius`: no travel cost, no water barrier, off-map as zero. The catchment the
winner then receives is a cost-bounded Dijkstra with watershed edges. So the day radius
means a *ring count* in the siting and a *cost budget* in the gather, a ridge beside a
candidate does not lower its score, and markets are ranked on countryside the built
catchment then fails to deliver. Six independent verifier runs in the PR #34 review
confirmed the mechanism (planting score ≈ 4× the real gather, its land term as
disconnected from the draw as its water term).

**Fixed once, calibrated twice, and reverted deliberately.** The rewrite exists and
works: score over a cached single-source day-reach Dijkstra plus the fishery rim on the
same terms `fishery_rim` grants it, deplete exactly what was scored, seed the lazy-greedy
heap with the ring disc as a provable upper bound (one ring slack for the rim) so
exactness survives and the Dijkstra only runs on candidates that pop. What it cannot do
is inherit the disc's calibration:

- The floor's units change (score ≈ real gather, not 4× it), and no single value threads
  the acceptance table. 12.0 reproduces disc-era density (15 markets on the 64×64
  temperate reference against 13) and keeps count monotone in measured food across all
  five climates — and then the tiers above and below break: a city stands away from
  navigable water, a landlocked arid map grows a city of 8,754, both 96×96 chokepoint
  fixtures grow zero villages, and every market's population moves across promotion
  because every market sits in some city's shadow.
- The deeper reason: cost-bounded scoring reads *local* concentration where the disc
  read regional total. Spread-thin fertility (taiga, leached tropics) scores lower and
  concentrated fertility (desert rivers, coasts with a fishery rim) scores higher, so
  the whole settlement economy — floor, `city_min_draw`, chokepoint gates, the rural
  share — needs recalibrating together, against the acceptance table the author wants,
  not one test at a time.

Sweep data from the review record (64×64, seed 42, island geometry, markets/medians):
floor 8 → arid 15/845, boreal 17/812, tropical 18/857, med 24/1020, temperate 30/930 (no
ordering violations, ~2× disc density); floor 12 → 6/6/7/14/15 (no violations, disc
density); floors 10, 14, 16 each break count-follows-food between boreal and tropical.

**Do this as its own branch**, with the acceptance table on the desk: pick the density,
re-tune `city_min_draw` and the chokepoint gates against it, and update the PR-table
numbers in the same change. The diff is small; the decision is not.


### 19 — One off-map river strips the floodplain off half the map's channels
**Category:** Model | **Priority: 21** | **Effort: M**

Two problems in the same field, found while merging the ported water model into
`feat/alluvium`. Both are invisible to the test suite, which exercises alluvium only at
48×48 — the size at which the first one does not yet bite.

**The field's magnitude falls away as the map grows.** Mean alluvium on a hex beside a
river, by world (`_distance_to_river` profile, seed 42):

| world | at the river | 1 | 2 | 3 | 4 | hexes above 0.5 |
|---|---|---|---|---|---|---|
| pre-merge 48×48 | 0.339 | 0.250 | 0.131 | 0.052 | 0.025 | 45 |
| pre-merge 96×96 | 0.331 | 0.259 | 0.156 | 0.082 | 0.044 | 363 |
| **post-merge 48×48** | 0.366 | 0.365 | 0.257 | 0.183 | 0.127 | 83 |
| **post-merge 96×96** | 0.123 | 0.078 | 0.037 | 0.015 | 0.009 | 71 |
| **post-merge 96×96, organic, south falloff** | 0.069 | 0.051 | 0.026 | 0.016 | 0.017 | 121 |

Before the merge the profile was flat in map size — 0.339 against 0.331, and the count of
deep hexes scaled with the area, which is what it should do. After it, 48×48 is if anything
stronger while 96×96 has fallen by a factor of three and the deep-hex count has stopped
scaling. The shape of the profile survives; the depth does not, and `food_alluvium_bonus`
multiplies the depth.

On the 96×96 organic world the claim `test_alluvium_sits_on_gentle_ground` makes has gone
hollow with it: ground carrying deep alluvium averages **82 m/km** against **96** for the
rest of the land. That still satisfies the assertion, which only asks that one be less than
the other, but 82 m/km is an 8% grade and silt does not sit on it. At 48×48 the same figures
are 39 against 123, which is the real claim.

**The cause, measured.** Not the quantile — the droplet term is stable across sizes
(normalised mean 0.0090 at 48×48 against 0.0080 at 96×96). It is the meander term, which
falls from a mean of 0.0997 to 0.0196 and from covering 18.6% of land to 4.1%.

`_widen_valleys` sizes every channel's belt as a fraction of the largest flow **on the
map**, and disqualifies a channel outright if the result is under one cell:

```python
reach = width_max * (flow[i, j] / max_flow) ** width_exponent
if reach < 1.0:
    continue          # no belt at all
```

At the shipped `width_max = 6.0` and `width_exponent = 0.6` a channel needs about 5.25% of
`max_flow` to get any belt. Off-map inflows seed a catchment of `river_inflow_volume` ×
land area — 0.15 of it — which is far more than any river the map raises for itself, so
one imported river sets `max_flow` for everything:

| world | `max_flow` | channels | of those, with a belt |
|---|---|---|---|
| 48×48, inflows on | 53.0 | 26 | 26 (100%) |
| 48×48, inflows off | 53.0 | 26 | 26 (100%) |
| **96×96, inflows on** | **1142.5** | 131 | **68 (52%)** |
| 96×96, inflows off | 174.0 | 131 | 131 (100%) |

At 48×48 the default sea ring leaves no border land, so no inlet is admitted and the two
columns agree — which is exactly why every alluvium fixture passes. At 96×96 the inlet
takes, `max_flow` goes up 6.6×, and **half the map's channels lose their floodplain
entirely**. Alluvium beside a river: 0.320 with inflows off, 0.123 with them on.

The terrain effect is a redistribution rather than a loss — one imported river gets a very
wide valley and the rest get none, so the mean slope profile barely moves (49.7 m/km at the
channel with inflows, 57.4 without). It is the *count* of floored cells that collapses, and
that is what the alluvium record is made of.

**This is the same defect the branch already fixed once, in the other term.** Belt *depth*
used to be scaled globally and was changed to scale against each channel's own reach,
because "`(flow/max_flow)**0.6` is tiny for anything but the trunk river, so every other
valley read as bare and the map showed one bright ribbon". Belt *width* still scales
globally, and now has a trunk river imported from off the map to be tiny against.

**The fix is a calibration decision, not a patch.** Scaling `reach` against a high quantile
of channel flow rather than the maximum is the smallest change and matches what
`_normalise_alluvium` already does for the same reason — but it widens valleys on every map
at every size, which moves the soil, the food and therefore the settlement economy. Excluding
the seeded catchment from `max_flow` is narrower and leaves the imported river's own valley
too small. Either way, measure at two sizes and put the acceptance table on the desk.

### The second problem: two mechanisms, one floodplain

`SoilQuality.PRIME` is documented as "alluvium: the floodplain of a river too big to wade"
and derived by `SoilStage.is_alluvium` from slope and catchment — a **rule** about where
silt ought to be. `Hex.alluvium` is a **measurement** of where the erosion model actually
put it. Both now feed `potential_food`, the first by choosing the soil class and the second
by multiplying it.

They identify almost disjoint ground (96×96 organic, seed 42, 7,757 land hexes):

- `PRIME`: 143 hexes, median **1** hop from a river, mean slope 17 m/km, 22% on the coast.
- measured alluvium > 0.5: 121 hexes, median **3** hops from a river, mean slope 82 m/km,
  8% on the coast.
- In both: **11 hexes** — 8% of `PRIME`, 9% of the measured set.

So the double pricing is real but small: `PRIME` scores 1.05× its configured base rather
than 1.00×, and the `PRIME`/`ARABLE` ratio comes out 1.45 against the 1.40 the settings ask
for. Not worth fixing on its own.

The disagreement is the interesting part. The rule finds riverside flats; the measurement
finds deltas and meander belts three hops out. `test_alluvium_falls_away_from_the_rivers`
checks that the two networks agree *statistically* — the profile is monotone — which is a
much weaker claim than agreeing per hex, and the numbers above are what that gap looks
like.

**The coherent end state is probably that the measurement replaces the rule inside
`is_alluvium`**: the erosion model knows where sediment went, and a rule inferring it from
slope and catchment is a second, worse answer to a question already answered. That would
retire the double count, remove a threshold, and make `PRIME` mean something measured. It
also moves the soil map and therefore the whole settlement economy, so it wants the
acceptance table on the desk — see item 18, which is the same kind of change and says the
same thing. **Fix the size dependence first**, or the measurement is not yet fit to be
promoted over the rule.


### 20 — An off-map river imports discharge but no sediment
**Category:** Model | **Priority: 18** | **Effort: M**

An inlet is seeded with a catchment it never earned on this map, and that imported
discharge is read everywhere it matters: `navigable` floats a boat on it, `_widen_valleys`
sizes a belt from it, `catchment_km2` records it. What it is *not* given is a load. Every
droplet in `_drop_particle` starts `sediment = 0.0` and `water = 1.0`, and droplets are
seeded at uniformly random land cells, so nothing arrives at an inlet carrying anything.

An imported river therefore has the discharge of a great river and the sediment budget of
whichever single hex it happened to enter on. Measured on a 96×96 map, seed 11, with two
inlets admitted (`continent_falloff_edges` dropping the north):

| | hexes | max catchment | droplet deposition, mean | meander term, mean |
|---|---|---|---|---|
| imported rivers | 51 | 1,528 km² | **−0.199** | 0.164 |
| native rivers | 583 | 1,588 km² | −0.188 | 0.068 |

And at the mouths, over each river's last three hexes:

| | droplet deposition, mean |
|---|---|
| imported | **−0.247** |
| native | −0.146 |

So a river draining 1,528 km² reaches the coast and **cuts**, more so than the native
rivers around it. `_deposit_delta` exists precisely to build a delta out of what a river
carries, and no droplet ever carries the imported catchment's load to it. The only alluvium
an imported river gets is the meander term — planed, not aggraded — which is 2.4× the native
figure purely because the imported discharge buys it a wider belt.

**This is why the earlier attempt failed.** Off-map inlet erosion by droplets was tried
twice and reverted, on the grounds that "a droplet is one raindrop wherever it starts, so
seeding them at a mouth digs a pit that inverts the inland fall and disqualifies the very
cell it was meant to serve". That is exactly right, and it diagnoses the instrument rather
than the idea: a droplet seeded at an inlet arrives with `water = 1.0` and `sediment = 0.0`,
which is full erosive capacity and nothing to drop, so of course it cuts. A droplet standing
for a river should arrive *pre-loaded* — some `water` and some `sediment` proportional to
the catchment being imported — and would then deposit on entry rather than excavate.

**The shape of the fix:** give `_drop_particle` initial `water` and `sediment` arguments,
default them to the present `1.0` and `0.0` so nothing else moves, and seed a small number
of pre-loaded droplets at each inlet in proportion to `river_inflow_volume`. Then the test
is the one the earlier attempt should have had: an imported trunk river builds a delta at
its mouth rather than trenching one, measured against the native rivers on the same map.

Worth doing **after item 19**, which changes how much floodplain any of these rivers get in
the first place.
