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
hex_traffic: dict = defaultdict(float)        # should be dict[HexCoord, float]
canonical_routes: dict = {}                    # should be dict[tuple, list[HexCoord]]
hex_tier: dict = {}                            # should be dict[HexCoord, RoadTier]
metadata: dict = field(default_factory=dict)   # should be dict[str, Any]
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
