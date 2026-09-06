# Worldgen — Technical Debt Audit

**Audited:** 2026-05-13 · **Re-verified against master:** 2026-09-05 (`8750ea4`)  
**Scope:** All source under `worldgen/`, tests, CI, and dependencies  
**Scoring:** Priority = (Impact + Risk) × (6 − Effort), 1–5 scale each

At re-verification the suite was green — ~760 tests passing, `ruff check` clean, **93%
coverage**. Twelve of the original twenty-one items are now closed — six by the work that landed
between the two dates, item 17 by the removal of `MINING` and `FORTRESS`, and items 3, 5,
8, 9 and 11 by the Phase A sweep, all on the day of the re-audit. Two more had premises
that the same work invalidated and have been re-scoped rather than deleted; one has grown
materially worse. Every remaining item below
was checked against the file and line it names.

---

## Summary — live items

| # | Item | Category | Priority | Effort | Since audit |
|---|------|----------|----------|--------|-------------|
| 1 | `erosion.py` valley-widening and droplet paths untested | Test | **24** | S | re-scoped from "6 stages untested" |
| 19 | One off-map river strips the floodplain off half the map's channels | Model | **21** | M | unchanged |
| 18 | Market siting scores a plain hex disc, not the day-reach the catchment walks | Model | **20** | L | unchanged |
| 13 | `hydrology.py` is 1,402 lines — prime split candidate | Code | **18** | M | **worse**: was 781 |
| 20 | An off-map river imports discharge but no sediment | Model | **18** | M | unchanged |
| 21 | Runoff is linearised at the map's mean rainfall | Model | **18** | M | unchanged |
| 16 | Moisture smear blends the ocean's carrier value into coastal rainfall | Model | **12** | M | unchanged |
| 8a | `ImageElevationStage` imports a reader from `export/` | Architecture | **8** | S | unchanged |
| 14 | Bare `dict`/`list` type annotations throughout stages | Code | **8** | S | 58 occurrences |
| 15 | Rivers stay a list of paths while roads became a graph | Architecture | — | — | **do not "fix"** — see entry |

---

## Closed since the audit

Recorded so they are not raised again. Each was verified absent from master.

| # | Item | How it closed |
|---|------|---------------|
| 2 | `settlements.py` is dead code | File deleted |
| 4 | `RoadStage` ≈ `InterurbanRoadStage` | `roads.py` deleted; only `InterurbanRoadStage` remains. Resolved by removing the duplicate stage rather than extracting `_road_shared.py` |
| 6 | `_assign_role` in three places | One definition at `city_town.py:8`, now taking `cfg`; imported by `chokepoints`, `land_use`, `village_placement` |
| 7 | Magic numbers not in `WorldConfig` | City shadow radius and multiplier gone entirely (`town_min_separation` does the spreading); village thresholds moved to `habitability_village`. The one survivor was the `0.70` in `_assign_role`, closed in turn as item 17 below |
| 10 | Pipeline assembly buried in the CLI | Extracted to `stages/__init__.py` as `default_stages(model)` / `stages_for(config, model)`; `cli.py:129` and the test fixtures both consume it |
| 12 | `_get_lake_components` dead code | No longer dead — called at `hydrology.py:946` and `:1342` |
| 3 | CI coverage check silently passes | Closed 2026-09-05. `|| true` removed and the floor raised 50 → 85, against an actual 92.86%. Both halves were needed: a floor of 50 passes at half the suite's real coverage |
| 5 | `networkx` never imported | Closed 2026-09-05. Removed from `pyproject.toml` |
| 8 | `WorldState.from_json` violates layer rule | Closed 2026-09-05. Method deleted; its one remaining caller, a test, now imports `json_export.load` directly |
| 9 | `__post_init__` references fields before declaration | Closed 2026-09-05, and it was larger than recorded: **89 of the 150 fields** were declared after the method, not just `wind_direction`. Fixed by moving the *method* below every field rather than reordering any field — a pure method move, so dataclass field order, positional arguments and `asdict` output are untouched |
| 11 | `presets/` empty, `worldgen presets` shows nothing | Closed 2026-09-05, and the root cause was not the empty directory. The command globbed a `presets/` beside the installed package — the source tree in an editable install, site-packages in a real one — while the README documents `--config presets/foo.json`, a path relative to the working directory. It now reads `./presets` and says so when it finds nothing. No presets are shipped: the README says none do by design, so an empty result is the normal state and exits 0 |
| 17 | `_assign_role` compares metre elevation against 0.70 | Closed 2026-09-05 by removing `MINING` and `FORTRESS` from `SettlementRole` outright. The threshold existed only to split those two roles, nothing has ever read `Settlement.role`, and defining a fortress was the blocker — so the roles went rather than the number being guessed at. Ground with steep neighbours now falls through to the fertility test |

---

## Item Details

### 1 — `erosion.py`'s valley-widening and droplet paths are untested
**Category:** Test debt | **Priority: 24** | **Effort: S (1–2 days)**

*Re-scoped 2026-09-05.* The original entry read "six stages have no test file at all"
and named `city_town`, `interurban_roads`, `village_placement`, `village_tracks`,
`terrain_class` and `erosion`. That is no longer true: every stage is now exercised, the
six named files run 96–98% except `erosion.py`, and total coverage is 93%.

What survives is one file. `worldgen/stages/erosion.py` is at **72% — 83 uncovered lines
of 296**, the lowest in the project by 18 points, and the uncovered region is not
incidental: it is `_widen_valleys` and the droplet loop, which is exactly where items 19
and 20 record live, shipping defects. The two facts are the same fact. A defect that
"is invisible to the test suite, which exercises alluvium only at 48×48" is invisible
because that code path has no test standing on it.

**Fix:** Cover `_widen_valleys` and `_drop_particle` directly, at two map sizes rather
than one. The size dependence in item 19 is only detectable across sizes, so a fixture
that runs 48×48 alone will keep passing through the bug — which is how it got shipped.

---

### 8a — `ImageElevationStage` imports a reader from `export/`
**Category:** Architecture debt | **Priority: 8** | **Effort: S**

Unchanged, including the reasoning for leaving it. `worldgen/stages/image_elevation.py:167`
still does `from ..export.heightmap_import import load_luminance` inside `run()`, and
`worldgen/export/__init__.py` still eagerly imports `png_export` and `svg_export` — so the
one concrete cost the entry names, pulling matplotlib into any programmatic pipeline that
uses the stage, is still being paid.

The alternative was rejected on balance and the reasoning is worth keeping:
`GeneratorPipeline` registers stage *classes* and discards the `stage_config` dict
`add_stage` accepts, so injecting a pre-loaded array would mean reviving that parameter,
giving two sources of truth (config names the file, the injected array holds the pixels)
and forcing every pipeline assembler to remember to pre-load. Stashing the array on
`state.metadata` is worse still, since that dict is serialised verbatim into `world.json`.

**Fix (if it becomes a problem):** make `export/__init__.py` lazy, which removes the
matplotlib cost and leaves only the layering question. Only revive `stage_config` if a
second stage needs injected data too; otherwise the honest cleanup is to *delete* the
dead parameter.

---

### 13 — `hydrology.py` is 1,402 lines and a candidate for splitting
**Category:** Code debt | **Priority: 18** | **Effort: M (2 days)**

*Re-scored 2026-09-05: priority raised from 9.* The file was 781 lines at audit time and
is now **1,402** — it has not been split, it has doubled. It remains the largest file in
the project, now by 2× over the next (`erosion.py`, 679) rather than the 2× over
everything the audit recorded.

It still handles Priority-Flood sink filling, epsilon-tilt BFS, flow direction, flow
accumulation, river tracing with three fallback strategies, lake drainage and expansion,
and confluence splitting. The audit's judgement that "the code is correct and
well-commented" still holds and is why this is not higher. But the growth is the point:
the file absorbed the water-model port and the off-map inflow work without ever being
divided, and items 19–21 all name lines inside it or inside the comparably-grown
`erosion.py`. Difficulty navigating this file is now a tax on the model work queued
behind it.

**Fix:** Split into `hydrology_fill.py` (Priority-Flood, flow direction, accumulation)
and `hydrology_rivers.py` (river tracing, lake drainage, confluence split). The public
`HydrologyStage` imports from both. Worth doing *before* items 19–21 rather than after.

---

### 14 — Bare `dict`/`list` type annotations throughout stage code
**Category:** Code debt | **Priority: 8** | **Effort: S (1 day)**

Unchanged in character, and now countable: **58 unparameterised `dict` / `list`
annotations** across `worldgen/`, concentrated in `chokepoints.py` and `haulage.py`
(both of which postdate the audit, so the pattern is still being written).

```python
degree: dict = {}          # chokepoints.py:184 — should be dict[HexCoord, int]
owner: dict = {}           # haulage.py:231
metadata: dict = field(default_factory=dict)   # world_state.py:189 — dict[str, Any]
```

**Fix:** Annotate with full generic types, and enable `ruff`'s `ANN` rules at warning
level so new occurrences are caught rather than counted later.

---

### 15 — Rivers stay a list of paths while roads became a graph
**Category:** Architecture | **Priority: —** | **Effort: —** | *not a task*

**Re-verified 2026-09-05:** `WorldState.rivers` is still `list[River]` and everything
below still holds. This entry is a *do not fix* note, not a task — leave it in place.

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

**Re-verified 2026-09-05:** live. `climate.py:81` still smears the whole array with
`gaussian_filter`, water hexes included, and rescales on the land mean at line 100.

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

### 18 — Market siting scores a plain hex disc, not the day-reach the catchment walks
**Category:** Model | **Priority: 20** | **Effort: L**

**Re-verified 2026-09-05:** live. `MarketStage._plant` still scores with `hex_range` and
`ring` (`markets.py:97`, `:150`) while `allocate_catchments` walks the cost-bounded
Dijkstra (`markets.py:78`) — the two halves still disagree exactly as described.

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

**Re-verified 2026-09-05:** live. `erosion.py:412` still takes `max_flow = float(flow.max())`
and line 433 still disqualifies a channel outright on `reach < 1.0`. See item 1: the code
carrying this defect sits inside the 83 uncovered lines of `erosion.py`, which is why the
suite does not see it.

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

**Re-verified 2026-09-05:** live. `_drop_particle` still opens with `water = 1.0` and
`sediment = 0.0` (`erosion.py:121`), and nothing seeds pre-loaded droplets at the inlets
`_inflow_mouths` returns.

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


### 21 — Runoff is linearised at the map's mean rainfall
**Category:** Model | **Priority: 18** | **Effort: M**

**Re-verified 2026-09-05:** live. `hydrology.py:100` and `haulage.py:59` both still
evaluate `runoff_mm` once at `mean_precip_mm`. Worth adding: `biomes.py:100` already calls
`runoff_mm(h.moisture, h.temperature)` per hex, so the per-hex form is live in the codebase
and the three consumers do not agree with each other.

`runoff_mm` is Pike's curve, and it is markedly non-linear in rainfall: the ground takes
most of the rain in a dry region and little of it in a wet one, which is the whole reason
the curve replaced a flat subtraction. But it is evaluated **once, at the map's mean**, and
the result used as a single figure for every hex:

```python
runoff_mm = self.config.runoff_mm(self.config.mean_precip_mm)   # hydrology.py:100
discharge = hx.catchment_km2 * cfg.runoff_mm(cfg.mean_precip_mm) # haulage.py:59
```

Half of what this used to cost has already been paid. `rain_per_hex` weights flow
accumulation by the orographic pattern, so `acc` counts hexes of *relative* rain rather
than bare area and a rain-shadowed catchment already raises a smaller river. What is left
is that converting that accumulation to a discharge assumes runoff scales **linearly** with
rainfall, when it does not.

Measured on a 96×96 temperate map (seed 42, rainfall spanning 158–4,586 mm against a mean
of 800):

| | rainfall | runoff, actual | runoff, as linearised | error |
|---|---|---|---|---|
| driest decile | 401 mm | 137.5 mm | 240.5 mm | **1.75× too much** |
| median | 588 mm | 286.8 mm | 352.0 mm | 1.23× too much |
| wettest decile | 1,582 mm | 1,239.9 mm | 947.7 mm | **0.76× too little** |

So dry country is credited with nearly twice the water it sheds and wet uplands with three
quarters of theirs — a spread of 2.3× applied the wrong way round. Channels therefore form
too readily in rain shadows and too reluctantly on windward slopes, and `navigable` inherits
the same error through `haulage.py`, floating boats on desert rivers that would not carry
them.

**The fix is small; the recalibration is not.** Weighting each hex's contribution by
`runoff_mm(hx.moisture, hx.temperature) / runoff_mm(mean_precip_mm)` inside `rain_per_hex`
puts the non-linearity where the rain already is, and leaves the threshold a single figure —
one function call and no new setting. What it changes is how much of every map is under
channel, which is `channel_min_discharge`'s calibration and the wetland and navigability
figures with it. Measure across all five climates before and after; the point of the change
is that arid and tropical should move in *opposite* directions.

---

## Phased Remediation Plan

Rewritten 2026-09-05, and Phase A struck the same day. Phases C, D and E of the original
plan were already done or moot; what remains is one real test gap, one file that needs
dividing before it can be worked in, and a queue of model decisions blocked on a
calibration table rather than on effort.

### Phase A — Quick wins — **done 2026-09-05**
Items 3, 5, 8, 9 and 11, in one sitting as scoped. Two ran deeper than the audit recorded
and both are written up in the closed table: `__post_init__` sat ahead of 89 fields rather
than one, and `worldgen presets` was looking in the wrong directory entirely rather than
merely at an empty one. The coverage ratchet from item 3 now guards everything below.

### Phase B — Close the one remaining test gap (1–2 days) — **next**
1. Cover `_widen_valleys` and `_drop_particle` in `erosion.py`, **at two map sizes**
   (**item 1**). This is a prerequisite for Phase D, not a parallel track: items 19 and
   20 live in this code, and a 48×48-only fixture passes straight through both.

### Phase C — Make the water model navigable (2 days)
2. Split `hydrology.py` (**item 13**). Promoted ahead of the model work rather than
   filed as optional: the file doubled while items 19–21 accumulated inside it and
   `erosion.py`, and every remaining model change has to be made in there.

### Phase D — The model decisions (each its own branch, acceptance table on the desk)
These do not queue behind effort, they queue behind a calibration decision. Order is
load-bearing where noted:

3. Off-map river vs. `max_flow` in `_widen_valleys` (**item 19**) — do first. It is the
   only one of the five whose defect is silently shipping on every map above ~64×64.
4. Pre-loaded droplets at the inlets (**item 20**) — explicitly after 19, which changes
   how much floodplain any river gets in the first place.
5. Per-hex runoff instead of runoff at the mean (**item 21**) — smallest diff, largest
   recalibration; arid and tropical must move in opposite directions or it is wrong.
6. Day-reach market siting (**item 18**) — the rewrite exists and works; what it needs
   is the density decision and a re-tune of `city_min_draw` and the chokepoint gates
   together.
7. Land-only moisture smear (**item 16**) — blocked on deciding what makes a 550 mm
   climate pastoral once the artifact is gone.

### Phase E — Low priority
8. Parameterise `dict` / `list` annotations and turn on `ruff`'s `ANN` rules
   (**item 14**)
9. `export/__init__.py` lazy imports, if the matplotlib cost ever bites (**item 8a**)
