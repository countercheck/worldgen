# Worldgen

Hex-based procedural world generator for TTRPGs, worldbuilding, and wargaming.
1 hex = 1 km. Python 3.11+. Every run is fully reproducible from a single integer seed.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Commands

```bash
python3 -m pytest          # run tests
python3 -m ruff check .    # lint
python3 -m ruff format .   # format
```

## Architecture (never violate)

See `docs/ARCHITECTURE.md` for Mermaid diagrams of the layer structure and the
generation pipeline in run order.

- `core/` — data types and pipeline only; no rendering, no file I/O
- `stages/` — pure transformers: `stage.run(WorldState) -> WorldState`
- `export/` — all file I/O lives here; stages never write files
- `render/` — matplotlib debug viewer; never imported by stages or core
- `analysis/` — measurements over a finished world (drainage-network shape, and whatever
  follows); imports `core/` only, does no file I/O, and is **never imported by a stage** —
  a stage that could read its own report card would start tuning itself against a metric

These are checked, not just described: `tests/test_layering.py` parses every module and
fails on a layer importing one above it. Add a rule there when you add one here.
- All random calls use a seeded `numpy.random.Generator` passed explicitly — no global state
- All tunable parameters live in `WorldConfig`; nothing hardcoded in stage logic

## Testing rules

- Every stage gets a corresponding test file in `tests/`
- Write tests for new code before or immediately after implementing it — never skip
- Tests assert structural invariants, not exact values (outputs are seed-dependent)
- Key invariants: rivers reach ocean, no uphill flow, city separation respected,
  road paths connected, JSON round-trips losslessly, same seed → same output

## Pre-commit checks

The pre-commit hook runs `ruff check` and `pytest` before every commit.
Install after cloning: the hook is at `.git/hooks/pre-commit` (set up automatically
if you run `pip install -e ".[dev]"` and the hook script is present).

To bypass in an emergency: `git commit --no-verify` (use sparingly).

## Implementation order

Follow the phase order in `worldgen_plan.md`. Do not start a new phase until
the debug viewer confirms the current phase output looks geographically plausible.
