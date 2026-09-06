"""Build the small generated-world fixture the TypeScript loader is tested against.

`campaign/shared/src/world.ts` parses the generator's `world.json`. That parser has to
survive real output, not a hand-written approximation of it — the schema has ~24 fields
per hex, four of them nullable enums, and two edge collections keyed by coordinate pairs.
So one genuinely generated world is committed and the loader is asserted against it.

It is deliberately small and deliberately *organic*: the organic pipeline is the one that
runs `CrossingStage`, so this is the only fixture that carries `ford` and `bridge` tags,
which the river-crossing rules key on.

Floats are rounded, which is the only departure from real output. Full double precision
roughly doubles the file for digits no test can meaningfully assert on; the schema, the
nulls, the enums and the coordinate pairs — everything the parser actually has to get
right — are untouched.

Regenerate with:

    python3 scripts/make_world_fixture.py

A diff here means the generator's output changed, which is worth looking at rather than
committing blind.
"""

import sys
from pathlib import Path

# Put the repository root ahead of anything else on the path.
#
# `python scripts/foo.py` sets sys.path[0] to `scripts/`, not to the repo root, so an
# editable install of `worldgen` elsewhere on the machine wins the import — which in a
# git worktree means this script silently runs against the *main* checkout's code while
# the tests run against the worktree's. That divergence is invisible until output stops
# matching what the tests say it should.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from pathlib import Path

from worldgen.core.config import WorldConfig
from worldgen.core.pipeline import GeneratorPipeline
from worldgen.stages import stages_for

FIXTURE = (
    Path(__file__).resolve().parent.parent
    / "campaign"
    / "shared"
    / "test"
    / "fixtures"
    / "world-32x32.json"
)

SEED = 7
SIZE = 32
MODEL = "organic"
PLACES = 4


def round_floats(value, places: int = PLACES):
    """Recursively round every float, leaving ints, strings, bools and None alone."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        return round(value, places)
    if isinstance(value, dict):
        return {k: round_floats(v, places) for k, v in value.items()}
    if isinstance(value, list):
        return [round_floats(v, places) for v in value]
    return value


def main() -> None:
    cfg = WorldConfig(width=SIZE, height=SIZE, model=MODEL)
    pipeline = GeneratorPipeline(SEED, cfg)
    for stage in stages_for(cfg, cfg.model):
        pipeline.add_stage(stage)
    world = pipeline.run()

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(round_floats(world.to_dict()), indent=1) + "\n")

    data = json.loads(FIXTURE.read_text())
    tags = sorted({t for h in data["hexes"] for t in h["tags"]})
    print(f"wrote {FIXTURE} ({FIXTURE.stat().st_size // 1024} KB)")
    print(f"  hexes={len(data['hexes'])} rivers={len(data['rivers'])}")
    print(f"  settlements={len(data['settlements'])} road_edges={len(data['road_edges'])}")
    print(f"  sea_edges={len(data['sea_edges'])} ferries={len(data['ferries'])}")
    print(f"  terrain={sorted({h['terrain_class'] for h in data['hexes']})}")
    print(f"  road_tiers={sorted({e['tier'] for e in data['road_edges']})}")
    print(f"  tags={tags}")


if __name__ == "__main__":
    main()
