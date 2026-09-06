"""Render a masked world and its unmasked original side by side, for eyeballing.

CLAUDE.md asks that the debug viewer confirm output looks plausible before moving on.
Fog of war is a case where that matters more than usual: a masking bug produces a file
that loads, parses and passes every structural assertion while showing the wrong ground.

    python3 scripts/preview_fog.py fog.json out.png
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
import sys
from pathlib import Path

from PIL import Image

from worldgen.export import png_export


def main() -> None:
    masked_path = Path(sys.argv[1])
    out = Path(sys.argv[2])

    from worldgen.core.world_state import WorldState

    masked = WorldState.from_dict(json.loads(masked_path.read_text()))

    fixture = (
        Path(__file__).resolve().parent.parent
        / "campaign"
        / "shared"
        / "test"
        / "fixtures"
        / "world-32x32.json"
    )
    full = WorldState.from_dict(json.loads(fixture.read_text()))

    cfg = png_export.PNGConfig(hex_size=14)
    left = png_export.render(full, cfg)
    right = png_export.render(masked, cfg)

    canvas = Image.new(
        "RGB", (left.width + right.width + 20, max(left.height, right.height)), "white"
    )
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + 20, 0))
    canvas.save(out)
    print(f"wrote {out} ({canvas.width}x{canvas.height})")
    print(f"  left: the world as generated, {len(full.hexes)} hexes")
    fogged = sum(1 for h in masked.hexes.values() if "fog" in h.tags)
    print(f"  right: red's map, {len(masked.hexes) - fogged} seen of {len(masked.hexes)}")


if __name__ == "__main__":
    main()
