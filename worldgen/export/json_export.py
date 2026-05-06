import json
from pathlib import Path

from ..core.world_state import WorldState


def save(ws: WorldState, path) -> None:
    """Serialize WorldState to a JSON file."""
    Path(path).write_text(json.dumps(ws.to_dict(), indent=2))


def load(path) -> WorldState:
    """Deserialize WorldState from a JSON file produced by save()."""
    return WorldState.from_dict(json.loads(Path(path).read_text()))
