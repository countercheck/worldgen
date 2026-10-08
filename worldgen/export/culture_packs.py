"""Reading culture packs off disk.

The packs that ship live beside `naming.packs` as YAML package data; a user adds their
own by listing folders in `naming_pack_dirs`. Folders are read in order after the
built-in packs, and a pack whose key is already taken replaces the earlier one — so a
user can fork a built-in pack by copying its file and editing it under the same key.

Here rather than in `naming/`, because this is file I/O and `naming/` does none; it hands
each file's contents to `naming.packs.schema.parse_pack`, which checks them. The stage
reads its packs through this, as `ImageElevationStage` reads its image through
`heightmap_import`.
"""

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path

import yaml

from ..naming.packs import CulturePackError, Pack, pack_hash, parse_pack


@dataclass(frozen=True)
class LoadedPack:
    pack: Pack
    # "builtin" or "user".
    source: str
    # The file it came from, for messages.
    path: str
    # A short fingerprint of the pack's content; see `pack_hash`.
    hash: str


def _parse(text: str, name: str, source: str) -> LoadedPack:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise CulturePackError(f"{name}: is not valid YAML ({exc})") from exc
    return LoadedPack(parse_pack(data, name), source, name, pack_hash(data))


def _add(found: dict[str, LoadedPack], loaded: LoadedPack, seen_here: dict[str, str]) -> None:
    key = loaded.pack.key
    if key in seen_here:
        raise CulturePackError(
            f"{loaded.path}: key {key!r} is already used by {seen_here[key]} in the same folder"
        )
    seen_here[key] = loaded.path
    found[key] = loaded


@lru_cache(maxsize=1)
def _builtin() -> tuple[LoadedPack, ...]:
    """The shipped packs. Package data does not change under a running process."""
    folder = resources.files("worldgen.naming.packs")
    entries = sorted(
        (e for e in folder.iterdir() if e.name.endswith(".yaml")), key=lambda e: e.name
    )
    found: dict[str, LoadedPack] = {}
    seen: dict[str, str] = {}
    for entry in entries:
        _add(found, _parse(entry.read_text(encoding="utf-8"), entry.name, "builtin"), seen)
    return tuple(found.values())


def load_packs(extra_dirs: tuple[str, ...] = ()) -> dict[str, LoadedPack]:
    """Every culture pack available: the built-in ones, then each user folder in order.

    A pack in a later folder replaces one of the same key from anywhere earlier.
    """
    found = {p.pack.key: p for p in _builtin()}
    for folder in extra_dirs:
        path = Path(folder)
        if not path.is_dir():
            raise CulturePackError(f"naming_pack_dirs: {folder!r} is not a folder")
        seen: dict[str, str] = {}
        for file in sorted(path.glob("*.yaml")):
            _add(found, _parse(file.read_text(encoding="utf-8"), str(file), "user"), seen)
    return found


__all__ = ["LoadedPack", "load_packs"]
