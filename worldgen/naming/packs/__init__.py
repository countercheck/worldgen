"""Culture packs: naming cultures written by hand from real place-name elements.

See `base` for how a pack is written. Each module here is one people's naming habits;
`PACKS` is the registry the stage looks them up in by key, and `core.config.CULTURE_PACKS`
lists the same keys so a config can be checked without importing this.
"""

from . import (
    arabic,
    dutch,
    english,
    french,
    german,
    hive,
    hobbitish,
    italian,
    khuzdul,
    latin,
    norse,
    quenya,
    rohirric,
    sindarin,
    slavic,
    spanish,
    welsh,
)
from .base import Pack, PackCulture

PACKS: dict[str, Pack] = {
    p.key: p
    for p in (
        english.PACK,
        norse.PACK,
        welsh.PACK,
        french.PACK,
        slavic.PACK,
        arabic.PACK,
        spanish.PACK,
        german.PACK,
        dutch.PACK,
        italian.PACK,
        latin.PACK,
        # Tolkien's Middle-earth.
        sindarin.PACK,
        quenya.PACK,
        khuzdul.PACK,
        rohirric.PACK,
        hobbitish.PACK,
        # Not human at all.
        hive.PACK,
    )
}

__all__ = ["PACKS", "Pack", "PackCulture"]
