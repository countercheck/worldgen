"""Culture packs: naming cultures written by hand from real place-name elements.

See `base` for how a pack is written. Each module here is one people's naming habits;
`PACKS` is the registry the stage looks them up in by key, and `core.config.CULTURE_PACKS`
lists the same keys so a config can be checked without importing this.
"""

from . import arabic, english, french, norse, slavic, welsh
from .base import Pack, PackCulture

PACKS: dict[str, Pack] = {
    p.key: p for p in (english.PACK, norse.PACK, welsh.PACK, french.PACK, slavic.PACK, arabic.PACK)
}

__all__ = ["PACKS", "Pack", "PackCulture"]
