"""Place names, built from what a place is rather than from letters.

A real place name is a compound whose parts once meant something about the site: Oxford
is the oxen's ford, Aberystwyth the mouth of the Ystwyth, Kirkby the farm by the church.
So naming here is two steps with a seam between them.

`site` reads the ground and says what a site *means* — a ford, a marsh, a hill, a mine —
in words no culture owns. A `Culture` turns meanings into words of its own, and one
culture's word for "ford" is the same everywhere on the map, which is what makes a
generated language read as a language rather than as noise.

Pure logic over `core/` types. Nothing here reads a file or draws, and nothing here is a
stage: `stages.naming` is the one that runs it.
"""

from .conlang import Language
from .culture import Culture, Qualifier
from .regions import culture_regions
from .registry import NameRegistry, edit_distance
from .site import GENERICS, GLOSS, SPECIFICS, Site, add_direction, read_site

__all__ = [
    "GENERICS",
    "GLOSS",
    "SPECIFICS",
    "Culture",
    "Language",
    "NameRegistry",
    "Qualifier",
    "Site",
    "add_direction",
    "culture_regions",
    "edit_distance",
    "read_site",
]
