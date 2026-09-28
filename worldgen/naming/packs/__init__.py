"""Culture packs: naming cultures written as data rather than invented from a seed.

A generated language is endless and alien. A pack is the other thing a map sometimes
wants: names a reader half-recognises, because they are built from the elements real
places are built from — English -ford and -ton, Norse -by and -thwaite, Welsh aber- and
pont-, French -ville and plessis — or from a fiction's own, like Sindarin or a hive's.

Each pack is a YAML file in this folder (or in a user's folder named by
`naming_pack_dirs`): for every meaning in `site.GENERICS` and `site.SPECIFICS`, the words
that culture used for it, and templates for how those words go together. `schema`
checks a file and builds a `Pack`; `processor.PackCulture` runs any pack. Reading the
files is `export.culture_packs`'s job. The format is documented for authors in
`docs/CULTURE_PACKS.md`.
"""

from .processor import PackCulture
from .schema import CulturePackError, Pack, ProperNames, pack_hash, parse_pack

__all__ = ["CulturePackError", "Pack", "PackCulture", "ProperNames", "pack_hash", "parse_pack"]
