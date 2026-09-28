"""Khuzdul: the secret tongue of Tolkien's Dwarves.

Tolkien left only a few dozen words of it, nearly all from place names: dûm (halls,
delvings), gathol (fortress), gunud (a delving), zahar (mansion), tumun (hollow), bizar
(dale), zâram (lake), nâla (river course), ul (streams), baraz (red), kibil (silver),
azan (dark), kheled (glass), zigil (spike), bund (head), gabil (great). Those carry the
pack. Everything else is invented in their sound — hard consonants, a and u, the
circumflex — and marked below, so nobody mistakes it for Tolkien's.

Names join with a hyphen and keep the second part in lower case, as Khazad-dûm and
Kheled-zâram do. Founders take names from the Dvergatal, the list of dwarves in the Old
Norse Völuspá that Tolkien drew his own dwarves from.
"""

from .base import Pack

PACK = Pack(
    key="khuzdul",
    name="Khuzdul",
    heads={
        "ford": ("gundab",),  # invented
        "bridge": ("baruk",),  # invented
        "confluence": ("ulbad",),  # ul (streams) + invented
        "mouth": ("ulnar",),  # ul (streams) + invented
        "falls": ("ulkhaz",),  # ul (streams) + invented
        "pass": ("nargun",),  # invented
        "harbour": ("khelek",),  # invented
        "shore": ("zâram",),
        "market": ("tharbul",),  # invented
        "mine": ("gunud",),
        "clearing": ("felak",),  # to hew
        "spring": ("ulun",),  # ul (streams)
        "hill": ("zigil", "bund"),
        "steading": ("zahar",),
        "hamlet": ("tumun",),
        "enclosure": ("gundar",),  # invented
        "cottages": ("khiz",),  # invented
        "field": ("bizar",),
        "town": ("dûm",),
        "wick": ("tharbul",),  # invented
        "stow": ("aglab",),  # speech
        "burgh": ("gathol",),
        "hall": ("dûm", "zahar"),
        "wall": ("gathol",),
    },
    quals={
        "marsh": ("ulzar",),  # invented
        "fen": ("nulun",),  # invented
        "deepwood": ("azan",),  # dark
        "wood": ("rukh",),  # invented
        "thorn": ("zigil",),
        "meadow": ("bizar",),
        "cold": ("kheled",),  # glass
        "sand": ("rakhab",),  # invented
        "stone": ("tharkh",),  # invented
        "salt": ("mizim",),  # invented
        "rich": ("mazar",),  # invented
        "high": ("bund",),  # head
        "oak": ("gorth",),  # invented
        "ash": ("azar",),  # invented
        "elm": ("ulmuk",),  # invented
        "birch": ("kibrak",),  # invented
        "pine": ("narduk",),  # invented
        "palm": ("zundur",),  # invented
        "broom": ("khurz",),  # invented
        "reed": ("ulsid",),  # invented
        "ox": ("barak",),  # invented
        "deer": ("zhelek",),  # invented
        "swine": ("gorog",),  # invented
        "horse": ("ruzhan",),  # invented
        "hare": ("kithik",),  # invented
        "elk": ("zhelek",),  # invented
        "wolf": ("narg",),  # invented
        "bear": ("bundab",),  # invented
        "goat": ("gamil",),  # invented
        "crane": ("shurak",),  # invented
        "eagle": ("khuzal",),  # invented
        "white": ("kibil",),  # silver
        "black": ("azan",),  # dark
        "red": ("baraz",),
        "green": ("ruzak",),  # invented
        "north": ("tharak",),  # invented
        "south": ("zudân",),  # invented
        "east": ("bazûk",),  # invented
        "west": ("ushar",),  # invented
        "great": ("gabil",),
        "little": ("nuluk",),  # invented
    },
    compound=("{q}-{h}", "{q}-{h}", "{q}{h}"),
    founder=("{p}-{h}",),
    river_on=("{r}-{h}",),
    person_first=(
        "Nar",
        "Frar",
        "Hepti",
        "Vili",
        "Hanar",
        "Svior",
        "Bruni",
        "Bild",
        "Buri",
        "Fundin",
        "Nali",
        "Loni",
        "Dolgthrasir",
        "Hlevang",
        "Duf",
        "Andvari",
        "Skirfir",
        "Virfir",
        "Skafid",
        "Alf",
        "Yngvi",
        "Fjalar",
        "Frosti",
        "Fid",
        "Ginnar",
        "Mjodvitnir",
    ),
    river_roots=(
        "kibil",
        "baraz",
        "azan",
        "kheled",
        "gabil",
        "sigin",
        "zirak",
        "narag",
        "ruzak",
        "tharak",
        "bund",
        "mizim",
    ),
    river_patterns=("{root}-nâla", "{root}ul"),
    hyphen_caps=False,
)
