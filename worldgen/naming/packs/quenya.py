"""Quenya: the High-elven tongue of Tolkien's Middle-earth.

Quenya compounds put the qualifier first and the head last, run together: Alqualondë is
swan-haven, Valimar the home of the Valar. The vocabulary is from Tolkien's own lists and
*The Etymologies*: londë a haven, yanta a bridge, osto a fortified town, ambo a hill,
tavar a wood, ninquë white, morë black, formen north, alta great.

A word-final ë keeps its diaeresis at the end of a name and loses it inside one, as
Tolkien wrote them. Where no word is attested it is marked below.

Rivers are built rather than borrowed: an element and sírë (river) or nen (water).
"""

from .base import Pack

PACK = Pack(
    key="quenya",
    name="Quenya",
    heads={
        "ford": ("tarna",),  # a crossing
        "bridge": ("yanta",),
        "confluence": ("yomenië",),  # a meeting
        "mouth": ("etsir",),
        "falls": ("lanta",),  # a fall
        "pass": ("tië",),  # a path
        "harbour": ("londë",),
        "shore": ("hresta",),
        "market": ("mancalë",),  # trade
        "mine": ("rotto",),  # a cave
        "clearing": ("landë",),  # an open space
        "spring": ("ehtelë",),
        "hill": ("ambo", "oron"),
        "steading": ("mar",),
        "hamlet": ("opelë",),
        "enclosure": ("peler",),
        "cottages": ("coa",),
        "field": ("resta",),
        "town": ("osto",),
        "wick": ("osto", "mancalë"),
        "stow": ("omentië",),
        "burgh": ("arta", "mindon"),
        "hall": ("sambë",),
        "wall": ("ramba",),
    },
    quals={
        "marsh": ("mixa",),  # wet
        "fen": ("nen",),
        "deepwood": ("taurë",),
        "wood": ("tavar",),
        "thorn": ("necel",),
        "meadow": ("salquë",),  # grass
        "cold": ("ringa",),
        "sand": ("litsë",),
        "stone": ("ondo",),
        "salt": ("singë",),
        "rich": ("alya",),
        "high": ("tar",),
        "oak": ("norno",),
        "ash": ("ornë",),  # a tree; no ash is attested
        "elm": ("alalmë",),
        "birch": ("silwin",),  # invented
        "pine": ("sonda",),  # invented
        "palm": ("malinornë",),
        "broom": ("aica",),  # sharp, of gorse
        "reed": ("liscë",),
        "ox": ("mundo",),
        "deer": ("arassë",),
        "swine": ("polca",),
        "horse": ("rocco",),
        "hare": ("lapat",),
        "elk": ("arassë",),
        "wolf": ("narmo", "ráca"),
        "bear": ("morco",),
        "goat": ("nyéni",),
        "crane": ("alqua",),  # a swan
        "eagle": ("soron",),
        "white": ("ninquë", "losse"),
        "black": ("morë",),
        "red": ("carnë",),
        "green": ("laiquë",),
        "north": ("formen",),
        "south": ("hyarmen",),
        "east": ("rómen",),
        "west": ("númen",),
        "great": ("alta",),
        "little": ("pitya",),
    },
    compound=("{q}{h}",),
    founder=("{p}{h}",),
    river_on=("{r}{h}",),
    person_first=("Elen", "Anar", "Isil", "Tar", "Cal", "Vin", "Ear", "Fin", "Mir", "Ingo", "Nol"),
    person_second=("dil", "wë", "ion", "ondo", "mir", "indë", "atan"),
    river_roots=(
        "ninquë",
        "morë",
        "carnë",
        "laiquë",
        "luin",
        "telpë",
        "laurë",
        "silma",
        "ringa",
        "mixa",
        "alta",
        "tára",
    ),
    river_patterns=("{root}sírë", "{root}nen"),
    # A final ë loses its diaeresis once another element follows it.
    fixes=((r"ë(?=[a-zà-ÿ])", "e"),),
)
