"""Sindarin: the Grey-elven tongue of Tolkien's Middle-earth.

Names are mostly a noun and what follows it, written as two words — Amon Hen, Dol
Guldur, Ethir Anduin — or run together with the qualifier first where Tolkien made a true
compound (Caradhras, red-horn). Heads and qualifiers are Sindarin vocabulary from
Tolkien's glossaries and *The Etymologies*: athrad a ford, iant a bridge, lond a haven,
cirith a cleft or pass, ost and barad a fortress, taur a great forest, nim white, morn
black, beleg mighty.

Soft mutation, which Sindarin applies inside compounds and after the article, is not
modelled; names come out in their citation forms. Where Tolkien left no word — salt, the
hare, the goat — a plausible element is used and marked below.

Rivers are built rather than borrowed: an element and duin (great river), sîr (river) or
nen (water), so a map gets Morduin and Nimsîr, not a copy of Middle-earth's own rivers.
"""

from .base import Pack

PACK = Pack(
    key="sindarin",
    name="Sindarin",
    heads={
        "ford": ("athrad",),
        "bridge": ("iant",),
        "confluence": ("govad",),  # "meeting", a later coinage
        "mouth": ("ethir",),
        "falls": ("lanthir",),
        "pass": ("cirith",),
        "harbour": ("lond", "lonn"),
        "shore": ("falas",),
        "market": ("bach",),  # an article of trade
        "mine": ("groth", "grod"),
        "clearing": ("lant",),
        "spring": ("eithel",),
        "hill": ("amon", "dol"),
        "steading": ("bar",),
        "hamlet": ("gobel",),
        "enclosure": ("iâth", "echor"),
        "cottages": ("car",),
        "field": ("parth", "talath"),
        "town": ("caras",),
        "wick": ("caras", "bach"),
        "stow": ("echad",),
        "burgh": ("ost", "barad"),
        "hall": ("tham",),
        "wall": ("ram", "rammas"),
    },
    quals={
        "marsh": ("lô",),
        "fen": ("nîn",),
        "deepwood": ("taur",),
        "wood": ("eryn", "glad"),
        "thorn": ("ereg",),
        "meadow": ("sâdh",),
        "cold": ("ring", "helch"),
        "sand": ("lith",),
        "stone": ("gond", "sarn"),
        "salt": ("hâl",),  # no attested word; after Quenya singë's sense
        "rich": ("laur",),  # golden
        "high": ("tar",),
        "oak": ("doron",),
        "ash": ("fêr",),  # beech; no ash is attested
        "elm": ("lalorn",),
        "birch": ("brethil",),
        "pine": ("thôn",),
        "palm": ("mallorn",),
        "broom": ("athel",),
        "reed": ("lisc",),
        "ox": ("mund",),
        "deer": ("aras",),
        "swine": ("hû",),  # a hound; no swine is attested
        "horse": ("roch",),
        "hare": ("lau",),  # invented
        "elk": ("aras",),
        "wolf": ("draug", "garaf"),
        "bear": ("brôg",),
        "goat": ("naew",),  # invented
        "crane": ("gwael",),  # a gull
        "eagle": ("thoron",),
        "white": ("nim", "faen"),
        "black": ("morn", "mor"),
        "red": ("caran",),
        "green": ("calen", "galen"),
        "north": ("forn", "for"),
        "south": ("har", "harn"),
        "east": ("rhûn",),
        "west": ("dûn",),
        "great": ("beleg",),
        "little": ("tithen", "niben"),
    },
    compound=("{h} {q}", "{h} {q}", "{q}{h}"),
    founder=("{h} {p}", "{p}{h}"),
    river_on=("{h} {r}",),
    person_first=(
        "Gal",
        "Cel",
        "Ara",
        "Elen",
        "Fin",
        "Mith",
        "Bel",
        "Thar",
        "Hal",
        "Glor",
        "Aeg",
        "Ered",
        "Lin",
        "Nim",
    ),
    person_second=("dir", "ion", "ril", "wen", "thor", "dor", "las", "born", "hir", "iel"),
    river_roots=(
        "mor",
        "nim",
        "calen",
        "caran",
        "celeb",
        "glos",
        "hith",
        "lin",
        "gwath",
        "ring",
        "mith",
        "faen",
        "baran",
        "luin",
        "ninn",
        "taur",
    ),
    river_patterns=("{root}duin", "{root}sîr", "{root}nen"),
    particles=frozenset({"in", "i", "en"}),
)
