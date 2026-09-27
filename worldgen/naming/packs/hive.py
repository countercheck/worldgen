"""Hive: an insect people, named in translation.

Their names are given as a translator renders them — what the place *is* to the colony —
and only what cannot be translated is left in their own tongue: the names of rivers,
which mean nothing but themselves, and the names of the swarms that founded each place.

**Rank shapes the name.** A queen's hive is old and named with ceremony, as a phrase:
Queen-Hive-of-the-Sweet-Rot. A daughter colony is named more briefly, and an outpost —
a larder, a nursery, a fungus garden — in a terse stack the way workers mark a trail:
Sweet-Rot-Larder.

**Rank shapes the sound, too.** Their tongue has two registers. The queen's is a buzz —
long sibilants, z and v and r, hummed — and the workers' is a click of hard stops. A
queen's hive is founded by a swarm with a buzzing name, an outpost by one with a clicking
name, and a daughter colony by either. Great rivers buzz; lesser ones click.

**Water is a resource**, not a danger: a ford is a drink-crossing, a harbour a wet-edge,
a river mouth a great-drink. **Senses stand in** where a sense is what an insect would
notice — a marsh is sweet-rot, stone is cold-still, deep forest deep-shade, north the
cold side — and plain words where they are not: oak, hare, green.

**Qualifiers are nouns**, so they read in the queen's phrases as well as the workers'
stacks: the many and the few rather than great and little (Hive-of-the-Many,
Few-Larder), the heights rather than high.
"""

import zlib

import numpy as np

from .base import Pack

# Onsets, vowels and codas for the two registers of the tongue.
_BUZZ = (
    ("zz", "vr", "sh", "zh", "hm", "z", "v", "ss", "th"),
    ("aa", "u", "i", "ee", "o", "au"),
    ("rr", "mm", "z", "", "ng", "ss"),
)
_CLICK = (
    ("k'", "tz'", "tch", "k", "t", "q'", "x", "tk"),
    ("i", "a", "e", "u"),
    ("k", "t", "tch", "", "ik", "x"),
)


def tongue(key: str, syllables: int, rank: int) -> str:
    """A word in the hive's own tongue, stable for its key.

    Rank 0 — a queen's hive, a great river — buzzes; rank 2 clicks; rank 1 may do either,
    decided by the key so it is the same every time.
    """
    rng = np.random.default_rng([zlib.crc32(key.encode()), max(1, syllables), rank])
    if rank <= 0:
        onsets, vowels, codas = _BUZZ
    elif rank >= 2:
        onsets, vowels, codas = _CLICK
    else:
        onsets, vowels, codas = _BUZZ if rng.random() < 0.5 else _CLICK
    return "".join(
        onsets[int(rng.integers(len(onsets)))]
        + vowels[int(rng.integers(len(vowels)))]
        + codas[int(rng.integers(len(codas)))]
        for _ in range(max(1, syllables))
    )


PACK = Pack(
    key="hive",
    name="Hive",
    heads={
        # Water, as something to drink and build with.
        "ford": ("drink-crossing",),
        "bridge": ("dry-crossing",),
        "confluence": ("two-waters",),
        "mouth": ("great-drink",),
        "falls": ("loud-water",),
        "pass": ("narrow-way",),
        "harbour": ("wet-edge",),
        "shore": ("wet-rim",),
        "market": ("sharing-place",),
        "mine": ("deep-dig",),
        "clearing": ("cut-ground",),
        "spring": ("clean-drink",),
        "hill": ("mound", "sun-mound"),
        # Outposts: what a forager's colony keeps away from the queen.
        "steading": ("forage-post",),
        "hamlet": ("nursery",),
        "enclosure": ("larder",),
        "cottages": ("cells",),
        "field": ("fungus-garden",),
        # Daughter colonies.
        "town": ("daughter-nest",),
        "wick": ("sharing-nest",),
        "stow": ("gathering",),
        # The queen's seat.
        "burgh": ("queen-hive", "queen-mound"),
        "hall": ("great-comb",),
        "wall": ("sealed-hive",),
    },
    quals={
        "marsh": ("sweet-rot",),
        "fen": ("soft-rot",),
        "deepwood": ("deep-shade",),
        "wood": ("shade",),
        "thorn": ("thorn",),
        "meadow": ("nectar",),
        "cold": ("cold",),
        "sand": ("loose-ground",),
        "stone": ("cold-still",),
        "salt": ("salt",),
        "rich": ("sweetness",),
        "high": ("heights",),
        "oak": ("oak",),
        "ash": ("ash",),
        "elm": ("elm",),
        "birch": ("birch",),
        "pine": ("pine",),
        "palm": ("palm",),
        "broom": ("broom",),
        "reed": ("reed",),
        "ox": ("ox",),
        "deer": ("deer",),
        "swine": ("swine",),
        "horse": ("horse",),
        "hare": ("hare",),
        "elk": ("elk",),
        "wolf": ("wolf",),
        "bear": ("bear",),
        "goat": ("goat",),
        "crane": ("crane",),
        "eagle": ("eagle",),
        "white": ("pale-light",),
        "black": ("dark",),
        "red": ("red-earth",),
        "green": ("green",),
        "north": ("cold-side",),
        "south": ("warm-side",),
        "east": ("dawn-side",),
        "west": ("dusk-side",),
        "great": ("many",),
        "little": ("few",),
    },
    # Rank-blind fallbacks; the ranked templates below are what the stage uses.
    compound=("{q}-{h}",),
    founder=("{p}-{h}",),
    river_on=("{r}-{h}",),
    ranked={
        "compound": (
            ("{h}-of-the-{q}", "{h}-under-{q}", "{h}-beside-{q}"),
            ("{h}-by-{q}", "{q}-{h}"),
            ("{q}-{h}",),
        ),
        "founder": (
            ("{h}-of-{p}-swarm",),
            ("{p}-swarm-{h}", "{h}-of-{p}-swarm"),
            ("{p}-{h}",),
        ),
        "river_on": (
            ("{h}-on-the-{r}",),
            ("{h}-on-the-{r}", "{r}-{h}"),
            ("{r}-{h}",),
        ),
    },
    person_first=(),
    river_roots=(),
    proper=tongue,
    founder_gloss="{h} of the {p} swarm",
    particles=frozenset({"of", "the", "on"}),
)
