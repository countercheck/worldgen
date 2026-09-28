"""A naming language invented from a seed.

After Martin O'Leary's naming languages for his fantasy maps: a language is a sound
inventory, a syllable shape and a way of spelling, and it makes one short word for each
meaning the first time it is asked and the same word every time after. That last property
is the whole trick. "Ford" is the same syllable in every ford town, so a map's names share
parts the way real ones do, and a reader starts to hear a language rather than noise.

Sounds are held as single characters from a small phonetic alphabet — `ʃ` for the sound
in *ship*, `ʧ` for *church*, `ŋ` for *sing*, capitals for long vowels — and only turned
into letters at the end, by a spelling table the language picks. The same sounds spelt
`sh` or `š` or `sch` read as three different peoples, which is cheap variety.
"""

import re
import zlib
from dataclasses import dataclass, field

import numpy as np

from .culture import Qualifier

_CONSONANTS = (
    "ptkmnls",
    "ptkbdgmnlrsʃzʧ",
    "ptkmnh",
    "hklmnpw",
    "ptkqvsgrmnŋlj",
    "tksʃdbqɣxmnlrwj",
    "tkdgmnsʃ",
    "ptkbdgmnszʒʧhjw",
)
_SIBILANTS = ("s", "sʃ", "sʃf")
_LIQUIDS = ("rl", "r", "l", "wj", "rlwj")
_FINALS = ("mn", "sk", "mnŋ", "sʃzʒ")
_VOWELS = ("aeiou", "aiu", "aeiouAEI", "aeiouU", "aiuAI", "eou", "aeiouAOU")
# C consonant, V vowel, S sibilant, L liquid, F final; `?` makes the one before optional.
_STRUCTURES = (
    "CVC",
    "CVV?C",
    "CVVC?",
    "CVC?",
    "CV",
    "VC",
    "CVF",
    "C?VC",
    "CVF?",
    "CL?VC",
    "CL?VF",
    "S?CVC",
    "S?CVF",
    "S?CVC?",
    "C?VF",
    "C?VC?",
    "C?VF?",
    "C?L?VC",
    "CVL?C?",
    "C?VL?C",
    "C?VLC?",
)
_CONSONANT_SPELLINGS = (
    {"ʃ": "sh", "ʒ": "zh", "ʧ": "ch", "ŋ": "ng", "j": "y", "x": "kh", "ɣ": "gh"},
    {"ʃ": "š", "ʒ": "ž", "ʧ": "č", "ŋ": "ng", "j": "j", "x": "h", "ɣ": "gh"},
    {"ʃ": "sch", "ʒ": "zh", "ʧ": "tsch", "ŋ": "ng", "j": "j", "x": "ch", "ɣ": "gh"},
    {"ʃ": "ch", "ʒ": "j", "ʧ": "tch", "ŋ": "ng", "j": "y", "x": "kh", "ɣ": "gh"},
    {"ʃ": "sh", "ʒ": "j", "ʧ": "tch", "ŋ": "ng", "j": "y", "x": "ch", "ɣ": "g"},
)
_VOWEL_SPELLINGS = (
    {"A": "á", "E": "é", "I": "í", "O": "ó", "U": "ú"},
    {"A": "aa", "E": "ee", "I": "ii", "O": "oo", "U": "uu"},
    {"A": "ä", "E": "ë", "I": "ï", "O": "ö", "U": "ü"},
    {"A": "â", "E": "ê", "I": "î", "O": "ô", "U": "û"},
    {"A": "au", "E": "ei", "I": "ie", "O": "ou", "U": "oo"},
)
_VOWEL_SOUNDS = frozenset("aeiouAEIOU")
# Sequences no language here allows inside a word: a doubled sound, two sibilants, two
# liquids. They read as typos rather than as a foreign language.
_FORBIDDEN = re.compile(r"(.)\1|[sʃfzʒ][sʃzʒ]|[rl][rl]")

_BARE_WORD_MAX_TRIES = 200


@dataclass
class Language:
    """One invented naming language. Build with `Language.generate`."""

    seed: int
    consonants: str
    sibilants: str
    liquids: str
    finals: str
    vowels: str
    structure: str
    spelling: dict[str, str]
    # How often a word is two syllables rather than one. Low in a language of long
    # syllables, where one is already plenty.
    long_words: float
    # Whether the head of a compound comes last (Ash-ford) or first (Aber-ystwyth).
    head_final: bool
    # What joins the parts of a compound: nothing, a hyphen, or a space.
    joiner: str
    # How often a name is a phrase — "Ford of the Oxen" — rather than a compound.
    phrase_rate: float
    # How often a compound wears down where its parts meet, as Asceford became Ashford.
    wear_rate: float
    # Whether a bare name takes an article — "the Ford" — or stands alone.
    articles: bool
    name: str = ""
    _words: dict[str, str] = field(default_factory=dict, repr=False)
    _taken: set[str] = field(default_factory=set, repr=False)

    @classmethod
    def generate(cls, rng: np.random.Generator) -> "Language":
        """Invent a language. Everything about it is drawn from *rng* and nothing else."""

        def pick(options):
            return options[int(rng.integers(len(options)))]

        def shuffled(sounds: str) -> str:
            # Order is frequency: earlier sounds are drawn more often, so shuffling is what
            # makes two languages with one inventory still sound unalike.
            return "".join(rng.permutation(list(sounds)))

        spelling = {**pick(_CONSONANT_SPELLINGS), **pick(_VOWEL_SPELLINGS)}
        lang = cls(
            seed=int(rng.integers(0, 2**32)),
            consonants=shuffled(pick(_CONSONANTS)),
            sibilants=shuffled(pick(_SIBILANTS)),
            liquids=shuffled(pick(_LIQUIDS)),
            finals=shuffled(pick(_FINALS)),
            vowels=shuffled(pick(_VOWELS)),
            structure=pick(_STRUCTURES),
            spelling=spelling,
            long_words=float(rng.uniform(0.0, 0.35)),
            head_final=bool(rng.random() < 0.6),
            joiner=pick(("", "", "", "-", " ")),
            phrase_rate=float(rng.uniform(0.0, 0.3)),
            wear_rate=float(rng.uniform(0.1, 0.6)),
            articles=bool(rng.random() < 0.3),
        )
        lang.name = lang.proper_name("people", 2)
        return lang

    # -- sounds ------------------------------------------------------------------------

    def _syllable(self, rng: np.random.Generator) -> str:
        sets = {
            "C": self.consonants,
            "V": self.vowels,
            "S": self.sibilants,
            "L": self.liquids,
            "F": self.finals,
        }
        out = []
        s = self.structure
        for i, ch in enumerate(s):
            if ch == "?":
                continue
            if i + 1 < len(s) and s[i + 1] == "?" and rng.random() < 0.5:
                continue
            sounds = sets[ch]
            # Squaring a uniform draw favours the front of the list: a few common sounds
            # and a tail of rare ones, which is how real inventories are used.
            out.append(sounds[int(rng.random() ** 2 * len(sounds))])
        return "".join(out)

    def _fresh(self, rng: np.random.Generator, syllables: int, min_sounds: int) -> str:
        """A new word of *syllables* syllables that breaks no rule and is not yet taken.

        *min_sounds* keeps out the one-vowel word a vowel-initial syllable shape can make:
        a word for "ford" may be two sounds, but a river called "U" reads as a typo.
        """
        for attempt in range(_BARE_WORD_MAX_TRIES):
            # A small inventory runs out of one-syllable words fast — a CV language of
            # seven consonants and three vowels has twenty-one — so a word that keeps
            # colliding is allowed to grow.
            n = syllables + attempt // 40
            word = "".join(self._syllable(rng) for _ in range(n))
            if len(word) >= min_sounds and not _FORBIDDEN.search(word) and word not in self._taken:
                return word
        raise RuntimeError(f"{self.name or 'language'} could not make a new word")

    def word(self, meaning: str) -> str:
        """This language's word for *meaning*, in sounds. The same meaning, the same word.

        Seeded from the language and the meaning, not from call order, so asking for
        "ford" first or last gives the same ford — except where two meanings would land
        on one word, when whichever came second takes its next draw.
        """
        if meaning not in self._words:
            rng = np.random.default_rng([self.seed, zlib.crc32(meaning.encode())])
            syllables = 2 if rng.random() < self.long_words else 1
            word = self._fresh(rng, syllables, min_sounds=2)
            self._words[meaning] = word
            self._taken.add(word)
        return self._words[meaning]

    def spell(self, sounds: str) -> str:
        return "".join(self.spelling.get(ch, ch) for ch in sounds)

    # -- names -------------------------------------------------------------------------

    def _compound(self, first: str, second: str, rng: np.random.Generator) -> str:
        """Two words run together, in sounds, worn where they meet if this one wears."""
        if rng.random() < self.wear_rate and len(first) > 2 and first[-1] in _VOWEL_SOUNDS:
            first = first[:-1]
        if first and second and first[-1] == second[0]:
            second = second[1:]
        return first + second

    def _title(self, words: list[str]) -> str:
        """Capitalise a name, leaving its particles in lower case after the first word."""
        particles = {self.spell(self.word(k)) for k in ("of", "on", "the")}
        out = []
        for i, w in enumerate(words):
            out.append(w if i and w in particles else w[:1].upper() + w[1:])
        return " ".join(out)

    def place_name(
        self,
        generic: str,
        qualifier: Qualifier | None,
        rng: np.random.Generator,
        rank: int = 1,
    ) -> str:
        head = self.word(generic)
        if qualifier is None:
            words = [self.spell(head)]
            if self.articles:
                words.insert(0, self.spell(self.word("the")))
            return self._title(words)

        if qualifier.kind == "meaning":
            spec = self.word(qualifier.value)
            if rng.random() < self.phrase_rate:
                return self._title(
                    [self.spell(head), self.spell(self.word("of")), self.spell(spec)]
                )
            first, second = (spec, head) if self.head_final else (head, spec)
            if self.joiner:
                return self._title_joined(self.spell(first), self.spell(second))
            return self._title([self.spell(self._compound(first, second, rng))])

        # A proper name — a river's or a founder's — is already spelled, perhaps by another
        # people, so it is used as it stands rather than respelled.
        proper = qualifier.value
        particle = "on" if qualifier.relation == "on" else "of"
        if rng.random() < max(self.phrase_rate, 0.35 if particle == "on" else 0.0):
            return self._title([self.spell(head), self.spell(self.word(particle)), proper])
        if self.head_final:
            return self._title_joined(proper, self.spell(head))
        return self._title_joined(self.spell(head), proper)

    def _title_joined(self, first: str, second: str) -> str:
        if self.joiner == " ":
            return self._title([first, second])
        # Hyphenated parts each keep a capital (Stoke-Poges); run together, only the first.
        second = second[:1].upper() + second[1:] if self.joiner else second.lower()
        return first[:1].upper() + first[1:] + self.joiner + second

    def proper_name(self, key: str, syllables: int, rank: int = 1) -> str:
        return self.spell(self.word_of_length(key, syllables)).capitalize()

    def word_of_length(self, key: str, syllables: int) -> str:
        """A meaningless root of a set length, stable per key, like `word`."""
        tagged = f"{key}#{syllables}"
        if tagged not in self._words:
            rng = np.random.default_rng([self.seed, zlib.crc32(tagged.encode())])
            word = self._fresh(rng, syllables, min_sounds=3)
            self._words[tagged] = word
            self._taken.add(word)
        return self._words[tagged]

    def alphabet(self) -> set[str]:
        """Every letter this language can write, for checking that it wrote nothing else."""
        sounds = self.consonants + self.sibilants + self.liquids + self.finals + self.vowels
        return {ch for s in sounds for ch in self.spell(s)} | {
            ch.upper() for s in sounds for ch in self.spell(s)
        }


__all__ = ["Language"]
