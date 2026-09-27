"""Name the rivers, then the places on them.

Last in both pipelines, and that is deliberate twice over. Tier, role and population are
not final until every settlement stage has run — promotion turns a market into a city, and
the resource stage founds ports and mines — so a name given earlier would be for the wrong
place. And the pipeline draws one child seed per stage, in order, so a stage appended at
the end leaves every seed before it alone: a world made before names existed comes out the
same ground, the same towns and the same roads, only named.

Rivers come first because towns borrow from them. The names are an older people's — the
substrate — when there is one, which is how England came to have Celtic rivers flowing
through English towns: newcomers name their own settlements and keep the names of the
water they found.

Placeholders are what the stages found settlements with, and `CityPromotionStage` still
edits them in place; this replaces them all at once at the end, rather than asking every
stage that founds a settlement to know about languages.
"""

import numpy as np

from ..core.hex import Hex, HexCoord, Settlement, SettlementTier, TerrainClass
from ..core.hex_grid import distance, neighbors
from ..core.pipeline import GeneratorStage
from ..core.world_state import WorldState
from ..naming import (
    GLOSS,
    Language,
    NameRegistry,
    Qualifier,
    add_direction,
    culture_regions,
    read_site,
)

_WATER = (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER)
_TIER_ORDER = {SettlementTier.CITY: 0, SettlementTier.TOWN: 1, SettlementTier.VILLAGE: 2}
# Heads that sit naturally before a river's name. A ford or a mouth is *of* a river; a
# farm on one is still a fair name, but a mine or a pass named for the river below it is
# not how anyone would put it.
_RIVER_HEADS = {
    "ford": 2.0,
    "bridge": 2.0,
    "mouth": 2.0,
    "falls": 2.0,
    "confluence": 1.5,
    "harbour": 1.0,
    "market": 1.0,
    "town": 1.0,
    "burgh": 1.0,
    "steading": 0.5,
    "spring": 1.0,
    "shore": 0.5,
}
# A bare head — the Ford, the Burgh — is a good name exactly once per language, and the
# registry will refuse the second; so it is offered, but seldom.
_BARE_WEIGHT = 0.2
_ATTEMPTS = 24


def _pick(options: dict, rng: np.random.Generator):
    """A key of *options* drawn in proportion to its weight. Keys are sorted first so the
    draw depends on the weights and the seed, never on dict insertion order."""
    keys = sorted(options, key=str)
    weights = np.array([options[k] for k in keys], dtype=float)
    total = weights.sum()
    if total <= 0:
        return keys[0]
    return keys[int(np.searchsorted(np.cumsum(weights) / total, rng.random(), side="right"))]


def _letters(name: str) -> int:
    return sum(ch.isalpha() for ch in name)


class NamingStage(GeneratorStage):
    def run(self, state: WorldState) -> WorldState:
        cfg = self.config
        if cfg.naming_cultures <= 0:
            return state
        rng = self.rng

        cultures = self._languages(cfg.naming_cultures, rng)
        substrate = (
            self._languages(1, rng, taken={c.name for c in cultures})[0]
            if (cfg.naming_substrate)
            else None
        )
        regions = culture_regions(
            state.hexes,
            len(cultures),
            rng,
            cfg.naming_region_climb_m,
            cfg.naming_region_river_cost,
            cfg.naming_great_river_km2,
            cfg.naming_region_water_cost,
        )
        registry = NameRegistry(cfg.naming_min_edit_distance)

        river_at = self._name_rivers(state, cultures, substrate, regions, registry, rng)
        anchors = self._anchors(state.settlements, cfg.naming_direction_radius)
        for s in sorted(
            state.settlements, key=lambda s: (_TIER_ORDER[s.tier], -s.population, s.coord)
        ):
            culture = cultures[regions.get(s.coord, 0)]
            self._name_settlement(
                state.hexes, s, culture, river_at, anchors.get(s.coord), registry, rng
            )

        state.metadata["cultures"] = [{"name": c.name, "role": "regional"} for c in cultures] + (
            [{"name": substrate.name, "role": "substrate"}] if substrate else []
        )
        return state

    @staticmethod
    def _languages(n: int, rng: np.random.Generator, taken: set[str] | None = None):
        """*n* languages, no two sharing a name, so a culture can be looked up by it."""
        taken = set(taken or ())
        out = []
        while len(out) < n:
            lang = Language.generate(rng)
            if lang.name in taken:
                lang.name = lang.proper_name("people", 2)
            if lang.name in taken:
                continue
            taken.add(lang.name)
            out.append(lang)
        return out

    def _name_rivers(self, state, cultures, substrate, regions, registry, rng):
        """Name every river big enough, largest first, and index the names by hex.

        Largest first so the great rivers get the short names: in real languages the big
        rivers' names are the oldest, and worn down to a syllable or two.
        """
        cfg = self.config
        hexes = state.hexes

        def mouth(river) -> tuple[float, HexCoord | None]:
            land = [c for c in river.hexes if c in hexes and hexes[c].terrain_class not in _WATER]
            if not land:
                return 0.0, None
            last = max(land, key=lambda c: (hexes[c].catchment_km2, c))
            return hexes[last].catchment_km2, last

        sized = [(mouth(r), i, r) for i, r in enumerate(state.rivers)]
        sized = [x for x in sized if x[0][1] is not None]
        sized.sort(key=lambda x: (-x[0][0], x[0][1]))

        river_at: dict[HexCoord, tuple[float, str]] = {}
        for (catchment, last), i, river in sized:
            if catchment < cfg.naming_river_min_catchment_km2:
                break
            namer = substrate or cultures[regions.get(last, 0)]
            # A great river's name is the oldest on the map and worn shortest; a lesser one
            # sometimes runs to three syllables.
            syllables = (
                2 if catchment >= cfg.naming_great_river_km2 else 2 + int(rng.random() < 0.3)
            )
            name = ""
            for attempt in range(_ATTEMPTS):
                candidate = namer.proper_name(f"river:{i}:{attempt}", syllables + attempt // 8)
                if registry.accepts(candidate):
                    name = candidate
                    break
            if not name:
                continue
            registry.add(name)
            river.name = name
            for c in river.hexes:
                on_land = c in hexes and hexes[c].terrain_class not in _WATER
                if on_land and catchment > river_at.get(c, (-1.0, ""))[0]:
                    river_at[c] = (catchment, name)
        return river_at

    @staticmethod
    def _anchors(settlements: list[Settlement], radius: int) -> dict[HexCoord, HexCoord]:
        """For each settlement, the nearest bigger one within *radius*, if any.

        What a place is north or west *of*: a village is placed against the town it
        markets in, a town against the city, and a city against nothing.
        """
        out = {}
        for s in settlements:
            bigger = [
                (distance(s.coord, b.coord), b.coord)
                for b in settlements
                if _TIER_ORDER[b.tier] < _TIER_ORDER[s.tier]
                and distance(s.coord, b.coord) <= radius
            ]
            if bigger:
                out[s.coord] = min(bigger)[1]
        return out

    @staticmethod
    def _river_near(coord: HexCoord, river_at) -> str | None:
        """The biggest named river through this hex or beside it."""
        options = [river_at[c] for c in (coord, *neighbors(coord)) if c in river_at]
        return max(options)[1] if options else None

    def _name_settlement(
        self,
        hexes: dict[HexCoord, Hex],
        s: Settlement,
        culture: Language,
        river_at,
        anchor: HexCoord | None,
        registry: NameRegistry,
        rng: np.random.Generator,
    ) -> None:
        cfg = self.config
        site = read_site(hexes, s, cfg.naming_hill_relief_m, cfg.naming_high_elevation_m)
        if anchor is not None:
            add_direction(site, anchor)
        river = self._river_near(s.coord, river_at)
        # Squared, so a site's most striking feature wins most of the time without always
        # winning: a map of towns each named for its single best feature is monotonous.
        heads = {k: w * w for k, w in site.generics.items()}

        for attempt in range(_ATTEMPTS):
            generic = _pick(heads, rng)
            options: dict = {("bare", ""): _BARE_WEIGHT, ("founder", ""): cfg.naming_founder_weight}
            for key, weight in site.specifics.items():
                options[("meaning", key)] = weight
            if river and generic in _RIVER_HEADS:
                options[("river", river)] = cfg.naming_river_weight * _RIVER_HEADS[generic]
            kind, value = _pick(options, rng)

            if kind == "bare":
                qualifier, etymology = None, f"the {GLOSS[generic]}"
            elif kind == "meaning":
                qualifier = Qualifier("meaning", value)
                etymology = f"{GLOSS[value]} {GLOSS[generic]}"
            elif kind == "river":
                qualifier = Qualifier("proper", value, "on")
                etymology = f"{GLOSS[generic]} on the {value}"
            else:
                # One syllable first, as the names buried in -ingham and -by mostly were;
                # a second only if the short ones keep colliding.
                founder = culture.proper_name(f"person:{s.coord}:{attempt}", 1 + attempt // 8)
                qualifier = Qualifier("proper", founder, "of")
                etymology = f"{founder}'s {GLOSS[generic]}"

            name = culture.place_name(generic, qualifier, rng)
            if _letters(name) <= cfg.naming_max_letters and registry.accepts(name):
                break
        else:
            # Every draw collided or ran long. A founder's name grows a syllable at a time
            # until it is unique, which always ends: there are more long names than towns.
            generic = max(site.generics, key=lambda k: (site.generics[k], k))
            syllables = 1
            while True:
                founder = culture.proper_name(f"person:{s.coord}:fallback", syllables)
                name = culture.place_name(generic, Qualifier("proper", founder, "of"), rng)
                if name not in registry:
                    etymology = f"{founder}'s {GLOSS[generic]}"
                    break
                syllables += 1

        registry.add(name)
        s.name = name
        s.culture = culture.name
        s.etymology = etymology


__all__ = ["NamingStage"]
