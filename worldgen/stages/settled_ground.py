"""The ground a settlement stands on is not wild.

`LandCoverStage` describes the country before anybody works it, and `LandUseStage` clears
the fields a market's hinterland ploughs and grazes; neither touches the hex a town is
built on, so a city of ten thousand stood in dense forest. This clears it: every town and
village hex, and for a city the ring of market gardens, orchards and paddocks round its
walls too (`settled_clear_radius_city`).

A lumber camp keeps its trees — they are what it is there for. So does wetland and bare
rock under a settlement: a town on a fen is built on the fen, and there is nothing to
clear on scree.

Its own stage, and the last before names, because the settlements are not all founded
until then: `CityPromotionStage` makes cities, `ResourceStage` founds ports and mines,
`ChokepointStage` the organic villages. Naming reads the cover, so it runs after this and
names what stands there now. It draws nothing from its generator, and adding it moved only
the naming stage's seed.
"""

from ..core.hex import LandCover, LandUse, SettlementRole, SettlementTier, TerrainClass
from ..core.hex_grid import hex_range
from ..core.pipeline import GeneratorStage
from ..core.world_state import WorldState

_WATER = (TerrainClass.OPEN_WATER, TerrainClass.INLAND_WATER)
# The cover a settlement clears. Wetland and bare ground stay what they are.
_CLEARED = frozenset({LandCover.DENSE_FOREST, LandCover.WOODLAND, LandCover.SCRUB})
_PLOUGHABLE_USE = (LandUse.WOOD,)


class SettledGroundStage(GeneratorStage):
    def run(self, state: WorldState) -> WorldState:
        hexes = state.hexes
        cfg = self.config
        for s in state.settlements:
            if s.role is SettlementRole.LUMBER:
                continue
            radius = cfg.settled_clear_radius_city if s.tier is SettlementTier.CITY else 0
            for coord in hex_range(s.coord, radius):
                hx = hexes.get(coord)
                if hx is None or hx.terrain_class in _WATER:
                    continue
                if hx.land_cover in _CLEARED:
                    hx.land_cover = LandCover.OPEN
                # In the organic model the use goes with it: wood round a city is grazed,
                # its owners' paddocks. The classic model assigns no land use.
                if hx.land_use in _PLOUGHABLE_USE:
                    hx.land_use = LandUse.PASTURE
        return state


__all__ = ["SettledGroundStage"]
