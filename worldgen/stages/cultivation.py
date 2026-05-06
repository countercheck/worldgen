from ..core.hex import LandCover, SettlementTier
from ..core.hex_grid import hex_range
from ..core.pipeline import GeneratorStage
from ..core.world_state import WorldState

RESISTANT = frozenset(
    {
        LandCover.BOG,
        LandCover.MARSH,
        LandCover.BARE_ROCK,
        LandCover.ALPINE,
        LandCover.TUNDRA,
        LandCover.DESERT,
        LandCover.OPEN_WATER,
    }
)


class CultivationStage(GeneratorStage):
    """Marks hexes as cultivated within clearing radii of cities and towns."""

    def run(self, state: WorldState) -> WorldState:
        hexes = state.hexes
        cfg = self.config
        radii = {
            SettlementTier.CITY: cfg.cultivation_city_radius,
            SettlementTier.TOWN: cfg.cultivation_town_radius,
        }
        for s in state.settlements:
            radius = radii.get(s.tier)
            if radius is None:
                continue
            for coord in hex_range(s.coord, radius):
                hx = hexes.get(coord)
                if hx is not None and hx.land_cover not in RESISTANT:
                    hx.cultivated = True
        return state


class VillageCultivationStage(GeneratorStage):
    """Marks hexes as cultivated within clearing radii of villages."""

    def run(self, state: WorldState) -> WorldState:
        hexes = state.hexes
        cfg = self.config
        radius = cfg.cultivation_village_radius
        for s in state.settlements:
            if s.tier != SettlementTier.VILLAGE:
                continue
            for coord in hex_range(s.coord, radius):
                hx = hexes.get(coord)
                if hx is not None and hx.land_cover not in RESISTANT:
                    hx.cultivated = True
        return state
