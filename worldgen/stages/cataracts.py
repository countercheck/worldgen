"""Where a river big enough for boats falls too fast to carry them.

A cataract is not a small river. It is a large one — enough water to float a barge — that
drops so steeply through a reach that nothing loaded can go up it and nothing sensible goes
down it. Three things follow from one, and they are the reason to mark it:

**It is a barrier.** `navigable` is false on a cataract, so a cargo afloat has to land
above it and load again below. The river is cut into navigable reaches, and moving along
it means leaving it.

**It makes a town.** Landing and loading again is a change of mode, which is what
`CityPromotionStage` pays a quay for. The portage at the falls is where the trade changes
hands, and `ResourceStage` founds a port there if nobody stands at it already — Aswan at
the First Cataract, Louisville at the Falls of the Ohio, the fall-line towns of the
American east coast.

**It turns a wheel.** A great fall of water is power, and mills were built at it before
they were built anywhere else. `site_bonus` pays a site beside one
`habitability_mill_bonus`.

**Rapids** are the same fall on a smaller river: white water, too fast to wade and too
rough to paddle, but on a river no barge was ever going to use. They are tagged `rapids`
for the map to draw, and change nothing else. A cataract is not also a rapid.

Tagged here, straight after hydrology, because `navigable` is read as early as
`HabitabilityStage` scoring harbours — a cataract has to exist before anyone asks whether a
boat can load there.
"""

from ..core.pipeline import GeneratorStage
from ..core.world_state import WorldState
from .haulage import catchment_carries_a_barge
from .hydrology import mirror_on_band
from .riverside import side_gradients


class CataractStage(GeneratorStage):
    def run(self, state: WorldState) -> WorldState:
        cfg = self.config
        # Measured before any tag is set, so one reach's cataract cannot change how the
        # next one reads: the fall is a fact about the ground, not about the tagging order.
        gradient = side_gradients(state)
        for side, rs in state.river_sides.items():
            fall = gradient.get(side, 0.0)
            if (
                cfg.cataract_min_drop_m > 0
                and catchment_carries_a_barge(rs.catchment_km2, cfg)
                and fall >= cfg.cataract_min_drop_m
            ):
                rs.tags.add("cataract")
            elif (
                cfg.rapids_min_drop_m > 0
                and rs.catchment_km2 >= cfg.rapids_min_catchment_km2
                and fall >= cfg.rapids_min_drop_m
            ):
                rs.tags.add("rapids")
        # Until navigation and the map read sides, each is mirrored onto the river's hex.
        mirror_on_band(state, ("cataract", "rapids"))
        return state


__all__ = ["CataractStage"]
