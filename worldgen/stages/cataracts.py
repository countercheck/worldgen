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
from .crossings import channel_drop_m
from .haulage import carries_a_barge
from .road_cost import is_river


class CataractStage(GeneratorStage):
    def run(self, state: WorldState) -> WorldState:
        cfg = self.config
        hexes = state.hexes
        # Measured before any tag is set, so one reach's cataract cannot change how the
        # next one reads: the drop is a fact about the ground, not about the tagging order.
        falls = set()
        rapids = set()
        for coord, hx in hexes.items():
            if not is_river(hx):
                continue
            drop = channel_drop_m(hx, hexes, cfg)
            if (
                cfg.cataract_min_drop_m > 0
                and carries_a_barge(hx, cfg)
                and drop >= cfg.cataract_min_drop_m
            ):
                falls.add(coord)
                continue
            if (
                cfg.rapids_min_drop_m > 0
                and hx.catchment_km2 >= cfg.rapids_min_catchment_km2
                and drop >= cfg.rapids_min_drop_m
            ):
                rapids.add(coord)
        for coord in falls:
            hexes[coord].tags.add("cataract")
        for coord in rapids:
            hexes[coord].tags.add("rapids")
        return state


__all__ = ["CataractStage"]
