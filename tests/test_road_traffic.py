"""The road traffic `InterurbanRoadStage` routes, kept on the world rather than thrown away.

The tiers are cut from it, so a world always had it for a moment; what is new is that it
stays: journeys per edge on `RoadEdge.traffic`, per hex on `Hex.traffic`, and every
journey with its path on `WorldState.journeys` for the stages after the roads. Keeping it
must change nothing else, and it must survive a save.
"""

import pytest

from tests.worlds import build_pipeline, build_world
from worldgen.core.world_state import WorldState
from worldgen.export import json_export

_KW = dict(seed=11, width=64, height=64, model="organic", regional_climate="temperate")


@pytest.fixture(scope="module")
def routed():
    return build_world(until="InterurbanRoadStage", **_KW)


def test_the_roads_keep_their_traffic(routed):
    carried = [e.traffic for e in routed.road_edges.values()]
    assert carried and max(carried) > 0.0
    assert all(t >= 0.0 for t in carried)
    assert any(h.traffic > 0.0 for h in routed.hexes.values())


def test_every_journey_starts_and_ends_at_a_settlement(routed):
    seats = {s.coord for s in routed.settlements}
    assert routed.journeys
    for (a, b), (n, path) in routed.journeys.items():
        assert n > 0.0 and a in seats and b in seats
        assert {path[0], path[-1]} == {a, b}


def test_the_journeys_add_up_to_the_traffic_on_each_hex(routed):
    """The kept journeys are the traffic, not a second account of it."""
    through: dict[tuple[int, int], float] = {}
    for n, path in routed.journeys.values():
        for c in path:
            through[c] = through.get(c, 0.0) + n
    for coord, hx in routed.hexes.items():
        assert hx.traffic == pytest.approx(through.get(coord, 0.0), abs=1e-2)


def test_traffic_survives_a_json_round_trip(routed, tmp_path):
    path = tmp_path / "w.json"
    json_export.save(routed, path)
    back = json_export.load(path)
    assert {k: e.traffic for k, e in back.road_edges.items()} == {
        k: e.traffic for k, e in routed.road_edges.items()
    }
    assert {c: h.traffic for c, h in back.hexes.items()} == {
        c: h.traffic for c, h in routed.hexes.items()
    }


def test_a_world_saved_before_traffic_was_kept_still_loads(routed):
    data = routed.to_dict()
    for h in data["hexes"]:
        del h["traffic"]
    for e in data["road_edges"] + data["sea_edges"]:
        del e["traffic"]
    back = WorldState.from_dict(data)
    assert all(h.traffic == 0.0 for h in back.hexes.values())
    assert all(e.traffic == 0.0 for e in back.road_edges.values())


def test_same_seed_same_traffic():
    a = build_pipeline(until="InterurbanRoadStage", **_KW).run()
    b = build_pipeline(until="InterurbanRoadStage", **_KW).run()
    assert a.to_dict() == b.to_dict()
    assert a.journeys == b.journeys
