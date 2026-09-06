"""Unseen ground must draw as absent, not as terrain.

A masked world keeps every hex, because `WorldState.from_dict` requires six fields on each
and because every renderer sizes its canvas from the hexes present — omitting them would
fail to load and would give each faction a differently-scaled map. So an unseen hex is
present but blank: elevation zero, no biome, no soil, and the `fog` tag.

Without a branch for that tag the renderers read those defaults as measurements and draw
flat green land at sea level. That is the worst kind of wrong: the file loads, every
structural assertion passes, and the map looks entirely plausible while showing grassland
where there may be a mountain range or an ocean. It was caught by looking at the output,
not by a test, which is why these exist.

`worldgen` has no other use for the tag — it is written only by the campaign layer — but
the rendering of it belongs here, with the palettes and the exporters that share them.
"""

import pytest

from worldgen.core.hex import Hex, TerrainClass
from worldgen.core.world_state import WorldState
from worldgen.export import legend, png_export, svg_export
from worldgen.render import debug_viewer
from worldgen.render.debug_viewer import FOG_COLOR, FOG_TAG, is_fog


def _world(fog_coords=(), size=4) -> WorldState:
    ws = WorldState.empty(seed=1, width=size, height=size)
    for q in range(size):
        for r in range(size):
            hx = Hex(coord=(q, r))
            hx.elevation = 250.0
            hx.terrain_class = TerrainClass.LAND
            if (q, r) in fog_coords:
                hx.elevation = 0.0
                hx.tags.add(FOG_TAG)
            ws.hexes[(q, r)] = hx
    return ws


def test_is_fog_reads_the_tag():
    ws = _world(fog_coords={(0, 0)})
    assert is_fog(ws.hexes[(0, 0)])
    assert not is_fog(ws.hexes[(1, 1)])


@pytest.mark.parametrize("color_mode", ["biome", "terrain", "land_cover", "soil", "land_use"])
def test_the_png_exporter_draws_fog_as_fog(color_mode):
    """Whatever attribute is being drawn. An unseen hex has no value for any of them."""
    ws = _world(fog_coords={(0, 0)})
    expected = png_export._rgb_int(*FOG_COLOR)

    assert png_export._get_hex_fill(ws.hexes[(0, 0)], color_mode) == expected
    assert png_export._get_hex_fill(ws.hexes[(1, 1)], color_mode) != expected


@pytest.mark.parametrize("color_mode", ["biome", "terrain", "land_cover", "soil", "land_use"])
def test_the_svg_exporter_draws_fog_as_fog(color_mode):
    ws = _world(fog_coords={(0, 0)})
    expected = svg_export._rgb_to_hex(*FOG_COLOR)

    assert svg_export._get_hex_fill(ws.hexes[(0, 0)], color_mode) == expected
    assert svg_export._get_hex_fill(ws.hexes[(1, 1)], color_mode) != expected


def test_the_elevation_ramp_does_not_read_a_fog_hex_as_sea_level():
    """The subtlest case. A continuous ramp has no 'unknown' end, so an unseen hex would
    otherwise paint as the lowest ground on the map and read as a valley floor."""
    ws = _world(fog_coords={(0, 0)})
    fog = png_export._get_hex_fill(ws.hexes[(0, 0)], "elevation")
    assert fog == png_export._rgb_int(*FOG_COLOR)


def test_the_debug_viewer_draws_fog_as_fog():
    ws = _world(fog_coords={(0, 0), (1, 0)})
    svg = debug_viewer.render_svg(ws, "biome")
    expected = debug_viewer._rgb_to_hex(*FOG_COLOR)
    assert expected in svg


def test_fog_is_not_a_terrain_category_in_the_legend():
    """It would otherwise enter the key as whatever its defaults happen to label — "Flat",
    on a map that is mostly unexplored."""
    ws = _world(fog_coords={(0, 0)})
    labels = [r.label for r in legend.rows(ws, "biome", {"terrain"})]
    assert "Unseen" in labels
    assert labels[-1] == "Unseen", (
        "unseen ground is the absence of the categories, so it sorts last"
    )


def test_a_world_with_no_fog_gains_no_unseen_row():
    """An ordinary export must be unchanged by any of this."""
    ws = _world()
    labels = [r.label for r in legend.rows(ws, "biome", {"terrain"})]
    assert "Unseen" not in labels


def test_a_fully_unseen_map_still_renders():
    """A commander who has scouted nothing still gets a map, rather than an exception."""
    coords = {(q, r) for q in range(4) for r in range(4)}
    ws = _world(fog_coords=coords)

    img = png_export.render(ws, png_export.PNGConfig(hex_size=10))
    assert img.width > 0 and img.height > 0

    svg = svg_export.render(ws, svg_export.SVGConfig(hex_size=10))
    assert "<svg" in svg


def test_masking_does_not_change_the_canvas_size():
    """The reason unseen hexes are kept rather than dropped: two commanders' maps, and the
    referee's, have to lie on top of one another."""
    full = _world()
    masked = _world(fog_coords={(0, 0), (1, 1), (2, 2)})

    cfg = png_export.PNGConfig(hex_size=10)
    assert png_export.render(full, cfg).size == png_export.render(masked, cfg).size
