"""The settlement icons: one shape each, drawn by every renderer, and listed in the legend."""

import re
from pathlib import Path

from PIL import Image, ImageDraw

from worldgen.core.hex import Settlement, SettlementRole, SettlementTier
from worldgen.core.world_state import WorldState
from worldgen.export import legend
from worldgen.export.png_export import _draw_settlement
from worldgen.export.svg_export import _settlement_marker
from worldgen.render import glyphs

_TS = Path(__file__).resolve().parent.parent / "campaign" / "client" / "src" / "map" / "glyphs.ts"


def _numbers(text: str) -> list[float]:
    return [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", text)]


def _ts_glyph(name: str) -> list[float]:
    block = re.search(rf"export const {name}: Glyph = \{{(.*?)\}};", _TS.read_text(), re.S)
    assert block, f"{name} is missing from {_TS.name}"
    return _numbers(block.group(1))


def test_the_campaign_client_draws_the_same_shapes():
    """The client cannot import the Python geometry, so it keeps a copy; this is the guard."""
    for name, glyph in (("AXE", glyphs.AXE), ("PICKAXE", glyphs.PICKAXE)):
        python = [v for p in (*glyph.handle, *glyph.head) for v in p] + [glyph.handle_width]
        assert _ts_glyph(name) == python, f"{name} differs between glyphs.py and glyphs.ts"


def test_the_campaign_client_draws_the_same_city():
    block = re.search(r"export const CITY: Emblem = \{(.*?)\};", _TS.read_text(), re.S)
    assert block, f"CITY is missing from {_TS.name}"
    python = [v for shape in glyphs.CITY.shapes for p in shape for v in p]
    assert _numbers(block.group(1)) == python, "CITY differs between glyphs.py and glyphs.ts"
    colours = {
        name: re.search(rf"export const {name} = '(#[0-9a-f]{{6}})'", _TS.read_text()).group(1)
        for name in ("CITY_DISC", "CITY_INK")
    }
    assert colours == {"CITY_DISC": glyphs.CITY_DISC, "CITY_INK": glyphs.CITY_INK}


def test_a_city_is_a_church_among_roofs_on_a_gold_disc():
    city = _settlement_marker(SettlementTier.CITY, 10, 10)
    assert glyphs.CITY_DISC in city and city.count("<polygon") == len(glyphs.CITY.shapes)
    img = Image.new("RGB", (60, 60), (200, 200, 200))
    _draw_settlement(ImageDraw.Draw(img), SettlementTier.CITY, 30, 30, scale=3)
    pixels = list(img.getdata())
    assert sum(1 for p in pixels if p == glyphs.CITY_DISC_RGB) > 50, "no gold disc"
    assert sum(1 for p in pixels if p == glyphs.CITY_INK_RGB) > 50, "no silhouette"


def test_a_mine_and_a_camp_are_drawn_as_their_tools():
    mine = _settlement_marker(SettlementTier.VILLAGE, 10, 10, role=SettlementRole.MINING)
    camp = _settlement_marker(SettlementTier.VILLAGE, 10, 10, role=SettlementRole.LUMBER)
    plain = _settlement_marker(SettlementTier.VILLAGE, 10, 10, role=SettlementRole.MARKET)
    assert "<polygon" in mine and "<line" in mine
    assert "<polygon" in camp and mine != camp
    assert plain.startswith("<circle") and "<polygon" not in plain


def test_the_png_exporter_puts_the_tool_in_the_disc():
    img = Image.new("RGB", (40, 40), (200, 200, 200))
    _draw_settlement(
        ImageDraw.Draw(img), SettlementTier.VILLAGE, 20, 20, scale=3, role=SettlementRole.MINING
    )
    dark = sum(1 for p in img.getdata() if max(p) < 90)
    assert dark > 20, "the pickaxe drew nothing inside its disc"


def test_the_legend_names_the_trades_present():
    ws = WorldState.empty(1, 4, 4, "axial")
    ws.settlements = [
        Settlement((0, 0), SettlementTier.VILLAGE, SettlementRole.MINING, 300, "m"),
        Settlement((1, 0), SettlementTier.TOWN, SettlementRole.MARKET, 900, "t"),
    ]
    labels = {r.label for r in legend.rows(ws, "terrain", {"settlements"}) if r.kind == "resource"}
    assert labels == {"Mine"}
