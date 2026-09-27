"""The pickaxe and the axe: one shape, drawn by every renderer, and listed in the legend."""

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
