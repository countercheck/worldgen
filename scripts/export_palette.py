"""Emit the renderer's colour palettes as TypeScript, so the web map cannot drift.

`worldgen/render/debug_viewer.py` owns the colours, and both SVG and PNG export import
them from there. The browser map is a third renderer of the same worlds, and a map that
disagrees with the exports about what a marsh looks like is worse than one that is simply
ugly — a referee comparing the screen against a printed SVG would be reading two different
maps.

So the palette is generated rather than retyped. Regenerate with:

    python3 scripts/export_palette.py

`campaign/shared/test/palette.test.ts` asserts the committed output is complete and
well-formed; if the Python palette gains a value and this is not re-run, that test fails
rather than the browser quietly falling back to grey.
"""

from pathlib import Path

from worldgen.core.hex import Biome, LandCover, LandUse, SoilQuality, TerrainLabel
from worldgen.core.world_state import RoadTier
from worldgen.render.debug_viewer import (
    _ROAD_STYLE,
    BIOME_COLORS,
    LAND_COVER_COLORS,
    LAND_USE_COLORS,
    PRECIP_RAMP_MM,
    SOIL_COLORS,
    TEMPERATURE_RAMP_C,
    TERRAIN_COLORS,
)

OUT = Path(__file__).resolve().parent.parent / "campaign" / "shared" / "src" / "palette.ts"

HEADER = """/**
 * Colour palettes, generated from `worldgen/render/debug_viewer.py`.
 *
 * DO NOT EDIT. Regenerate with `python3 scripts/export_palette.py`.
 *
 * The Python renderer owns these, and the SVG and PNG exporters import them from it. The
 * browser map is a third renderer of the same worlds, so it reads the same values rather
 * than an approximation of them — otherwise a referee comparing the screen against a
 * printed export would be reading two maps that disagree about what a marsh looks like.
 */

/** A colour as a CSS hex string, e.g. `#33cc66`. */
export type Color = string;
"""


def to_hex(rgb: tuple[float, float, float]) -> str:
    """Matplotlib-style 0..1 floats to a CSS hex string."""
    return "#" + "".join(f"{round(c * 255):02x}" for c in rgb)


def emit(name: str, colors: dict, enum_cls, doc: str) -> str:
    """One palette as a TypeScript record keyed by the enum's serialised value.

    Keyed by `.value` rather than by the Python enum name, because the value is what
    `world.json` carries and therefore what the parser produces.
    """
    lines = [f"\n/** {doc} */", f"export const {name}: Readonly<Record<string, Color>> = {{"]
    for member in enum_cls:
        if member in colors:
            lines.append(f"  {member.value!r}: '{to_hex(colors[member])}',".replace('"', "'"))
    lines.append("};")
    return "\n".join(lines)


def render() -> str:
    """The whole palette module as a string. Kept separate from writing it so the test
    can compare against the committed file without shelling out."""
    parts = [HEADER]

    parts.append(
        emit(
            "TERRAIN_COLORS",
            TERRAIN_COLORS,
            TerrainLabel,
            "Presentation bands from `terrain_label()`, not the stored `TerrainClass`.",
        )
    )
    parts.append(emit("BIOME_COLORS", BIOME_COLORS, Biome, "Biome fills."))
    parts.append(
        emit("LAND_COVER_COLORS", LAND_COVER_COLORS, LandCover, "What grows on the ground.")
    )
    parts.append(
        emit("SOIL_COLORS", SOIL_COLORS, SoilQuality, "A ranked ramp, buff through to alluvium.")
    )
    parts.append(emit("LAND_USE_COLORS", LAND_USE_COLORS, LandUse, "Categorical, unranked."))

    road = ["\n/** Road styling by tier, matching the SVG exporter's strokes. */"]
    road.append("export const ROAD_STYLE: Readonly<")
    road.append("  Record<string, { color: Color; width: number; dash?: number[] }>")
    road.append("> = {")
    for tier in RoadTier:
        style = _ROAD_STYLE[tier]
        dash = style.get("dasharray")
        dash_part = ""
        if dash:
            dash_part = f", dash: [{', '.join(dash.split())}]"
        road.append(
            f"  '{tier.value}': {{ color: '{style['stroke']}', "
            f"width: {float(style['stroke-width'])}{dash_part} }},"
        )
    road.append("};")
    parts.append("\n".join(road))

    ramps = [
        "\n/** Fixed ramps, so two maps compare by eye. Deliberately not data-driven. */",
        "export const TEMPERATURE_RAMP_C: readonly [number, number] = "
        f"[{TEMPERATURE_RAMP_C[0]}, {TEMPERATURE_RAMP_C[1]}];",
        "export const PRECIP_RAMP_MM: readonly [number, number] = "
        f"[{PRECIP_RAMP_MM[0]}, {PRECIP_RAMP_MM[1]}];",
    ]
    parts.append("\n".join(ramps))

    parts.append(
        "\n/** Ground a faction has never observed. Owned here so nothing hand-writes it. */\n"
        "export const FOG_COLOR: Color = '#17171c';\n"
    )

    return "\n".join(parts)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
