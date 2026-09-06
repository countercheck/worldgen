/**
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


/** Presentation bands from `terrain_label()`, not the stored `TerrainClass`. */
export const TERRAIN_COLORS: Readonly<Record<string, Color>> = {
  'ocean': '#3366cc',
  'lake': '#5999d9',
  'coast': '#e6cc66',
  'flat': '#66cc66',
  'rolling': '#b2994c',
  'steep': '#808080',
  'escarpment': '#524a47',
};

/** Biome fills. */
export const BIOME_COLORS: Readonly<Record<string, Color>> = {
  'tundra': '#e6f2f2',
  'boreal': '#4c804c',
  'temperate_forest': '#339933',
  'grassland': '#99cc4c',
  'shrubland': '#ccb24c',
  'desert': '#f2e699',
  'tropical': '#1a801a',
  'wetland': '#669966',
  'ocean': '#3366cc',
  'alpine': '#b2b2b2',
};

/** What grows on the ground. */
export const LAND_COVER_COLORS: Readonly<Record<string, Color>> = {
  'open_water': '#4169e1',
  'bog': '#556b2f',
  'marsh': '#6b8e6b',
  'dense_forest': '#1a4a1a',
  'woodland': '#3a7a3a',
  'scrub': '#8b7455',
  'open': '#c8d870',
  'tundra': '#b0c4c4',
  'desert': '#d2b48c',
  'alpine': '#a0a0a0',
  'bare_rock': '#606060',
};

/** A ranked ramp, buff through to alluvium. */
export const SOIL_COLORS: Readonly<Record<string, Color>> = {
  'unusable': '#e0daca',
  'grazing': '#c1cd9d',
  'marginal': '#96b176',
  'arable': '#817a51',
  'prime': '#5c472d',
};

/** Categorical, unranked. */
export const LAND_USE_COLORS: Readonly<Record<string, Color>> = {
  'water': '#4169e1',
  'waste': '#bdbab3',
  'wood': '#2e6638',
  'pasture': '#adcd8c',
  'arable': '#d4a44c',
};

/** Road styling by tier, matching the SVG exporter's strokes. */
export const ROAD_STYLE: Readonly<
  Record<string, { color: Color; width: number; dash?: number[] }>
> = {
  'primary': { color: '#5c3d1e', width: 2.0 },
  'secondary': { color: '#8b6914', width: 1.2 },
  'track': { color: '#b8a070', width: 0.6, dash: [4, 2] },
};

/** Fixed ramps, so two maps compare by eye. Deliberately not data-driven. */
export const TEMPERATURE_RAMP_C: readonly [number, number] = [-20.0, 35.0];
export const PRECIP_RAMP_MM: readonly [number, number] = [0.0, 2500.0];

/** Ground a faction has never observed. Owned here so nothing hand-writes it. */
export const FOG_COLOR: Color = '#17171c';
