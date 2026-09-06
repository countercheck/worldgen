/**
 * How a campaign is drawn.
 *
 * `palette.ts` is generated from the Python renderer and is the *default* look, not the
 * only one. A campaign may override any part of it — a referee running a winter scenario,
 * or one who simply wants their own land-use colours, should not need a code change.
 *
 * Nothing outside this module should import `palette.ts` directly. Renderers take a
 * `Theme`, which makes per-campaign overrides a matter of passing a different value
 * rather than of threading a flag through the drawing code.
 *
 * Future work: imported tilesets. A tileset would supply sprites rather than flat colours,
 * which means `Theme` grows a per-key image reference alongside `color` and renderers
 * learn to prefer one. Deliberately not modelled yet — the shape of that depends on what
 * a tileset turns out to look like, and guessing it now would be a fiction to work around
 * later. What matters today is that the colours are a value the campaign owns rather than
 * a constant the renderer reaches for.
 */

import {
  BIOME_COLORS,
  FOG_COLOR,
  LAND_COVER_COLORS,
  LAND_USE_COLORS,
  ROAD_STYLE,
  SOIL_COLORS,
  TERRAIN_COLORS,
  type Color,
} from './palette.js';

export interface RoadStyle {
  readonly color: Color;
  readonly width: number;
  readonly dash?: readonly number[];
}

/** A palette keyed by the enum values `world.json` carries. */
export type ColorMap = Readonly<Record<string, Color>>;

export interface Theme {
  readonly terrain: ColorMap;
  readonly biome: ColorMap;
  readonly landCover: ColorMap;
  readonly soil: ColorMap;
  readonly landUse: ColorMap;
  readonly road: Readonly<Record<string, RoadStyle>>;
  /** Ground a faction has never observed. */
  readonly fog: Color;
  /** Drawn when a key is missing, so a gap is visible rather than invisible. */
  readonly fallback: Color;
}

/** The look generated from `worldgen/render/debug_viewer.py`. */
export const DEFAULT_THEME: Theme = {
  terrain: TERRAIN_COLORS,
  biome: BIOME_COLORS,
  landCover: LAND_COVER_COLORS,
  soil: SOIL_COLORS,
  landUse: LAND_USE_COLORS,
  road: ROAD_STYLE,
  fog: FOG_COLOR,
  // Magenta on purpose. A missing colour should look like a bug, not like terrain — a
  // tasteful grey fallback is how an incomplete palette ships unnoticed.
  fallback: '#ff00ff',
};

/** A campaign's overrides: any subset of any palette. */
export interface ThemeOverrides {
  readonly terrain?: ColorMap;
  readonly biome?: ColorMap;
  readonly landCover?: ColorMap;
  readonly soil?: ColorMap;
  readonly landUse?: ColorMap;
  readonly road?: Readonly<Record<string, RoadStyle>>;
  readonly fog?: Color;
  readonly fallback?: Color;
}

/**
 * Merge overrides onto the default theme.
 *
 * Merged per key rather than per palette: overriding one land-use colour should not
 * require restating the other four, which is the difference between a usable override
 * and one nobody uses.
 */
export function resolveTheme(overrides?: ThemeOverrides): Theme {
  if (overrides === undefined) return DEFAULT_THEME;
  return {
    terrain: { ...DEFAULT_THEME.terrain, ...overrides.terrain },
    biome: { ...DEFAULT_THEME.biome, ...overrides.biome },
    landCover: { ...DEFAULT_THEME.landCover, ...overrides.landCover },
    soil: { ...DEFAULT_THEME.soil, ...overrides.soil },
    landUse: { ...DEFAULT_THEME.landUse, ...overrides.landUse },
    road: { ...DEFAULT_THEME.road, ...overrides.road },
    fog: overrides.fog ?? DEFAULT_THEME.fog,
    fallback: overrides.fallback ?? DEFAULT_THEME.fallback,
  };
}

/** Look a colour up, falling back visibly rather than silently. */
export const colorFor = (map: ColorMap, key: string | null, theme: Theme): Color =>
  (key === null ? undefined : map[key]) ?? theme.fallback;
