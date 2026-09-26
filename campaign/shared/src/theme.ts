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

/**
 * One class of watercourse. `width` is a fraction of the hex radius.
 *
 * Keyed by the rules' Major/Minor split (`riverClass`) rather than by the generator's
 * flow rank, so the map shows the line that decides whether a column can ford — a river
 * drawn wide is one you cannot wade.
 */
export interface RiverStyle {
  readonly color: Color;
  readonly width: number;
}

/** A palette keyed by the enum values `world.json` carries. */
export type ColorMap = Readonly<Record<string, Color>>;

/**
 * The wash laid over ground nobody is currently watching.
 *
 * Distinct from `fog`, and the distinction is the whole point. `fog` is a *colour*, used
 * when `terrainFog` is on and the server has replaced a hex with a blank that carries no
 * biome and no elevation — there is nothing to draw, so something stands in for it. This
 * is a *veil* over terrain that is perfectly well known: the ground is on the map, but no
 * one of yours is looking at it, and what is standing there now is anybody's guess.
 *
 * The two compose. With `terrainFog` on, a hex can be both blank and unwatched, and it
 * should read as both. Reusing one for the other is how that stops being possible.
 *
 * Alphas rather than three colours, because the wash has to sit over snow and over forest
 * and stay legible on both; a flat opaque tone that suits one drowns the other.
 */
export interface WashStyle {
  /**
   * Desaturated blue-black rather than pure black.
   *
   * Black drains the hue out of what it covers and the map goes grey; a cold dark tone
   * leaves enough terrain colour showing to be read as terrain, which is the point —
   * unwatched ground is still ground you have marched over and mapped.
   */
  readonly color: Color;
  /** Under observation right now. Zero: nothing between the reader and the ground. */
  readonly observed: number;
  /** Marched over before, watched by nobody now. Terrain remembered, occupants not. */
  readonly surveyed: number;
  /** Never seen by anybody under their command. */
  readonly unseen: number;
}

export interface Theme {
  readonly terrain: ColorMap;
  readonly biome: ColorMap;
  readonly landCover: ColorMap;
  readonly soil: ColorMap;
  readonly landUse: ColorMap;
  readonly road: Readonly<Record<string, RoadStyle>>;
  readonly river: Readonly<Record<'major' | 'minor', RiverStyle>>;
  /**
   * A pale band drawn under every road. The road colours are browns, and a brown line on
   * dense forest or escarpment is close to invisible; the casing gives it an edge on any
   * ground without changing the colour that says which tier it is.
   */
  readonly roadCasing: Color;
  /** Ground a faction has never observed. */
  readonly fog: Color;
  /** The veil over ground not under observation. See `WashStyle`. */
  readonly wash: WashStyle;
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
  // Not from `palette.ts` either: the Python renderer draws rivers by flow rank and has no
  // Major/Minor line. Width carries the distinction and hue backs it up, so it survives
  // both the wash and a reader who cannot tell the two blues apart.
  river: {
    major: { color: '#1d4fa8', width: 0.34 },
    minor: { color: '#5fa0e6', width: 0.1 },
  },
  roadCasing: 'rgba(250,244,228,0.9)',
  fog: FOG_COLOR,
  // Not from `palette.ts`: the Python renderer draws a finished map of a whole country and
  // has no notion of who is looking at it. The wash only means anything on a screen being
  // read by one person.
  wash: { color: '#0b1020', observed: 0, surveyed: 0.25, unseen: 0.55 },
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
  readonly river?: Partial<Record<'major' | 'minor', RiverStyle>>;
  readonly roadCasing?: Color;
  readonly fog?: Color;
  readonly wash?: Partial<WashStyle>;
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
    river: { ...DEFAULT_THEME.river, ...overrides.river },
    roadCasing: overrides.roadCasing ?? DEFAULT_THEME.roadCasing,
    fog: overrides.fog ?? DEFAULT_THEME.fog,
    // Per key like the palettes above: a referee who wants the unwatched ground darker
    // should not have to restate the colour and the other two bands to say so.
    wash: { ...DEFAULT_THEME.wash, ...overrides.wash },
    fallback: overrides.fallback ?? DEFAULT_THEME.fallback,
  };
}

/** Look a colour up, falling back visibly rather than silently. */
export const colorFor = (map: ColorMap, key: string | null, theme: Theme): Color =>
  (key === null ? undefined : map[key]) ?? theme.fallback;
