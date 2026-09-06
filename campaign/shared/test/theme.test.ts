/**
 * Per-campaign theming.
 *
 * The generated palette is the default look, not the only one. A referee should be able
 * to override any part of it — their own land-use colours, a winter scenario — without a
 * code change, and without restating the parts they are happy with.
 */

import { describe, expect, it } from 'vitest';

import { colorFor, DEFAULT_THEME, resolveTheme } from '../src/theme.js';

describe('resolveTheme', () => {
  it('is the default when nothing is overridden', () => {
    expect(resolveTheme()).toBe(DEFAULT_THEME);
    expect(resolveTheme({}).landUse).toEqual(DEFAULT_THEME.landUse);
  });

  it('merges per key, so one colour can be changed alone', () => {
    // The difference between a usable override and one nobody uses: overriding a single
    // land-use colour must not require restating the other four.
    const theme = resolveTheme({ landUse: { arable: '#ff0000' } });

    expect(theme.landUse.arable).toBe('#ff0000');
    expect(theme.landUse.pasture).toBe(DEFAULT_THEME.landUse.pasture);
    expect(theme.landUse.wood).toBe(DEFAULT_THEME.landUse.wood);
    expect(Object.keys(theme.landUse).sort()).toEqual(Object.keys(DEFAULT_THEME.landUse).sort());
  });

  it('leaves the other palettes alone', () => {
    const theme = resolveTheme({ landUse: { arable: '#ff0000' } });
    expect(theme.biome).toEqual(DEFAULT_THEME.biome);
    expect(theme.terrain).toEqual(DEFAULT_THEME.terrain);
    expect(theme.road).toEqual(DEFAULT_THEME.road);
  });

  it('accepts a key the default does not have', () => {
    // A campaign may name land uses of its own, which is the first step towards an
    // imported tileset defining its own categories.
    const theme = resolveTheme({ landUse: { vineyard: '#804060' } });
    expect(theme.landUse.vineyard).toBe('#804060');
  });

  it('overrides fog and road styling', () => {
    const theme = resolveTheme({
      fog: '#000000',
      road: { primary: { color: '#111111', width: 3 } },
    });
    expect(theme.fog).toBe('#000000');
    expect(theme.road.primary).toEqual({ color: '#111111', width: 3 });
    expect(theme.road.track).toEqual(DEFAULT_THEME.road.track);
  });

  it('does not mutate the default', () => {
    const before = { ...DEFAULT_THEME.landUse };
    resolveTheme({ landUse: { arable: '#ff0000' } });
    expect(DEFAULT_THEME.landUse).toEqual(before);
  });
});

describe('colorFor', () => {
  it('finds a colour that is there', () => {
    const theme = resolveTheme();
    expect(colorFor(theme.landUse, 'arable', theme)).toBe(DEFAULT_THEME.landUse.arable);
  });

  it('falls back visibly rather than silently', () => {
    // Magenta on purpose: a missing colour should look like a bug, not like terrain. A
    // tasteful grey is how an incomplete palette ships unnoticed.
    const theme = resolveTheme();
    expect(colorFor(theme.landUse, 'nonsense', theme)).toBe(theme.fallback);
    expect(colorFor(theme.landUse, null, theme)).toBe(theme.fallback);
    expect(theme.fallback).toBe('#ff00ff');
  });
});
