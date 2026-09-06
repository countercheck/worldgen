/**
 * The generated palette must cover everything a world can actually contain.
 *
 * A missing entry is not a crash, it is a hex quietly drawn in the fallback colour — the
 * kind of bug that survives review because the map still looks like a map. So the
 * coverage is asserted against the enum values the world parser accepts, and against the
 * real fixture world.
 */

import { describe, expect, it } from 'vitest';

import world32 from './fixtures/world-32x32.json' with { type: 'json' };

import {
  BIOME_COLORS,
  FOG_COLOR,
  LAND_COVER_COLORS,
  LAND_USE_COLORS,
  PRECIP_RAMP_MM,
  ROAD_STYLE,
  SOIL_COLORS,
  TEMPERATURE_RAMP_C,
  TERRAIN_COLORS,
  type Color,
} from '../src/palette.js';
import { parseWorld } from '../src/world.js';

const HEX_COLOR = /^#[0-9a-f]{6}$/;

const palettes: [string, Readonly<Record<string, Color>>][] = [
  ['TERRAIN_COLORS', TERRAIN_COLORS],
  ['BIOME_COLORS', BIOME_COLORS],
  ['LAND_COVER_COLORS', LAND_COVER_COLORS],
  ['SOIL_COLORS', SOIL_COLORS],
  ['LAND_USE_COLORS', LAND_USE_COLORS],
];

describe('every colour is well formed', () => {
  for (const [name, palette] of palettes) {
    it(`${name} holds only #rrggbb strings`, () => {
      expect(Object.keys(palette).length).toBeGreaterThan(0);
      for (const [k, v] of Object.entries(palette)) {
        expect(v, `${name}.${k}`).toMatch(HEX_COLOR);
      }
    });
  }

  it('the fog colour is a real colour', () => {
    expect(FOG_COLOR).toMatch(HEX_COLOR);
  });
});

describe('coverage of what a world can contain', () => {
  it('colours every land cover the parser accepts', () => {
    const covers = [
      'open_water', 'bog', 'marsh', 'dense_forest', 'woodland', 'scrub',
      'open', 'tundra', 'desert', 'alpine', 'bare_rock',
    ];
    for (const c of covers) {
      expect(LAND_COVER_COLORS[c], `no colour for land cover ${c}`).toBeDefined();
    }
  });

  it('styles every road tier', () => {
    for (const tier of ['primary', 'secondary', 'track']) {
      const style = ROAD_STYLE[tier];
      expect(style, `no style for road tier ${tier}`).toBeDefined();
      expect(style!.color).toMatch(HEX_COLOR);
      expect(style!.width).toBeGreaterThan(0);
    }
  });

  it('draws the track dashed, so tiers are told apart without a key', () => {
    expect(ROAD_STYLE.track!.dash).toBeDefined();
    expect(ROAD_STYLE.primary!.dash).toBeUndefined();
  });

  it('colours everything actually present in the fixture world', () => {
    const w = parseWorld(world32);
    for (const h of w.hexes.values()) {
      if (h.landCover !== null) {
        expect(LAND_COVER_COLORS[h.landCover], `land cover ${h.landCover}`).toBeDefined();
      }
      if (h.biome !== null) {
        expect(BIOME_COLORS[h.biome], `biome ${h.biome}`).toBeDefined();
      }
    }
    for (const e of w.roadEdges.values()) {
      expect(ROAD_STYLE[e.tier], `road tier ${e.tier}`).toBeDefined();
    }
  });
});

describe('ramps', () => {
  it('run low to high', () => {
    expect(TEMPERATURE_RAMP_C[0]).toBeLessThan(TEMPERATURE_RAMP_C[1]);
    expect(PRECIP_RAMP_MM[0]).toBeLessThan(PRECIP_RAMP_MM[1]);
  });
});

describe('fog', () => {
  it('is darker than every terrain fill', () => {
    // Unseen ground has to read as absent rather than as another kind of terrain, and
    // the cheapest guarantee of that is that nothing on the map is darker.
    const luma = (c: Color): number => {
      const n = parseInt(c.slice(1), 16);
      return 0.2126 * ((n >> 16) & 255) + 0.7152 * ((n >> 8) & 255) + 0.0722 * (n & 255);
    };
    const fog = luma(FOG_COLOR);
    for (const [name, palette] of palettes) {
      for (const [k, v] of Object.entries(palette)) {
        expect(luma(v), `${name}.${k} is no lighter than fog`).toBeGreaterThan(fog);
      }
    }
  });
});
