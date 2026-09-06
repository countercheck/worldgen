/**
 * The world parser, against genuinely generated output.
 *
 * The fixture is a real 32x32 organic world from `scripts/make_world_fixture.py`, not a
 * hand-written approximation — the point is that the parser survives what the generator
 * actually emits, including its nullable enums and its coordinate-pair edge collections.
 */

import { describe, expect, it } from 'vitest';

import world32 from './fixtures/world-32x32.json' with { type: 'json' };

import { distance, key, type Hex } from '../src/hex.js';
import {
  edgeKey,
  hexAt,
  isWater,
  parseWorld,
  roadBetween,
  ROAD_TIER_RANK,
  WorldParseError,
  type World,
} from '../src/world.js';

const load = (): World => parseWorld(world32);

describe('parseWorld', () => {
  it('reads the generated world whole', () => {
    const w = load();
    expect(w.schemaVersion).toBe('1.8');
    expect(w.layout).toBe('axial');
    expect(w.width).toBe(32);
    expect(w.height).toBe(32);
    expect(w.hexes.size).toBe(32 * 32);
  });

  it('keeps every hex reachable by coordinate', () => {
    const w = load();
    for (let q = 0; q < w.width; q++) {
      for (let r = 0; r < w.height; r++) {
        expect(hexAt(w, { q, r }), `hex ${q},${r} missing`).toBeDefined();
      }
    }
  });

  it('carries the config the world was generated with', () => {
    const w = load();
    // Not the parser's defaults — these must come from metadata.config, or a world
    // generated before a threshold moved would have its rivers reclassified.
    expect(w.config.navigableMinDischarge).toBeGreaterThan(0);
    expect(w.config.fordMaxCatchmentKm2).toBeGreaterThan(0);
    expect(w.config.model).toBe('organic');
  });
});

describe('terrain and rivers', () => {
  it('parses every terrain class the generator produced', () => {
    const w = load();
    const classes = new Set([...w.hexes.values()].map((h) => h.terrainClass));
    expect(classes).toContain('land');
    expect(classes).toContain('open_water');
    expect(classes).toContain('coast');
  });

  it('agrees with isWater about which hexes are wet', () => {
    const w = load();
    for (const h of w.hexes.values()) {
      const wet = h.terrainClass === 'open_water' || h.terrainClass === 'inland_water';
      expect(isWater(h)).toBe(wet);
    }
  });

  it('gives river hexes an upstream catchment', () => {
    const w = load();
    const river = [...w.hexes.values()].filter((h) => h.tags.has('river'));
    expect(river.length).toBeGreaterThan(0);
    for (const h of river) expect(h.catchmentKm2).toBeGreaterThan(0);
  });

  it('carries the ford tags the organic pipeline writes', () => {
    // The river-crossing rules key on these, and only the organic model produces them.
    const w = load();
    const fords = [...w.hexes.values()].filter((h) => h.tags.has('ford'));
    expect(fords.length).toBeGreaterThan(0);
  });

  it('runs each river along a connected chain of hexes that exist', () => {
    const w = load();
    expect(w.rivers.length).toBeGreaterThan(0);
    for (const river of w.rivers) {
      for (const c of river.hexes) expect(hexAt(w, c), `river hex ${key(c)}`).toBeDefined();
      for (let i = 1; i < river.hexes.length; i++) {
        expect(distance(river.hexes[i - 1]!, river.hexes[i]!)).toBe(1);
      }
    }
  });
});

describe('road edges', () => {
  it('parses all three tiers', () => {
    const w = load();
    const tiers = new Set([...w.roadEdges.values()].map((e) => e.tier));
    expect(tiers).toEqual(new Set(['primary', 'secondary', 'track']));
    for (const t of tiers) expect(ROAD_TIER_RANK[t]).toBeTypeOf('number');
  });

  it('joins adjacent hexes that both exist', () => {
    const w = load();
    expect(w.roadEdges.size).toBeGreaterThan(0);
    for (const e of w.roadEdges.values()) {
      expect(hexAt(w, e.a), `road endpoint ${key(e.a)}`).toBeDefined();
      expect(hexAt(w, e.b), `road endpoint ${key(e.b)}`).toBeDefined();
      expect(distance(e.a, e.b), `road edge ${key(e.a)}->${key(e.b)} is not one step`).toBe(1);
    }
  });

  it('is found from either end', () => {
    // Movement asks about the edge it is crossing without knowing which end the
    // generator happened to write first, so the key must be direction-free.
    const w = load();
    for (const e of w.roadEdges.values()) {
      expect(roadBetween(w, e.a, e.b)).toBe(e);
      expect(roadBetween(w, e.b, e.a)).toBe(e);
    }
  });

  it('carries a per-edge elevation delta, which is the on-road grade term', () => {
    const w = load();
    for (const e of w.roadEdges.values()) {
      expect(Number.isFinite(e.deltaElevationM)).toBe(true);
    }
    // A road that climbs and descends somewhere, or the grade term is meaningless.
    const deltas = [...w.roadEdges.values()].map((e) => e.deltaElevationM);
    expect(Math.max(...deltas)).toBeGreaterThan(0);
    expect(Math.min(...deltas)).toBeLessThan(0);
  });

  it('keeps sea edges separate from road edges', () => {
    const w = load();
    expect(w.seaEdges.size).toBeGreaterThan(0);
    for (const k of w.seaEdges.keys()) expect(w.roadEdges.has(k)).toBe(false);
  });
});

describe('settlements', () => {
  it('sits every settlement on a hex, and back-references it', () => {
    const w = load();
    expect(w.settlements.length).toBeGreaterThan(0);
    for (const s of w.settlements) {
      const h = hexAt(w, s.coord);
      expect(h, `settlement ${s.name} at ${key(s.coord)}`).toBeDefined();
      expect(h!.settlementName).toBe(s.name);
    }
  });

  it('leaves empty hexes without a settlement name', () => {
    const w = load();
    const named = [...w.hexes.values()].filter((h) => h.settlementName !== null);
    expect(named.length).toBe(w.settlements.length);
  });
});

describe('edgeKey', () => {
  it('is the same whichever way round the pair is given', () => {
    const pairs: [Hex, Hex][] = [
      [{ q: 0, r: 0 }, { q: 1, r: 0 }],
      [{ q: 5, r: -3 }, { q: 5, r: -2 }],
      [{ q: -2, r: 7 }, { q: -3, r: 7 }],
      [{ q: 4, r: 4 }, { q: 4, r: 4 }],
    ];
    for (const [a, b] of pairs) expect(edgeKey(a, b)).toBe(edgeKey(b, a));
  });

  it('distinguishes different pairs', () => {
    const seen = new Set<string>();
    for (let q = 0; q < 6; q++) {
      for (let r = 0; r < 6; r++) {
        seen.add(edgeKey({ q, r }, { q: q + 1, r }));
      }
    }
    expect(seen.size).toBe(36);
  });
});

describe('rejecting malformed input', () => {
  it('refuses a schema version it does not read', () => {
    expect(() => parseWorld({ ...world32, version: '1.4' })).toThrow(WorldParseError);
    expect(() => parseWorld({ ...world32, version: '1.4' })).toThrow(/not supported/);
  });

  it('refuses a non-object', () => {
    for (const bad of [null, 42, 'world', []]) {
      expect(() => parseWorld(bad), `${JSON.stringify(bad)}`).toThrow(WorldParseError);
    }
  });

  it('names the field that is wrong', () => {
    // A missing field surfacing as undefined inside a movement calculation is far worse
    // than a clear error at the door, so the message has to point at the field.
    const broken = { ...world32, hexes: [{ q: 0, r: 0, terrain_class: 'land' }] };
    expect(() => parseWorld(broken)).toThrow(/hexes\[0\]\.elevation/);
  });

  it('refuses a terrain class it does not know', () => {
    const broken = {
      ...world32,
      hexes: [{ q: 0, r: 0, elevation: 0, terrain_class: 'swamp' }],
    };
    expect(() => parseWorld(broken)).toThrow(/terrain_class/);
  });

  it('refuses a road tier it does not know', () => {
    const broken = {
      ...world32,
      road_edges: [{ a: [0, 0], b: [1, 0], tier: 'motorway', delta_elevation_m: 0 }],
    };
    expect(() => parseWorld(broken)).toThrow(/tier/);
  });
});
