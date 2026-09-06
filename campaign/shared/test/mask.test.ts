/**
 * Masking a world down to one faction's knowledge.
 *
 * The output has to remain a legal `world.json`, so most of these assert on the shape of
 * the document rather than on the parsed result. `tests/test_masked_world.py` completes
 * the picture by loading a masked file with the real Python.
 */

import { describe, expect, it } from 'vitest';

import world32 from './fixtures/world-32x32.json' with { type: 'json' };

import { key, type HexKey } from '../src/hex.js';
import { FOG_TAG, isFog, keysOf, maskWorld, REMEMBERED_TAG } from '../src/mask.js';
import { parseWorld } from '../src/world.js';

type Doc = Record<string, unknown>;
const doc = world32 as unknown as Doc;
const full = parseWorld(world32);

const allCoords = [...full.hexes.values()].map((h) => h.coord);

/** A blob of knowledge around one point. */
function around(centre: { q: number; r: number }, radius: number): Set<HexKey> {
  const out = new Set<HexKey>();
  for (const h of full.hexes.values()) {
    const d =
      (Math.abs(h.coord.q - centre.q) +
        Math.abs(h.coord.r - centre.r) +
        Math.abs(h.coord.q + h.coord.r - centre.q - centre.r)) /
      2;
    if (d <= radius) out.add(key(h.coord));
  }
  return out;
}

const seen = around({ q: 16, r: 16 }, 6);
const visible = around({ q: 16, r: 16 }, 2);

const mask = (over: Partial<Parameters<typeof maskWorld>[1]> = {}): Doc =>
  maskWorld(doc, { seen, visible, faction: 'red', clockHours: 24, ...over });

describe('the masked document is still a world', () => {
  it('keeps every hex, so the canvas bounds do not move', () => {
    // Dropping hexes would size each faction's map differently, and two commanders'
    // maps could then not be laid over one another or over the referee's.
    const masked = mask();
    expect((masked.hexes as Doc[]).length).toBe((doc.hexes as Doc[]).length);
  });

  it('keeps the schema version and dimensions', () => {
    const masked = mask();
    expect(masked.version).toBe(doc.version);
    expect(masked.width).toBe(doc.width);
    expect(masked.height).toBe(doc.height);
    expect(masked.layout).toBe(doc.layout);
  });

  it('reparses as a world', () => {
    const reparsed = parseWorld(mask());
    expect(reparsed.hexes.size).toBe(full.hexes.size);
  });

  it('keeps every field the Python loader requires on every hex', () => {
    // WorldState.from_dict hard-requires these, so an omitted one does not load at all.
    const required = [
      'q', 'r', 'elevation', 'moisture', 'temperature',
      'terrain_class', 'river_flow', 'cultivated',
    ];
    for (const h of mask().hexes as Doc[]) {
      for (const f of required) expect(h[f], `${f} on ${h.q},${h.r}`).toBeDefined();
    }
  });

  it('preserves fields the parser does not model', () => {
    // Masking must not quietly become a lossy re-serialisation: a commander's map should
    // carry soil and land use even though the movement rules never read them.
    const masked = mask();
    const known = (masked.hexes as Doc[]).find((h) => seen.has(key({ q: h.q as number, r: h.r as number })))!;
    for (const f of ['soil', 'land_use', 'alluvium', 'habitability_city', 'territory_cost']) {
      expect(known, `${f} was dropped`).toHaveProperty(f);
    }
  });
});

describe('what is hidden', () => {
  it('tags unseen hexes and blanks their values', () => {
    const masked = mask();
    for (const h of masked.hexes as Doc[]) {
      const k = key({ q: h.q as number, r: h.r as number });
      if (seen.has(k)) continue;
      expect(isFog(h.tags as string[]), `${k} is not tagged fog`).toBe(true);
      expect(h.elevation).toBe(0);
      expect(h.biome).toBeNull();
      expect(h.land_cover).toBeNull();
      expect(h.road_connections).toEqual([]);
    }
  });

  it('leaves known hexes untagged and intact', () => {
    const masked = mask();
    const original = new Map((doc.hexes as Doc[]).map((h) => [key({ q: h.q as number, r: h.r as number }), h]));

    for (const h of masked.hexes as Doc[]) {
      const k = key({ q: h.q as number, r: h.r as number });
      if (!seen.has(k)) continue;
      expect(isFog(h.tags as string[])).toBe(false);
      expect(h.elevation).toBe(original.get(k)!.elevation);
      expect(h.biome).toBe(original.get(k)!.biome);
    }
  });

  it('prunes roads running on into the dark', () => {
    // Otherwise a commander could read the enemy's lines of communication off the shape
    // of a road leaving their own map.
    for (const h of mask().hexes as Doc[]) {
      for (const c of h.road_connections as number[][]) {
        expect(seen.has(key({ q: c[0]!, r: c[1]! })), `road to unseen ${c}`).toBe(true);
      }
    }
  });

  it('keeps an edge only when both ends are known', () => {
    const masked = mask();
    for (const field of ['road_edges', 'sea_edges', 'ferries']) {
      for (const e of masked[field] as Doc[]) {
        const a = e.a as number[];
        const b = e.b as number[];
        expect(seen.has(key({ q: a[0]!, r: a[1]! })), `${field} end a`).toBe(true);
        expect(seen.has(key({ q: b[0]!, r: b[1]! })), `${field} end b`).toBe(true);
      }
    }
  });

  it('keeps a settlement only on ground the faction has seen', () => {
    for (const s of mask().settlements as Doc[]) {
      const c = s.coord as number[];
      expect(seen.has(key({ q: c[0]!, r: c[1]! }))).toBe(true);
    }
  });

  it('hides most of the map when little has been scouted', () => {
    const masked = mask();
    const fogged = (masked.hexes as Doc[]).filter((h) => isFog(h.tags as string[]));
    expect(fogged.length).toBeGreaterThan((doc.hexes as Doc[]).length * 0.5);
  });
});

describe('rivers', () => {
  it('splits a river into the runs actually seen', () => {
    // Keeping only the known hexes of a chain would leave a renderer drawing a straight
    // line between two banks either side of unseen country.
    const masked = mask();
    for (const river of masked.rivers as Doc[]) {
      const hexes = river.hexes as number[][];
      expect(hexes.length).toBeGreaterThanOrEqual(2);
      for (const c of hexes) {
        expect(seen.has(key({ q: c[0]!, r: c[1]! }))).toBe(true);
      }
      // Each run is contiguous.
      for (let i = 1; i < hexes.length; i++) {
        const a = hexes[i - 1]!;
        const b = hexes[i]!;
        const d = (Math.abs(a[0]! - b[0]!) + Math.abs(a[1]! - b[1]!) +
          Math.abs(a[0]! + a[1]! - b[0]! - b[1]!)) / 2;
        expect(d).toBe(1);
      }
    }
  });

  it('drops a lone hex of river, which is a dot rather than a watercourse', () => {
    const oneHex = new Set([key({ q: 16, r: 16 })]);
    const masked = maskWorld(doc, {
      seen: oneHex,
      visible: oneHex,
      faction: 'red',
      clockHours: 0,
    });
    for (const river of masked.rivers as Doc[]) {
      expect((river.hexes as number[][]).length).toBeGreaterThanOrEqual(2);
    }
  });
});

describe('memory versus observation', () => {
  it('marks known-but-not-visible ground as remembered', () => {
    const masked = mask();
    for (const h of masked.hexes as Doc[]) {
      const k = key({ q: h.q as number, r: h.r as number });
      const tags = h.tags as string[];
      if (!seen.has(k)) continue;
      expect(tags.includes(REMEMBERED_TAG), `${k}`).toBe(!visible.has(k));
    }
  });

  it('discards memory when asked to show only what is in view', () => {
    const now = mask({ remember: false });
    const kept = (now.hexes as Doc[]).filter((h) => !isFog(h.tags as string[]));
    expect(kept.length).toBe(visible.size);
  });

  it('shows strictly less without memory than with it', () => {
    const withMemory = (mask().hexes as Doc[]).filter((h) => !isFog(h.tags as string[])).length;
    const without = (mask({ remember: false }).hexes as Doc[]).filter(
      (h) => !isFog(h.tags as string[]),
    ).length;
    expect(without).toBeLessThan(withMemory);
  });
});

describe('metadata', () => {
  it('says whose map this is and what the fog tag means', () => {
    // The tag is the contract with any consumer, so the file states it rather than
    // relying on the reader knowing.
    const fog = (mask().metadata as Doc).fog as Doc;
    expect(fog.faction).toBe('red');
    expect(fog.clock_hours).toBe(24);
    expect(fog.fog_tag).toBe(FOG_TAG);
    expect(fog.remembered_tag).toBe(REMEMBERED_TAG);
    expect(fog.known).toBe(seen.size);
    expect(fog.total).toBe((doc.hexes as Doc[]).length);
    expect(fog.note).toContain('do not read them as terrain');
  });

  it('keeps the generator metadata, so the config still travels', () => {
    const masked = mask();
    expect((masked.metadata as Doc).config).toEqual((doc.metadata as Doc).config);
  });
});

describe('edge cases', () => {
  it('produces a legal empty map for a faction that has seen nothing', () => {
    const nothing = new Set<HexKey>();
    const masked = maskWorld(doc, {
      seen: nothing,
      visible: nothing,
      faction: 'red',
      clockHours: 0,
    });

    expect((masked.hexes as Doc[]).length).toBe((doc.hexes as Doc[]).length);
    expect(masked.settlements).toEqual([]);
    expect(masked.road_edges).toEqual([]);
    expect(masked.rivers).toEqual([]);
    expect(() => parseWorld(masked)).not.toThrow();
  });

  it('returns the whole world to a faction that has seen everything', () => {
    const everything = keysOf(allCoords);
    const masked = maskWorld(doc, {
      seen: everything,
      visible: everything,
      faction: 'red',
      clockHours: 0,
    });

    expect((masked.settlements as Doc[]).length).toBe((doc.settlements as Doc[]).length);
    expect((masked.road_edges as Doc[]).length).toBe((doc.road_edges as Doc[]).length);
    for (const h of masked.hexes as Doc[]) expect(isFog(h.tags as string[])).toBe(false);
  });

  it('does not touch the document it was given', () => {
    const before = JSON.stringify(doc);
    mask();
    expect(JSON.stringify(doc)).toBe(before);
  });
});
