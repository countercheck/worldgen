/**
 * What the map marks besides the ground: names, crossings and landings.
 *
 * Each mirrors a function in the Python exporters, so the campaign map and the atlas the
 * generator wrote mark the same places. These pin the rules rather than the pixels.
 */

import { describe, expect, it } from 'vitest';

import { key, type Hex, type HexKey, type RoadEdge, type World } from '@campaign/shared';

import {
  ALL_LAYERS,
  anchoragePoints,
  crossingMarks,
  groundFill,
  reliefColor,
} from '../src/map/draw.js';
import { estimateWidth, MIN_LABEL_PX, placeLabels, type PlacedLabel } from '../src/map/labels.js';
import { normaliseLayers } from '../src/session.js';

type Cell = { tags?: string[]; water?: boolean };

/** A world of the cells given, keyed `q,r`, and whatever else the test names. */
function worldOf(cells: Record<string, Cell>, rest: Partial<World> = {}): World {
  const hexes = new Map<HexKey, unknown>();
  for (const [k, cell] of Object.entries(cells)) {
    const [q, r] = k.split(',').map(Number) as [number, number];
    hexes.set(key({ q, r }), {
      coord: { q, r },
      tags: new Set(cell.tags ?? []),
      terrainClass: cell.water === true ? 'open_water' : 'land',
    });
  }
  return {
    hexes,
    rivers: [],
    settlements: [],
    roadEdges: new Map(),
    seaEdges: new Map(),
    ferries: [],
    ...rest,
  } as unknown as World;
}

const edge = (a: Hex, b: Hex): [string, RoadEdge] => [
  `${key(a)}|${key(b)}`,
  { a, b, tier: 'track', deltaElevationM: 0 },
];

describe('fords and bridges', () => {
  it('marks every tagged crossing, and a bridge over a ford on the same hex', () => {
    const world = worldOf({
      '0,0': { tags: ['river', 'ford'] },
      '1,0': { tags: ['river', 'ford', 'bridge'] },
      '2,0': { tags: ['river'] },
    });
    expect(crossingMarks(world).map((c) => [key(c.coord), c.kind])).toEqual([
      ['0,0', 'ford'],
      ['1,0', 'bridge'],
    ]);
  });

  it('takes the river’s bearing through the hex, so the mark can lie across it', () => {
    const world = worldOf(
      { '0,0': {}, '1,0': { tags: ['ford'] }, '2,0': {} },
      { rivers: [{ hexes: [{ q: 0, r: 0 }, { q: 1, r: 0 }, { q: 2, r: 0 }], flowVolume: 1, name: '' }] },
    );
    const [ford] = crossingMarks(world);
    // Downstream along q on the flat-top layout runs to the right and a little down.
    expect(Math.cos(ford!.bearing)).toBeGreaterThan(0);
  });
});

describe('landings', () => {
  const shore = { q: 1, r: 0 };
  const inland = { q: 0, r: 0 };
  const sea = { q: 2, r: 0 };
  const cells = { '0,0': {}, '1,0': {}, '2,0': { water: true }, '5,5': {}, '6,6': {} };

  it('puts an anchor where a road reaches a sea leg', () => {
    const world = worldOf(cells, {
      roadEdges: new Map([edge(inland, shore)]),
      seaEdges: new Map([edge(shore, sea)]),
    });
    expect(anchoragePoints(world)).toEqual([shore]);
  });

  it('leaves off a landing no road reaches', () => {
    const world = worldOf(cells, { seaEdges: new Map([edge(shore, sea)]) });
    expect(anchoragePoints(world)).toEqual([]);
  });

  it('puts one at each end of a ferry', () => {
    const world = worldOf(cells, { ferries: [{ a: { q: 5, r: 5 }, b: { q: 6, r: 6 } }] });
    expect(anchoragePoints(world)).toEqual([
      { q: 5, r: 5 },
      { q: 6, r: 6 },
    ]);
  });
});

describe('names', () => {
  /** One hex is `size` pixels square, laid out on a plain grid: enough to test the rules. */
  const grid = (size: number) => (c: Hex) => ({ x: c.q * size, y: c.r * size });

  const town = (name: string, q: number, r: number, tier = 'town', population = 100) => ({
    coord: { q, r },
    tier,
    role: 'market',
    population,
    name,
    culture: '',
    etymology: '',
  });

  const boxOf = (l: PlacedLabel) => {
    const { w, h } = estimateWidth(l.text, l.size, l.bold);
    return [l.x - w / 2, l.y - h / 2, l.x + w / 2, l.y + h / 2] as const;
  };

  it('never sets two names on top of each other', () => {
    const settlements = Array.from({ length: 30 }, (_, i) =>
      town(`Place ${i}`, i % 6, Math.floor(i / 6)),
    );
    const placed = placeLabels(worldOf({}, { settlements }), grid(20), 20);
    expect(placed.length).toBeGreaterThan(0);
    for (let i = 0; i < placed.length; i++) {
      for (let j = i + 1; j < placed.length; j++) {
        const [a0, b0, a1, b1] = boxOf(placed[i]!);
        const [c0, d0, c1, d1] = boxOf(placed[j]!);
        const overlap = a1 > c0 && a0 < c1 && b1 > d0 && b0 < d1;
        expect(overlap, `${placed[i]!.text} / ${placed[j]!.text}`).toBe(false);
      }
    }
  });

  it('drops a crowded village before the city beside it', () => {
    const settlements = [
      town('Hamlet', 0, 0, 'village', 50),
      town('Metropolis', 1, 0, 'city', 9000),
    ];
    // So close together that only one name can stand.
    const placed = placeLabels(worldOf({}, { settlements }), grid(4), 12);
    expect(placed.map((l) => l.text)).toContain('Metropolis');
    expect(placed[0]!.text).toBe('Metropolis');
  });

  it('leaves off a name too small to read at this zoom', () => {
    const settlements = [town('Somewhere', 0, 0, 'village')];
    const size = (MIN_LABEL_PX - 1) / 0.58;
    expect(placeLabels(worldOf({}, { settlements }), grid(size), size)).toEqual([]);
  });

  it('sets a river’s name along it, never upside down', () => {
    const hexes = Array.from({ length: 7 }, (_, i): Hex => ({ q: 6 - i, r: 0 }));
    const cells = Object.fromEntries(hexes.map((c) => [key(c), {}]));
    const world = worldOf(cells, { rivers: [{ hexes, flowVolume: 1, name: 'Vassa' }] });
    const [label] = placeLabels(world, grid(30), 30);
    expect(label).toMatchObject({ text: 'Vassa', river: true, italic: true });
    // Flowing right to left, the name still reads left to right.
    expect(Math.abs(label!.angle)).toBeLessThanOrEqual(Math.PI / 2);
  });
});

describe('layers', () => {
  it('reads back what was stored, and fills in anything a stored value is missing', () => {
    expect(normaliseLayers(null)).toEqual(ALL_LAYERS);
    expect(normaliseLayers('nonsense')).toEqual(ALL_LAYERS);
    // An older build stored no ports, and a ground this build does not know.
    expect(normaliseLayers({ ground: 'satellite', names: false })).toEqual({
      ...ALL_LAYERS,
      names: false,
    });
    expect(normaliseLayers({ ...ALL_LAYERS, ground: 'plain', roads: 'yes' })).toEqual({
      ...ALL_LAYERS,
      ground: 'plain',
    });
  });

  it('tints height across the world’s own range, and shades the steep', () => {
    const range = { lo: 100, hi: 1100 };
    const rgb = (c: string) => c.match(/\d+/g)!.map(Number);
    const low = rgb(reliefColor(100, 0, range));
    const high = rgb(reliefColor(1100, 0, range));
    // Low ground is green; the top of the range is near white.
    expect(low[1]!).toBeGreaterThan(low[0]!);
    expect(Math.min(...high)).toBeGreaterThan(220);
    // The same height is darker on a slope.
    const flat = rgb(reliefColor(600, 0, range));
    const steep = rgb(reliefColor(600, 300, range));
    expect(steep.reduce((a, b) => a + b)).toBeLessThan(flat.reduce((a, b) => a + b));
  });

  it('draws water as water and land as paper on the plain ground', () => {
    const hex = (terrainClass: string) =>
      ({ tags: new Set(), terrainClass, elevation: 0, slope: 0 }) as never;
    const range = { lo: 0, hi: 1 };
    expect(groundFill(hex('land'), 'plain', range)).not.toBe(groundFill(hex('open_water'), 'plain', range));
    expect(groundFill(hex('land'), 'plain', range)).toBe(groundFill(hex('coast'), 'plain', range));
  });

  it('sets no river names when the rivers are off', () => {
    const hexes = Array.from({ length: 7 }, (_, i): Hex => ({ q: i, r: 0 }));
    const cells = Object.fromEntries(hexes.map((c) => [key(c), {}]));
    const world = worldOf(cells, { rivers: [{ hexes, flowVolume: 1, name: 'Vassa' }] });
    const at = (c: Hex) => ({ x: c.q * 30, y: c.r * 30 });
    expect(placeLabels(world, at, 30, estimateWidth, { rivers: false })).toEqual([]);
  });
});
