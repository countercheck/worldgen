/**
 * That the map draws a river in the class the crossing rules give it.
 *
 * The map is how a commander learns which rivers they can wade. If it drew rivers by the
 * generator's flow rank instead, a line that looked like a stream could still refuse a
 * column without a bridge — so the split is asserted here against `riverClass` itself.
 */

import { describe, expect, it } from 'vitest';

import { key, type Hex, type HexKey, type River, type World } from '@campaign/shared';

import { riverRuns } from '../src/map/draw.js';

const NAVIGABLE = 1000;

/** A row of hexes along r = 0, each with the catchment given. Zero means no river. */
function worldOf(catchments: readonly number[]): World {
  const hexes = new Map<HexKey, unknown>();
  catchments.forEach((catchmentKm2, q) => {
    hexes.set(key({ q, r: 0 }), {
      coord: { q, r: 0 },
      tags: new Set(catchmentKm2 > 0 ? ['river'] : []),
      catchmentKm2,
    });
  });
  return {
    hexes,
    config: { navigableMinDischarge: NAVIGABLE, meanPrecipMm: 1 },
  } as unknown as World;
}

const along = (n: number): River => ({
  hexes: Array.from({ length: n }, (_, q): Hex => ({ q, r: 0 })),
  flowVolume: 0,
});

describe('splitting a river by class', () => {
  it('is one minor run when it never becomes navigable', () => {
    const runs = riverRuns(along(3), worldOf([10, 20, 30]));
    expect(runs.map((r) => r.cls)).toEqual(['minor']);
    expect(runs[0].hexes).toHaveLength(3);
  });

  it('turns major where the discharge crosses the line, and stays joined', () => {
    const runs = riverRuns(along(5), worldOf([10, 20, 2000, 3000, 4000]));

    expect(runs.map((r) => r.cls)).toEqual(['minor', 'major']);
    // The two runs share the first major hex, so they draw as one river.
    expect(runs[0].hexes.at(-1)).toEqual(runs[1].hexes[0]);
    expect(runs[0].hexes).toHaveLength(3);
    expect(runs[1].hexes).toHaveLength(3);
  });

  it('keeps a minor tributary thin right up to the major river it joins', () => {
    // Rivers run source to mouth; a tributary's last hex is on the major river.
    const runs = riverRuns(along(3), worldOf([10, 20, 5000]));
    expect(runs.map((r) => r.cls)).toEqual(['minor']);
  });

  it('carries a major river into its mouth', () => {
    // The last hex is sea: no river tag, so `riverClass` calls it none.
    const runs = riverRuns(along(3), worldOf([2000, 3000, 0]));
    expect(runs.map((r) => r.cls)).toEqual(['major']);
    expect(runs[0].hexes).toHaveLength(3);
  });

  it('leaves a lake wide when a major river flows out of it', () => {
    // The third hex is lake: no river tag, so `riverClass` calls it none.
    const runs = riverRuns(along(4), worldOf([2000, 3000, 0, 4000]));
    expect(runs.map((r) => r.cls)).toEqual(['major']);
    expect(runs[0].hexes).toHaveLength(4);
  });

  it('draws a hex the mask has blanked as minor rather than dropping it', () => {
    const world = worldOf([10, 20]);
    const runs = riverRuns({ hexes: [...along(2).hexes, { q: 9, r: 9 }], flowVolume: 0 }, world);
    expect(runs.map((r) => r.cls)).toEqual(['minor']);
    expect(runs[0].hexes).toHaveLength(3);
  });
});
