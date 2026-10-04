/**
 * That the map draws a river in the class the crossing rules give it.
 *
 * The map is how a commander learns which rivers they can wade. If it drew rivers by the
 * generator's flow rank instead, a line that looked like a stream could still refuse a
 * column without a bridge — so the split is asserted here against `riverClass` itself.
 */

import { describe, expect, it } from 'vitest';

import {
  cornerId,
  sideId,
  sideJoining,
  type Corner,
  type River,
  type RiverSide,
  type World,
} from '@campaign/shared';

import { riverMarks, riverRuns } from '../src/map/draw.js';

const NAVIGABLE = 1000;

/** The i-th corner of a chain running down the column of hexes at q = 0. */
const corner = (i: number): Corner => ({ q: 0, r: Math.floor(i / 2), k: i % 2 });

/** A course of `n` corners down that chain. */
const along = (n: number): River => ({
  corners: Array.from({ length: n }, (_, i) => corner(i)),
  flowVolume: 0,
  name: '',
});

/**
 * A world with a river side between each pair of corners, with the catchment given.
 * Zero means the side has no record — the fog cut it.
 */
function worldOf(catchments: readonly number[], tags: readonly string[][] = []): World {
  const riverSides = new Map<string, RiverSide>();
  catchments.forEach((catchmentKm2, i) => {
    if (catchmentKm2 <= 0) return;
    const side = sideJoining(corner(i), corner(i + 1));
    riverSides.set(sideId(side), {
      side,
      catchmentKm2,
      flow: 0,
      dropM: 0,
      tags: new Set(tags[i] ?? []),
    });
  });
  return {
    hexes: new Map(),
    rivers: [along(catchments.length + 1)],
    riverSides,
    riverCorners: new Map(),
    config: { navigableMinDischarge: NAVIGABLE, runoffMm: 1 },
  } as unknown as World;
}

describe('splitting a river by class', () => {
  it('is one minor run when it never becomes navigable', () => {
    const runs = riverRuns(along(4), worldOf([10, 20, 30]));
    expect(runs.map((r) => r.cls)).toEqual(['minor']);
    expect(runs[0]!.corners).toHaveLength(4);
  });

  it('turns major where the discharge crosses the line, and stays joined', () => {
    const runs = riverRuns(along(6), worldOf([10, 20, 2000, 3000, 4000]));
    expect(runs.map((r) => r.cls)).toEqual(['minor', 'major']);
    // The two runs share the corner where the river turns major, so they draw as one.
    expect(runs[0]!.corners.at(-1)).toEqual(runs[1]!.corners[0]);
    expect(runs[0]!.corners).toHaveLength(3);
    expect(runs[1]!.corners).toHaveLength(4);
  });

  it('carries a major river all the way to its last side', () => {
    const runs = riverRuns(along(3), worldOf([2000, 3000]));
    expect(runs.map((r) => r.cls)).toEqual(['major']);
    expect(runs[0]!.corners).toHaveLength(3);
  });

  it('draws a side the mask has no record of as minor rather than dropping it', () => {
    const runs = riverRuns(along(4), worldOf([10, 0, 20]));
    expect(runs.map((r) => r.cls)).toEqual(['minor']);
    expect(runs[0]!.corners).toHaveLength(4);
  });
});

describe('marking where a river rises, ends and runs white', () => {
  /** A course of three sides, with a source and an end on its corners and a side tagged. */
  function marked(sideTags: readonly string[][]): World {
    const w = worldOf([10, 10, 10], sideTags as string[][]);
    const riverCorners = new Map<string, ReadonlySet<string>>([
      [cornerId(corner(0)), new Set(['river_source'])],
      [cornerId(corner(3)), new Set(['river_end'])],
    ]);
    return { ...w, riverCorners };
  }

  it('marks a source and an end on their corners, and white water on its side', () => {
    const marks = riverMarks(marked([[], ['rapids'], []]));
    expect(marks.map((m) => m.kind).sort()).toEqual(['end', 'rapids', 'source']);
  });

  it('draws a cataract as white water too', () => {
    const kinds = riverMarks(marked([[], ['cataract'], []])).map((m) => m.kind);
    expect(kinds.filter((k) => k === 'rapids')).toHaveLength(1);
  });

  it('turns a mark along its river, downstream', () => {
    // Down the column at q = 0 is down the screen on the flat-top layout.
    const mark = riverMarks(marked([[], ['rapids'], []])).find((m) => m.kind === 'rapids');
    expect(Math.sin(mark!.bearing)).toBeGreaterThan(0);
  });
});
