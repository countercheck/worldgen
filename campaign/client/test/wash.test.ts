/**
 * That the wash draws every hex it classifies, and not merely the last one.
 *
 * `board.test.ts` covers the classification, which is the part with rules in it. This
 * covers the part with a canvas in it, and it exists because of a real bug: the first
 * version accumulated hexes by calling `hexPath` in a loop, and `hexPath` begins a new
 * path, so each hexagon discarded the one before it and exactly one hex per band was ever
 * filled. The classifier was perfectly correct and the map was wrong, which is precisely
 * the seam a pure-function test cannot see across.
 *
 * Stubs rather than a real canvas: these tests run in node, and what is being asserted is
 * how many subpaths reach the fill, which a stub can count and a bitmap cannot.
 */

import { beforeAll, describe, expect, it } from 'vitest';

import { DEFAULT_THEME, key, type HexKey, type World } from '@campaign/shared';

import { drawWash, hexSubPath, type View } from '../src/map/draw.js';

/** Counts the hexagons added to it. One `moveTo` begins each. */
class FakePath {
  moves = 0;
  moveTo(): void {
    this.moves += 1;
  }
  lineTo(): void {}
  closePath(): void {}
}

beforeAll(() => {
  (globalThis as { Path2D?: unknown }).Path2D = FakePath;
});

interface Filled {
  readonly alpha: number;
  readonly hexes: number;
}

function fakeContext(): { ctx: CanvasRenderingContext2D; fills: Filled[] } {
  const fills: Filled[] = [];
  const ctx = {
    canvas: { width: 100, height: 100 },
    globalAlpha: 1,
    fillStyle: '',
    clearRect: () => {},
    save: () => {},
    restore: () => {},
    fill: (path: FakePath) => {
      fills.push({ alpha: ctx.globalAlpha, hexes: path.moves });
    },
  };
  return { ctx: ctx as unknown as CanvasRenderingContext2D, fills };
}

/** A 4x4 rhombus, which is all `drawWash` reads of a world. */
const world = ((): World => {
  const hexes = new Map<HexKey, { coord: { q: number; r: number } }>();
  for (let q = 0; q < 4; q += 1) {
    for (let r = 0; r < 4; r += 1) hexes.set(key({ q, r }), { coord: { q, r } });
  }
  return { hexes } as unknown as World;
})();

const view: View = { size: 10, offsetX: 0, offsetY: 0 };
const at = (q: number, r: number): HexKey => key({ q, r });

describe('drawing the wash', () => {
  it('fills every hex of a band, not just the last one', () => {
    const { ctx, fills } = fakeContext();
    const visible = new Set([at(0, 0)]);
    const surveyed = new Set([at(0, 0), at(1, 0), at(2, 0)]);

    drawWash(ctx, world, view, { visible, surveyed, mode: 'three' }, DEFAULT_THEME);

    // 16 hexes: 1 watched, 2 more surveyed, 13 never seen.
    const surveyedFill = fills.find((f) => f.alpha === DEFAULT_THEME.wash.surveyed);
    const unseenFill = fills.find((f) => f.alpha === DEFAULT_THEME.wash.unseen);
    expect(surveyedFill?.hexes).toBe(2);
    expect(unseenFill?.hexes).toBe(13);
  });

  it('never washes the hexes under observation', () => {
    const { ctx, fills } = fakeContext();
    const visible = new Set([at(0, 0), at(1, 1)]);

    drawWash(ctx, world, view, { visible, surveyed: visible, mode: 'three' }, DEFAULT_THEME);

    // 14 unseen, and no surveyed band at all: everything surveyed is also watched.
    expect(fills.map((f) => f.hexes)).toEqual([14]);
    expect(fills.every((f) => f.alpha > 0)).toBe(true);
  });

  it('puts remembered ground in the dark band at two tones', () => {
    const { ctx, fills } = fakeContext();
    const visible = new Set([at(0, 0)]);
    const surveyed = new Set([at(0, 0), at(1, 0), at(2, 0)]);

    drawWash(ctx, world, view, { visible, surveyed, mode: 'two' }, DEFAULT_THEME);

    expect(fills).toHaveLength(1);
    expect(fills[0]!.hexes).toBe(15);
    expect(fills[0]!.alpha).toBe(DEFAULT_THEME.wash.unseen);
  });

  it('draws nothing when it is off, and nothing for a referee', () => {
    const off = fakeContext();
    drawWash(off.ctx, world, view, {
      visible: new Set([at(0, 0)]),
      surveyed: new Set(),
      mode: 'off',
    }, DEFAULT_THEME);
    expect(off.fills).toEqual([]);

    const referee = fakeContext();
    drawWash(referee.ctx, world, view, {
      visible: new Set(),
      surveyed: new Set(),
      mode: 'three',
    }, DEFAULT_THEME);
    expect(referee.fills).toEqual([]);
  });
});

describe('hexSubPath', () => {
  it('appends rather than starting over, which is the whole reason it exists', () => {
    const path = new FakePath();
    hexSubPath(path as unknown as Path2D, { x: 0, y: 0 }, 5);
    hexSubPath(path as unknown as Path2D, { x: 20, y: 0 }, 5);
    expect(path.moves).toBe(2);
  });
});
