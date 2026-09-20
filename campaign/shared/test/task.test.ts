/**
 * Which waypoints a column has behind it.
 *
 * Small, and worth its own file: the scheduler and the reducer both count waypoints off
 * through these two functions rather than each deciding for itself, and a disagreement
 * between them would route a column back through ground it had already passed. The
 * marching itself is covered in `scheduler.test.ts`.
 */

import { describe, expect, it } from 'vitest';

import type { Hex } from '../src/hex.js';
import { isRunning, viaAhead, viaIndexAt, type Task } from '../src/task.js';

const task = (via: readonly Hex[], viaIndex = 0): Task => ({
  unitId: 'red-1',
  destination: { q: 9, r: 9 },
  via,
  setAtHours: 6,
  fromDespatchId: null,
  nextHex: { q: 1, r: 0 },
  progressHours: 0,
  complete: false,
  viaIndex,
});

const A = { q: 3, r: 3 };
const B = { q: 6, r: 6 };

describe('viaIndexAt', () => {
  it('counts nothing off on ground that is not a waypoint', () => {
    expect(viaIndexAt(task([A, B]), { q: 1, r: 1 })).toBe(0);
  });

  it('counts a waypoint off when the head stands on it', () => {
    expect(viaIndexAt(task([A, B]), A)).toBe(1);
  });

  it('clears two waypoints named on the same hex', () => {
    // Otherwise the second sits forever on ground the column is already standing on, and
    // the march never ends.
    expect(viaIndexAt(task([A, A, B]), A)).toBe(2);
  });

  it('does not count off a later waypoint the route happens to cross early', () => {
    // Order is the whole meaning of "by way of A, then B". Standing on B before A is
    // reached counts nothing: the column still has to make A.
    expect(viaIndexAt(task([A, B]), B)).toBe(0);
  });

  it('never counts back', () => {
    // Having passed both, standing on the first again is not a waypoint still to make.
    expect(viaIndexAt(task([A, B], 2), A)).toBe(2);
  });

  it('is unmoved by a march with no waypoints at all', () => {
    expect(viaIndexAt(task([]), A)).toBe(0);
  });
});

describe('viaAhead', () => {
  it('is everything still to make', () => {
    expect(viaAhead(task([A, B]), { q: 1, r: 1 })).toEqual([A, B]);
    expect(viaAhead(task([A, B]), A)).toEqual([B]);
    expect(viaAhead(task([A, B], 1), B)).toEqual([]);
  });

  it('is empty for a march that named none', () => {
    expect(viaAhead(task([]), A)).toEqual([]);
  });
});

describe('isRunning', () => {
  it('is true while a column has somewhere to be', () => {
    expect(isRunning(task([A]))).toBe(true);
    expect(isRunning({ ...task([A]), complete: true })).toBe(false);
    expect(isRunning({ ...task([A]), nextHex: null })).toBe(false);
  });
});
