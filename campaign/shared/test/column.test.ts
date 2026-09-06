/**
 * The column.
 *
 * The two worked examples in the rules are the only golden numbers in this suite, and
 * they earn it: they come from the specification rather than from this implementation, so
 * reproducing them is evidence the formula was read correctly rather than evidence the
 * code still does what it did yesterday.
 */

import { describe, expect, it } from 'vitest';

import {
  advanceColumn,
  catchupHours,
  columnCovers,
  columnHexes,
  columnIsConnected,
  columnLengthKm,
  effectiveLengthKm,
  occupied,
} from '../src/column.js';
import type { Hex } from '../src/hex.js';
import type { Unit } from '../src/unit.js';

const base: Unit = {
  id: 'u',
  faction: 'red',
  kind: 'infantry',
  effectives: 4000,
  fatigue: 0,
  experience: 0,
  morale: 30,
  provisions: 40,
  maxProvisions: 40,
  equipment: 30,
  maxEquipment: 30,
  guns: 6,
  marchSpeedKmh: 3,
  spacingM: 0.5,
  spacingMultiplier: 1.3,
  traits: [],
  formation: 'march',
  column: [{ q: 0, r: 0 }],
  hoursMarchedToday: 0,
  corps: null,
};

/** Rules, "Regular Infantry Division": 4000 at 0.5 m, SM 130%, 6 guns. */
const REGULAR_INFANTRY: Unit = base;

/** Rules, "Guards Cuirassiers Division": 4000 at 3 m, SM 150%, 6 guns, 4 km/h. */
const GUARDS_CAVALRY: Unit = {
  ...base,
  id: 'c',
  kind: 'cavalry',
  spacingM: 3,
  spacingMultiplier: 1.5,
  marchSpeedKmh: 4,
};

describe("the rules' worked examples", () => {
  // The rules print these truncated, not rounded: 2.6004 km appears as "2.6 km" and
  // 0.8668 h as "0.86 h", where rounding would have given 0.87. Asserted as the interval
  // the printed figure stands for, so the test says what agreement actually means.

  it('gives the infantry division a 2.6 km column and 0.86 h catchup', () => {
    expect(columnLengthKm(REGULAR_INFANTRY)).toBeGreaterThanOrEqual(2.6);
    expect(columnLengthKm(REGULAR_INFANTRY)).toBeLessThan(2.7);

    const catchup = catchupHours(REGULAR_INFANTRY, 3);
    expect(catchup).toBeGreaterThanOrEqual(0.86);
    expect(catchup).toBeLessThan(0.87);
  });

  it('gives the cavalry division an 18 km column and 4.5 h catchup', () => {
    expect(columnLengthKm(GUARDS_CAVALRY)).toBeGreaterThanOrEqual(18);
    expect(columnLengthKm(GUARDS_CAVALRY)).toBeLessThan(18.1);

    const catchup = catchupHours(GUARDS_CAVALRY, 4);
    expect(catchup).toBeGreaterThanOrEqual(4.5);
    expect(catchup).toBeLessThan(4.6);
  });

  it('reads the guns term in metres, like the rest of the formula', () => {
    // The one ambiguity in the rules' formula. Five centimetres a gun is not a gun and
    // 0.05 was probably meant as kilometres — but taken as metres it reproduces both
    // examples to the digit, and taken as kilometres it does not.
    const withGuns = columnLengthKm({ ...REGULAR_INFANTRY, guns: 6 });
    const withoutGuns = columnLengthKm({ ...REGULAR_INFANTRY, guns: 0 });
    expect(withGuns).toBeCloseTo(2.6004, 4);
    expect(withoutGuns).toBeCloseTo(2.6, 4);
    // Read as kilometres the guns would add 390 m here and the example would read 3.0.
    expect(withGuns - withoutGuns).toBeLessThan(0.001);
  });
});

describe('columnLengthKm', () => {
  it('scales with the number of men', () => {
    const small = columnLengthKm({ ...base, effectives: 4000 });
    const large = columnLengthKm({ ...base, effectives: 8000 });
    expect(large).toBeCloseTo(small * 2, 2);
  });

  it('scales with spacing, which is why cavalry are so much longer', () => {
    const foot = columnLengthKm({ ...base, spacingM: 0.5 });
    const horse = columnLengthKm({ ...base, spacingM: 3 });
    expect(horse).toBeCloseTo(foot * 6, 2);
  });

  it('applies the baggage multiplier', () => {
    const tight = columnLengthKm({ ...base, spacingMultiplier: 1 });
    const trailing = columnLengthKm({ ...base, spacingMultiplier: 2 });
    expect(trailing).toBeCloseTo(tight * 2, 2);
  });
});

describe('effectiveLengthKm', () => {
  it('halves the column on a highway', () => {
    // A metalled road takes a column several abreast where a track takes it in file.
    const full = columnLengthKm(REGULAR_INFANTRY);
    expect(effectiveLengthKm(REGULAR_INFANTRY, 'highway')).toBeCloseTo(full / 2, 5);
    expect(effectiveLengthKm(REGULAR_INFANTRY, 'road')).toBeCloseTo(full, 5);
    expect(effectiveLengthKm(REGULAR_INFANTRY, 'off_road')).toBeCloseTo(full, 5);
    expect(effectiveLengthKm(REGULAR_INFANTRY, 'bad_going')).toBeCloseTo(full, 5);
  });
});

describe('columnHexes', () => {
  it('is at least one — a unit is always somewhere', () => {
    expect(columnHexes({ ...base, effectives: 1, guns: 0 })).toBe(1);
  });

  it('rounds up, because a column spilling into a hex is in it', () => {
    expect(columnHexes(REGULAR_INFANTRY)).toBe(3);
  });

  it('puts an 18 km cavalry column across eighteen hexes', () => {
    // The fact the whole game turns on: this division is exposed along 18 km of road.
    expect(columnHexes(GUARDS_CAVALRY)).toBe(19);
  });
});

describe('occupied', () => {
  const path: Hex[] = Array.from({ length: 30 }, (_, i) => ({ q: i, r: 0 }));

  it('covers the head and as much tail as the column reaches', () => {
    const unit = { ...GUARDS_CAVALRY, column: path };
    const cells = occupied(unit);
    expect(cells).toHaveLength(columnHexes(unit));
    expect(cells[0]).toEqual({ q: 0, r: 0 });
  });

  it('takes what it has when the unit has only just been placed', () => {
    const unit = { ...GUARDS_CAVALRY, column: [{ q: 5, r: 5 }] };
    expect(occupied(unit)).toEqual([{ q: 5, r: 5 }]);
  });

  it('shortens on a highway', () => {
    const unit = { ...GUARDS_CAVALRY, column: path };
    expect(occupied(unit, 'highway').length).toBeLessThan(occupied(unit, 'road').length);
  });
});

describe('columnCovers', () => {
  it('finds the tail as well as the head', () => {
    // Recon and interception both work against the whole column, not against the head.
    const unit = { ...GUARDS_CAVALRY, column: Array.from({ length: 30 }, (_, i) => ({ q: i, r: 0 })) };
    expect(columnCovers(unit, { q: 0, r: 0 })).toBe(true);
    expect(columnCovers(unit, { q: 10, r: 0 })).toBe(true);
    expect(columnCovers(unit, { q: 25, r: 0 })).toBe(false);
  });
});

describe('advanceColumn', () => {
  it('puts the new hex at the head', () => {
    const next = advanceColumn(REGULAR_INFANTRY, { q: 1, r: 0 });
    expect(next[0]).toEqual({ q: 1, r: 0 });
    expect(next[1]).toEqual({ q: 0, r: 0 });
  });

  it('keeps the stored path bounded', () => {
    // A unit marching for a week would otherwise accumulate an unbounded history that
    // serialises into every save.
    let unit = REGULAR_INFANTRY;
    for (let i = 1; i <= 200; i++) {
      unit = { ...unit, column: advanceColumn(unit, { q: i, r: 0 }) };
    }
    expect(unit.column.length).toBeLessThan(10);
  });

  it('keeps enough tail to survive a spell on a highway', () => {
    let unit = { ...GUARDS_CAVALRY };
    for (let i = 1; i <= 40; i++) {
      unit = { ...unit, column: advanceColumn(unit, { q: i, r: 0 }, 'highway') };
    }
    // Back onto ordinary road, the full column is still accounted for.
    expect(unit.column.length).toBeGreaterThanOrEqual(columnHexes(unit, 'road'));
  });
});

describe('columnIsConnected', () => {
  it('accepts a column marched hex by hex', () => {
    let unit = REGULAR_INFANTRY;
    for (let i = 1; i <= 10; i++) {
      unit = { ...unit, column: advanceColumn(unit, { q: i, r: 0 }) };
    }
    expect(columnIsConnected(unit)).toBe(true);
  });

  it('rejects a column with a gap in it', () => {
    const unit = { ...REGULAR_INFANTRY, column: [{ q: 0, r: 0 }, { q: 9, r: 9 }] };
    expect(columnIsConnected(unit)).toBe(false);
  });
});
