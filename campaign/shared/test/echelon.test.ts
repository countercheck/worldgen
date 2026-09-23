/**
 * Echelons: the size mark above a formation's symbol.
 *
 * Display only — no rule keys off one — so what matters is that the list a form offers,
 * the marks the map draws, and the guess made when nobody said all agree.
 */

import { describe, expect, it } from 'vitest';

import { ECHELON_MARKS, ECHELONS, echelonOf, KIND_DEFAULTS, type Unit } from '../src/unit.js';

const unit = (over: Partial<Unit> = {}): Unit => ({
  id: 'u',
  name: 'U',
  faction: 'red',
  kind: 'infantry',
  paperStrength: 6000,
  fatigue: 0,
  experience: 0,
  morale: 30,
  provisions: 40,
  maxProvisions: 40,
  equipment: 30,
  maxEquipment: 30,
  guns: 6,
  marchSpeedKmh: KIND_DEFAULTS.infantry.marchSpeedKmh,
  spacingM: KIND_DEFAULTS.infantry.spacingM,
  spacingMultiplier: 1.3,
  traits: [],
  formation: 'march',
  formationChange: null,
  column: [{ q: 0, r: 0 }],
  hoursMarchedToday: 0,
  corps: null,
  parentUnitId: null,
  ...over,
});

describe('echelons', () => {
  it('offers exactly the echelons the map can mark, smallest first', () => {
    expect([...ECHELONS].sort()).toEqual(Object.keys(ECHELON_MARKS).sort());
    expect(ECHELONS.slice(-3)).toEqual(['corps', 'army', 'army_group']);
  });

  it('marks an army with four crosses and an army group with five', () => {
    expect(ECHELON_MARKS.army).toBe('XXXX');
    expect(ECHELON_MARKS.army_group).toBe('XXXXX');
  });

  it('takes a stated echelon over any guess, including the top two', () => {
    expect(echelonOf(unit({ kind: 'hq', paperStrength: 400, echelon: 'army' }))).toBe('army');
    expect(echelonOf(unit({ kind: 'hq', paperStrength: 400, echelon: 'army_group' }))).toBe(
      'army_group',
    );
  });

  it('never guesses an army: an unstated headquarters is drawn at corps', () => {
    // An army headquarters is a few hundred staff; strength cannot tell it from a corps HQ.
    expect(echelonOf(unit({ kind: 'hq', paperStrength: 400 }))).toBe('corps');
    expect(echelonOf(unit({ paperStrength: 90000 }))).toBe('division');
  });
});
