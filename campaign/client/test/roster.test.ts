/**
 * What the order-of-battle drawer lists.
 *
 * The interesting part is not the table, it is which facts a role is shown. A commander
 * reading a live position for a formation he has not heard from in six hours would be
 * reading the thing this whole design exists to withhold.
 */

import { describe, expect, it } from 'vitest';

import type { Unit, UnitReport } from '@campaign/shared';

import { copy } from '../src/copy.js';
import { rosterGroups } from '../src/roster.js';

const unit = (id: string, faction: string, at = { q: 1, r: 1 }): Unit => ({
  id,
  name: `${id} Division`,
  faction,
  kind: 'infantry',
  paperStrength: 4000,
  fatigue: 10,
  experience: 0,
  morale: 30,
  provisions: 40,
  maxProvisions: 40,
  equipment: 30,
  maxEquipment: 30,
  guns: 6,
  marchSpeedKmh: 3,
  spacingM: 0.5,
  spacingMultiplier: 1,
  traits: [],
  formation: 'march',
  formationChange: null,
  column: [at],
  hoursMarchedToday: 0,
  corps: null,
  parentUnitId: null,
});

const report = (id: string, faction: string, atHours: number): UnitReport => ({
  unitId: id,
  name: `${id} Division`,
  faction,
  kind: 'infantry',
  echelon: 'division',
  atHours,
  head: { q: 9, r: 9 },
  paperStrength: 4000,
  fatigue: 20,
  formation: 'march',
  provisions: 30,
  corps: null,
});

const factionName = (id: string) => (id === 'red' ? 'Armée du Nord' : 'Coalition');

describe('the referee’s order of battle', () => {
  it('groups every formation by side', () => {
    const groups = rosterGroups({
      role: 'referee',
      units: [unit('blue-1', 'blue'), unit('red-2', 'red'), unit('red-1', 'red')],
      reports: [],
      factionName,
    });

    expect(groups.map((g) => g.title)).toEqual(['Coalition', 'Armée du Nord']);
    expect(groups.find((g) => g.title === 'Armée du Nord')!.lines.map((l) => l.unitId)).toEqual([
      'red-1',
      'red-2',
    ]);
  });

  it('dates nothing, because a referee is never reading the past', () => {
    const groups = rosterGroups({
      role: 'referee',
      units: [unit('red-1', 'red')],
      reports: [],
      factionName,
    });
    for (const line of groups.flatMap((g) => g.lines)) {
      expect(line.asOfHours).toBeNull();
      expect(line.unit).not.toBeNull();
    }
  });
});

describe('a commander’s order of battle', () => {
  it('separates what he can see from what he was told', () => {
    const groups = rosterGroups({
      role: 'commander',
      units: [unit('red-1', 'red')],
      reports: [report('red-2', 'red', 3), report('red-3', 'red', 1)],
      factionName,
    });

    expect(groups.map((g) => g.title)).toEqual([
      copy.roster.groupWithYou,
      copy.roster.groupUnderCommand,
    ]);
    expect(groups[0]!.lines.map((l) => l.unitId)).toEqual(['red-1']);
    expect(groups[1]!.lines.map((l) => l.unitId)).toEqual(['red-2', 'red-3']);
  });

  it('marks every remembered row with the hour it describes', () => {
    const groups = rosterGroups({
      role: 'commander',
      units: [unit('red-1', 'red')],
      reports: [report('red-2', 'red', 3)],
      factionName,
    });

    const own = groups[0]!.lines[0]!;
    const heard = groups[1]!.lines[0]!;

    expect(own.asOfHours).toBeNull();
    expect(own.unit).not.toBeNull();
    // A report carries no unit, so nothing downstream can reach for a live field through
    // a remembered row — the type is the guard rather than a convention.
    expect(heard.asOfHours).toBe(3);
    expect(heard.unit).toBeNull();
  });

  it('never lists a formation twice, once true and once remembered', () => {
    // His own formation reports itself. Showing both rows would leave a reader with two
    // positions for one division and no way to tell which to believe.
    const groups = rosterGroups({
      role: 'commander',
      units: [unit('red-1', 'red', { q: 4, r: 4 })],
      reports: [report('red-1', 'red', 2), report('red-2', 'red', 2)],
      factionName,
    });

    const ids = groups.flatMap((g) => g.lines.map((l) => l.unitId));
    expect(ids).toEqual(['red-1', 'red-2']);
    // And the one that survived is the live one.
    expect(groups[0]!.lines[0]!.at).toEqual({ q: 4, r: 4 });
  });

  it('says what the hours mean, but only when there are any', () => {
    const withNone = rosterGroups({
      role: 'commander',
      units: [unit('red-1', 'red')],
      reports: [],
      factionName,
    });
    expect(withNone[1]!.note).toBeNull();

    const withSome = rosterGroups({
      role: 'commander',
      units: [unit('red-1', 'red')],
      reports: [report('red-2', 'red', 3)],
      factionName,
    });
    expect(withSome[1]!.note).toContain('when word reached you');
  });

  it('takes the report’s position, not a live one', () => {
    const groups = rosterGroups({
      role: 'commander',
      units: [],
      reports: [report('red-2', 'red', 3)],
      factionName,
    });
    expect(groups[1]!.lines[0]!.at).toEqual({ q: 9, r: 9 });
  });
});
