/**
 * The chain of command.
 *
 * One field carries the whole hierarchy, so these are mostly about what that field is not
 * allowed to become. A loop in the chain makes every tree walk in the engine
 * non-terminating, and a state that cannot be walked cannot be drawn, ordered through, or
 * reasoned about — so the guards are tested from both sides: `check` refuses to create a
 * cycle, and the walks survive one that got in anyway.
 *
 * That second half is not paranoia for its own sake. A log written by an older build folds
 * into today's state without passing through today's `check`, so the walks have to be
 * total over states the checks would never have permitted.
 */

import { describe, expect, it } from 'vitest';

import {
  commanderIds,
  commanderOf,
  directSubordinates,
  formationOf,
  formationsUnder,
  mayOrder,
  mayWriteTo,
  ridersOf,
  subordinates,
  superiors,
  uncommanded,
  wouldCycle,
  type Commander,
} from '../src/commander.js';
import { EMPTY_STATE, type CampaignState } from '../src/state.js';
import { KIND_DEFAULTS, type Unit } from '../src/unit.js';

const commander = (
  id: string,
  faction: string,
  unitId: string,
  superiorId: string | null = null,
): Commander => ({
  id,
  name: `General ${id}`,
  faction,
  unitId,
  superiorId,
  autoCascade: true,
});

const unit = (id: string, faction: string, parentUnitId: string | null = null): Unit => ({
  id,
  name: `${id} Division`,
  faction,
  kind: 'infantry',
  paperStrength: 5000,
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
  column: [{ q: 0, r: 0 }],
  hoursMarchedToday: 0,
  corps: null,
  parentUnitId,
});

/**
 * An army: Ney over Kellermann and Soult, Soult over Girard. Wellington opposite.
 *
 * Three levels deep on one side, because two would not tell a transitive walk apart from
 * a direct one.
 */
function army(): CampaignState {
  const commanders: Commander[] = [
    commander('ney', 'red', 'red-1'),
    commander('kellermann', 'red', 'red-2', 'ney'),
    commander('soult', 'red', 'red-3', 'ney'),
    commander('girard', 'red', 'red-4', 'soult'),
    commander('wellington', 'blue', 'blue-1'),
  ];
  const units = ['red-1', 'red-2', 'red-3', 'red-4'].map((id) => unit(id, 'red'));
  units.push(unit('blue-1', 'blue'));

  return {
    ...EMPTY_STATE,
    commanders: new Map(commanders.map((c) => [c.id, c])),
    units: new Map(units.map((u) => [u.id, u])),
  };
}

describe('the tree', () => {
  const s = army();

  it('lists commanders in a fixed order, whatever order they arrived in', () => {
    expect(commanderIds(s)).toEqual([
      'girard',
      'kellermann',
      'ney',
      'soult',
      'wellington',
    ]);
  });

  it('reports direct subordinates only', () => {
    expect(directSubordinates(s, 'ney').map((c) => c.id)).toEqual(['kellermann', 'soult']);
    expect(directSubordinates(s, 'soult').map((c) => c.id)).toEqual(['girard']);
    expect(directSubordinates(s, 'girard')).toEqual([]);
  });

  it('reports everyone beneath a commander, at any depth', () => {
    expect(subordinates(s, 'ney').map((c) => c.id).sort()).toEqual([
      'girard',
      'kellermann',
      'soult',
    ]);
    expect(subordinates(s, 'soult').map((c) => c.id)).toEqual(['girard']);
  });

  it('never counts a commander among their own subordinates', () => {
    expect(subordinates(s, 'ney').some((c) => c.id === 'ney')).toBe(false);
  });

  it('walks upward to the army commander', () => {
    expect(superiors(s, 'girard').map((c) => c.id)).toEqual(['soult', 'ney']);
    expect(superiors(s, 'ney')).toEqual([]);
  });
});

describe('who may be written to', () => {
  const s = army();

  it('lets a commander order anyone beneath them, at any depth', () => {
    expect(mayOrder(s, 'ney', 'kellermann')).toBe(true);
    // Past a level: Napoleon wrote directly to divisions constantly, and the skipped
    // commander simply does not find out.
    expect(mayOrder(s, 'ney', 'girard')).toBe(true);
  });

  it('refuses an order upward or sideways', () => {
    expect(mayOrder(s, 'girard', 'ney')).toBe(false);
    expect(mayOrder(s, 'kellermann', 'soult')).toBe(false);
    expect(mayOrder(s, 'ney', 'ney')).toBe(false);
  });

  it('lets a commander write to anyone on their own side', () => {
    // Lateral coordination between corps commanders was real and mattered enormously.
    expect(mayWriteTo(s, 'kellermann', 'soult')).toBe(true);
    expect(mayWriteTo(s, 'girard', 'ney')).toBe(true);
  });

  it('refuses a despatch to the enemy or to themselves', () => {
    expect(mayWriteTo(s, 'ney', 'wellington')).toBe(false);
    expect(mayWriteTo(s, 'ney', 'ney')).toBe(false);
    expect(mayWriteTo(s, 'ney', 'nobody')).toBe(false);
  });
});

describe('formations', () => {
  const s = army();

  it('finds the one a commander rides with', () => {
    expect(formationOf(s, 'ney')?.id).toBe('red-1');
    expect(formationOf(s, 'nobody')).toBeUndefined();
  });

  it('collects their own and every subordinate\'s', () => {
    expect(formationsUnder(s, 'ney').map((u) => u.id)).toEqual([
      'red-1',
      'red-2',
      'red-3',
      'red-4',
    ]);
    expect(formationsUnder(s, 'soult').map((u) => u.id)).toEqual(['red-3', 'red-4']);
  });

  it('lists a formation once when two commanders ride with it', () => {
    // A corps commander whose headquarters has been destroyed falls back on one of their
    // divisions, and both they and its own commander are then with the same column.
    const shared: CampaignState = {
      ...s,
      commanders: new Map(s.commanders).set(
        'soult',
        commander('soult', 'red', 'red-2', 'ney'),
      ),
    };
    const ids = formationsUnder(shared, 'ney').map((u) => u.id);
    expect(ids).toEqual([...new Set(ids)]);
  });

  it('skips a formation that no longer exists', () => {
    const bereaved: CampaignState = { ...s, units: new Map() };
    expect(formationsUnder(bereaved, 'ney')).toEqual([]);
  });
});

describe('cycles', () => {
  const s = army();

  it('sees the loop a reassignment would close', () => {
    // Making Ney answer to their own subordinate closes the chain on itself.
    expect(wouldCycle(s, 'ney', 'girard')).toBe(true);
    expect(wouldCycle(s, 'ney', 'ney')).toBe(true);
  });

  it('permits a legitimate reassignment', () => {
    expect(wouldCycle(s, 'girard', 'kellermann')).toBe(false);
    expect(wouldCycle(s, 'girard', null)).toBe(false);
  });

  it('terminates on a state that already contains one', () => {
    // `check` refuses to create this, but a log written by an older build folds into
    // today's state without passing through today's checks, so the walks must be total
    // over states the checks would never have permitted. Without the guards these three
    // calls do not return.
    const looped: CampaignState = {
      ...s,
      commanders: new Map(s.commanders).set('ney', commander('ney', 'red', 'red-1', 'girard')),
    };

    expect(subordinates(looped, 'ney').length).toBeLessThan(10);
    expect(superiors(looped, 'girard').length).toBeLessThan(10);
    expect(formationsUnder(looped, 'ney').length).toBeLessThan(10);
  });
});

describe('who rides with a formation', () => {
  const s = army();

  it('is the one commander appointed to it', () => {
    expect(ridersOf(s, 'red-3').map((c) => c.id)).toEqual(['soult']);
  });

  it('is nobody for a formation nobody was appointed to, or that does not exist', () => {
    const vacant: CampaignState = {
      ...s,
      units: new Map(s.units).set('red-5', unit('red-5', 'red')),
    };
    expect(ridersOf(vacant, 'red-5')).toEqual([]);
    expect(ridersOf(s, 'nowhere')).toEqual([]);
  });

  it('puts the senior officer first when two share a column', () => {
    // Soult's headquarters is gone and they have fallen back on Girard's division. Soult
    // is higher in the chain, so a despatch addressed to the formation is theirs.
    const fallenBack: CampaignState = {
      ...s,
      commanders: new Map(s.commanders).set('soult', commander('soult', 'red', 'red-4', 'ney')),
    };
    expect(ridersOf(fallenBack, 'red-4').map((c) => c.id)).toEqual(['soult', 'girard']);
  });

  it('breaks a tie in seniority on id, not on the order of appointment', () => {
    const equals = (first: string, second: string): CampaignState => ({
      ...s,
      commanders: new Map([
        [first, commander(first, 'red', 'red-2', 'ney')],
        [second, commander(second, 'red', 'red-2', 'ney')],
        ['ney', commander('ney', 'red', 'red-1')],
      ]),
    });
    expect(ridersOf(equals('b', 'a'), 'red-2').map((c) => c.id)).toEqual(['a', 'b']);
    expect(ridersOf(equals('a', 'b'), 'red-2').map((c) => c.id)).toEqual(['a', 'b']);
  });
});

describe('who commands a unit', () => {
  const s = army();

  it('is the officer riding with it', () => {
    expect(commanderOf(s, 'red-2')?.id).toBe('kellermann');
  });

  it('is the senior of two riding with it', () => {
    const fallenBack: CampaignState = {
      ...s,
      commanders: new Map(s.commanders).set('soult', commander('soult', 'red', 'red-4', 'ney')),
    };
    expect(commanderOf(fallenBack, 'red-4')?.id).toBe('soult');
  });

  it('is, for a patrol, the commander of the formation it came off', () => {
    const patrolled: CampaignState = {
      ...s,
      units: new Map(s.units)
        .set('red-2-p1', unit('red-2-p1', 'red', 'red-2'))
        .set('red-2-p1-p1', unit('red-2-p1-p1', 'red', 'red-2-p1')),
    };
    expect(commanderOf(patrolled, 'red-2-p1')?.id).toBe('kellermann');
    // Transitively, however far down the detachments go.
    expect(commanderOf(patrolled, 'red-2-p1-p1')?.id).toBe('kellermann');
  });

  it('is nobody for a vacant formation, or a patrol off one', () => {
    const vacant: CampaignState = {
      ...s,
      units: new Map(s.units)
        .set('red-5', unit('red-5', 'red'))
        .set('red-5-p1', unit('red-5-p1', 'red', 'red-5')),
    };
    expect(commanderOf(vacant, 'red-5')).toBeUndefined();
    expect(commanderOf(vacant, 'red-5-p1')).toBeUndefined();
  });

  it('is nobody for a unit that does not exist, or a patrol whose parent is gone', () => {
    const orphaned: CampaignState = {
      ...s,
      units: new Map(s.units).set('lost-p1', unit('lost-p1', 'red', 'lost')),
    };
    expect(commanderOf(s, 'nowhere')).toBeUndefined();
    expect(commanderOf(orphaned, 'lost-p1')).toBeUndefined();
  });

  it('terminates on a loop of patrols', () => {
    // Nothing should create one, but the walk follows `parentUnitId` and a state folded
    // from an older log has not been through today's checks.
    const looped: CampaignState = {
      ...s,
      units: new Map(s.units)
        .set('p1', unit('p1', 'red', 'p2'))
        .set('p2', unit('p2', 'red', 'p1')),
    };
    expect(commanderOf(looped, 'p1')).toBeUndefined();
  });
});

describe('formations nobody commands', () => {
  const s = army();

  it('are none in an army where every formation has an officer', () => {
    expect(uncommanded(s)).toEqual([]);
  });

  it('are found, in id order', () => {
    const vacant: CampaignState = {
      ...s,
      units: new Map(s.units).set('red-9', unit('red-9', 'red')).set('red-5', unit('red-5', 'red')),
    };
    expect(uncommanded(vacant).map((u) => u.id)).toEqual(['red-5', 'red-9']);
  });

  it('include a formation whose only commander has been removed', () => {
    const commanders = new Map(s.commanders);
    commanders.delete('kellermann');
    expect(uncommanded({ ...s, commanders }).map((u) => u.id)).toEqual(['red-2']);
  });

  it('do not include a patrol, which answers to its parent', () => {
    const patrolled: CampaignState = {
      ...s,
      units: new Map(s.units).set('red-2-p1', unit('red-2-p1', 'red', 'red-2')),
    };
    expect(uncommanded(patrolled)).toEqual([]);
  });

  it('do not include a patrol off a vacant formation — the formation is the vacancy', () => {
    const vacant: CampaignState = {
      ...s,
      units: new Map(s.units)
        .set('red-5', unit('red-5', 'red'))
        .set('red-5-p1', unit('red-5-p1', 'red', 'red-5')),
    };
    expect(uncommanded(vacant).map((u) => u.id)).toEqual(['red-5']);
  });
});
