/**
 * The order of battle as a chain of command.
 *
 * Two things matter. The shape: officers under the officer they answer to, each with the
 * formation they ride with hung beneath them, and a tree that survives a bad log. And which
 * facts a role is shown: a commander reading a live position for a formation they have not
 * heard from in six hours would be reading the thing this whole design exists to withhold.
 */

import { describe, expect, it } from 'vitest';

import type { Unit, UnitReport } from '@campaign/shared';

import { beneath, commandTrees, type CommandNode, type CommandTree } from '../src/roster.js';

const unit = (
  id: string,
  faction: string,
  at = { q: 1, r: 1 },
  parentUnitId: string | null = null,
): Unit => ({
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
  parentUnitId,
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

const officer = (
  id: string,
  unitId: string,
  superiorId: string | null,
  faction = 'red',
  unitName = `${unitId} Division`,
) => ({ id, name: `General ${id}`, faction, unitId, unitName, superiorId });

/** Every officer in a tree, depth first, as `id`s. */
const everyone = (t: CommandTree): string[] => {
  const walk = (n: CommandNode): string[] => [n.id, ...n.subordinates.flatMap(walk)];
  return t.roots.flatMap(walk);
};

const find = (t: CommandTree, id: string): CommandNode | undefined => {
  const walk = (n: CommandNode): CommandNode | undefined =>
    n.id === id ? n : n.subordinates.map(walk).find((x) => x !== undefined);
  return t.roots.map(walk).find((x) => x !== undefined);
};

/** Ney over Soult and Kellermann, Soult over Girard. Wellington opposite. */
const army = {
  commanders: [
    officer('ney', 'red-1', null),
    officer('soult', 'red-2', 'ney'),
    officer('kellermann', 'red-3', 'ney'),
    officer('girard', 'red-4', 'soult'),
    officer('wellington', 'blue-1', null, 'blue'),
  ],
  units: ['red-1', 'red-2', 'red-3', 'red-4'].map((id) => unit(id, 'red')).concat(unit('blue-1', 'blue')),
};

describe('the referee’s order of battle', () => {
  const trees = commandTrees({ factions: ['red', 'blue'], ...army, reports: [] });
  const red = trees[0]!;

  it('builds one tree per side, in the order asked for', () => {
    expect(trees.map((t) => t.faction)).toEqual(['red', 'blue']);
    expect(everyone(trees[1]!)).toEqual(['wellington']);
  });

  it('nests every officer under the officer they answer to', () => {
    expect(red.roots.map((r) => r.id)).toEqual(['ney']);
    expect(find(red, 'ney')!.subordinates.map((s) => s.id)).toEqual(['kellermann', 'soult']);
    expect(find(red, 'soult')!.subordinates.map((s) => s.id)).toEqual(['girard']);
    expect(find(red, 'girard')!.subordinates).toEqual([]);
  });

  it('hangs the formation each rides with under them, not their command', () => {
    // Ney rides with red-1 and commands the whole army. The army is under Ney, and red-1
    // is where Ney is: the two are not the same thing and are not drawn the same way.
    expect(find(red, 'ney')!.formation.unitId).toBe('red-1');
    expect(find(red, 'soult')!.formation.unitId).toBe('red-2');
  });

  it('holds every formation live, and dates none of them', () => {
    const ney = find(red, 'ney')!.formation;
    expect(ney.line?.unit).not.toBeNull();
    expect(ney.line?.asOfHours).toBeNull();
  });

  it('lists an empty side as empty, not as missing', () => {
    const [green] = commandTrees({ factions: ['green'], ...army, reports: [] });
    expect(green).toEqual({ faction: 'green', roots: [], uncommanded: [] });
  });

  it('shows a formation under each officer riding with it, saying who else is there', () => {
    // Soult's headquarters has gone and they have fallen back on Girard's division.
    const commanders = army.commanders.map((c) => (c.id === 'soult' ? { ...c, unitId: 'red-4' } : c));
    const [tree] = commandTrees({ factions: ['red'], commanders, units: army.units, reports: [] });

    expect(find(tree!, 'soult')!.formation.unitId).toBe('red-4');
    expect(find(tree!, 'soult')!.formation.alsoRiding).toEqual(['General girard']);
    expect(find(tree!, 'girard')!.formation.alsoRiding).toEqual(['General soult']);
    // red-2 is left with nobody.
    expect(tree!.uncommanded.map((f) => f.unitId)).toEqual(['red-2']);
  });

  it('puts a formation with one officer under them alone', () => {
    expect(find(red, 'ney')!.formation.alsoRiding).toEqual([]);
  });

  it('nests a patrol under the formation it came off', () => {
    const units = [...army.units, unit('red-3-p1', 'red', { q: 2, r: 2 }, 'red-3')];
    const [tree] = commandTrees({ factions: ['red'], commanders: army.commanders, units, reports: [] });

    expect(find(tree!, 'kellermann')!.formation.patrols.map((p) => p.unitId)).toEqual(['red-3-p1']);
    expect(tree!.uncommanded).toEqual([]);
  });

  it('lists formations nobody rides with apart, with their patrols', () => {
    const units = [
      ...army.units,
      unit('red-9', 'red'),
      unit('red-9-p1', 'red', { q: 2, r: 2 }, 'red-9'),
    ];
    const [tree] = commandTrees({ factions: ['red'], commanders: army.commanders, units, reports: [] });

    expect(tree!.uncommanded.map((f) => f.unitId)).toEqual(['red-9']);
    expect(tree!.uncommanded[0]!.patrols.map((p) => p.unitId)).toEqual(['red-9-p1']);
  });

  it('lists a patrol whose formation is gone on its own, rather than losing it', () => {
    const units = [...army.units, unit('lost-p1', 'red', { q: 2, r: 2 }, 'lost')];
    const [tree] = commandTrees({ factions: ['red'], commanders: army.commanders, units, reports: [] });
    expect(tree!.uncommanded.map((f) => f.unitId)).toEqual(['lost-p1']);
  });

  it('makes a root of an officer whose superior is on the other side, or gone', () => {
    const commanders = [
      ...army.commanders,
      officer('turncoat', 'red-5', 'wellington'),
      officer('bereaved', 'red-6', 'nobody'),
    ];
    const [tree] = commandTrees({ factions: ['red'], commanders, units: army.units, reports: [] });
    expect(tree!.roots.map((r) => r.id).sort()).toEqual(['bereaved', 'ney', 'turncoat']);
  });

  it('counts everyone beneath an officer, at every level, for a folded row', () => {
    expect(beneath(find(red, 'ney')!)).toBe(3);
    expect(beneath(find(red, 'soult')!)).toBe(1);
    expect(beneath(find(red, 'girard')!)).toBe(0);
  });

  it('marks each formation with its size, stated or guessed', () => {
    const units = [...army.units.slice(1), { ...unit('red-1', 'red'), kind: 'hq' as const, echelon: 'army_group' as const }];
    const [tree] = commandTrees({ factions: ['red'], commanders: army.commanders, units, reports: [] });
    expect(find(tree!, 'ney')!.formation.line?.echelon).toBe('army_group');
    // Guessed from strength when nobody said: 4,000 paper is a division.
    expect(find(tree!, 'soult')!.formation.line?.echelon).toBe('division');
  });

  it('lists everyone exactly once when the chain loops, and stops', () => {
    // `check` refuses a loop, but a log from an older build folds without it.
    const commanders = [
      officer('a', 'red-1', 'c'),
      officer('b', 'red-2', 'a'),
      officer('c', 'red-3', 'b'),
      officer('d', 'red-4', null),
    ];
    const [tree] = commandTrees({ factions: ['red'], commanders, units: army.units, reports: [] });
    expect(everyone(tree!).sort()).toEqual(['a', 'b', 'c', 'd']);
    // Broken at the lowest id.
    expect(tree!.roots.map((r) => r.id)).toContain('a');
  });
});

describe('a commander’s order of battle', () => {
  // Soult's view: their own division live, a report of Girard's, and no word of Ney's or
  // Kellermann's — which they know by name only.
  const [tree] = commandTrees({
    factions: ['red'],
    commanders: army.commanders.filter((c) => c.faction === 'red'),
    units: [unit('red-2', 'red', { q: 4, r: 4 })],
    reports: [report('red-4', 'red', 3)],
  });

  it('holds their own formation live', () => {
    const own = find(tree!, 'soult')!.formation;
    expect(own.line?.unit).not.toBeNull();
    expect(own.line?.asOfHours).toBeNull();
  });

  it('holds a subordinate’s formation as last heard, from the report', () => {
    const heard = find(tree!, 'girard')!.formation;
    expect(heard.line?.asOfHours).toBe(3);
    // A report carries no unit, so nothing downstream can reach for a live field through
    // a remembered row — the type is the guard rather than a convention.
    expect(heard.line?.unit).toBeNull();
    expect(heard.line?.at).toEqual({ q: 9, r: 9 });
  });

  it('knows a formation it has no word of by name, and nothing else', () => {
    const ney = find(tree!, 'ney')!.formation;
    expect(ney.name).toBe('red-1 Division');
    expect(ney.line).toBeNull();
  });

  it('never holds a formation twice, once true and once remembered', () => {
    const [both] = commandTrees({
      factions: ['red'],
      commanders: army.commanders.filter((c) => c.faction === 'red'),
      units: [unit('red-2', 'red', { q: 4, r: 4 })],
      reports: [report('red-2', 'red', 1)],
    });
    // The live one wins.
    expect(find(both!, 'soult')!.formation.line?.at).toEqual({ q: 4, r: 4 });
    expect(find(both!, 'soult')!.formation.line?.asOfHours).toBeNull();
  });

  it('does not invent a vacancy out of formations it has simply not heard of', () => {
    expect(tree!.uncommanded).toEqual([]);
  });
});
