/**
 * Raising a formation.
 *
 * What a new unit *is* matters more than the form that collects it: a division with the
 * wrong spacing is eighteen kilometres of road nobody notices until it will not fit
 * through a defile, and a morale above its ceiling is a number the rules cannot reduce.
 */

import { describe, expect, it } from 'vitest';

import { DEFAULT_CONFIG, columnLengthKm, maxMorale } from '@campaign/shared';

import { copy } from '../src/copy.js';
import { commanderFrom, draftProblems, emptyDraft, idFor, unitFrom } from '../src/orbat.js';

const cfg = DEFAULT_CONFIG;
const at = { q: 4, r: 4 };

describe('an id from a name', () => {
  it('is short, legible, and made of the name', () => {
    expect(idFor('1re Division', new Set())).toBe('1re-division');
    expect(idFor('The Duke’s Own', new Set())).toBe('the-duke-s-own');
  });

  it('strips accents rather than dropping the letters', () => {
    expect(idFor('Armée du Nord', new Set())).toBe('armee-du-nord');
  });

  it('only pays for a suffix when it has to', () => {
    expect(idFor('Guard', new Set(['guard']))).toBe('guard-2');
    expect(idFor('Guard', new Set(['guard', 'guard-2']))).toBe('guard-3');
  });

  it('always produces something, even from nothing', () => {
    expect(idFor('', new Set())).toBe(copy.orbat.fallbackId);
    expect(idFor('!!!', new Set())).toBe(copy.orbat.fallbackId);
  });
});

describe('what a draft has to say before it can be raised', () => {
  it('wants a name and a place', () => {
    const problems = draftProblems(emptyDraft('red'), cfg, new Set());
    expect(problems).toContain(copy.orbat.needsName);
    expect(problems).toContain(copy.orbat.needsGround);
  });

  it('refuses an id already on the map', () => {
    const draft = { ...emptyDraft('red'), name: 'Guard', id: 'guard', at };
    expect(draftProblems(draft, cfg, new Set(['guard']))).toContain(copy.orbat.idTaken('guard'));
  });

  it('is satisfied by a name, a commander and a hex', () => {
    const draft = { ...emptyDraft('red'), name: 'Guard', id: 'guard', commanderName: 'Mortier', at };
    expect(draftProblems(draft, cfg, new Set())).toEqual([]);
  });

  it('wants a commander', () => {
    const draft = { ...emptyDraft('red'), name: 'Guard', id: 'guard', at };
    expect(draftProblems(draft, cfg, new Set())).toEqual([copy.orbat.needsCommander]);
  });

  it('does not take whitespace for a commander', () => {
    const draft = { ...emptyDraft('red'), name: 'Guard', id: 'guard', commanderName: '   ', at };
    expect(draftProblems(draft, cfg, new Set())).toContain(copy.orbat.needsCommander);
  });

  it('does not need a superior: the first formation of a side is army command', () => {
    const draft = { ...emptyDraft('red'), name: 'Guard', id: 'guard', commanderName: 'Mortier', at };
    expect(draft.superiorId).toBe('');
    expect(draftProblems(draft, cfg, new Set())).toEqual([]);
  });
});

describe('the commander a draft appoints', () => {
  const appoint = (over: Partial<ReturnType<typeof emptyDraft>> = {}, taken = new Set<string>()) =>
    commanderFrom(
      { ...emptyDraft('red'), name: 'Guard', id: 'guard', commanderName: 'Marshal Mortier', ...over },
      'guard',
      taken,
    );

  it('rides with the formation being raised, on its side', () => {
    const c = appoint();
    expect(c.unitId).toBe('guard');
    expect(c.faction).toBe('red');
  });

  it('takes the name as written, less stray whitespace, and an id made of it', () => {
    const c = appoint({ commanderName: '  Marshal Mortier ' });
    expect(c.name).toBe('Marshal Mortier');
    expect(c.id).toBe('marshal-mortier');
  });

  it('does not reuse the id of a commander already appointed', () => {
    expect(appoint({}, new Set(['marshal-mortier'])).id).toBe('marshal-mortier-2');
  });

  it('answers to army command when no superior is chosen', () => {
    expect(appoint({ superiorId: '' }).superiorId).toBeNull();
  });

  it('answers to the superior chosen', () => {
    expect(appoint({ superiorId: 'ney' }).superiorId).toBe('ney');
  });

  it('passes orders down on its own until somebody is given the seat', () => {
    expect(appoint().autoCascade).toBe(true);
  });
});

describe('the formation a draft describes', () => {
  const raise = (over: Partial<ReturnType<typeof emptyDraft>> = {}) =>
    unitFrom({ ...emptyDraft('red'), name: 'Guard', id: 'guard', ...over }, cfg, at);

  it('takes its speed and spacing from the rules, not from the form', () => {
    expect(raise({ kind: 'cavalry' }).spacingM).toBe(cfg.kindDefaults.cavalry.spacingM);
    expect(raise({ kind: 'infantry' }).marchSpeedKmh).toBe(cfg.kindDefaults.infantry.marchSpeedKmh);
  });

  it('starts fresh: no fatigue, full supply, morale at its ceiling', () => {
    const veteran = raise({ experience: 1 });
    expect(veteran.fatigue).toBe(0);
    expect(veteran.provisions).toBe(veteran.maxProvisions);
    expect(veteran.equipment).toBe(veteran.maxEquipment);
    // The ceiling is what experience buys, and a unit that has not been fought has not
    // lost any of it.
    expect(veteran.morale).toBe(maxMorale(veteran, cfg.maxMorale));
  });

  it('stands where it was pointed at, and nowhere else yet', () => {
    expect(raise().column).toEqual([at]);
    expect(raise().hoursMarchedToday).toBe(0);
    expect(raise().formationChange).toBeNull();
  });

  it('makes a long tail an actual tail', () => {
    // Otherwise the trait is decorative, which is what it was before anything raised a
    // unit with it.
    const plain = raise({ paperStrength: 6000 });
    const trailing = raise({ paperStrength: 6000, traits: ['long_tail'] });
    expect(columnLengthKm(trailing)).toBeGreaterThan(columnLengthKm(plain));
  });

  it('is a formation in its own right, not somebody’s detachment', () => {
    expect(raise().parentUnitId).toBeNull();
  });

  it('gives guns to an artillery reserve, and a handful to everyone else', () => {
    expect(raise({ kind: 'artillery_reserve' }).guns).toBeGreaterThan(raise().guns);
  });
});
