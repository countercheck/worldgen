/**
 * Filing sightings: one contact or two?
 *
 * The whole of this module is one judgement made on a commander's behalf, because they are
 * not allowed the information it needs. The engine knows which formation was seen; they
 * must not, or they could correlate two sightings hours apart for free. So identity is
 * decided here, by continuity, and these tests are about where that line falls.
 */

import { describe, expect, it } from 'vitest';

import type { Commander } from '../src/commander.js';
import { DEFAULT_CONFIG } from '../src/config.js';
import type { EventPayload } from '../src/events.js';
import { key, type Hex, type HexKey } from '../src/hex.js';
import { fileSightings, sightingEvents } from '../src/knowledge.js';
import { contactFrom, SIGHTING_INTEL, type Sighting } from '../src/recon.js';
import { EMPTY_STATE, reduce, type CampaignState } from '../src/state.js';
import type { Unit, UnitKind } from '../src/unit.js';
import type { World, WorldHex } from '../src/world.js';

const cfg = DEFAULT_CONFIG;

function flatWorld(size = 30): World {
  const hexes = new Map<HexKey, WorldHex>();
  for (let q = 0; q < size; q++) {
    for (let r = 0; r < size; r++) {
      const coord = { q, r };
      hexes.set(key(coord), {
        coord,
        elevation: 100,
        slope: 0,
        relief: 0,
        terrainClass: 'land',
        landCover: 'open',
        biome: 'grassland',
        riverFlow: 0,
        catchmentKm2: 0,
        settlementName: null,
        tags: new Set<string>(),
        roadConnections: [],
      });
    }
  }
  return {
    schemaVersion: '1.8',
    seed: 1,
    width: size,
    height: size,
    layout: 'axial',
    hexes,
    rivers: [],
    settlements: [],
    roadEdges: new Map(),
    seaEdges: new Map(),
    ferries: [],
    config: {
      navigableMinDischarge: 60000,
      fordMaxCatchmentKm2: 60,
      crossingReliefM: 60,
      meanPrecipMm: 800,
      model: 'organic',
    },
  };
}

const world = flatWorld();

function unit(id: string, faction: string, at: Hex, kind: UnitKind = 'infantry'): Unit {
  return {
    id,
    name: id,
    faction,
    kind,
    paperStrength: 4000,
    fatigue: 0,
    experience: 0,
    morale: 30,
    provisions: 40,
    maxProvisions: 40,
    equipment: 30,
    maxEquipment: 30,
    guns: 0,
    marchSpeedKmh: 3,
    spacingM: 0.5,
    spacingMultiplier: 1,
    traits: [],
    formation: 'march',
    column: [at],
    hoursMarchedToday: 0,
    corps: null,
    parentUnitId: null,
  };
}

const commander = (id: string, faction: string, unitId: string): Commander => ({
  id,
  name: id,
  faction,
  unitId,
  superiorId: null,
  autoCascade: true,
});

const stateWith = (units: Unit[], commanders: Commander[], clockHours = 6): CampaignState => ({
  ...EMPTY_STATE,
  clockHours,
  nextSeq: 10,
  units: new Map(units.map((u) => [u.id, u])),
  commanders: new Map(commanders.map((c) => [c.id, c])),
});

function fold(state: CampaignState, payloads: readonly EventPayload[]): CampaignState {
  let s = state;
  for (const payload of payloads) {
    s = reduce(s, {
      seq: s.nextSeq,
      clockHours: s.clockHours,
      actor: { kind: 'referee' },
      payload,
      forced: false,
      strictness: 'strict',
      bypassed: [],
    });
  }
  return s;
}

const sightingOf = (u: Unit, atHours: number): Sighting =>
  contactFrom(u, SIGHTING_INTEL, atHours);

const contactsOf = (s: CampaignState, id: string): ReturnType<typeof Array.from> =>
  [...(s.knowledge.get(id)?.contacts.values() ?? [])];

describe('labelling a sighting', () => {
  const enemy = unit('blue-1', 'blue', { q: 9, r: 5 });
  const base = stateWith([unit('red-1', 'red', { q: 5, r: 5 }), enemy], [
    commander('ney', 'red', 'red-1'),
  ]);

  it('gives it a label of the commander’s own, not the enemy’s id', () => {
    const after = fold(base, fileSightings(base, cfg, 'ney', [sightingOf(enemy, 6)], 6));
    const [contact] = contactsOf(after, 'ney') as { id: string; unitId: string }[];

    expect(contact!.id).toBe('c1');
    // The engine keeps the observed formation — it needs it to decide continuity — and
    // `publicContact` is what stops it reaching a client.
    expect(contact!.unitId).toBe('blue-1');
  });

  it('numbers each new contact in turn', () => {
    const second = unit('blue-2', 'blue', { q: 10, r: 5 });
    const after = fold(
      base,
      fileSightings(base, cfg, 'ney', [sightingOf(enemy, 6), sightingOf(second, 6)], 6),
    );
    expect((contactsOf(after, 'ney') as { id: string }[]).map((c) => c.id)).toEqual([
      'c1',
      'c2',
    ]);
  });

  it('does not reuse a number when two are filed in the same pass', () => {
    // The counter lives in state, which is not folded between the payloads of one pass.
    // Reading it fresh for each would mint `c1` twice and the second would overwrite the
    // first — a column silently disappearing off the map.
    const second = unit('blue-2', 'blue', { q: 10, r: 5 });
    const payloads = fileSightings(
      base,
      cfg,
      'ney',
      [sightingOf(enemy, 6), sightingOf(second, 6)],
      6,
    );
    const ids = payloads.map((p) => (p.kind === 'contact_filed' ? p.contact.id : ''));
    expect(new Set(ids).size).toBe(ids.length);
  });
});

describe('continuity is eyes on, not elapsed time', () => {
  const enemyAt = (coord: Hex): Unit => unit('blue-1', 'blue', coord);
  const ney = commander('ney', 'red', 'red-1');
  const observer = unit('red-1', 'red', { q: 5, r: 5 });

  /** A pass of the real thing: what their own formation can see, right now. */
  const look = (s: CampaignState): CampaignState =>
    fold(s, sightingEvents(s, world, cfg));

  const withEnemyAt = (s: CampaignState, coord: Hex, clockHours: number): CampaignState => ({
    ...s,
    clockHours,
    units: new Map(s.units).set('blue-1', enemyAt(coord)),
  });

  it('keeps one label while the picket never looks away', () => {
    let s = look(stateWith([observer, enemyAt({ q: 6, r: 5 })], [ney], 0));
    // Twelve hours of watching, in whatever steps the referee happened to advance in.
    for (const hour of [1, 6, 12]) s = look(withEnemyAt(s, { q: 6, r: 5 }, hour));

    const held = contactsOf(s, 'ney') as { id: string; seenAtHours: number }[];
    // One contact. An earlier version measured the gap between *filings* against a window
    // and minted a new number every advance — for an enemy standing still in front of a
    // division that had never once stopped looking at it.
    expect(held).toHaveLength(1);
    expect(held[0]!.id).toBe('c1');
    expect(held[0]!.seenAtHours).toBe(12);
  });

  it('follows a column across the ground as one contact', () => {
    let s = look(stateWith([observer, enemyAt({ q: 6, r: 5 })], [ney], 0));
    // (5,6) is a neighbour of the observer at (5,5); (6,6) is two hexes off and would be
    // out of sight, which is a different test.
    s = look(withEnemyAt(s, { q: 5, r: 6 }, 2));

    const held = contactsOf(s, 'ney') as { id: string; coord: Hex }[];
    expect(held).toHaveLength(1);
    expect(held[0]!.coord).toEqual({ q: 5, r: 6 });
  });

  it('marks a contact lost when it goes out of view, and keeps it on the map', () => {
    let s = look(stateWith([observer, enemyAt({ q: 6, r: 5 })], [ney], 0));
    s = look(withEnemyAt(s, { q: 25, r: 25 }, 3));

    const held = contactsOf(s, 'ney') as {
      id: string;
      coord: Hex;
      inSight: boolean;
      seenAtHours: number;
    }[];
    expect(held).toHaveLength(1);
    expect(held[0]!.inSight).toBe(false);
    // Still at the hex they last saw it, and still dated then. Losing sight of something
    // does not unsee it.
    expect(held[0]!.coord).toEqual({ q: 6, r: 5 });
    expect(held[0]!.seenAtHours).toBe(0);
  });

  it('gives a reappearing column a new number, and keeps the old sighting', () => {
    let s = look(stateWith([observer, enemyAt({ q: 6, r: 5 })], [ney], 0));
    s = look(withEnemyAt(s, { q: 25, r: 25 }, 3));
    s = look(withEnemyAt(s, { q: 5, r: 4 }, 9));

    const held = contactsOf(s, 'ney') as { id: string; coord: Hex }[];
    // Two marks. Whether they are the same corps is a judgement they have to make, and the
    // numbering deliberately does not make it for them.
    expect(held).toHaveLength(2);
    expect(new Set(held.map((c) => c.id))).toEqual(new Set(['c1', 'c2']));
  });

  it('writes nothing down when nothing has changed', () => {
    const s = look(stateWith([observer, enemyAt({ q: 6, r: 5 })], [ney], 0));
    // Same hour, same hex, same grade: a second pass has nothing to record. Without this
    // every command in the campaign would append an identical event.
    expect(sightingEvents(s, world, cfg)).toEqual([]);
  });

  it('refreshes a watched contact’s hour at the logging cadence', () => {
    const s = look(stateWith([observer, enemyAt({ q: 6, r: 5 })], [ney], 0));
    const later = { ...s, clockHours: cfg.contactRefreshHours };
    expect(sightingEvents(later, world, cfg)).toHaveLength(1);
  });

  it('never lets an older sighting overwrite a newer one', () => {
    // Riders overtake each other, so a report can arrive carrying word older than the map
    // already holds. The later hour is the better fact whichever landed first.
    let s = look(stateWith([observer, enemyAt({ q: 6, r: 5 })], [ney], 8));
    const stale: Sighting = { ...sightingOf(enemyAt({ q: 9, r: 9 }), 2), coord: { q: 9, r: 9 } };
    s = fold(s, fileSightings(s, cfg, 'ney', [stale], 2));

    const held = contactsOf(s, 'ney') as { coord: Hex; seenAtHours: number }[];
    const watched = held.find((c) => c.seenAtHours === 8);
    expect(watched!.coord).toEqual({ q: 6, r: 5 });
  });
});

describe('word from somebody else', () => {
  const ney = commander('ney', 'red', 'red-1');
  const observer = unit('red-1', 'red', { q: 5, r: 5 });
  const enemy = unit('blue-1', 'blue', { q: 6, r: 5 });

  it('starts its own contact even when their own pickets are watching the same column', () => {
    let s = fold(
      stateWith([observer, enemy], [ney], 0),
      sightingEvents(stateWith([observer, enemy], [ney], 0), world, cfg),
    );
    s = fold(s, fileSightings(s, cfg, 'ney', [sightingOf(enemy, 0)], 0, 'reported'));

    const held = contactsOf(s, 'ney') as { id: string; inSight: boolean }[];
    // A headquarters told of an enemy on its flank has no way to know it is the same body
    // of troops its own screen can see. Merging them would be the engine deciding
    // something a staff has to decide.
    expect(held).toHaveLength(2);
    expect(held.filter((c) => c.inSight)).toHaveLength(1);
  });

  it('is never marked in sight, however fresh it is', () => {
    const s0 = stateWith([observer, enemy], [ney], 0);
    const s = fold(s0, fileSightings(s0, cfg, 'ney', [sightingOf(enemy, 0)], 0, 'reported'));
    expect((contactsOf(s, 'ney') as { inSight: boolean }[])[0]!.inSight).toBe(false);
  });
});

describe('what a commander’s own eyes file', () => {
  it('records only what the formation they ride with can see', () => {
    const near = unit('red-1', 'red', { q: 5, r: 5 });
    const detached = unit('red-2', 'red', { q: 25, r: 25 });
    const enemyByHim = unit('blue-1', 'blue', { q: 6, r: 5 });
    const enemyByThem = unit('blue-2', 'blue', { q: 26, r: 25 });

    const s = stateWith(
      [near, detached, enemyByHim, enemyByThem],
      [commander('ney', 'red', 'red-1')],
    );
    const after = fold(s, sightingEvents(s, world, cfg));

    const held = contactsOf(after, 'ney') as { unitId: string }[];
    // Their own column sees one of them. What their detached division is looking at reaches
    // them by rider or not at all — merging the two is the telepathy this design denies.
    expect(held.map((c) => c.unitId)).toEqual(['blue-1']);
  });

  it('files nothing for a commander with no formation', () => {
    const s = stateWith([unit('blue-1', 'blue', { q: 6, r: 5 })], [
      commander('ghost', 'red', 'red-gone'),
    ]);
    expect(sightingEvents(s, world, cfg)).toEqual([]);
  });

  it('is stable across runs, so a log replays byte for byte', () => {
    const s = stateWith(
      [
        unit('red-1', 'red', { q: 5, r: 5 }),
        unit('blue-1', 'blue', { q: 6, r: 5 }),
        unit('blue-2', 'blue', { q: 4, r: 5 }),
      ],
      [commander('ney', 'red', 'red-1')],
    );
    expect(JSON.stringify(sightingEvents(s, world, cfg))).toBe(
      JSON.stringify(sightingEvents(s, world, cfg)),
    );
  });
});
