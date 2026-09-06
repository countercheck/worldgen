/**
 * Reconnaissance: what a faction can see, and how well.
 */

import { describe, expect, it } from 'vitest';

import { DEFAULT_CONFIG } from '../src/config.js';
import { key, type Hex, type HexKey } from '../src/hex.js';
import {
  contactFrom,
  detectionDice,
  factionVisible,
  hearsGunfire,
  reconRadius,
  reconZone,
  spotted,
  SIGHTING_INTEL,
} from '../src/recon.js';
import { EMPTY_STATE, type CampaignState } from '../src/state.js';
import type { Trait, Unit, UnitKind } from '../src/unit.js';
import type { World, WorldHex } from '../src/world.js';

const cfg = DEFAULT_CONFIG;

function flatWorld(size = 40): World {
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

function unit(
  id: string,
  faction: string,
  column: Hex[],
  traits: Trait[] = [],
  kind: UnitKind = 'infantry',
  opts: Partial<Unit> = {},
): Unit {
  return {
    id,
    faction,
    kind,
    effectives: 4000,
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
    traits,
    formation: 'march',
    column,
    hoursMarchedToday: 0,
    corps: null,
    ...opts,
  };
}

const stateWith = (...units: Unit[]): CampaignState => ({
  ...EMPTY_STATE,
  clockHours: 12,
  units: new Map(units.map((u) => [u.id, u])),
});

describe('reconRadius', () => {
  it('is one hex, or two with scouts', () => {
    expect(reconRadius(cfg, unit('a', 'red', [{ q: 5, r: 5 }]))).toBe(1);
    expect(reconRadius(cfg, unit('a', 'red', [{ q: 5, r: 5 }], ['scout']))).toBe(2);
  });
});

describe('reconZone', () => {
  it('covers the unit and its immediate ring', () => {
    const u = unit('a', 'red', [{ q: 10, r: 10 }]);
    const zone = reconZone(world, cfg, u);
    expect(zone.size).toBe(7); // the hex itself plus six neighbours
    expect(zone.has(key({ q: 10, r: 10 }))).toBe(true);
    expect(zone.has(key({ q: 11, r: 10 }))).toBe(true);
    expect(zone.has(key({ q: 12, r: 10 }))).toBe(false);
  });

  it('widens to two hexes with scouts', () => {
    const u = unit('a', 'red', [{ q: 10, r: 10 }], ['scout']);
    const zone = reconZone(world, cfg, u);
    expect(zone.size).toBe(19); // 1 + 6 + 12
    expect(zone.has(key({ q: 12, r: 10 }))).toBe(true);
    expect(zone.has(key({ q: 13, r: 10 }))).toBe(false);
  });

  it('sees along the whole column, not just from the head', () => {
    // The point of measuring recon from the column: a division strung across twenty
    // kilometres of road observes the country beside all of it.
    const long = unit(
      'a',
      'red',
      Array.from({ length: 20 }, (_, i) => ({ q: 10 + i, r: 10 })),
      [],
      'cavalry',
      { effectives: 4000, spacingM: 3, spacingMultiplier: 1.5 },
    );
    const zone = reconZone(world, cfg, long);

    // The tail hex, eighteen km back, is still observing.
    expect(zone.has(key({ q: 28, r: 10 }))).toBe(true);
    expect(zone.has(key({ q: 28, r: 9 }))).toBe(true);
    // A point unit at the head would see none of that.
    const point = unit('b', 'red', [{ q: 10, r: 10 }]);
    expect(reconZone(world, cfg, point).has(key({ q: 28, r: 10 }))).toBe(false);
  });

  it('sweeps a corridor, so a marching column sees far more than a halted one', () => {
    const halted = unit('a', 'red', [{ q: 10, r: 10 }], ['scout']);
    const marching = unit(
      'b',
      'red',
      Array.from({ length: 20 }, (_, i) => ({ q: 10 + i, r: 10 })),
      ['scout'],
      'cavalry',
      { spacingM: 3, spacingMultiplier: 1.5 },
    );
    expect(reconZone(world, cfg, marching).size).toBeGreaterThan(
      reconZone(world, cfg, halted).size * 5,
    );
  });

  it('does not see off the map', () => {
    const corner = unit('a', 'red', [{ q: 0, r: 0 }], ['scout']);
    for (const k of reconZone(world, cfg, corner)) {
      expect(world.hexes.has(k), `${k} is not on the map`).toBe(true);
    }
  });

  it('shrinks on a highway, where the column closes up', () => {
    const u = unit(
      'a',
      'red',
      Array.from({ length: 20 }, (_, i) => ({ q: 10 + i, r: 10 })),
      [],
      'cavalry',
      { spacingM: 3, spacingMultiplier: 1.5 },
    );
    expect(reconZone(world, cfg, u, 'highway').size).toBeLessThan(
      reconZone(world, cfg, u, 'road').size,
    );
  });
});

describe('factionVisible', () => {
  it('unions every unit of the faction and nobody else', () => {
    const state = stateWith(
      unit('r1', 'red', [{ q: 5, r: 5 }]),
      unit('r2', 'red', [{ q: 20, r: 20 }]),
      unit('b1', 'blue', [{ q: 30, r: 30 }]),
    );
    const red = factionVisible(state, world, cfg, 'red');
    expect(red.has(key({ q: 5, r: 5 }))).toBe(true);
    expect(red.has(key({ q: 20, r: 20 }))).toBe(true);
    expect(red.has(key({ q: 30, r: 30 }))).toBe(false);
  });

  it('is empty for a faction with nothing in the field', () => {
    expect(factionVisible(stateWith(), world, cfg, 'red').size).toBe(0);
  });
});

describe('spotted', () => {
  it('finds an enemy standing in the recon zone', () => {
    const state = stateWith(
      unit('r1', 'red', [{ q: 10, r: 10 }]),
      unit('b1', 'blue', [{ q: 11, r: 10 }]),
    );
    const contacts = spotted(state, world, cfg, 'red');
    expect(contacts.has('b1')).toBe(true);
    expect(contacts.get('b1')!.coord).toEqual({ q: 11, r: 10 });
    expect(contacts.get('b1')!.seenAtHours).toBe(12);
  });

  it('does not find an enemy beyond it', () => {
    const state = stateWith(
      unit('r1', 'red', [{ q: 10, r: 10 }]),
      unit('b1', 'blue', [{ q: 20, r: 20 }]),
    );
    expect(spotted(state, world, cfg, 'red').size).toBe(0);
  });

  it("spots a corps by its baggage when its head is clear", () => {
    // The whole column counts. A formation whose head is away but whose tail is strung
    // across open country has been seen.
    const enemy = unit(
      'b1',
      'blue',
      Array.from({ length: 20 }, (_, i) => ({ q: 30 - i, r: 10 })),
      [],
      'cavalry',
      { spacingM: 3, spacingMultiplier: 1.5 },
    );
    const state = stateWith(unit('r1', 'red', [{ q: 15, r: 10 }]), enemy);

    const contacts = spotted(state, world, cfg, 'red');
    expect(contacts.has('b1')).toBe(true);
    // Reported where it was actually seen — its tail, not its head.
    expect(contacts.get('b1')!.coord).not.toEqual({ q: 30, r: 10 });
  });

  it('never reports a faction its own units', () => {
    const state = stateWith(
      unit('r1', 'red', [{ q: 10, r: 10 }]),
      unit('r2', 'red', [{ q: 11, r: 10 }]),
    );
    expect(spotted(state, world, cfg, 'red').size).toBe(0);
  });

  it('gives a plain sighting presence and location only', () => {
    const state = stateWith(
      unit('r1', 'red', [{ q: 10, r: 10 }]),
      unit('b1', 'blue', [{ q: 11, r: 10 }], [], 'cavalry', { corps: 'I Corps' }),
    );
    const contact = spotted(state, world, cfg, 'red').get('b1')!;
    expect(contact.intelLevel).toBe(SIGHTING_INTEL);
    expect(contact.kind).toBeNull();
    expect(contact.corps).toBeNull();
  });
});

describe('contactFrom', () => {
  it('redacts what the intel level does not earn', () => {
    // The redaction is here rather than at display time on purpose. Anything a commander
    // is not entitled to know should never reach their client — filtering on the way to
    // the screen is exactly how it leaks.
    const enemy = unit('b1', 'blue', [{ q: 4, r: 4 }], [], 'cavalry', { corps: 'II Corps' });

    expect(contactFrom(enemy, 2, 0).kind).toBeNull();
    expect(contactFrom(enemy, 4, 0).kind).toBeNull();
    expect(contactFrom(enemy, 5, 0).kind).toBe('cavalry');
    expect(contactFrom(enemy, 5, 0).corps).toBeNull();
    expect(contactFrom(enemy, 6, 0).corps).toBe('II Corps');
  });

  it('always carries presence and location', () => {
    const enemy = unit('b1', 'blue', [{ q: 4, r: 4 }]);
    for (const level of [1, 2, 3, 4, 5, 6] as const) {
      const c = contactFrom(enemy, level, 7);
      expect(c.coord).toEqual({ q: 4, r: 4 });
      expect(c.seenAtHours).toBe(7);
      expect(c.faction).toBe('blue');
    }
  });
});

describe('detectionDice', () => {
  it('follows the rules modifiers', () => {
    const infantry = unit('a', 'red', [{ q: 0, r: 0 }]);
    const cavalry = unit('b', 'red', [{ q: 0, r: 0 }], [], 'cavalry');
    const scoutCav = unit('c', 'red', [{ q: 0, r: 0 }], ['scout'], 'cavalry');
    const convoy = unit('d', 'red', [{ q: 0, r: 0 }], [], 'convoy');

    expect(detectionDice(cfg, infantry)).toBe(1); // a division
    expect(detectionDice(cfg, cavalry)).toBe(2); // division + cavalry
    expect(detectionDice(cfg, scoutCav)).toBe(3); // division + cavalry + scout
    expect(detectionDice(cfg, convoy)).toBe(0); // neither a division nor cavalry
  });
});

describe('hearsGunfire', () => {
  it('carries thirty kilometres', () => {
    expect(hearsGunfire(cfg, { q: 0, r: 0 }, { q: 30, r: 0 })).toBe(true);
    expect(hearsGunfire(cfg, { q: 0, r: 0 }, { q: 31, r: 0 })).toBe(false);
  });
});
