/**
 * Reconnaissance: what a faction can see, and how well.
 */

import { describe, expect, it } from 'vitest';

import { DEFAULT_CONFIG } from '../src/config.js';
import { key, type Hex, type HexKey } from '../src/hex.js';
import {
  contactFrom,
  detectionDice,
  commanderVisible,
  commandVisible,
  hearsGunfire,
  reconRadius,
  reconZone,
  spottedBy,
  spottedUnder,
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
    traits,
    formation: 'march',
    column,
    hoursMarchedToday: 0,
    corps: null,
    parentUnitId: null,
    ...opts,
  };
}

const stateWith = (...units: Unit[]): CampaignState => ({
  ...EMPTY_STATE,
  clockHours: 12,
  units: new Map(units.map((u) => [u.id, u])),
});

/** A commander for each formation, the first of them commanding the rest of their side. */
function commanded(...units: Unit[]): CampaignState {
  const base = stateWith(...units);
  const chiefs = new Map<string, string>();
  const commanders = new Map<string, Commander>();

  for (const u of units) {
    const chief = chiefs.get(u.faction);
    commanders.set(`c-${u.id}`, {
      id: `c-${u.id}`,
      name: `Commander of ${u.id}`,
      faction: u.faction,
      unitId: u.id,
      superiorId: chief ?? null,
    });
    if (chief === undefined) chiefs.set(u.faction, `c-${u.id}`);
  }
  return { ...base, commanders };
}

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
      { paperStrength: 4000, spacingM: 3, spacingMultiplier: 1.5 },
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

describe('commanderVisible', () => {
  it('is what they can see from where they stand, and no further', () => {
    // The whole point of the model. A commander riding with r1 at 5,5 does not see the country
    // around r2 at 20,20 merely because r2 is theirs; that arrives by despatch or not at all.
    const state = commanded(
      unit('r1', 'red', [{ q: 5, r: 5 }]),
      unit('r2', 'red', [{ q: 20, r: 20 }]),
      unit('b1', 'blue', [{ q: 30, r: 30 }]),
    );
    const seen = commanderVisible(state, world, cfg, 'c-r1');
    expect(seen.has(key({ q: 5, r: 5 }))).toBe(true);
    expect(seen.has(key({ q: 20, r: 20 })), 'they saw their own subordinate forty km away').toBe(
      false,
    );
    expect(seen.has(key({ q: 30, r: 30 }))).toBe(false);
  });

  it('is exactly their formation\'s recon zone', () => {
    const u = unit('r1', 'red', [{ q: 5, r: 5 }]);
    const state = commanded(u);
    expect([...commanderVisible(state, world, cfg, 'c-r1')].sort()).toEqual(
      [...reconZone(world, cfg, u)].sort(),
    );
  });

  it('is empty for a commander who is nobody, or who rides with nothing', () => {
    expect(commanderVisible(stateWith(), world, cfg, 'nobody').size).toBe(0);

    const orphaned: CampaignState = {
      ...EMPTY_STATE,
      commanders: new Map([
        [
          'ghost',
          {
            id: 'ghost',
            name: 'A commander with no army',
            faction: 'red',
            unitId: 'gone',
            superiorId: null,
          },
        ],
      ]),
    };
    expect(commanderVisible(orphaned, world, cfg, 'ghost').size).toBe(0);
  });
});

describe('commandVisible', () => {
  it('unions everything their formations can see, which is not what they know', () => {
    // The observing side of the transaction: the ground their divisions are looking at, from
    // which reports are made. Distinct from `commanderVisible` on purpose.
    const state = commanded(
      unit('r1', 'red', [{ q: 5, r: 5 }]),
      unit('r2', 'red', [{ q: 20, r: 20 }]),
      unit('b1', 'blue', [{ q: 30, r: 30 }]),
    );
    const all = commandVisible(state, world, cfg, 'c-r1');
    expect(all.has(key({ q: 5, r: 5 }))).toBe(true);
    expect(all.has(key({ q: 20, r: 20 }))).toBe(true);
    expect(all.has(key({ q: 30, r: 30 }))).toBe(false);
  });

  it('is only their own formation for a commander with no subordinates', () => {
    const state = commanded(
      unit('r1', 'red', [{ q: 5, r: 5 }]),
      unit('r2', 'red', [{ q: 20, r: 20 }]),
    );
    const junior = commandVisible(state, world, cfg, 'c-r2');
    expect(junior.has(key({ q: 20, r: 20 }))).toBe(true);
    expect(junior.has(key({ q: 5, r: 5 }))).toBe(false);
  });

  it('is empty for nobody', () => {
    expect(commandVisible(stateWith(), world, cfg, 'nobody').size).toBe(0);
  });
});

describe('spottedBy', () => {
  it('finds an enemy standing in the recon zone', () => {
    const state = stateWith(
      unit('r1', 'red', [{ q: 10, r: 10 }]),
      unit('b1', 'blue', [{ q: 11, r: 10 }]),
    );
    const contacts = spottedBy(state, world, cfg, state.units.get('r1')!);
    expect(contacts.has('b1')).toBe(true);
    expect(contacts.get('b1')!.coord).toEqual({ q: 11, r: 10 });
    expect(contacts.get('b1')!.seenAtHours).toBe(12);
  });

  it('does not find an enemy beyond it', () => {
    const state = stateWith(
      unit('r1', 'red', [{ q: 10, r: 10 }]),
      unit('b1', 'blue', [{ q: 20, r: 20 }]),
    );
    expect(spottedBy(state, world, cfg, state.units.get('r1')!).size).toBe(0);
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

    const contacts = spottedBy(state, world, cfg, state.units.get('r1')!);
    expect(contacts.has('b1')).toBe(true);
    // Reported where it was actually seen — its tail, not its head.
    expect(contacts.get('b1')!.coord).not.toEqual({ q: 30, r: 10 });
  });

  it('never reports a faction its own units', () => {
    const state = stateWith(
      unit('r1', 'red', [{ q: 10, r: 10 }]),
      unit('r2', 'red', [{ q: 11, r: 10 }]),
    );
    expect(spottedBy(state, world, cfg, state.units.get('r1')!).size).toBe(0);
  });

  it('gives a plain sighting presence and location only', () => {
    const state = stateWith(
      unit('r1', 'red', [{ q: 10, r: 10 }]),
      unit('b1', 'blue', [{ q: 11, r: 10 }], [], 'cavalry', { corps: 'I Corps' }),
    );
    const contact = spottedBy(state, world, cfg, state.units.get('r1')!).get('b1')!;
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

describe('spottedUnder', () => {
  it('merges what every formation under a commander can see', () => {
    // The interim rule: until riders exist, subordinates report the instant they see
    // anything. A chief learns of an enemy their cavalry found forty kilometres away.
    const state = commanded(
      unit('r1', 'red', [{ q: 5, r: 5 }]),
      unit('r2', 'red', [{ q: 20, r: 20 }]),
      unit('b1', 'blue', [{ q: 21, r: 20 }]),
    );

    const chief = spottedUnder(state, world, cfg, 'c-r1');
    expect(chief.has('b1'), 'the chief never heard from their cavalry').toBe(true);
    expect(chief.get('b1')!.coord).toEqual({ q: 21, r: 20 });
  });

  it('tells a subordinate nothing about what their chief can see', () => {
    // Reports travel up the tree and never down it.
    const state = commanded(
      unit('r1', 'red', [{ q: 5, r: 5 }]),
      unit('r2', 'red', [{ q: 20, r: 20 }]),
      unit('b1', 'blue', [{ q: 6, r: 5 }]),
    );

    expect(spottedUnder(state, world, cfg, 'c-r1').has('b1')).toBe(true);
    expect(spottedUnder(state, world, cfg, 'c-r2').has('b1')).toBe(false);
  });

  it('keeps the better of two sightings of the same enemy', () => {
    // Two despatches about one column; a headquarters believes the more informative.
    const state = commanded(
      unit('r1', 'red', [{ q: 5, r: 5 }]),
      unit('r2', 'red', [{ q: 20, r: 20 }]),
      unit('b1', 'blue', [{ q: 6, r: 5 }]),
    );
    const merged = spottedUnder(state, world, cfg, 'c-r1');
    const direct = spottedBy(state, world, cfg, state.units.get('r1')!);

    expect(merged.get('b1')!.intelLevel).toBeGreaterThanOrEqual(
      direct.get('b1')!.intelLevel,
    );
  });

  it('is empty for a commander whose formations see nothing', () => {
    const state = commanded(
      unit('r1', 'red', [{ q: 5, r: 5 }]),
      unit('b1', 'blue', [{ q: 25, r: 25 }]),
    );
    expect(spottedUnder(state, world, cfg, 'c-r1').size).toBe(0);
  });
});
