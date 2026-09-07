/**
 * Despatches: what a sender is shown, and what he must never be.
 *
 * The assertions here are about *shape* rather than about riding — the scheduler's tests
 * cover the ride. What matters in this file is that the copy a commander receives is
 * built rather than filtered, so that a field added to `Despatch` tomorrow does not
 * silently appear in an outbox.
 */

import { describe, expect, it } from 'vitest';

import {
  addresseeCopy,
  captorCopy,
  deliveredAt,
  formationsTouch,
  isSuperseded,
  planRide,
  ridePath,
  senderCopy,
  type Despatch,
} from '../src/despatch.js';
import { DEFAULT_CONFIG } from '../src/config.js';
import { key, type Hex, type HexKey } from '../src/hex.js';
import type { Unit } from '../src/unit.js';
import type { World, WorldHex } from '../src/world.js';

const cfg = DEFAULT_CONFIG;

function flatWorld(size = 20): World {
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

const despatch = (over: Partial<Despatch> = {}): Despatch => ({
  id: 'd1',
  kind: 'order',
  from: 'ney',
  to: 'kellermann',
  faction: 'red',
  sentAtHours: 4,
  body: { text: 'Move on Quatre Bras with all speed.' },
  via: [],
  forwardedFrom: null,
  inReplyTo: null,
  route: [
    { q: 1, r: 1 },
    { q: 2, r: 1 },
    { q: 3, r: 1 },
  ],
  progress: 1,
  fate: { kind: 'in_transit' },
  handed: false,
  ...over,
});

function unit(id: string, faction: string, column: Hex[], effectives = 4000): Unit {
  return {
    id,
    name: id,
    faction,
    kind: 'infantry',
    effectives,
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
    column,
    hoursMarchedToday: 0,
    corps: null,
  };
}

describe('what a sender is shown', () => {
  it('carries no route, no fate, and no progress', () => {
    const copy = senderCopy(despatch(), false);
    expect(Object.keys(copy).sort()).toEqual([
      'acknowledged',
      'body',
      'handed',
      'id',
      'inReplyTo',
      'kind',
      'sentAtHours',
      'to',
      'via',
    ]);
  });

  it('shows his own waypoints, which he chose himself', () => {
    const via = [{ q: 9, r: 9 }];
    expect(senderCopy(despatch({ via }), false).via).toEqual(via);
  });

  it('says nothing different when the rider was captured', () => {
    const taken = despatch({
      fate: { kind: 'captured', by: 'blue', atHours: 6, dice: [1, 1] },
    });
    // The whole mechanic in one assertion: a commander whose order was read by the enemy
    // sees precisely what he saw the moment he sent it.
    expect(senderCopy(taken, false)).toEqual(senderCopy(despatch(), false));
  });

  it('learns of an arrival only when an acknowledgement comes back', () => {
    expect(senderCopy(despatch(), false).acknowledged).toBe(false);
    expect(senderCopy(despatch(), true).acknowledged).toBe(true);
  });
});

describe('what an addressee is shown', () => {
  it('leads with the hour it describes, not the hour it arrived', () => {
    const arrived = despatch({ fate: { kind: 'delivered', atHours: 11 } });
    const copy = addresseeCopy(arrived, false);
    expect(copy.sentAtHours).toBe(4);
    expect(copy.receivedAtHours).toBe(11);
  });

  it('carries no route either: the ride is the referee’s business', () => {
    const arrived = despatch({ fate: { kind: 'delivered', atHours: 11 } });
    expect('route' in addresseeCopy(arrived, false)).toBe(false);
  });
});

describe('a captor', () => {
  it('reads the body and knows when he took it', () => {
    const taken = despatch({
      fate: { kind: 'captured', by: 'blue', atHours: 6, dice: [1, 1] },
    });
    const copy = captorCopy(taken);
    expect(copy.body.text).toContain('Quatre Bras');
    expect(copy.capturedAtHours).toBe(6);
    // Not the route: knowing where the rider was going is knowing where the addressee
    // stands, which is the one thing capture must not hand over for free.
    expect('route' in copy).toBe(false);
  });
});

describe('dated orders', () => {
  const at = (id: string, sent: number, delivered: number): Despatch =>
    despatch({ id, sentAtHours: sent, fate: { kind: 'delivered', atHours: delivered } });

  it('disregards an order overtaken by a later one already in hand', () => {
    const early = at('d1', 4, 12);
    const late = at('d2', 6, 9);
    expect(isSuperseded(early, [early, late])).toBe(true);
    expect(isSuperseded(late, [early, late])).toBe(false);
  });

  it('stands when the later order has not arrived yet', () => {
    const early = at('d1', 4, 9);
    const late = despatch({ id: 'd2', sentAtHours: 6 });
    expect(isSuperseded(early, [early, late])).toBe(false);
  });

  it('never supersedes a report — an old fact is still a fact', () => {
    const old = at('d1', 4, 12);
    const report = despatch({ id: 'd2', kind: 'report', sentAtHours: 6 });
    expect(isSuperseded({ ...old, kind: 'report' }, [old, report])).toBe(false);
  });

  it('reads a delivery hour off a fate, and null off anything else', () => {
    expect(deliveredAt(at('d1', 4, 9))).toBe(9);
    expect(deliveredAt(despatch())).toBeNull();
  });
});

describe('riding', () => {
  it('finds a path between two hexes on open ground', () => {
    const path = ridePath(world, cfg, { q: 2, r: 2 }, { q: 8, r: 2 });
    expect(path).not.toBeNull();
    expect(path![0]).toEqual({ q: 2, r: 2 });
    expect(path!.at(-1)).toEqual({ q: 8, r: 2 });
  });

  it('goes the long way when the sender insists on a waypoint', () => {
    const direct = ridePath(world, cfg, { q: 2, r: 2 }, { q: 8, r: 2 })!;
    const around = planRide(world, cfg, { q: 2, r: 2 }, { q: 8, r: 2 }, [{ q: 5, r: 10 }])!;
    // The whole point of `via`: a rider ordered around a wood he thinks holds pickets
    // takes longer, and takes it willingly.
    expect(around.length).toBeGreaterThan(direct.length);
    expect(around.some((h) => h.q === 5 && h.r === 10)).toBe(true);
  });

  it('returns null when there is nowhere to ride to', () => {
    expect(ridePath(world, cfg, { q: 2, r: 2 }, { q: 99, r: 99 })).toBeNull();
  });
});

describe('formations touching', () => {
  it('counts a column, not a marker', () => {
    // A big division is several kilometres of road. Its tail touches what its head
    // cannot reach, which is exactly why this is measured against `occupied`.
    const long = unit('a', 'red', [
      { q: 5, r: 5 },
      { q: 6, r: 5 },
      { q: 7, r: 5 },
      { q: 8, r: 5 },
    ], 8000);
    const near = unit('b', 'red', [{ q: 9, r: 5 }]);
    const far = unit('c', 'red', [{ q: 15, r: 5 }]);

    expect(formationsTouch(long, near)).toBe(true);
    expect(formationsTouch(long, far)).toBe(false);
  });

  it('is symmetric', () => {
    const a = unit('a', 'red', [{ q: 5, r: 5 }]);
    const b = unit('b', 'red', [{ q: 6, r: 5 }]);
    expect(formationsTouch(a, b)).toBe(formationsTouch(b, a));
  });
});
