/**
 * The hours of the day: the referee's sunrise and sunset, and a commander's standing orders.
 *
 * Structural, like the scheduler's own tests: the head is never on the road outside the
 * hours its commander allowed, a column halted by them camps and is formed up again by the
 * hour they named, and the referee's sunset is the one night fatigue is charged by.
 */

import { describe, expect, it } from 'vitest';

import { DEFAULT_CONFIG } from '../src/config.js';
import { apply } from '../src/engine.js';
import type { EventPayload } from '../src/events.js';
import { isDark } from '../src/fatigue.js';
import { key, type Hex, type HexKey } from '../src/hex.js';
import type { Rng } from '../src/rng.js';
import { CODES } from '../src/ruling.js';
import { advance } from '../src/scheduler.js';
import { configAt, EMPTY_STATE, reduce, type CampaignState } from '../src/state.js';
import { roadHoursWithin } from '../src/movement.js';
import {
  hourOfDay,
  standingHoursLeft,
  standingOrdersProblems,
  type StandingOrders,
} from '../src/standing.js';
import type { Task } from '../src/task.js';
import type { Unit } from '../src/unit.js';
import { REFEREE_ROLE, commanderRole, viewFor, assertMasked } from '../src/view.js';
import type { World, WorldHex } from '../src/world.js';
import { marched } from './fixtures/road.js';

const cfg = DEFAULT_CONFIG;

function flatWorld(size = 60): World {
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

const division = (at: Hex, opts: Partial<Unit> = {}): Unit => ({
  id: 'red-1',
  name: '1re Division',
  faction: 'red',
  kind: 'infantry',
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
  formationChange: null,
  column: [at],
  hoursMarchedToday: 0,
  corps: null,
  parentUnitId: null,
  ...opts,
});

/** A long march east across open ground, far enough that it never finishes in a test. */
const eastward = (u: Unit, atHours: number): Task => ({
  unitId: u.id,
  destination: { q: 55, r: 5 },
  via: [],
  setAtHours: atHours,
  fromDespatchId: null,
  nextHex: { q: u.column[0]!.q + 1, r: u.column[0]!.r },
  progressHours: 0,
  complete: false,
  viaIndex: 0,
});

const stateWith = (
  u: Unit,
  clockHours: number,
  orders?: StandingOrders,
  task: Task | null = eastward(u, clockHours),
): CampaignState => ({
  ...EMPTY_STATE,
  clockHours,
  nextSeq: 100,
  factions: new Map([['red', { id: 'red', name: 'Red', color: '#c00' }]]),
  units: new Map([[u.id, u]]),
  commanders: new Map([
    ['ney', { id: 'ney', name: 'Ney', faction: 'red', unitId: u.id, superiorId: null }],
  ]),
  tasks: task === null ? new Map() : new Map([[u.id, task]]),
  standingOrders: orders === undefined ? new Map() : new Map([[u.id, orders]]),
});

const dawn = cfg.sunriseHour;

const clean: Rng = { next: () => 0.9, int: () => 5, d6: () => 6, pool: (n) => Array(n).fill(6) };

const orders = (o: Partial<StandingOrders>): StandingOrders => ({
  startHour: null,
  latestHour: null,
  maxHoursOnRoad: null,
  ...o,
});

type Of<K extends EventPayload['kind']> = Extract<EventPayload, { kind: K }>;
const only = <K extends EventPayload['kind']>(ps: readonly EventPayload[], k: K): Of<K>[] =>
  ps.filter((p): p is Of<K> => p.kind === k);

describe('the hours standing orders leave', () => {
  it('are unlimited when there are no orders', () => {
    expect(standingHoursLeft(undefined, 5, 13, dawn)).toBe(Infinity);
    expect(standingHoursLeft(orders({}), 5, 13, dawn)).toBe(Infinity);
  });

  it('are none before the hour to step off, on any day', () => {
    const o = orders({ startHour: 5 });
    expect(standingHoursLeft(o, 0, 4, dawn)).toBe(0);
    expect(standingHoursLeft(o, 0, 24 * 3 + 4, dawn)).toBe(0);
    expect(standingHoursLeft(o, 0, 24 * 3 + 5, dawn)).toBe(Infinity);
  });

  it('run out at the hour to be off the road', () => {
    const o = orders({ latestHour: 19 });
    expect(standingHoursLeft(o, 0, 17, dawn)).toBe(2);
    expect(standingHoursLeft(o, 0, 19, dawn)).toBe(0);
    expect(standingHoursLeft(o, 0, 23, dawn)).toBe(0);
  });

  it('run out when the head has spent its hours on the road', () => {
    const o = orders({ maxHoursOnRoad: 10 });
    expect(standingHoursLeft(o, 7.5, 12, dawn)).toBe(2.5);
    expect(standingHoursLeft(o, 10, 12, dawn)).toBe(0);
  });

  it('take the tightest of the three', () => {
    const o = orders({ startHour: 5, latestHour: 19, maxHoursOnRoad: 10 });
    expect(standingHoursLeft(o, 9, 17, dawn)).toBe(1);
    expect(standingHoursLeft(o, 2, 17, dawn)).toBe(2);
  });

  it('begin at dawn when the orders name no hour to step off', () => {
    const o = orders({ maxHoursOnRoad: 10 });
    expect(standingHoursLeft(o, 0, 24 * 2 + dawn - 1, dawn)).toBe(0);
    expect(standingHoursLeft(o, 0, 24 * 2 + dawn, dawn)).toBe(10);
    // The referee's dawn, whatever it is.
    expect(standingHoursLeft(o, 0, 4, 4)).toBe(10);
    expect(hourOfDay(24 * 2 + 7)).toBe(7);
  });
});

describe('what a set of standing orders may say', () => {
  it('accepts a march day', () => {
    expect(
      standingOrdersProblems(orders({ startHour: 5, latestHour: 19, maxHoursOnRoad: 10 }), cfg),
    ).toEqual([]);
  });

  it('refuses a window that ends before it starts', () => {
    expect(standingOrdersProblems(orders({ startHour: 19, latestHour: 5 }), cfg)).toHaveLength(1);
    expect(standingOrdersProblems(orders({ startHour: 8, latestHour: 8 }), cfg)).toHaveLength(1);
  });

  it('refuses hours that are not on the clock face, and more road than the rules allow', () => {
    expect(standingOrdersProblems(orders({ startHour: 24 }), cfg)).toHaveLength(1);
    expect(standingOrdersProblems(orders({ startHour: 5.5 }), cfg)).toHaveLength(1);
    expect(standingOrdersProblems(orders({ latestHour: 0 }), cfg)).toHaveLength(1);
    expect(standingOrdersProblems(orders({ maxHoursOnRoad: 0 }), cfg)).toHaveLength(1);
    expect(
      standingOrdersProblems(orders({ maxHoursOnRoad: cfg.maxMarchHoursPerDay + 1 }), cfg),
    ).toHaveLength(1);
  });

  it('refuses an hour to be off the road that comes before a dawn step-off', () => {
    expect(standingOrdersProblems(orders({ latestHour: cfg.sunriseHour }), cfg)).toHaveLength(1);
    expect(standingOrdersProblems(orders({ latestHour: cfg.sunriseHour + 1 }), cfg)).toEqual([]);
  });

  it('refuses anything that is not the three hours, rather than storing it', () => {
    const problems = (raw: unknown): string[] => standingOrdersProblems(raw, cfg);
    expect(problems(undefined)).toHaveLength(1);
    expect(problems('5 to 19')).toHaveLength(1);
    expect(problems([5, 19, 10])).toHaveLength(1);
    expect(problems({ startHour: 5, latestHour: 19 })).toHaveLength(1);
    expect(problems({ ...orders({ startHour: 5 }), note: 'march at dawn' })).toHaveLength(1);
    expect(problems({ ...orders({}), startHour: '5' })).toHaveLength(1);
  });
});

describe('a column under standing orders', () => {
  it('never has its head on the road outside the hours allowed', () => {
    const u = division({ q: 2, r: 5 });
    const o = orders({ startHour: 8, latestHour: 12 });
    const { payloads } = advance(stateWith(u, 6, o), world, cfg, clean, { hours: 48 });

    const marched = [
      ...only(payloads, 'unit_marched').map((p) => p.atHours),
      ...only(payloads, 'march_progressed').map((p) => p.atHours),
    ];
    expect(marched.length).toBeGreaterThan(0);
    for (const at of marched) {
      expect(hourOfDay(at)).toBeGreaterThanOrEqual(8);
      expect(hourOfDay(at)).toBeLessThan(12);
    }
    // Both days, not just the first: the orders stand until they are changed.
    expect(new Set(marched.map((at) => Math.floor(at / 24))).size).toBe(2);
  });

  it('camps when its hours are up, and is formed up again by the hour it may march', () => {
    const u = division({ q: 2, r: 5 });
    const o = orders({ startHour: 8, latestHour: 12 });
    const { payloads } = advance(stateWith(u, 6, o), world, cfg, clean, { hours: 30 });

    const changes = only(payloads, 'formation_change_began');
    const camp = changes.find((c) => c.to === 'rest');
    expect(camp?.reason).toBe('standing_orders');
    expect(camp?.atHours).toBe(12);

    // Striking camp takes the rules' two hours, so it begins at six and the head steps off
    // at eight — not at ten, which is what waiting for eight to start striking would give.
    const strike = changes.find((c) => c.to === 'march' && c.atHours > 12);
    const stepOff = cfg.formationChangeHours.rest.march;
    expect(strike?.atHours).toBe(24 + 8 - stepOff);
    const secondDay = only(payloads, 'unit_marched').filter((p) => p.atHours >= 24);
    expect(secondDay[0]?.atHours).toBe(24 + 8);
  });

  it('stops for the day when the head has had its hours on the road', () => {
    const u = division({ q: 2, r: 5 });
    const o = orders({ maxHoursOnRoad: 3 });
    const state = stateWith(u, 6, o);
    const { payloads } = advance(state, world, cfg, clean, { hours: 10 });

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
    expect(roadHoursWithin(s.units.get(u.id)!, s.clockHours)).toBeCloseTo(3);
    expect(only(payloads, 'formation_change_began').some((c) => c.reason === 'standing_orders')).toBe(
      true,
    );
  });

  describe('stops the head at whichever limit it reaches first', () => {
    /** The hours the head was on the road on the first day, and when it camped. */
    const firstDay = (o: StandingOrders, clockHours = 0) => {
      const u = division({ q: 2, r: 5 });
      const { payloads } = advance(stateWith(u, clockHours, o), world, cfg, clean, {
        hours: 24 - clockHours,
      });
      const walking = [
        ...only(payloads, 'unit_marched').map((p) => ({ at: p.atHours, spent: p.stepHours })),
        ...only(payloads, 'march_progressed').map((p) => ({ at: p.atHours, spent: p.spentHours })),
      ];
      const camp = only(payloads, 'formation_change_began').find((c) => c.to === 'rest');
      return {
        onRoad: walking.reduce((sum, w) => sum + w.spent, 0),
        lastHour: Math.max(...walking.map((w) => w.at)),
        campedAt: camp?.atHours,
        reason: camp?.reason,
      };
    };

    it('the hours on the road, when they run out before the hour to be off it', () => {
      const day = firstDay(orders({ startHour: 6, latestHour: 20, maxHoursOnRoad: 4 }));
      expect(day.onRoad).toBeCloseTo(4);
      expect(day.lastHour).toBe(9);
      expect(day.campedAt).toBe(10);
      expect(day.reason).toBe('standing_orders');
    });

    it('the hour to be off the road, when it comes before the hours run out', () => {
      const day = firstDay(orders({ startHour: 6, latestHour: 9, maxHoursOnRoad: 10 }));
      expect(day.onRoad).toBeCloseTo(3);
      expect(day.lastHour).toBe(8);
      expect(day.campedAt).toBe(9);
    });

    it('the hour to be off the road alone, stepping off at dawn', () => {
      const day = firstDay(orders({ latestHour: dawn + 3 }));
      expect(day.onRoad).toBeCloseTo(3);
      expect(day.lastHour).toBe(dawn + 2);
      expect(day.campedAt).toBe(dawn + 3);
    });

    it('the hours on the road alone, stepping off at dawn', () => {
      const day = firstDay(orders({ maxHoursOnRoad: 5 }));
      expect(day.onRoad).toBeCloseTo(5);
      expect(day.campedAt).toBe(dawn + 5);
    });

    it('any limit already past when the orders are given, without taking another step', () => {
      const late = firstDay(orders({ latestHour: 12 }), 14);
      expect(late.onRoad).toBe(0);

      const u = division({ q: 2, r: 5 }, marched(6, 14));
      const { payloads } = advance(
        stateWith(u, 14, orders({ maxHoursOnRoad: 5 })),
        world,
        cfg,
        clean,
        { hours: 6 },
      );
      expect(only(payloads, 'unit_marched')).toHaveLength(0);
      expect(only(payloads, 'march_progressed')).toHaveLength(0);
    });
  });

  describe('does not take midnight as a fresh start', () => {
    /** Campaign hours the head was on the road, over a run. */
    const onRoadAt = (o: StandingOrders, from: number, hours: number): number[] => {
      const u = division({ q: 2, r: 5 });
      const { payloads } = advance(stateWith(u, from, o), world, cfg, clean, { hours });
      return [
        ...only(payloads, 'unit_marched').map((p) => p.atHours),
        ...only(payloads, 'march_progressed').map((p) => p.atHours),
      ];
    };

    it('stops at midnight when the orders say to be off the road by it', () => {
      const hours = onRoadAt(orders({ latestHour: 24 }), 20, 12);
      expect(hours).toContain(23);
      // Nothing between midnight and dawn; at dawn it is a new day under the same orders.
      for (const at of hours) expect(at < 24 || hourOfDay(at) >= dawn).toBe(true);
    });

    it('waits for dawn, not midnight, when the orders name no hour to step off', () => {
      const hours = onRoadAt(orders({ latestHour: 22 }), 12, 24);
      for (const at of hours) {
        expect(hourOfDay(at)).toBeGreaterThanOrEqual(dawn);
        expect(hourOfDay(at)).toBeLessThan(22);
      }
      expect(hours.some((at) => at >= 24)).toBe(true);
    });

    it('counts hours on the road across it', () => {
      // Three hours from ten at night: two before midnight and one after, and none more.
      const hours = onRoadAt(orders({ startHour: 0, maxHoursOnRoad: 3 }), 22, 6);
      expect(new Set(hours.map(Math.floor))).toEqual(new Set([22, 23, 24]));
    });
  });

  it('marches as before when nobody has given any', () => {
    const u = division({ q: 2, r: 5 });
    const bare = advance(stateWith(u, 6), world, cfg, clean, { hours: 10 }).payloads;
    const lifted = advance(stateWith(u, 6, orders({})), world, cfg, clean, { hours: 10 }).payloads;
    expect(lifted).toEqual(bare);
  });
});

describe('setting them', () => {
  const u = division({ q: 2, r: 5 });

  it('is logged and folded, and lifted by null', () => {
    const o = orders({ startHour: 5, latestHour: 19, maxHoursOnRoad: 10 });
    const set = apply({ kind: 'set_standing_orders', unitId: u.id, orders: o }, stateWith(u, 6), world, 'strict');
    expect(set.ok).toBe(true);
    expect(set.state.standingOrders.get(u.id)).toEqual(o);

    const lifted = apply(
      { kind: 'set_standing_orders', unitId: u.id, orders: null },
      set.state,
      world,
      'strict',
    );
    expect(lifted.state.standingOrders.has(u.id)).toBe(false);
  });

  it('refuses orders that are not there, rather than failing on them', () => {
    const out = apply(
      { kind: 'set_standing_orders', unitId: u.id } as unknown as Parameters<typeof apply>[0],
      stateWith(u, 6),
      world,
      'permissive',
      { force: true },
    );
    expect(out.ok).toBe(false);
    expect(out.violations[0]?.code).toBe(CODES.MALFORMED);
  });

  it('refuses orders that cannot be followed, even when forced', () => {
    const out = apply(
      { kind: 'set_standing_orders', unitId: u.id, orders: orders({ startHour: 19, latestHour: 5 }) },
      stateWith(u, 6),
      world,
      'permissive',
      { force: true },
    );
    expect(out.ok).toBe(false);
    expect(out.violations[0]?.code).toBe(CODES.MALFORMED);
  });

  it('shows a commander their own and a referee everyone’s', () => {
    const o = orders({ startHour: 5 });
    const state = stateWith(u, 6, o);
    const input = { campaignId: 'c', state, worldDoc: { hexes: [] }, world };

    expect(viewFor(input, REFEREE_ROLE).standingOrders).toEqual({ [u.id]: o });
    const mine = viewFor(input, commanderRole('ney'));
    expect(mine.standingOrders).toEqual({ [u.id]: o });
    expect(() => assertMasked(mine)).not.toThrow();
  });
});

describe('the referee’s daylight', () => {
  const u = division({ q: 2, r: 5 });

  it('replaces the campaign’s sunrise and sunset from the moment it is set', () => {
    const out = apply(
      { kind: 'set_daylight', sunriseHour: 4, sunsetHour: 21 },
      stateWith(u, 6),
      world,
      'strict',
    );
    expect(out.ok).toBe(true);
    const now = configAt(cfg, out.state);
    expect(isDark(now, 20)).toBe(false);
    expect(isDark(now, 21)).toBe(true);
    expect(isDark(now, 24 + 4)).toBe(false);
    // And the console is sent the same sun.
    const view = viewFor(
      { campaignId: 'c', state: out.state, worldDoc: { hexes: [] }, world },
      commanderRole('ney'),
    );
    expect(view.config.sunsetHour).toBe(21);
  });

  it('refuses a sun that sets before it rises', () => {
    const out = apply(
      { kind: 'set_daylight', sunriseHour: 20, sunsetHour: 6 },
      stateWith(u, 6),
      world,
      'strict',
    );
    expect(out.ok).toBe(false);
  });

  it('is what night fatigue is charged by', () => {
    const at = cfg.sunsetHour;
    const night = (s: CampaignState): number =>
      only(advance(s, world, cfg, clean, { hours: 2 }).payloads, 'fatigue_accrued').reduce(
        (sum, p) => sum + p.fromNight,
        0,
      );

    const byTheBook = stateWith(u, at);
    const longEvening = { ...byTheBook, daylight: { sunriseHour: 4, sunsetHour: at + 4 } };
    expect(night(byTheBook)).toBeGreaterThan(0);
    expect(night(longEvening)).toBe(0);
  });
});
