/**
 * The clock: marching, riding, and stopping to ask.
 *
 * These assert structure and ordering rather than exact hours wherever the number is a
 * consequence of terrain — but the hours a march takes over known ground *are* pinned,
 * because "one hex is one kilometre and the speed table is in km/h" is the correspondence
 * the whole ruleset hangs on and a drift in it would be silent.
 */

import { describe, expect, it } from 'vitest';

import type { Commander } from '../src/commander.js';
import { DEFAULT_CONFIG, type CampaignConfig } from '../src/config.js';
import type { Despatch } from '../src/despatch.js';
import type { EventPayload } from '../src/events.js';
import { key, type Hex, type HexKey } from '../src/hex.js';
import { makeRng, type Rng } from '../src/rng.js';
import { advance, despatchNow } from '../src/scheduler.js';
import { EMPTY_STATE, reduce, type CampaignState } from '../src/state.js';
import type { Task } from '../src/task.js';
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
  at: Hex,
  opts: Partial<Unit> = {},
  kind: UnitKind = 'infantry',
  traits: Trait[] = [],
): Unit {
  return {
    id,
    name: id,
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
    // Half a metre a man over four thousand is two kilometres of road: two hexes, which
    // keeps the column visible in these tests without dominating them.
    spacingM: 0.5,
    spacingMultiplier: 1,
    traits,
    formation: 'march',
    column: [at],
    hoursMarchedToday: 0,
    corps: null,
    ...opts,
  };
}

const commander = (
  id: string,
  faction: string,
  unitId: string,
  superiorId: string | null = null,
  autoCascade = true,
): Commander => ({ id, name: id, faction, unitId, superiorId, autoCascade });

interface Setup {
  units?: Unit[];
  commanders?: Commander[];
  tasks?: Task[];
  despatches?: Despatch[];
  clockHours?: number;
}

const stateFrom = (s: Setup): CampaignState => ({
  ...EMPTY_STATE,
  clockHours: s.clockHours ?? 6,
  nextSeq: 100,
  units: new Map((s.units ?? []).map((u) => [u.id, u])),
  commanders: new Map((s.commanders ?? []).map((c) => [c.id, c])),
  tasks: new Map((s.tasks ?? []).map((t) => [t.unitId, t])),
  despatches: new Map((s.despatches ?? []).map((d) => [d.id, d])),
});

/** Fold a run's payloads, exactly as `apply` will. */
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

/** A march from a hex to a hex, with the first leg already scheduled as `set_task` does. */
function marchTo(u: Unit, destination: Hex, atHours: number, next: Hex): Task {
  return {
    unitId: u.id,
    destination,
    via: [],
    setAtHours: atHours,
    fromDespatchId: null,
    nextHex: next,
    // Infantry off-road is 2 km/h, so half an hour a hex.
    arrivesAtHours: atHours + 0.5,
    complete: false,
  };
}

/** Dice that never come up 1, and dice that always do. */
const clean: Rng = { next: () => 0.9, int: () => 5, d6: () => 6, pool: (n) => Array(n).fill(6) };
const cursed: Rng = { next: () => 0, int: () => 0, d6: () => 1, pool: (n) => Array(n).fill(1) };

const kinds = (payloads: readonly EventPayload[]): string[] => payloads.map((p) => p.kind);

describe('marching', () => {
  const red = unit('red-1', 'red', { q: 5, r: 5 });
  const ney = commander('ney', 'red', 'red-1');
  const task = marchTo(red, { q: 10, r: 5 }, 6, { q: 6, r: 5 });

  it('covers open ground at the rate the speed table gives', () => {
    const state = stateFrom({ units: [red], commanders: [ney], tasks: [task] });
    const { payloads } = advance(state, world, cfg, clean, { hours: 3 });
    const marches = payloads.filter((p) => p.kind === 'unit_marched');

    // Five hexes off-road at 2 km/h is two and a half hours, and the whole of it fits in
    // the three the referee asked for.
    expect(marches).toHaveLength(5);
    const after = fold(state, payloads);
    expect(after.units.get('red-1')!.column[0]).toEqual({ q: 10, r: 5 });
  });

  it('drags its tail along behind it', () => {
    const state = stateFrom({ units: [red], commanders: [ney], tasks: [task] });
    const after = fold(state, advance(state, world, cfg, clean, { hours: 3 }).payloads);
    const column = after.units.get('red-1')!.column;
    // Two kilometres of division is two hexes of road, and the second one is the ground
    // it has just left rather than the ground in front of it.
    expect(column[0]).toEqual({ q: 10, r: 5 });
    expect(column[1]).toEqual({ q: 9, r: 5 });
  });

  it('stops when it arrives, and says so', () => {
    const state = stateFrom({ units: [red], commanders: [ney], tasks: [task] });
    const { payloads } = advance(state, world, cfg, clean, { hours: 6 });

    expect(kinds(payloads)).toContain('task_completed');
    const raised = payloads.filter((p) => p.kind === 'decision_raised');
    expect(raised).toHaveLength(1);
    expect(raised[0]).toMatchObject({ decision: { trigger: 'objective_reached' } });

    // And having arrived, it does not wander: a second advance changes nothing.
    const after = fold(state, payloads);
    const again = advance(after, world, cfg, clean, { hours: 6 });
    expect(again.payloads.filter((p) => p.kind === 'unit_marched')).toHaveLength(0);
  });

  it('halts at the twenty-hour cap and starts again at midnight', () => {
    // Nineteen and a half hours already marched leaves half an hour: one hex, and then
    // the column stands until the day rolls over.
    const tired = unit('red-1', 'red', { q: 5, r: 5 }, { hoursMarchedToday: 19.5 });
    const state = stateFrom({
      units: [tired],
      commanders: [ney],
      tasks: [marchTo(tired, { q: 20, r: 5 }, 20, { q: 6, r: 5 })],
      clockHours: 20,
    });

    const overnight = advance(state, world, cfg, clean, { hours: 6 });
    const marched = overnight.payloads
      .filter((p) => p.kind === 'unit_marched')
      .map((p) => (p.kind === 'unit_marched' ? p.atHours : 0));

    expect(kinds(overnight.payloads)).toContain('day_rolled');
    // One hex before the cap bites, then a four-hour halt in the dark, then the column
    // steps off again on the far side of midnight.
    expect(marched.slice(0, 3)).toEqual([20.5, 24.5, 25]);
    expect(fold(state, overnight.payloads).units.get('red-1')!.hoursMarchedToday).toBeLessThan(
      cfg.maxMarchHoursPerDay,
    );
  });
});

describe('the clock', () => {
  it('does not move on a quiet advance beyond the hour asked for', () => {
    const state = stateFrom({ units: [unit('red-1', 'red', { q: 5, r: 5 })] });
    const { payloads, toHours } = advance(state, world, cfg, clean, { hours: 4 });

    // Nothing is happening, so nothing is written but the hour itself.
    expect(kinds(payloads)).toEqual(['clock_advanced']);
    expect(toHours).toBe(10);
  });

  it('stamps a march at the hour it happened, not at the hour it was asked for', () => {
    const red = unit('red-1', 'red', { q: 5, r: 5 });
    const state = stateFrom({
      units: [red],
      commanders: [commander('ney', 'red', 'red-1')],
      tasks: [marchTo(red, { q: 7, r: 5 }, 6, { q: 6, r: 5 })],
    });
    const { payloads } = advance(state, world, cfg, clean, { hours: 12 });

    const clocks = payloads.filter((p) => p.kind === 'clock_advanced');
    expect(clocks.map((p) => (p.kind === 'clock_advanced' ? p.toHours : 0))).toContain(6.5);
  });

  it('refuses to run further than config allows in one go', () => {
    const state = stateFrom({ units: [unit('red-1', 'red', { q: 5, r: 5 })] });
    const capped: CampaignConfig = { ...cfg, maxAdvanceHours: 5 };
    expect(advance(state, world, capped, clean, { hours: 5000 }).toHours).toBe(11);
  });
});

describe('discovery', () => {
  const red = unit('red-1', 'red', { q: 5, r: 5 });
  const blue = unit('blue-1', 'blue', { q: 12, r: 5 });
  const ney = commander('ney', 'red', 'red-1');
  const wellington = commander('wellington', 'blue', 'blue-1');

  const marching = (): CampaignState =>
    stateFrom({
      units: [red, blue],
      commanders: [ney, wellington],
      tasks: [marchTo(red, { q: 20, r: 5 }, 6, { q: 6, r: 5 })],
    });

  it('halts the clock at the first contact when asked to', () => {
    const { payloads, halted } = advance(marching(), world, cfg, clean, {
      hours: 12,
      untilDecision: true,
    });

    expect(halted).not.toBeNull();
    expect(halted!.trigger).toBe('enemy_contact');
    // And it really stopped: the column is nowhere near its destination.
    const after = fold(marching(), payloads);
    expect(after.units.get('red-1')!.column[0]!.q).toBeLessThan(20);
  });

  it('marches straight past when it is not asked to', () => {
    const { payloads, halted } = advance(marching(), world, cfg, clean, { hours: 12 });
    expect(halted).not.toBeNull();
    const after = fold(marching(), payloads);
    // The decision was still raised — the referee will see it in his queue — but the
    // clock ran on, which is the difference between the two controls.
    expect(after.units.get('red-1')!.column[0]).toEqual({ q: 20, r: 5 });
  });

  it('raises contact for both sides, since both can see', () => {
    const { payloads } = advance(marching(), world, cfg, clean, { hours: 12 });
    const raised = payloads
      .filter((p) => p.kind === 'decision_raised')
      .map((p) => (p.kind === 'decision_raised' ? p.decision : null));

    const contacts = raised.filter((d) => d?.trigger === 'enemy_contact');
    expect(contacts.map((d) => d!.commanderId).sort()).toEqual(['ney', 'wellington']);
  });

  it('does not halt on an enemy it was already looking at', () => {
    const close = stateFrom({
      units: [red, unit('blue-1', 'blue', { q: 6, r: 5 })],
      commanders: [ney, wellington],
      tasks: [marchTo(red, { q: 5, r: 20 }, 6, { q: 5, r: 6 })],
    });
    const { halted } = advance(close, world, cfg, clean, { hours: 2, untilDecision: true });
    // They are already in contact. An advance that stopped instantly, every time, would
    // make two armies in touch impossible to run at all.
    expect(halted).toBeNull();
  });

  it('sends a report up the chain automatically', () => {
    const state = stateFrom({
      units: [red, blue, unit('red-2', 'red', { q: 30, r: 30 })],
      commanders: [
        commander('soult', 'red', 'red-2'),
        commander('ney', 'red', 'red-1', 'soult'),
        wellington,
      ],
      tasks: [marchTo(red, { q: 20, r: 5 }, 6, { q: 6, r: 5 })],
    });

    const { payloads } = advance(state, world, cfg, clean, { hours: 12 });
    const reports = payloads
      .filter((p) => p.kind === 'despatch_sent')
      .map((p) => (p.kind === 'despatch_sent' ? p.despatch : null))
      .filter((d) => d?.kind === 'report');

    expect(reports).toHaveLength(1);
    expect(reports[0]).toMatchObject({ from: 'ney', to: 'soult' });
    expect(reports[0]!.body.contacts).toHaveLength(1);
    // A division thirty kilometres away is a rider's journey, not a shout.
    expect(reports[0]!.handed).toBe(false);
  });
});

describe('riders', () => {
  const near = unit('red-1', 'red', { q: 5, r: 5 });
  const far = unit('red-2', 'red', { q: 25, r: 5 });
  const ney = commander('ney', 'red', 'red-1');
  const kellermann = commander('kellermann', 'red', 'red-2', 'ney', false);

  const apart = (): CampaignState =>
    stateFrom({ units: [near, far], commanders: [ney, kellermann] });

  const order = { text: 'Move on Quatre Bras with all speed.' };

  it('takes real hours to arrive', () => {
    const state = apart();
    const sent = despatchNow(state, world, cfg, clean, {
      from: 'ney',
      to: 'kellermann',
      kind: 'order',
      body: order,
    });

    expect(kinds(sent)).toEqual(['despatch_sent']);
    const riding = fold(state, sent);
    const d = [...riding.despatches.values()][0]!;
    expect(d.fate.kind).toBe('in_transit');

    // Twenty hexes off-road at six km/h is three and a third hours. One hour is not
    // enough; four is.
    const short = advance(riding, world, cfg, clean, { hours: 1 });
    expect(kinds(short.payloads)).not.toContain('despatch_delivered');

    const long = advance(riding, world, cfg, clean, { hours: 4 });
    const delivered = long.payloads.find((p) => p.kind === 'despatch_delivered');
    expect(delivered).toBeDefined();
    expect(delivered!.kind === 'despatch_delivered' && delivered.atHours).toBeCloseTo(9.5, 0);
  });

  it('is handed over on the spot between formations that touch', () => {
    const state = stateFrom({
      units: [near, unit('red-2', 'red', { q: 6, r: 5 })],
      commanders: [ney, kellermann],
    });
    const sent = despatchNow(state, world, cfg, clean, {
      from: 'ney',
      to: 'kellermann',
      kind: 'order',
      body: order,
    });

    expect(kinds(sent)).toEqual(['despatch_sent', 'despatch_delivered', 'decision_raised']);
    const d = (sent[0] as { despatch: Despatch }).despatch;
    expect(d.handed).toBe(true);
    // No ride at all, so nothing to intercept and nothing to betray a position.
    expect(d.route).toHaveLength(1);
  });

  it('chases a corps that has marched on', () => {
    const state = apart();
    const sent = fold(
      state,
      despatchNow(state, world, cfg, clean, {
        from: 'ney',
        to: 'kellermann',
        kind: 'order',
        body: order,
      }),
    );

    // The addressee marches away while the rider is on the road.
    const moving: CampaignState = {
      ...sent,
      tasks: new Map([['red-2', marchTo(far, { q: 25, r: 20 }, 6, { q: 25, r: 6 })]]),
    };

    const { payloads } = advance(moving, world, cfg, clean, { hours: 12 });
    expect(kinds(payloads)).toContain('despatch_progressed');
    expect(kinds(payloads)).toContain('despatch_delivered');
  });

  it('cascades an arriving order to subordinates who are run by the referee', () => {
    const state = stateFrom({
      units: [near, unit('red-2', 'red', { q: 6, r: 5 }), unit('red-3', 'red', { q: 30, r: 5 })],
      commanders: [
        ney,
        commander('kellermann', 'red', 'red-2', 'ney', true),
        commander('soult', 'red', 'red-3', 'kellermann', true),
      ],
    });

    const sent = despatchNow(state, world, cfg, clean, {
      from: 'ney',
      to: 'kellermann',
      kind: 'order',
      body: order,
    });

    const onward = sent
      .filter((p) => p.kind === 'despatch_sent')
      .map((p) => (p.kind === 'despatch_sent' ? p.despatch : null));

    expect(onward).toHaveLength(2);
    expect(onward[1]).toMatchObject({ from: 'kellermann', to: 'soult', kind: 'order' });
    // Verbatim, and pointing back at the order it came from.
    expect(onward[1]!.body.text).toBe(order.text);
    expect(onward[1]!.inReplyTo).toBe(onward[0]!.id);
  });

  it('does not cascade past a commander a player is holding', () => {
    const state = stateFrom({
      units: [near, unit('red-2', 'red', { q: 6, r: 5 }), unit('red-3', 'red', { q: 30, r: 5 })],
      commanders: [
        ney,
        commander('kellermann', 'red', 'red-2', 'ney', false),
        commander('soult', 'red', 'red-3', 'kellermann', true),
      ],
    });
    const sent = despatchNow(state, world, cfg, clean, {
      from: 'ney',
      to: 'kellermann',
      kind: 'order',
      body: order,
    });
    expect(sent.filter((p) => p.kind === 'despatch_sent')).toHaveLength(1);
  });
});

describe('interception', () => {
  const red = unit('red-1', 'red', { q: 5, r: 5 });
  const far = unit('red-2', 'red', { q: 25, r: 5 });
  // Sitting squarely across the road the rider must take.
  const picket = unit('blue-1', 'blue', { q: 15, r: 5 }, {}, 'cavalry', ['scout']);

  const inTheWay = (): CampaignState =>
    stateFrom({
      units: [red, far, picket],
      commanders: [
        commander('ney', 'red', 'red-1'),
        commander('kellermann', 'red', 'red-2', 'ney', false),
        commander('wellington', 'blue', 'blue-1'),
      ],
    });

  const ride = (state: CampaignState): CampaignState =>
    fold(
      state,
      despatchNow(state, world, cfg, clean, {
        from: 'ney',
        to: 'kellermann',
        kind: 'order',
        body: { text: 'Hold the crossroads.' },
      }),
    );

  it('lets the rider through when the dice are kind', () => {
    const { payloads } = advance(ride(inTheWay()), world, cfg, clean, { hours: 6 });
    expect(kinds(payloads)).toContain('despatch_delivered');
    expect(kinds(payloads)).not.toContain('despatch_stopped');
  });

  it('takes the paper when they are not, and records the dice', () => {
    const riding = ride(inTheWay());
    const { payloads } = advance(riding, world, cfg, cursed, { hours: 6 });

    const stopped = payloads.find((p) => p.kind === 'despatch_stopped');
    expect(stopped).toBeDefined();
    if (stopped?.kind !== 'despatch_stopped') throw new Error('unreachable');

    // A scouting cavalry division is one die plus three: four ones is a capture twice
    // over, and the dice are in the event so a referee can show them.
    expect(stopped.outcome).toBe('captured');
    expect(stopped.dice).toEqual([1, 1, 1, 1]);
    expect(stopped.by).toBe('blue');

    const after = fold(riding, payloads);
    const taken = [...after.despatches.values()][0]!;
    expect(taken.fate).toMatchObject({ kind: 'captured', by: 'blue' });
    // And it never arrives, however long the clock runs.
    const later = advance(after, world, cfg, clean, { hours: 24 });
    expect(kinds(later.payloads)).not.toContain('despatch_delivered');
  });

  it('loses the rider without the paper on a single one', () => {
    const oneOne: Rng = {
      ...clean,
      pool: (n) => [1, ...Array(Math.max(0, n - 1)).fill(6)],
    };
    const riding = ride(inTheWay());
    const stopped = advance(riding, world, cfg, oneOne, { hours: 6 }).payloads.find(
      (p) => p.kind === 'despatch_stopped',
    );
    expect(stopped?.kind === 'despatch_stopped' && stopped.outcome).toBe('lost');
  });
});

describe('determinism', () => {
  it('produces the same events from the same state and the same dice', () => {
    const red = unit('red-1', 'red', { q: 5, r: 5 });
    const state = stateFrom({
      units: [red, unit('blue-1', 'blue', { q: 12, r: 5 })],
      commanders: [
        commander('ney', 'red', 'red-1'),
        commander('wellington', 'blue', 'blue-1'),
      ],
      tasks: [marchTo(red, { q: 20, r: 5 }, 6, { q: 6, r: 5 })],
    });

    const a = advance(state, world, cfg, makeRng(7), { hours: 12 });
    const b = advance(state, world, cfg, makeRng(7), { hours: 12 });
    expect(JSON.stringify(a.payloads)).toBe(JSON.stringify(b.payloads));
  });

  it('reaches the same state in one advance as in six', () => {
    const red = unit('red-1', 'red', { q: 5, r: 5 });
    const state = stateFrom({
      units: [red],
      commanders: [commander('ney', 'red', 'red-1')],
      tasks: [marchTo(red, { q: 15, r: 5 }, 6, { q: 6, r: 5 })],
    });

    const once = fold(state, advance(state, world, cfg, clean, { hours: 6 }).payloads);

    let stepwise = state;
    for (let i = 0; i < 6; i++) {
      stepwise = fold(stepwise, advance(stepwise, world, cfg, clean, { hours: 1 }).payloads);
    }

    expect(stepwise.clockHours).toBe(once.clockHours);
    expect(stepwise.units.get('red-1')!.column).toEqual(once.units.get('red-1')!.column);
  });

  it('mutates nothing it was given', () => {
    const red = unit('red-1', 'red', { q: 5, r: 5 });
    const state = stateFrom({
      units: [red],
      commanders: [commander('ney', 'red', 'red-1')],
      tasks: [marchTo(red, { q: 15, r: 5 }, 6, { q: 6, r: 5 })],
    });
    const before = JSON.stringify([...state.units.values()]);

    advance(state, world, cfg, clean, { hours: 6 });
    expect(JSON.stringify([...state.units.values()])).toBe(before);
    expect(state.clockHours).toBe(6);
  });
});
