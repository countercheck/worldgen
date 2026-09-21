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
import { marchFatigueAt } from '../src/fatigue.js';
import { advance, despatchNow } from '../src/scheduler.js';
import { EMPTY_STATE, reduce, type CampaignState } from '../src/state.js';
import { contestants, contestedHex, type Task } from '../src/task.js';
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

/** The same ground with a wall of water down one column of hexes. */
function splitWorld(atQ: number, size = 40): World {
  const w = flatWorld(size);
  for (let r = 0; r < size; r++) {
    const hex = w.hexes.get(key({ q: atQ, r }))!;
    w.hexes.set(key({ q: atQ, r }), { ...hex, terrainClass: 'open_water' });
  }
  return w;
}

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
    // Half a metre a soldier over four thousand is two kilometres of road: two hexes, which
    // keeps the column visible in these tests without dominating them.
    spacingM: 0.5,
    spacingMultiplier: 1,
    traits,
    formation: 'march',
    formationChange: null,
    column: [at],
    hoursMarchedToday: 0,
    corps: null,
    parentUnitId: null,
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
  /** Ground being fought over. */
  battle?: Hex[];
}

const stateFrom = (s: Setup): CampaignState => ({
  ...EMPTY_STATE,
  clockHours: s.clockHours ?? 6,
  nextSeq: 100,
  units: new Map((s.units ?? []).map((u) => [u.id, u])),
  commanders: new Map((s.commanders ?? []).map((c) => [c.id, c])),
  tasks: new Map((s.tasks ?? []).map((t) => [t.unitId, t])),
  despatches: new Map((s.despatches ?? []).map((d) => [d.id, d])),
  battle: new Set((s.battle ?? []).map(key)),
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
function marchTo(
  u: Unit,
  destination: Hex,
  atHours: number,
  next: Hex,
  via: readonly Hex[] = [],
): Task {
  return {
    unitId: u.id,
    destination,
    via,
    setAtHours: atHours,
    fromDespatchId: null,
    nextHex: next,
    progressHours: 0,
    complete: false,
    viaIndex: 0,
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

    const overnight = advance(state, world, cfg, clean, { hours: 10 });
    const marched = overnight.payloads
      .filter((p) => p.kind === 'unit_marched')
      .map((p) => (p.kind === 'unit_marched' ? p.atHours : 0));
    const after = fold(state, overnight.payloads);

    expect(kinds(overnight.payloads)).toContain('day_rolled');

    // One hex, and the twentieth hour is spent. The column builds a camp where it stands —
    // nobody decides to stop after twenty hours on the road, they simply stop — and it is
    // then in a camp it has to get out of. Midnight resets the day but not the camp: it
    // breaks camp first, two hours of it, and steps off on the far side of that.
    expect(marched[0]).toBe(20);
    expect(marched[1]).toBe(26);
    expect(after.units.get('red-1')!.formation).toBe('march');
    expect(after.units.get('red-1')!.hoursMarchedToday).toBeLessThan(cfg.maxMarchHoursPerDay);

    const changes = overnight.payloads.filter((p) => p.kind === 'formation_changed');
    expect(changes.map((p) => (p.kind === 'formation_changed' ? p.to : ''))).toEqual([
      'rest',
      'march',
    ]);
  });
});

describe('marching by way of somewhere', () => {
  const red = unit('red-1', 'red', { q: 5, r: 5 });
  const ney = commander('ney', 'red', 'red-1');

  /** The hexes a run's column actually entered, head first, in order. */
  const entered = (payloads: readonly EventPayload[]): HexKey[] =>
    payloads.filter((p) => p.kind === 'unit_marched').map((p) => key(p.to));

  it('goes through the waypoint rather than straight to the destination', () => {
    // Due east to { q: 10, r: 5 }, but by way of ground four hexes south of the line. A
    // scheduler that ignored `via` would march the five hexes east and stop.
    const waypoint = { q: 7, r: 9 };
    const task = marchTo(red, { q: 10, r: 5 }, 6, { q: 6, r: 5 }, [waypoint]);
    const state = stateFrom({ units: [red], commanders: [ney], tasks: [task] });

    const { payloads } = advance(state, world, cfg, clean, { hours: 24 });
    const hexes = entered(payloads);

    expect(hexes).toContain(key(waypoint));
    expect(hexes.indexOf(key(waypoint))).toBeLessThan(hexes.indexOf(key({ q: 10, r: 5 })));
    expect(fold(state, payloads).units.get('red-1')!.column[0]).toEqual({ q: 10, r: 5 });
  });

  it('makes them in the order given', () => {
    const first = { q: 5, r: 9 };
    const second = { q: 9, r: 9 };
    const task = marchTo(red, { q: 10, r: 5 }, 6, { q: 6, r: 5 }, [first, second]);
    const state = stateFrom({ units: [red], commanders: [ney], tasks: [task] });

    const hexes = entered(advance(state, world, cfg, clean, { hours: 24 }).payloads);
    expect(hexes.indexOf(key(first))).toBeGreaterThanOrEqual(0);
    expect(hexes.indexOf(key(second))).toBeGreaterThan(hexes.indexOf(key(first)));
  });

  it('does not finish early when the route crosses its own destination', () => {
    // The destination lies between the column and the waypoint, so the head stands on it
    // hours before the march is done. Arrival is "on the destination *and* nothing left to
    // make" — testing only the hex would halt the column on ground it was passing over.
    const destination = { q: 8, r: 5 };
    const task = marchTo(red, destination, 6, { q: 6, r: 5 }, [{ q: 12, r: 5 }]);
    const state = stateFrom({ units: [red], commanders: [ney], tasks: [task] });

    const { payloads } = advance(state, world, cfg, clean, { hours: 24 });
    const hexes = entered(payloads);

    // Passed over it, went on to the waypoint, and came back to it.
    expect(hexes.filter((h) => h === key(destination))).toHaveLength(2);
    expect(hexes).toContain(key({ q: 12, r: 5 }));

    const after = fold(state, payloads);
    expect(after.units.get('red-1')!.column[0]).toEqual(destination);
    expect(after.tasks.get('red-1')!.complete).toBe(true);
  });

  it('counts each waypoint off as the head passes it, and never counts back', () => {
    const task = marchTo(red, { q: 10, r: 5 }, 6, { q: 6, r: 5 }, [
      { q: 5, r: 9 },
      { q: 9, r: 9 },
    ]);
    const state = stateFrom({ units: [red], commanders: [ney], tasks: [task] });

    // Fold one event at a time, so the index is read at every step of the march rather
    // than only at the end.
    let s = state;
    const seen: number[] = [];
    for (const payload of advance(state, world, cfg, clean, { hours: 24 }).payloads) {
      s = fold(s, [payload]);
      const t = s.tasks.get('red-1');
      if (t !== undefined) seen.push(t.viaIndex);
    }

    expect(seen.at(-1)).toBe(2);
    for (let i = 1; i < seen.length; i++) expect(seen[i]).toBeGreaterThanOrEqual(seen[i - 1]!);
  });

  it('halts at a waypoint it cannot reach rather than skipping it', () => {
    // Water across the whole of column 8. The destination is on this side and perfectly
    // reachable; the waypoint is on the far shore. The column has nowhere to go, and that
    // is not arrival.
    const split = flatWorld();
    for (let r = 0; r < 40; r++) {
      const h = split.hexes.get(key({ q: 8, r }))!;
      split.hexes.set(key({ q: 8, r }), { ...h, terrainClass: 'open_water' });
    }

    const task = marchTo(red, { q: 6, r: 5 }, 6, { q: 6, r: 5 }, [{ q: 12, r: 5 }]);
    const state = stateFrom({ units: [red], commanders: [ney], tasks: [task] });
    const { payloads } = advance(state, split, cfg, clean, { hours: 24 });

    expect(kinds(payloads)).not.toContain('task_completed');
    expect(payloads.filter((p) => p.kind === 'decision_raised')[0]).toMatchObject({
      decision: { trigger: 'crossing_impassable' },
    });
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
      // Six hexes at two km/h is three hours, so the march spans several of them.
      tasks: [marchTo(red, { q: 11, r: 5 }, 6, { q: 6, r: 5 })],
    });
    const { payloads } = advance(state, world, cfg, clean, { hours: 12 });

    // On the hour, because the hour is the only time there is. A column given the hour
    // from six to seven is on the road at six.
    const clocks = payloads.filter((p) => p.kind === 'clock_advanced');
    const stamps = clocks.map((p) => (p.kind === 'clock_advanced' ? p.toHours : 0));
    expect(stamps).toContain(7);
    expect(stamps.every((h) => Number.isInteger(h))).toBe(true);
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
    // The decision was still raised — the referee will see it in their queue — but the
    // clock ran on, which is the difference between the two controls.
    //
    // It does not reach { q: 20, r: 5 }, and not because the clock stopped: Wellington's
    // division is standing on { q: 12, r: 5 }, square across the line of march, and a
    // column cannot walk through one that is already there. It closes up to the hex
    // before and stops, which is traffic rather than a decision.
    expect(after.units.get('red-1')!.column[0]).toEqual({ q: 11, r: 5 });
    expect(kinds(payloads)).toContain('march_blocked');
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
    // On the hour: a rider who would have reached their addressee at half past nine reaches
    // them at ten, because there is no half past ten for them to arrive at.
    expect(delivered!.kind === 'despatch_delivered' && delivered.atHours).toBe(10);
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

    // The report is not an extra: every despatch carries word of where its sender stood
    // when they sealed it, so arriving paper refreshes the recipient's picture of the commander
    // who wrote it as well as delivering what they wrote.
    expect(kinds(sent)).toEqual([
      'despatch_sent',
      'despatch_delivered',
      'report_filed',
      'decision_raised',
    ]);
    const d = (sent[0] as { despatch: Despatch }).despatch;
    expect(d.handed).toBe(true);
    expect(d.body.unitReport?.unitId).toBe('red-1');
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

describe('the pace of a march', () => {
  // Cavalry at five km/h crosses a hex in 0.2 hours, which is less than one tick. A
  // scheduler that moved each column at most once a tick would cap every formation at
  // 1 / tickHours km/h — four, on the default — and it would look like mud rather than
  // like a bug, which is why the rate is pinned rather than the count.
  const fast = unit('red-1', 'red', { q: 5, r: 5 }, {}, 'cavalry', ['very_fast']);
  const murat = commander('murat', 'red', 'red-1');

  const task: Task = {
    unitId: 'red-1',
    destination: { q: 15, r: 5 },
    via: [],
    setAtHours: 6,
    fromDespatchId: null,
    nextHex: { q: 6, r: 5 },
    progressHours: 0,
    complete: false,
    viaIndex: 0,
  };

  it('covers more than one hex in a tick when the speed table says so', () => {
    const state = stateFrom({ units: [fast], commanders: [murat], tasks: [task] });
    const { payloads } = advance(state, world, cfg, clean, { hours: 1 });

    // Five km/h for an hour is five hexes, not the four a tick-per-hex would allow.
    expect(payloads.filter((p) => p.kind === 'unit_marched')).toHaveLength(5);
    const after = fold(state, payloads);
    expect(after.units.get('red-1')!.column[0]).toEqual({ q: 10, r: 5 });
  });

  it('stamps every step of an hour at that hour', () => {
    const state = stateFrom({ units: [fast], commanders: [murat], tasks: [task] });
    const { payloads } = advance(state, world, cfg, clean, { hours: 1 });
    const hours = payloads
      .filter((p) => p.kind === 'unit_marched')
      .map((p) => (p.kind === 'unit_marched' ? Number(p.atHours.toFixed(3)) : 0));

    // Five hexes, all of them inside the hour from six to seven, and all stamped at six.
    // There is no six-twelve for the second of them to happen at: the hour is the smallest
    // thing this clock has, and what a column does inside one happens at that hour.
    expect(hours).toEqual([6, 6, 6, 6, 6]);
  });
});

describe('a column that cannot get there', () => {
  const split = splitWorld(8);
  const red = unit('red-1', 'red', { q: 5, r: 5 });
  const ney = commander('ney', 'red', 'red-1');

  /** The far bank: reachable ground on the map, with a wall of water in front of it. */
  const stranded = (): CampaignState =>
    stateFrom({
      units: [red],
      commanders: [ney],
      tasks: [marchTo(red, { q: 12, r: 5 }, 6, { q: 6, r: 5 })],
    });

  it('raises the crossing, and does not call the march done', () => {
    const state = stranded();
    const { payloads } = advance(state, split, cfg, clean, { hours: 12 });

    const raised = payloads
      .filter((p) => p.kind === 'decision_raised')
      .map((p) => (p.kind === 'decision_raised' ? p.decision.trigger : null));
    expect(raised).toContain('crossing_impassable');

    // The console reads the task, and "complete" beside a formation stuck on the wrong
    // bank is a lie the referee would act on.
    expect(kinds(payloads)).not.toContain('task_completed');
    const after = fold(state, payloads);
    expect(after.tasks.get('red-1')!.complete).toBe(false);
    // It stopped where the route ran out rather than at the bank: the router plans from
    // where the head now stands, and from here there is no path to the far side at all.
    expect(after.units.get('red-1')!.column[0]).toEqual({ q: 6, r: 5 });
  });

  it('still calls an arrival an arrival', () => {
    const state = stateFrom({
      units: [red],
      commanders: [ney],
      tasks: [marchTo(red, { q: 7, r: 5 }, 6, { q: 6, r: 5 })],
    });
    const { payloads } = advance(state, split, cfg, clean, { hours: 12 });
    expect(kinds(payloads)).toContain('task_completed');
    expect(fold(state, payloads).tasks.get('red-1')!.complete).toBe(true);
  });
});

describe('a rider with nowhere to ride', () => {
  const split = splitWorld(15);
  const near = unit('red-1', 'red', { q: 5, r: 5 });
  const far = unit('red-2', 'red', { q: 25, r: 5 });
  const ney = commander('ney', 'red', 'red-1');
  const kellermann = commander('kellermann', 'red', 'red-2', 'ney', false);

  const sundered = (): CampaignState =>
    stateFrom({ units: [near, far], commanders: [ney, kellermann] });

  it('does not put the paper in their hand across an ocean', () => {
    const state = sundered();
    const payloads = despatchNow(state, split, cfg, clean, {
      from: 'ney',
      to: 'kellermann',
      kind: 'order',
      body: { text: 'Close on me.' },
    });

    // `check` has already raised this as a soft violation; a referee may send them anyway.
    // What they must not get is a delivery at the hour it was written.
    expect(kinds(payloads)).not.toContain('despatch_delivered');
    const after = fold(state, payloads);
    const sent = [...after.despatches.values()][0]!;
    expect(sent.fate.kind).toBe('in_transit');
    expect(sent.handed).toBe(false);
  });

  it('is still out there, and finds their addressee if the ground ever allows it', () => {
    const state = sundered();
    const after = fold(
      state,
      despatchNow(state, split, cfg, clean, {
        from: 'ney',
        to: 'kellermann',
        kind: 'order',
        body: { text: 'Close on me.' },
      }),
    );

    // Hours of riding change nothing while the water is in the way.
    const stuck = advance(after, split, cfg, clean, { hours: 12 });
    expect(kinds(stuck.payloads)).not.toContain('despatch_delivered');

    // Put the addressee on this side of it and the same rider re-plans and gets through.
    const moved: CampaignState = {
      ...fold(after, stuck.payloads),
      units: new Map(after.units).set('red-2', unit('red-2', 'red', { q: 9, r: 5 })),
    };
    const found = advance(moved, split, cfg, clean, { hours: 12 });
    expect(kinds(found.payloads)).toContain('despatch_delivered');
  });
});

describe('what a despatch says about its sender', () => {
  const near = unit('red-1', 'red', { q: 5, r: 5 });
  const far = unit('red-2', 'red', { q: 25, r: 5 });
  const ney = commander('ney', 'red', 'red-1');
  const kellermann = commander('kellermann', 'red', 'red-2', 'ney', false);

  it('is the engine\'s own return, not the one the caller wrote', () => {
    // A forged report is believed — it is filed as knowledge the moment it arrives — so a
    // body that could overwrite the server's own would let anyone holding a seat feed their
    // own side false intelligence signed by a real subordinate.
    const state = stateFrom({ units: [near, far], commanders: [ney, kellermann] });
    const payloads = despatchNow(state, world, cfg, clean, {
      from: 'ney',
      to: 'kellermann',
      kind: 'report',
      body: {
        text: 'All quiet.',
        unitReport: {
          unitId: 'FORGED',
          name: 'FORGED',
          faction: 'red',
          kind: 'infantry',
          echelon: 'division',
          atHours: 0,
          head: { q: 39, r: 39 },
          paperStrength: 1,
          fatigue: 0,
          formation: 'march',
          provisions: 0,
          corps: null,
        },
      },
    });

    const sent = payloads.find((p) => p.kind === 'despatch_sent');
    const report = sent?.kind === 'despatch_sent' ? sent.despatch.body.unitReport : undefined;
    expect(report?.unitId).toBe('red-1');
    expect(report?.head).toEqual({ q: 5, r: 5 });
    // The prose is their own and is left alone.
    expect(sent?.kind === 'despatch_sent' ? sent.despatch.body.text : null).toBe('All quiet.');
  });

  it('is the cascading commander\'s own position, not the original sender\'s', () => {
    // A rider coming from the corps commander knows where the corps commander was. They have
    // never been near the army headquarters that wrote the order in the first place.
    // Standing beside them, so the paper is handed over and the cascade happens at once
    // rather than a rider's journey later.
    const army = unit('red-3', 'red', { q: 4, r: 5 });
    const state = stateFrom({
      units: [near, far, army],
      commanders: [
        commander('napoleon', 'red', 'red-3'),
        commander('ney', 'red', 'red-1', 'napoleon', true),
        commander('kellermann', 'red', 'red-2', 'ney', false),
      ],
    });

    const payloads = despatchNow(state, world, cfg, clean, {
      from: 'napoleon',
      to: 'ney',
      kind: 'order',
      body: { text: 'Take Quatre Bras.' },
    });

    const cascaded = payloads
      .filter((p) => p.kind === 'despatch_sent')
      .map((p) => (p.kind === 'despatch_sent' ? p.despatch : null))
      .find((d) => d?.from === 'ney');

    expect(cascaded).toBeDefined();
    expect(cascaded!.body.unitReport?.unitId).toBe('red-1');
    expect(cascaded!.body.text).toBe('Take Quatre Bras.');
  });
});

describe('who decides for a formation', () => {
  it('is the senior commander riding with it, by depth in the chain', () => {
    // A corps commander fallen back on one of their own divisions decides for it. Neither
    // they nor the divisional commander is the army commander, so "has no superior" cannot
    // tell them apart and alphabetical order would hand it to the junior commander.
    const red = unit('red-1', 'red', { q: 5, r: 5 });
    const army = unit('red-9', 'red', { q: 30, r: 30 });
    const state = stateFrom({
      units: [red, army],
      commanders: [
        commander('zieten', 'red', 'red-9'),
        commander('mouton', 'red', 'red-1', 'zieten'),
        commander('bertrand', 'red', 'red-1', 'mouton'),
      ],
      tasks: [marchTo(red, { q: 7, r: 5 }, 6, { q: 6, r: 5 })],
    });

    const { payloads } = advance(state, world, cfg, clean, { hours: 3 });
    const raised = payloads
      .filter((p) => p.kind === 'decision_raised')
      .map((p) => (p.kind === 'decision_raised' ? p.decision : null));

    expect(raised).toHaveLength(1);
    expect(raised[0]!.commanderId).toBe('mouton');
  });
});

describe('what the commander who saw it keeps', () => {
  it('files the contact for the observer, not only for their superior', () => {
    // The store asks every commander what they can see after the whole command, from the
    // final state. An enemy sighted and lost again during a long advance would never
    // reach the observer's own contacts at all — their superior would get it by despatch
    // and they would not, which is precisely backwards.
    const red = unit('red-1', 'red', { q: 5, r: 5 });
    const blue = unit('blue-1', 'blue', { q: 12, r: 5 });
    const soult = commander('soult', 'red', 'red-2');
    const state = stateFrom({
      units: [red, blue, unit('red-2', 'red', { q: 30, r: 30 })],
      commanders: [
        soult,
        commander('ney', 'red', 'red-1', 'soult'),
        commander('wellington', 'blue', 'blue-1'),
      ],
      tasks: [marchTo(red, { q: 20, r: 5 }, 6, { q: 6, r: 5 })],
    });

    const { payloads } = advance(state, world, cfg, clean, { hours: 12 });
    const filed = payloads
      .filter((p) => p.kind === 'contact_filed')
      .map((p) => (p.kind === 'contact_filed' ? p.commanderId : null));

    expect(filed).toContain('ney');

    // And it is their own eyes, so it continues rather than being reminted next time.
    const after = fold(state, payloads);
    const contacts = [...(after.knowledge.get('ney')?.contacts.values() ?? [])];
    expect(contacts).toHaveLength(1);
    expect(contacts[0]!.inSight).toBe(true);
  });
});

describe('two columns wanting the same ground', () => {
  const ney = commander('ney', 'red', 'red-1');

  /** The decisions a run raised, by trigger. */
  const raised = (payloads: readonly EventPayload[], trigger: string) =>
    payloads.filter((p) => p.kind === 'decision_raised' && p.decision.trigger === trigger);

  it('stops a head that runs into a column standing in its way', () => {
    // Blue is parked square across the line of march and is going nowhere. Red closes up
    // to the hex before it and stands: a column cannot walk through one that is there.
    const red = unit('red-1', 'red', { q: 5, r: 5 });
    const blue = unit('blue-1', 'blue', { q: 9, r: 5 });
    const state = stateFrom({
      units: [red, blue],
      commanders: [ney],
      tasks: [marchTo(red, { q: 12, r: 5 }, 6, { q: 6, r: 5 })],
    });

    const { payloads } = advance(state, world, cfg, clean, { hours: 12 });
    const after = fold(state, payloads);

    expect(after.units.get('red-1')!.column[0]).toEqual({ q: 8, r: 5 });
    expect(kinds(payloads)).not.toContain('task_completed');
    expect(raised(payloads, 'column_blocked')).toHaveLength(1);
  });

  it('asks only once while the road stays blocked', () => {
    // The column tries again every tick, and a referee who had to clear one decision per
    // quarter hour of traffic would stop running the clock at all.
    const red = unit('red-1', 'red', { q: 5, r: 5 });
    const blue = unit('blue-1', 'blue', { q: 9, r: 5 });
    const state = stateFrom({
      units: [red, blue],
      commanders: [ney],
      tasks: [marchTo(red, { q: 12, r: 5 }, 6, { q: 6, r: 5 })],
    });

    const { payloads } = advance(state, world, cfg, clean, { hours: 12 });
    expect(raised(payloads, 'column_blocked')).toHaveLength(1);
    expect(payloads.filter((p) => p.kind === 'march_blocked').length).toBeGreaterThan(1);
  });

  it('marches on again once the road clears', () => {
    // Blue steps aside of its own accord. Nothing re-orders red: its task never ended, so
    // the moment the ground is free it carries on.
    const red = unit('red-1', 'red', { q: 5, r: 5 });
    const blue = unit('blue-1', 'blue', { q: 9, r: 5 });
    const state = stateFrom({
      units: [red, blue],
      commanders: [ney],
      tasks: [
        marchTo(red, { q: 12, r: 5 }, 6, { q: 6, r: 5 }),
        marchTo(blue, { q: 9, r: 12 }, 6, { q: 9, r: 6 }),
      ],
    });

    const after = fold(state, advance(state, world, cfg, clean, { hours: 24 }).payloads);
    expect(after.units.get('red-1')!.column[0]).toEqual({ q: 12, r: 5 });
  });

  it('gives a contested hex to the faster column', () => {
    // Both heads are one hex from { q: 9, r: 5 } and both are due within the tick, but
    // blue is cavalry: a fifth of an hour a hex against red's half. Blue takes it.
    const red = unit('red-1', 'red', { q: 8, r: 5 });
    const blue = unit('blue-1', 'blue', { q: 10, r: 5 }, {}, 'cavalry');
    const contested = { q: 9, r: 5 };
    const state = stateFrom({
      units: [red, blue],
      tasks: [
        marchTo(red, { q: 12, r: 5 }, 6, contested),
        marchTo(blue, { q: 4, r: 5 }, 6, contested),
      ],
    });

    const { payloads } = advance(state, world, cfg, clean, { hours: 1 });
    const after = fold(state, payloads);

    expect(after.units.get('blue-1')!.column[0]).toEqual(contested);
    expect(after.units.get('red-1')!.column[0]).toEqual({ q: 8, r: 5 });
    // No tie, so nothing for the referee to break — the rules settled it.
    expect(raised(payloads, 'column_contested')).toHaveLength(0);
  });

  it('gives it to the faster column whichever of the two is looked at first', () => {
    // The same meeting as above with the identifiers the other way round, so the slower
    // column is the one the scheduler considers first. Who wins is a fact about the
    // ground and the pace, not about how the two formations happen to sort.
    const slow = unit('a-1', 'red', { q: 8, r: 5 });
    const fast = unit('z-1', 'blue', { q: 10, r: 5 }, {}, 'cavalry');
    const contested = { q: 9, r: 5 };
    const state = stateFrom({
      units: [slow, fast],
      tasks: [
        marchTo(slow, { q: 12, r: 5 }, 6, contested),
        marchTo(fast, { q: 4, r: 5 }, 6, contested),
      ],
    });

    const { payloads } = advance(state, world, cfg, clean, { hours: 1 });
    const after = fold(state, payloads);

    expect(after.units.get('z-1')!.column[0]).toEqual(contested);
    expect(after.units.get('a-1')!.column[0]).toEqual({ q: 8, r: 5 });
    // Still no tie. A contest raised here would hold the hex against the winner too.
    expect(raised(payloads, 'column_contested')).toHaveLength(0);
  });

  it('hands an even contest to the referee, and stops the clock', () => {
    // Two infantry divisions on identical ground: the same half hour into the same hex.
    // This is the common case rather than the rare one.
    const red = unit('red-1', 'red', { q: 8, r: 5 });
    const blue = unit('blue-1', 'blue', { q: 10, r: 5 });
    const contested = { q: 9, r: 5 };
    const state = stateFrom({
      units: [red, blue],
      tasks: [
        marchTo(red, { q: 12, r: 5 }, 6, contested),
        marchTo(blue, { q: 4, r: 5 }, 6, contested),
      ],
    });

    const { payloads, halted } = advance(state, world, cfg, clean, {
      hours: 6,
      untilDecision: true,
    });
    const after = fold(state, payloads);

    expect(halted?.trigger).toBe('column_contested');
    // Neither took it. A tie is not "first sorted wins".
    expect(after.units.get('red-1')!.column[0]).toEqual({ q: 8, r: 5 });
    expect(after.units.get('blue-1')!.column[0]).toEqual({ q: 10, r: 5 });
  });

  it('names the other column, so the referee has two sides to choose between', () => {
    const red = unit('red-1', 'red', { q: 8, r: 5 });
    const blue = unit('blue-1', 'blue', { q: 10, r: 5 });
    const contested = { q: 9, r: 5 };
    const state = stateFrom({
      units: [red, blue],
      tasks: [
        marchTo(red, { q: 12, r: 5 }, 6, contested),
        marchTo(blue, { q: 4, r: 5 }, 6, contested),
      ],
    });

    const { payloads } = advance(state, world, cfg, clean, { hours: 1 });
    const decision = raised(payloads, 'column_contested')[0];
    expect(decision).toBeDefined();
    if (decision?.kind !== 'decision_raised') throw new Error('unreachable');

    expect(contestants(decision.decision)).toHaveLength(1);
    expect([decision.decision.unitId, ...contestants(decision.decision)].sort()).toEqual([
      'blue-1',
      'red-1',
    ]);
    expect(contestedHex(decision.decision)).toEqual(contested);
  });

  it('raises traffic for the referee even where no commander rides with either column', () => {
    // Nobody is appointed to either formation. Two columns meeting on a road is a fact
    // about the ground, not about what anyone believes, and it still needs settling.
    const red = unit('red-1', 'red', { q: 8, r: 5 });
    const blue = unit('blue-1', 'blue', { q: 10, r: 5 });
    const contested = { q: 9, r: 5 };
    const state = stateFrom({
      units: [red, blue],
      tasks: [
        marchTo(red, { q: 12, r: 5 }, 6, contested),
        marchTo(blue, { q: 4, r: 5 }, 6, contested),
      ],
    });

    const { payloads } = advance(state, world, cfg, clean, { hours: 1 });
    const decision = raised(payloads, 'column_contested')[0];
    if (decision?.kind !== 'decision_raised') throw new Error('unreachable');
    expect(decision.decision.commanderId).toBeNull();
  });

  it('lets the column the referee named through, once they have ruled', () => {
    const red = unit('red-1', 'red', { q: 8, r: 5 });
    const blue = unit('blue-1', 'blue', { q: 10, r: 5 });
    const contested = { q: 9, r: 5 };
    const state = stateFrom({
      units: [red, blue],
      tasks: [
        marchTo(red, { q: 12, r: 5 }, 6, contested),
        marchTo(blue, { q: 4, r: 5 }, 6, contested),
      ],
    });

    const stuck = fold(state, advance(state, world, cfg, clean, { hours: 1 }).payloads);
    const decision = [...stuck.decisions.values()].find((d) => d.trigger === 'column_contested')!;
    expect(decision).toBeDefined();

    const ruled = fold(stuck, [
      {
        kind: 'decision_resolved',
        decisionId: decision.id,
        atHours: stuck.clockHours,
        note: null,
        favouring: 'red-1',
      },
    ]);

    const after = fold(ruled, advance(ruled, world, cfg, clean, { hours: 2 }).payloads);
    expect(after.units.get('red-1')!.column[0]).toEqual(contested);
    expect(after.units.get('blue-1')!.column[0]).toEqual({ q: 10, r: 5 });
  });

  it('charges the wait against the day, once a column has broken camp', () => {
    // Red marches one hex and is then stopped by blue. Standing formed up in column of
    // march is work: the hours keep running, and it will reach the twenty-hour cap having
    // spent most of the day in a road it could not get out of.
    const red = unit('red-1', 'red', { q: 5, r: 5 });
    const blue = unit('blue-1', 'blue', { q: 7, r: 5 });
    const state = stateFrom({
      units: [red, blue],
      tasks: [marchTo(red, { q: 12, r: 5 }, 6, { q: 6, r: 5 })],
    });

    const after = fold(state, advance(state, world, cfg, clean, { hours: 10 }).payloads);
    const marched = after.units.get('red-1')!.hoursMarchedToday;

    // Half an hour of marching, then standing for the rest of the ten. Not pinned to the
    // hour, because what is being asserted is that waiting costs time at all.
    expect(marched).toBeGreaterThan(5);
    expect(marched).toBeLessThanOrEqual(10);
  });

  it('never charges a blocked column past the day it has', () => {
    // Twenty hours is the most any formation spends under arms, and a column that stood in
    // a lane all day would otherwise carry a number the fatigue table has no row for.
    const red = unit('red-1', 'red', { q: 5, r: 5 });
    const blue = unit('blue-1', 'blue', { q: 7, r: 5 });
    const state = stateFrom({
      units: [red, blue],
      tasks: [marchTo(red, { q: 12, r: 5 }, 0, { q: 6, r: 5 })],
      clockHours: 0,
    });

    // Stops short of midnight, so the accumulator is never reset by the day rolling.
    const after = fold(state, advance(state, world, cfg, clean, { hours: 23 }).payloads);
    expect(after.units.get('red-1')!.hoursMarchedToday).toBe(cfg.maxMarchHoursPerDay);
  });

  it('charges nothing to a column blocked before it took a step', () => {
    // Stopped on the first hex of the march: it never broke camp, and a day's fatigue for
    // standing in a field it was already standing in would be the wrong kind of wrong.
    const red = unit('red-1', 'red', { q: 5, r: 5 });
    const blue = unit('blue-1', 'blue', { q: 6, r: 5 });
    const state = stateFrom({
      units: [red, blue],
      tasks: [marchTo(red, { q: 12, r: 5 }, 6, { q: 6, r: 5 })],
    });

    const after = fold(state, advance(state, world, cfg, clean, { hours: 10 }).payloads);
    expect(after.units.get('red-1')!.hoursMarchedToday).toBe(0);
    expect(after.units.get('red-1')!.column[0]).toEqual({ q: 5, r: 5 });
  });

  it('still counts the wait after midnight only once the column moves again', () => {
    // The day rolls, the accumulator zeroes, and a column still stuck behind the same
    // obstruction starts the new day owing nothing until it actually steps off.
    const red = unit('red-1', 'red', { q: 5, r: 5 }, { hoursMarchedToday: 3 });
    const blue = unit('blue-1', 'blue', { q: 6, r: 5 });
    const state = stateFrom({
      units: [red, blue],
      tasks: [marchTo(red, { q: 12, r: 5 }, 22, { q: 6, r: 5 })],
      clockHours: 22,
    });

    const after = fold(state, advance(state, world, cfg, clean, { hours: 6 }).payloads);
    expect(kinds(advance(state, world, cfg, clean, { hours: 6 }).payloads)).toContain('day_rolled');
    expect(after.units.get('red-1')!.hoursMarchedToday).toBe(0);
  });
});

describe('what a day on the road costs', () => {
  const ney = commander('ney', 'red', 'red-1');

  /** March a unit for `hours` from a start hour, and hand back what it cost. */
  const marchFor = (hours: number, startHour: number, u = unit('red-1', 'red', { q: 5, r: 5 })) => {
    const state = stateFrom({
      units: [u],
      commanders: [ney],
      tasks: [marchTo(u, { q: 39, r: 5 }, startHour, { q: 6, r: 5 })],
      clockHours: startHour,
    });
    const { payloads } = advance(state, world, cfg, clean, { hours });
    const after = fold(state, payloads);
    return { after, payloads, unit: after.units.get('red-1')! };
  };

  it('costs nothing for the first four hours', () => {
    // The rules give infantry four free hours, and a game that charged for the first would
    // make every short march a decision.
    const { unit: u } = marchFor(4, 8);
    expect(u.hoursMarchedToday).toBeCloseTo(4, 6);
    expect(u.fatigue).toBe(0);
  });

  it('starts charging at the fifth hour', () => {
    const { unit: u } = marchFor(6, 8);
    expect(u.hoursMarchedToday).toBeGreaterThan(5);
    expect(u.fatigue).toBeGreaterThan(0);
  });

  it('charges the head, not the hexes', () => {
    // Two divisions marching the same hours over the same ground pay the same, whether
    // one covered twice the distance because it was on a road.
    const { unit: slow } = marchFor(9, 8);
    const { unit: fast } = marchFor(
      9,
      8,
      unit('red-1', 'red', { q: 5, r: 5 }, {}, 'infantry', ['fast']),
    );

    // The fast division covered half as much ground again, and paid exactly what the table
    // says for the hours it spent — not for the hexes. Nine hours of daylight, so there is
    // no night term in either.
    expect(fast.column[0]!.q).toBeGreaterThan(slow.column[0]!.q);
    expect(slow.fatigue).toBe(marchFatigueAt(cfg, slow, slow.hoursMarchedToday));
    expect(fast.fatigue).toBe(marchFatigueAt(cfg, fast, fast.hoursMarchedToday));
  });

  it('charges a night march more than the same hours by day', () => {
    // Eight hours from four in the morning is half in the dark; eight from eight is not.
    const byNight = marchFor(8, 2).unit;
    const byDay = marchFor(8, 8).unit;
    expect(byNight.fatigue).toBeGreaterThan(byDay.fatigue);
  });

  it('charges the rear for the dark it marched in after the head halted', () => {
    // The head halts in the last of the light and would pay nothing. The troops at the back
    // are still on the road when the sun goes down, and that is the whole point of a
    // column having a length.
    //
    // Eight hexes at two km/h from hour fourteen puts the head on its destination exactly
    // at sunset. Everything charged after that was charged to the tail.
    const red = unit('red-1', 'red', { q: 5, r: 5 });
    const state = stateFrom({
      units: [red],
      commanders: [ney],
      tasks: [marchTo(red, { q: 13, r: 5 }, 14, { q: 6, r: 5 })],
      clockHours: 14,
    });

    const { payloads } = advance(state, world, cfg, clean, { hours: 6 });
    const tail = payloads.filter(
      (p) => p.kind === 'fatigue_accrued' && p.fromNight > 0 && p.fromMarching === 0,
    );

    // Nothing the head did was in the dark: it arrived as the sun went down.
    const marchedInDark = payloads.filter(
      (p) => p.kind === 'fatigue_accrued' && p.fromNight > 0 && p.fromMarching > 0,
    );
    expect(marchedInDark).toHaveLength(0);
    expect(tail).toHaveLength(1);
    expect(fold(state, payloads).units.get('red-1')!.fatigue).toBeGreaterThan(0);
  });

  it('writes what the fatigue was for, so the log reads as an account', () => {
    const { payloads } = marchFor(10, 2);
    const accrued = payloads.filter((p) => p.kind === 'fatigue_accrued');
    expect(accrued.length).toBeGreaterThan(0);
    for (const p of accrued) {
      if (p.kind !== 'fatigue_accrued') continue;
      expect(p.fatigue).toBeCloseTo(p.fromMarching + p.fromNight, 9);
      expect(p.fatigue).toBeGreaterThan(0);
    }
  });

  it('never runs past a hundred, which is the whole unit', () => {
    // Fatigue is read as a percentage of the troops no longer fit to stand in the line, so a
    // number above a hundred is not a worse day, it is a broken scale.
    const wrecked = unit('red-1', 'red', { q: 5, r: 5 }, { fatigue: 99 });
    const { unit: u } = marchFor(20, 2, wrecked);
    expect(u.fatigue).toBeLessThanOrEqual(100);
  });
});

describe('making and breaking camp', () => {
  const ney = commander('ney', 'red', 'red-1');

  it('does not camp merely because it arrived', () => {
    // Arriving is a decision for the referee, not a reason to unpack. The column stands
    // where it was sent, in column of march, until somebody says otherwise.
    const red = unit('red-1', 'red', { q: 5, r: 5 });
    const state = stateFrom({
      units: [red],
      commanders: [ney],
      tasks: [marchTo(red, { q: 8, r: 5 }, 6, { q: 6, r: 5 })],
    });

    const { payloads } = advance(state, world, cfg, clean, { hours: 6 });
    const after = fold(state, payloads);

    expect(kinds(payloads)).toContain('task_completed');
    expect(after.units.get('red-1')!.formation).toBe('march');
    expect(kinds(payloads)).not.toContain('formation_change_began');
  });

  it('alerts the referee when it arrives, rather than acting on its own', () => {
    const red = unit('red-1', 'red', { q: 5, r: 5 });
    const state = stateFrom({
      units: [red],
      commanders: [ney],
      tasks: [marchTo(red, { q: 8, r: 5 }, 6, { q: 6, r: 5 })],
    });

    const raised = advance(state, world, cfg, clean, { hours: 6 }).payloads.filter(
      (p) => p.kind === 'decision_raised',
    );
    expect(raised).toHaveLength(1);
    expect(raised[0]).toMatchObject({ decision: { trigger: 'objective_reached' } });
  });

  it('camps of its own accord only when the day is spent', () => {
    const tired = unit('red-1', 'red', { q: 5, r: 5 }, { hoursMarchedToday: 19.5 });
    const state = stateFrom({
      units: [tired],
      commanders: [ney],
      tasks: [marchTo(tired, { q: 30, r: 5 }, 20, { q: 6, r: 5 })],
      clockHours: 20,
    });

    const { payloads } = advance(state, world, cfg, clean, { hours: 3 });
    const began = payloads.filter((p) => p.kind === 'formation_change_began');

    expect(began).toHaveLength(1);
    expect(began[0]).toMatchObject({ to: 'rest', reason: 'day_spent' });
  });

  it('takes two hours to build, and the unit is still in march formation until it is', () => {
    const tired = unit('red-1', 'red', { q: 5, r: 5 }, { hoursMarchedToday: 19.5 });
    const state = stateFrom({
      units: [tired],
      commanders: [ney],
      tasks: [marchTo(tired, { q: 30, r: 5 }, 20, { q: 6, r: 5 })],
      clockHours: 20,
    });

    const began = advance(state, world, cfg, clean, { hours: 1 }).payloads.find(
      (p) => p.kind === 'formation_change_began',
    );
    if (began?.kind !== 'formation_change_began') throw new Error('no camp was begun');
    expect(began.completesAtHours - began.atHours).toBe(cfg.formationChangeHours.march.rest);
    expect(began.completesAtHours - began.atHours).toBe(2);

    // Halfway through building it, it is still what it was.
    const half = fold(state, advance(state, world, cfg, clean, { hours: 1.5 }).payloads);
    expect(half.units.get('red-1')!.formation).toBe('march');
    expect(half.units.get('red-1')!.formationChange).toMatchObject({ to: 'rest' });
  });

  it('breaks camp before it steps off, when ordered to march again', () => {
    // A resting division does not simply walk away. Two hours to get out of the camp, and
    // only then does the first hex begin.
    const resting = unit('red-1', 'red', { q: 5, r: 5 }, { formation: 'rest' });
    const state = stateFrom({
      units: [resting],
      commanders: [ney],
      tasks: [marchTo(resting, { q: 10, r: 5 }, 6, { q: 6, r: 5 })],
    });

    const { payloads } = advance(state, world, cfg, clean, { hours: 6 });
    const began = payloads.find((p) => p.kind === 'formation_change_began');
    if (began?.kind !== 'formation_change_began') throw new Error('camp was never broken');

    expect(began).toMatchObject({ from: 'rest', to: 'march', reason: 'break_camp' });
    const firstMarch = payloads.find((p) => p.kind === 'unit_marched');
    if (firstMarch?.kind !== 'unit_marched') throw new Error('never marched');
    expect(firstMarch.atHours).toBeGreaterThanOrEqual(began.completesAtHours);
  });

  it('breaks a camp it was only halfway through building', () => {
    // An order arriving mid-camp does not get the tents back for free: the troops have to
    // undo what they have done, and the rules charge the change either way.
    const tired = unit('red-1', 'red', { q: 5, r: 5 }, { hoursMarchedToday: 19.5 });
    const state = stateFrom({
      units: [tired],
      commanders: [ney],
      tasks: [marchTo(tired, { q: 30, r: 5 }, 20, { q: 6, r: 5 })],
      clockHours: 20,
    });

    const { payloads } = advance(state, world, cfg, clean, { hours: 12 });
    const reasons = payloads
      .filter((p) => p.kind === 'formation_change_began')
      .map((p) => (p.kind === 'formation_change_began' ? p.reason : ''));

    expect(reasons[0]).toBe('day_spent');
    expect(reasons).toContain('break_camp');
  });
});

describe('telling the referee a formation needs them', () => {
  it('reports an arrival even where nobody rides with the column', () => {
    // The referee decides what a formation does next, so a column that has finished its
    // orders is their business whether or not a commander was appointed to it. Without this it
    // stands in a field, orders complete, with nothing in the queue to say so.
    const red = unit('red-1', 'red', { q: 5, r: 5 });
    const state = stateFrom({ units: [red], tasks: [marchTo(red, { q: 8, r: 5 }, 6, { q: 6, r: 5 })] });

    const { payloads } = advance(state, world, cfg, clean, { hours: 6 });
    const raised = payloads.filter((p) => p.kind === 'decision_raised');

    expect(raised).toHaveLength(1);
    expect(raised[0]).toMatchObject({ decision: { trigger: 'objective_reached', commanderId: null } });
  });

  it('reports a column that cannot go on, likewise', () => {
    const red = unit('red-1', 'red', { q: 5, r: 5 });
    const split = splitWorld(8);
    const state = stateFrom({ units: [red], tasks: [marchTo(red, { q: 12, r: 5 }, 6, { q: 6, r: 5 })] });

    const raised = advance(state, split, cfg, clean, { hours: 12 }).payloads.filter(
      (p) => p.kind === 'decision_raised',
    );
    expect(raised[0]).toMatchObject({ decision: { trigger: 'crossing_impassable' } });
  });
});

describe('the clock runs in whole hours', () => {
  const ney = commander('ney', 'red', 'red-1');

  it('stamps every event on the hour', () => {
    // The claim the whole model rests on. Nothing happens at twenty past: a referee
    // adjudicates by the hour and an event they cannot address is an event in the wrong
    // place.
    const red = unit('red-1', 'red', { q: 5, r: 5 });
    const blue = unit('blue-1', 'blue', { q: 20, r: 5 }, {}, 'cavalry');
    const state = stateFrom({
      units: [red, blue],
      commanders: [ney],
      tasks: [
        marchTo(red, { q: 30, r: 5 }, 6, { q: 6, r: 5 }),
        marchTo(blue, { q: 20, r: 20 }, 6, { q: 20, r: 6 }),
      ],
    });

    const { payloads, toHours } = advance(state, world, cfg, clean, { hours: 30 });
    const stamps = payloads.flatMap((p) =>
      'atHours' in p && typeof p.atHours === 'number'
        ? [p.atHours]
        : p.kind === 'clock_advanced' || p.kind === 'day_rolled'
          ? [p.toHours]
          : [],
    );

    expect(stamps.length).toBeGreaterThan(10);
    for (const h of stamps) expect(Number.isInteger(h), `${h}`).toBe(true);
    expect(Number.isInteger(toHours)).toBe(true);
  });

  it('gives a column an hour of movement and lets it spend the lot', () => {
    // Infantry off-road is two km/h, so two hexes an hour — not one hex per tick and not
    // one hex per hour. The speed table is in kilometres an hour and a hex is a kilometre.
    const red = unit('red-1', 'red', { q: 5, r: 5 });
    const state = stateFrom({
      units: [red],
      commanders: [ney],
      tasks: [marchTo(red, { q: 30, r: 5 }, 6, { q: 6, r: 5 })],
    });

    const after = fold(state, advance(state, world, cfg, clean, { hours: 3 }).payloads);
    expect(after.units.get('red-1')!.column[0]).toEqual({ q: 11, r: 5 });
    expect(after.units.get('red-1')!.hoursMarchedToday).toBe(3);
  });

  it('banks the part of a hex an hour could not finish', () => {
    // A convoy off-road makes two thirds of a kilometre an hour. Without banking it would
    // never move at all, and a whole row of the movement table would be a dead letter.
    const convoy = unit('red-1', 'red', { q: 5, r: 5 }, {}, 'convoy');
    const state = stateFrom({
      units: [convoy],
      tasks: [marchTo(convoy, { q: 10, r: 5 }, 6, { q: 6, r: 5 })],
    });

    const afterOne = fold(state, advance(state, world, cfg, clean, { hours: 1 }).payloads);
    expect(afterOne.units.get('red-1')!.column[0]).toEqual({ q: 5, r: 5 });
    expect(afterOne.tasks.get('red-1')!.progressHours).toBeCloseTo(1, 9);

    // Two hexes in three hours: the remainder is carried, not lost.
    const afterThree = fold(state, advance(state, world, cfg, clean, { hours: 3 }).payloads);
    expect(afterThree.units.get('red-1')!.column[0]).toEqual({ q: 7, r: 5 });
  });

  it('spends a river crossing over as many hours as it takes', () => {
    // A crossing costs an hour on top of the march, so it cannot fit in the hour that
    // reaches the bank. The column walks at the ford and finishes the next hour.
    const red = unit('red-1', 'red', { q: 5, r: 5 });
    const w = flatWorld();
    for (const r of [4, 5, 6]) {
      const hex = w.hexes.get(key({ q: 7, r }))!;
      w.hexes.set(key({ q: 7, r }), {
        ...hex,
        catchmentKm2: 10,
        tags: new Set(['river', 'ford']),
      });
    }

    const state = stateFrom({
      units: [red],
      tasks: [marchTo(red, { q: 9, r: 5 }, 6, { q: 6, r: 5 })],
    });

    const after = fold(state, advance(state, w, cfg, clean, { hours: 6 }).payloads);
    expect(after.units.get('red-1')!.column[0]).toEqual({ q: 9, r: 5 });
  });

  it('gives a referee who asks for part of an hour the whole hours only', () => {
    // There is no half hour for the extra to happen in, and rounding up would run the
    // clock past what they asked for.
    const state = stateFrom({ units: [unit('red-1', 'red', { q: 5, r: 5 })] });
    expect(advance(state, world, cfg, clean, { hours: 2.5 }).toHours).toBe(8);
    expect(advance(state, world, cfg, clean, { hours: 0.5 }).toHours).toBe(6);
  });
});

describe('a patrol running into something', () => {
  /** Twenty troopers, detached, with a parent that is elsewhere. */
  const patrol = (id: string, faction: string, at: Hex): Unit => ({
    ...unit(id, faction, at, {}, 'cavalry', ['scout']),
    paperStrength: 20,
    morale: 0,
    provisions: 0,
    maxProvisions: 0,
    equipment: 0,
    maxEquipment: 0,
    parentUnitId: `${faction}-1`,
  });

  const raised = (payloads: readonly EventPayload[], trigger: string) =>
    payloads.filter((p) => p.kind === 'decision_raised' && p.decision.trigger === trigger);

  it('is reported rather than treated as traffic', () => {
    // A division standing in the way is a traffic problem. Twenty troopers riding into one
    // is a die roll that may destroy them, and the rules hand that roll to the referee.
    const scout = patrol('red-p1', 'red', { q: 5, r: 5 });
    const blue = unit('blue-1', 'blue', { q: 7, r: 5 });
    const state = stateFrom({
      units: [scout, blue],
      tasks: [marchTo(scout, { q: 12, r: 5 }, 6, { q: 6, r: 5 })],
    });

    const { payloads } = advance(state, world, cfg, clean, { hours: 6 });
    expect(raised(payloads, 'patrol_contact')).toHaveLength(1);
    expect(raised(payloads, 'column_blocked')).toHaveLength(0);
  });

  it('reads the same way round when the division is the one moving', () => {
    // The same meeting. Reporting it only when the patrol happened to be the mover would
    // lose half of them, and which half would depend on the order the ids sort in.
    const scout = patrol('blue-p1', 'blue', { q: 7, r: 5 });
    const red = unit('red-1', 'red', { q: 5, r: 5 });
    const state = stateFrom({
      units: [red, scout],
      tasks: [marchTo(red, { q: 12, r: 5 }, 6, { q: 6, r: 5 })],
    });

    const { payloads } = advance(state, world, cfg, clean, { hours: 6 });
    expect(raised(payloads, 'patrol_contact')).toHaveLength(1);
    expect(raised(payloads, 'column_blocked')).toHaveLength(0);
  });

  it('stops the clock, because the roll may destroy it', () => {
    const scout = patrol('red-p1', 'red', { q: 5, r: 5 });
    const blue = unit('blue-1', 'blue', { q: 7, r: 5 });
    const state = stateFrom({
      units: [scout, blue],
      tasks: [marchTo(scout, { q: 12, r: 5 }, 6, { q: 6, r: 5 })],
    });

    const { halted } = advance(state, world, cfg, clean, { hours: 12, untilDecision: true });
    expect(halted?.trigger).toBe('patrol_contact');
  });

  it('hands the referee the pool the rules call for', () => {
    // One die to start, and the modifiers for what was met: a division, cavalry, and a
    // scouting formation hardest of all.
    const scout = patrol('red-p1', 'red', { q: 5, r: 5 });
    const horse = unit('blue-1', 'blue', { q: 7, r: 5 }, {}, 'cavalry', ['scout']);
    const state = stateFrom({
      units: [scout, horse],
      tasks: [marchTo(scout, { q: 12, r: 5 }, 6, { q: 6, r: 5 })],
    });

    const decision = raised(advance(state, world, cfg, clean, { hours: 6 }).payloads, 'patrol_contact')[0];
    if (decision?.kind !== 'decision_raised') throw new Error('nothing was reported');

    const c = decision.decision.context as { dice: number; hostile: boolean; patrolUnitId: string };
    expect(c.dice).toBe(
      cfg.interceptDiceBase +
        cfg.interceptDiceDivision +
        cfg.interceptDiceCavalry +
        cfg.interceptDiceScout,
    );
    expect(c.hostile).toBe(true);
    expect(c.patrolUnitId).toBe('red-p1');
  });

  it('rides straight through its own side', () => {
    // Its own traffic is not an obstacle to twenty troopers: they ride down the column,
    // through the halt, and out the far end. Nothing is reported, because nothing
    // happened that a referee has to judge.
    const scout = patrol('red-p1', 'red', { q: 5, r: 5 });
    const friend = unit('red-2', 'red', { q: 7, r: 5 });
    const state = stateFrom({
      units: [scout, friend],
      tasks: [marchTo(scout, { q: 12, r: 5 }, 6, { q: 6, r: 5 })],
    });

    const { payloads } = advance(state, world, cfg, clean, { hours: 12 });
    const after = fold(state, payloads);

    expect(raised(payloads, 'patrol_contact')).toHaveLength(0);
    expect(raised(payloads, 'column_blocked')).toHaveLength(0);
    expect(kinds(payloads)).not.toContain('march_blocked');
    expect(after.units.get('red-p1')!.column[0]).toEqual({ q: 12, r: 5 });
  });

  it('is not an obstacle to its own side either', () => {
    // Symmetric. A division is not stopped by its own vedettes — they get out of the road.
    const scout = patrol('blue-p1', 'blue', { q: 7, r: 5 });
    const friend = unit('blue-2', 'blue', { q: 5, r: 5 });
    const state = stateFrom({
      units: [friend, scout],
      tasks: [marchTo(friend, { q: 12, r: 5 }, 6, { q: 6, r: 5 })],
    });

    const { payloads } = advance(state, world, cfg, clean, { hours: 12 });
    expect(kinds(payloads)).not.toContain('march_blocked');
    expect(fold(state, payloads).units.get('blue-2')!.column[0]).toEqual({ q: 12, r: 5 });
  });

  it('still stops dead for an enemy one', () => {
    // The exemption is for its own side only. An enemy column is the whole reason the
    // patrol is out there.
    const scout = patrol('red-p1', 'red', { q: 5, r: 5 });
    const friend = unit('red-2', 'red', { q: 7, r: 5 });
    const enemy = unit('blue-1', 'blue', { q: 9, r: 5 });
    const state = stateFrom({
      units: [scout, friend, enemy],
      tasks: [marchTo(scout, { q: 12, r: 5 }, 6, { q: 6, r: 5 })],
    });

    const { payloads } = advance(state, world, cfg, clean, { hours: 12 });
    const after = fold(state, payloads);

    // Past its own division at 7, stopped short of the enemy at 9.
    expect(after.units.get('red-p1')!.column[0]).toEqual({ q: 8, r: 5 });
    const contact = raised(payloads, 'patrol_contact');
    expect(contact).toHaveLength(1);
    if (contact[0]?.kind !== 'decision_raised') throw new Error('unreachable');
    expect((contact[0].decision.context as { hostile: boolean }).hostile).toBe(true);
  });

  it('is not held by a contest it is no part of', () => {
    // Two divisions arguing over a crossroads is not a reason twenty troopers cannot ride
    // over it. Holding them there would be the traffic rules reaching a formation they
    // were never written for.
    const scout = patrol('red-p1', 'red', { q: 5, r: 5 });
    const a = unit('red-2', 'red', { q: 8, r: 4 });
    const b = unit('red-3', 'red', { q: 8, r: 6 });
    const contested = { q: 8, r: 5 };
    const state = stateFrom({
      units: [scout, a, b],
      tasks: [
        marchTo(scout, { q: 12, r: 5 }, 6, { q: 6, r: 5 }),
        marchTo(a, { q: 8, r: 8 }, 6, contested),
        marchTo(b, { q: 8, r: 2 }, 6, contested),
      ],
    });

    const { payloads } = advance(state, world, cfg, clean, { hours: 12 });
    const after = fold(state, payloads);

    // The two divisions are still arguing about it. The patrol went through and onward.
    expect(raised(payloads, 'column_contested').length).toBeGreaterThan(0);
    expect(after.units.get('red-p1')!.column[0]).toEqual({ q: 12, r: 5 });
  });

  it('does not enter the hex it ran into', () => {
    const scout = patrol('red-p1', 'red', { q: 5, r: 5 });
    const blue = unit('blue-1', 'blue', { q: 7, r: 5 });
    const state = stateFrom({
      units: [scout, blue],
      tasks: [marchTo(scout, { q: 12, r: 5 }, 6, { q: 6, r: 5 })],
    });

    const after = fold(state, advance(state, world, cfg, clean, { hours: 6 }).payloads);
    expect(after.units.get('red-p1')!.column[0]).toEqual({ q: 6, r: 5 });
  });

  it('asks once, not once an hour, while the two stand facing each other', () => {
    const scout = patrol('red-p1', 'red', { q: 5, r: 5 });
    const blue = unit('blue-1', 'blue', { q: 7, r: 5 });
    const state = stateFrom({
      units: [scout, blue],
      tasks: [marchTo(scout, { q: 12, r: 5 }, 6, { q: 6, r: 5 })],
    });

    expect(raised(advance(state, world, cfg, clean, { hours: 18 }).payloads, 'patrol_contact')).toHaveLength(1);
  });
});

/**
 * Ground being fought over.
 *
 * The campaign layer does not resolve battles. What it has to get right is that the rules
 * written for open country stop applying on ground where formations are intermingled —
 * otherwise brigades in the same fight bounce off each other as though the field were a
 * crossroads.
 */
describe('a battlefield', () => {
  it('is not traffic', () => {
    const holding = unit('blue-1', 'blue', { q: 7, r: 5 });
    const coming = unit('red-1', 'red', { q: 5, r: 5 });
    const state = stateFrom({
      units: [holding, coming],
      tasks: [marchTo(coming, { q: 12, r: 5 }, 6, { q: 6, r: 5 })],
      battle: [{ q: 7, r: 5 }],
    });

    const { payloads } = advance(state, world, cfg, clean, { hours: 12 });
    expect(kinds(payloads)).not.toContain('march_blocked');
    expect(fold(state, payloads).units.get('red-1')!.column[0]).toEqual({ q: 12, r: 5 });
  });

  it('is traffic again the moment the fighting is over', () => {
    // The same ground, the same two formations, and nothing declared. This is the control:
    // without it the test above would pass just as well if marching were broken.
    const holding = unit('blue-1', 'blue', { q: 7, r: 5 });
    const coming = unit('red-1', 'red', { q: 5, r: 5 });
    const state = stateFrom({
      units: [holding, coming],
      tasks: [marchTo(coming, { q: 12, r: 5 }, 6, { q: 6, r: 5 })],
    });

    const { payloads } = advance(state, world, cfg, clean, { hours: 12 });
    expect(kinds(payloads)).toContain('march_blocked');
  });

  it('exempts only the ground declared, not the road up to it', () => {
    // A column walking into a formation short of the fighting is stopped as usual. The
    // exemption is about where troops are intermingled, not about being near a battle.
    const holding = unit('blue-1', 'blue', { q: 7, r: 5 });
    const coming = unit('red-1', 'red', { q: 5, r: 5 });
    const state = stateFrom({
      units: [holding, coming],
      tasks: [marchTo(coming, { q: 12, r: 5 }, 6, { q: 6, r: 5 })],
      battle: [{ q: 10, r: 5 }],
    });

    const { payloads } = advance(state, world, cfg, clean, { hours: 12 });
    expect(kinds(payloads)).toContain('march_blocked');
  });
});
