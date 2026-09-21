/**
 * The engine: commands in, events out, and nothing written when a command is refused.
 *
 * The invariants here are the ones the whole design rests on — that `check` cannot
 * mutate, that a refusal leaves no trace, that an override leaves an audit trail, and
 * that a log folds to the same state every time.
 */

import { describe, expect, it } from 'vitest';

import world32 from './fixtures/world-32x32.json' with { type: 'json' };

import {
  apply,
  applyAll,
  applyOrThrow,
  byCommander,
  check,
  decide,
  EMPTY_STATE,
  REFEREE,
  type Command,
} from '../src/engine.js';
import type { Faction, LoggedEvent, WorldRef } from '../src/events.js';
import { key, type Hex } from '../src/hex.js';
import { makeRng, rngFor } from '../src/rng.js';
import { CODES, RuleViolation } from '../src/ruling.js';
import {
  engaged,
  hasSurveyed,
  isEngaged,
  parentOf,
  patrolsOf,
  reduce,
  replay,
  type CampaignState,
} from '../src/state.js';
import {
  isBroken,
  isDivision,
  isPatrol,
  isStarving,
  KIND_DEFAULTS,
  PATROL_PAPER_STRENGTH,
  presentUnderArms,
  type Unit,
} from '../src/unit.js';
import type { Commander } from '../src/commander.js';
import { parseWorld, type World } from '../src/world.js';

const world: World = parseWorld(world32);

const worldRef: WorldRef = {
  seed: world.seed,
  width: world.width,
  height: world.height,
  layout: world.layout,
  schemaVersion: world.schemaVersion,
  hash: 'sha256:test',
};

const RED: Faction = { id: 'red', name: 'Red', color: '#c02020' };
const BLUE: Faction = { id: 'blue', name: 'Blue', color: '#2040c0' };

/** A land hex, so units are placed somewhere they could legally stand. */
const landHex = (): Hex => {
  for (const h of world.hexes.values()) {
    if (h.terrainClass === 'land') return h.coord;
  }
  throw new Error('the fixture world has no land');
};

const LAND = landHex();

const NEY: Commander = {
  id: 'ney',
  name: 'Marshal Ney',
  faction: 'red',
  unitId: 'red-1',
  superiorId: null,
  autoCascade: true,
};

const WELLINGTON: Commander = {
  id: 'wellington',
  name: 'The Duke of Wellington',
  faction: 'blue',
  unitId: 'blue-1',
  superiorId: null,
  autoCascade: true,
};

function division(id: string, faction: string, at: Hex = LAND, paperStrength = 5000): Unit {
  return {
    id,
    name: `${id} Division`,
    faction,
    kind: 'infantry',
    paperStrength,
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
    formationChange: null,
    column: [at],
    hoursMarchedToday: 0,
    corps: null,
    parentUnitId: null,
  };
}

const CREATE: Command = {
  kind: 'create_campaign',
  name: 'Test',
  world: worldRef,
  seed: 1234,
};

/** A campaign with two factions and one red division. */
function setUp(): CampaignState {
  const out = applyAll(
    [
      CREATE,
      { kind: 'add_faction', faction: RED },
      { kind: 'add_faction', faction: BLUE },
      { kind: 'add_unit', unit: division('red-1', 'red') },
      { kind: 'add_unit', unit: division('blue-1', 'blue') },
      { kind: 'add_commander', commander: NEY },
      { kind: 'add_commander', commander: WELLINGTON },
    ],
    EMPTY_STATE,
    world,
    'strict',
  );
  expect(out.ok, out.violations.map((v) => v.message).join('; ')).toBe(true);
  return out.state;
}

describe('check', () => {
  it('mutates nothing', () => {
    // The requirement that makes variable enforcement safe: validation cannot have side
    // effects, so a refused command cannot have left anything behind. Deep-freezing both
    // arguments turns any attempt into a TypeError.
    const state = setUp();
    const frozen = deepFreeze(structuredCloneState(state));
    const frozenWorld = Object.freeze(world);

    const commands: Command[] = [
      { kind: 'add_unit', unit: division('red-2', 'red') },
      { kind: 'remove_unit', unitId: 'red-1' },
      { kind: 'advance_clock', hours: 6 },
      { kind: 'teleport_unit', unitId: 'red-1', column: [LAND] },
      { kind: 'reveal', commanderId: 'ney', coords: [LAND] },
      { kind: 'set_unit_stats', unitId: 'red-1', changes: { morale: 10 } },
      // The commands that route, ride and run the clock. These do the most work of any
      // checker — planning a ride crosses the whole map — so they are the likeliest to
      // reach for something they should not.
      {
        kind: 'send_despatch',
        from: 'ney',
        to: 'ney',
        despatchKind: 'order',
        body: { text: 'Hold the crossroads.' },
      },
      { kind: 'set_task', unitId: 'red-1', destination: LAND },
      { kind: 'clear_task', unitId: 'red-1' },
      { kind: 'resolve_decision', decisionId: 'k1' },
    ];
    for (const cmd of commands) {
      expect(() => check(cmd, frozen, frozenWorld), cmd.kind).not.toThrow();
    }
  });

  it('the purity test above would actually catch a mutation', () => {
    // Guarding the guard. `Object.freeze` on its own leaves `Map.set` working, so a
    // naive deep-freeze would let `state.units.set(...)` through and the test above
    // would pass while proving nothing.
    const frozen = deepFreeze(structuredCloneState(setUp()));

    expect(() => {
      (frozen.units as Map<string, Unit>).set('x', division('x', 'red'));
    }).toThrow(TypeError);

    expect(() => {
      (frozen as { clockHours: number }).clockHours = 99;
    }).toThrow(TypeError);
  });

  it('rejects a unit that does not exist, hard', () => {
    const state = setUp();
    const v = check({ kind: 'remove_unit', unitId: 'ghost' }, state, world);
    expect(v).toHaveLength(1);
    expect(v[0]!.code).toBe(CODES.NO_SUCH_UNIT);
    expect(v[0]!.severity).toBe('hard');
  });

  it('rejects a hex off the map, hard', () => {
    const state = setUp();
    const v = check(
      { kind: 'teleport_unit', unitId: 'red-1', column: [{ q: 999, r: 999 }] },
      state,
      world,
    );
    expect(v.map((x) => x.code)).toContain(CODES.OFF_MAP);
    expect(v.every((x) => x.severity === 'hard')).toBe(true);
  });

  it('rejects a duplicate id, hard', () => {
    const state = setUp();
    const v = check({ kind: 'add_unit', unit: division('red-1', 'red') }, state, world);
    expect(v.map((x) => x.code)).toContain(CODES.DUPLICATE_ID);
  });

  it('rejects a unit in a faction that does not exist, hard', () => {
    const state = setUp();
    const v = check({ kind: 'add_unit', unit: division('green-1', 'green') }, state, world);
    expect(v.map((x) => x.code)).toContain(CODES.NO_SUCH_FACTION);
  });

  it('flags an under-strength division softly, so a referee may still place it', () => {
    const state = setUp();
    const v = check({ kind: 'add_unit', unit: division('red-2', 'red', LAND, 900) }, state, world);
    const small = v.find((x) => x.code === CODES.UNIT_TOO_SMALL);
    expect(small).toBeDefined();
    expect(small!.severity).toBe('soft');
  });

  it('refuses to run the clock backwards, hard', () => {
    // Rewinding is replaying a prefix of the log, not running the clock down; a negative
    // advance would leave events stamped after the hour they happened at.
    const state = setUp();
    const v = check({ kind: 'advance_clock', hours: -3 }, state, world);
    expect(v.map((x) => x.code)).toContain(CODES.TIME_REVERSED);
    expect(v[0]!.severity).toBe('hard');
  });

  it('refuses a world that is not the one the campaign names', () => {
    const v = check(
      { ...CREATE, world: { ...worldRef, seed: worldRef.seed + 1 } },
      EMPTY_STATE,
      world,
    );
    expect(v.map((x) => x.code)).toContain(CODES.WORLD_MISMATCH);
  });
});

/**
 * Tasks, despatches and the clock, as they arrive through the engine.
 *
 * The scheduler's own suite covers the mechanics. What matters here is that they come in
 * through `check`/`decide` like everything else, so a march is refused, audited and
 * logged by the same machinery as a teleport.
 */
describe('tasks and the clock', () => {
  /** A land hex some distance from the first, so a march has somewhere to go. */
  const somewhereElse = (): Hex => {
    const land = [...world.hexes.values()].filter((h) => h.terrainClass === 'land');
    const far = land.find(
      (h) => Math.abs(h.coord.q - LAND.q) + Math.abs(h.coord.r - LAND.r) > 6,
    );
    if (far === undefined) throw new Error('the fixture world is too small');
    return far.coord;
  };

  it('schedules the first leg when a task is set', () => {
    const state = setUp();
    const out = applyOrThrow(
      { kind: 'set_task', unitId: 'red-1', destination: somewhereElse() },
      state,
      world,
      'lenient',
    );

    const task = out.state.tasks.get('red-1')!;
    expect(task.destination).toEqual(somewhereElse());
    // A destination, not a path — but the engine has to know which hex is next, or the
    // clock has nothing to run. Nothing is walked yet: an order is not a head start.
    expect(task.nextHex).not.toBeNull();
    expect(task.progressHours).toBe(0);
    expect(task.complete).toBe(false);
  });

  it('completes a march to where the column already stands', () => {
    const out = applyOrThrow(
      { kind: 'set_task', unitId: 'red-1', destination: LAND },
      setUp(),
      world,
      'lenient',
    );
    expect(out.state.tasks.get('red-1')!.complete).toBe(true);
  });

  it('does not complete one that stands on its destination but has somewhere to go first', () => {
    const out = applyOrThrow(
      { kind: 'set_task', unitId: 'red-1', destination: LAND, via: [somewhereElse()] },
      setUp(),
      world,
      'lenient',
    );
    const task = out.state.tasks.get('red-1')!;
    expect(task.complete).toBe(false);
    expect(task.nextHex).not.toBeNull();
    expect(task.viaIndex).toBe(0);
  });

  it('keeps the waypoints a referee insisted on', () => {
    const out = applyOrThrow(
      { kind: 'set_task', unitId: 'red-1', destination: somewhereElse(), via: [LAND] },
      setUp(),
      world,
      'lenient',
    );
    expect(out.state.tasks.get('red-1')!.via).toEqual([LAND]);
  });

  it('names the waypoint it cannot reach, not the destination', () => {
    // The destination is reachable and the waypoint is not, which is exactly the case a
    // message about the destination would send a referee looking in the wrong place.
    const water = [...world.hexes.values()].find((h) => h.terrainClass === 'open_water');
    if (water === undefined) throw new Error('the fixture world has no water');

    const v = check(
      { kind: 'set_task', unitId: 'red-1', destination: somewhereElse(), via: [water.coord] },
      setUp(),
      world,
    );
    const route = v.find((x) => x.code === CODES.NO_MARCH_ROUTE);
    expect(route?.severity).toBe('soft');
    expect(route?.message).toContain(key(water.coord));
  });

  it('still issues a march through a waypoint it cannot reach, when the referee insists', () => {
    // Soft, so `lenient` lets it stand. The column discovers the problem where it stands,
    // which is the point of the whole design.
    const water = [...world.hexes.values()].find((h) => h.terrainClass === 'open_water');
    if (water === undefined) throw new Error('the fixture world has no water');

    const out = applyOrThrow(
      { kind: 'set_task', unitId: 'red-1', destination: somewhereElse(), via: [water.coord] },
      setUp(),
      world,
      'lenient',
    );
    expect(out.state.tasks.get('red-1')!.via).toEqual([water.coord]);
  });

  it('refuses a waypoint off the map outright', () => {
    const v = check(
      {
        kind: 'set_task',
        unitId: 'red-1',
        destination: somewhereElse(),
        via: [{ q: -50, r: -50 }],
      },
      setUp(),
      world,
    );
    expect(v.filter((x) => x.code === CODES.OFF_MAP).map((x) => x.severity)).toContain('hard');
  });

  it('advances the clock by marching, not by fiat', () => {
    const state = applyOrThrow(
      { kind: 'set_task', unitId: 'red-1', destination: somewhereElse() },
      setUp(),
      world,
      'lenient',
    ).state;

    const out = applyOrThrow({ kind: 'advance_clock', hours: 6 }, state, world, 'lenient');
    const kinds = out.events.map((e) => e.payload.kind);

    expect(kinds).toContain('unit_marched');
    // Every one of them is a logged event with an actor and a sequence number, which is
    // what makes a march reviewable after the fact rather than a number that changed.
    expect(out.events.every((e) => Number.isInteger(e.seq))).toBe(true);
    expect(out.state.units.get('red-1')!.column[0]).not.toEqual(LAND);
  });

  it('refuses to run the clock backwards, whatever the strictness', () => {
    for (const strictness of ['strict', 'lenient', 'open'] as const) {
      const out = apply({ kind: 'advance_clock', hours: -1 }, setUp(), world, strictness, {
        force: true,
      });
      expect(out.ok).toBe(false);
    }
  });

  it('refuses a decision nobody raised', () => {
    const v = check({ kind: 'resolve_decision', decisionId: 'k99' }, setUp(), world);
    expect(v[0]!.code).toBe(CODES.NO_SUCH_DECISION);
    expect(v[0]!.severity).toBe('hard');
  });

  /** A second red seat, so there is somebody for Ney to write to. */
  const withSubordinate = (): CampaignState =>
    applyAll(
      [
        { kind: 'add_unit', unit: division('red-2', 'red') },
        {
          kind: 'add_commander',
          commander: {
            id: 'kellermann',
            name: 'General Kellermann',
            faction: 'red',
            unitId: 'red-2',
            superiorId: 'ney',
            autoCascade: false,
          },
        },
      ],
      setUp(),
      world,
      'lenient',
    ).state;

  it('puts an order on the road as an event like any other', () => {
    const out = applyOrThrow(
      {
        kind: 'send_despatch',
        from: 'ney',
        to: 'kellermann',
        despatchKind: 'order',
        body: { text: 'Hold the crossroads.' },
      },
      withSubordinate(),
      world,
      'strict',
      { actor: byCommander('ney') },
    );

    const despatches = [...out.state.despatches.values()];
    expect(despatches).toHaveLength(1);
    expect(despatches[0]!.from).toBe('ney');
    // The two divisions were placed on the same hex, so this one is handed over rather
    // than ridden — which is the concentration rule, arriving through the engine.
    expect(despatches[0]!.handed).toBe(true);
    expect(out.events[0]!.actor).toEqual(byCommander('ney'));
  });

  it('refuses to write to the other side, at every strictness', () => {
    for (const strictness of ['strict', 'lenient', 'open'] as const) {
      const out = apply(
        {
          kind: 'send_despatch',
          from: 'ney',
          to: 'wellington',
          despatchKind: 'report',
          body: { text: 'I surrender.' },
        },
        withSubordinate(),
        world,
        strictness,
        { force: true },
      );
      expect(out.ok, strictness).toBe(false);
    }
  });

  it('refuses a despatch with nothing written on it', () => {
    const v = check(
      {
        kind: 'send_despatch',
        from: 'ney',
        to: 'kellermann',
        despatchKind: 'order',
        body: {},
      },
      withSubordinate(),
      world,
    );
    expect(v.some((x) => x.code === CODES.MALFORMED && x.severity === 'hard')).toBe(true);
  });
});

describe('apply', () => {
  it('writes nothing when a command is refused', () => {
    const state = setUp();
    const before = snapshot(state);

    const out = apply({ kind: 'remove_unit', unitId: 'ghost' }, state, world, 'strict');

    expect(out.ok).toBe(false);
    expect(out.events).toEqual([]);
    expect(out.state).toBe(state);
    expect(snapshot(out.state)).toEqual(before);
  });

  it('lets force through a soft violation, and records what it bent', () => {
    const state = setUp();
    const cmd: Command = { kind: 'add_unit', unit: division('red-2', 'red', LAND, 900) };

    expect(apply(cmd, state, world, 'strict').ok).toBe(false);

    const out = apply(cmd, state, world, 'strict', { force: true });
    expect(out.ok).toBe(true);
    expect(out.state.units.has('red-2')).toBe(true);

    const event = out.events[0]!;
    expect(event.forced).toBe(true);
    expect(event.bypassed.map((v) => v.code)).toEqual([CODES.UNIT_TOO_SMALL]);
  });

  it('lets a permissive strictness through, and still records the violation', () => {
    // Open is bookkeeping, not blindness: the rule was still broken and the log says so.
    const state = setUp();
    const cmd: Command = { kind: 'add_unit', unit: division('red-2', 'red', LAND, 900) };

    const out = apply(cmd, state, world, 'open');
    expect(out.ok).toBe(true);
    expect(out.events[0]!.forced).toBe(false);
    expect(out.events[0]!.bypassed.map((v) => v.code)).toEqual([CODES.UNIT_TOO_SMALL]);
  });

  it('refuses a hard violation even when forced and open', () => {
    const state = setUp();
    const out = apply(
      { kind: 'teleport_unit', unitId: 'red-1', column: [{ q: -50, r: -50 }] },
      state,
      world,
      'open',
      { force: true },
    );
    expect(out.ok).toBe(false);
    expect(out.events).toEqual([]);
  });

  it('overrides strictness for one command without loosening the campaign', () => {
    const state = setUp();
    const cmd: Command = { kind: 'add_unit', unit: division('red-2', 'red', LAND, 900) };

    const once = apply(cmd, state, world, 'strict', { strictness: 'open' });
    expect(once.ok).toBe(true);

    // The campaign's own setting is untouched — the next command is strict again.
    const after = apply(
      { kind: 'add_unit', unit: division('red-3', 'red', LAND, 900) },
      once.state,
      world,
      'strict',
    );
    expect(after.ok).toBe(false);
  });

  it('records nothing as bypassed when nothing was bent', () => {
    const state = setUp();
    const out = apply({ kind: 'advance_clock', hours: 4 }, state, world, 'strict');
    expect(out.ok).toBe(true);
    expect(out.events[0]!.bypassed).toEqual([]);
    expect(out.events[0]!.forced).toBe(false);
  });

  it('stamps events with the hour they happened, not the hour they produced', () => {
    const state = setUp();
    const out = apply({ kind: 'advance_clock', hours: 6 }, state, world, 'strict');
    expect(out.events[0]!.clockHours).toBe(0);
    expect(out.state.clockHours).toBe(6);
  });

  it('numbers events consecutively', () => {
    const state = setUp();
    let s = state;
    const seqs: number[] = [];
    for (let i = 0; i < 5; i++) {
      const out = apply({ kind: 'advance_clock', hours: 1 }, s, world, 'strict');
      seqs.push(out.events[0]!.seq);
      s = out.state;
    }
    expect(seqs).toEqual([seqs[0]!, seqs[0]! + 1, seqs[0]! + 2, seqs[0]! + 3, seqs[0]! + 4]);
  });

  it('names the actor', () => {
    const state = setUp();
    const asRed = apply({ kind: 'advance_clock', hours: 1 }, state, world, 'strict', {
      actor: byCommander('ney'),
    });
    expect(asRed.events[0]!.actor).toEqual({ kind: 'commander', id: 'ney' });

    const asRef = apply({ kind: 'advance_clock', hours: 1 }, state, world, 'strict');
    expect(asRef.events[0]!.actor).toEqual(REFEREE);
  });
});

describe('applyOrThrow', () => {
  it('throws a RuleViolation carrying the findings', () => {
    const state = setUp();
    try {
      applyOrThrow({ kind: 'remove_unit', unitId: 'ghost' }, state, world, 'strict');
      expect.unreachable('should have thrown');
    } catch (e) {
      expect(e).toBeInstanceOf(RuleViolation);
      expect((e as RuleViolation).unavoidable).toBe(true);
    }
  });
});

describe('applyAll', () => {
  it('stops at the first refusal and reports what got through', () => {
    const state = setUp();
    const out = applyAll(
      [
        { kind: 'advance_clock', hours: 1 },
        { kind: 'remove_unit', unitId: 'ghost' },
        { kind: 'advance_clock', hours: 1 },
      ],
      state,
      world,
      'strict',
    );
    expect(out.ok).toBe(false);
    expect(out.events).toHaveLength(1);
    expect(out.state.clockHours).toBe(1);
  });
});

describe('referee overrides', () => {
  it('teleports a unit with no movement rule applying', () => {
    const state = setUp();
    const far = [...world.hexes.values()].find(
      (h) => h.terrainClass === 'land' && key(h.coord) !== key(LAND),
    )!.coord;

    const out = apply({ kind: 'teleport_unit', unitId: 'red-1', column: [far] }, state, world, 'strict');

    expect(out.ok).toBe(true);
    // Not an override: nothing was bypassed, because no rule applies to a teleport.
    expect(out.events[0]!.bypassed).toEqual([]);
    expect(out.state.units.get('red-1')!.column[0]).toEqual(far);
  });

  it('reveals ground a faction has not earned', () => {
    const state = setUp();
    const coords = [...world.hexes.values()].slice(0, 5).map((h) => h.coord);

    const out = apply({ kind: 'reveal', commanderId: 'ney', coords }, state, world, 'strict');

    expect(out.ok).toBe(true);
    for (const c of coords) expect(hasSurveyed(out.state, 'ney', c)).toBe(true);
    for (const c of coords) expect(hasSurveyed(out.state, 'wellington', c)).toBe(false);
  });

  it('conceals ground, the one thing that shrinks what a faction knows', () => {
    const state = setUp();
    const coords = [...world.hexes.values()].slice(0, 5).map((h) => h.coord);

    const revealed = apply({ kind: 'reveal', commanderId: 'ney', coords }, state, world, 'strict').state;
    const concealed = apply(
      { kind: 'conceal', commanderId: 'ney', coords: coords.slice(0, 2) },
      revealed,
      world,
      'strict',
    ).state;

    expect(hasSurveyed(concealed, 'ney', coords[0]!)).toBe(false);
    expect(hasSurveyed(concealed, 'ney', coords[4]!)).toBe(true);
  });

  it('sets a stat without touching the others', () => {
    const state = setUp();
    const before = state.units.get('red-1')!;

    const out = apply(
      { kind: 'set_unit_stats', unitId: 'red-1', changes: { morale: 12 } },
      state,
      world,
      'strict',
    );

    const after = out.state.units.get('red-1')!;
    expect(after.morale).toBe(12);
    expect(after.paperStrength).toBe(before.paperStrength);
    expect(after.fatigue).toBe(before.fatigue);
    expect(after.corps).toBe(before.corps);
  });

  it('treats an absent field as "leave alone", not as "set to undefined"', () => {
    // `corps` is legitimately null, so spreading undefined over it would be a silent
    // difference between "not mentioned" and "cleared".
    const state = setUp();
    const withCorps = apply(
      { kind: 'set_unit_stats', unitId: 'red-1', changes: { corps: 'I Corps' } },
      state,
      world,
      'strict',
    ).state;

    const out = apply(
      { kind: 'set_unit_stats', unitId: 'red-1', changes: { morale: 5 } },
      withCorps,
      world,
      'strict',
    );

    expect(out.state.units.get('red-1')!.corps).toBe('I Corps');
  });
});

describe('replay', () => {
  it('folds a log to the same state the engine produced', () => {
    const events: LoggedEvent[] = [];
    let s = EMPTY_STATE;
    const commands: Command[] = [
      CREATE,
      { kind: 'add_faction', faction: RED },
      { kind: 'add_faction', faction: BLUE },
      { kind: 'add_unit', unit: division('red-1', 'red') },
      { kind: 'add_unit', unit: division('blue-1', 'blue') },
      { kind: 'add_commander', commander: NEY },
      { kind: 'advance_clock', hours: 6 },
      { kind: 'reveal', commanderId: 'ney', coords: [LAND] },
      { kind: 'set_unit_stats', unitId: 'red-1', changes: { fatigue: 12 } },
      { kind: 'remove_unit', unitId: 'blue-1' },
    ];
    for (const cmd of commands) {
      const out = apply(cmd, s, world, 'strict');
      expect(out.ok, cmd.kind).toBe(true);
      events.push(...out.events);
      s = out.state;
    }

    expect(snapshot(replay(events))).toEqual(snapshot(s));
  });

  it('folds the same log to the same state every time', () => {
    const events: LoggedEvent[] = [];
    let s = EMPTY_STATE;
    for (const cmd of [
      CREATE,
      { kind: 'add_faction', faction: RED } as Command,
      { kind: 'add_unit', unit: division('red-1', 'red') } as Command,
      { kind: 'advance_clock', hours: 3 } as Command,
    ]) {
      const out = apply(cmd, s, world, 'strict');
      events.push(...out.events);
      s = out.state;
    }
    expect(snapshot(replay(events))).toEqual(snapshot(replay(events)));
  });

  it('rewinds by folding a prefix', () => {
    // How the referee's rewind works: there is no undo, only a shorter log.
    const events: LoggedEvent[] = [];
    let s = EMPTY_STATE;
    for (const cmd of [
      CREATE,
      { kind: 'add_faction', faction: RED } as Command,
      { kind: 'add_unit', unit: division('red-1', 'red') } as Command,
      { kind: 'advance_clock', hours: 3 } as Command,
      { kind: 'advance_clock', hours: 3 } as Command,
    ]) {
      const out = apply(cmd, s, world, 'strict');
      events.push(...out.events);
      s = out.state;
    }

    expect(s.clockHours).toBe(6);
    expect(replay(events.slice(0, -1)).clockHours).toBe(3);
    expect(replay(events.slice(0, -2)).clockHours).toBe(0);
  });

  it('ignores an event naming something that has since gone', () => {
    // A log must always fold. A campaign that cannot be opened is far worse than an
    // event that turns out to be a no-op.
    const state = setUp();
    const orphan: LoggedEvent = {
      seq: 99,
      clockHours: 0,
      actor: REFEREE,
      payload: { kind: 'unit_teleported', unitId: 'ghost', column: [LAND] },
      forced: false,
      strictness: 'strict',
      bypassed: [],
    };
    expect(() => replay([orphan], state)).not.toThrow();
  });
});

describe('decide', () => {
  it('is a pure function of state and command', () => {
    const state = setUp();
    const cmd: Command = { kind: 'advance_clock', hours: 5 };
    const a = decide(cmd, state, world, makeRng(1));
    const b = decide(cmd, state, world, makeRng(1));
    expect(a).toEqual(b);
  });

  it('draws the same dice for the same campaign and sequence number', () => {
    // Deriving the generator from (seed, seq) rather than carrying it is what keeps
    // decide pure and makes replaying a prefix impossible to desynchronise.
    expect(rngFor(1234, 7).pool(4)).toEqual(rngFor(1234, 7).pool(4));
    expect(rngFor(1234, 7).pool(4)).not.toEqual(rngFor(1234, 8).pool(4));
  });
});

/** A comparable, order-independent view of the state. */
function snapshot(s: CampaignState) {
  return {
    name: s.name,
    seed: s.seed,
    clockHours: s.clockHours,
    factions: [...s.factions.keys()].sort(),
    units: [...s.units.values()]
      .map((u) => ({ ...u, traits: [...u.traits].sort() }))
      .sort((a, b) => (a.id < b.id ? -1 : 1)),
    commanders: [...s.commanders.values()].sort((a, b) => (a.id < b.id ? -1 : 1)),
    knowledge: [...s.knowledge.values()]
      .map((k) => ({ commanderId: k.commanderId, surveyed: [...k.surveyed].sort() }))
      .sort((a, b) => (a.commanderId < b.commanderId ? -1 : 1)),
  };
}

/** Structural copy that preserves Maps and Sets, which structuredClone would too. */
function structuredCloneState(s: CampaignState): CampaignState {
  return {
    ...s,
    factions: new Map(s.factions),
    commanders: new Map(s.commanders),
    units: new Map(s.units),
    knowledge: new Map(s.knowledge),
  };
}

/**
 * Freeze deeply, including collections.
 *
 * `Object.freeze` alone is not enough: it stops property assignment but leaves
 * `Map.set`, `Map.delete`, `Set.add` and friends working, so a frozen state would still
 * happily accept `state.units.set(...)`. The mutators are replaced with throwing stubs so
 * the purity test actually bites.
 */
function deepFreeze<T>(o: T): T {
  if (o === null || typeof o !== 'object') return o;
  // Reachable twice over: two commanders can share a superior, and the walk would then
  // try to redefine an already-stubbed mutator, which throws.
  if (Object.isFrozen(o)) return o;

  if (o instanceof Map || o instanceof Set) {
    for (const v of o.values()) deepFreeze(v);
    for (const m of ['set', 'delete', 'clear', 'add'] as const) {
      if (m in o) {
        Object.defineProperty(o, m, {
          value: () => {
            throw new TypeError(`${o.constructor.name}.${m} on a frozen collection`);
          },
        });
      }
    }
    return Object.freeze(o);
  }

  for (const v of Object.values(o as Record<string, unknown>)) deepFreeze(v);
  return Object.freeze(o);
}

describe('formation, as the referee orders it', () => {
  /** A land hex some distance from the first, so a march has somewhere to go. */
  const far = (): Hex => {
    const land = [...world.hexes.values()].filter((h) => h.terrainClass === 'land');
    const out = land.find((h) => Math.abs(h.coord.q - LAND.q) + Math.abs(h.coord.r - LAND.r) > 6);
    if (out === undefined) throw new Error('the fixture world is too small');
    return out.coord;
  };

  it('begins a change at the cost the rules give it', () => {
    const out = applyOrThrow(
      { kind: 'set_formation', unitId: 'red-1', formation: 'battle' },
      setUp(),
      world,
      'lenient',
    );

    const change = out.state.units.get('red-1')!.formationChange!;
    expect(change.to).toBe('battle');
    // Forming for battle from column of march is an hour, and the unit is still in march
    // formation until that hour is up.
    expect(change.completesAtHours - out.state.clockHours).toBe(1);
    expect(out.state.units.get('red-1')!.formation).toBe('march');
  });

  it('calls off a march when it orders anything but a march', () => {
    // A formation told to make camp is no longer going anywhere. Leaving the task standing
    // would have it break camp again the moment the camp was finished.
    const marching = applyOrThrow(
      { kind: 'set_task', unitId: 'red-1', destination: far() },
      setUp(),
      world,
      'lenient',
    ).state;
    expect(marching.tasks.get('red-1')!.complete).toBe(false);

    const out = applyOrThrow(
      { kind: 'set_formation', unitId: 'red-1', formation: 'rest' },
      marching,
      world,
      'lenient',
    );
    expect(out.events.map((e) => e.payload.kind)).toContain('task_cleared');
    expect(out.state.tasks.has('red-1')).toBe(false);
  });

  it('leaves a march standing when it orders march formation', () => {
    const resting = applyOrThrow(
      { kind: 'set_formation', unitId: 'red-1', formation: 'rest' },
      setUp(),
      world,
      'lenient',
    ).state;

    const marching = applyOrThrow(
      { kind: 'set_task', unitId: 'red-1', destination: far() },
      resting,
      world,
      'lenient',
    ).state;

    const out = applyOrThrow(
      { kind: 'set_formation', unitId: 'red-1', formation: 'march' },
      marching,
      world,
      'lenient',
    );
    expect(out.events.map((e) => e.payload.kind)).not.toContain('task_cleared');
  });

  it('writes nothing when the formation is already what was asked for', () => {
    const out = applyOrThrow(
      { kind: 'set_formation', unitId: 'red-1', formation: 'march' },
      setUp(),
      world,
      'lenient',
    );
    expect(out.events).toHaveLength(0);
  });

  it('writes nothing when the change already under way is the one asked for', () => {
    // A second click on the same button is the referee saying the same thing twice.
    // Restarting the change would charge the hours again from the present hour, so a
    // formation could be held mid-change indefinitely by repeating the order.
    const changing = applyOrThrow(
      { kind: 'set_formation', unitId: 'red-1', formation: 'occupation' },
      setUp(),
      world,
      'lenient',
    ).state;
    const completesAt = changing.units.get('red-1')!.formationChange!.completesAtHours;

    const again = applyOrThrow(
      { kind: 'set_formation', unitId: 'red-1', formation: 'occupation' },
      changing,
      world,
      'lenient',
    );
    expect(again.events).toHaveLength(0);
    expect(again.state.units.get('red-1')!.formationChange!.completesAtHours).toBe(completesAt);
  });

  it('refuses a formation for a unit that does not exist', () => {
    const v = check(
      { kind: 'set_formation', unitId: 'ghost', formation: 'rest' },
      setUp(),
      world,
    );
    expect(v.map((x) => x.code)).toContain(CODES.NO_SUCH_UNIT);
  });

  it('will not simply order a routing formation to form up, but can be made to', () => {
    // Rallying a broken division is an adjudication, not an order. Soft, so a referee who
    // has decided it rallied may say so.
    const broken = applyOrThrow(
      { kind: 'set_unit_stats', unitId: 'red-1', changes: { morale: 0 } },
      setUp(),
      world,
      'lenient',
    ).state;
    const routing = reduce(broken, {
      seq: broken.nextSeq,
      clockHours: broken.clockHours,
      actor: { kind: 'referee' },
      payload: { kind: 'formation_changed', unitId: 'red-1', to: 'rout', atHours: broken.clockHours },
      forced: false,
      strictness: 'strict',
      bypassed: [],
    });

    const v = check({ kind: 'set_formation', unitId: 'red-1', formation: 'battle' }, routing, world);
    const found = v.find((x) => x.code === CODES.NOT_IN_COMMAND);
    expect(found?.severity).toBe('soft');

    const out = applyOrThrow(
      { kind: 'set_formation', unitId: 'red-1', formation: 'battle' },
      routing,
      world,
      'lenient',
    );
    expect(out.ok).toBe(true);
  });
});

describe('patrols', () => {
  const scout = (id: string): Unit => ({
    ...division(id, 'red'),
    traits: ['scout'],
  });

  const withScout = (): CampaignState =>
    applyOrThrow(
      { kind: 'add_unit', unit: scout('red-2') },
      setUp(),
      world,
      'lenient',
    ).state;

  it('detaches twenty troopers, and remembers whose they are', () => {
    const out = applyOrThrow(
      { kind: 'detach_patrol', unitId: 'red-2' },
      withScout(),
      world,
      'lenient',
    );

    const patrol = [...out.state.units.values()].find((u) => u.parentUnitId === 'red-2');
    expect(patrol).toBeDefined();
    expect(patrol!.paperStrength).toBe(PATROL_PAPER_STRENGTH);
    expect(patrol!.faction).toBe('red');
    // Moves as cavalry and sees as a scout, which is what a patrol is for.
    expect(patrol!.kind).toBe('cavalry');
    expect(patrol!.traits).toContain('scout');
    expect(isPatrol(patrol!)).toBe(true);
  });

  it('starts where its parent stands, unless told otherwise', () => {
    const state = withScout();
    const head = state.units.get('red-2')!.column[0]!;

    const here = applyOrThrow(
      { kind: 'detach_patrol', unitId: 'red-2' },
      state,
      world,
      'lenient',
    ).state;
    expect(patrolsOf(here, 'red-2')[0]!.column[0]).toEqual(head);

    const elsewhere = [...world.hexes.values()].find(
      (h) => h.terrainClass === 'land' && key(h.coord) !== key(head),
    )!.coord;
    const there = applyOrThrow(
      { kind: 'detach_patrol', unitId: 'red-2', at: elsewhere },
      state,
      world,
      'lenient',
    ).state;
    expect(patrolsOf(there, 'red-2')[0]!.column[0]).toEqual(elsewhere);
  });

  it('costs nothing for the first three, which is the rules allowance', () => {
    let state = withScout();
    const before = state.units.get('red-2')!.paperStrength;

    for (let i = 0; i < 3; i++) {
      const out = applyOrThrow({ kind: 'detach_patrol', unitId: 'red-2' }, state, world, 'strict');
      expect(out.ok, `patrol ${i + 1}`).toBe(true);
      state = out.state;
    }

    expect(patrolsOf(state, 'red-2')).toHaveLength(3);
    expect(state.units.get('red-2')!.paperStrength).toBe(before);
  });

  it('takes a hundred troopers off the rolls for the fourth, permanently', () => {
    let state = withScout();
    for (let i = 0; i < 3; i++) {
      state = applyOrThrow({ kind: 'detach_patrol', unitId: 'red-2' }, state, world, 'lenient').state;
    }
    const before = state.units.get('red-2')!.paperStrength;

    // Soft, and it says what it will cost before it costs it.
    const v = check({ kind: 'detach_patrol', unitId: 'red-2' }, state, world);
    const warned = v.find((x) => x.code === CODES.UNIT_TOO_SMALL);
    expect(warned?.severity).toBe('soft');
    expect(warned?.message).toContain('100');

    const out = applyOrThrow({ kind: 'detach_patrol', unitId: 'red-2' }, state, world, 'lenient');
    expect(out.state.units.get('red-2')!.paperStrength).toBe(before - 100);
    expect(patrolsOf(out.state, 'red-2')).toHaveLength(4);
  });

  it('does not reuse the identifier of a patrol that is still out', () => {
    // Two out, the first lost. Numbering the next from the count of those still riding
    // would name it after the one that is — and quietly overwrite it.
    let state = withScout();
    for (let i = 0; i < 2; i++) {
      state = applyOrThrow({ kind: 'detach_patrol', unitId: 'red-2' }, state, world, 'lenient').state;
    }
    expect(patrolsOf(state, 'red-2').map((p) => p.id)).toEqual(['red-2-p1', 'red-2-p2']);

    state = applyOrThrow({ kind: 'remove_unit', unitId: 'red-2-p1' }, state, world, 'lenient').state;
    state = applyOrThrow({ kind: 'detach_patrol', unitId: 'red-2' }, state, world, 'lenient').state;

    const ids = patrolsOf(state, 'red-2').map((p) => p.id).sort();
    expect(ids).toEqual(['red-2-p2', 'red-2-p3']);
    // The survivor is still the survivor, not a fresh patrol wearing its name.
    expect(state.units.get('red-2-p2')!.name).toBe('red-2 Division patrol 2');
  });

  it('will not let a formation without Scout field one as of right', () => {
    // The rules give patrols to Scout. Soft, so a referee running a scenario where a line
    // division pushes out vedettes can say so.
    const v = check({ kind: 'detach_patrol', unitId: 'red-1' }, setUp(), world);
    const found = v.find((x) => x.code === CODES.NOT_IN_COMMAND);
    expect(found?.severity).toBe('soft');

    const out = applyOrThrow({ kind: 'detach_patrol', unitId: 'red-1' }, setUp(), world, 'lenient');
    expect(patrolsOf(out.state, 'red-1')).toHaveLength(1);
  });

  it('refuses a patrol from a formation that does not exist', () => {
    const v = check({ kind: 'detach_patrol', unitId: 'ghost' }, setUp(), world);
    expect(v.map((x) => x.code)).toContain(CODES.NO_SUCH_UNIT);
  });

  it('gives each patrol a distinct id, and refuses a duplicate', () => {
    let state = withScout();
    state = applyOrThrow(
      { kind: 'detach_patrol', unitId: 'red-2', patrolId: 'vedette' },
      state,
      world,
      'lenient',
    ).state;

    const v = check(
      { kind: 'detach_patrol', unitId: 'red-2', patrolId: 'vedette' },
      state,
      world,
    );
    expect(v.map((x) => x.code)).toContain(CODES.DUPLICATE_ID);

    const second = applyOrThrow({ kind: 'detach_patrol', unitId: 'red-2' }, state, world, 'lenient');
    const ids = patrolsOf(second.state, 'red-2').map((u) => u.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it('carries no morale, no supply and no fatigue', () => {
    // A patrol is not a formation in miniature. Twenty troopers do not hold a line, do not
    // run out of food on a two-day ride, and are not worn down by a table written for a
    // division marching in column.
    const out = applyOrThrow(
      { kind: 'detach_patrol', unitId: 'red-2' },
      withScout(),
      world,
      'lenient',
    );
    const patrol = patrolsOf(out.state, 'red-2')[0]!;

    expect(patrol.morale).toBe(0);
    expect(patrol.provisions).toBe(0);
    expect(patrol.equipment).toBe(0);
    expect(patrol.fatigue).toBe(0);
  });

  it('is not broken or starving for carrying none of it', () => {
    // The zeroes mean "not tracked", and every rule that reads them asks whose detachment
    // this is before it reads the number. Without that a patrol would be born broken and
    // starving on the hour it was sent out.
    const out = applyOrThrow(
      { kind: 'detach_patrol', unitId: 'red-2' },
      withScout(),
      world,
      'lenient',
    );
    const patrol = patrolsOf(out.state, 'red-2')[0]!;

    expect(isBroken(patrol)).toBe(false);
    expect(isStarving(patrol)).toBe(false);
    // And all twenty are present, because there is no fatigue to take any of them off.
    expect(presentUnderArms(patrol)).toBe(PATROL_PAPER_STRENGTH);
  });

  it('takes no fatigue from a day in the saddle', () => {
    const state = applyOrThrow(
      { kind: 'detach_patrol', unitId: 'red-2', patrolId: 'vedette' },
      withScout(),
      world,
      'lenient',
    ).state;

    const land = [...world.hexes.values()].filter((h) => h.terrainClass === 'land');
    const start = state.units.get('vedette')!.column[0]!;
    const goal = land.find(
      (h) => Math.abs(h.coord.q - start.q) + Math.abs(h.coord.r - start.r) > 6,
    )!.coord;

    const ordered = applyOrThrow(
      { kind: 'set_task', unitId: 'vedette', destination: goal },
      state,
      world,
      'lenient',
    ).state;
    const out = applyOrThrow({ kind: 'advance_clock', hours: 14 }, ordered, world, 'lenient');

    // It rode — the hours are counted, because the twenty-hour cap still applies to a
    // horse — but it was charged nothing for them.
    const rider = out.state.units.get('vedette')!;
    expect(rider.hoursMarchedToday).toBeGreaterThan(4);
    expect(rider.fatigue).toBe(0);
    expect(
      out.events.filter((e) => e.payload.kind === 'fatigue_accrued'),
    ).toHaveLength(0);
  });

  it('is not a division, so nothing treats twenty troopers as one', () => {
    const out = applyOrThrow(
      { kind: 'detach_patrol', unitId: 'red-2' },
      withScout(),
      world,
      'lenient',
    );
    expect(isDivision(patrolsOf(out.state, 'red-2')[0]!)).toBe(false);
  });

  it('marches like any other unit, because it is one', () => {
    const state = applyOrThrow(
      { kind: 'detach_patrol', unitId: 'red-2', patrolId: 'vedette' },
      withScout(),
      world,
      'lenient',
    ).state;

    const land = [...world.hexes.values()].filter((h) => h.terrainClass === 'land');
    const start = state.units.get('vedette')!.column[0]!;
    const goal = land.find(
      (h) => Math.abs(h.coord.q - start.q) + Math.abs(h.coord.r - start.r) > 4,
    )!.coord;

    const ordered = applyOrThrow(
      { kind: 'set_task', unitId: 'vedette', destination: goal },
      state,
      world,
      'lenient',
    ).state;
    const after = applyOrThrow({ kind: 'advance_clock', hours: 4 }, ordered, world, 'lenient').state;

    expect(key(after.units.get('vedette')!.column[0]!)).not.toBe(key(start));
    // And it is still its parent's.
    expect(after.units.get('vedette')!.parentUnitId).toBe('red-2');
  });
});

describe('a patrol and the troops it came from', () => {
  const scout = (id: string): Unit => ({ ...division(id, 'red'), traits: ['scout'] });

  const detached = (): CampaignState => {
    const base = applyOrThrow(
      { kind: 'add_unit', unit: scout('red-2') },
      setUp(),
      world,
      'lenient',
    ).state;
    return applyOrThrow(
      { kind: 'detach_patrol', unitId: 'red-2', patrolId: 'vedette' },
      base,
      world,
      'lenient',
    ).state;
  };

  it('points back at its parent, which is where its condition is read from', () => {
    const state = detached();
    const patrol = state.units.get('vedette')!;
    expect(parentOf(state, patrol)?.id).toBe('red-2');
    // A formation of its own has no parent to read.
    expect(parentOf(state, state.units.get('red-1')!)).toBeUndefined();
  });

  it('stays immune while its parent falls apart', () => {
    // The parent is wrecked: starving, broken and exhausted. The patrol is twenty troopers
    // a day's ride away and none of that reaches them — which is the point of the immunity
    // rather than an oversight in it.
    const state = applyOrThrow(
      {
        kind: 'set_unit_stats',
        unitId: 'red-2',
        changes: { morale: 0, provisions: 0, fatigue: 90 },
      },
      detached(),
      world,
      'lenient',
    ).state;

    const parent = state.units.get('red-2')!;
    const patrol = state.units.get('vedette')!;

    expect(isBroken(parent)).toBe(true);
    expect(isStarving(parent)).toBe(true);
    expect(isBroken(patrol)).toBe(false);
    expect(isStarving(patrol)).toBe(false);
    expect(presentUnderArms(patrol)).toBe(PATROL_PAPER_STRENGTH);

    // And the condition a reader wants is the parent's, live — not a copy taken when the
    // patrol rode out, which would still be reading full morale and forty provisions.
    expect(parentOf(state, patrol)!.morale).toBe(0);
    expect(parentOf(state, patrol)!.fatigue).toBe(90);
  });
});

/**
 * Battlefields.
 *
 * Ground rather than an object. The campaign layer does not resolve battles — a division's
 * frontage is a kilometre and a hex is a kilometre, so everything that makes a battle a
 * battle happens below this map. What it tracks is only which ground has stopped behaving
 * like open country, and who is standing on it.
 */
describe('declare_battle', () => {
  it('marks ground as fought over, and gives it back', () => {
    const state = setUp();

    const declared = applyOrThrow(
      { kind: 'declare_battle', coords: [LAND] },
      state,
      world,
      'strict',
    ).state;
    expect(declared.battle.has(key(LAND))).toBe(true);

    const ended = applyOrThrow(
      { kind: 'end_battle', coords: [LAND] },
      declared,
      world,
      'strict',
    ).state;
    expect(ended.battle.has(key(LAND))).toBe(false);
  });

  it('refuses a battle nowhere, and a battle off the map', () => {
    const state = setUp();

    expect(check({ kind: 'declare_battle', coords: [] }, state, world, 'strict')).toContainEqual(
      expect.objectContaining({ code: CODES.MALFORMED }),
    );
    expect(
      check({ kind: 'declare_battle', coords: [{ q: 999, r: 999 }] }, state, world, 'strict'),
    ).toContainEqual(expect.objectContaining({ code: CODES.OFF_MAP }));
  });

  it('is a referee command that bypasses no rule', () => {
    // Declaring a battle is not an illegal march. There is nothing for `force` to do here,
    // so the checker has only hard violations to give.
    const state = setUp();
    const out = apply({ kind: 'declare_battle', coords: [] }, state, world, 'lenient', {
      force: true,
    });
    expect(out.ok).toBe(false);
  });

  it('folds the same way on replay', () => {
    const state = setUp();
    const log: LoggedEvent[] = [];
    let s = state;
    for (const cmd of [
      { kind: 'declare_battle', coords: [LAND] },
      { kind: 'end_battle', coords: [LAND] },
      { kind: 'declare_battle', coords: [LAND] },
    ] as Command[]) {
      const out = apply(cmd, s, world, 'strict');
      expect(out.ok).toBe(true);
      log.push(...out.events);
      s = out.state;
    }
    expect([...replay(log).battle].sort()).toEqual([...s.battle].sort());
  });

  it('counts a formation as engaged wherever its footprint touches the fighting', () => {
    let s = setUp();
    // Strung out along a road: the head three hexes away from where the fighting is.
    const path = [0, 1, 2, 3].map((i) => ({ q: LAND.q + i, r: LAND.r }));
    s = applyOrThrow({ kind: 'teleport_unit', unitId: 'red-1', column: path }, s, world, 'strict')
      .state;

    // The tail, not the head. A division trailing through a battle has not let go of it.
    const tail = path[path.length - 1]!;
    s = applyOrThrow({ kind: 'declare_battle', coords: [tail] }, s, world, 'strict').state;

    const unit = s.units.get('red-1')!;
    expect(isEngaged(s, unit)).toBe(true);
    expect(engaged(s).map((u) => u.id)).toEqual(['red-1']);

    // And it disengages by marching out, with no flag for anyone to forget to clear.
    const away = applyOrThrow(
      { kind: 'teleport_unit', unitId: 'red-1', column: [LAND] },
      s,
      world,
      'strict',
    ).state;
    expect(isEngaged(away, away.units.get('red-1')!)).toBe(false);
  });

  it('has nobody engaged where there is no fighting', () => {
    const s = setUp();
    expect(s.battle.size).toBe(0);
    expect(engaged(s)).toEqual([]);
  });
});
