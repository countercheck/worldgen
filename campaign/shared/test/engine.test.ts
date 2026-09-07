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
import { hasSurveyed, replay, type CampaignState } from '../src/state.js';
import { KIND_DEFAULTS, type Unit } from '../src/unit.js';
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

function division(id: string, faction: string, at: Hex = LAND, effectives = 5000): Unit {
  return {
    id,
    name: `${id} Division`,
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
    guns: 6,
    marchSpeedKmh: KIND_DEFAULTS.infantry.marchSpeedKmh,
    spacingM: KIND_DEFAULTS.infantry.spacingM,
    spacingMultiplier: 1.3,
    traits: [],
    formation: 'march',
    column: [at],
    hoursMarchedToday: 0,
    corps: null,
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
    expect(after.effectives).toBe(before.effectives);
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
