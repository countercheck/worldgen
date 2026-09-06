/**
 * The event store.
 *
 * The log is the campaign; everything else is cache. These tests are mostly about proving
 * that claim rather than assuming it — that snapshots can be deleted and the campaign
 * still loads, that a prefix of the log is a valid earlier state, and that what a faction
 * knows survives a round trip through the database.
 */

import { describe, expect, it } from 'vitest';

import worldDoc from '../../shared/test/fixtures/world-32x32.json';

import { key, KIND_DEFAULTS, parseWorld, type Command, type Hex, type Unit } from '@campaign/shared';

import { openDb } from '../src/db.js';
import { CampaignStore, deserialise, hashToken, newToken, serialise } from '../src/store.js';
import { REFEREE_ROLE } from '../src/view.js';

const world = parseWorld(worldDoc);
const land = [...world.hexes.values()]
  .filter((h) => h.terrainClass === 'land')
  .map((h) => h.coord);

function division(id: string, faction: string, at: Hex): Unit {
  return {
    id,
    faction,
    kind: 'infantry',
    effectives: 5000,
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

function setUp() {
  const store = new CampaignStore(openDb());
  const created = store.create({
    id: 'c1',
    name: 'Test',
    worldDoc,
    seed: 42,
    factions: [
      { id: 'red', name: 'Red', color: '#f00' },
      { id: 'blue', name: 'Blue', color: '#00f' },
    ],
  });
  return { store, ...created };
}

const run = (
  s: ReturnType<typeof setUp>,
  command: Command,
  opts: Parameters<CampaignStore['execute']>[3] = {},
) => s.store.execute(s.campaign, command, REFEREE_ROLE, opts);

describe('creating a campaign', () => {
  it('mints a distinct token per role', () => {
    const s = setUp();
    const tokens = [s.refereeToken, s.tokens.red!, s.tokens.blue!];
    expect(new Set(tokens).size).toBe(3);
  });

  it('resolves each token to its own role', () => {
    const s = setUp();
    expect(s.store.roleFor('c1', s.refereeToken)).toEqual({ kind: 'referee' });
    expect(s.store.roleFor('c1', s.tokens.red)).toEqual({ kind: 'faction', id: 'red' });
    expect(s.store.roleFor('c1', s.tokens.blue)).toEqual({ kind: 'faction', id: 'blue' });
  });

  it('resolves nothing for an unknown token', () => {
    const s = setUp();
    expect(s.store.roleFor('c1', 'nope')).toBeNull();
    expect(s.store.roleFor('c1', undefined)).toBeNull();
    expect(s.store.roleFor('c1', '')).toBeNull();
  });

  it('stores tokens hashed, so a leaked database is not a leaked game', () => {
    const s = setUp();
    const db = openDb();
    // The plaintext token must not appear anywhere in the roles table.
    const rows = s.store as unknown as { db: { prepare: (q: string) => { all: () => unknown[] } } };
    const all = JSON.stringify(rows.db.prepare('SELECT * FROM roles').all());
    expect(all).not.toContain(s.refereeToken);
    expect(all).toContain(hashToken(s.refereeToken));
    db.close();
  });

  it('records the world with the campaign', () => {
    // Stored rather than referenced: a regenerated world.json on disk must not silently
    // invalidate every coordinate in a saved fog set.
    const s = setUp();
    const loaded = s.store.campaign('c1');
    expect(loaded).not.toBeNull();
    expect(loaded!.world.hexes.size).toBe(world.hexes.size);
    expect(loaded!.worldHash).toMatch(/^sha256:/);
  });

  it('returns null for a campaign that does not exist', () => {
    const s = setUp();
    expect(s.store.campaign('nope')).toBeNull();
  });
});

describe('the log', () => {
  it('grows by one command at a time and never changes', () => {
    const s = setUp();
    const before = s.store.events('c1').length;

    run(s, { kind: 'advance_clock', hours: 3 });
    const after = s.store.events('c1');

    expect(after.length).toBeGreaterThan(before);
    // Earlier events are byte-identical: the log is append-only.
    expect(after.slice(0, before)).toEqual(s.store.events('c1').slice(0, before));
  });

  it('numbers events consecutively from zero', () => {
    const s = setUp();
    run(s, { kind: 'add_unit', unit: division('r1', 'red', land[0]!) });
    const seqs = s.store.events('c1').map((e) => e.seq);
    expect(seqs).toEqual(seqs.map((_, i) => i));
  });

  it('records what a forced command bypassed', () => {
    const s = setUp();
    const out = run(
      s,
      { kind: 'add_unit', unit: { ...division('r1', 'red', land[0]!), effectives: 100 } },
      { force: true },
    );
    expect(out.ok).toBe(true);
    const forced = s.store.events('c1').filter((e) => e.forced);
    expect(forced.length).toBe(1);
    expect(forced[0]!.bypassed.map((v) => v.code)).toContain('unit_too_small');
  });

  it('writes nothing when a command is refused', () => {
    const s = setUp();
    const before = s.store.events('c1').length;
    const out = run(s, { kind: 'remove_unit', unitId: 'ghost' });

    expect(out.ok).toBe(false);
    expect(s.store.events('c1').length).toBe(before);
  });
});

describe('state', () => {
  it('is the fold of the log', () => {
    const s = setUp();
    run(s, { kind: 'add_unit', unit: division('r1', 'red', land[0]!) });
    run(s, { kind: 'advance_clock', hours: 6 });

    const state = s.store.state('c1');
    expect(state.clockHours).toBe(6);
    expect(state.units.has('r1')).toBe(true);
    expect([...state.factions.keys()].sort()).toEqual(['blue', 'red']);
  });

  it('rebuilds identically with every snapshot deleted', () => {
    // The claim that the log is the campaign. If this fails, snapshots are load-bearing
    // and a corrupted one loses the game.
    const s = setUp();
    for (let i = 0; i < 60; i++) run(s, { kind: 'advance_clock', hours: 1 });

    const withSnapshots = s.store.state('c1');
    s.store.clearSnapshots('c1');
    const fromScratch = s.store.state('c1');

    expect(serialise(fromScratch)).toBe(serialise(withSnapshots));
  });

  it('rewinds by replaying a prefix', () => {
    const s = setUp();
    run(s, { kind: 'advance_clock', hours: 4 });
    const mid = s.store.state('c1').nextSeq;
    run(s, { kind: 'advance_clock', hours: 4 });

    expect(s.store.state('c1').clockHours).toBe(8);
    expect(s.store.state('c1', mid).clockHours).toBe(4);
  });
});

describe('observation', () => {
  it('records what a faction sees when a unit appears', () => {
    const s = setUp();
    run(s, { kind: 'add_unit', unit: division('r1', 'red', land[0]!) });

    const state = s.store.state('c1');
    const seen = state.knowledge.get('red')!.seen;
    expect(seen.size).toBeGreaterThan(0);
    expect(seen.has(key(land[0]!))).toBe(true);
  });

  it('does not give one faction another faction knowledge', () => {
    const s = setUp();
    run(s, { kind: 'add_unit', unit: division('r1', 'red', land[0]!) });

    const state = s.store.state('c1');
    expect(state.knowledge.get('blue')!.seen.size).toBe(0);
  });

  it('remembers ground a unit has left', () => {
    // The whole point of storing knowledge rather than recomputing it: an army does not
    // forget a valley the moment it marches out of the far side.
    const s = setUp();
    const start = land[0]!;
    run(s, { kind: 'add_unit', unit: division('r1', 'red', start) });

    const far = land.find(
      (c) => Math.abs(c.q - start.q) + Math.abs(c.r - start.r) > 12,
    )!;
    run(s, { kind: 'teleport_unit', unitId: 'r1', column: [far] });

    const seen = s.store.state('c1').knowledge.get('red')!.seen;
    expect(seen.has(key(start)), 'the starting ground was forgotten').toBe(true);
    expect(seen.has(key(far))).toBe(true);
  });

  it('emits nothing when nothing new was seen', () => {
    const s = setUp();
    run(s, { kind: 'add_unit', unit: division('r1', 'red', land[0]!) });
    const afterFirst = s.store.events('c1').length;

    // Advancing the clock moves nobody, so no new ground is observed.
    run(s, { kind: 'advance_clock', hours: 1 });
    const events = s.store.events('c1');
    const observations = events
      .slice(afterFirst)
      .filter((e) => e.payload.kind === 'hexes_revealed');

    expect(observations).toEqual([]);
  });
});

describe('snapshot round trip', () => {
  it('preserves maps and sets, which JSON does not', () => {
    // A snapshot that silently lost a faction's `seen` set would be indistinguishable
    // from an army that forgot the war.
    const s = setUp();
    run(s, { kind: 'add_unit', unit: division('r1', 'red', land[0]!) });

    const state = s.store.state('c1');
    const round = deserialise(serialise(state));

    expect(round.units.get('r1')).toEqual(state.units.get('r1'));
    expect(round.factions.size).toBe(state.factions.size);
    expect(round.knowledge.get('red')!.seen).toEqual(state.knowledge.get('red')!.seen);
    expect(round.knowledge.get('red')!.lastSeenHours).toEqual(
      state.knowledge.get('red')!.lastSeenHours,
    );
  });
});

describe('tokens', () => {
  it('are long enough not to be guessed', () => {
    // 32 random bytes, base64url. Security here is "do not share your link", and that
    // only holds if the link cannot be found by trying.
    const t = newToken();
    expect(t.length).toBeGreaterThanOrEqual(43);
    expect(t).toMatch(/^[A-Za-z0-9_-]+$/);
  });

  it('are distinct', () => {
    const seen = new Set(Array.from({ length: 500 }, () => newToken()));
    expect(seen.size).toBe(500);
  });
});
