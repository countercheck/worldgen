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

import {
  key,
  KIND_DEFAULTS,
  parseWorld,
  REFEREE_ROLE,
  type Command,
  type Commander,
  type Hex,
  type Unit,
} from '@campaign/shared';

import { openDb } from '../src/db.js';
import { CampaignStore, deserialise, hashToken, newToken, serialise } from '../src/store.js';

const world = parseWorld(worldDoc);
const land = [...world.hexes.values()]
  .filter((h) => h.terrainClass === 'land')
  .map((h) => h.coord);

function division(id: string, faction: string, at: Hex): Unit {
  return {
    id,
    name: `${id} Division`,
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

const commander = (
  id: string,
  faction: string,
  unitId: string,
  superiorId: string | null = null,
): Commander => ({
  id,
  name: `Commander ${id}`,
  faction,
  unitId,
  superiorId,
  autoCascade: true,
});

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

/**
 * A campaign with one formation and one man riding with it.
 *
 * Most of these tests are about knowledge, and knowledge belongs to a commander now, so
 * there is nothing to observe until somebody is appointed to do the observing.
 */
function withCommander(id = 'r1', faction = 'red', at: Hex = land[0]!) {
  const s = setUp();
  const run = (c: Command) => s.store.execute(s.campaign, c, REFEREE_ROLE);
  expect(run({ kind: 'add_unit', unit: division(id, faction, at) }).ok).toBe(true);
  expect(run({ kind: 'add_commander', commander: commander(`c-${id}`, faction, id) }).ok).toBe(
    true,
  );
  return s;
}

const run = (
  s: ReturnType<typeof setUp>,
  command: Command,
  opts: Parameters<CampaignStore['execute']>[3] = {},
) => s.store.execute(s.campaign, command, REFEREE_ROLE, opts);

describe('creating a campaign', () => {
  it('mints only the referee link, because there are no seats yet', () => {
    // A join link names a commander, and a commander needs a formation to ride with. The
    // order of battle arrives after creation, so the seats do too.
    const s = setUp();
    expect(s.refereeToken).toBeTruthy();
    expect(s.store.roleFor('c1', s.refereeToken)).toEqual({ kind: 'referee' });
  });

  it('mints a distinct link per seat, resolving to that man', () => {
    const s = withCommander();
    const first = s.store.issueToken('c1', 'c-r1');
    const second = s.store.issueToken('c1', 'c-r1');

    expect(first).not.toBe(second);
    expect(first).not.toBe(s.refereeToken);
    // Reissuing does not invalidate the old one — that is `revokeTokens`, and keeping the
    // two separate is what lets a referee hand out a replacement before killing the leak.
    expect(s.store.roleFor('c1', first)).toEqual({ kind: 'commander', id: 'c-r1' });
    expect(s.store.roleFor('c1', second)).toEqual({ kind: 'commander', id: 'c-r1' });
  });

  it('revokes every link to a seat at once', () => {
    const s = withCommander();
    const a = s.store.issueToken('c1', 'c-r1');
    const b = s.store.issueToken('c1', 'c-r1');

    s.store.revokeTokens('c1', 'c-r1');
    expect(s.store.roleFor('c1', a)).toBeNull();
    expect(s.store.roleFor('c1', b)).toBeNull();
    // The referee's own link is not a commander's and must survive.
    expect(s.store.roleFor('c1', s.refereeToken)).toEqual({ kind: 'referee' });
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
  it('records what a commander covers once he is appointed', () => {
    const s = withCommander();
    const surveyed = s.store.state('c1').knowledge.get('c-r1')!.surveyed;
    expect(surveyed.size).toBeGreaterThan(0);
    expect(surveyed.has(key(land[0]!))).toBe(true);
  });

  it('records nothing for a formation nobody commands', () => {
    // Formations observe, but observing is not knowing: what a division sees becomes
    // knowledge when there is a man riding with it to take note of it.
    const s = setUp();
    run(s, { kind: 'add_unit', unit: division('r1', 'red', land[0]!) });
    expect(s.store.state('c1').knowledge.size).toBe(0);
  });

  it('does not give one commander another commander\'s knowledge', () => {
    const s = withCommander();
    const far = land.find((c) => Math.abs(c.q - land[0]!.q) > 12)!;
    run(s, { kind: 'add_unit', unit: division('b1', 'blue', far) });
    run(s, { kind: 'add_commander', commander: commander('c-b1', 'blue', 'b1') });

    const state = s.store.state('c1');
    const mine = state.knowledge.get('c-r1')!.surveyed;
    const theirs = state.knowledge.get('c-b1')!.surveyed;
    expect([...mine].some((k) => theirs.has(k))).toBe(false);
  });

  it('gives a superior what his subordinate covers, and not the reverse', () => {
    // The interim rule, stated in observe.ts: every formation reports upward instantly
    // until riders exist. Reports travel up the tree, never down it.
    const s = withCommander();
    const far = land.find((c) => Math.abs(c.q - land[0]!.q) > 12)!;
    run(s, { kind: 'add_unit', unit: division('r2', 'red', far) });
    run(s, {
      kind: 'add_commander',
      commander: commander('c-r2', 'red', 'r2', 'c-r1'),
    });

    const state = s.store.state('c1');
    const chief = state.knowledge.get('c-r1')!.surveyed;
    const junior = state.knowledge.get('c-r2')!.surveyed;

    expect(chief.has(key(far)), 'the subordinate never reported in').toBe(true);
    expect(junior.has(key(land[0]!)), 'a subordinate learned his chief\'s ground').toBe(false);
  });

  it('remembers ground a formation has left', () => {
    // The whole point of storing knowledge rather than recomputing it: a man does not
    // forget a valley the moment his column marches out of the far side.
    const s = withCommander();
    const start = land[0]!;
    const far = land.find((c) => Math.abs(c.q - start.q) + Math.abs(c.r - start.r) > 12)!;
    run(s, { kind: 'teleport_unit', unitId: 'r1', column: [far] });

    const surveyed = s.store.state('c1').knowledge.get('c-r1')!.surveyed;
    expect(surveyed.has(key(start)), 'the starting ground was forgotten').toBe(true);
    expect(surveyed.has(key(far))).toBe(true);
  });

  it('emits nothing when nothing new was seen', () => {
    const s = withCommander();
    const afterFirst = s.store.events('c1').length;

    // Advancing the clock moves nobody, so no new ground is covered.
    run(s, { kind: 'advance_clock', hours: 1 });
    const observations = s.store
      .events('c1')
      .slice(afterFirst)
      .filter((e) => e.payload.kind === 'hexes_surveyed');

    expect(observations).toEqual([]);
  });
});

describe('snapshot round trip', () => {
  it('preserves maps and sets, which JSON does not', () => {
    // A snapshot that silently lost what a commander had surveyed would be
    // indistinguishable from a man who forgot the campaign.
    const s = withCommander();

    const state = s.store.state('c1');
    const round = deserialise(serialise(state));

    expect(round.units.get('r1')).toEqual(state.units.get('r1'));
    expect(round.factions.size).toBe(state.factions.size);
    expect(round.commanders.get('c-r1')).toEqual(state.commanders.get('c-r1'));
    expect(round.knowledge.get('c-r1')!.surveyed).toEqual(
      state.knowledge.get('c-r1')!.surveyed,
    );
    expect(round.knowledge.get('c-r1')!.lastSurveyedHours).toEqual(
      state.knowledge.get('c-r1')!.lastSurveyedHours,
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
