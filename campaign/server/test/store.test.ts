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
    paperStrength: 5000,
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
 * A campaign with one formation and one commander riding with it.
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

  it('mints a distinct link per seat, resolving to that commander', () => {
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
      { kind: 'add_unit', unit: { ...division('r1', 'red', land[0]!), paperStrength: 100 } },
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

  it('snapshots by distance along the log, not by a coincidence of sequence numbers', () => {
    // `nextSeq % 50` only fires when a command happens to land the sequence exactly on a
    // multiple, and a command emits as many events as it produced — an advance through a
    // day of marching emits dozens. In practice none were ever written after setup, and
    // every read replayed the whole log, which is the thing snapshots exist to stop.
    const db = openDb();
    const store = new CampaignStore(db);
    const created = store.create({
      id: 'c1',
      name: 'Test',
      worldDoc,
      seed: 42,
      factions: [{ id: 'red', name: 'Red', color: '#f00' }],
    });

    const drive = (command: Command) => store.execute(created.campaign, command, REFEREE_ROLE);
    expect(drive({ kind: 'add_unit', unit: division('r1', 'red', land[0]!) }).ok).toBe(true);
    expect(
      drive({ kind: 'add_commander', commander: commander('c-r1', 'red', 'r1') }).ok,
    ).toBe(true);

    // A column actually marching, so each advance emits several events rather than one.
    // That is the case the modulus missed: it lands on a multiple of 50 only by accident
    // once a command can carry the sequence past one in a single step.
    const far = land[land.length - 1]!;
    expect(drive({ kind: 'set_task', unitId: 'r1', destination: far }).ok).toBe(true);
    for (let i = 0; i < 60; i++) drive({ kind: 'advance_clock', hours: 3 });

    const rows = db
      .prepare(`SELECT seq FROM snapshots WHERE campaign_id = ? ORDER BY seq`)
      .all('c1') as { seq: number }[];

    // One per interval the log has actually covered — not the two the modulus happened
    // to catch out of the four this campaign earned.
    const head = store.state('c1').nextSeq;
    expect(rows.length).toBeGreaterThanOrEqual(Math.floor(head / 50));

    // And each is a real saving: no two closer together than the interval, and the newest
    // close enough to the head that a read replays a handful of events, not the campaign.
    let previous = 0;
    for (const row of rows) {
      expect(row.seq - previous).toBeGreaterThanOrEqual(50);
      previous = row.seq;
    }
    expect(head - rows[rows.length - 1]!.seq).toBeLessThan(50);
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
  it('records what a commander covers once they are appointed', () => {
    const s = withCommander();
    const surveyed = s.store.state('c1').knowledge.get('c-r1')!.surveyed;
    expect(surveyed.size).toBeGreaterThan(0);
    expect(surveyed.has(key(land[0]!))).toBe(true);
  });

  it('records nothing for a formation nobody commands', () => {
    // Formations observe, but observing is not knowing: what a division sees becomes
    // knowledge when there is a commander riding with it to take note of it.
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

  it('gives a superior what their subordinate covers, and not the reverse', () => {
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
    expect(junior.has(key(land[0]!)), 'a subordinate learned their chief\'s ground').toBe(false);
  });

  it('remembers ground a formation has left', () => {
    // The whole point of storing knowledge rather than recomputing it: a commander does not
    // forget a valley the moment their column marches out of the far side.
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
    // indistinguishable from a commander who forgot the campaign.
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

  it('keeps a march and the waypoints it still has to make', () => {
    const s = withCommander();
    const far = land[land.length - 1]!;
    const waypoint = land[Math.floor(land.length / 2)]!;
    expect(
      run(s, { kind: 'set_task', unitId: 'r1', destination: far, via: [waypoint] }).ok,
    ).toBe(true);
    run(s, { kind: 'advance_clock', hours: 3 });

    const state = s.store.state('c1');
    const round = deserialise(serialise(state));
    expect(round.tasks.get('r1')).toEqual(state.tasks.get('r1'));
  });

  it('reads a snapshot written before waypoints existed as a march with none', () => {
    // Snapshots are not versioned, and one written by the previous build has neither
    // field. A march that named no waypoints is exactly what such a task was.
    const s = withCommander();
    const far = land[land.length - 1]!;
    expect(run(s, { kind: 'set_task', unitId: 'r1', destination: far }).ok).toBe(true);

    const doc = JSON.parse(serialise(s.store.state('c1'))) as {
      tasks: Record<string, unknown>[];
    };
    for (const t of doc.tasks) {
      delete t.via;
      delete t.viaIndex;
    }

    const round = deserialise(JSON.stringify(doc));
    expect(round.tasks.get('r1')!.via).toEqual([]);
    expect(round.tasks.get('r1')!.viaIndex).toBe(0);
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

describe('a campaign carries its own numbers', () => {
  it('records the ruleset it was started under', () => {
    const store = new CampaignStore(openDb());
    const { campaign } = store.create({
      id: 'c-brisk',
      name: 'Brisk',
      worldDoc,
      factions: [{ id: 'red', name: 'Red', color: '#f00' }],
      ruleset: 'brisk',
    });

    expect(campaign.ruleset).toBe('brisk');
    expect(campaign.config.maxMarchHoursPerDay).toBe(12);
  });

  it('reads them back off the row, not off the defaults', () => {
    const db = openDb();
    const store = new CampaignStore(db);
    store.create({
      id: 'c-brisk',
      name: 'Brisk',
      worldDoc,
      factions: [{ id: 'red', name: 'Red', color: '#f00' }],
      ruleset: 'brisk',
      config: { freePatrols: 7 },
    });

    // A second store over the same database: nothing is carried in memory between them.
    const reopened = new CampaignStore(db).campaign('c-brisk')!;
    expect(reopened.ruleset).toBe('brisk');
    expect(reopened.config.maxMarchHoursPerDay).toBe(12);
    expect(reopened.config.freePatrols).toBe(7);
  });

  it('keeps two campaigns on different rules in one database', () => {
    // The point of naming them. A referee running the book and the house amendments at the
    // same time must not have one quietly re-tune the other.
    const db = openDb();
    const store = new CampaignStore(db);
    const factions = [{ id: 'red', name: 'Red', color: '#f00' }];

    store.create({ id: 'by-the-book', name: 'A', worldDoc, factions });
    store.create({ id: 'house', name: 'B', worldDoc, factions, ruleset: 'brisk' });

    expect(store.campaign('by-the-book')!.config.maxMarchHoursPerDay).toBe(20);
    expect(store.campaign('house')!.config.maxMarchHoursPerDay).toBe(12);
  });

  it('is not re-tuned by a ruleset edited afterwards', () => {
    // Resolved at creation and stored. The numbers a campaign has been played under are a
    // fact about it, and a house rule changed next month must not rewrite its history.
    const db = openDb();
    new CampaignStore(db).create({
      id: 'c1',
      name: 'A',
      worldDoc,
      factions: [{ id: 'red', name: 'Red', color: '#f00' }],
      ruleset: 'brisk',
    });

    const row = db
      .prepare(`SELECT config_json FROM campaigns WHERE id = ?`)
      .get('c1') as { config_json: string };
    const stored = JSON.parse(row.config_json) as { maxMarchHoursPerDay: number };
    expect(stored.maxMarchHoursPerDay).toBe(12);
  });

  it('falls back to an unknown ruleset rather than refusing to start', () => {
    const store = new CampaignStore(openDb());
    const { campaign } = store.create({
      id: 'c1',
      name: 'A',
      worldDoc,
      factions: [{ id: 'red', name: 'Red', color: '#f00' }],
      ruleset: 'nonesuch',
    });
    expect(campaign.ruleset).toBe('standard');
  });

  it('gives a row written before rulesets existed the rules as written', () => {
    // The migration case. An older database has the columns added on open, and its
    // campaigns have been playing under the standard numbers all along.
    const db = openDb();
    const store = new CampaignStore(db);
    store.create({
      id: 'c1',
      name: 'A',
      worldDoc,
      factions: [{ id: 'red', name: 'Red', color: '#f00' }],
    });
    db.prepare(`UPDATE campaigns SET config_json = NULL WHERE id = ?`).run('c1');

    const loaded = store.campaign('c1')!;
    expect(loaded.config.maxMarchHoursPerDay).toBe(20);
    expect(loaded.config.speeds.infantry.road).toBe(3);
  });

  it("runs commands on the campaign's own numbers", () => {
    // The whole point. A campaign under a ruleset that halves the day must cap a march at
    // that, not at the store's.
    const db = openDb();
    const store = new CampaignStore(db);
    const { campaign } = store.create({
      id: 'c1',
      name: 'A',
      worldDoc,
      factions: [{ id: 'red', name: 'Red', color: '#f00' }],
      ruleset: 'brisk',
    });

    expect(
      store.execute(campaign, { kind: 'add_unit', unit: division('r1', 'red', land[0]!) }, REFEREE_ROLE).ok,
    ).toBe(true);
    const far = land[land.length - 1]!;
    store.execute(campaign, { kind: 'set_task', unitId: 'r1', destination: far }, REFEREE_ROLE);
    store.execute(campaign, { kind: 'advance_clock', hours: 20 }, REFEREE_ROLE);

    // Twelve hours of marching, not twenty. To a tolerance, because a day is accumulated
    // out of thirds of an hour and binary floating point does not sum them to exactly
    // twelve — the cap holds, the last bit does not.
    expect(store.state('c1').units.get('r1')!.hoursMarchedToday).toBeLessThanOrEqual(12 + 1e-9);
    expect(store.state('c1').units.get('r1')!.hoursMarchedToday).toBeGreaterThan(11);
  });
});

describe('a log written before a field was renamed', () => {
  it('still folds, because the log is append-only and cannot be rewritten', () => {
    // `effectives` became `paperStrength`. An event written under the old name is a fact
    // in the past tense: it cannot be edited, so the rename is honoured on the way in.
    const db = openDb();
    const store = new CampaignStore(db);
    const { campaign } = store.create({
      id: 'c1',
      name: 'A',
      worldDoc,
      factions: [{ id: 'red', name: 'Red', color: '#f00' }],
    });
    expect(
      store.execute(campaign, { kind: 'add_unit', unit: division('r1', 'red', land[0]!) }, REFEREE_ROLE).ok,
    ).toBe(true);

    // Rewrite that event as an older build would have written it, and drop the snapshots
    // so the campaign has to be rebuilt from the log alone.
    const row = db
      .prepare(`SELECT seq, payload_json FROM events WHERE campaign_id = ? AND kind = 'unit_added'`)
      .get('c1') as { seq: number; payload_json: string };
    const old = JSON.parse(row.payload_json) as { unit: Record<string, unknown> };
    old.unit['effectives'] = old.unit['paperStrength'];
    delete old.unit['paperStrength'];
    db.prepare(`UPDATE events SET payload_json = ? WHERE campaign_id = ? AND seq = ?`)
      .run(JSON.stringify(old), 'c1', row.seq);
    db.prepare(`DELETE FROM snapshots WHERE campaign_id = ?`).run('c1');

    const state = new CampaignStore(db).state('c1');
    expect(state.units.get('r1')!.paperStrength).toBe(5000);
  });

  it('never undoes an upgrade already applied', () => {
    // A payload carrying both names — which should not happen, but a half-migrated
    // database is exactly the case that eats a campaign — keeps the new one.
    const db = openDb();
    const store = new CampaignStore(db);
    const { campaign } = store.create({
      id: 'c1',
      name: 'A',
      worldDoc,
      factions: [{ id: 'red', name: 'Red', color: '#f00' }],
    });
    store.execute(campaign, { kind: 'add_unit', unit: division('r1', 'red', land[0]!) }, REFEREE_ROLE);

    const row = db
      .prepare(`SELECT seq, payload_json FROM events WHERE campaign_id = ? AND kind = 'unit_added'`)
      .get('c1') as { seq: number; payload_json: string };
    const both = JSON.parse(row.payload_json) as { unit: Record<string, unknown> };
    both.unit['effectives'] = 1;
    db.prepare(`UPDATE events SET payload_json = ? WHERE campaign_id = ? AND seq = ?`)
      .run(JSON.stringify(both), 'c1', row.seq);
    db.prepare(`DELETE FROM snapshots WHERE campaign_id = ?`).run('c1');

    expect(new CampaignStore(db).state('c1').units.get('r1')!.paperStrength).toBe(5000);
  });

  it('upgrades a snapshot as well as a log', () => {
    const db = openDb();
    const store = new CampaignStore(db);
    const { campaign } = store.create({
      id: 'c1',
      name: 'A',
      worldDoc,
      factions: [{ id: 'red', name: 'Red', color: '#f00' }],
    });
    store.execute(campaign, { kind: 'add_unit', unit: division('r1', 'red', land[0]!) }, REFEREE_ROLE);

    const doc = JSON.parse(serialise(store.state('c1'))) as { units: Record<string, unknown>[] };
    for (const u of doc.units) {
      u['effectives'] = u['paperStrength'];
      delete u['paperStrength'];
    }
    expect(deserialise(JSON.stringify(doc)).units.get('r1')!.paperStrength).toBe(5000);
  });
});
