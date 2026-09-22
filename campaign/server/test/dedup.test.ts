/**
 * Storing a world once, however many campaigns are played on it.
 *
 * A world used to live in the campaign row, so a referee running five games on one map
 * stored that map five times. At 23 MB for a 200x200 world that is 23 GB across a thousand
 * campaigns against 225 MB — and the hash was already being computed and stored, so the
 * row knew the world's identity and simply did not act on it.
 *
 * The migration is the delicate half. It moves data rather than adding a column, and it
 * runs against the only copy of somebody's campaign, so the tests below care more about
 * what happens when it goes wrong than about what happens when it goes right.
 */

import { readFileSync } from 'node:fs';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { DatabaseSync, openDb, type Db } from '../src/db.js';
import { CampaignStore } from '../src/store.js';

const world = (): unknown =>
  JSON.parse(
    readFileSync(new URL('../../shared/test/fixtures/world-32x32.json', import.meta.url), 'utf8'),
  );

const FACTIONS = [{ id: 'red', name: 'Red', color: '#c00' }];

const count = (db: Db, table: string): number =>
  (db.prepare(`SELECT COUNT(*) AS n FROM ${table}`).get() as { n: number }).n;

const columns = (db: Db): string[] =>
  (db.prepare(`PRAGMA table_info(campaigns)`).all() as { name: string }[]).map((c) => c.name);

describe('one world, many campaigns', () => {
  it('stores a shared map once', () => {
    const db = openDb();
    const store = new CampaignStore(db);
    const doc = world();

    for (const id of ['c1', 'c2', 'c3', 'c4', 'c5']) {
      store.create({ id, name: id, worldDoc: doc, factions: FACTIONS });
    }

    expect(count(db, 'campaigns')).toBe(5);
    expect(count(db, 'worlds')).toBe(1);
  });

  it('keeps different maps apart', () => {
    const db = openDb();
    const store = new CampaignStore(db);
    const a = world() as { seed: number };
    const b = { ...(world() as object), seed: 999 };

    store.create({ id: 'c1', name: 'a', worldDoc: a, factions: FACTIONS });
    store.create({ id: 'c2', name: 'b', worldDoc: b, factions: FACTIONS });

    expect(count(db, 'worlds')).toBe(2);
    expect(store.campaign('c1')!.worldHash).not.toBe(store.campaign('c2')!.worldHash);
  });

  it('gives every campaign the world it was created with', () => {
    const db = openDb();
    const store = new CampaignStore(db);
    const a = world();
    const b = { ...(world() as object), seed: 999 };

    store.create({ id: 'c1', name: 'a', worldDoc: a, factions: FACTIONS });
    store.create({ id: 'c2', name: 'b', worldDoc: b, factions: FACTIONS });

    // Read back rather than trusted from creation: the join is what could go wrong.
    expect((store.campaign('c2')!.worldDoc as { seed: number }).seed).toBe(999);
    expect((store.campaign('c1')!.worldDoc as { seed: number }).seed).not.toBe(999);
  });
});

describe('migrating a database written before worlds were shared', () => {
  /** A database in the old shape: `world_blob` on the campaign, no `worlds` table. */
  const legacy = (): string => {
    const path = join(mkdtempSync(join(tmpdir(), 'campaign-migrate-')), 'old.db');
    const db = new DatabaseSync(path);
    db.exec(`CREATE TABLE campaigns (
      id TEXT PRIMARY KEY, name TEXT NOT NULL, world_hash TEXT NOT NULL,
      world_blob TEXT NOT NULL, strictness TEXT NOT NULL DEFAULT 'strict',
      ruleset TEXT NOT NULL DEFAULT 'standard', config_json TEXT, created_at TEXT NOT NULL)`);
    // Two campaigns on one map and one on another: the case the migration exists for.
    db.exec(`INSERT INTO campaigns (id, name, world_hash, world_blob, created_at) VALUES
      ('c1','a','hash-a','{"hexes":[1]}','1805'),
      ('c2','b','hash-a','{"hexes":[1]}','1805'),
      ('c3','c','hash-b','{"hexes":[2]}','1805')`);
    db.close();
    return path;
  };

  it('moves every world across, sharing what is shared', () => {
    const db = openDb(legacy());

    expect(count(db, 'worlds')).toBe(2);
    expect(count(db, 'campaigns')).toBe(3);
    const blobs = db.prepare(`SELECT hash, blob FROM worlds ORDER BY hash`).all();
    expect(blobs).toEqual([
      { hash: 'hash-a', blob: '{"hexes":[1]}' },
      { hash: 'hash-b', blob: '{"hexes":[2]}' },
    ]);
  });

  it('drops the column it no longer writes', () => {
    const db = openDb(legacy());
    expect(columns(db)).not.toContain('world_blob');
    expect(columns(db)).toContain('world_hash');
  });

  it('runs once and is harmless afterwards', () => {
    // `openDb` runs on every start, so the second start must be a no-op rather than an
    // error or a second pass over data that has already moved.
    const path = legacy();
    openDb(path);
    const again = openDb(path);
    expect(count(again, 'worlds')).toBe(2);
    expect(columns(again)).not.toContain('world_blob');
  });

  it('changes nothing at all when it cannot finish', () => {
    // The failure that matters. A campaign with no world to move would be left pointing
    // at nothing — a database that opens cleanly and dies on the first request, which is
    // the worst moment to discover it. So the migration refuses, and refusing has to mean
    // the database is exactly as it was rather than half-moved.
    const path = join(mkdtempSync(join(tmpdir(), 'campaign-migrate-')), 'old.db');
    const db = new DatabaseSync(path);
    db.exec(`CREATE TABLE campaigns (
      id TEXT PRIMARY KEY, name TEXT NOT NULL, world_hash TEXT NOT NULL,
      world_blob TEXT NOT NULL, strictness TEXT NOT NULL DEFAULT 'strict',
      ruleset TEXT NOT NULL DEFAULT 'standard', config_json TEXT, created_at TEXT NOT NULL)`);
    db.exec(`INSERT INTO campaigns (id, name, world_hash, world_blob, created_at) VALUES
      ('c1','good','hash-a','{"hexes":[1]}','1805'),
      ('c2','empty','hash-b','','1805')`);
    db.close();

    expect(() => openDb(path)).toThrow(/without a world/);

    // Reopened raw, bypassing the migration, to see what is actually on disk.
    const after = new DatabaseSync(path);
    const names = (after.prepare(`PRAGMA table_info(campaigns)`).all() as { name: string }[])
      .map((c) => c.name);
    expect(names, 'the column was dropped despite the failure').toContain('world_blob');
    expect(
      (after.prepare(`SELECT world_blob FROM campaigns WHERE id = 'c1'`).get() as {
        world_blob: string;
      }).world_blob,
      'a world was lost despite the failure',
    ).toBe('{"hexes":[1]}');
    after.close();
  });

  it('leaves a fresh database alone', () => {
    const db = openDb();
    expect(count(db, 'worlds')).toBe(0);
    expect(columns(db)).not.toContain('world_blob');
  });
});
