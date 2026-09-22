/**
 * Worlds on disk, compressed.
 *
 * A world is tens of thousands of objects carrying the same dozen keys, which is about as
 * compressible as text gets: 23 MB becomes 4.5 MB. Multiplied against deduplication, a
 * thousand campaigns over fifty maps is tens of megabytes rather than tens of gigabytes.
 *
 * The half worth testing is the reading. A database written before this change holds plain
 * JSON in the same column, and a server that could not read it would present as every
 * campaign created before today having vanished.
 */

import { mkdtempSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { DatabaseSync, openDb, packWorld, unpackWorld, type Db } from '../src/db.js';
import { CampaignStore } from '../src/store.js';

const FACTIONS = [{ id: 'red', name: 'Red', color: '#c00' }];

/** A world that compresses like a real one: many hexes, few distinct keys. */
const world = (hexes = 4000): unknown => ({
  version: '1.8',
  seed: 1,
  width: hexes,
  height: 1,
  layout: 'axial',
  hexes: Array.from({ length: hexes }, (_, q) => ({
    q,
    r: 0,
    elevation: 100.5,
    slope: 1.25,
    relief: 0.5,
    terrain_class: 'land',
    land_cover: 'open',
    biome: 'temperate_grassland',
    river_flow: 0,
    catchment_km2: 0,
    tags: [],
    road_connections: [],
  })),
  rivers: [],
  settlements: [],
  road_edges: [],
  sea_edges: [],
  ferries: [],
});

const storedType = (db: Db): string =>
  (db.prepare(`SELECT typeof(blob) AS t FROM worlds LIMIT 1`).get() as { t: string }).t;

describe('the codec', () => {
  it('round-trips', () => {
    const json = JSON.stringify(world(200));
    expect(unpackWorld(packWorld(json))).toBe(json);
  });

  it('reads plain JSON from before worlds were compressed', () => {
    // The type is the discriminator, and this is the case it exists for.
    expect(unpackWorld('{"hexes":[]}')).toBe('{"hexes":[]}');
  });

  it('actually compresses, and substantially', () => {
    const json = JSON.stringify(world());
    expect(packWorld(json).length).toBeLessThan(json.length / 5);
  });
});

describe('a world written today', () => {
  it('is stored as a blob and read back whole', () => {
    const db = openDb();
    const store = new CampaignStore(db);
    const doc = world(500);
    store.create({ id: 'c1', name: 'c', worldDoc: doc, factions: FACTIONS });

    expect(storedType(db)).toBe('blob');
    expect(store.campaign('c1')!.world.hexes.size).toBe(500);
  });

  it('takes far less disk than the JSON it came from', () => {
    const path = join(mkdtempSync(join(tmpdir(), 'campaign-gzip-')), 'c.db');
    const store = new CampaignStore(openDb(path));
    const doc = world();
    store.create({ id: 'c1', name: 'c', worldDoc: doc, factions: FACTIONS });

    const json = JSON.stringify(doc).length;
    // Loose, because a database is more than its worlds — page overhead, the event log and
    // the referee's role row are all in here too. A file larger than the raw JSON would
    // mean the compression is simply not happening.
    expect(statSync(path).size).toBeLessThan(json / 2);
  });
});

describe('a database written before worlds were compressed', () => {
  /** `worlds` as it was first introduced: plain JSON in a TEXT column. */
  const legacy = (): string => {
    const path = join(mkdtempSync(join(tmpdir(), 'campaign-gzip-')), 'old.db');
    const db = new DatabaseSync(path);
    db.exec(`CREATE TABLE worlds (hash TEXT PRIMARY KEY, blob TEXT NOT NULL)`);
    db.exec(`INSERT INTO worlds VALUES ('hash-a', '{"hexes":[1]}')`);
    db.close();
    return path;
  };

  it('is compressed on the way in, and still reads the same', () => {
    const db = openDb(legacy());
    expect(storedType(db)).toBe('blob');
    const row = db.prepare(`SELECT blob FROM worlds`).get() as { blob: Uint8Array };
    expect(unpackWorld(row.blob)).toBe('{"hexes":[1]}');
  });

  it('runs once and leaves an already-compressed database alone', () => {
    const path = legacy();
    const first = openDb(path);
    const before = first.prepare(`SELECT blob FROM worlds`).get() as { blob: Uint8Array };

    const again = openDb(path);
    const after = again.prepare(`SELECT blob FROM worlds`).get() as { blob: Uint8Array };
    // Gzipping a gzip would still round-trip through `unpackWorld` once and return
    // garbage, so this is the assertion that a second pass did not happen.
    expect(Buffer.from(after.blob).equals(Buffer.from(before.blob))).toBe(true);
    expect(unpackWorld(after.blob)).toBe('{"hexes":[1]}');
  });
});
