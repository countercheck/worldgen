/**
 * Storage.
 *
 * SQLite through Node's built-in `node:sqlite`, which needs no native build step. That
 * matters more than it sounds: `better-sqlite3` is excellent but compiles on install, and
 * a referee self-hosting this on whatever machine they have should not meet node-gyp. The
 * module is still marked experimental upstream, so it may warn on startup and its API
 * could shift; the surface used here is `exec`, `prepare`, `run` and `all`, which is the
 * part least likely to move.
 *
 * The `events` table is append-only and is the campaign. Everything else — snapshots
 * especially — is cache that can be deleted and rebuilt by replaying it.
 */

import { createRequire } from 'node:module';
import { gunzipSync, gzipSync } from 'node:zlib';

/**
 * `node:sqlite` is loaded through `createRequire` rather than imported.
 *
 * Vite — which vitest builds on — carries its own list of Node builtins to leave alone,
 * and that list predates `node:sqlite`. A static import is therefore rewritten into a
 * search for a package called "sqlite" on disk, which does not exist, and the whole test
 * suite fails to load. A runtime `require` is not a specifier Vite can rewrite, so it
 * reaches Node untouched.
 *
 * Delete this the day Vite knows about the module; the static import is the better code.
 */
const nodeRequire = createRequire(import.meta.url);

interface Statement {
  run(...params: unknown[]): { changes: number; lastInsertRowid: number };
  get(...params: unknown[]): unknown;
  all(...params: unknown[]): unknown[];
}

export interface Db {
  exec(sql: string): void;
  prepare(sql: string): Statement;
  close(): void;
}

/**
 * The raw handle, exported so the one `node:sqlite` workaround above lives in one file.
 *
 * `openDb` is what everything should use. This is for the one caller that must open a
 * database *without* creating it — a backup that helpfully created an empty campaign and
 * then copied it would report success and lose the game.
 */
export const { DatabaseSync } = nodeRequire('node:sqlite') as {
  DatabaseSync: new (path: string) => Db;
};

/**
 * Schema.
 *
 * `events` has a composite primary key of (campaign, seq), which is what makes an
 * out-of-order or duplicated append fail loudly at the database rather than quietly
 * corrupting the log. Nothing in here is ever UPDATEd; a correction is a later event, and
 * a rewind is replaying a prefix.
 */
const SCHEMA = `
-- The ruleset and config columns are the campaign's own numbers, written once at creation.
-- Resolved and stored rather than resolved on read: a house rule edited next month must
-- not silently re-tune a game already in progress, and a campaign that cannot say what it
-- was played under cannot be reviewed afterwards.
CREATE TABLE IF NOT EXISTS campaigns (
  id            TEXT PRIMARY KEY,
  name          TEXT NOT NULL,
  world_hash    TEXT NOT NULL,
  world_blob    TEXT NOT NULL,
  strictness    TEXT NOT NULL DEFAULT 'strict',
  ruleset       TEXT NOT NULL DEFAULT 'standard',
  config_json   TEXT,
  created_at    TEXT NOT NULL
);

-- Worlds, stored once each and named by their own hash.
--
-- A world used to live in the campaign row, which meant a referee running five games on
-- one map stored that map five times. At 23 MB for a 200x200 that is the difference
-- between 225 MB and 23 GB across a thousand campaigns, and the hash was already being
-- computed and stored — the row knew the world's identity and simply did not act on it.
--
-- Content-addressed, so there is no version to get wrong: two identical documents are the
-- same row by construction, and a regenerated world with so much as a different seed is a
-- different hash and a different row.
--
-- Nothing reaps these. Campaigns are never deleted, so a world is never orphaned; add
-- reference counting on the day that stops being true.
-- The blob is gzipped JSON. A world is tens of thousands of objects carrying the same
-- dozen keys, which is about as compressible as text gets: 23 MB becomes 4.5 MB.
--
-- Declared BLOB, and a database written before this was declared TEXT. That does not
-- matter and no table needs rebuilding: SQLite stores a blob as a blob whatever the
-- column's affinity says, and unpackWorld reads a string as the plain JSON it is.
CREATE TABLE IF NOT EXISTS worlds (
  hash          TEXT PRIMARY KEY,
  blob          BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
  campaign_id   TEXT NOT NULL,
  seq           INTEGER NOT NULL,
  clock_hours   REAL NOT NULL,
  kind          TEXT NOT NULL,
  payload_json  TEXT NOT NULL,
  actor_json    TEXT NOT NULL,
  forced        INTEGER NOT NULL,
  strictness    TEXT NOT NULL,
  bypassed_json TEXT NOT NULL,
  created_at    TEXT NOT NULL,
  PRIMARY KEY (campaign_id, seq)
);

CREATE TABLE IF NOT EXISTS snapshots (
  campaign_id   TEXT NOT NULL,
  seq           INTEGER NOT NULL,
  state_json    TEXT NOT NULL,
  PRIMARY KEY (campaign_id, seq)
);

-- One row per join link. A token is stored hashed: a database that leaks should not hand
-- out the ability to play, and nothing ever needs the token back — it arrives in a request
-- and is hashed to be compared.
--
-- A link names a commander, not a side. Two commanders on the same side see different wars, which
-- is the point of the whole design, so a per-faction token could not express who is asking.
-- Commanders cannot exist until there are units for them to ride with, so these rows are
-- written after creation rather than during it.
CREATE TABLE IF NOT EXISTS roles (
  campaign_id   TEXT NOT NULL,
  token_hash    TEXT NOT NULL,
  role_kind     TEXT NOT NULL,
  commander_id  TEXT,
  PRIMARY KEY (campaign_id, token_hash)
);

CREATE INDEX IF NOT EXISTS roles_by_hash ON roles (token_hash);
`;

/**
 * Columns added after the first databases were written.
 *
 * `CREATE TABLE IF NOT EXISTS` does nothing to a table that already exists, so a new
 * column needs saying twice: once in the schema for a fresh database and once here for
 * every database already on disk. Each is attempted and its failure ignored, because the
 * only expected failure is "duplicate column name" — which means the work is already done.
 */
const ADDED_COLUMNS: readonly string[] = [
  `ALTER TABLE campaigns ADD COLUMN ruleset TEXT NOT NULL DEFAULT 'standard'`,
  `ALTER TABLE campaigns ADD COLUMN config_json TEXT`,
];

/**
 * A world on its way to disk.
 *
 * Level 6 rather than 9: on a 23 MB world the last three levels buy about two per cent for
 * several times the CPU, and this runs while a referee waits on the one request that
 * already takes the longest.
 */
export const packWorld = (json: string): Uint8Array => gzipSync(Buffer.from(json), { level: 6 });

/**
 * A world on its way back.
 *
 * A string is plain JSON from a database written before worlds were compressed. The type
 * is the discriminator and needs no magic bytes: `node:sqlite` hands back a `Uint8Array`
 * for a blob and a `string` for text, and nothing else can appear in this column.
 */
export const unpackWorld = (stored: string | Uint8Array): string =>
  typeof stored === 'string' ? stored : gunzipSync(Buffer.from(stored)).toString('utf8');

/**
 * Compress any world still stored as plain JSON.
 *
 * Separate from `moveWorldsOut` and run after it, so each migration does one thing and
 * either can be read without the other in mind. The cost is that a database old enough to
 * need both writes its worlds twice, once on a migration that runs once.
 *
 * Idempotent: a row already stored as a blob is left alone, so this is a no-op on every
 * start after the first.
 */
function compressWorlds(db: Db): void {
  const plain = db
    .prepare(`SELECT hash, blob FROM worlds WHERE typeof(blob) = 'text'`)
    .all() as { hash: string; blob: string }[];
  if (plain.length === 0) return;

  db.exec('BEGIN IMMEDIATE');
  try {
    const update = db.prepare(`UPDATE worlds SET blob = ? WHERE hash = ?`);
    for (const row of plain) update.run(packWorld(row.blob), row.hash);
    db.exec('COMMIT');
  } catch (err) {
    db.exec('ROLLBACK');
    throw err;
  }
}

/**
 * Move each campaign's world into `worlds`, then stop storing it on the campaign.
 *
 * The one migration in this file that is not an `ALTER TABLE ADD COLUMN`, because it moves
 * data rather than making room for some. It runs inside a transaction and drops the old
 * column only after every campaign has been shown to have a world row to read instead: a
 * crash halfway leaves the database exactly as it was, which for the only copy of somebody
 * else's campaign is the only acceptable behaviour.
 *
 * Idempotent by inspection rather than by a version table — the column is either there or
 * it is not, and that is the whole of the question.
 */
function moveWorldsOut(db: Db): void {
  const columns = db.prepare(`PRAGMA table_info(campaigns)`).all() as { name: string }[];
  if (!columns.some((c) => c.name === 'world_blob')) return;

  db.exec('BEGIN IMMEDIATE');
  try {
    // `OR IGNORE`, because the whole point is that many campaigns share one hash.
    db.exec(
      `INSERT OR IGNORE INTO worlds (hash, blob)
       SELECT world_hash, world_blob FROM campaigns WHERE world_blob <> ''`,
    );

    // Checked before anything is dropped. A campaign left without a world would open
    // cleanly and fail on the first request, which is the worst time to find out.
    const { stranded } = db
      .prepare(
        `SELECT COUNT(*) AS stranded FROM campaigns c
         LEFT JOIN worlds w ON w.hash = c.world_hash
         WHERE w.hash IS NULL`,
      )
      .get() as { stranded: number };
    if (stranded > 0) {
      throw new Error(`${stranded} campaigns would be left without a world; not migrating`);
    }

    db.exec(`ALTER TABLE campaigns DROP COLUMN world_blob`);
    db.exec('COMMIT');
  } catch (err) {
    db.exec('ROLLBACK');
    throw err;
  }
}

export function openDb(path = ':memory:'): Db {
  const db = new DatabaseSync(path);
  db.exec('PRAGMA journal_mode = WAL');
  db.exec('PRAGMA foreign_keys = ON');
  db.exec(SCHEMA);
  for (const sql of ADDED_COLUMNS) {
    try {
      db.exec(sql);
    } catch {
      // Already there. Adding a column was for a long time the only migration this schema
      // needed, and re-running it is how a database written by an older build catches up.
    }
  }
  moveWorldsOut(db);
  compressWorlds(db);
  return db;
}
