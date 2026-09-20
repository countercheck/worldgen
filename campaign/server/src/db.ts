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
-- A link names a commander, not a side. Two men on the same side see different wars, which
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

export function openDb(path = ':memory:'): Db {
  const db = new DatabaseSync(path);
  db.exec('PRAGMA journal_mode = WAL');
  db.exec('PRAGMA foreign_keys = ON');
  db.exec(SCHEMA);
  for (const sql of ADDED_COLUMNS) {
    try {
      db.exec(sql);
    } catch {
      // Already there. Adding a column is the only migration this schema has ever needed,
      // and re-running it is how a database written by an older build catches up.
    }
  }
  return db;
}
