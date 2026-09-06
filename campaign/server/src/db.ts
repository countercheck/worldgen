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

const { DatabaseSync } = nodeRequire('node:sqlite') as {
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
CREATE TABLE IF NOT EXISTS campaigns (
  id            TEXT PRIMARY KEY,
  name          TEXT NOT NULL,
  world_hash    TEXT NOT NULL,
  world_blob    TEXT NOT NULL,
  strictness    TEXT NOT NULL DEFAULT 'strict',
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

-- The join token is stored hashed. A database that leaks should not hand out the ability
-- to play, and nothing ever needs the token back: it arrives in a request and is hashed
-- to be compared.
CREATE TABLE IF NOT EXISTS roles (
  campaign_id   TEXT NOT NULL,
  token_hash    TEXT NOT NULL,
  role_kind     TEXT NOT NULL,
  faction_id    TEXT,
  PRIMARY KEY (campaign_id, token_hash)
);

CREATE INDEX IF NOT EXISTS roles_by_hash ON roles (token_hash);
`;

export function openDb(path = ':memory:'): Db {
  const db = new DatabaseSync(path);
  db.exec('PRAGMA journal_mode = WAL');
  db.exec('PRAGMA foreign_keys = ON');
  db.exec(SCHEMA);
  return db;
}
