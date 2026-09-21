/**
 * Take a consistent copy of a campaign.
 *
 *     node server/dist/backup.js /data/campaign.db /data/backups/2026-09-20.db
 *
 * The documentation says a referee backs a campaign up by copying a file, and that is
 * true of the campaign but not quite true of the file. The database runs in WAL mode —
 * `db.ts` sets it, because it is what lets a referee advance the clock while five people
 * are reading — and in WAL mode the committed state is spread across `campaign.db` and
 * `campaign.db-wal`. Copying the first while the server is running gets whatever was last
 * checkpointed: a campaign missing its most recent evening, which looks like a valid
 * database and so is not noticed until it is restored.
 *
 * `VACUUM INTO` is SQLite's own answer. It writes a new database holding the committed
 * state as of one point in time, from inside the engine, with no cooperation needed from
 * the running server and no pause in play. The result is a single file with no sidecars —
 * which is what a referee thought they were copying in the first place.
 */

import { existsSync } from 'node:fs';

// Not `node:sqlite` directly: see the note in `db.ts` about Vite's builtin list.
import { DatabaseSync } from './db.js';

/**
 * Write a snapshot of the database at `source` to `destination`.
 *
 * SQLite refuses to overwrite, so `destination` must not exist: a backup that quietly
 * replaced the last one would turn a typo into the loss of both.
 */
export function backup(source: string, destination: string): void {
  // Checked rather than trusted. `node:sqlite` creates a database it cannot find, so a
  // mistyped source would otherwise produce an empty one, copy it, and report success.
  if (!existsSync(source)) throw new Error(`no database at ${source}`);

  // Read-write rather than read-only, because `VACUUM INTO` on a WAL database may need to
  // take a read lock the engine records, and a truly read-only handle cannot.
  const db = new DatabaseSync(source);
  try {
    // The path is a SQL string literal here, so a quote in it would end the literal. That
    // is an escaping bug rather than an injection one — nobody but the operator names
    // this file — but a campaign directory with an apostrophe in it is not exotic.
    db.exec(`VACUUM INTO '${destination.replace(/'/g, "''")}'`);
  } finally {
    db.close();
  }
}

// Run directly, rather than imported by a test.
if (process.argv[1]?.endsWith('backup.js') === true) {
  const [source, destination] = process.argv.slice(2);
  if (source === undefined || destination === undefined) {
    console.error('usage: node backup.js <source.db> <destination.db>');
    process.exit(2);
  }
  backup(source, destination);
  console.log(`wrote ${destination}`);
}
