/**
 * Entry point.
 *
 *     CAMPAIGN_DB=./campaign.db node dist/index.js
 *
 * One process, one SQLite file. That is the whole deployment: a referee running a game
 * for five people does not need anything else, and a database that is a file is a
 * database a referee can back up by copying it.
 */

import { buildApp } from './app.js';
import { openDb } from './db.js';

const port = Number(process.env.PORT ?? 3000);
const host = process.env.HOST ?? '127.0.0.1';
const dbPath = process.env.CAMPAIGN_DB ?? './campaign.db';

const app = buildApp({ db: openDb(dbPath), logger: true });

try {
  await app.listen({ port, host });
  app.log.info(`campaign server on http://${host}:${port} using ${dbPath}`);
} catch (err) {
  app.log.error(err);
  process.exit(1);
}
