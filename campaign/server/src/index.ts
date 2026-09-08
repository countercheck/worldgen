/**
 * Entry point.
 *
 *     CAMPAIGN_DB=./campaign.db node dist/index.js
 *
 * One process, one SQLite file. That is the whole deployment: a referee running a game
 * for five people does not need anything else, and a database that is a file is a
 * database a referee can back up by copying it.
 *
 * `CAMPAIGN_CLIENT` points at the built client, and when it is set this process serves
 * the whole application on one port. The container does that; development does not,
 * because Vite serving the client itself is what gives hot reloading.
 */

import { buildApp } from './app.js';
import { openDb } from './db.js';

const port = Number(process.env.PORT ?? 3000);
// Loopback by default: this is a game among people who know each other, and a server that
// binds every interface the moment it starts is not a default anybody chose. The container
// sets 0.0.0.0 explicitly, because inside a container that is the only useful address.
const host = process.env.HOST ?? '127.0.0.1';
const dbPath = process.env.CAMPAIGN_DB ?? './campaign.db';
const clientDir = process.env.CAMPAIGN_CLIENT;

const app = buildApp({
  db: openDb(dbPath),
  logger: true,
  ...(clientDir === undefined ? {} : { clientDir }),
});

try {
  await app.listen({ port, host });
  app.log.info(
    `campaign server on http://${host}:${port} using ${dbPath}` +
      (clientDir === undefined ? '' : `, serving the client from ${clientDir}`),
  );
} catch (err) {
  app.log.error(err);
  process.exit(1);
}
