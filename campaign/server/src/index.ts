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

import { buildApp, DEFAULT_BODY_LIMIT, DEFAULT_RATE_LIMITS } from './app.js';
import { openDb } from './db.js';

/**
 * `TRUST_PROXY=true` believes the forwarding chain; anything else is read as the
 * addresses or CIDR ranges to believe, comma separated.
 */
function trustSetting(v: string): boolean | string[] {
  if (v === 'true') return true;
  if (v === 'false') return false;
  return v.split(',').map((s) => s.trim()).filter((s) => s !== '');
}

const port = Number(process.env.PORT ?? 3000);
// Loopback by default: this is a game among people who know each other, and a server that
// binds every interface the moment it starts is not a default anybody chose. The container
// sets 0.0.0.0 explicitly, because inside a container that is the only useful address.
const host = process.env.HOST ?? '127.0.0.1';
const dbPath = process.env.CAMPAIGN_DB ?? './campaign.db';
const clientDir = process.env.CAMPAIGN_CLIENT;

// Behind a reverse proxy — a platform's router, or a Caddy in front — the connecting
// address is the proxy's, so every player would share one rate-limit bucket and the first
// one would spend it. Off unless asked, because trusting a forwarding header nobody put
// there lets any client claim any address. Set it to `1` behind exactly one proxy.
const trustProxy = process.env.TRUST_PROXY;

// Limiting is a property of being long-running, not of being built, so it is switched on
// here rather than in `buildApp`: the tests and the development server stand the same app
// up dozens of times a second and should not be fighting a budget to do it.
const rateLimit =
  process.env.CAMPAIGN_RATE_LIMIT === 'off'
    ? false
    : {
        global: Number(process.env.CAMPAIGN_RATE_GLOBAL ?? DEFAULT_RATE_LIMITS.global),
        create: Number(process.env.CAMPAIGN_RATE_CREATE ?? DEFAULT_RATE_LIMITS.create),
      };

const app = await buildApp({
  db: openDb(dbPath),
  logger: true,
  bodyLimit: Number(process.env.CAMPAIGN_BODY_LIMIT ?? DEFAULT_BODY_LIMIT),
  rateLimit,
  trustProxy: trustProxy === undefined ? false : trustSetting(trustProxy),
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
