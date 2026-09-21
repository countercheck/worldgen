/**
 * The API.
 *
 * Built as a factory so tests can stand up a server against an in-memory database
 * without a port or a process. Every route that returns campaign state calls `viewFor`
 * and nothing else — there is deliberately no route that serialises `CampaignState`
 * directly, because that is the single mistake that would end the game.
 *
 * Identity is a join token. A referee creates a campaign and receives one secret link per
 * faction plus their own; anyone holding a link plays that side. No accounts, nothing to
 * set up, and the `roles` table already has room for a user id when accounts are wanted.
 * Security here is "do not share your link", which is the right level for a refereed game
 * among people who know each other, and is stated plainly rather than dressed up.
 */

import compress from '@fastify/compress';
import cookie from '@fastify/cookie';
import rateLimit from '@fastify/rate-limit';
import fastifyStatic from '@fastify/static';
import websocket from '@fastify/websocket';
import Fastify, {
  type FastifyInstance,
  type FastifyRequest,
  type FastifyServerOptions,
} from 'fastify';

import {
  assertMasked,
  DEFAULT_CONFIG,
  type ConfigOverrides,
  inboxOf,
  logFor,
  viewFor,
  type CampaignConfig,
  type Command,
  type Role,
  type Strictness,
} from '@campaign/shared';

import { openDb, type Db } from './db.js';
import { CampaignStore, type CampaignRow } from './store.js';

export const TOKEN_COOKIE = 'campaign_token';

export interface AppOptions {
  readonly db?: Db;
  readonly cfg?: CampaignConfig;
  readonly logger?: boolean;
  /**
   * Where the built client lives, if this process is to serve it.
   *
   * Unset in development, where Vite serves the client on its own port and proxies the
   * API here — two processes, because that is what gives hot reloading. Set in a
   * container, where the whole application is one process on one port and there is no
   * CORS story to have.
   */
  readonly clientDir?: string;

  /**
   * The largest request body accepted, in bytes.
   *
   * This is a world upload limit and nothing else: a campaign is created by posting a
   * generated `world.json`, and that document is by far the biggest thing this server is
   * ever sent. Fastify's own default is 1 MiB, which is under the size of the world the
   * documentation tells a referee to generate — a 32x32 world is already 641 KB and the
   * recommended 64x64 is about four times that, so the default refused the ordinary case
   * with a 413.
   */
  readonly bodyLimit?: number;

  /**
   * Whether to believe `X-Forwarded-For`, and from whom.
   *
   * Off by default, because a server that trusts a forwarding header nobody put there is
   * letting any client claim any address. On behind a reverse proxy — a container
   * platform's router, or a Caddy in front — where without it every request appears to
   * come from the proxy and per-address rate limiting collapses into one bucket.
   *
   * `true` believes the chain, which is right where the platform is the only way in. A
   * string or list names the addresses or CIDR ranges to believe and nothing else, which
   * is better where the server is also reachable directly.
   */
  readonly trustProxy?: boolean | string | string[];

  /**
   * Whether to rate limit, and how hard.
   *
   * Off by default so tests and development are not fighting a budget, and set by the
   * entry point for anything long-running. The limit that matters is on creation: it is
   * the only unauthenticated write this server has, and each one puts a whole world on
   * disk. Everything else needs a token first.
   */
  readonly rateLimit?: { readonly global: number; readonly create: number } | false;
}

/**
 * Bounds the upload while clearing the real maps with room to spare.
 *
 * Measured, not guessed: a generated `world.json` is 3.1 MB at 64x64 and **32 MB at
 * 200x200**, which is the size this project needs to handle. The first figure here was
 * 16 MiB, chosen against an estimate of 2.5 MB for 64x64 and an assumption that nothing
 * much larger was coming; a 200x200 upload met it as a 413.
 *
 * 64 MiB clears 200x200 with most of the room again to spare, and would take a 256x256
 * map at roughly 50 MB. It is still a bound: this is the only unauthenticated write, and
 * an unbounded one is a disk-fill waiting for whoever finds the URL.
 *
 * Note that the upload is not what the projection shrank. `projectWorld` governs what
 * leaves the server; what arrives is the generator's document, every field of it.
 */
export const DEFAULT_BODY_LIMIT = 64 * 1024 * 1024;

/** Per minute, per address. Generous, because a referee mid-evening is not an attacker. */
export const DEFAULT_RATE_LIMITS = { global: 600, create: 5 } as const;

/** Where a request's token may come from, in order of preference. */
function tokenFrom(req: FastifyRequest): string | undefined {
  const header = req.headers['x-campaign-token'];
  if (typeof header === 'string' && header !== '') return header;

  const auth = req.headers.authorization;
  if (typeof auth === 'string' && auth.startsWith('Bearer ')) return auth.slice(7);

  const query = (req.query as { token?: string } | undefined)?.token;
  if (typeof query === 'string' && query !== '') return query;

  return req.cookies[TOKEN_COOKIE];
}

/**
 * Asynchronous because plugins are.
 *
 * `app.register` defers: the plugin is loaded when the instance is readied, in a child
 * context, and a route added synchronously afterwards is in the parent and never sees its
 * hooks. For a decorator that is invisible — `@fastify/cookie` hoists itself — but a rate
 * limiter registered that way loads, reports itself loaded, and limits nothing at all.
 * Awaiting each registration puts the hooks where the routes are.
 */
export async function buildApp(opts: AppOptions = {}): Promise<FastifyInstance> {
  const db = opts.db ?? openDb();
  // A fallback only. Every request resolves the campaign's own numbers from the row, so
  // that two campaigns in one database can be played under different rules.
  const cfg = opts.cfg ?? DEFAULT_CONFIG;
  const store = new CampaignStore(db, cfg);

  /** The numbers this campaign runs on. */
  const configOf = (campaign: { config?: CampaignConfig }): CampaignConfig =>
    campaign.config ?? cfg;

  const limits = opts.rateLimit ?? false;

  // Annotated rather than passed inline. Fastify chooses between its plain-HTTP and
  // HTTP/2 overloads by reading the options literal, and an object carrying `trustProxy`
  // resolves to the HTTP/2 one — which then types every request as an
  // `Http2ServerRequest` and fails against every helper here.
  const serverOptions: FastifyServerOptions = {
    logger: opts.logger ?? false,
    bodyLimit: opts.bodyLimit ?? DEFAULT_BODY_LIMIT,
    trustProxy: opts.trustProxy ?? false,
  };

  const app = Fastify(serverOptions);
  await app.register(cookie);
  // `@fastify/compress` covers HTTP replies and nothing else — a WebSocket frame never
  // passes through the reply pipeline. The push is where the repeated cost is: a view goes
  // out to every connected commander on every clock tick, so an uncompressed socket sends
  // the map again in full each time the hour advances.
  //
  // `permessage-deflate` is the protocol's own answer and the browser negotiates it. The
  // threshold matches the HTTP one, and `ws` keeps a compressor per connection — which is
  // memory worth watching at thousands of sockets and irrelevant at the five a referee's
  // evening actually has.
  await app.register(websocket, {
    options: { perMessageDeflate: { threshold: 1024 } },
  });

  // A world is the largest thing this server sends and the most compressible: tens of
  // thousands of hexes, each an object with the same dozen keys. On a 200x200 map the view
  // is 9 MB of JSON and 1.1 MB once deflated, and it is sent on every load and pushed to
  // every commander on every clock tick.
  //
  // `threshold` so that a 200-byte acknowledgement is not wrapped in a gzip header for no
  // reason. Encodings in preference order: brotli compresses this shape better than gzip
  // and every browser that will ever open this console supports it, but gzip stays for
  // `curl` and for anything speaking through a proxy that strips `br`.
  await app.register(compress, {
    global: true,
    threshold: 1024,
    encodings: ['br', 'gzip', 'deflate'],
  });
  if (limits !== false) {
    // Registered globally so an unauthenticated flood cannot reach a route handler at
    // all, and overridden per route where the cost of a request is not the same. The
    // allowance is per address, which is the reason `trustProxy` has to be right behind
    // a proxy: without it every player shares one bucket and the first one spends it.
    await app.register(rateLimit, { max: limits.global, timeWindow: '1 minute' });
  }

  app.decorate('store', store);

  /** Resolve the campaign and the caller's role, or answer for them. */
  const authorise = (
    req: FastifyRequest,
    reply: { code: (n: number) => { send: (b: unknown) => unknown } },
  ): { campaign: CampaignRow; role: Role } | null => {
    const { id } = req.params as { id: string };
    const campaign = store.campaign(id);
    if (campaign === null) {
      reply.code(404).send({ error: 'no such campaign' });
      return null;
    }
    const role = store.roleFor(id, tokenFrom(req));
    if (role === null) {
      // 404 rather than 403 on a bad token would be tidier about not confirming that a
      // campaign exists, but these links are shared by hand among people who know each
      // other, and a clear error is worth more than that much obscurity.
      reply.code(401).send({ error: 'a valid join token is required' });
      return null;
    }
    return { campaign, role };
  };

  // ---- create -----------------------------------------------------------

  app.post(
    '/api/campaigns',
    limits === false
      ? {}
      : { config: { rateLimit: { max: limits.create, timeWindow: '1 minute' } } },
    async (req, reply) => {
      const body = req.body as {
        id?: string;
        name?: string;
        world?: unknown;
        factions?: { id: string; name: string; color: string }[];
        seed?: number;
        strictness?: Strictness;
        ruleset?: string;
        config?: ConfigOverrides;
      };

      if (body?.world === undefined || !Array.isArray(body.factions) || body.factions.length === 0) {
        return reply.code(400).send({ error: 'a world and at least one faction are required' });
      }

      const id = body.id ?? `c${Date.now().toString(36)}`;
      try {
        const created = store.create({
          id,
          name: body.name ?? 'Campaign',
          worldDoc: body.world,
          factions: body.factions,
          ...(body.seed !== undefined ? { seed: body.seed } : {}),
          ...(body.strictness !== undefined ? { strictness: body.strictness } : {}),
          ...(body.ruleset !== undefined ? { ruleset: body.ruleset } : {}),
          ...(body.config !== undefined ? { config: body.config } : {}),
        });

        // The only time this token exists in plaintext anywhere. It is not stored and
        // cannot be recovered — a lost link is reissued, not looked up.
        //
        // No commander links yet: a link names a seat, and there are no seats until the
        // referee has put formations on the map and appointed men to them.
        return reply.code(201).send({
          id,
          refereeToken: created.refereeToken,
          ruleset: created.campaign.ruleset,
        });
      } catch (err) {
        return reply.code(400).send({ error: String((err as Error).message ?? err) });
      }
    },
  );

  // ---- the one read path ------------------------------------------------

  app.get('/api/campaigns/:id/view', async (req, reply) => {
    const auth = authorise(req, reply);
    if (auth === null) return;

    const view = viewFor(
      {
        campaignId: auth.campaign.id,
        state: store.state(auth.campaign.id),
        worldDoc: auth.campaign.worldDoc,
        world: auth.campaign.world,
        cfg: configOf(auth.campaign),
        ruleset: auth.campaign.ruleset,
      },
      auth.role,
    );

    // Checked at the boundary as well as guaranteed by construction. A masking bug is
    // exactly the kind that ships quietly, so it fails the request rather than the game.
    assertMasked(view, configOf(auth.campaign));
    return reply.send(view);
  });

  /** The masked world alone, as a file a referee can put through the Python tooling. */
  app.get('/api/campaigns/:id/export', async (req, reply) => {
    const auth = authorise(req, reply);
    if (auth === null) return;

    const asCommander = (req.query as { commander?: string }).commander;
    // A referee may look through any commander's eyes; a commander only through his own.
    const role: Role =
      auth.role.kind === 'referee' && asCommander !== undefined
        ? { kind: 'commander', id: asCommander }
        : auth.role;

    const view = viewFor(
      {
        campaignId: auth.campaign.id,
        state: store.state(auth.campaign.id),
        worldDoc: auth.campaign.worldDoc,
        world: auth.campaign.world,
        cfg: configOf(auth.campaign),
        ruleset: auth.campaign.ruleset,
      },
      role,
    );
    assertMasked(view, configOf(auth.campaign));

    return reply
      .header('content-type', 'application/json')
      .header(
        'content-disposition',
        `attachment; filename="${auth.campaign.id}-${view.commander?.id ?? 'truth'}.json"`,
      )
      .send(view.world);
  });

  // ---- seats ------------------------------------------------------------

  /**
   * Mint a join link for a commander's seat. Referee only, obviously.
   *
   * Returned once. The token is stored only as a hash, so a link that is lost is reissued
   * rather than recovered, and issuing a second one for the same seat is the supported
   * way to replace a link that went to the wrong person — followed by a revoke.
   */
  app.post('/api/campaigns/:id/commanders/:commanderId/token', async (req, reply) => {
    const auth = authorise(req, reply);
    if (auth === null) return;
    if (auth.role.kind !== 'referee') {
      return reply.code(403).send({ error: 'only the referee issues join links' });
    }

    const { commanderId } = req.params as { commanderId: string };
    if (!store.state(auth.campaign.id).commanders.has(commanderId)) {
      return reply.code(404).send({ error: `there is no commander ${commanderId}` });
    }

    const token = store.issueToken(auth.campaign.id, commanderId);
    return reply.code(201).send({
      commanderId,
      token,
      joinLink: `/j/${auth.campaign.id}/${token}`,
    });
  });

  /** Invalidate every link to a seat. */
  app.delete('/api/campaigns/:id/commanders/:commanderId/token', async (req, reply) => {
    const auth = authorise(req, reply);
    if (auth === null) return;
    if (auth.role.kind !== 'referee') {
      return reply.code(403).send({ error: 'only the referee revokes join links' });
    }
    const { commanderId } = req.params as { commanderId: string };
    store.revokeTokens(auth.campaign.id, commanderId);
    return reply.send({ ok: true });
  });

  // ---- commands ---------------------------------------------------------

  app.post('/api/campaigns/:id/commands', async (req, reply) => {
    const auth = authorise(req, reply);
    if (auth === null) return;

    const body = req.body as {
      command?: Command;
      force?: boolean;
      strictness?: Strictness;
    };
    if (body?.command === undefined) {
      return reply.code(400).send({ error: 'a command is required' });
    }

    // A commander may do exactly one thing: write. Everything else — tasks, the clock,
    // the order of battle — is the referee's, and rejecting it here rather than in
    // `check` keeps the engine free of any notion of who is connected.
    let command = body.command;
    if (auth.role.kind !== 'referee') {
      if (command.kind !== 'send_despatch') {
        return reply.code(403).send({ error: 'a commander may only send despatches' });
      }
      // The sender is who the token says he is, never who the payload claims. Trusting
      // `from` would let anyone holding any seat write in another man's name, which is
      // both forgery and — since a forged report would be believed — a way to feed the
      // enemy's commander false intelligence signed by his own subordinate.
      command = { ...command, from: auth.role.id };
    }

    const result = store.execute(auth.campaign, command, auth.role, {
      ...(body.force !== undefined ? { force: body.force } : {}),
      ...(body.strictness !== undefined ? { strictness: body.strictness } : {}),
    });

    if (!result.ok) {
      return reply.code(409).send({ ok: false, violations: result.violations });
    }

    broadcast(auth.campaign.id);
    return reply.send({
      ok: true,
      events: result.events.length,
      bypassed: result.events.flatMap((e) => e.bypassed),
    });
  });

  app.post('/api/campaigns/:id/advance', async (req, reply) => {
    const auth = authorise(req, reply);
    if (auth === null) return;
    if (auth.role.kind !== 'referee') {
      return reply.code(403).send({ error: 'only the referee advances the clock' });
    }

    const body = req.body as { hours?: number; untilDecision?: boolean };
    const hours = body?.hours ?? 1;
    const result = store.execute(
      auth.campaign,
      { kind: 'advance_clock', hours, untilDecision: body?.untilDecision ?? false },
      auth.role,
    );
    if (!result.ok) return reply.code(409).send({ ok: false, violations: result.violations });

    broadcast(auth.campaign.id);
    const state = store.state(auth.campaign.id);
    // What stopped the clock, if anything did. The referee's whole workflow is "run it
    // until something needs me", so the answer to *what* needs him belongs in the reply
    // rather than in a second request he has to know to make.
    const halts = new Set(configOf(auth.campaign).haltTriggers);
    const halted = result.events
      .map((e) => e.payload)
      .find((p) => p.kind === 'decision_raised' && halts.has(p.decision.trigger));

    return reply.send({
      ok: true,
      clockHours: state.clockHours,
      halted:
        body?.untilDecision === true && halted?.kind === 'decision_raised'
          ? halted.decision
          : null,
    });
  });

  // ---- log --------------------------------------------------------------

  app.get('/api/campaigns/:id/log', async (req, reply) => {
    const auth = authorise(req, reply);
    if (auth === null) return;

    const events = store.events(auth.campaign.id);
    if (auth.role.kind === 'referee') return reply.send(events);

    // A commander gets his outbox, rebuilt from the log rather than filtered out of it.
    // The raw events are a rich source of exactly what the fog exists to withhold — every
    // march in order, and every rider's route, which is a position — and filtering by
    // actor does not remove them: a cascaded order is stamped with the name of the man
    // who started the chain but carries a route to somebody else's subordinate. See
    // `logFor`, which builds by construction for the same reason `senderCopy` does.
    const mine = auth.role.id;
    const acknowledged = new Set(
      inboxOf(store.state(auth.campaign.id), mine)
        .filter((d) => d.kind === 'acknowledgement')
        .map((d) => d.inReplyTo),
    );
    return reply.send(logFor(events, mine, acknowledged));
  });

  // ---- live updates -----------------------------------------------------

  const sockets = new Map<string, Set<{ send: (s: string) => void; role: Role }>>();

  function broadcast(campaignId: string): void {
    const listeners = sockets.get(campaignId);
    if (listeners === undefined || listeners.size === 0) return;
    const campaign = store.campaign(campaignId);
    if (campaign === null) return;
    const state = store.state(campaignId);

    // Rebuilt per socket, because each is entitled to a different picture. Sending one
    // payload to everybody is how a fog-of-war server leaks.
    for (const listener of listeners) {
      const view = viewFor(
        {
          campaignId,
          state,
          worldDoc: campaign.worldDoc,
          world: campaign.world,
          cfg: configOf(campaign),
          ruleset: campaign.ruleset,
        },
        listener.role,
      );
      assertMasked(view, configOf(campaign));
      listener.send(JSON.stringify({ type: 'view', view }));
    }
  }

  app.register(async (scoped) => {
    scoped.get('/api/campaigns/:id/stream', { websocket: true }, (socket, req) => {
      const { id } = req.params as { id: string };
      const campaign = store.campaign(id);
      const role = campaign === null ? null : store.roleFor(id, tokenFrom(req));

      if (campaign === null || role === null) {
        socket.send(JSON.stringify({ type: 'error', error: 'unauthorised' }));
        socket.close();
        return;
      }

      const listener = { send: (s: string) => socket.send(s), role };
      const set = sockets.get(id) ?? new Set();
      set.add(listener);
      sockets.set(id, set);

      const view = viewFor(
        {
          campaignId: id,
          state: store.state(id),
          worldDoc: campaign.worldDoc,
          world: campaign.world,
          cfg: configOf(campaign),
          ruleset: campaign.ruleset,
        },
        role,
      );
      assertMasked(view, configOf(campaign));
      socket.send(JSON.stringify({ type: 'view', view }));

      socket.on('close', () => {
        set.delete(listener);
      });
    });
  });

  // ---- join -------------------------------------------------------------

  /** Exchange a join link for a cookie, so the client need not carry the token itself. */
  app.get('/j/:id/:token', async (req, reply) => {
    const { id, token } = req.params as { id: string; token: string };
    if (store.roleFor(id, token) === null) {
      return reply.code(401).send({ error: 'that link is not valid for this campaign' });
    }
    return reply
      .setCookie(TOKEN_COOKIE, token, {
        path: '/',
        httpOnly: true,
        sameSite: 'lax',
      })
      .send({ ok: true, campaign: id });
  });

  app.get('/health', async () => ({ ok: true }));

  // ---- the client -------------------------------------------------------

  if (opts.clientDir !== undefined) {
    app.register(fastifyStatic, { root: opts.clientDir, wildcard: false });

    // Anything not matched by a route above is the client's own. It routes on the URL
    // fragment — join links are `#/j/<campaign>/<token>`, deliberately, so a token never
    // reaches a server log — so in practice this only ever serves `/`, but a deep link
    // somebody typed should land on the app rather than on a 404.
    //
    // `/api` is excluded so a mistyped endpoint returns a JSON 404 rather than a page,
    // which is far easier to diagnose from a fetch that suddenly parses as HTML.
    app.setNotFoundHandler((req, reply) => {
      if (req.url.startsWith('/api')) {
        return reply.code(404).send({ error: 'no such endpoint' });
      }
      return reply.sendFile('index.html');
    });
  }

  return app;
}
