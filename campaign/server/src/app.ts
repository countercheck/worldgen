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

import cookie from '@fastify/cookie';
import fastifyStatic from '@fastify/static';
import websocket from '@fastify/websocket';
import Fastify, { type FastifyInstance, type FastifyRequest } from 'fastify';

import {
  assertMasked,
  DEFAULT_CONFIG,
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
}

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

export function buildApp(opts: AppOptions = {}): FastifyInstance {
  const db = opts.db ?? openDb();
  const cfg = opts.cfg ?? DEFAULT_CONFIG;
  const store = new CampaignStore(db, cfg);

  const app = Fastify({ logger: opts.logger ?? false });
  app.register(cookie);
  app.register(websocket);

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

  app.post('/api/campaigns', async (req, reply) => {
    const body = req.body as {
      id?: string;
      name?: string;
      world?: unknown;
      factions?: { id: string; name: string; color: string }[];
      seed?: number;
      strictness?: Strictness;
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
      });

      // The only time this token exists in plaintext anywhere. It is not stored and
      // cannot be recovered — a lost link is reissued, not looked up.
      //
      // No commander links yet: a link names a seat, and there are no seats until the
      // referee has put formations on the map and appointed men to them.
      return reply.code(201).send({ id, refereeToken: created.refereeToken });
    } catch (err) {
      return reply.code(400).send({ error: String((err as Error).message ?? err) });
    }
  });

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
        cfg,
      },
      auth.role,
    );

    // Checked at the boundary as well as guaranteed by construction. A masking bug is
    // exactly the kind that ships quietly, so it fails the request rather than the game.
    assertMasked(view, cfg);
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
        cfg,
      },
      role,
    );
    assertMasked(view, cfg);

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
    const halts = new Set(cfg.haltTriggers);
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

    // A commander sees his own actions and nothing else. The log is a rich source of
    // exactly the information the fog exists to withhold — an enemy's marches are all in
    // there, and so are his own subordinates' — so it is filtered rather than trimmed.
    const mine = auth.role.id;
    return reply.send(
      events.filter((e) => e.actor.kind === 'commander' && e.actor.id === mine),
    );
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
        { campaignId, state, worldDoc: campaign.worldDoc, world: campaign.world, cfg },
        listener.role,
      );
      assertMasked(view, cfg);
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
          cfg,
        },
        role,
      );
      assertMasked(view, cfg);
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
