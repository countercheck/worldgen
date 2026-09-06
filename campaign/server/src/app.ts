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

      // The only time the tokens exist in plaintext anywhere. They are not stored and
      // cannot be recovered — a lost link is reissued, not looked up.
      return reply.code(201).send({
        id,
        refereeToken: created.refereeToken,
        factionTokens: created.tokens,
        joinLinks: Object.fromEntries(
          Object.entries(created.tokens).map(([f, t]) => [f, `/j/${id}/${t}`]),
        ),
      });
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
    assertMasked(view);
    return reply.send(view);
  });

  /** The masked world alone, as a file a referee can put through the Python tooling. */
  app.get('/api/campaigns/:id/export', async (req, reply) => {
    const auth = authorise(req, reply);
    if (auth === null) return;

    const asFaction = (req.query as { faction?: string }).faction;
    // A referee may ask for any faction's map; a commander may only have their own.
    const role: Role =
      auth.role.kind === 'referee' && asFaction !== undefined
        ? { kind: 'faction', id: asFaction }
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
    assertMasked(view);

    return reply
      .header('content-type', 'application/json')
      .header(
        'content-disposition',
        `attachment; filename="${auth.campaign.id}-${view.faction ?? 'truth'}.json"`,
      )
      .send(view.world);
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

    // Every command in this build is a referee command — the player-issued orders arrive
    // with the courier system. Rejecting here rather than in `check` keeps the engine
    // free of any notion of who is connected.
    if (auth.role.kind !== 'referee') {
      return reply.code(403).send({ error: 'only the referee may issue commands' });
    }

    const result = store.execute(auth.campaign, body.command, auth.role, {
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

    const hours = (req.body as { hours?: number })?.hours ?? 1;
    const result = store.execute(auth.campaign, { kind: 'advance_clock', hours }, auth.role);
    if (!result.ok) return reply.code(409).send({ ok: false, violations: result.violations });

    broadcast(auth.campaign.id);
    return reply.send({ ok: true, clockHours: store.state(auth.campaign.id).clockHours });
  });

  // ---- log --------------------------------------------------------------

  app.get('/api/campaigns/:id/log', async (req, reply) => {
    const auth = authorise(req, reply);
    if (auth === null) return;

    const events = store.events(auth.campaign.id);
    if (auth.role.kind === 'referee') return reply.send(events);

    // A commander sees their own actions and nothing else. The log is a rich source of
    // exactly the information the fog exists to withhold — an enemy's marches are all in
    // there — so it is filtered rather than trimmed.
    const mine = auth.role.id;
    return reply.send(
      events.filter((e) => e.actor.kind === 'faction' && e.actor.id === mine),
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
      assertMasked(view);
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
      assertMasked(view);
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

  return app;
}
