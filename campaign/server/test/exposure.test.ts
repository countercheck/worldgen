/**
 * The edges that only matter once this is reachable from somewhere else.
 *
 * Everything here is invisible on a laptop: a world small enough to upload, one client so
 * there is nothing to limit, and no proxy in front so the connecting address is the real
 * one. All three change the moment a referee hosts this, and each of them fails quietly
 * rather than loudly — a 413 on the one upload that matters, a rate limit that buckets
 * every player together, a budget spent by the first arrival.
 */

import { readFileSync } from 'node:fs';

import { describe, expect, it } from 'vitest';

import { buildApp, DEFAULT_BODY_LIMIT, DEFAULT_RATE_LIMITS } from '../src/app.js';
import { openDb } from '../src/db.js';

/** A body of roughly `mb` megabytes, as a world document would be. */
const worldOf = (mb: number) => ({ pad: 'x'.repeat(Math.round(mb * 1_000_000)) });

const FACTIONS = [{ id: 'red', name: 'Red', color: '#c00' }];

describe('the size of a world', () => {
  it('accepts one the size the documentation tells a referee to generate', async () => {
    // The 32x32 fixture is 641 KB on disk and `docs/CAMPAIGN.md` recommends 64x64, which
    // is four times the hexes. Fastify's own default is 1 MiB, so before this was set the
    // ordinary case was refused with a 413 and nothing said why.
    const app = await buildApp({ db: openDb() });
    const res = await app.inject({
      method: 'POST',
      url: '/api/campaigns',
      payload: { name: 'c', world: worldOf(2.5), factions: FACTIONS },
    });

    expect(res.statusCode).not.toBe(413);
    await app.close();
  });

  it('still refuses one that is not a world at all', async () => {
    // A bound, not an absence of one. The limit is configurable so that a referee with a
    // 256x256 map can raise it deliberately rather than discovering it at upload time.
    const app = await buildApp({ db: openDb(), bodyLimit: 1024 });
    const res = await app.inject({
      method: 'POST',
      url: '/api/campaigns',
      payload: { name: 'c', world: worldOf(1), factions: FACTIONS },
    });

    expect(res.statusCode).toBe(413);
    await app.close();
  });

  it('defaults to a limit that clears a 128x128 world', () => {
    expect(DEFAULT_BODY_LIMIT).toBeGreaterThan(10 * 1024 * 1024);
  });
});

describe('rate limiting', () => {
  it('is off unless asked for, so tests and development are not on a budget', async () => {
    const app = await buildApp({ db: openDb() });
    for (let i = 0; i < DEFAULT_RATE_LIMITS.create + 5; i++) {
      const res = await app.inject({ method: 'GET', url: '/health' });
      expect(res.statusCode, `request ${i + 1}`).toBe(200);
    }
    await app.close();
  });

  it('spends a creation budget and then refuses, with the header saying so', async () => {
    const app = await buildApp({ db: openDb(), rateLimit: { global: 100, create: 2 } });
    const post = () =>
      app.inject({
        method: 'POST',
        url: '/api/campaigns',
        payload: { name: 'c', world: { hexes: [] }, factions: FACTIONS },
      });

    // The two allowed ones are answered on their merits, whatever those are — a rejected
    // world is still a request that was let through to be rejected.
    for (let i = 0; i < 2; i++) expect((await post()).statusCode).not.toBe(429);

    const refused = await post();
    expect(refused.statusCode).toBe(429);
    expect(refused.headers['x-ratelimit-limit']).toBe('2');

    await app.close();
  });

  it('keeps the tight budget to creation, which is the only unauthenticated write', async () => {
    // Reading `/health` is not creating a campaign, and a referee whose players are all
    // watching one evening should not find the console rationed because of it.
    const app = await buildApp({ db: openDb(), rateLimit: { global: 100, create: 2 } });
    for (let i = 0; i < 20; i++) {
      expect((await app.inject({ method: 'GET', url: '/health' })).statusCode).toBe(200);
    }
    await app.close();
  });
});

describe('trusting a proxy', () => {
  it('does not, by default', async () => {
    // Without this a client sets its own address by sending a header, which on a rate
    // limited server means setting its own budget too.
    const app = await buildApp({ db: openDb(), rateLimit: { global: 2, create: 2 } });
    const spoofed = () =>
      app.inject({
        method: 'GET',
        url: '/health',
        headers: { 'x-forwarded-for': `10.0.0.${Math.floor(Math.random() * 250)}` },
      });

    await spoofed();
    await spoofed();
    expect((await spoofed()).statusCode).toBe(429);
    await app.close();
  });

  it('does when told to, so a proxied player is limited as himself', async () => {
    const app = await buildApp({
      db: openDb(),
      trustProxy: true,
      rateLimit: { global: 2, create: 2 },
    });
    const from = (ip: string) =>
      app.inject({ method: 'GET', url: '/health', headers: { 'x-forwarded-for': ip } });

    await from('10.0.0.1');
    await from('10.0.0.1');
    expect((await from('10.0.0.1')).statusCode).toBe(429);
    // A different player, behind the same proxy, still has his own allowance.
    expect((await from('10.0.0.2')).statusCode).toBe(200);
    await app.close();
  });
});

describe('compressing what is sent', () => {
  const FACTIONS = [{ id: 'red', name: 'Red', color: '#c00' }];

  /** The real generated fixture, so this is measured against a world rather than a shape. */
  const world = (): unknown =>
    JSON.parse(
      readFileSync(
        new URL('../../shared/test/fixtures/world-32x32.json', import.meta.url),
        'utf8',
      ),
    );

  it('deflates a view, because a map is the most compressible thing here', async () => {
    const app = await buildApp({ db: openDb() });
    const created = await app.inject({
      method: 'POST',
      url: '/api/campaigns',
      payload: { name: 'c', world: world(), factions: FACTIONS },
    });
    expect(created.statusCode, created.body.slice(0, 200)).toBe(201);
    const { id, refereeToken } = created.json() as { id: string; refereeToken: string };

    const plain = await app.inject({
      method: 'GET',
      url: `/api/campaigns/${id}/view`,
      headers: { 'x-campaign-token': refereeToken },
    });
    const zipped = await app.inject({
      method: 'GET',
      url: `/api/campaigns/${id}/view`,
      headers: { 'x-campaign-token': refereeToken, 'accept-encoding': 'gzip' },
    });

    expect(zipped.headers['content-encoding']).toBe('gzip');
    // Thousands of hexes carrying the same dozen keys: the ratio here is not marginal.
    expect(zipped.rawPayload.length).toBeLessThan(plain.rawPayload.length / 4);
    await app.close();
  });

  it('leaves a small reply alone, rather than framing 11 bytes', async () => {
    const app = await buildApp({ db: openDb() });
    const res = await app.inject({
      method: 'GET',
      url: '/health',
      headers: { 'accept-encoding': 'gzip' },
    });
    expect(res.headers['content-encoding']).toBeUndefined();
    await app.close();
  });
});

describe('one create at a time', () => {
  const FACTIONS = [{ id: 'red', name: 'Red', color: '#c00' }];
  const world = (): unknown =>
    JSON.parse(
      readFileSync(
        new URL('../../shared/test/fixtures/world-32x32.json', import.meta.url),
        'utf8',
      ),
    );

  it('serialises creations, so two worlds are never parsed at once', async () => {
    // The measurement behind this: one 200x200 create takes resident memory from 93 MB to
    // 348 MB, and three at once reach 677 MB. On a container with a fixed limit that is
    // not a slow request, it is the process dying and every socket with it.
    const app = await buildApp({ db: openDb() });

    let inFlight = 0;
    let peak = 0;
    app.addHook('preHandler', async (req) => {
      if (req.url !== '/api/campaigns') return;
      inFlight += 1;
      peak = Math.max(peak, inFlight);
      await new Promise((r) => setTimeout(r, 15));
      inFlight -= 1;
    });

    const results = await Promise.all(
      [1, 2, 3].map((n) =>
        app.inject({
          method: 'POST',
          url: '/api/campaigns',
          payload: { name: `c${n}`, world: world(), factions: FACTIONS },
        }),
      ),
    );

    expect(results.map((r) => r.statusCode)).toEqual([201, 201, 201]);
    expect(peak, 'two creates were in flight together').toBe(1);
    await app.close();
  });

  it('opens again after a create fails, rather than shutting for good', async () => {
    // A release reached only on the happy path is the classic form of this bug, and it
    // presents as creation hanging forever with nothing in the log.
    const app = await buildApp({ db: openDb() });

    const bad = await app.inject({
      method: 'POST',
      url: '/api/campaigns',
      payload: { name: 'no world here' },
    });
    expect(bad.statusCode).toBe(400);

    const good = await app.inject({
      method: 'POST',
      url: '/api/campaigns',
      payload: { name: 'c', world: world(), factions: FACTIONS },
    });
    expect(good.statusCode, 'the gate stayed shut after a failure').toBe(201);
    await app.close();
  });

  it('can be widened for a container with memory to spare', async () => {
    const app = await buildApp({ db: openDb(), createConcurrency: 3 });
    let inFlight = 0;
    let peak = 0;
    app.addHook('preHandler', async (req) => {
      if (req.url !== '/api/campaigns') return;
      inFlight += 1;
      peak = Math.max(peak, inFlight);
      await new Promise((r) => setTimeout(r, 15));
      inFlight -= 1;
    });

    await Promise.all(
      [1, 2, 3].map((n) =>
        app.inject({
          method: 'POST',
          url: '/api/campaigns',
          payload: { name: `c${n}`, world: world(), factions: FACTIONS },
        }),
      ),
    );
    expect(peak).toBe(3);
    await app.close();
  });
});
