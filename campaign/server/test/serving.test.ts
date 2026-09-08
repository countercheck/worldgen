/**
 * Serving the client from the same process.
 *
 * Development runs two processes and lets Vite proxy the API, because that is what gives
 * hot reloading. The container runs one, on one port, with no CORS story to have — and
 * that second mode is only exercised in a container, which means it is exactly the sort
 * of path that works on a laptop and is broken in the image nobody built yet.
 *
 * The interesting assertion is the last one: a mistyped API endpoint must not fall through
 * to the single-page app. A fetch that suddenly parses as HTML is one of the more annoying
 * things to diagnose from inside a browser.
 */

import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, describe, expect, it } from 'vitest';

import { buildApp } from '../src/app.js';
import { openDb } from '../src/db.js';

const INDEX = '<!doctype html><title>Campaign</title><div id="root"></div>';

function servedApp() {
  const dir = mkdtempSync(join(tmpdir(), 'campaign-client-'));
  writeFileSync(join(dir, 'index.html'), INDEX);
  writeFileSync(join(dir, 'app.js'), 'export const ok = true;\n');
  return buildApp({ db: openDb(), clientDir: dir });
}

let app: ReturnType<typeof buildApp> | null = null;
afterEach(async () => {
  await app?.close();
  app = null;
});

describe('with a client directory', () => {
  it('serves the app at the root', async () => {
    app = servedApp();
    const res = await app.inject({ method: 'GET', url: '/' });

    expect(res.statusCode).toBe(200);
    expect(res.body).toContain('id="root"');
    expect(res.headers['content-type']).toContain('text/html');
  });

  it('serves its assets', async () => {
    app = servedApp();
    const res = await app.inject({ method: 'GET', url: '/app.js' });
    expect(res.statusCode).toBe(200);
    expect(res.body).toContain('export const ok');
  });

  it('sends a deep link to the app rather than a 404', async () => {
    app = servedApp();
    // Join links are URL fragments, so this should never happen in practice — but a link
    // somebody retyped by hand should land on the application, not on an error.
    const res = await app.inject({ method: 'GET', url: '/somewhere/nobody/routed' });
    expect(res.statusCode).toBe(200);
    expect(res.body).toContain('id="root"');
  });

  it('answers a mistyped API path with JSON, not with the app', async () => {
    app = servedApp();
    const res = await app.inject({ method: 'GET', url: '/api/campaigns/nope/vieww' });

    expect(res.statusCode).toBe(404);
    expect(res.headers['content-type']).toContain('application/json');
    expect(res.json()).toEqual({ error: 'no such endpoint' });
  });

  it('still answers the health check', async () => {
    app = servedApp();
    const res = await app.inject({ method: 'GET', url: '/health' });
    expect(res.json()).toEqual({ ok: true });
  });
});

describe('without a client directory', () => {
  it('serves no application at all', async () => {
    app = buildApp({ db: openDb() });
    const res = await app.inject({ method: 'GET', url: '/' });
    // Development's shape: this process is an API and nothing else, and a root that
    // returned something would only ever be a stale copy of the real client.
    expect(res.statusCode).toBe(404);
  });

  it('still answers the health check', async () => {
    app = buildApp({ db: openDb() });
    expect((await app.inject({ method: 'GET', url: '/health' })).json()).toEqual({ ok: true });
  });
});
