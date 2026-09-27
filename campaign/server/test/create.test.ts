/**
 * Starting a campaign with no sides, and adding them afterwards.
 *
 * A campaign used to arrive with two stock sides the referee had not chosen, red and blue,
 * and the only way to be rid of them was to start again. Now it arrives empty, and each
 * side is added by hand, named and coloured, over the ordinary command endpoint.
 */

import { readFileSync } from 'node:fs';

import { describe, expect, it } from 'vitest';

import { buildApp } from '../src/app.js';
import { openDb } from '../src/db.js';

const world = (): unknown =>
  JSON.parse(
    readFileSync(new URL('../../shared/test/fixtures/world-32x32.json', import.meta.url), 'utf8'),
  );

describe('creating a campaign', () => {
  it('starts with no sides when none are given', async () => {
    const app = await buildApp({ db: openDb() });
    const created = await app.inject({
      method: 'POST',
      url: '/api/campaigns',
      payload: { name: 'c', world: world() },
    });
    expect(created.statusCode, created.body.slice(0, 200)).toBe(201);
    const { id, refereeToken } = created.json() as { id: string; refereeToken: string };

    const view = await app.inject({
      method: 'GET',
      url: `/api/campaigns/${id}/view`,
      headers: { 'x-campaign-token': refereeToken },
    });
    expect((view.json() as { factions: unknown[] }).factions).toEqual([]);
    await app.close();
  });

  it('takes a side the referee adds afterwards', async () => {
    const app = await buildApp({ db: openDb() });
    const created = await app.inject({
      method: 'POST',
      url: '/api/campaigns',
      payload: { name: 'c', world: world() },
    });
    const { id, refereeToken } = created.json() as { id: string; refereeToken: string };

    const faction = { id: 'grande-armee', name: 'Grande Armée', color: '#1f4e9c' };
    const added = await app.inject({
      method: 'POST',
      url: `/api/campaigns/${id}/commands`,
      headers: { 'x-campaign-token': refereeToken },
      payload: { command: { kind: 'add_faction', faction } },
    });
    expect(added.statusCode, added.body).toBe(200);

    const view = await app.inject({
      method: 'GET',
      url: `/api/campaigns/${id}/view`,
      headers: { 'x-campaign-token': refereeToken },
    });
    expect((view.json() as { factions: unknown[] }).factions).toEqual([faction]);
    await app.close();
  });

  it('still refuses a create with no world', async () => {
    const app = await buildApp({ db: openDb() });
    const res = await app.inject({ method: 'POST', url: '/api/campaigns', payload: { name: 'c' } });
    expect(res.statusCode).toBe(400);
    await app.close();
  });
});
