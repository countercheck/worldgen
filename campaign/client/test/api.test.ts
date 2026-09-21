/**
 * Compressing the one request that is worth compressing.
 *
 * A create carries a generated world; nothing else the console sends comes close. A
 * 200x200 map is 32 MB, which on a domestic upstream is most of a minute, and 4.6 MB
 * gzipped. Everything here is about that one request — the rest of the wire is orders and
 * despatches measured in hundreds of bytes, where a gzip header would cost more than it
 * saved.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';

import { createCampaign } from '../src/api.js';

const FACTIONS = [{ id: 'red', name: 'Red', color: '#c00' }];

/** Capture what `fetch` was handed, and answer as the server would. */
function captureFetch() {
  const calls: { url: string; init: RequestInit }[] = [];
  vi.stubGlobal('fetch', async (url: string, init: RequestInit) => {
    calls.push({ url, init });
    return new Response(JSON.stringify({ id: 'c1', refereeToken: 't' }), {
      status: 201,
      headers: { 'content-type': 'application/json' },
    });
  });
  return calls;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('creating a campaign', () => {
  it('sends the world gzipped, and says so', async () => {
    const calls = captureFetch();
    await createCampaign({ name: 'c', world: { hexes: [] }, factions: FACTIONS });

    const [call] = calls;
    expect(call).toBeDefined();
    const headers = call!.init.headers as Record<string, string>;
    expect(headers['content-encoding']).toBe('gzip');
    expect(call!.init.body).toBeInstanceOf(Blob);
  });

  it('actually compresses, and to the same bytes back', async () => {
    const calls = captureFetch();
    // Repetitive like a real world: forty thousand hexes with the same dozen keys is why
    // this is worth doing at all.
    const world = { hexes: Array.from({ length: 2000 }, (_, q) => ({ q, r: 0, biome: 'x' })) };
    await createCampaign({ name: 'c', world, factions: FACTIONS });

    const body = calls[0]!.init.body as Blob;
    const plain = JSON.stringify({ name: 'c', world, factions: FACTIONS });
    expect(body.size).toBeLessThan(plain.length / 4);

    // Round-trips: what the server inflates must be what we meant to send.
    const back = await new Response(
      body.stream().pipeThrough(new DecompressionStream('gzip')),
    ).text();
    expect(JSON.parse(back)).toEqual({ name: 'c', world, factions: FACTIONS });
  });

  it('falls back to plain JSON where the browser cannot compress', async () => {
    // Not every context has `CompressionStream` — an older browser, or a page not served
    // over a secure context. Compression is an optimisation, never the point, and a
    // referee in that position must still be able to start a campaign.
    const calls = captureFetch();
    vi.stubGlobal('CompressionStream', undefined);

    await createCampaign({ name: 'c', world: { hexes: [] }, factions: FACTIONS });

    const headers = (calls[0]!.init.headers ?? {}) as Record<string, string>;
    expect(headers['content-encoding']).toBeUndefined();
    expect(typeof calls[0]!.init.body).toBe('string');
  });
});
