/**
 * The three addresses.
 *
 * Two properties are worth a test here and they pull against each other. A campaign must
 * have an address a player can bookmark and come back to — that is the whole point of
 * `#/c/<id>` — and a token must never end up in an address anybody would bookmark. So
 * these check both: that a campaign address carries no token, and that the one address
 * which does carry a token is a fragment the server never sees.
 */

import { beforeAll, describe, expect, it, vi } from 'vitest';

import { absolute, campaignHash, joinHash, parseRoute } from '../src/route.js';
import { joinLink } from '../src/session.js';

// Only an origin is read from the DOM, so a stub is enough — pulling in a whole browser
// environment to read one property would make this suite slower for no more assurance.
beforeAll(() => {
  vi.stubGlobal('location', { origin: 'https://campaign.test' });
});

describe('reading an address', () => {
  it('reads a campaign and a token out of a join link', () => {
    expect(parseRoute('#/j/c1/abc123')).toEqual({
      kind: 'join',
      campaignId: 'c1',
      token: 'abc123',
    });
  });

  it('decodes escaped segments, so a token is never truncated', () => {
    // Tokens are base64url and need no escaping, but a campaign name might, and half a
    // token silently becomes a wrong token rather than an error.
    expect(parseRoute('#/j/c%201/a-b_c')).toEqual({
      kind: 'join',
      campaignId: 'c 1',
      token: 'a-b_c',
    });
  });

  it('reads a campaign address, which carries no token at all', () => {
    expect(parseRoute('#/c/c1')).toEqual({ kind: 'campaign', campaignId: 'c1' });
    expect(parseRoute('#/c/c%201')).toEqual({ kind: 'campaign', campaignId: 'c 1' });
  });

  it('reads anything else as the front page', () => {
    // Including malformed links. Somebody who mistypes an address should land somewhere
    // they can act from, and the front page lists what their browser holds.
    for (const hash of ['', '#', '#/', '#/j/c1', '#/j/c1/tok/extra', '#/other/c1', '#j/c1/tok']) {
      expect(parseRoute(hash), `${hash} was not read as the front page`).toEqual({
        kind: 'home',
      });
    }
  });

  it('round-trips the addresses it produces', () => {
    // The only property that really matters: what is handed to a player must be readable
    // by the page they land on.
    expect(parseRoute(joinHash('c1', 'x-Y_z09'))).toEqual({
      kind: 'join',
      campaignId: 'c1',
      token: 'x-Y_z09',
    });
    expect(parseRoute(campaignHash('c 1'))).toEqual({ kind: 'campaign', campaignId: 'c 1' });
  });
});

describe('what an address may carry', () => {
  it('puts a join token after the fragment marker, never before it', () => {
    // If this ever produced a query parameter the token would reach the server on every
    // request and land in its logs, which is exactly what a fragment avoids.
    const link = joinLink({ campaignId: 'c1', token: 'secret-token' });
    expect(link.split('#')[0]).not.toContain('secret-token');
    expect(link).toContain('#/j/c1/secret-token');
  });

  it('keeps every token out of a campaign address', () => {
    // The bookmarkable one. A token here would end up in history, in bookmarks and in
    // the first screenshot anybody takes — the whole reason the campaign route exists
    // separately from the join route.
    const link = absolute(campaignHash('c1'));
    expect(link).toBe('https://campaign.test/#/c/c1');
    expect(link).not.toContain('secret');
    expect(parseRoute(campaignHash('c1'))).not.toHaveProperty('token');
  });
});
