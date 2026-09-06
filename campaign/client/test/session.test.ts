/**
 * Join links.
 *
 * The token in a link is the whole of a player's identity, so where it is carried is a
 * security decision rather than a formatting one. It goes in the URL fragment, which the
 * browser never sends to the server: it stays out of access logs, out of the `Referer`
 * of anything the page later loads, and out of any proxy in between.
 */

import { beforeAll, describe, expect, it, vi } from 'vitest';

import { joinLink, sessionFromHash } from '../src/session.js';

// `joinLink` needs an origin to make an absolute link and nothing else from the DOM, so
// a stub is enough — pulling in a whole browser environment to read one property would
// make this suite slower for no additional assurance.
beforeAll(() => {
  vi.stubGlobal('location', { origin: 'https://campaign.test' });
});

describe('reading a join link', () => {
  it('reads a campaign and a token out of the fragment', () => {
    expect(sessionFromHash('#/j/c1/abc123')).toEqual({ campaignId: 'c1', token: 'abc123' });
  });

  it('decodes escaped segments, so a token is never truncated', () => {
    // Tokens are base64url and need no escaping, but a campaign name might, and half a
    // token silently becomes a wrong token rather than an error.
    expect(sessionFromHash('#/j/c%201/a-b_c')).toEqual({ campaignId: 'c 1', token: 'a-b_c' });
  });

  it('ignores anything that is not a join link', () => {
    for (const hash of ['', '#', '#/j/c1', '#/j/c1/tok/extra', '#/other/c1/tok', '#j/c1/tok']) {
      expect(sessionFromHash(hash), `${hash} was read as a link`).toBeNull();
    }
  });

  it('round-trips a link it produced', () => {
    // The only property that really matters: what is handed to a player must be readable
    // by the page they land on.
    const session = { campaignId: 'c1', token: 'x-Y_z09' };
    const link = joinLink(session);
    expect(sessionFromHash(link.slice(link.indexOf('#')))).toEqual(session);
  });

  it('puts the token after the fragment marker, never before it', () => {
    // If this ever produced a query parameter the token would reach the server on every
    // request and land in its logs, which is exactly what a fragment avoids.
    const link = joinLink({ campaignId: 'c1', token: 'secret-token' });
    expect(link.split('#')[0]).not.toContain('secret-token');
    expect(link).toContain('#/j/c1/secret-token');
  });
});
