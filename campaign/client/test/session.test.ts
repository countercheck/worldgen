/**
 * What this browser holds, and why it is held per campaign.
 *
 * Storage used to be one session: whatever was last opened. That is what made the front
 * page unreachable — the campaign was a property of the browser rather than of the
 * address, so there was no address that meant "not in a campaign", and a referee running
 * two games had the second quietly evict the first.
 *
 * These are the properties that had to become true for `#/c/<id>` to work at all: a
 * campaign can be found by its id, two of them coexist, and a browser that was holding a
 * session under the old scheme still finds it after the upgrade.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';

import {
  forgetCampaign,
  listCampaigns,
  loadCampaign,
  rememberCampaign,
  saveCampaign,
} from '../src/session.js';

/** A `localStorage` that lives in a Map. The tests run in node, which has none. */
class FakeStorage {
  private items = new Map<string, string>();
  getItem(k: string): string | null {
    return this.items.get(k) ?? null;
  }
  setItem(k: string, v: string): void {
    this.items.set(k, v);
  }
  removeItem(k: string): void {
    this.items.delete(k);
  }
}

let store: FakeStorage;

beforeEach(() => {
  store = new FakeStorage();
  vi.stubGlobal('localStorage', store);
});

const held = (campaignId: string, token: string, name?: string) => ({
  session: { campaignId, token },
  held: {},
  ownToken: token,
  ...(name === undefined ? {} : { name }),
});

describe('holding a campaign', () => {
  it('finds one by its id, which is what a campaign address has to do', () => {
    saveCampaign(held('c1', 'tok1'));
    expect(loadCampaign('c1')?.session.token).toBe('tok1');
  });

  it('holds nothing for a campaign it was never given a link to', () => {
    expect(loadCampaign('never-heard-of-it')).toBeNull();
  });

  it('holds two campaigns at once', () => {
    // The old single-key store could not: opening the second evicted the first, and a
    // referee running two games lost the way into one of them by looking at the other.
    saveCampaign(held('c1', 'tok1'));
    saveCampaign(held('c2', 'tok2'));

    expect(loadCampaign('c1')?.session.token).toBe('tok1');
    expect(loadCampaign('c2')?.session.token).toBe('tok2');
  });
});

describe('the list the front page shows', () => {
  it('names each campaign and says which seat this browser holds', () => {
    saveCampaign({ ...held('c1', 'tok1', 'Waterloo'), held: { ney: 'seat-token' } });
    saveCampaign(held('c2', 'tok2', 'Austerlitz'));

    expect(listCampaigns()).toEqual([
      { campaignId: 'c2', name: 'Austerlitz', isReferee: false },
      { campaignId: 'c1', name: 'Waterloo', isReferee: true },
    ]);
  });

  it('takes the name and the seat from the last view seen', () => {
    saveCampaign(held('c1', 'tok1'));
    expect(listCampaigns()[0]!.name).toBeNull();

    expect(rememberCampaign('c1', { name: 'Waterloo', role: 'commander' })).toBe(true);
    expect(listCampaigns()[0]!.name).toBe('Waterloo');

    // Nothing to write, so nothing is written — the console calls this on every view.
    expect(rememberCampaign('c1', { name: 'Waterloo', role: 'commander' })).toBe(false);
  });

  it('believes the server about the seat, not the seat tokens it happens to hold', () => {
    // A referee who arrives on their own join link holds no seat tokens: those are minted
    // once, when the campaign is created. Counting them filed the referee's own link
    // under "a commander" on the front page.
    saveCampaign(held('c1', 'ref-token'));
    expect(listCampaigns()[0]!.isReferee).toBe(false);

    rememberCampaign('c1', { name: 'Waterloo', role: 'referee' });
    expect(listCampaigns()[0]!.isReferee).toBe(true);
  });

  it('does not invent a campaign for a name it was never given a link to', () => {
    expect(rememberCampaign('c1', { name: 'Waterloo', role: 'referee' })).toBe(false);
    expect(listCampaigns()).toEqual([]);
  });
});

describe('forgetting one', () => {
  it('drops that campaign and leaves the rest', () => {
    saveCampaign(held('c1', 'tok1'));
    saveCampaign(held('c2', 'tok2'));

    forgetCampaign('c1');

    expect(loadCampaign('c1')).toBeNull();
    expect(loadCampaign('c2')?.session.token).toBe('tok2');
  });
});

describe('a browser upgraded mid-campaign', () => {
  it('still finds the session it was holding under the old scheme', () => {
    // The app is deployed and people hold links to games in progress. Losing those on
    // the deploy would mean every player having to be sent a new link.
    store.setItem(
      'campaign.session',
      JSON.stringify({
        session: { campaignId: 'old', token: 'old-token' },
        held: {},
        ownToken: 'old-token',
      }),
    );

    expect(loadCampaign('old')?.session.token).toBe('old-token');
    expect(listCampaigns().map((c) => c.campaignId)).toEqual(['old']);
  });

  it('picks up the seats and tokens the old scheme kept in their own keys', () => {
    store.setItem(
      'campaign.session',
      JSON.stringify({ session: { campaignId: 'old', token: 'ref' }, ownToken: 'ref' }),
    );
    store.setItem('campaign.tokens', JSON.stringify({ old: { ney: 'ney-token' } }));
    store.setItem(
      'campaign.seats',
      JSON.stringify({ old: { ney: { name: 'Marshal Ney', faction: 'blue' } } }),
    );

    const stored = loadCampaign('old');
    expect(stored?.held).toEqual({ ney: 'ney-token' });
    expect(stored?.seats).toEqual({ ney: { name: 'Marshal Ney', faction: 'blue' } });
    expect(listCampaigns()[0]!.isReferee).toBe(true);
  });

  it('does not resurrect a migrated campaign after it is forgotten', () => {
    store.setItem(
      'campaign.session',
      JSON.stringify({
        session: { campaignId: 'old', token: 'old-token' },
        held: {},
        ownToken: 'old-token',
      }),
    );
    expect(loadCampaign('old')).not.toBeNull();

    forgetCampaign('old');

    expect(loadCampaign('old')).toBeNull();
    expect(listCampaigns()).toEqual([]);
  });
});
