/**
 * Holding a parsed world, and the assumption that makes it safe.
 *
 * `authorise` calls `campaign()` before any route sees a request, so on a 200x200 map
 * every authenticated call was reading 23 MB out of SQLite, parsing it, and rebuilding
 * forty thousand hexes — 56 ms and a fresh allocation of the whole map, per request, per
 * commander watching.
 *
 * Caching that is only correct because a campaign row never changes. That is a property of
 * the code rather than a wish, so it is asserted here: everything that happens to a
 * campaign is an event, and events are not in this row.
 */

import { readFileSync } from 'node:fs';

import { describe, expect, it } from 'vitest';

import { CampaignStore } from '../src/store.js';
import { openDb } from '../src/db.js';

const SOURCE = new URL('../src/store.ts', import.meta.url);

const world = (): unknown =>
  JSON.parse(
    readFileSync(new URL('../../shared/test/fixtures/world-32x32.json', import.meta.url), 'utf8'),
  );

const FACTIONS = [{ id: 'red', name: 'Red', color: '#c00' }];

const storeWith = (ids: string[]) => {
  const store = new CampaignStore(openDb());
  for (const id of ids) store.create({ id, name: id, worldDoc: world(), factions: FACTIONS });
  return store;
};

describe('the assumption underneath the cache', () => {
  it('never updates or deletes a campaign row', () => {
    // The whole safety argument. An `UPDATE campaigns` anywhere in this file would mean a
    // cached row serving the old name, the old ruleset or the old numbers until the
    // process restarted — and only on whichever instance happened to hold it.
    //
    // Comments are stripped first. The prose above this very assertion contains the
    // words it looks for, and a test that matches its own documentation tests nothing.
    const code = readFileSync(SOURCE, 'utf8')
      .replace(/\/\*[\s\S]*?\*\//g, '')
      .replace(/\/\/.*$/gm, '');

    expect(code).not.toMatch(/UPDATE\s+campaigns/i);
    expect(code).not.toMatch(/DELETE\s+FROM\s+campaigns/i);
    // And that the stripping did not simply remove everything.
    expect(code).toMatch(/INSERT\s+INTO\s+campaigns/i);
  });
});

describe('caching a parsed campaign', () => {
  it('hands back the same object rather than parsing again', () => {
    const store = storeWith(['c1']);
    const first = store.campaign('c1')!;
    const second = store.campaign('c1')!;
    // Identity, not equality: a re-parse would be equal and would have cost the 56 ms.
    expect(second).toBe(first);
    expect(second.world).toBe(first.world);
  });

  it('is warm the moment a campaign is created', () => {
    const store = new CampaignStore(openDb());
    const { campaign } = store.create({
      id: 'c1', name: 'c', worldDoc: world(), factions: FACTIONS,
    });
    // The referee is about to look at the map they have just uploaded.
    expect(store.campaign('c1')).toBe(campaign);
  });

  it('still answers for a campaign that has fallen out of the cache', () => {
    const store = storeWith(['c1', 'c2', 'c3', 'c4', 'c5']);
    // c1 is five creations old and the cache holds four.
    const reloaded = store.campaign('c1');
    expect(reloaded).not.toBeNull();
    expect(reloaded!.id).toBe('c1');
    expect(reloaded!.world.hexes.size).toBeGreaterThan(0);
  });

  it('keeps the campaign being played, not the one most recently made', () => {
    const store = storeWith(['c1']);
    const played = store.campaign('c1')!;

    // Four more campaigns created while c1 is being read between each. Reading has to
    // count as use, or the cache holds whatever was made last and evicts the live game.
    for (const id of ['c2', 'c3', 'c4', 'c5']) {
      store.create({ id, name: id, worldDoc: world(), factions: FACTIONS });
      expect(store.campaign('c1'), `after ${id}`).toBe(played);
    }
  });

  it('bounds what it holds, rather than keeping every world forever', () => {
    const store = storeWith(['c1', 'c2', 'c3', 'c4', 'c5', 'c6']);
    // c1 and c2 are gone; each returns a freshly parsed row, not the original object.
    const a = store.campaign('c1')!;
    expect(store.campaign('c1')).toBe(a);
    expect(a.id).toBe('c1');
  });

  it('sends a client the projected world, whether cached or loaded', () => {
    const store = storeWith(['c1', 'c2', 'c3', 'c4', 'c5']);
    const fresh = store.campaign('c5')!;
    const reloaded = store.campaign('c1')!;

    const keys = (row: { worldDoc: unknown }) =>
      Object.keys((row.worldDoc as { hexes: object[] }).hexes[0]!).sort();

    expect(keys(reloaded)).toEqual(keys(fresh));
    expect(keys(fresh)).not.toContain('moisture');
  });

  it('parses the same world from the projection it holds', () => {
    const store = storeWith(['c1']);
    const row = store.campaign('c1')!;
    // The cached document is already projected and `viewFor` projects again on the way
    // out. That second pass has to be a no-op rather than a second reduction.
    expect(row.world.hexes.size).toBe(
      (JSON.parse(
        readFileSync(
          new URL('../../shared/test/fixtures/world-32x32.json', import.meta.url),
          'utf8',
        ),
      ) as { hexes: unknown[] }).hexes.length,
    );
  });
});
