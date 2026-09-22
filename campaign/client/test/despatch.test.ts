/**
 * The post, as the console reasons about it.
 *
 * Built through `viewFor` and the real engine rather than from hand-written objects, for
 * the reason `board.test.ts` gives: a hand-written view is a view whose fog nobody
 * enforced, and it would let this file assert the console does things with data the
 * server would never send.
 *
 * The estimate is the interesting one. It is computed on the client because the server
 * must never send a delivery time — an estimate is a distance and a distance is a
 * position — and the test that matters is that it is derived from the commander's own
 * stale report rather than from anything current.
 */

import { describe, expect, it } from 'vitest';

import worldDoc from '../../shared/test/fixtures/world-32x32.json';

import {
  applyAll,
  commanderRole,
  DEFAULT_CONFIG,
  DEMO_FACTIONS,
  demoCommands,
  EMPTY_STATE,
  knowledgeEvents,
  parseWorld,
  reduce,
  REFEREE,
  REFEREE_ROLE,
  viewFor,
  type CampaignState,
  type ClientView,
  type Role,
} from '@campaign/shared';

import {
  correspondents,
  estimateRide,
  forwardOf,
  inbox,
  labelling,
  outbox,
  REFEREE_ADDRESS,
} from '../src/despatch.js';
import { copy } from '../src/copy.js';
import { idFor } from '../src/orbat.js';

const world = parseWorld(worldDoc);

const built = applyAll(
  [
    {
      kind: 'create_campaign',
      name: 'Post',
      world: {
        seed: world.seed,
        width: world.width,
        height: world.height,
        layout: world.layout,
        schemaVersion: world.schemaVersion,
        hash: 'sha256:test',
      },
      seed: 1,
    },
    ...DEMO_FACTIONS.map((faction) => ({ kind: 'add_faction' as const, faction })),
    ...demoCommands(world),
  ],
  EMPTY_STATE,
  world,
  'strict',
);

if (!built.ok) throw new Error('the demo scenario is not legal');

const state: CampaignState = knowledgeEvents(built.state, world, DEFAULT_CONFIG).reduce(
  (acc, payload, i) =>
    reduce(acc, {
      seq: acc.nextSeq + i,
      clockHours: acc.clockHours,
      actor: REFEREE,
      payload,
      forced: false,
      strictness: 'strict',
      bypassed: [],
    }),
  built.state,
);

const view = (role: Role): ClientView =>
  viewFor({ campaignId: 'c1', state, worldDoc, world }, role);

/** Ney: the army commander, riding with 1re Division, with two commanders beneath them. */
const ney = view(commanderRole('ney'));

describe('who a commander may write to', () => {
  const kellermann = view(commanderRole('kellermann'));
  const ids = (v: ClientView, from?: string) => correspondents(v, from).map((c) => c.id);

  it('offers exactly who the server says may be reached', () => {
    // The same function `check` enforces computed the list, so the form cannot offer
    // somebody a despatch would then be refused for.
    expect([...ids(ney)].sort()).toEqual([...ney.addressees['ney']!]);
    expect([...ids(kellermann)].sort()).toEqual([...kellermann.addressees['kellermann']!]);
  });

  it('offers their superior and those directly beneath them, marked as such', () => {
    expect(correspondents(kellermann).find((c) => c.id === 'ney')?.relation).toBe('superior');
    expect(correspondents(ney).find((c) => c.id === 'kellermann')?.relation).toBe('subordinate');
  });

  it('offers nobody the server did not, even a subordinate', () => {
    const narrowed: ClientView = { ...ney, addressees: { ney: ['kellermann'] } };
    expect(ids(narrowed)).toEqual(['kellermann']);
  });

  it('marks anyone else as in sight, and lists them last', () => {
    // Soult is Kellermann's peer: reachable only because their column is in view.
    const inView: ClientView = { ...kellermann, addressees: { kellermann: ['soult', 'ney'] } };
    expect(correspondents(inView).map((c) => [c.id, c.relation])).toEqual([
      ['ney', 'superior'],
      ['soult', 'in_sight'],
    ]);
  });

  it('never offers the enemy', () => {
    // The list comes from `view.commanders`, which the server has already limited to their
    // own side. Asserting it here as well is cheap and catches the day somebody widens
    // that for a legitimate-looking reason.
    for (const c of correspondents(ney)) {
      expect(ney.commanders.find((x) => x.id === c.id)!.faction).toBe(
        ney.commander!.faction,
      );
    }
  });

  it('gives a commander with no seat nobody to write to', () => {
    expect(correspondents(view(REFEREE_ROLE))).toEqual([]);
  });

  it('gives a referee writing as a commander that commander\'s list', () => {
    const referee = view(REFEREE_ROLE);
    expect([...ids(referee, 'ney')].sort()).toEqual([...referee.addressees['ney']!]);
    expect([...ids(referee, 'ney')].sort()).toEqual([...ids(ney)].sort());
  });

  it('keeps the referee\'s address out of reach of any commander\'s id', () => {
    expect(idFor(REFEREE_ADDRESS, new Set())).not.toBe(REFEREE_ADDRESS);
    expect(idFor('referee', new Set())).not.toBe(REFEREE_ADDRESS);
  });
});

describe('naming an officer by what they command', () => {
  it('sends the name of the formation each commander rides with', () => {
    const neyHere = ney.commanders.find((c) => c.id === 'ney')!;
    expect(neyHere.unitName).toBe('1re Division');
    expect(ney.commander!.unitName).toBe('1re Division');
  });

  it('sends formation names for their own side only', () => {
    // The order of battle a commander already holds, and no more: the enemy's
    // establishment arrives by sighting or not at all.
    expect(ney.commanders.every((c) => c.faction === ney.commander!.faction)).toBe(true);
    expect(ney.commanders.some((c) => c.unitName === '3rd Division')).toBe(false);
  });

  it('sends the referee every side\'s', () => {
    const referee = view(REFEREE_ROLE);
    expect(referee.commanders.some((c) => c.unitName === '3rd Division')).toBe(true);
  });

  it('falls back to the unit id when the formation is gone', () => {
    const units = new Map(state.units);
    units.delete('red-2');
    const bereaved = viewFor(
      { campaignId: 'c1', state: { ...state, units }, worldDoc, world },
      commanderRole('ney'),
    );
    expect(bereaved.commanders.find((c) => c.id === 'kellermann')!.unitName).toBe('red-2');
  });

  it('puts the formation and side on every addressee', () => {
    const kellermann = correspondents(ney).find((c) => c.id === 'kellermann')!;
    expect(kellermann.unitName).toBe('Cuirassiers de la Garde');
    expect(kellermann.faction).toBe('red');
  });

  it('labels a known commander by name, formation and the side\'s name', () => {
    expect(labelling(ney)('kellermann')).toEqual({
      name: 'General Kellermann',
      unit: 'Cuirassiers de la Garde',
      // The name, not the id: "red" is a key, not something a reader should see.
      faction: 'Armée du Nord',
    });
  });

  it('labels a commander this view does not know by their id, not by nothing', () => {
    // An enemy officer to a commander, or one removed since the despatch was written.
    expect(labelling(ney)('wellington')).toEqual({
      name: 'wellington',
      unit: 'wellington',
      faction: '',
    });
  });

  it('falls back to the faction id when the side is not in the view', () => {
    const sideless: ClientView = { ...ney, factions: [] };
    expect(labelling(sideless)('kellermann').faction).toBe('red');
  });

  it('writes the addressee line and the byline the same way everywhere', () => {
    expect(copy.composer.correspondent('Ney', '1re Division', 'Armée du Nord')).toBe(
      'Ney — 1re Division, Armée du Nord',
    );
    expect(copy.post.commands('1re Division', 'Armée du Nord')).toBe(
      '1re Division · Armée du Nord',
    );
  });
});

describe('the ride estimate, which only a referee sees', () => {
  it('is null for a commander, who cannot know where the commander is', () => {
    // The rule, not an omission. A commander's view carries exactly one live unit — their
    // own — so there is nothing to measure to. A number here would be a distance, and a
    // distance is a position.
    const subordinate = correspondents(ney).find((c) => c.relation === 'subordinate')!;
    expect(estimateRide(ney, world, DEFAULT_CONFIG, subordinate.id)).toBeNull();
    expect(estimateRide(ney, world, DEFAULT_CONFIG, 'wellington')).toBeNull();
  });

  it('is null for a referee who has not said whose rider it is', () => {
    // They have no seat of their own to send from.
    expect(estimateRide(view(REFEREE_ROLE), world, DEFAULT_CONFIG, 'ney')).toBeNull();
  });

  it('measures the real ground when the referee names a sender', () => {
    const referee = view(REFEREE_ROLE);
    const subordinate = correspondents(ney).find((c) => c.relation === 'subordinate')!;
    const estimate = estimateRide(
      referee,
      world,
      DEFAULT_CONFIG,
      subordinate.id,
      [],
      'ney',
    );

    expect(estimate).not.toBeNull();
    expect(estimate!.hours).toBeGreaterThan(0);
  });

  it('is null when the sender or the addressee is not on the map', () => {
    const referee = view(REFEREE_ROLE);
    expect(
      estimateRide(referee, world, DEFAULT_CONFIG, 'nobody', [], 'ney'),
    ).toBeNull();
    expect(
      estimateRide(referee, world, DEFAULT_CONFIG, 'ney', [], 'nobody'),
    ).toBeNull();
  });
});

describe('the ledgers', () => {
  it('are empty before anybody writes anything', () => {
    expect(inbox(ney)).toEqual([]);
    expect(outbox(ney)).toEqual([]);
  });

  it('sorts the inbox by the hour described, newest first', () => {
    const withPost: ClientView = {
      ...ney,
      received: [
        received('d1', 4, 9),
        received('d2', 7, 8),
        received('d3', 1, 12),
      ],
    };
    // By what it tells them about, not by when it landed. `d2` arrived before `d1` but
    // describes a later hour, and the later hour is the more useful thing to read first.
    expect(inbox(withPost).map((d) => d.id)).toEqual(['d2', 'd1', 'd3']);
  });
});

describe('forwarding', () => {
  it('says where it came from and when it was written', () => {
    const body = forwardOf(received('d1', 4, 9, 'The enemy is at Ligny.'));
    expect(body.text).toContain('hour 4');
    expect(body.text).toContain('kellermann');
    // Two lags stack, which is very much the period — so the original hour has to
    // survive the forward, and the original text with it.
    expect(body.text).toContain('The enemy is at Ligny.');
  });

  it('carries attached sightings through', () => {
    const d = received('d1', 4, 9);
    const withContacts = { ...d, body: { ...d.body, contacts: [{ unitId: 'x' }] } };
    expect(forwardOf(withContacts as never).contacts).toHaveLength(1);
  });
});

function received(
  id: string,
  sentAtHours: number,
  receivedAtHours: number,
  text = 'Move on Quatre Bras.',
) {
  return {
    id,
    from: 'kellermann',
    sentAtHours,
    receivedAtHours,
    body: { text },
    forwardedFrom: null,
  };
}
