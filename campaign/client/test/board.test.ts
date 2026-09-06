/**
 * What the console makes of what it is sent.
 *
 * These build a real `ClientView` through `viewFor` — the same function the server calls
 * — rather than hand-writing one, because a hand-written view is a view whose fog nobody
 * enforced, and it would let this file assert that the console draws things the server
 * would never actually send.
 *
 * The assertion that matters is the negative one: an enemy contact must not turn into a
 * unit anywhere on this path. An earlier version of the console fabricated a `Unit` from
 * the enemy's real record so the map had something to draw, which worked perfectly and
 * put the entire enemy order of battle in the page.
 */

import { describe, expect, it } from 'vitest';

import worldDoc from '../../shared/test/fixtures/world-32x32.json';

import {
  applyAll,
  DEFAULT_THEME,
  DEFAULT_CONFIG,
  DEMO_FACTIONS,
  demoCommands,
  EMPTY_STATE,
  factionRole,
  key,
  observationEvents,
  occupied,
  parseWorld,
  reduce,
  REFEREE,
  REFEREE_ROLE,
  viewFor,
  type CampaignState,
  type ClientView,
  type Role,
} from '@campaign/shared';

import { boardFrom } from '../src/board.js';

const world = parseWorld(worldDoc);

const built = applyAll(
  [
    {
      kind: 'create_campaign',
      name: 'Demonstration',
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

/**
 * The observation pass, which is the store's job rather than the engine's.
 *
 * `applyAll` does not run it, so a state built straight from commands has every faction
 * knowing nothing — which would make the fog assertions below pass trivially against an
 * empty `seen`. Doing here what the server does after every command is the only way these
 * tests are looking at a view a client could actually receive.
 */
const state: CampaignState = observationEvents(built.state, world, DEFAULT_CONFIG).reduce(
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

describe('a referee board', () => {
  const board = boardFrom(view(REFEREE_ROLE), DEFAULT_THEME);

  it('holds every unit on both sides', () => {
    expect(board.units.size).toBe(state.units.size);
    expect(new Set([...board.units.values()].map((u) => u.faction))).toEqual(
      new Set(['red', 'blue']),
    );
  });

  it('has no contacts, because it has the units themselves', () => {
    expect(board.contacts.size).toBe(0);
  });

  it('draws each unit as the length of ground it occupies', () => {
    for (const unit of board.units.values()) {
      const mark = board.marks.find((m) => m.id === unit.id);
      expect(mark, `${unit.id} is not drawn`).toBeDefined();
      expect(mark!.column.map(key)).toEqual(occupied(unit).map(key));
      expect(mark!.visible).toBe(true);
    }
  });
});

describe('a commander board', () => {
  const raw = view(factionRole('red'));
  const board = boardFrom(raw, DEFAULT_THEME);

  it('holds only its own units', () => {
    expect(board.units.size).toBeGreaterThan(0);
    for (const unit of board.units.values()) expect(unit.faction).toBe('red');
  });

  it('never turns a contact into a unit', () => {
    // The whole point. A contact carries almost nothing; a unit carries everything.
    for (const contact of board.contacts.values()) {
      expect(board.units.has(contact.unitId), `${contact.unitId} became a unit`).toBe(false);
    }
  });

  it('draws a contact as one hex, not a column', () => {
    // Knowing an enemy was in a village is not knowing how far back its baggage was
    // strung out. Drawing a column there would invent a report nobody made.
    for (const contact of board.contacts.values()) {
      const mark = board.marks.find((m) => m.id === contact.unitId)!;
      expect(mark.column).toHaveLength(1);
      expect(key(mark.column[0]!)).toBe(key(contact.coord));
      expect(mark.visible).toBe(false);
    }
  });

  it('receives a world with the same number of hexes as the referee sees', () => {
    // Bounds are preserved by masking, so every player's canvas is the same size and the
    // maps overlay. A shorter hex list would re-centre this commander's map on its own.
    expect(board.world.hexes.size).toBe(world.hexes.size);
  });

  it('knows less ground than there is', () => {
    // If this ever equalled the map, the masking has stopped happening and every other
    // assertion here would still pass.
    expect(board.seen.size).toBeGreaterThan(0);
    expect(board.seen.size).toBeLessThan(world.hexes.size);
  });

  it('tags every unseen hex as fog rather than dropping it', () => {
    for (const [k, hex] of board.world.hexes) {
      if (board.seen.has(k)) continue;
      expect(hex.tags.has('fog'), `${k} is not fogged`).toBe(true);
    }
  });

  it('carries a name for every unit it can see', () => {
    // The name lives on the unit precisely so it survives this trip. A console that shows
    // "red-1" is showing the engine's bookkeeping to a player.
    for (const unit of board.units.values()) expect(unit.name).toBeTruthy();
  });
});
