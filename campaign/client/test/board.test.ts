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
  commanderRole,
  key,
  knowledgeEvents,
  occupied,
  presentUnderArms,
  parseWorld,
  reduce,
  REFEREE,
  REFEREE_ROLE,
  viewFor,
  type CampaignState,
  type ClientView,
  type Role,
} from '@campaign/shared';

import { boardFrom, dayHour, dayOf, timeOfDay } from '../src/board.js';
import { washBand } from '../src/map/draw.js';

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
      expect(mark!.kind).toBe('live');
    }
  });
});

describe('a commander board', () => {
  // Ney: the army commander, riding with red-1 and answering to nobody.
  const raw = view(commanderRole('ney'));
  const board = boardFrom(raw, DEFAULT_THEME);

  it('holds exactly one live formation: the one they ride with', () => {
    expect(board.units.size).toBe(1);
    const [only] = [...board.units.values()];
    expect(only!.id).toBe(raw.commander!.unitId);
    expect(only!.faction).toBe('red');
  });

  it('holds their subordinates as dated reports rather than units', () => {
    expect(board.reports.size).toBeGreaterThan(0);
    for (const report of board.reports.values()) {
      expect(report.faction).toBe('red');
      expect(board.units.has(report.unitId), `${report.unitId} arrived live`).toBe(false);
      expect(typeof report.atHours).toBe('number');
    }
  });

  it('draws a reported formation as one hex, not a column', () => {
    // A despatch says where a formation stood, not how it was strung out along the road.
    for (const report of board.reports.values()) {
      const mark = board.marks.find((m) => m.id === report.unitId)!;
      expect(mark.column).toHaveLength(1);
      expect(key(mark.column[0]!)).toBe(key(report.head));
      expect(mark.kind, 'a report was drawn as a live formation').toBe('reported');
    }
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
      expect(mark.kind).toBe('contact');
    }
  });

  it('receives the whole accurate map, because terrain fog is off', () => {
    // A stated decision rather than a leak: the tension is where the enemy is, not what
    // the country looks like, and a two-hex sight radius over darkness is unplayable.
    expect(board.world.hexes.size).toBe(world.hexes.size);
    for (const hex of board.world.hexes.values()) {
      expect(hex.tags.has('fog')).toBe(false);
    }
  });

  it('sees far less ground than it has covered', () => {
    // What they can see from where they stand, against everywhere their command has been. If
    // these ever converged, sight has stopped being limited to their own formation.
    expect(board.visible.size).toBeGreaterThan(0);
    expect(board.visible.size).toBeLessThan(board.surveyed.size);
  });

  it('carries a name for every unit it can see', () => {
    // The name lives on the unit precisely so it survives this trip. A console that shows
    // "red-1" is showing the engine's bookkeeping to a player.
    for (const unit of board.units.values()) expect(unit.name).toBeTruthy();
  });
});

describe('the symbols a commander is given', () => {
  const board = boardFrom(view(commanderRole('ney')), DEFAULT_THEME);
  const symbolOf = (id: string) => board.marks.find((m) => m.id === id)!.symbol;

  it('frames their own formation as friendly and solid', () => {
    const [own] = [...board.units.values()];
    const symbol = symbolOf(own!.id);
    expect(symbol.affiliation).toBe('friend');
    expect(symbol.dashed, 'a formation they are standing next to was drawn as reported').toBe(
      false,
    );
    expect(symbol.kind).toBe(own!.kind);
  });

  it('frames a reported formation as friendly and dashed', () => {
    // The standard's mark for a position reported rather than observed, which is exactly
    // what a despatch carries. Its arm and size are not in doubt — only where it is.
    for (const report of board.reports.values()) {
      const symbol = symbolOf(report.unitId);
      expect(symbol.affiliation).toBe('friend');
      expect(symbol.dashed).toBe(true);
      expect(symbol.kind).toBe(report.kind);
      expect(symbol.echelon).toBe(report.echelon);
    }
  });

  it('frames an enemy as hostile, and draws only what the sighting earned', () => {
    // A plain sighting is intel 2. The patrol table grants rough size at 4 and the arm at
    // 5, so both are null here and the symbol is an empty diamond — which is precisely
    // what an empty frame means in the standard.
    //
    // Taken from whichever commander's own formation is actually in contact: a commander
    // sees through the column they ride with and no other, so which of them has a sighting
    // is a fact about the scenario rather than something to assume.
    const seeing = ['ney', 'kellermann', 'soult']
      .map((id) => boardFrom(view(commanderRole(id)), DEFAULT_THEME))
      .find((b) => b.contacts.size > 0);

    expect(seeing, 'nobody on the red side can see the enemy at all').toBeDefined();
    const symbolOf = (id: string) => seeing!.marks.find((m) => m.id === id)!.symbol;

    for (const contact of seeing!.contacts.values()) {
      // Keyed by the commander's own label for the sighting. The observed unit's id is
      // not in the payload at all, which is what stops two marks being correlated.
      const symbol = symbolOf(contact.id);
      expect(symbol.affiliation).toBe('hostile');
      expect(symbol.dashed).toBe(true);
      expect(symbol.kind).toBe(contact.kind);
      expect(symbol.echelon).toBe(contact.echelon);
      if (contact.intelLevel < 5) expect(symbol.kind).toBeNull();
      if (contact.intelLevel < 4) expect(symbol.echelon).toBeNull();
    }
  });
});

describe('the symbols a referee is given', () => {
  const board = boardFrom(view(REFEREE_ROLE), DEFAULT_THEME);

  it('frames every formation as known, because they have no side to be hostile to', () => {
    expect(board.marks.length).toBeGreaterThan(0);
    for (const mark of board.marks) {
      expect(mark.symbol.affiliation).toBe('friend');
      expect(mark.symbol.dashed).toBe(false);
      expect(mark.symbol.kind).not.toBeNull();
    }
  });

  it('draws a brigade as a brigade and a division as a division', () => {
    // Stated on the unit rather than guessed from strength: the demo's Light Brigade is
    // four thousand strong, which the fallback would round up to a division.
    const brigade = [...board.units.values()].find((u) => u.name === 'Light Brigade');
    expect(brigade, 'the demo should still field a brigade').toBeDefined();
    expect(board.marks.find((m) => m.id === brigade!.id)!.symbol.echelon).toBe('brigade');

    const division = [...board.units.values()].find((u) => u.name === '1re Division');
    expect(board.marks.find((m) => m.id === division!.id)!.symbol.echelon).toBe('division');
  });
});

/**
 * Riders in flight, which only a referee ever sees.
 *
 * The negative half is the one that matters. A commander's board has no riders not because
 * the console filters them out but because the route is never in their payload — a rider's
 * path runs to where their addressee actually stands, so drawing one for the sender would
 * hand them the position of their own detached corps and end the game.
 */
describe('riders on the map', () => {
  const riding = (() => {
    const out = applyAll(
      [
        {
          kind: 'send_despatch',
          from: 'ney',
          to: 'kellermann',
          body: { text: 'Move on Quatre Bras with all speed.' },
        },
        { kind: 'advance_clock', hours: 1 },
      ],
      state,
      world,
      'lenient',
    );
    if (!out.ok) throw new Error('the despatch was refused');
    return out.state;
  })();

  const asRole = (role: Role): ClientView =>
    viewFor({ campaignId: 'c1', state: riding, worldDoc, world }, role);

  it('shows the referee where a courier has actually got to', () => {
    const board = boardFrom(asRole(REFEREE_ROLE), DEFAULT_THEME);
    expect(board.riders).toHaveLength(1);

    const rider = board.riders[0]!;
    // An hour of riding is several hexes at courier pace, so they are neither at the start
    // nor at the end: the whole value of the overlay is seeing them in the country between.
    expect(rider.ridden.length).toBeGreaterThan(1);
    expect(rider.ahead.length).toBeGreaterThan(1);
    // The two halves meet at the hex they are on, and nowhere else.
    expect(rider.ridden.at(-1)).toEqual(rider.at);
    expect(rider.ahead[0]).toEqual(rider.at);
  });

  it('gives the sender no rider at all, because they were sent no route', () => {
    const ney = asRole(commanderRole('ney'));
    expect(ney.despatches).toEqual([]);
    expect(boardFrom(ney, DEFAULT_THEME).riders).toEqual([]);
    // And the outbox says they wrote it, without a hint of where it is.
    expect(ney.sent).toHaveLength(1);
    expect(JSON.stringify(ney.sent)).not.toContain('route');
  });

  it('gives the addressee no rider either, and nothing in their hand yet', () => {
    const kellermann = asRole(commanderRole('kellermann'));
    expect(boardFrom(kellermann, DEFAULT_THEME).riders).toEqual([]);
    expect(kellermann.received).toEqual([]);
  });

  it('stops drawing a rider once they have arrived', () => {
    const arrived = applyAll(
      [{ kind: 'advance_clock', hours: 200 }],
      riding,
      world,
      'lenient',
    );
    const board = boardFrom(
      viewFor({ campaignId: 'c1', state: arrived.state, worldDoc, world }, REFEREE_ROLE),
      DEFAULT_THEME,
    );
    expect(board.riders).toEqual([]);
  });
});

/**
 * The wash: which of the three bands a hex falls in.
 *
 * Classification only. What it looks like is a matter of taste and lives in the theme;
 * what it *claims* is not, and getting that wrong would either hide ground a commander is
 * entitled to see or, worse, imply they are watching ground they are not.
 */
describe('the observation wash', () => {
  const ney = boardFrom(view(commanderRole('ney')), DEFAULT_THEME);
  const referee = boardFrom(view(REFEREE_ROLE), DEFAULT_THEME);

  const watched = [...ney.visible][0]!;
  const remembered = [...ney.surveyed].find((k) => !ney.visible.has(k))!;
  const stranger = [...world.hexes.keys()].find(
    (k) => !ney.visible.has(k) && !ney.surveyed.has(k),
  )!;

  it('has ground in all three states to begin with', () => {
    expect(watched).toBeDefined();
    expect(remembered).toBeDefined();
    expect(stranger).toBeDefined();
  });

  it('tells watched, marched-over and unknown ground apart', () => {
    expect(washBand(watched, ney.visible, ney.surveyed, 'three')).toBe('observed');
    expect(washBand(remembered, ney.visible, ney.surveyed, 'three')).toBe('surveyed');
    expect(washBand(stranger, ney.visible, ney.surveyed, 'three')).toBe('unseen');
  });

  it('collapses remembered ground into the dark at two tones', () => {
    expect(washBand(watched, ney.visible, ney.surveyed, 'two')).toBe('observed');
    expect(washBand(remembered, ney.visible, ney.surveyed, 'two')).toBe('unseen');
    expect(washBand(stranger, ney.visible, ney.surveyed, 'two')).toBe('unseen');
  });

  it('washes nothing at all when it is switched off', () => {
    for (const k of [watched, remembered, stranger]) {
      expect(washBand(k, ney.visible, ney.surveyed, 'off')).toBe('observed');
    }
  });

  it('counts a hex they are looking at as observed even if the survey lags', () => {
    // The two sets are maintained independently, and sight is the stronger claim.
    expect(washBand(watched, new Set([watched]), new Set(), 'three')).toBe('observed');
  });

  /**
   * The regression that would black out the one screen meant to see everything. A referee
   * is sent `visible: []` because they are not on the map, and that must read as "wash
   * nothing" rather than as "they can see nothing".
   */
  it('leaves the referee an unwashed map', () => {
    expect(referee.visible.size).toBe(0);
    expect(referee.surveyed.size).toBe(0);
    for (const k of [...world.hexes.keys()].slice(0, 50)) {
      expect(washBand(k, referee.visible, referee.surveyed, 'three')).toBe('observed');
    }
  });

  it('gives a scouting column more ground to watch than the commander it reports to', () => {
    // Kellermann's cavalry is eighteen kilometres of column with the scout trait, so it
    // sweeps a corridor where Ney's infantry sees a patch. That gap is the whole reason
    // for drawing this at all.
    const kellermann = boardFrom(view(commanderRole('kellermann')), DEFAULT_THEME);
    expect(kellermann.visible.size).toBeGreaterThan(ney.visible.size);
  });
});

describe('the campaign clock, as a referee reads it', () => {
  it('counts the first day as day one', () => {
    // Nobody calls the opening day of a campaign day zero.
    expect(dayOf(0)).toBe(1);
    expect(dayOf(23.99)).toBe(1);
    expect(dayOf(24)).toBe(2);
    expect(dayOf(62)).toBe(3);
  });

  it('reads the hour as a clock face', () => {
    expect(timeOfDay(0)).toBe('00:00');
    expect(timeOfDay(6)).toBe('06:00');
    expect(timeOfDay(14.5)).toBe('14:30');
    expect(timeOfDay(23.75)).toBe('23:45');
    // Past midnight, the same face comes round again.
    expect(timeOfDay(24)).toBe('00:00');
    expect(timeOfDay(38.25)).toBe('14:15');
  });

  it('never prints a time that does not exist', () => {
    // The engine works in fractions of an hour, so rounding to the minute is where a
    // 13:60 comes from. Rolling the hour is the fix; rolling the last hour of the day
    // wraps rather than printing 24:00.
    expect(timeOfDay(13.999)).toBe('14:00');
    expect(timeOfDay(23.9999)).toBe('00:00');
    for (let h = 0; h < 48; h += 0.017) {
      const [hh, mm] = timeOfDay(h).split(':').map(Number);
      expect(hh, `${h}`).toBeGreaterThanOrEqual(0);
      expect(hh, `${h}`).toBeLessThan(24);
      expect(mm, `${h}`).toBeLessThan(60);
    }
  });

  it('puts the day and the time together', () => {
    expect(dayHour(0)).toBe('Day 1, 00:00');
    expect(dayHour(62.5)).toBe('Day 3, 14:30');
  });

  it('rolls the day when the minute it rounds to is midnight', () => {
    // The day and the time are rounded off the same value. Rounding each for itself gives
    // "Day 1, 00:00" for an hour that is, to the minute, the start of the second day —
    // wrong in the one place a reader actually checks.
    expect(dayHour(23.9999)).toBe('Day 2, 00:00');
    expect(dayOf(23.9999)).toBe(2);
    expect(dayHour(47.9999)).toBe('Day 3, 00:00');
  });
});

/**
 * Deployment, and the ground it is fought over.
 *
 * Two separate things that meet on the map. A formation in line of battle is drawn as
 * ranks rather than as a column ribbon; a battlefield is ground, drawn under everything
 * and belonging to nobody.
 */
describe('formations drawn deployed', () => {
  const deployed = (role: Role, unitId: string): ClientView => {
    const v = view(role);
    return {
      ...v,
      units: v.units.map((u) => (u.id === unitId ? { ...u, formation: 'battle' as const } : u)),
    };
  };

  it('draws a formation in line of battle as ranks, not as a ribbon', () => {
    const anyUnit = [...state.units.values()][0]!;
    const before = boardFrom(view(REFEREE_ROLE), DEFAULT_THEME);
    const after = boardFrom(deployed(REFEREE_ROLE, anyUnit.id), DEFAULT_THEME);

    expect(before.marks.find((m) => m.id === anyUnit.id)!.deployed).toBe(false);
    expect(after.marks.find((m) => m.id === anyUnit.id)!.deployed).toBe(true);
  });

  it('collapses the deployed footprint to its frontage', () => {
    const anyUnit = [...state.units.values()][0]!;
    const before = boardFrom(view(REFEREE_ROLE), DEFAULT_THEME);
    const after = boardFrom(deployed(REFEREE_ROLE, anyUnit.id), DEFAULT_THEME);

    const marching = before.marks.find((m) => m.id === anyUnit.id)!.column.length;
    const standing = after.marks.find((m) => m.id === anyUnit.id)!.column.length;
    expect(standing).toBeLessThanOrEqual(marching);
    expect(standing).toBe(Math.max(1, Math.ceil(presentUnderArms(anyUnit) / 10000)));
  });

  it('never draws a reported or sighted formation as deployed', () => {
    // How an enemy is standing is intelligence. A sighting does not carry it, so the
    // drawing code is never handed it — not even as a false value it might mistake.
    const board = boardFrom(view(commanderRole('ney')), DEFAULT_THEME);
    for (const mark of board.marks) {
      if (mark.kind === 'live') continue;
      expect(mark.deployed, `${mark.id} is drawn deployed on hearsay`).toBe(false);
    }
  });

  it('carries the ground being fought over onto the board', () => {
    const v = view(REFEREE_ROLE);
    const here = [...state.units.values()][0]!.column[0]!;
    const board = boardFrom({ ...v, battle: [key(here)] }, DEFAULT_THEME);
    expect(board.battle.has(key(here))).toBe(true);
  });

  it('has no battlefield where none was declared', () => {
    expect(boardFrom(view(REFEREE_ROLE), DEFAULT_THEME).battle.size).toBe(0);
  });
});
