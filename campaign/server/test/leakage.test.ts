/**
 * The tests that matter most in this project.
 *
 * Everything else here is a wargame; this is the part where the wargame is about
 * incomplete information. One careless `reply.send(state)` and a commander's browser holds
 * every position on the map, and no amount of correct rules makes up for it.
 *
 * Two deliberate choices about how these are written:
 *
 * **They assert on the serialised response body, not the object graph.** A structural
 * assertion over a parsed response is convincing and insufficient: a getter, an enumerable
 * field added later, or a `toJSON` can put data on the wire that a shaped assertion never
 * looks at. So the tests take the raw payload string and search it for things that must
 * not be in it.
 *
 * **They assert absence, not presence.** It is easy to check that Ney sees their own
 * division. The bug that ends the game is Wellington's division being in the payload too,
 * in some field nobody thought to look at.
 *
 * ## What is secret now
 *
 * Terrain fog is off by default, so the ground is public and accurate. That does not
 * shrink this file — it moves it. The secrets are the enemy's positions and, carrying the
 * whole design, **the live position of any formation but the one a commander rides with**.
 * Their own corps reaches them as a dated report or not at all, and a `Unit` where a
 * `UnitReport` belongs is the leak this suite now exists to catch.
 *
 * The masking machinery is still exercised, under `describe('with terrain fog on')`, so
 * "switched off rather than removed" is a claim with a test behind it.
 */

import { beforeEach, describe, expect, it } from 'vitest';

import worldDoc from '../../shared/test/fixtures/world-32x32.json';

import {
  DEFAULT_CONFIG,
  key,
  KIND_DEFAULTS,
  parseWorld,
  unkey,
  type Command,
  type Commander,
  type Hex,
  type Unit,
  type UnitReport,
} from '@campaign/shared';

import { buildApp, TOKEN_COOKIE } from '../src/app.js';
import { openDb } from '../src/db.js';

const world = parseWorld(worldDoc);

const dist = (a: Hex, b: Hex): number =>
  (Math.abs(a.q - b.q) + Math.abs(a.r - b.r) + Math.abs(a.q + a.r - (b.q + b.r))) / 2;

/** Three land hexes far enough apart that nobody starts in anybody's recon zone. */
function spreadLand(): [Hex, Hex, Hex] {
  const land = [...world.hexes.values()]
    .filter((h) => h.terrainClass === 'land')
    .map((h) => h.coord);

  let best: [Hex, Hex] = [land[0]!, land[1]!];
  let bestDist = -1;
  for (const a of land) {
    for (const b of land) {
      if (dist(a, b) > bestDist) {
        bestDist = dist(a, b);
        best = [a, b];
      }
    }
  }

  // A third, well clear of both, for the subordinate nobody may see.
  let third = land[0]!;
  let bestClear = -1;
  for (const c of land) {
    const clear = Math.min(dist(c, best[0]), dist(c, best[1]));
    if (clear > bestClear) {
      bestClear = clear;
      third = c;
    }
  }
  return [best[0], best[1], third];
}

const [RED_HEX, BLUE_HEX, SUB_HEX] = spreadLand();

/**
 * Distinct strengths and names per formation, on purpose.
 *
 * These tests search the raw response for things that must not be in it, which only
 * discriminates if the value is unique. Giving both sides 5,000 paperStrength made an early
 * version of the strength assertion pass against red's own division while proving nothing
 * about blue's.
 */
const RED_STRENGTH = 5000;
const BLUE_STRENGTH = 7777;
const SUB_STRENGTH = 6333;

const RED_NAME = 'Division Morand';
const BLUE_NAME = 'Erzherzog Karl Grenadiers';
const SUB_NAME = 'Cuirassiers de Kellermann';

function division(
  id: string,
  faction: string,
  at: Hex,
  corps: string,
  name: string,
  paperStrength: number,
): Unit {
  return {
    id,
    name,
    faction,
    kind: 'infantry',
    paperStrength,
    fatigue: 0,
    experience: 0,
    morale: 30,
    provisions: 40,
    maxProvisions: 40,
    equipment: 30,
    maxEquipment: 30,
    guns: 6,
    marchSpeedKmh: KIND_DEFAULTS.infantry.marchSpeedKmh,
    spacingM: KIND_DEFAULTS.infantry.spacingM,
    spacingMultiplier: 1.3,
    traits: [],
    formation: 'march',
    formationChange: null,
    column: [at],
    hoursMarchedToday: 0,
    corps,
  };
}

const commander = (
  id: string,
  name: string,
  faction: string,
  unitId: string,
  superiorId: string | null,
): Commander => ({ id, name, faction, unitId, superiorId, autoCascade: true });

interface Fixture {
  app: ReturnType<typeof buildApp>;
  id: string;
  referee: string;
  /** Ney: army commander, rides with red-1, commands Kellermann. */
  ney: string;
  /** Kellermann: rides with red-2, answers to Ney, commands nobody. */
  kellermann: string;
  /** Wellington: the other side entirely. */
  wellington: string;
}

/**
 * A campaign with a chain of command on one side.
 *
 * Two red commanders rather than one, because the interesting leak is no longer only
 * between sides — it is between a commander and their own subordinate, whose position they are
 * supposed to learn by despatch and not by opening the response.
 */
async function setUp(opts: { terrainFog?: boolean } = {}): Promise<Fixture> {
  const app = await buildApp({
    db: openDb(),
    cfg: { ...DEFAULT_CONFIG, terrainFog: opts.terrainFog ?? false },
  });

  const created = await app.inject({
    method: 'POST',
    url: '/api/campaigns',
    payload: {
      id: 'test',
      name: 'Leakage',
      world: worldDoc,
      seed: 1,
      factions: [
        { id: 'red', name: 'Armée du Nord', color: '#d1495b' },
        { id: 'blue', name: 'Coalition', color: '#3d6fd1' },
      ],
    },
  });
  expect(created.statusCode).toBe(201);
  const referee = created.json().refereeToken as string;

  const run = async (command: Command): Promise<void> => {
    const res = await app.inject({
      method: 'POST',
      url: '/api/campaigns/test/commands',
      headers: { 'x-campaign-token': referee },
      payload: { command },
    });
    expect(res.statusCode, res.body).toBe(200);
  };

  await run({
    kind: 'add_unit',
    unit: division('red-1', 'red', RED_HEX, 'I Corps', RED_NAME, RED_STRENGTH),
  });
  await run({
    kind: 'add_unit',
    unit: division('red-2', 'red', SUB_HEX, 'Cavalry Reserve', SUB_NAME, SUB_STRENGTH),
  });
  await run({
    kind: 'add_unit',
    unit: division('blue-1', 'blue', BLUE_HEX, 'Grand Army of the Danube', BLUE_NAME, BLUE_STRENGTH),
  });

  await run({
    kind: 'add_commander',
    commander: commander('ney', 'Marshal Ney', 'red', 'red-1', null),
  });
  await run({
    kind: 'add_commander',
    commander: commander('kellermann', 'General Kellermann', 'red', 'red-2', 'ney'),
  });
  await run({
    kind: 'add_commander',
    commander: commander('wellington', 'The Duke', 'blue', 'blue-1', null),
  });

  const seat = async (commanderId: string): Promise<string> => {
    const res = await app.inject({
      method: 'POST',
      url: `/api/campaigns/test/commanders/${commanderId}/token`,
      headers: { 'x-campaign-token': referee },
    });
    expect(res.statusCode, res.body).toBe(201);
    return res.json().token as string;
  };

  return {
    app,
    id: 'test',
    referee,
    ney: await seat('ney'),
    kellermann: await seat('kellermann'),
    wellington: await seat('wellington'),
  };
}

const viewAs = (f: Fixture, token: string) =>
  f.app.inject({
    method: 'GET',
    url: `/api/campaigns/${f.id}/view`,
    headers: { 'x-campaign-token': token },
  });

describe('the view endpoint', () => {
  let f: Fixture;
  beforeEach(async () => {
    f = await setUp();
  });

  it('serves a referee ground truth', async () => {
    const view = (await viewAs(f, f.referee)).json();
    expect(view.role).toBe('referee');
    expect(view.units.map((u: Unit) => u.id).sort()).toEqual(['blue-1', 'red-1', 'red-2']);
    expect(view.reports).toEqual([]);
  });

  it('serves a commander exactly one live formation: the one they ride with', async () => {
    // The assertion this whole step exists for. A second live unit here is somebody
    // else's position arriving as fact when it should arrive as a dated report.
    const view = (await viewAs(f, f.ney)).json();
    expect(view.role).toBe('commander');
    expect(view.commander.id).toBe('ney');
    expect(view.units.map((u: Unit) => u.id)).toEqual(['red-1']);
  });

  it('gives them their subordinate as a report, not as a unit', async () => {
    const view = (await viewAs(f, f.ney)).json();
    const report = (view.reports as UnitReport[]).find((r) => r.unitId === 'red-2');

    expect(report, 'Ney should hear about their own cavalry').toBeDefined();
    expect(report!.atHours).toBe(0);
    // A report carries what a despatch would carry and no more. If any of these ever
    // appear, somebody has widened it toward a Unit and the distinction has collapsed.
    for (const field of ['traits', 'equipment', 'morale', 'spacingM', 'column']) {
      expect(report, `a report carries ${field}`).not.toHaveProperty(field);
    }
  });

  it('never puts an unseen enemy anywhere in the payload', async () => {
    // Made against the raw bytes. Red has never been near blue, so no trace of blue-1 —
    // its id, name, corps or strength — may be on the wire in any field, including ones
    // nobody thought to shape an assertion around.
    const raw = (await viewAs(f, f.ney)).body;

    expect(raw).not.toContain('blue-1');
    expect(raw).not.toContain(BLUE_NAME);
    expect(raw).not.toContain('Grand Army of the Danube');
    expect(raw).not.toContain(`"paperStrength":${BLUE_STRENGTH}`);
  });

  it('never names the enemy chain of command', async () => {
    // Who commands the other side's corps is intelligence. It arrives by sighting or
    // interrogation, not by being on the same map.
    const raw = (await viewAs(f, f.ney)).body;
    expect(raw).not.toContain('wellington');
    expect(raw).not.toContain('The Duke');
  });

  it('gives two commanders on the same side different pictures', async () => {
    // The point of a role being a seat rather than a side. Kellermann rides with red-2 and
    // sees it live; to Ney the same formation is a dated report. Neither has the other's
    // view, and they are on the same side.
    const ney = (await viewAs(f, f.ney)).json();
    const kellermann = (await viewAs(f, f.kellermann)).json();

    expect(ney.units.map((u: Unit) => u.id)).toEqual(['red-1']);
    expect(kellermann.units.map((u: Unit) => u.id)).toEqual(['red-2']);

    expect((ney.reports as UnitReport[]).map((r) => r.unitId)).toEqual(['red-2']);
    // A subordinate reports upward and is told nothing about their superior's column.
    expect(kellermann.reports).toEqual([]);

    expect(ney.visible).not.toEqual(kellermann.visible);
    const kellermannSees = new Set<string>(kellermann.visible);
    expect((ney.visible as string[]).some((k) => kellermannSees.has(k))).toBe(false);
  });

  it("does not send a subordinate their superior's live column", async () => {
    const res = await viewAs(f, f.kellermann);
    const raw = res.body;
    // As a unit, not as a name. The formation's name does reach them — `unitName` on
    // their superior, which is the order of battle every staff on a side holds — so the
    // canary is the unit object itself: its name field, its id among their units, and
    // its strength.
    expect(raw).not.toContain(`"name":"${RED_NAME}"`);
    expect((res.json().units as { id: string }[]).map((u) => u.id)).not.toContain('red-1');
    expect(raw).not.toContain(`"paperStrength":${RED_STRENGTH}`);
  });

  it('sends the ground unmasked when terrain fog is off', async () => {
    // Not a leak: a stated decision. The tension is where the enemy is, not what the
    // country looks like, and a two-hex sight radius over a blacked-out map is unplayable.
    const view = (await viewAs(f, f.ney)).json();
    expect(view.world.hexes.length).toBe(world.hexes.size);
    for (const h of view.world.hexes as { tags: string[] }[]) {
      expect(h.tags).not.toContain('fog');
    }
  });

  it('still reports what they can see and what they have surveyed', async () => {
    const view = (await viewAs(f, f.ney)).json();
    expect(view.visible.length).toBeGreaterThan(0);
    expect(view.surveyed.length).toBeGreaterThan(0);
    expect(view.visible.length).toBeLessThan(world.hexes.size);
  });

  it('tells a commander about a battle they can see, and no other', async () => {
    // Gunfire carries, but this is the campaign map and a battle out of sight is a battle
    // a rider has to bring word of. A referee sees every field; a commander sees the ones
    // on ground they are actually looking at.
    const ney = (await viewAs(f, f.ney)).json();
    const seen = (ney.visible as string[])[0]!;
    const unseen = [...world.hexes.keys()].find((k) => !(ney.visible as string[]).includes(k))!;

    await f.app.inject({
      method: 'POST',
      url: `/api/campaigns/${f.id}/commands`,
      headers: { 'x-campaign-token': f.referee },
      payload: { command: { kind: 'declare_battle', coords: [unkey(seen), unkey(unseen)] } },
    });

    const after = (await viewAs(f, f.ney)).json();
    expect(after.battle).toEqual([seen]);
    expect(after.battle).not.toContain(unseen);

    const ref = (await viewAs(f, f.referee)).json();
    expect([...(ref.battle as string[])].sort()).toEqual([seen, unseen].sort());
  });

  it('does not leak another commander through the log', async () => {
    // The log is a rich source of exactly what the fog withholds: every march, in order.
    const res = await f.app.inject({
      method: 'GET',
      url: `/api/campaigns/${f.id}/log`,
      headers: { 'x-campaign-token': f.ney },
    });
    expect(res.body).not.toContain('blue-1');
    expect(res.body).not.toContain(BLUE_NAME);
  });

  it('gives the referee the whole log', async () => {
    const res = await f.app.inject({
      method: 'GET',
      url: `/api/campaigns/${f.id}/log`,
      headers: { 'x-campaign-token': f.referee },
    });
    expect(res.body).toContain('blue-1');
  });

  it('does not hand a sender their own rider’s route back through the log', async () => {
    // The total, invisible leak. A `despatch_sent` event carries the whole `Despatch`,
    // route included, and the route is planned to where the addressee actually stands —
    // so a commander who could read their own outbox in the log would never need a report
    // again. Filtering the log by actor does not remove it: the event is theirs.
    const sent = await f.app.inject({
      method: 'POST',
      url: `/api/campaigns/${f.id}/commands`,
      headers: { 'x-campaign-token': f.ney },
      payload: {
        command: {
          kind: 'send_despatch',
          from: 'ney',
          to: 'kellermann',
          despatchKind: 'order',
          body: { text: 'Close on Ligny.' },
        },
      },
    });
    expect(sent.statusCode, sent.body).toBe(200);

    const res = await f.app.inject({
      method: 'GET',
      url: `/api/campaigns/${f.id}/log`,
      headers: { 'x-campaign-token': f.ney },
    });

    // They see that they wrote it, and what they wrote.
    expect(res.body).toContain('Close on Ligny.');
    // And nothing about where the rider is going, or whether they got there.
    for (const forbidden of ['route', 'fate', 'progress', 'in_transit']) {
      expect(res.body).not.toContain(forbidden);
    }
    expect(res.body).not.toContain(`"q":${SUB_HEX.q},"r":${SUB_HEX.r}`);
    expect(res.body).not.toContain(SUB_NAME);
    expect(res.body).not.toContain('red-2');
  });

  it('shows a commander nothing of the despatches other commanders wrote', async () => {
    const sent = await f.app.inject({
      method: 'POST',
      url: `/api/campaigns/${f.id}/commands`,
      headers: { 'x-campaign-token': f.ney },
      payload: {
        command: {
          kind: 'send_despatch',
          from: 'ney',
          to: 'kellermann',
          despatchKind: 'order',
          body: { text: 'Close on Ligny.' },
        },
      },
    });
    expect(sent.statusCode, sent.body).toBe(200);

    const res = await f.app.inject({
      method: 'GET',
      url: `/api/campaigns/${f.id}/log`,
      headers: { 'x-campaign-token': f.kellermann },
    });
    // The order is Ney's. Kellermann learns of it by its arrival, in their inbox, at the
    // hour the rider reaches them — and the log is not a second way in.
    expect(res.json()).toEqual([]);
    expect(res.body).not.toContain('Close on Ligny.');
  });
});

describe('once an enemy is actually spotted', () => {
  it('reports it, and no better than the sighting earned', async () => {
    const f = await setUp();

    const beside = { q: RED_HEX.q + 1, r: RED_HEX.r };
    const res = await f.app.inject({
      method: 'POST',
      url: `/api/campaigns/${f.id}/commands`,
      headers: { 'x-campaign-token': f.referee },
      payload: {
        command: {
          kind: 'add_unit',
          unit: division('blue-2', 'blue', beside, 'II Corps', BLUE_NAME, BLUE_STRENGTH),
        } satisfies Command,
      },
    });
    expect(res.statusCode, res.body).toBe(200);

    const view = (await viewAs(f, f.ney)).json();
    // Found by where it was seen, because that is all they have. There is deliberately no
    // way to ask "which contact is blue-2?" from inside a commander's payload — that
    // question is the correlation the rules make them buy with a patrol.
    const contact = (
      view.contacts as {
        id: string;
        coord: { q: number; r: number };
        kind: null;
        corps: null;
        echelon: null;
      }[]
    ).find((c) => c.coord.q === beside.q && c.coord.r === beside.r);

    expect(contact, 'a division one hex away should be seen').toBeDefined();
    // Presence and location only. A plain sighting is intel 2, and the patrol table grants
    // rough size at 4, arm at 5 and the corps at 6 — so all three are withheld, and the
    // symbol this draws is an empty frame, which is what the standard means by it.
    expect(contact!.kind).toBeNull();
    expect(contact!.echelon).toBeNull();
    expect(contact!.corps).toBeNull();

    const raw = (await viewAs(f, f.ney)).body;
    expect(raw).not.toContain('II Corps');
    expect(raw).not.toContain(BLUE_NAME);
    expect(raw).not.toContain(`"paperStrength":${BLUE_STRENGTH}`);

    // A contact carries exactly these fields and no others. Not a search of the whole
    // payload for `"morale"` — red's own division has one, so that would pass while
    // proving nothing at all.
    //
    // `unitId` is the field this list exists to keep out. A contact naming the formation
    // it is a sighting of would let a commander correlate two marks hours apart as the
    // same corps, which is top-level intelligence in these rules and is meant to cost a
    // patrol. `id` in its place is their own staff's label and means nothing to anybody
    // else — two commanders watching the same column hold two different numbers.
    expect(Object.keys(contact!).sort()).toEqual(
      ['corps', 'coord', 'echelon', 'faction', 'id', 'intelLevel', 'kind', 'seenAtHours'].sort(),
    );
    expect(contact!.id).not.toContain('blue');
  });

  it('keeps the enemy’s unit ids out of the payload entirely', async () => {
    const f = await setUp();

    const beside = { q: RED_HEX.q + 1, r: RED_HEX.r };
    await f.app.inject({
      method: 'POST',
      url: `/api/campaigns/${f.id}/commands`,
      headers: { 'x-campaign-token': f.referee },
      payload: {
        command: {
          kind: 'add_unit',
          unit: division('blue-2', 'blue', beside, 'II Corps', BLUE_NAME, BLUE_STRENGTH),
        } satisfies Command,
      },
    });

    // Against the bytes. The engine holds the observed formation on every contact — it has
    // to, or it could not tell this evening's column from this morning's — so the only
    // thing standing between that and the wire is `publicContact`, and this is the
    // assertion that says it ran.
    const raw = (await viewAs(f, f.ney)).body;
    expect(raw).not.toContain('blue-2');
    expect(raw).not.toContain('"unitId":"blue');
  });

  it('keeps a sighting after the enemy has marched out of view', async () => {
    const f = await setUp();

    const beside = { q: RED_HEX.q + 1, r: RED_HEX.r };
    const run = (command: Command) =>
      f.app.inject({
        method: 'POST',
        url: `/api/campaigns/${f.id}/commands`,
        headers: { 'x-campaign-token': f.referee },
        payload: { command },
      });

    await run({
      kind: 'add_unit',
      unit: division('blue-2', 'blue', beside, 'II Corps', BLUE_NAME, BLUE_STRENGTH),
    });
    const seen = (await viewAs(f, f.ney)).json().contacts;
    expect(seen).toHaveLength(1);

    // Teleported clean across the map, well outside anybody's recon zone.
    await run({ kind: 'teleport_unit', unitId: 'blue-2', column: [BLUE_HEX] });

    const after = (await viewAs(f, f.ney)).json().contacts;
    // Still there, and still saying where it was. A contact that vanished when the enemy
    // walked away would mean a commander's map could only ever show what their pickets can
    // see this second — which is the opposite of a fog-of-war map.
    expect(after).toHaveLength(1);
    expect(after[0].coord).toEqual(beside);
    expect(after[0].id).toBe(seen[0].id);
  });
});

describe('with terrain fog on', () => {
  // Switched off by default rather than removed, and this is what makes that a claim with
  // a test behind it. When the issued map lands, these are the assertions it inherits.
  let f: Fixture;
  beforeEach(async () => {
    f = await setUp({ terrainFog: true });
  });

  it('keeps every hex, so all commanders draw the same canvas', async () => {
    const view = (await viewAs(f, f.ney)).json();
    expect(view.world.hexes.length).toBe(world.hexes.size);
  });

  it('never puts unsurveyed ground in the payload as terrain', async () => {
    const view = (await viewAs(f, f.ney)).json();
    const surveyed = new Set<string>(view.surveyed);

    let fogged = 0;
    for (const h of view.world.hexes as {
      q: number;
      r: number;
      tags: string[];
      elevation: number;
      biome: string | null;
    }[]) {
      const k = key({ q: h.q, r: h.r });
      if (surveyed.has(k)) continue;
      fogged++;
      expect(h.tags, `${k} is not tagged fog`).toContain('fog');
      expect(h.elevation, `${k} carries a real elevation`).toBe(0);
      expect(h.biome).toBeNull();
    }
    expect(fogged, 'the fixture should leave most of the map unsurveyed').toBeGreaterThan(500);
  });

  it('leaves the ground the enemy stands on fogged', async () => {
    const view = (await viewAs(f, f.ney)).json();
    const hex = (view.world.hexes as { q: number; r: number; tags: string[] }[]).find(
      (h) => h.q === BLUE_HEX.q && h.r === BLUE_HEX.r,
    );
    expect(hex, 'the hex must still be present').toBeDefined();
    expect(hex!.tags).toContain('fog');
    expect(view.surveyed).not.toContain(key(BLUE_HEX));
  });

  it('does not leak the far side of the map through roads or settlements', async () => {
    const view = (await viewAs(f, f.ney)).json();
    const surveyed = new Set<string>(view.surveyed);

    for (const s of view.world.settlements as { coord: [number, number] }[]) {
      expect(surveyed.has(key({ q: s.coord[0], r: s.coord[1] }))).toBe(true);
    }
    for (const e of view.world.road_edges as { a: [number, number]; b: [number, number] }[]) {
      expect(surveyed.has(key({ q: e.a[0], r: e.a[1] }))).toBe(true);
      expect(surveyed.has(key({ q: e.b[0], r: e.b[1] }))).toBe(true);
    }
    for (const h of view.world.hexes as { q: number; r: number; road_connections: number[][] }[]) {
      for (const c of h.road_connections) {
        expect(surveyed.has(key({ q: c[0]!, r: c[1]! }))).toBe(true);
      }
    }
  });

  it('gives two commanders on the same side different ground', async () => {
    const ney = (await viewAs(f, f.ney)).json();
    const kellermann = (await viewAs(f, f.kellermann)).json();
    expect(new Set(ney.surveyed)).not.toEqual(new Set(kellermann.surveyed));
  });
});

describe('authorisation', () => {
  let f: Fixture;
  beforeEach(async () => {
    f = await setUp();
  });

  it('refuses a request with no token', async () => {
    const res = await f.app.inject({ method: 'GET', url: `/api/campaigns/${f.id}/view` });
    expect(res.statusCode).toBe(401);
  });

  it('refuses a token from another campaign', async () => {
    const other = await setUp();
    const res = await f.app.inject({
      method: 'GET',
      url: `/api/campaigns/${f.id}/view`,
      headers: { 'x-campaign-token': other.ney },
    });
    expect(res.statusCode).toBe(401);
  });

  it('refuses a made-up token', async () => {
    expect((await viewAs(f, 'not-a-real-token')).statusCode).toBe(401);
  });

  it('refuses a commander the referee routes', async () => {
    const commands = await f.app.inject({
      method: 'POST',
      url: `/api/campaigns/${f.id}/commands`,
      headers: { 'x-campaign-token': f.ney },
      payload: { command: { kind: 'advance_clock', hours: 3 } satisfies Command },
    });
    expect(commands.statusCode).toBe(403);

    const advance = await f.app.inject({
      method: 'POST',
      url: `/api/campaigns/${f.id}/advance`,
      headers: { 'x-campaign-token': f.ney },
      payload: { hours: 3 },
    });
    expect(advance.statusCode).toBe(403);
  });

  it('refuses a commander the power to mint join links', async () => {
    // Otherwise anybody could issue themselves a seat on the other side of the map.
    const res = await f.app.inject({
      method: 'POST',
      url: `/api/campaigns/${f.id}/commanders/wellington/token`,
      headers: { 'x-campaign-token': f.ney },
    });
    expect(res.statusCode).toBe(403);
  });

  it("does not let a commander export through another commander's eyes", async () => {
    const res = await f.app.inject({
      method: 'GET',
      url: `/api/campaigns/${f.id}/export?commander=wellington`,
      headers: { 'x-campaign-token': f.ney },
    });
    expect(res.statusCode).toBe(200);
    expect(res.body).not.toContain('blue-1');
  });

  it('revokes a seat, and the link stops working', async () => {
    expect((await viewAs(f, f.ney)).statusCode).toBe(200);

    const revoked = await f.app.inject({
      method: 'DELETE',
      url: `/api/campaigns/${f.id}/commanders/ney/token`,
      headers: { 'x-campaign-token': f.referee },
    });
    expect(revoked.statusCode).toBe(200);
    expect((await viewAs(f, f.ney)).statusCode).toBe(401);
  });

  it('accepts a token by cookie, header, bearer or query', async () => {
    for (const inject of [
      { headers: { 'x-campaign-token': f.ney } },
      { headers: { authorization: `Bearer ${f.ney}` } },
      { headers: { cookie: `${TOKEN_COOKIE}=${f.ney}` } },
      { url: `/api/campaigns/${f.id}/view?token=${f.ney}` },
    ]) {
      const res = await f.app.inject({
        method: 'GET',
        url: `/api/campaigns/${f.id}/view`,
        ...inject,
      });
      expect(res.statusCode).toBe(200);
    }
  });

  it('exchanges a join link for a cookie', async () => {
    const res = await f.app.inject({ method: 'GET', url: `/j/${f.id}/${f.ney}` });
    expect(res.statusCode).toBe(200);
    expect(res.cookies.some((c) => c.name === TOKEN_COOKIE)).toBe(true);
  });
});

/**
 * The despatch surface, where a leak would now be both invisible and total.
 *
 * A rider takes the least-time path to where the addressee **actually** is, so their route
 * is computed from ground truth. Show a commander that route and you have told them where
 * their detached corps stands — and they would never read a report again, because their own
 * outbox would be a better source than any of them.
 */
describe('despatches', () => {
  let f: Fixture;
  beforeEach(async () => {
    f = await setUp();
  });

  const post = (token: string, command: Command) =>
    f.app.inject({
      method: 'POST',
      url: `/api/campaigns/${f.id}/commands`,
      headers: { 'x-campaign-token': token },
      payload: { command },
    });

  const ORDER = 'Move on Quatre Bras with all speed; I expect you astride the crossroads.';

  const write = (
    token: string,
    from: string,
    to: string,
    text = ORDER,
    despatchKind: 'order' | 'report' = 'order',
  ) => post(token, { kind: 'send_despatch', from, to, despatchKind, body: { text } });

  const advance = (hours: number) =>
    f.app.inject({
      method: 'POST',
      url: `/api/campaigns/${f.id}/advance`,
      headers: { 'x-campaign-token': f.referee },
      payload: { hours },
    });

  it('never sends a commander a route, a fate, or a rider position', async () => {
    expect((await write(f.ney, 'ney', 'kellermann')).statusCode).toBe(200);

    const res = await viewAs(f, f.ney);
    const view = res.json();
    expect(view.sent).toHaveLength(1);
    expect(view.sent[0].body.text).toBe(ORDER);

    // Against the bytes, not the object graph: an enumerable field added to `Despatch`
    // later, or a `toJSON`, would sail straight past a shaped assertion.
    expect(res.body).not.toContain('"route"');
    expect(res.body).not.toContain('"fate"');
    expect(res.body).not.toContain('"progress"');
    // And no ETA, ever. An estimate is a distance, and a distance is a position.
    // Quoted, because "eta" unquoted matches `metadata` in the world document.
    expect(res.body).not.toContain('"eta"');
    expect(res.body).not.toContain('"etaHours"');
  });

  it('tells the sender nothing about whether it arrived', async () => {
    await write(f.ney, 'ney', 'kellermann');
    const before = (await viewAs(f, f.ney)).json().sent;

    await advance(200);

    const after = (await viewAs(f, f.ney)).json().sent;
    // Whatever became of the rider — delivered, lost, or read by the enemy — Ney's
    // outbox says exactly what it said the moment they sealed it. That is the mechanic.
    expect(after).toEqual(before);
    expect(after[0].acknowledged).toBe(false);
  });

  it('keeps a despatch out of the addressee’s hands until it arrives', async () => {
    await write(f.ney, 'ney', 'kellermann');

    const res = await viewAs(f, f.kellermann);
    expect(res.json().received).toEqual([]);
    // Not merely absent from the ledger — absent from the payload. A commander must not
    // be able to read their orders early by opening the developer tools.
    expect(res.body).not.toContain('Quatre Bras');
  });

  it('delivers it eventually, dated with the hour it was written', async () => {
    await write(f.ney, 'ney', 'kellermann');
    await advance(200);

    const view = (await viewAs(f, f.kellermann)).json();
    expect(view.received).toHaveLength(1);
    expect(view.received[0].body.text).toBe(ORDER);
    // The hour it describes and the hour it arrived are different numbers, and both are
    // on the page. The gap between them is the whole of the fog.
    expect(view.received[0].sentAtHours).toBeLessThan(view.received[0].receivedAtHours);
  });

  it('gives the referee the route, because that is what a referee is for', async () => {
    await write(f.ney, 'ney', 'kellermann');
    const res = await viewAs(f, f.referee);

    expect(res.json().despatches).toHaveLength(1);
    expect(res.body).toContain('"route"');
    expect(res.json().despatches[0].route.length).toBeGreaterThan(1);
  });

  it('sends no decision queue and no tasks to a commander', async () => {
    await write(f.ney, 'ney', 'kellermann');
    await advance(200);

    const view = (await viewAs(f, f.kellermann)).json();
    expect(view.despatches).toEqual([]);
    expect(view.decisions).toEqual([]);
    expect(view.tasks).toEqual([]);
  });

  it('writes in the name of the token, not the name in the payload', async () => {
    // Kellermann, claiming to be Ney. Forgery would be bad enough; a forged *report*
    // would let anyone feed a commander false intelligence signed by their own subordinate.
    const res = await write(f.kellermann, 'ney', 'kellermann', 'Fall back at once.');
    expect(res.statusCode).toBe(409);

    expect((await viewAs(f, f.referee)).json().despatches).toEqual([]);
  });

  it('refuses every other command from a commander', async () => {
    const res = await post(f.ney, { kind: 'set_task', unitId: 'red-2', destination: RED_HEX });
    expect(res.statusCode).toBe(403);
  });

  it('refuses an order sent sideways under strict rules', async () => {
    // Kellermann does not command Ney. It is a message, not an order — and a referee who
    // wants it to be one can force it, which is why this is soft rather than hard.
    const res = await write(f.kellermann, 'kellermann', 'ney');
    expect(res.statusCode).toBe(409);
    expect(res.json().violations[0].code).toBe('not_in_command');
  });

  /**
   * The fog that carries the game, now that the ground is public.
   *
   * A commander's picture of their own detached corps has to be capable of being *wrong* —
   * not merely delayed in some abstract sense, but showing a hex the formation is no
   * longer standing on. If this ever passes trivially, the reports have gone back to being
   * snapshotted live and the design has quietly stopped working.
   */
  it('lets a commander’s picture of their own corps go stale, and wrong', async () => {
    const far = [...world.hexes.values()]
      .filter((h) => h.terrainClass === 'land')
      .map((h) => h.coord)
      .find((c) => dist(c, SUB_HEX) > 6 && dist(c, SUB_HEX) < 14)!;

    // Kellermann's division marches. Ney, forty kilometres away, is told nothing.
    await post(f.referee, { kind: 'set_task', unitId: 'red-2', destination: far });
    await advance(12);

    const truth = (await viewAs(f, f.referee)).json();
    const actual = truth.units.find((u: Unit) => u.id === 'red-2').column[0];
    expect(actual, 'the division never moved, so this test proves nothing').not.toEqual(
      SUB_HEX,
    );

    const ney = (await viewAs(f, f.ney)).json();
    const report = ney.reports.find((r: UnitReport) => r.unitId === 'red-2');

    expect(report.atHours).toBe(0);
    expect(report.head).toEqual(SUB_HEX);
    // The whole of it: they are looking at a hex their division left hours ago, and nothing in
    // their payload tells them where it actually is.
    expect(report.head).not.toEqual(actual);
    // Nothing else in their payload knows better either: the report is their only record of
    // that formation, and there is no live unit beside it to contradict it.
    expect(ney.units.map((u: Unit) => u.id)).toEqual(['red-1']);
  });

  it('refreshes that picture when a despatch arrives from the commander themselves', async () => {
    await post(f.referee, { kind: 'set_task', unitId: 'red-2', destination: SUB_HEX });
    await advance(6);

    // Kellermann writes to Ney. The rider carries word of where Kellermann stood when they
    // sealed it, whether or not they thought to mention it.
    // A report, not an order: Kellermann does not command Ney, and writing upward is
    // exactly what a report is for.
    const sent = await write(f.kellermann, 'kellermann', 'ney', 'All quiet here.', 'report');
    expect(sent.statusCode, sent.body).toBe(200);
    await advance(200);

    const ney = (await viewAs(f, f.ney)).json();
    const report = ney.reports.find((r: UnitReport) => r.unitId === 'red-2');
    // Dated when it was written, not when it landed. The gap is the fog.
    expect(report.atHours).toBeGreaterThan(0);
    expect(report.atHours).toBeLessThan(ney.campaign.clockHours);
  });

  it('refuses a despatch to the other side outright', async () => {
    const res = await write(f.ney, 'ney', 'wellington');
    expect(res.statusCode).toBe(409);
    expect(
      res.json().violations.some((v: { code: string }) => v.code === 'wrong_faction'),
    ).toBe(true);
  });
});

/**
 * The referee's two controls, over the wire.
 *
 * "Run until something needs me" is the one a referee actually uses, and the answer to
 * *what* needs them belongs in the reply rather than in a second request they have to know to
 * make. The rest of this file is about what must not be sent; this is about the one role
 * that is entitled to all of it.
 */
describe('the referee console', () => {
  let f: Fixture;
  beforeEach(async () => {
    f = await setUp();
  });

  const run = (hours: number, untilDecision: boolean) =>
    f.app.inject({
      method: 'POST',
      url: `/api/campaigns/${f.id}/advance`,
      headers: { 'x-campaign-token': f.referee },
      payload: { hours, untilDecision },
    });

  const command = (token: string, cmd: Command) =>
    f.app.inject({
      method: 'POST',
      url: `/api/campaigns/${f.id}/commands`,
      headers: { 'x-campaign-token': token },
      payload: { command: cmd },
    });

  it('stops the clock and says what stopped it', async () => {
    // A march that ends somewhere is the simplest thing that raises a decision: the
    // column arrives, and arriving is something a referee has to have an opinion about.
    const near = [...world.hexes.values()]
      .filter((h) => h.terrainClass === 'land')
      .map((h) => h.coord)
      .find((c) => dist(c, RED_HEX) > 2 && dist(c, RED_HEX) < 5)!;

    await command(f.referee, { kind: 'set_task', unitId: 'red-1', destination: near });

    const res = await run(48, true);
    expect(res.statusCode, res.body).toBe(200);

    const halted = res.json().halted;
    expect(halted).not.toBeNull();
    expect(halted.unitId).toBe('red-1');
    // And it really stopped rather than running the full forty-eight hours it was offered.
    expect(res.json().clockHours).toBeLessThan(48);
  });

  it('runs the clock through a decision when it was not asked to stop', async () => {
    const near = [...world.hexes.values()]
      .filter((h) => h.terrainClass === 'land')
      .map((h) => h.coord)
      .find((c) => dist(c, RED_HEX) > 2 && dist(c, RED_HEX) < 5)!;

    await command(f.referee, { kind: 'set_task', unitId: 'red-1', destination: near });

    const res = await run(12, false);
    expect(res.json().clockHours).toBe(12);
    // The decision was still raised — it is in their queue — but the clock did not care.
    expect(res.json().halted).toBeNull();
    expect((await viewAs(f, f.referee)).json().decisions.length).toBeGreaterThan(0);
  });

  it('marks a decision dealt with', async () => {
    const near = [...world.hexes.values()]
      .filter((h) => h.terrainClass === 'land')
      .map((h) => h.coord)
      .find((c) => dist(c, RED_HEX) > 2 && dist(c, RED_HEX) < 5)!;

    await command(f.referee, { kind: 'set_task', unitId: 'red-1', destination: near });
    const halted = (await run(48, true)).json().halted;

    const done = await command(f.referee, {
      kind: 'resolve_decision',
      decisionId: halted.id,
      note: 'Told them to hold there.',
    });
    expect(done.statusCode, done.body).toBe(200);

    const queue = (await viewAs(f, f.referee)).json().decisions;
    const resolved = queue.find((d: { id: string }) => d.id === halted.id);
    // Kept rather than deleted: the queue is a history as well as a workload, and why a
    // referee decided something is the most interesting line in an after-action review.
    expect(resolved.resolvedAtHours).not.toBeNull();
    expect(resolved.note).toBe('Told them to hold there.');
  });

  it('lets nobody but the referee run the clock', async () => {
    const res = await f.app.inject({
      method: 'POST',
      url: `/api/campaigns/${f.id}/advance`,
      headers: { 'x-campaign-token': f.ney },
      payload: { hours: 1 },
    });
    expect(res.statusCode).toBe(403);
  });
});

describe('a referee writing in a commander’s name', () => {
  let f: Fixture;
  beforeEach(async () => {
    f = await setUp();
  });

  const post = (token: string, command: Command) =>
    f.app.inject({
      method: 'POST',
      url: `/api/campaigns/${f.id}/commands`,
      headers: { 'x-campaign-token': token },
      payload: { command },
    });

  const viewFor = async (token: string) =>
    JSON.parse(
      (
        await f.app.inject({
          method: 'GET',
          url: `/api/campaigns/${f.id}/view`,
          headers: { 'x-campaign-token': token },
        })
      ).body,
    ) as { despatches: { from: string; body: { text?: string } }[] };

  const logFor = async (token: string) =>
    JSON.parse(
      (
        await f.app.inject({
          method: 'GET',
          url: `/api/campaigns/${f.id}/log`,
          headers: { 'x-campaign-token': token },
        })
      ).body,
    ) as { actor: { kind: string }; payload: Record<string, never> }[];

  it('sends it from the commander they named', async () => {
    // Their ordinary work: they run most of the commanders on the map, and takes dictation
    // from the players who hold the rest.
    const res = await post(f.referee, {
      kind: 'send_despatch',
      from: 'ney',
      to: 'kellermann',
      despatchKind: 'order',
      body: { text: 'Move on the crossroads.' },
    });
    expect(res.statusCode).toBe(200);

    const view = await viewFor(f.referee);
    const sent = view.despatches.find((d) => d.body.text === 'Move on the crossroads.');
    expect(sent).toBeDefined();
    expect(sent!.from).toBe('ney');
  });

  it('records that it was the referee who wrote it', async () => {
    // The despatch is from the commander; the event is from the referee. An after-action
    // review has to be able to tell a commander's own order from one written for them.
    await post(f.referee, {
      kind: 'send_despatch',
      from: 'ney',
      to: 'kellermann',
      despatchKind: 'order',
      body: { text: 'By your hand, not mine.' },
    });

    const log = await logFor(f.referee);
    const event = log.find((e) => {
      const p = e.payload as unknown as {
        kind: string;
        despatch?: { from: string; body: { text?: string } };
      };
      return p.kind === 'despatch_sent' && p.despatch?.body.text === 'By your hand, not mine.';
    });
    expect(event).toBeDefined();
    expect(event!.actor).toEqual({ kind: 'referee' });
    expect(
      (event!.payload as unknown as { despatch: { from: string } }).despatch.from,
    ).toBe('ney');
  });

  it('still ignores a commander who claims to be somebody else', async () => {
    // The rule that makes the referee's power safe to grant: a seat writes as itself and
    // nothing else, whatever the payload says. A forged report would let anyone feed a
    // commander false intelligence signed by their own subordinate.
    const res = await post(f.ney, {
      kind: 'send_despatch',
      from: 'wellington',
      to: 'kellermann',
      despatchKind: 'order',
      body: { text: 'Fall back at once.' },
    });
    expect(res.statusCode).toBe(200);

    const view = await viewFor(f.referee);
    const sent = view.despatches.find((d) => d.body.text === 'Fall back at once.');
    expect(sent!.from).toBe('ney');
  });

  it('refuses to write across the lines, whoever asks', async () => {
    // Hard, and hard for the referee too: writing to the other side is not a despatch, and
    // letting it through would produce state neither the inbox nor the fog can describe.
    const res = await post(f.referee, {
      kind: 'send_despatch',
      from: 'ney',
      to: 'wellington',
      despatchKind: 'order',
      body: { text: 'Surrender.' },
    });
    expect(res.statusCode).toBe(409);
  });
});
