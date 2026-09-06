/**
 * The tests that matter most in this project.
 *
 * Everything else here is a wargame; this is the part where the wargame is about
 * incomplete information. One careless `reply.send(state)` and a commander's browser
 * holds the entire map and every enemy position, and no amount of correct rules makes up
 * for it.
 *
 * Two deliberate choices about how these are written:
 *
 * **They assert on the serialised response body, not the object graph.** A structural
 * assertion over a parsed response is convincing and insufficient: a getter, an
 * enumerable field added later, or a `toJSON` can put data on the wire that a shaped
 * assertion never looks at. So the tests take the raw payload string and search it for
 * things that must not be in it.
 *
 * **They assert absence, not presence.** It is easy to check that red sees its own
 * division. The bug that ends the game is blue's division being in the payload too, in
 * some field nobody thought to look at.
 */

import { beforeEach, describe, expect, it } from 'vitest';

import worldDoc from '../../shared/test/fixtures/world-32x32.json';

import {
  key,
  KIND_DEFAULTS,
  parseWorld,
  type Command,
  type Hex,
  type Unit,
} from '@campaign/shared';

import { buildApp, TOKEN_COOKIE } from '../src/app.js';
import { openDb } from '../src/db.js';

const world = parseWorld(worldDoc);

/** Two well-separated land hexes, so neither faction can see the other at the start. */
function farApartLand(): [Hex, Hex] {
  const land = [...world.hexes.values()]
    .filter((h) => h.terrainClass === 'land')
    .map((h) => h.coord);

  let best: [Hex, Hex] = [land[0]!, land[1]!];
  let bestDist = -1;
  for (const a of land) {
    for (const b of land) {
      const d =
        (Math.abs(a.q - b.q) + Math.abs(a.r - b.r) + Math.abs(a.q + a.r - (b.q + b.r))) / 2;
      if (d > bestDist) {
        bestDist = d;
        best = [a, b];
      }
    }
  }
  return best;
}

const [RED_HEX, BLUE_HEX] = farApartLand();

/**
 * Distinct strengths per side, on purpose.
 *
 * These tests search the raw response for things that must not be in it, which only
 * discriminates if the value is unique to one faction. Giving both sides 5,000 effectives
 * made an early version of the strength assertion pass against red's own division while
 * proving nothing about blue's.
 */
const RED_STRENGTH = 5000;
const BLUE_STRENGTH = 7777;

/** Blue's name, which red must never read. Distinctive so a byte search discriminates. */
const BLUE_NAME = 'Erzherzog Karl Grenadiers';

function division(id: string, faction: string, at: Hex, corps: string): Unit {
  return {
    id,
    name: faction === 'blue' ? BLUE_NAME : 'Division Morand',
    faction,
    kind: 'infantry',
    effectives: faction === 'blue' ? BLUE_STRENGTH : RED_STRENGTH,
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
    column: [at],
    hoursMarchedToday: 0,
    corps,
  };
}

interface Fixture {
  app: ReturnType<typeof buildApp>;
  id: string;
  referee: string;
  red: string;
  blue: string;
}

async function setUp(): Promise<Fixture> {
  const app = buildApp({ db: openDb() });

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
  const body = created.json();

  const add = async (unit: Unit): Promise<void> => {
    const res = await app.inject({
      method: 'POST',
      url: '/api/campaigns/test/commands',
      headers: { 'x-campaign-token': body.refereeToken },
      payload: { command: { kind: 'add_unit', unit } satisfies Command },
    });
    expect(res.statusCode, res.body).toBe(200);
  };

  await add(division('red-1', 'red', RED_HEX, 'I Corps'));
  await add(division('blue-1', 'blue', BLUE_HEX, 'Grand Army of the Danube'));

  return {
    app,
    id: 'test',
    referee: body.refereeToken,
    red: body.factionTokens.red,
    blue: body.factionTokens.blue,
  };
}

/**
 * Move blue next to red, so red actually spots it.
 *
 * Every other test here has the two sides at opposite ends of the map, which exercises
 * only the easy half of the problem: nothing has been observed, so nothing may be sent.
 * The interesting half is what a sighting is allowed to carry, and that only happens when
 * somebody is looking at somebody.
 */
async function bringTogether(f: Fixture): Promise<void> {
  const res = await f.app.inject({
    method: 'POST',
    url: `/api/campaigns/${f.id}/commands`,
    headers: { 'x-campaign-token': f.referee },
    payload: {
      command: {
        kind: 'teleport_unit',
        unitId: 'blue-1',
        column: [RED_HEX],
      } satisfies Command,
    },
  });
  expect(res.statusCode, res.body).toBe(200);
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
    const res = await viewAs(f, f.referee);
    expect(res.statusCode).toBe(200);
    const view = res.json();
    expect(view.role).toBe('referee');
    expect(view.units.map((u: Unit) => u.id).sort()).toEqual(['blue-1', 'red-1']);
  });

  it('serves a commander only their own units', async () => {
    const view = (await viewAs(f, f.red)).json();
    expect(view.role).toBe('faction');
    expect(view.faction).toBe('red');
    expect(view.units.map((u: Unit) => u.id)).toEqual(['red-1']);
  });

  it('never puts an unseen enemy anywhere in the payload', async () => {
    // The assertion that matters, made against the raw bytes. Red has never been near
    // blue, so no trace of blue-1 — its id, its corps, its strength — may be on the wire,
    // in any field, including ones nobody thought to shape an assertion around.
    const raw = (await viewAs(f, f.red)).body;

    expect(raw).not.toContain('blue-1');
    expect(raw).not.toContain(BLUE_NAME);
    expect(raw).not.toContain('Grand Army of the Danube');
    expect(raw).not.toContain(`"effectives":${BLUE_STRENGTH}`);
  });

  it('leaves the ground the enemy stands on fogged', async () => {
    // The coordinate itself is not a leak and cannot be tested for: every hex is present
    // in a masked world, blanked and tagged, which is what keeps the canvas the same size
    // for everyone. What must be true is that this one carries nothing.
    const view = (await viewAs(f, f.red)).json();
    const hex = (view.world.hexes as { q: number; r: number; tags: string[] }[]).find(
      (h) => h.q === BLUE_HEX.q && h.r === BLUE_HEX.r,
    );

    expect(hex, 'the hex must still be present').toBeDefined();
    expect(hex!.tags).toContain('fog');
    expect(view.seen).not.toContain(key(BLUE_HEX));
  });

  it('never puts unseen ground in the payload as terrain', async () => {
    const view = (await viewAs(f, f.red)).json();
    const seen = new Set<string>(view.seen);

    let fogged = 0;
    for (const h of view.world.hexes as { q: number; r: number; tags: string[] }[]) {
      const k = key({ q: h.q, r: h.r });
      if (seen.has(k)) continue;
      fogged++;
      expect(h.tags, `${k} is not tagged fog`).toContain('fog');
      expect(h.elevation, `${k} carries a real elevation`).toBe(0);
      expect(h.biome).toBeNull();
    }
    expect(fogged, 'the fixture should leave most of the map unseen').toBeGreaterThan(500);
  });

  it('does not leak the far side of the map through roads or settlements', async () => {
    const view = (await viewAs(f, f.red)).json();
    const seen = new Set<string>(view.seen);

    for (const s of view.world.settlements as { coord: [number, number] }[]) {
      expect(seen.has(key({ q: s.coord[0], r: s.coord[1] }))).toBe(true);
    }
    for (const e of view.world.road_edges as { a: [number, number]; b: [number, number] }[]) {
      expect(seen.has(key({ q: e.a[0], r: e.a[1] }))).toBe(true);
      expect(seen.has(key({ q: e.b[0], r: e.b[1] }))).toBe(true);
    }
    for (const h of view.world.hexes as { q: number; r: number; road_connections: number[][] }[]) {
      for (const c of h.road_connections) {
        expect(seen.has(key({ q: c[0]!, r: c[1]! }))).toBe(true);
      }
    }
  });

  it('gives each commander a different picture', async () => {
    const red = (await viewAs(f, f.red)).json();
    const blue = (await viewAs(f, f.blue)).json();

    expect(red.seen).not.toEqual(blue.seen);
    expect(new Set(red.seen)).not.toEqual(new Set(blue.seen));
    // And neither sees the other's ground.
    const blueSeen = new Set<string>(blue.seen);
    expect((red.seen as string[]).some((k) => blueSeen.has(k))).toBe(false);
  });

  it('does not leak another faction in the log', async () => {
    // The log is a rich source of exactly what the fog withholds: every march, in order.
    const res = await f.app.inject({
      method: 'GET',
      url: `/api/campaigns/${f.id}/log`,
      headers: { 'x-campaign-token': f.red },
    });
    expect(res.body).not.toContain('blue-1');
    expect(res.body).not.toContain('Grand Army of the Danube');
  });

  it('gives the referee the whole log', async () => {
    const res = await f.app.inject({
      method: 'GET',
      url: `/api/campaigns/${f.id}/log`,
      headers: { 'x-campaign-token': f.referee },
    });
    expect(res.body).toContain('blue-1');
  });
});

describe('once an enemy is actually spotted', () => {
  it('reports it, and no better than the sighting earned', async () => {
    const f = await setUp();

    // Put a blue division right next to the red one.
    const beside = { q: RED_HEX.q + 1, r: RED_HEX.r };
    const res = await f.app.inject({
      method: 'POST',
      url: `/api/campaigns/${f.id}/commands`,
      headers: { 'x-campaign-token': f.referee },
      payload: {
        command: {
          kind: 'add_unit',
          unit: division('blue-2', 'blue', beside, 'II Corps'),
        } satisfies Command,
      },
    });
    expect(res.statusCode, res.body).toBe(200);

    const view = (await viewAs(f, f.red)).json();
    const contact = (view.contacts as { unitId: string; kind: null; corps: null }[]).find(
      (c) => c.unitId === 'blue-2',
    );

    expect(contact, 'a division one hex away should be seen').toBeDefined();
    // Presence and location only. Strength, arm and corps are not known from a sighting.
    expect(contact!.kind).toBeNull();
    expect(contact!.corps).toBeNull();

    const raw = (await viewAs(f, f.red)).body;
    expect(raw).not.toContain('II Corps');
    expect(raw).not.toContain(BLUE_NAME);
    // Its strength is not on the wire either.
    expect(raw).not.toContain(`"effectives":${BLUE_STRENGTH}`);
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
      headers: { 'x-campaign-token': other.red },
    });
    expect(res.statusCode).toBe(401);
  });

  it('refuses a made-up token', async () => {
    const res = await viewAs(f, 'not-a-real-token');
    expect(res.statusCode).toBe(401);
  });

  it('refuses a commander the referee routes', async () => {
    const commands = await f.app.inject({
      method: 'POST',
      url: `/api/campaigns/${f.id}/commands`,
      headers: { 'x-campaign-token': f.red },
      payload: { command: { kind: 'advance_clock', hours: 3 } satisfies Command },
    });
    expect(commands.statusCode).toBe(403);

    const advance = await f.app.inject({
      method: 'POST',
      url: `/api/campaigns/${f.id}/advance`,
      headers: { 'x-campaign-token': f.red },
      payload: { hours: 3 },
    });
    expect(advance.statusCode).toBe(403);
  });

  it('does not let a commander export another faction map', async () => {
    // The referee may ask for anyone's map; a commander may only have their own, whatever
    // they put in the query string.
    const res = await f.app.inject({
      method: 'GET',
      url: `/api/campaigns/${f.id}/export?faction=blue`,
      headers: { 'x-campaign-token': f.red },
    });
    expect(res.statusCode).toBe(200);
    expect(res.body).not.toContain('blue-1');

    const asBlue = (
      await f.app.inject({
        method: 'GET',
        url: `/api/campaigns/${f.id}/export`,
        headers: { 'x-campaign-token': f.blue },
      })
    ).json();
    const asRed = res.json();
    expect(asRed.metadata.fog.faction).toBe('red');
    expect(asBlue.metadata.fog.faction).toBe('blue');
  });

  it('accepts a token by cookie, header, bearer or query', async () => {
    for (const inject of [
      { headers: { 'x-campaign-token': f.red } },
      { headers: { authorization: `Bearer ${f.red}` } },
      { headers: { cookie: `${TOKEN_COOKIE}=${f.red}` } },
      { url: `/api/campaigns/${f.id}/view?token=${f.red}` },
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
    const res = await f.app.inject({ method: 'GET', url: `/j/${f.id}/${f.red}` });
    expect(res.statusCode).toBe(200);
    expect(res.cookies.some((c) => c.name === TOKEN_COOKIE)).toBe(true);
    // httpOnly, so a script on the page cannot read the token back out.
    expect(res.cookies.find((c) => c.name === TOKEN_COOKIE)?.httpOnly).toBe(true);
  });

  it('refuses an invalid join link', async () => {
    const res = await f.app.inject({ method: 'GET', url: `/j/${f.id}/nonsense` });
    expect(res.statusCode).toBe(401);
  });
});

describe('what a sighting is allowed to carry', () => {
  // Found by running the console rather than by any assertion here: with the two sides
  // apart, every leak test passes trivially because there is nothing to leak. These cover
  // the case where red is looking straight at blue.
  let f: Fixture;
  beforeEach(async () => {
    f = await setUp();
    await bringTogether(f);
  });

  it('reports the enemy as a contact rather than a unit', async () => {
    const view = (await viewAs(f, f.red)).json();
    expect(view.units.map((u: Unit) => u.id)).toEqual(['red-1']);
    expect(view.contacts.length).toBe(1);
  });

  it('carries none of the enemy record a sighting does not earn', async () => {
    // A sighting is a presence and a position. Strength, name and corps are patrol work,
    // and none of them may be on the wire merely because the formation was seen.
    const raw = (await viewAs(f, f.red)).body;

    expect(raw).not.toContain(BLUE_NAME);
    expect(raw).not.toContain('Grand Army of the Danube');
    expect(raw).not.toContain(`"effectives":${BLUE_STRENGTH}`);

    // Not `raw.includes('"morale"')`: red's own division has a morale, so searching the
    // whole payload for the field name would pass while proving nothing. The question is
    // what the contact itself carries.
    const contact = (await viewAs(f, f.red)).json().contacts[0];
    expect(Object.keys(contact).sort()).toEqual(
      ['corps', 'coord', 'faction', 'intelLevel', 'kind', 'seenAtHours', 'unitId'].sort(),
    );
  });

  it('does not report the enemy column, only where it was seen', async () => {
    // Knowing an enemy was in a village is not knowing how far back its baggage was
    // strung out. A contact is one coordinate and must stay one.
    const view = (await viewAs(f, f.red)).json();
    const contact = view.contacts[0];
    expect(contact.coord).toBeDefined();
    expect(contact.column).toBeUndefined();
  });
});
