/**
 * What a client is owed, and the proof that it is enough.
 *
 * `ClientView.world` was typed `unknown`, so nothing said what crossed the boundary and
 * the answer drifted to "the whole generated document" — twenty-four fields a hex, twelve
 * of which the engine has never read. On a 200x200 map that is 23.6 MB to every browser on
 * every view, and again down every socket on every clock tick.
 *
 * The contract is not a new invention: `parseWorld` already defines it, by reading twelve
 * fields and ignoring the rest. So the test of the projection is not a field list somebody
 * agreed with — it is that the engine cannot tell the difference.
 */

import { readFileSync } from 'node:fs';

import { describe, expect, it } from 'vitest';

import { DEFAULT_CONFIG } from '../src/config.js';
import { EMPTY_STATE } from '../src/state.js';
import { viewFor } from '../src/view.js';
import { parseWorld, projectWorld, WORLD_HEX_FIELDS } from '../src/world.js';

const FIXTURE = new URL('./fixtures/world-32x32.json', import.meta.url);

const doc = (): unknown => JSON.parse(readFileSync(FIXTURE, 'utf8'));

/**
 * A `World` flattened for comparison, with every measurement rounded to the precision the
 * projection sends at.
 *
 * Both sides are rounded, which is the honest form of the claim. The projection *is* lossy
 * — it sends `elevation` to three decimals instead of seventeen, and that is most of why
 * the compressed payload halves, because full-precision floats are incompressible noise.
 * What it does not lose is anything above a millimetre of elevation or a thousandth of a
 * degree of slope, and nothing in the rules reads below that. Rounding both sides asserts
 * exactly that and nothing weaker: a dropped field or a reordered structure still fails.
 */
const PRECISION = 3;

const round = (v: unknown): unknown =>
  typeof v === 'number' && !Number.isInteger(v) ? Number(v.toFixed(PRECISION)) : v;

const comparable = (raw: unknown) => {
  const w = parseWorld(raw);
  return {
    ...w,
    hexes: [...w.hexes.entries()]
      .sort(([a], [b]) => (a < b ? -1 : 1))
      .map(([k, h]) => [
        k,
        {
          ...Object.fromEntries(Object.entries(h).map(([f, v]) => [f, round(v)])),
          tags: [...h.tags].sort(),
        },
      ]),
    roadEdges: [...w.roadEdges.entries()].sort(([a], [b]) => (a < b ? -1 : 1)),
    seaEdges: [...w.seaEdges.entries()].sort(([a], [b]) => (a < b ? -1 : 1)),
  };
};

describe('projecting a world for a client', () => {
  it('parses to the same world it would have without the projection', () => {
    // The whole argument in one assertion. Every field dropped is one `parseWorld` never
    // looked at, so a `World` built from the projection is indistinguishable from a
    // `World` built from the full document — and the engine only ever sees a `World`.
    expect(comparable(projectWorld(doc()))).toEqual(comparable(doc()));
  });

  it('drops the generator fields the engine has never read', () => {
    const before = new Set(Object.keys((doc() as { hexes: object[] }).hexes[0]!));
    const after = new Set(
      Object.keys((projectWorld(doc()) as { hexes: object[] }).hexes[0]!),
    );

    const dropped = [...before].filter((k) => !after.has(k)).sort();
    // Named rather than counted, so that a field moving from ignored to read is a
    // deliberate edit here rather than a silent change in payload.
    expect(dropped).toEqual([
      'alluvium',
      'cultivated',
      'habitability_city',
      'habitability_town',
      'habitability_village',
      'land_use',
      'moisture',
      'rural_population',
      'soil',
      'temperature',
      'territory',
      'territory_cost',
    ]);
  });

  it('sends nothing that is not in the declared list', () => {
    const declared = new Set<string>(WORLD_HEX_FIELDS);
    for (const hex of (projectWorld(doc()) as { hexes: object[] }).hexes) {
      for (const field of Object.keys(hex)) {
        expect(declared.has(field), `undeclared field ${field}`).toBe(true);
      }
    }
  });

  it('keeps everything above the hex array whole', () => {
    // Rivers, settlements, road and sea edges, ferries and metadata are all read, and
    // together they are a fraction of the payload. Dropping a hex field is a measurement
    // nobody takes; dropping a river is a map that is wrong.
    const before = doc() as Record<string, unknown>;
    const after = projectWorld(doc()) as Record<string, unknown>;
    for (const k of Object.keys(before)) {
      if (k === 'hexes') continue;
      expect(after[k], k).toEqual(before[k]);
    }
  });

  it('makes the document substantially smaller', () => {
    const bytes = (o: unknown) => JSON.stringify(o).length;
    const ratio = bytes(projectWorld(doc())) / bytes(doc());
    // Measured at 0.38 on a real 200x200 map. Asserted loosely, because the exact figure
    // moves with terrain; a projection that stopped saving anything is the failure.
    expect(ratio).toBeLessThan(0.55);
  });

  it('reaches every role, including the referee', () => {
    // The referee's branch of `viewFor` set `world` straight from the stored document,
    // because no masking applies to him — so a projection wired only into the masking
    // path saved nothing at all for the role that loads the whole map most often. Checked
    // per role rather than once, since each builds its view separately.
    const world = parseWorld(doc());
    const state = { ...EMPTY_STATE, factions: new Map(), commanders: new Map() };
    const input = {
      campaignId: 'c',
      state,
      worldDoc: doc(),
      world,
      cfg: DEFAULT_CONFIG,
      ruleset: 'standard',
    };

    for (const role of [{ kind: 'referee' } as const]) {
      const sent = viewFor(input, role).world as { hexes: object[] };
      const declared = new Set<string>(WORLD_HEX_FIELDS);
      for (const field of Object.keys(sent.hexes[0]!)) {
        expect(declared.has(field), `${role.kind} was sent ${field}`).toBe(true);
      }
    }
  });

  it('leaves a document it does not recognise alone', () => {
    expect(projectWorld(null)).toBeNull();
    expect(projectWorld({ version: '1.8' })).toEqual({ version: '1.8' });
  });
});
