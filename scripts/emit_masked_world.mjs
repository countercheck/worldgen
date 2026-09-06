/**
 * Mask the fixture world down to one faction's knowledge and print it to stdout.
 *
 * A bridge for `tests/test_masked_world.py`, which loads the result with the real
 * `WorldState.from_dict`. The claim that a commander's map is still a legal `world.json`
 * is only worth something if the Python actually reads one, and only the Python can say
 * whether it does.
 *
 * Imports the built output rather than the sources: the TypeScript uses `.js` import
 * specifiers, which bare Node cannot resolve against `.ts` files even with type
 * stripping. Run `npm --prefix campaign run build` first — the Python test does.
 *
 *     node scripts/emit_masked_world.mjs [radius] [--no-remember] > fog.json
 */

import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const shared = join(here, '..', 'campaign', 'shared', 'dist');

const { key } = await import(join(shared, 'hex.js'));
const { maskWorld } = await import(join(shared, 'mask.js'));

const fixture = join(here, '..', 'campaign', 'shared', 'test', 'fixtures', 'world-32x32.json');
const doc = JSON.parse(readFileSync(fixture, 'utf8'));

const radius = Number(process.argv[2] ?? 6);
const remember = !process.argv.includes('--no-remember');

const centre = { q: 16, r: 16 };
const dist = (q, r) =>
  (Math.abs(q - centre.q) + Math.abs(r - centre.r) + Math.abs(q + r - centre.q - centre.r)) / 2;

const seen = new Set();
const visible = new Set();
for (const h of doc.hexes) {
  const d = dist(h.q, h.r);
  if (d <= radius) seen.add(key({ q: h.q, r: h.r }));
  if (d <= Math.max(1, Math.floor(radius / 3))) visible.add(key({ q: h.q, r: h.r }));
}

process.stdout.write(
  JSON.stringify(maskWorld(doc, { seen, visible, faction: 'red', clockHours: 24, remember })),
);
