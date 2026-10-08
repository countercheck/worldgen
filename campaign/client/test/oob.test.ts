/**
 * An order of battle read from YAML.
 *
 * The file is the referee's whole preparation, so what matters is that what it describes
 * is what the engine is sent: every formation, in an order the engine will accept, with the
 * same shape a formation raised by hand has. And that a mistake anywhere stops all of it,
 * with the place it is — a half-imported army is worse than none.
 */

import { readFileSync } from 'node:fs';

import { describe, expect, it } from 'vitest';

import {
  DEFAULT_CONFIG,
  ECHELONS,
  EMPTY_STATE,
  TRAITS,
  UNIT_KINDS,
  applyAll,
  isWater,
  key,
  parseWorld,
  type Command,
  type Unit,
} from '@campaign/shared';

import worldDoc from '../../shared/test/fixtures/world-32x32.json';
import { emptyDraft, unitFrom } from '../src/orbat.js';
import { parseOob, rehearse } from '../src/oob.js';
import { SAMPLE_OOB, SAMPLE_OOB_HREF } from '../src/sample.js';

const cfg = DEFAULT_CONFIG;
const world = parseWorld(worldDoc);
const example = readFileSync(new URL('../../examples/order-of-battle.yaml', import.meta.url), 'utf8');

const ok = (text: string, existing?: Parameters<typeof parseOob>[3]) => {
  const out = parseOob(text, world, cfg, existing);
  if (!out.ok) throw new Error(out.problems.join('\n'));
  return out;
};
const problems = (text: string): readonly string[] => {
  const out = parseOob(text, world, cfg);
  return out.ok ? [] : out.problems;
};

const units = (cmds: readonly Command[]): Unit[] =>
  cmds.flatMap((c) => (c.kind === 'add_unit' ? [c.unit] : []));

/** One side, one formation, with whatever the test changes. */
const one = (formation: string): string =>
  `sides:\n  - name: Red\n    army:\n${formation
    .trim()
    .split('\n')
    .map((l) => `      ${l}`)
    .join('\n')}\n`;

describe('the example order of battle', () => {
  const out = ok(example);

  it('is accepted in full by the engine at its strictest', () => {
    const run = applyAll(
      [
        {
          kind: 'create_campaign',
          name: 'Example',
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
        ...out.newFactions.map((faction) => ({ kind: 'add_faction' as const, faction })),
        ...out.commands,
      ],
      EMPTY_STATE,
      world,
      'strict',
    );
    expect(run.violations.map((v) => v.message)).toEqual([]);
    expect(run.state.units.size).toBe(units(out.commands).length);
    expect(run.state.commanders.size).toBe(run.state.units.size);
  });

  it('stands every formation on dry land', () => {
    for (const u of units(out.commands)) {
      const hex = world.hexes.get(key(u.column[0]!));
      expect(hex, `${u.id} is off the map`).toBeDefined();
      expect(isWater(hex!), `${u.id} stands in water`).toBe(false);
    }
  });

  it('lists every trait, arm and echelon in its header', () => {
    // So the file a referee copies from cannot fall behind the rules it is written for.
    const header = example.slice(0, example.indexOf('\nsides:'));
    for (const word of [...TRAITS, ...UNIT_KINDS, ...ECHELONS]) {
      expect(header, `the header does not mention ${word}`).toMatch(new RegExp(`\\b${word}\\b`));
    }
  });
});

describe('reading a formation', () => {
  it('raises exactly what the form would, given the same answers', () => {
    const out = ok(one('unit: 1re Division\ncommander: Ney\nat: {q: 14, r: 15}'));
    const [unit] = units(out.commands);
    const byHand = unitFrom(
      { ...emptyDraft('red'), id: '1re-division', name: '1re Division', commanderName: 'Ney' },
      cfg,
      { q: 14, r: 15 },
    );
    expect(unit).toEqual(byHand);
  });

  it('gives any formation the guns the file says, whatever its arm', () => {
    const out = ok(
      one(`
unit: Garde
commander: Ney
guns: 40
at: {q: 14, r: 15}
subordinates:
  - {unit: Hussards, commander: Lasalle, kind: cavalry, guns: 0, at: {q: 14, r: 15}}
  - {unit: Ligne, commander: Gérard, at: {q: 14, r: 15}}`),
    );
    const guns = Object.fromEntries(units(out.commands).map((u) => [u.name, u.guns]));
    // Left out, the arm's usual battery, exactly as the raise form gives it.
    expect(guns).toEqual({ Garde: 40, Hussards: 0, Ligne: 6 });
  });

  it('refuses guns that are not a whole number of 0 or more', () => {
    for (const bad of ['-2', '1.5', 'many']) {
      expect(problems(one(`unit: Garde\ncommander: Ney\nguns: ${bad}\nat: {q: 14, r: 15}`))).toEqual([
        expect.stringMatching(/`guns`/),
      ]);
    }
  });

  it('puts every superior ahead of those who answer to them', () => {
    const appointed = new Set<string>();
    for (const c of ok(example).commands) {
      if (c.kind !== 'add_commander') continue;
      if (c.commander.superiorId !== null) expect(appointed).toContain(c.commander.superiorId);
      appointed.add(c.commander.id);
    }
  });

  it('sends every unit before any commander, who must have one to ride with', () => {
    const kinds = ok(example).commands.map((c) => c.kind);
    expect(kinds.lastIndexOf('add_unit')).toBeLessThan(kinds.indexOf('add_commander'));
  });

  it('gives every unit and commander its own id, around what the campaign has', () => {
    const out = ok(
      one(`
unit: Garde
commander: Ney
at: {q: 14, r: 15}
subordinates:
  - {unit: Garde, commander: Ney, at: {q: 14, r: 15}}`),
      { factions: [], unitIds: new Set(['garde']), commanderIds: new Set() },
    );
    const ids = units(out.commands).map((u) => u.id);
    expect(new Set(ids).size).toBe(ids.length);
    expect(ids).not.toContain('garde');
  });

  it('adds to a side the campaign already has, found by its name', () => {
    const out = ok(one('unit: Garde\ncommander: Ney\nat: {q: 14, r: 15}'), {
      factions: [{ id: 'red-side', name: 'Red', color: '#ff0000' }],
      unitIds: new Set(),
      commanderIds: new Set(),
    });
    expect(out.newFactions).toEqual([]);
    expect(units(out.commands)[0]!.faction).toBe('red-side');
  });
});

describe('a file with mistakes in it', () => {
  it('names each mistake with where it is, and sends nothing', () => {
    const out = parseOob(
      one('unit: Garde\ncommander: Ney\nkind: dragoons\nat: {q: 99, r: 99}'),
      world,
      cfg,
    );
    expect(out.ok).toBe(false);
    if (out.ok) return;
    expect(out.problems).toHaveLength(2);
    for (const p of out.problems) expect(p).toMatch(/^Red › Garde: /);
    expect(out.problems.join()).toMatch(/dragoons/);
    expect(out.problems.join()).toMatch(/99, 99/);
  });

  it('wants somewhere to stand', () => {
    expect(problems(one('unit: Garde\ncommander: Ney'))).toEqual([
      expect.stringMatching(/`at`/),
    ]);
  });

  it('wants a commander', () => {
    expect(problems(one('unit: Garde\nat: {q: 14, r: 15}'))).toEqual([
      expect.stringMatching(/`commander`/),
    ]);
  });

  it('refuses a colour the map cannot draw', () => {
    const text = `sides:\n  - name: Red\n    color: red\n    army: {unit: G, commander: N, at: {q: 14, r: 15}}\n`;
    expect(problems(text)).toEqual([expect.stringMatching(/not a colour/)]);
  });

  it('catches a misspelt key rather than ignoring it', () => {
    expect(problems(one('unit: Garde\ncommander: Ney\nstrenght: 9000\nat: {q: 14, r: 15}'))).toEqual(
      [expect.stringMatching(/strenght/)],
    );
  });

  it('catches an unknown trait', () => {
    expect(
      problems(one('unit: Garde\ncommander: Ney\ntraits: [scout, sneaky]\nat: {q: 14, r: 15}')),
    ).toEqual([expect.stringMatching(/sneaky/)]);
  });

  it('says so when the file is not YAML at all', () => {
    expect(problems('sides: [unclosed')).toHaveLength(1);
    expect(problems('just: words')).toEqual([expect.stringMatching(/sides/)]);
  });
});

describe('rehearsing before a campaign exists', () => {
  it('finds nothing to refuse in the example', () => {
    expect(rehearse(world, ok(example))).toEqual([]);
  });

  it('finds what the engine would refuse that the file reader cannot know', () => {
    // A division under the floor is well-formed YAML and a legal formation; it is the
    // rules, at a new campaign's strictness, that turn it away.
    const out = ok(one('unit: Garde\ncommander: Ney\nstrength: 900\nat: {q: 14, r: 15}'));
    expect(rehearse(world, out)).toEqual([expect.stringMatching(/division is at least/)]);
  });
});

describe('the sample a referee downloads', () => {
  it('is the example file itself, which everything above proves', () => {
    expect(SAMPLE_OOB).toBe(example);
  });

  it('downloads as exactly that text', () => {
    const [head, body] = SAMPLE_OOB_HREF.split(',', 2) as [string, string];
    expect(head).toBe('data:application/yaml;charset=utf-8');
    expect(decodeURIComponent(body)).toBe(example);
  });
});
