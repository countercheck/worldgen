/**
 * The enforcement policy, asserted as a table.
 *
 * This is the heart of "variable rule enforcement", so it is tested exhaustively rather
 * than by example: every strictness against every severity against forced and not. A
 * conditional buried in `refuses` is easy to get subtly wrong and very hard to notice,
 * because the wrong answer still produces a playable game.
 */

import { describe, expect, it } from 'vitest';

import {
  anyHard,
  bypassed,
  CODES,
  hard,
  refuses,
  RuleViolation,
  soft,
  STRICTNESSES,
  type Strictness,
  type Violation,
} from '../src/ruling.js';

const NONE: Violation[] = [];
const SOFT = [soft(CODES.IMPASSABLE, 'the ground is impassable')];
const HARD = [hard(CODES.OFF_MAP, 'that hex is not on the map')];
const BOTH = [...SOFT, ...HARD];

describe('refuses', () => {
  it('allows a clean command at every strictness', () => {
    for (const s of STRICTNESSES) {
      expect(refuses(NONE, s, false), s).toBe(false);
      expect(refuses(NONE, s, true), s).toBe(false);
    }
  });

  it('refuses a hard violation at every strictness, forced or not', () => {
    // The point of the hard/soft split. There is no setting, and no override, that lets
    // a unit stand on a hex that does not exist.
    for (const s of STRICTNESSES) {
      expect(refuses(HARD, s, false), `${s} unforced`).toBe(true);
      expect(refuses(HARD, s, true), `${s} forced`).toBe(true);
      expect(refuses(BOTH, s, true), `${s} forced, mixed`).toBe(true);
    }
  });

  it('refuses a soft violation only under strict, and only unforced', () => {
    expect(refuses(SOFT, 'strict', false)).toBe(true);
    expect(refuses(SOFT, 'strict', true)).toBe(false);
    expect(refuses(SOFT, 'lenient', false)).toBe(false);
    expect(refuses(SOFT, 'open', false)).toBe(false);
  });

  it('is the full table', () => {
    const table: [Strictness, 'none' | 'soft' | 'hard', boolean, boolean][] = [
      ['strict', 'none', false, false],
      ['strict', 'none', true, false],
      ['strict', 'soft', false, true],
      ['strict', 'soft', true, false],
      ['strict', 'hard', false, true],
      ['strict', 'hard', true, true],
      ['lenient', 'none', false, false],
      ['lenient', 'none', true, false],
      ['lenient', 'soft', false, false],
      ['lenient', 'soft', true, false],
      ['lenient', 'hard', false, true],
      ['lenient', 'hard', true, true],
      ['open', 'none', false, false],
      ['open', 'none', true, false],
      ['open', 'soft', false, false],
      ['open', 'soft', true, false],
      ['open', 'hard', false, true],
      ['open', 'hard', true, true],
    ];
    const pick = { none: NONE, soft: SOFT, hard: HARD };
    for (const [strictness, kind, forced, expected] of table) {
      expect(refuses(pick[kind], strictness, forced), `${strictness}/${kind}/forced=${forced}`).toBe(
        expected,
      );
    }
  });
});

describe('bypassed', () => {
  it('records what a command proceeded in spite of', () => {
    // The audit trail. A soft violation allowed through has to be visible afterwards, or
    // a referee reviewing a contested campaign cannot see what was bent.
    expect(bypassed(SOFT, 'strict', true)).toEqual(SOFT);
    expect(bypassed(SOFT, 'lenient', false)).toEqual(SOFT);
    expect(bypassed(SOFT, 'open', false)).toEqual(SOFT);
  });

  it('records nothing when the command was refused', () => {
    expect(bypassed(SOFT, 'strict', false)).toEqual([]);
    expect(bypassed(HARD, 'open', true)).toEqual([]);
  });

  it('records nothing when there was nothing to bend', () => {
    for (const s of STRICTNESSES) expect(bypassed(NONE, s, false)).toEqual([]);
  });
});

describe('anyHard', () => {
  it('finds a hard violation among soft ones', () => {
    expect(anyHard(BOTH)).toBe(true);
    expect(anyHard(SOFT)).toBe(false);
    expect(anyHard(NONE)).toBe(false);
  });
});

describe('RuleViolation', () => {
  it('carries the violations and says which they were', () => {
    const err = new RuleViolation(SOFT);
    expect(err.violations).toEqual(SOFT);
    expect(err.message).toContain(CODES.IMPASSABLE);
    expect(err.message).toContain('impassable');
  });

  it('reports whether the refusal could have been overridden', () => {
    expect(new RuleViolation(HARD).unavoidable).toBe(true);
    expect(new RuleViolation(SOFT).unavoidable).toBe(false);
  });
});
