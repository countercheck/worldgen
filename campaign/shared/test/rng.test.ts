/**
 * Seeded randomness.
 *
 * The distribution tests are loose on purpose — this is a game generator, not a
 * cryptographic one, and a tight bound would make the suite flaky for no benefit. What
 * they actually guard is the failure that matters: a generator that is stuck, biased to
 * one face, or correlated between consecutive commands.
 */

import { describe, expect, it } from 'vitest';

import { lowest, makeRng, ones, rngFor } from '../src/rng.js';

describe('makeRng', () => {
  it('is reproducible from a seed', () => {
    const a = makeRng(42);
    const b = makeRng(42);
    for (let i = 0; i < 100; i++) expect(a.next()).toBe(b.next());
  });

  it('differs between seeds', () => {
    expect(makeRng(1).pool(10)).not.toEqual(makeRng(2).pool(10));
  });

  it('stays in [0, 1)', () => {
    const rng = makeRng(7);
    for (let i = 0; i < 10000; i++) {
      const v = rng.next();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });

  it('rolls only faces a die has', () => {
    const rng = makeRng(9);
    for (let i = 0; i < 10000; i++) {
      const d = rng.d6();
      expect(Number.isInteger(d)).toBe(true);
      expect(d).toBeGreaterThanOrEqual(1);
      expect(d).toBeLessThanOrEqual(6);
    }
  });

  it('rolls every face at roughly even odds', () => {
    // Guards a stuck or badly biased generator, not the quality of the distribution.
    // The band is deliberately wide: this is a game generator, and a tight bound would
    // make the suite flaky for no benefit.
    const rng = makeRng(11);
    const n = 60000;
    const counts = new Map<number, number>();
    for (let i = 0; i < n; i++) {
      const face = rng.d6();
      counts.set(face, (counts.get(face) ?? 0) + 1);
    }

    expect([...counts.values()].reduce((a, c) => a + c, 0)).toBe(n);

    const expected = n / 6;
    for (let face = 1; face <= 6; face++) {
      const seen = counts.get(face) ?? 0;
      expect(seen, `face ${face} came up ${seen} times, expected about ${expected}`).toBeGreaterThan(
        expected * 0.9,
      );
      expect(seen, `face ${face} came up ${seen} times, expected about ${expected}`).toBeLessThan(
        expected * 1.1,
      );
    }
  });

  it('does not repeat itself over a long run', () => {
    // A generator stuck in a short cycle passes every test above.
    const rng = makeRng(13);
    const seen = new Set<number>();
    for (let i = 0; i < 5000; i++) seen.add(rng.next());
    expect(seen.size).toBeGreaterThan(4900);
  });

  it('gives a pool of the size asked for', () => {
    const rng = makeRng(3);
    expect(rng.pool(0)).toEqual([]);
    expect(rng.pool(1)).toHaveLength(1);
    expect(rng.pool(5)).toHaveLength(5);
    expect(rng.pool(-2)).toEqual([]);
  });

  it('bounds int()', () => {
    const rng = makeRng(5);
    for (let i = 0; i < 1000; i++) {
      const v = rng.int(4);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(4);
    }
  });
});

describe('rngFor', () => {
  it('is reproducible for a campaign and sequence number', () => {
    expect(rngFor(1234, 7).pool(6)).toEqual(rngFor(1234, 7).pool(6));
  });

  it('decorrelates consecutive sequence numbers', () => {
    // Sequence numbers are small and consecutive. Fed to a generator raw they give
    // visibly similar first draws, so commands 1 and 2 would roll suspiciously alike.
    const draws = [];
    for (let seq = 0; seq < 40; seq++) draws.push(rngFor(99, seq).pool(3).join(''));
    expect(new Set(draws).size).toBeGreaterThan(20);
  });

  it('differs between campaigns at the same sequence number', () => {
    expect(rngFor(1, 5).pool(6)).not.toEqual(rngFor(2, 5).pool(6));
  });
});

describe('reading a pool', () => {
  it('counts ones, which is how interception and patrol contact read', () => {
    // The rules say "if a 1 is rolled ... if two 1s are rolled", so the count is the
    // primitive rather than the sum.
    expect(ones([1, 3, 5])).toBe(1);
    expect(ones([1, 1, 4])).toBe(2);
    expect(ones([2, 3, 4])).toBe(0);
    expect(ones([])).toBe(0);
  });

  it('takes the lowest, which is how patrol intelligence reads', () => {
    expect(lowest([4, 2, 6])).toBe(2);
    expect(lowest([5])).toBe(5);
  });
});
