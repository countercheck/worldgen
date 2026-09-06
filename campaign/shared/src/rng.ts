/**
 * Seeded randomness.
 *
 * Every roll in the game goes through here. Nothing calls `Math.random()`, for the same
 * reason `worldgen` passes an explicit `numpy.random.Generator` everywhere: a campaign
 * that cannot be reproduced cannot be debugged, and a referee who has to adjudicate a
 * disputed interception needs to be able to show the dice.
 *
 * The generator is not carried in campaign state. Instead a fresh one is derived per
 * command from `(campaign seed, sequence number)`, so `decide()` is a pure function of
 * facts already in the log. There is no RNG cursor to keep in step, and replaying a
 * prefix of the log cannot desynchronise anything.
 *
 * That said, the recorded *outcome* is what replay uses — see `events.ts`. A roll is
 * written into the event that describes it, so a log replays identically even if this
 * algorithm is ever changed. This generator decides what happens; it is not load-bearing
 * for reconstructing what already happened.
 */

export interface Rng {
  /** A float in [0, 1). */
  next(): number;
  /** An integer in [0, n). */
  int(n: number): number;
  /** One six-sided die, 1..6. */
  d6(): number;
  /** `n` six-sided dice. */
  pool(n: number): number[];
}

/**
 * mulberry32 — a small, fast, well-distributed 32-bit generator.
 *
 * Chosen for being short enough to audit at a glance and exactly reproducible across
 * JavaScript engines: it uses only `Math.imul`, `>>>` and `+`, all of which are defined
 * on 32-bit integers with no floating-point rounding anywhere in the state update.
 */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Mix two 32-bit values into a seed.
 *
 * Sequence numbers are small and consecutive, and feeding those straight into a
 * generator gives visibly correlated first draws — command 1 and command 2 would roll
 * suspiciously similar dice. This is the finalising mix from MurmurHash3, which
 * decorrelates them.
 */
function mix(a: number, b: number): number {
  let h = (a ^ Math.imul(b, 0x9e3779b9)) >>> 0;
  h ^= h >>> 16;
  h = Math.imul(h, 0x85ebca6b);
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35);
  h ^= h >>> 16;
  return h >>> 0;
}

export function makeRng(seed: number): Rng {
  const next = mulberry32(seed);
  const rng: Rng = {
    next,
    int: (n: number) => Math.floor(next() * n),
    d6: () => Math.floor(next() * 6) + 1,
    pool: (n: number) => Array.from({ length: Math.max(0, n) }, () => rng.d6()),
  };
  return rng;
}

/**
 * The generator for one command, derived from the campaign seed and the sequence number
 * the command's first event will take.
 *
 * Deriving rather than carrying is what keeps `decide()` pure: two calls with the same
 * state and command roll the same dice, which is what makes the engine testable at all.
 */
export const rngFor = (seed: number, seq: number): Rng => makeRng(mix(seed, seq));

/**
 * Count the 1s in a pool — the shape both courier interception and patrol combat take.
 *
 * The rules phrase these as "if a 1 is rolled ... if two 1s are rolled", so the count is
 * the primitive rather than the sum.
 */
export const ones = (dice: readonly number[]): number => dice.filter((d) => d === 1).length;

/** The lowest die in a pool — how patrol intelligence is read. */
export const lowest = (dice: readonly number[]): number => Math.min(...dice);
