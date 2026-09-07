/**
 * Rule enforcement, with a dial the referee controls.
 *
 * This game is refereed, and a referee legitimately needs to break its rules: to correct
 * a mistake made three hours ago, to model something the rules do not cover, or simply to
 * keep play moving. An engine that cannot be overridden gets worked around outside the
 * system, and then the log stops describing the game.
 *
 * So validation is physically separate from mutation. Checks are pure functions returning
 * violations; nothing in this module changes anything. The engine is the only place the
 * two meet, and it records what it bent.
 *
 * The important distinction is that **hard and soft is not the same axis as strict and
 * open.**
 *
 * - A soft violation is a rule of the *game*: a river that cannot be forded, a march that
 *   exceeds twenty hours, an order sent without a courier. Breaking one produces a
 *   coherent campaign the rules happen to disallow, which is exactly the referee's
 *   prerogative.
 * - A hard violation is a rule of *arithmetic*: a hex that is not on the map, a unit that
 *   does not exist, a world that is not the one the campaign was created against.
 *   Breaking one produces state that cannot be reduced, serialised or drawn.
 *
 * So hard violations are refused at every strictness, including `open`, and including
 * `force`. There is no setting that lets a unit stand on a hex that is not there. That is
 * what makes "permissive bookkeeping" a safe mode to run in rather than an unvalidated
 * one — the referee can bend every rule that is a rule, and none that is a fact.
 */

export type Severity = 'hard' | 'soft';

export type Strictness =
  /** Any violation refuses the command. The default, and how a rules-tight game runs. */
  | 'strict'
  /** Soft violations are allowed through and recorded. Hard ones still refuse. */
  | 'lenient'
  /** Bookkeeping: soft violations are ignored entirely. Hard ones still refuse. */
  | 'open';

export const STRICTNESSES: readonly Strictness[] = ['strict', 'lenient', 'open'];

export interface Violation {
  /** A stable machine-readable code. Logged, so keep them stable across versions. */
  readonly code: string;
  /** Addressed to the referee, so say what is wrong rather than which check failed. */
  readonly message: string;
  readonly severity: Severity;
}

export const hard = (code: string, message: string): Violation => ({
  code,
  message,
  severity: 'hard',
});

export const soft = (code: string, message: string): Violation => ({
  code,
  message,
  severity: 'soft',
});

/** Violation codes with a fixed meaning across the engine. */
export const CODES = {
  // Hard — never overridable.
  NO_SUCH_UNIT: 'no_such_unit',
  NO_SUCH_FACTION: 'no_such_faction',
  NO_SUCH_COMMANDER: 'no_such_commander',
  /** A man riding with the other side's baggage, or answering to their headquarters. */
  WRONG_FACTION: 'wrong_faction',
  /** A loop in the chain of command: every tree walk in the engine would hang. */
  COMMAND_CYCLE: 'command_cycle',
  DUPLICATE_ID: 'duplicate_id',
  OFF_MAP: 'off_map',
  WORLD_MISMATCH: 'world_mismatch',
  TIME_REVERSED: 'time_reversed',
  MALFORMED: 'malformed',
  // Soft — the referee's to bend.
  IMPASSABLE: 'impassable',
  NOT_ADJACENT: 'not_adjacent',
  MAJOR_RIVER_UNBRIDGED: 'major_river_unbridged',
  NO_CROSSING: 'no_crossing',
  MARCH_LIMIT: 'march_limit',
  NIGHT_MOVE: 'night_move',
  NOT_YOUR_UNIT: 'not_your_unit',
  NO_COURIER_ROUTE: 'no_courier_route',
  UNIT_TOO_SMALL: 'unit_too_small',
} as const;

export const anyHard = (violations: readonly Violation[]): boolean =>
  violations.some((v) => v.severity === 'hard');

/**
 * Whether these violations stop the command.
 *
 * The whole enforcement policy, in one place, so it can be read and tested as a table
 * rather than inferred from scattered conditionals.
 */
export function refuses(
  violations: readonly Violation[],
  strictness: Strictness,
  forced: boolean,
): boolean {
  if (anyHard(violations)) return true;
  if (forced) return false;
  switch (strictness) {
    case 'strict':
      return violations.length > 0;
    case 'lenient':
    case 'open':
      return false;
  }
}

/**
 * The violations a command proceeded in spite of — what the log records as bypassed.
 *
 * Empty when nothing was bent, which is the common case; a non-empty list on an applied
 * command is precisely the audit trail a referee needs after the fact.
 */
export function bypassed(
  violations: readonly Violation[],
  strictness: Strictness,
  forced: boolean,
): readonly Violation[] {
  return refuses(violations, strictness, forced) ? [] : violations;
}

/** Thrown when a command is refused. Carries the violations so a UI can show them. */
export class RuleViolation extends Error {
  override readonly name = 'RuleViolation';
  readonly violations: readonly Violation[];

  constructor(violations: readonly Violation[]) {
    const detail = violations.map((v) => `${v.code}: ${v.message}`).join('; ');
    super(detail || 'refused');
    this.violations = violations;
  }

  /** Whether this refusal is one no strictness or force could have avoided. */
  get unavoidable(): boolean {
    return anyHard(this.violations);
  }
}
