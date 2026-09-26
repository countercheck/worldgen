/**
 * What a referee may write onto a unit by hand, and what they may not.
 *
 * The referee is the rules' last word, so nearly every number is theirs to set: a battered
 * division's strength after a fight the engine did not see, a column's hours on the road
 * after a night the log does not know about. What is checked here is only that the value
 * is the kind of thing the field holds. A morale above the experience ceiling, or
 * provisions over the waggons' capacity, is a referee's call and not a typo.
 *
 * Checked against whatever arrived rather than the type, as standing orders are: an edit
 * comes off the wire, goes into the log for good, and is replayed into every view after.
 */

import type { UnitStatChanges } from './events.js';
import { ECHELONS, EXPERIENCES, FORMATIONS, TRAITS, UNIT_KINDS } from './unit.js';

const isNumber = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);

/** A field's check: a sentence when the value will not do, null when it will. */
type Check = (v: unknown) => string | null;

const atLeast =
  (min: number, what: string): Check =>
  (v) =>
    isNumber(v) && v >= min ? null : `${what} is a number from ${min} up`;

const above =
  (min: number, what: string): Check =>
  (v) =>
    isNumber(v) && v > min ? null : `${what} is a number above ${min}`;

const between =
  (min: number, max: number, what: string): Check =>
  (v) =>
    isNumber(v) && v >= min && v <= max ? null : `${what} runs from ${min} to ${max}`;

const oneOf =
  <T>(values: readonly T[], what: string): Check =>
  (v) =>
    values.includes(v as T) ? null : `${what} is one of ${values.join(', ')}`;

const CHECKS: { readonly [K in keyof Required<UnitStatChanges>]: Check } = {
  name: (v) => (typeof v === 'string' && v.trim() !== '' ? null : 'a unit has to be called something'),
  kind: oneOf(UNIT_KINDS, 'kind'),
  paperStrength: atLeast(0, 'paper strength'),
  fatigue: between(0, 100, 'fatigue'),
  morale: atLeast(0, 'morale'),
  provisions: atLeast(0, 'provisions'),
  maxProvisions: atLeast(0, 'the most provisions it can carry'),
  equipment: atLeast(0, 'equipment'),
  maxEquipment: atLeast(0, 'the most equipment it can carry'),
  guns: atLeast(0, 'guns'),
  experience: oneOf(EXPERIENCES, 'experience'),
  marchSpeedKmh: above(0, 'march speed'),
  spacingM: above(0, 'spacing'),
  spacingMultiplier: between(1, 2, 'the spacing multiplier'),
  formation: oneOf(FORMATIONS, 'formation'),
  formationChange: (v) => {
    if (v === null) return null;
    const c = v as { to?: unknown; completesAtHours?: unknown } | undefined;
    return typeof v === 'object' &&
      !Array.isArray(v) &&
      Object.keys(v as object).every((k) => k === 'to' || k === 'completesAtHours') &&
      FORMATIONS.includes(c?.to as never) &&
      isNumber(c?.completesAtHours)
      ? null
      : 'a change of formation is what it is becoming and the hour it finishes, or none';
  },
  traits: (v) =>
    Array.isArray(v) &&
    v.every((t) => TRAITS.includes(t)) &&
    new Set(v).size === v.length
      ? null
      : `traits are a list of ${TRAITS.join(', ')}, each once`,
  hoursMarchedToday: atLeast(0, 'hours since the last rest'),
  // Laid down as the hours just gone, the earliest of twenty-four would already have left
  // the window the cap reads: 23 is the most that reads back as it was set.
  roadHoursLast24: between(0, 23, 'hours on the road in the last 24'),
  corps: (v) => (v === null || typeof v === 'string' ? null : 'a corps is a name, or none'),
  echelon: oneOf(ECHELONS, 'echelon'),
};

/** What is wrong with a referee's edit to a unit, as sentences. Empty when there is nothing. */
export function unitStatProblems(raw: unknown): string[] {
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) {
    return ['the changes are a set of values to write onto the unit'];
  }
  const out: string[] = [];
  for (const [k, v] of Object.entries(raw)) {
    const check = (CHECKS as Record<string, Check | undefined>)[k];
    if (check === undefined) {
      out.push(`${k} is not a value the referee sets by hand`);
      continue;
    }
    const problem = check(v);
    if (problem !== null) out.push(problem);
  }
  return out;
}

/** The changes and nothing else, for the log: only the keys the referee may set. */
export function unitStatChangesOf(c: UnitStatChanges): UnitStatChanges {
  const out = Object.fromEntries(
    Object.entries(c).filter(([k, v]) => k in CHECKS && v !== undefined),
  ) as { -readonly [K in keyof UnitStatChanges]: UnitStatChanges[K] };
  if (c.formationChange != null) {
    out.formationChange = { to: c.formationChange.to, completesAtHours: c.formationChange.completesAtHours };
  }
  return out;
}
