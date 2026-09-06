/**
 * Units, as the rules define them.
 *
 * Field names and defaults come from the Napoleonic Campaign Rules (1 km hexes) v4. The
 * numbers are the rules' numbers; where a value is tunable it belongs in campaign config
 * rather than here, because the ruleset is still moving.
 *
 * The division is the primary manoeuvre unit, 4,000 to 10,000 strong. What makes this
 * game unusual is that a division is not a point: at its own spacing it forms a column
 * between about 2.6 and 18 km long, which on a 1 km hex grid means it physically occupies
 * many hexes at once. `column` is that path, and `column.ts` derives its length.
 */

import type { Hex } from './hex.js';

export type UnitKind = 'infantry' | 'cavalry' | 'hq' | 'artillery_reserve' | 'garrison' | 'convoy';

export type Formation = 'march' | 'battle' | 'rest' | 'occupation' | 'rout';

export type Trait =
  /** Wider recon zone, and may field patrols without spending effectives. */
  | 'scout'
  /** Advantage in combat. */
  | 'heavy'
  | 'slow'
  | 'very_slow'
  | 'fast'
  | 'very_fast'
  /** A longer baggage tail: spacing multiplier between 110% and 200%. */
  | 'long_tail'
  /** Recovers provisions in the field, given scenario conditions. */
  | 'foraging'
  /** Can build, repair and demolish bridges. */
  | 'pontooneers';

/** Experience shifts the fatigue table and sets the morale ceiling. */
export type Experience = -2 | -1 | 0 | 1 | 2;

export const EXPERIENCE_NAMES: Readonly<Record<Experience, string>> = {
  [-2]: 'raw',
  [-1]: 'reserve',
  0: 'regular',
  1: 'veteran',
  2: 'elite',
};

/** The highest morale a unit of each experience can hold. */
export const MAX_MORALE: Readonly<Record<Experience, number>> = {
  [-2]: 10,
  [-1]: 20,
  0: 30,
  1: 40,
  2: 50,
};

/** A division may not be formed or split below this. */
export const MIN_DIVISION_EFFECTIVES = 4000;

export interface Unit {
  readonly id: string;
  /**
   * What the unit is called: "1re Division", "Light Brigade".
   *
   * On the unit rather than in a lookup beside it, because it has to survive the trip to
   * a client. A commander who is shown "red-1" is being shown the engine's bookkeeping,
   * and a name kept in a side map is a name the server never sends.
   */
  readonly name: string;
  readonly faction: string;
  readonly kind: UnitKind;

  /** Combat troops on the rolls. */
  readonly effectives: number;
  /** Accumulated fatigue and illness, 0..100, read as a percentage. */
  readonly fatigue: number;
  readonly experience: Experience;
  /** Willingness to stand. At 0 the unit breaks and must rout. */
  readonly morale: number;
  /** Food. At 0 the unit can neither march nor fight. */
  readonly provisions: number;
  readonly maxProvisions: number;
  /** Ammunition. At 0 the unit disengages immediately. */
  readonly equipment: number;
  readonly maxEquipment: number;
  readonly guns: number;

  /** Base march speed in km/h before terrain grade and traits. */
  readonly marchSpeedKmh: number;
  /** Metres of column occupied per soldier. */
  readonly spacingM: number;
  /** Column-length multiplier, 1.0 or 1.10..2.00 with `long_tail`. */
  readonly spacingMultiplier: number;

  readonly traits: readonly Trait[];
  readonly formation: Formation;

  /**
   * The hexes the column occupies, head first.
   *
   * Head is `column[0]`. The tail runs back along the path the unit marched, as far as
   * its length reaches — so this is both "where the unit is" and "where it has just
   * been", which is what recon and interception both need.
   */
  readonly column: readonly Hex[];

  /** Hours marched since the last midnight. Drives fatigue; capped at 20. */
  readonly hoursMarchedToday: number;

  /** Corps grouping. Presentation and combat only — everything tracks individually. */
  readonly corps: string | null;
}

export const hasTrait = (u: Unit, t: Trait): boolean => u.traits.includes(t);

/** Where the unit is, for every purpose that wants a single hex. */
export const head = (u: Unit): Hex => {
  const h = u.column[0];
  if (h === undefined) throw new Error(`unit ${u.id} has no position`);
  return h;
};

/**
 * Troops actually able to fight: effectives reduced by fatigue percent.
 *
 * Derived, never stored. The rules define it as a function of two other fields, and a
 * stored copy is a third fact that can disagree with them.
 */
export const presentUnderArms = (u: Unit): number =>
  Math.round(u.effectives * (1 - u.fatigue / 100));

export const maxMorale = (u: Unit): number => MAX_MORALE[u.experience];

/** A unit at zero morale is broken and must rout. */
export const isBroken = (u: Unit): boolean => u.morale <= 0;

/** At zero provisions a unit can neither march nor fight. */
export const isStarving = (u: Unit): boolean => u.provisions <= 0;

/**
 * Whether this unit is a division for rules that distinguish one.
 *
 * Courier interception and patrol contact both add a die "if the enemy is a division",
 * and garrisons and convoys are neither.
 */
export const isDivision = (u: Unit): boolean =>
  u.kind === 'infantry' || u.kind === 'cavalry' || u.kind === 'artillery_reserve';

/** Sensible starting values by kind, from the rules' worked examples. */
export const KIND_DEFAULTS: Readonly<
  Record<UnitKind, { marchSpeedKmh: number; spacingM: number }>
> = {
  infantry: { marchSpeedKmh: 3, spacingM: 0.5 },
  cavalry: { marchSpeedKmh: 5, spacingM: 3 },
  hq: { marchSpeedKmh: 5, spacingM: 1 },
  artillery_reserve: { marchSpeedKmh: 3, spacingM: 3 },
  garrison: { marchSpeedKmh: 3, spacingM: 0.5 },
  convoy: { marchSpeedKmh: 1, spacingM: 3 },
};
