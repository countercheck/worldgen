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
  /** Wider recon zone, and may field patrols without spending troops. */
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

/**
 * The rules' own numbers, kept here as the shape of a unit rather than as its tuning.
 *
 * These are the *defaults*: the live values are on `CampaignConfig`, which a campaign
 * carries, so a ruleset can change what a division is without changing this file. Anything
 * deciding a rule reads config; these exist so a caller with no config in hand still gets
 * the rules as written rather than a zero.
 */
export const MAX_MORALE: Readonly<Record<Experience, number>> = {
  [-2]: 10,
  [-1]: 20,
  0: 30,
  1: 40,
  2: 50,
};

/** A division may not be formed or split below this. See `cfg.minDivisionPaperStrength`. */
export const MIN_DIVISION_PAPER_STRENGTH = 4000;

/**
 * A patrol's strength, in troops.
 *
 * Small enough to be a handful of troopers and large enough to be worth counting. The
 * rules do not give a number — patrols are free until the fourth — so this is what a
 * patrol *is* rather than what it costs, and the cost lives in config beside the rest of
 * the game's numbers.
 */
export const PATROL_PAPER_STRENGTH = 20;

/** Whether this formation is a detachment of another rather than a formation in its own right. */
export const isPatrol = (u: Unit): boolean => u.parentUnitId != null;

/**
 * How large a formation is, in the sense the map cares about.
 *
 * Only ever used for display: it is the size marker above a NATO symbol, and nothing in
 * the rules keys off it. Optional on `Unit` because a scenario that does not say gets a
 * sensible guess from strength rather than a wrong assertion.
 */
export type Echelon =
  | 'none'
  | 'battalion'
  | 'regiment'
  | 'brigade'
  | 'division'
  | 'corps'
  | 'army'
  | 'army_group';

/** Echelon marks, in the order they are drawn above the frame. */
export const ECHELON_MARKS: Readonly<Record<Echelon, string>> = {
  none: '',
  battalion: 'II',
  regiment: 'III',
  brigade: 'X',
  division: 'XX',
  corps: 'XXX',
  // Above corps, for the headquarters of a wing or an army and of the whole: never guessed
  // from strength — an army headquarters is a few hundred staff — only ever stated.
  army: 'XXXX',
  army_group: 'XXXXX',
};

/** Every echelon, smallest first: the order a form should offer them in. */
export const ECHELONS: readonly Echelon[] = [
  'none',
  'battalion',
  'regiment',
  'brigade',
  'division',
  'corps',
  'army',
  'army_group',
];

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

  /**
   * Troops on the rolls: what the returns say the formation has.
   *
   * Paper strength rather than strength, because it is not what would stand in a line
   * tomorrow — `presentUnderArms` is, and the gap between the two is fatigue. A
   * commander plans with the first number and fights with the second, and a game about
   * not knowing things should keep them visibly apart.
   */
  readonly paperStrength: number;
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
   * A change of formation under way, and the hour it finishes.
   *
   * A formation is not a setting: going from column of march to a camp takes two hours of
   * real work, and the unit is neither one thing nor the other while it happens. Held as a
   * pending change rather than as an intermediate `Formation` so that everything reading
   * `formation` sees what the unit still *is* — a division halfway into camp is still in
   * march formation, which is exactly why breaking camp again costs it nothing it has not
   * already spent.
   */
  readonly formationChange: { readonly to: Formation; readonly completesAtHours: number } | null;

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
  /**
   * The formation this was detached from, for a patrol or picket.
   *
   * Null for everything that stands on its own. A patrol is twenty troops off a division's
   * strength and it stays that division's — it reports to it, it is recalled to it, and
   * what it sees is what that division's commander comes to know. Kept on the patrol
   * rather than as a list on the parent so that there is one place a patrol's parentage
   * is recorded and no way for the two to disagree.
   */
  readonly parentUnitId: string | null;

  /**
   * How large the formation is, for the size marker on its map symbol.
   *
   * Optional: absent means "work it out from strength", which is right often enough and
   * wrong quietly rather than loudly. A scenario that knows better says so — the demo's
   * Light Brigade is four thousand strong and is still a brigade.
   */
  readonly echelon?: Echelon;
}

export const hasTrait = (u: Unit, t: Trait): boolean => u.traits.includes(t);

/** Where the unit is, for every purpose that wants a single hex. */
export const head = (u: Unit): Hex => {
  const h = u.column[0];
  if (h === undefined) throw new Error(`unit ${u.id} has no position`);
  return h;
};

/**
 * Troops actually able to fight: paper strength reduced by fatigue percent.
 *
 * Derived, never stored. The rules define it as a function of two other fields, and a
 * stored copy is a third fact that can disagree with them.
 *
 * A patrol carries no fatigue, so its twenty are its twenty.
 */
export const presentUnderArms = (u: Unit): number =>
  isPatrol(u) ? u.paperStrength : Math.round(u.paperStrength * (1 - u.fatigue / 100));

/**
 * The morale ceiling for this unit's experience.
 *
 * Takes the campaign's table where there is one. A caller without config in hand falls
 * back to the rules as written, which is right for a scenario builder and for a test.
 */
export const maxMorale = (
  u: Unit,
  table: Readonly<Record<Experience, number>> = MAX_MORALE,
): number => table[u.experience];

/**
 * A unit at zero morale is broken and must rout.
 *
 * Never a patrol. Twenty troopers have no morale to lose in the rules' sense — what
 * happens to a patrol that meets something is a die roll that destroys it or recoils it,
 * not a morale check — and their zero means "not tracked" rather than "broken".
 */
export const isBroken = (u: Unit): boolean => !isPatrol(u) && u.morale <= 0;

/**
 * At zero provisions a unit can neither march nor fight.
 *
 * Never a patrol, for the same reason: a patrol carries no supply, and reading its zero as
 * starvation would freeze every picket in the field on the hour it was sent out.
 */
export const isStarving = (u: Unit): boolean => !isPatrol(u) && u.provisions <= 0;

/**
 * Whether this unit is a division for rules that distinguish one.
 *
 * Courier interception and patrol contact both add a die "if the enemy is a division",
 * and garrisons and convoys are neither. Nor is a patrol: twenty troopers ride as cavalry
 * and would otherwise pass the kind test, which would hand a picket the die the rules give
 * to the division it was detached from.
 */
export const isDivision = (u: Unit): boolean =>
  !isPatrol(u) &&
  (u.kind === 'infantry' || u.kind === 'cavalry' || u.kind === 'artillery_reserve');

/**
 * The formation's echelon, guessed from strength when it was not stated.
 *
 * A guess, and only ever drawn — never used by a rule. The thresholds are the ordinary
 * Napoleonic ones: a division is several thousand, a brigade a couple, a regiment under a
 * thousand. A headquarters is drawn at corps level because that is what an HQ formation in
 * these rules represents, and a convoy gets no marker at all because it is not a
 * manoeuvre unit and a size mark on it would be a category error.
 */
export function echelonOf(u: Unit): Echelon {
  if (u.echelon !== undefined) return u.echelon;
  if (u.kind === 'hq') return 'corps';
  if (u.kind === 'convoy') return 'none';
  if (u.paperStrength >= MIN_DIVISION_PAPER_STRENGTH) return 'division';
  if (u.paperStrength >= 1500) return 'brigade';
  if (u.paperStrength >= 500) return 'regiment';
  return 'battalion';
}

/**
 * A formation as somebody last heard of it.
 *
 * Deliberately not a `Unit`. A dated snapshot and a live record are different things, and
 * anything handed a `Unit` will draw it as though it were true now — which is exactly the
 * belief this design exists to deny. Everything here is what a despatch would carry.
 *
 * It lives beside `Unit` rather than beside `viewFor` because a commander's *knowledge*
 * holds these: what they last heard is state, folded from the log like anything else, not
 * something computed when a client happens to ask.
 */
export interface UnitReport {
  readonly unitId: string;
  readonly name: string;
  readonly faction: string;
  /** Your own formation, so its arm and size are not in doubt — only its position is. */
  readonly kind: UnitKind;
  readonly echelon: Echelon;
  /** The hour the report describes, which is not the hour it arrived. */
  readonly atHours: number;
  readonly head: Hex;
  readonly paperStrength: number;
  readonly fatigue: number;
  readonly formation: Formation;
  readonly provisions: number;
  readonly corps: string | null;
}

/** Snapshot a formation as of a given hour. What a rider would carry away with them. */
export const reportOf = (unit: Unit, atHours: number): UnitReport => ({
  unitId: unit.id,
  name: unit.name,
  faction: unit.faction,
  kind: unit.kind,
  echelon: echelonOf(unit),
  atHours,
  head: unit.column[0] ?? { q: 0, r: 0 },
  paperStrength: unit.paperStrength,
  fatigue: unit.fatigue,
  formation: unit.formation,
  provisions: unit.provisions,
  corps: unit.corps,
});

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
