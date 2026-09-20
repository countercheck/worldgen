/**
 * Every tunable number in the game.
 *
 * The ruleset is at v4 and still moving, so nothing from it is hardcoded in rule logic —
 * the same discipline `worldgen` applies with `WorldConfig`. A rules revision should be a
 * config edit, and a scenario that wants slower cavalry should not need a build.
 *
 * Defaults are the rules' own numbers. Where a default is a judgement rather than a
 * quotation, the comment says so.
 */

import type { DecisionTrigger } from './task.js';
import type { LandCover } from './world.js';
import { FOOTPRINT, type FootprintShape } from './column.js';
import type { Experience, Formation, Trait, UnitKind } from './unit.js';

/**
 * How hard the ground is to move over.
 *
 * The rules' four columns. This is the whole terrain model for movement: no gradient
 * term, no per-hex cost, just which of four categories a step falls into.
 */
export type Grade = 'highway' | 'road' | 'off_road' | 'bad_going';

export const GRADES: readonly Grade[] = ['highway', 'road', 'off_road', 'bad_going'];

/** Things that move, for speed-table purposes. Patrols move as cavalry, then as couriers. */
export type Mover = UnitKind | 'courier';

export type SpeedTable = Readonly<Record<Mover, Readonly<Record<Grade, number>>>>;

/**
 * Movement rates in km/h, straight from the rules' table.
 *
 * One hex is one kilometre, so this is also hexes per hour, and the time to enter a hex
 * is simply `1 / speed`. That correspondence is why the ruleset and the generator fit
 * together without a conversion factor anywhere.
 */
export const DEFAULT_SPEEDS: SpeedTable = {
  infantry: { highway: 3, road: 3, off_road: 2, bad_going: 1 },
  cavalry: { highway: 5, road: 5, off_road: 3, bad_going: 1 },
  hq: { highway: 5, road: 5, off_road: 3, bad_going: 1 },
  courier: { highway: 10, road: 10, off_road: 6, bad_going: 2 },
  convoy: { highway: 1, road: 1, off_road: 2 / 3, bad_going: 1 / 3 },
  // The rules give no row for these. Artillery moves with the guns, so it takes the
  // infantry rate; a garrison cannot manoeuvre at all until collected, and its rate is
  // only ever used if a referee marches one anyway.
  artillery_reserve: { highway: 3, road: 3, off_road: 2, bad_going: 1 },
  garrison: { highway: 3, road: 3, off_road: 2, bad_going: 1 },
};

/** Traits that shift march speed, in km/h. */
export const DEFAULT_TRAIT_SPEED: Readonly<Partial<Record<Trait, number>>> = {
  fast: 1,
  very_fast: 2,
  slow: -1,
  very_slow: -2,
};

/**
 * Land cover that counts as Bad Going.
 *
 * The rules name the category but not which ground falls in it, so this is a judgement
 * against `worldgen`'s cover classes: standing water and closed canopy, plus high ground
 * with no soil. Scrub and woodland are deliberately not here — they are slower than open
 * country, which the off-road rate already captures, but they are not a bog.
 */
export const DEFAULT_BAD_GOING_COVERS: readonly LandCover[] = [
  'bog',
  'marsh',
  'dense_forest',
  'alpine',
  'bare_rock',
];

export interface CampaignConfig {
  // ---- movement -------------------------------------------------------
  readonly speeds: SpeedTable;
  readonly traitSpeedKmh: Readonly<Partial<Record<Trait, number>>>;
  /** A speed floor, so a trait stack cannot bring a unit to a halt or below zero. */
  readonly minSpeedKmh: number;
  readonly badGoingCovers: readonly LandCover[];
  /**
   * Gradient above which open ground is Bad Going, in metres of rise per km.
   *
   * A judgement, not from the rules. 150 m/km is about 8.5 degrees — the point at which
   * a column with guns and waggons stops being able to take the ground straight on.
   */
  readonly badGoingSlopeMPerKm: number;
  /** The rules' hard cap: no unit, patrol or convoy marches more than this in a day. */
  readonly maxMarchHoursPerDay: number;

  // ---- river crossings ------------------------------------------------
  /** Crossing a minor river at a ford, per division. Free if there is a bridge. */
  readonly fordHours: number;
  /** Crossing a major river at a bridge, per division. Impossible without one. */
  readonly majorCrossingHours: number;
  /** What a pontooneer unit spends to bridge a major river itself. A judgement. */
  readonly pontoonBuildHours: number;

  // ---- reconnaissance -------------------------------------------------
  /** Hexes either side of the march column a unit observes. */
  /**
   * Whether ground a commander's formations have not covered is hidden from him.
   *
   * Off. The tension the ruleset turns on is where the enemy is and where one's own
   * detached corps is, not what the country looks like — a commander in 1815 had a map.
   * And with sight limited to the formation he rides with, terrain fog would leave him a
   * two-hex bubble and darkness beyond it, unable to plan a march at all.
   *
   * The masking machinery is unchanged and still tested in both positions, so turning
   * this on is a switch rather than a rebuild. It comes back with the issued map — the
   * period survey that is wrong about minor roads, fords and river courses — because that
   * turns fog from darkness into doubt, which is the version worth having.
   */
  readonly terrainFog: boolean;
  readonly reconRadius: number;
  /** The same, for a unit with the scout trait. */
  readonly scoutReconRadius: number;
  /** Patrols a scout unit may field without spending men. */
  readonly freePatrols: number;
  /** PaperStrength permanently lost per patrol beyond the free ones. */
  readonly extraPatrolCost: number;
  /** How far gunfire carries, in km. */
  readonly gunfireRangeKm: number;
  /**
   * How often a contact somebody is still watching is written down again.
   *
   * A logging cadence, not a rule — nothing keys off it. A picket watching a stationary
   * column would otherwise write an identical event on every command; this bounds that to
   * once an hour, which is also how stale the hour on a watched contact may read.
   *
   * Contact *identity* does not use this. Whether a sighting continues a contact or starts
   * a new one turns on whether his men ever lost sight of it, which is a fact about the
   * world rather than about elapsed time — see `knowledge.ts`.
   */
  readonly contactRefreshHours: number;

  // ---- the clock ------------------------------------------------------
  /**
   * Daylight, as hours of the day.
   *
   * Fixed rather than astronomical: the generator models no latitude and the rules ask
   * only for "daylight hours". Convoys and patrols are bound by these; couriers ride
   * regardless.
   */
  readonly sunriseHour: number;
  readonly sunsetHour: number;

  // ---- couriers -------------------------------------------------------
  /** Extra dice added to an interception roll, by what the courier is passing. */
  readonly interceptDiceCavalry: number;
  readonly interceptDiceScout: number;
  readonly interceptDiceDivision: number;
  /** The die every rider throws when he passes an enemy column, before modifiers. */
  readonly interceptDiceBase: number;
  /** Ones that lose the rider, and ones that lose the paper as well. */
  readonly interceptLoseOnes: number;
  readonly interceptCaptureOnes: number;
  /**
   * What a river costs a lone rider where it would stop a division.
   *
   * A courier fords, finds the boat, or swims the horse. Not from the rules, which do not
   * discuss it; a courier system in which one river ends communication altogether is not
   * the period.
   */
  readonly courierMajorCrossingHours: number;

  // ---- the scheduler --------------------------------------------------
  /**
   * Which discoveries halt an advance and put a decision in the referee's queue.
   *
   * A dial rather than a constant because the cost of this game falls on the referee: a
   * large campaign should be able to march through a distant sighting and stop only for
   * what its referee actually wants to adjudicate.
   *
   * `gunfire_heard` is out by default — thirty kilometres is a wide net and it would halt
   * every advance on a day anybody was fighting.
   */
  readonly haltTriggers: readonly DecisionTrigger[];
  /** A ceiling on one advance, so a mistyped `advance 1000` cannot lock the server up. */
  readonly maxAdvanceHours: number;

  // ---- fatigue and formation ------------------------------------------
  /**
   * Cumulative fatigue for a day's marching, indexed by whole hours on the road.
   *
   * The rules give this as a table of bands rather than a rate, and the shape is the
   * point: four hours is free, the cost climbs slowly to the twelfth hour and then turns
   * brutal. Held as a cumulative curve rather than per-hour increments because that is
   * what makes it composable — the cost of marching from hour *a* to hour *b* is one
   * subtraction, whatever happened in between.
   *
   * Beyond the table, `marchFatiguePerHourBeyond` applies per further hour.
   */
  readonly marchFatigue: Readonly<Record<FatigueClass, readonly number[]>>;
  readonly marchFatiguePerHourBeyond: Readonly<Record<FatigueClass, number>>;
  /** Which curve each kind of formation reads. */
  readonly fatigueClass: Readonly<Record<UnitKind, FatigueClass>>;
  /**
   * Extra fatigue per hour any part of the column is on the road in the dark.
   *
   * Any part, which is why the tail matters: a column is not off the road until its rear
   * is in, and a march that ends at dusk has men still marching well into the night.
   */
  readonly nightFatiguePerHour: number;
  /**
   * Hours to change formation, `from` then `to`.
   *
   * The rules' own matrix. Making camp is the one that fires by itself — a formation that
   * has spent its day stops and builds one — and everything else is ordered.
   */
  readonly formationChangeHours: Readonly<Record<Formation, Readonly<Record<Formation, number>>>>;
  /**
   * The ground each formation stands on, as a fold of its column of march.
   *
   * A column is long and thin because it is on a road; everything else gathers it up. See
   * `FOOTPRINT` in `column.ts` for what the numbers mean and why camp is a thicker line
   * rather than a disc.
   */
  readonly footprint: Readonly<Record<Formation, FootprintShape>>;

  // ---- what a formation is --------------------------------------------
  /** A division may not be formed or split below this. */
  readonly minDivisionPaperStrength: number;
  /** How many troopers ride in a patrol. */
  readonly patrolPaperStrength: number;
  /** The highest morale a unit of each experience can hold. */
  readonly maxMorale: Readonly<Record<Experience, number>>;
  /** March speed and column spacing each kind starts with, before traits. */
  readonly kindDefaults: Readonly<Record<UnitKind, { marchSpeedKmh: number; spacingM: number }>>;
}

/**
 * Which fatigue curve a formation reads.
 *
 * Two, because the rules give two. Horses tire differently from men and the table says so
 * from the first hour: cavalry starts the day a point down and reaches every band an hour
 * before the infantry does.
 */
export type FatigueClass = 'infantry' | 'cavalry';

/**
 * The rules' fatigue table, as cumulative totals by hour marched.
 *
 * Infantry: nothing to four hours, then 1, 1, 2, 2, 3, 4, 5, 6 through the twelfth, and
 * +2 an hour after that. Cavalry is the same curve one hour earlier, and starts at 1.
 */
/** The rules' morale ceilings, by experience: raw 10 through elite 50. */
export const DEFAULT_MAX_MORALE: Readonly<Record<Experience, number>> = {
  [-2]: 10,
  [-1]: 20,
  0: 30,
  1: 40,
  2: 50,
};

/**
 * What each kind of formation is made of, before anything is done to it.
 *
 * Spacing is the rules' metres-per-man, and it is why a cavalry division is six times the
 * length of an infantry one at the same strength.
 */
export const DEFAULT_KIND_DEFAULTS: Readonly<
  Record<UnitKind, { marchSpeedKmh: number; spacingM: number }>
> = {
  infantry: { marchSpeedKmh: 3, spacingM: 0.5 },
  cavalry: { marchSpeedKmh: 5, spacingM: 3 },
  hq: { marchSpeedKmh: 5, spacingM: 1 },
  artillery_reserve: { marchSpeedKmh: 3, spacingM: 3 },
  garrison: { marchSpeedKmh: 3, spacingM: 0.5 },
  convoy: { marchSpeedKmh: 1, spacingM: 3 },
};

export const DEFAULT_MARCH_FATIGUE: Readonly<Record<FatigueClass, readonly number[]>> = {
  //        0  1  2  3  4  5  6  7  8  9 10 11 12
  infantry: [0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 4, 5, 6],
  //        0  1  2  3  4  5  6  7  8  9 10 11
  cavalry: [1, 1, 1, 1, 1, 1, 2, 2, 3, 4, 5, 6],
};

export const DEFAULT_MARCH_FATIGUE_BEYOND: Readonly<Record<FatigueClass, number>> = {
  infantry: 2,
  cavalry: 2,
};

/** Guns and baggage march with the infantry; a headquarters rides. */
export const DEFAULT_FATIGUE_CLASS: Readonly<Record<UnitKind, FatigueClass>> = {
  infantry: 'infantry',
  cavalry: 'cavalry',
  hq: 'cavalry',
  artillery_reserve: 'infantry',
  garrison: 'infantry',
  convoy: 'infantry',
};

/**
 * The rules' formation change matrix, in hours: `[from][to]`.
 *
 * Occupation is a day at either end, which is what makes it a decision rather than a
 * manoeuvre. Staying put costs nothing, so the diagonal is zero.
 */
export const DEFAULT_FORMATION_CHANGE_HOURS: Readonly<
  Record<Formation, Readonly<Record<Formation, number>>>
> = {
  march: { march: 0, battle: 1, rest: 2, occupation: 24, rout: 0 },
  battle: { march: 2, battle: 0, rest: 1, occupation: 24, rout: 0 },
  rest: { march: 2, battle: 1, rest: 0, occupation: 24, rout: 0 },
  occupation: { march: 24, battle: 24, rest: 24, occupation: 0, rout: 0 },
  // A formation that has broken does not change formation in any orderly sense. Rallying
  // it is the referee's to adjudicate, and costs whatever he says it costs.
  rout: { march: 0, battle: 0, rest: 0, occupation: 0, rout: 0 },
};

export const DEFAULT_CONFIG: CampaignConfig = {
  speeds: DEFAULT_SPEEDS,
  traitSpeedKmh: DEFAULT_TRAIT_SPEED,
  minSpeedKmh: 0.25,
  badGoingCovers: DEFAULT_BAD_GOING_COVERS,
  badGoingSlopeMPerKm: 150,
  maxMarchHoursPerDay: 20,

  fordHours: 1,
  majorCrossingHours: 1,
  pontoonBuildHours: 6,

  terrainFog: false,
  reconRadius: 1,
  scoutReconRadius: 2,
  freePatrols: 3,
  extraPatrolCost: 100,
  gunfireRangeKm: 30,
  contactRefreshHours: 1,

  sunriseHour: 6,
  sunsetHour: 18,

  interceptDiceCavalry: 1,
  interceptDiceScout: 1,
  interceptDiceDivision: 1,
  interceptDiceBase: 1,
  interceptLoseOnes: 1,
  interceptCaptureOnes: 2,
  courierMajorCrossingHours: 1,

  haltTriggers: [
    'enemy_contact',
    'crossing_impassable',
    'objective_reached',
    'despatch_arrived',
    // A tie for a hex is the referee's to break and nothing moves until he does, so the
    // clock has to hand back. `column_blocked` is deliberately not here: one column
    // waiting for another to clear a road settles itself the moment the road clears.
    'column_contested',
    // Twenty troopers running into anything is a die roll that may destroy them, and the
    // rules hand that roll to the referee. Marching the clock past it would be marching
    // past the one thing a patrol exists to produce.
    'patrol_contact',
  ],
  maxAdvanceHours: 24 * 14,

  minDivisionPaperStrength: 4000,
  patrolPaperStrength: 20,
  maxMorale: DEFAULT_MAX_MORALE,
  kindDefaults: DEFAULT_KIND_DEFAULTS,

  marchFatigue: DEFAULT_MARCH_FATIGUE,
  marchFatiguePerHourBeyond: DEFAULT_MARCH_FATIGUE_BEYOND,
  fatigueClass: DEFAULT_FATIGUE_CLASS,
  nightFatiguePerHour: 1,
  formationChangeHours: DEFAULT_FORMATION_CHANGE_HOURS,
  footprint: FOOTPRINT,
};

/** A campaign's overrides, merged onto the defaults. */
export type ConfigOverrides = {
  readonly [K in keyof CampaignConfig]?: CampaignConfig[K];
};

/**
 * A named set of rules, and what it changes.
 *
 * Namespaced rather than global because a referee running two campaigns is often running
 * two different games: one by the book, one with the house amendments he has been arguing
 * about for a year. A campaign records which ruleset it was started under, so a change to
 * the house rules tomorrow does not silently re-tune a game already in progress — the
 * campaign carries its own resolved numbers, and the name is there to say where they came
 * from.
 *
 * `standard` is the rules as written and is the one every other set is expressed against.
 */
export interface Ruleset {
  readonly id: string;
  readonly name: string;
  readonly description: string;
  readonly overrides: ConfigOverrides;
}

export const RULESETS: Readonly<Record<string, Ruleset>> = {
  standard: {
    id: 'standard',
    name: 'Napoleonic Campaign Rules v4',
    description: 'The rules as written. Every number here is the document\u2019s own.',
    overrides: {},
  },
  brisk: {
    id: 'brisk',
    name: 'Brisk',
    description:
      'For a short scenario or a demonstration: columns form up and make camp in half ' +
      'the time, and a day holds fewer hours of marching so fatigue bites sooner.',
    overrides: {
      maxMarchHoursPerDay: 12,
      formationChangeHours: {
        march: { march: 0, battle: 1, rest: 1, occupation: 12, rout: 0 },
        battle: { march: 1, battle: 0, rest: 1, occupation: 12, rout: 0 },
        rest: { march: 1, battle: 1, rest: 0, occupation: 12, rout: 0 },
        occupation: { march: 12, battle: 12, rest: 12, occupation: 0, rout: 0 },
        rout: { march: 0, battle: 0, rest: 0, occupation: 0, rout: 0 },
      },
    },
  },
  quiet: {
    id: 'quiet',
    name: 'Quiet clock',
    description:
      'The clock stops for less. Traffic and arrivals go into the queue without halting ' +
      'an advance, which suits a large campaign with one referee.',
    overrides: {
      haltTriggers: ['enemy_contact', 'crossing_impassable', 'patrol_contact'],
    },
  },
};

export const DEFAULT_RULESET = 'standard';

/**
 * The numbers a campaign actually runs on.
 *
 * A ruleset first, then whatever this particular campaign overrides on top — so a referee
 * can take the house rules and still bend one number for one game without inventing a
 * fourth ruleset to hold it.
 *
 * Shallow: a table given here replaces the table it names rather than merging into it.
 * Merging a speed table row by row would let a half-specified override produce a table
 * that is neither the rules' nor the referee's, and no error anywhere to say so.
 *
 * `base` is what a deployment starts from — normally the rules as written, but a server
 * run with its own numbers passes those, and campaigns created on it inherit them.
 */
export function resolveConfig(
  overrides?: ConfigOverrides,
  rulesetId: string = DEFAULT_RULESET,
  base: CampaignConfig = DEFAULT_CONFIG,
): CampaignConfig {
  const ruleset = RULESETS[rulesetId] ?? RULESETS[DEFAULT_RULESET]!;
  return { ...base, ...ruleset.overrides, ...(overrides ?? {}) };
}

/** Whether a name refers to a ruleset this build knows. */
export const isRuleset = (id: string): boolean => Object.hasOwn(RULESETS, id);
