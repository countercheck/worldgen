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
import type { Trait, UnitKind } from './unit.js';

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
  /** Patrols a scout unit may field without spending effectives. */
  readonly freePatrols: number;
  /** Effectives permanently lost per patrol beyond the free ones. */
  readonly extraPatrolCost: number;
  /** How far gunfire carries, in km. */
  readonly gunfireRangeKm: number;

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
  /**
   * The clock's granularity while advancing, in hours.
   *
   * Riders and columns are stepped hex by hex, so this only bounds how finely two events
   * in the same quarter-hour are ordered against each other. Smaller is more faithful and
   * slower; a quarter of an hour is well under the time anything takes to cross a hex.
   */
  readonly tickHours: number;
  /** A ceiling on one advance, so a mistyped `advance 1000` cannot lock the server up. */
  readonly maxAdvanceHours: number;
}

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
  ],
  tickHours: 0.25,
  maxAdvanceHours: 24 * 14,
};

/** A campaign's overrides, merged onto the defaults. */
export type ConfigOverrides = {
  readonly [K in keyof CampaignConfig]?: CampaignConfig[K];
};

export function resolveConfig(overrides?: ConfigOverrides): CampaignConfig {
  return overrides === undefined ? DEFAULT_CONFIG : { ...DEFAULT_CONFIG, ...overrides };
}
