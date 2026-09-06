/**
 * Marching: how fast, how long, and by what route.
 *
 * One hex is one kilometre, so the cost of entering a hex is a *time*: `1 / speed` hours,
 * plus whatever a river crossing adds. There are no movement points anywhere in this
 * game — a unit has hours in the day, and terrain decides how many hexes they buy.
 *
 * The router and the mover share one cost function, deliberately. A pathfinder that
 * disagrees with the rules about what a step costs is the classic wargame bug: it plans
 * routes the unit then cannot follow, and the discrepancy shows up as units mysteriously
 * stopping short.
 */

import { columnHexes } from './column.js';
import { GRADES, type CampaignConfig, type Grade, type Mover } from './config.js';
import { crossingFor } from './crossing.js';
import { astar, distance, key, neighbors, type Hex } from './hex.js';
import { gradeOf, isPassable } from './terrain.js';
import { hasTrait, type Trait, type Unit } from './unit.js';
import { hexAt, type World, type WorldHex } from './world.js';

/** March speed in km/h for a mover on a given grade, after traits. */
export function speedKmh(
  cfg: CampaignConfig,
  mover: Mover,
  grade: Grade,
  traits: readonly Trait[] = [],
): number {
  const base = cfg.speeds[mover][grade];
  let modifier = 0;
  for (const t of traits) modifier += cfg.traitSpeedKmh[t] ?? 0;
  return Math.max(cfg.minSpeedKmh, base + modifier);
}

/** A unit's speed on a grade. */
export const unitSpeedKmh = (cfg: CampaignConfig, unit: Unit, grade: Grade): number =>
  speedKmh(cfg, unit.kind, grade, unit.traits);

export interface StepCost {
  readonly grade: Grade;
  /** Total hours to enter, march plus crossing. `Infinity` if the step cannot be made. */
  readonly hours: number;
  readonly marchHours: number;
  readonly crossingHours: number;
  readonly crossing: ReturnType<typeof crossingFor>;
}

const IMPASSABLE: StepCost = {
  grade: 'bad_going',
  hours: Infinity,
  marchHours: Infinity,
  crossingHours: 0,
  crossing: { river: 'none', hours: 0, how: 'blocked', violations: [] },
};

/**
 * What it costs a unit to step from one hex to an adjacent one.
 *
 * Water is not slow, it is impassable — the rules have no naval movement, and a land
 * unit does not enter a lake at any speed.
 */
export function stepCost(
  world: World,
  cfg: CampaignConfig,
  unit: Unit,
  from: Hex,
  to: Hex,
): StepCost {
  if (!isPassable(world, to)) return IMPASSABLE;

  const grade = gradeOf(world, cfg, from, to);
  const marchHours = 1 / unitSpeedKmh(cfg, unit, grade);
  const crossing = crossingFor(world, cfg, unit, from, to);

  return {
    grade,
    marchHours,
    crossingHours: crossing.hours,
    hours: marchHours + crossing.hours,
    crossing,
  };
}

/** Hours to enter a hex — the number the router sorts on. */
export const hoursToEnter = (
  world: World,
  cfg: CampaignConfig,
  unit: Unit,
  from: Hex,
  to: Hex,
): number => stepCost(world, cfg, unit, from, to).hours;

/**
 * The least-*time* route from a unit's head to a goal.
 *
 * Least time, not least distance, and the difference is the point: a road that runs the
 * long way round is often quicker than the direct line over broken ground, and a courier
 * routed by distance would ignore every road on the map.
 *
 * Uses the shared `astar`, whose heuristic counts hexes at 1.0. That is only admissible
 * if no step can cost less than an hour — which is false on a highway, where a courier
 * covers ten hexes in one. So the heuristic is scaled by the fastest step available to
 * this unit, keeping it a lower bound and the result optimal.
 */
export function planMarch(
  world: World,
  cfg: CampaignConfig,
  unit: Unit,
  goal: Hex,
  from?: Hex,
): Hex[] | null {
  const start = from ?? unit.column[0];
  if (start === undefined) return null;

  const perHex = fastestHoursPerHex(cfg, unit);

  return astar<WorldHex>(
    world.hexes,
    start,
    goal,
    // Node cost is unused: everything here depends on the pair of hexes, not on the
    // destination alone, so it is all charged as an edge.
    () => 0,
    (_a, _b, fromCoord, toCoord) => hoursToEnter(world, cfg, unit, fromCoord, toCoord),
    (a, b) => distance(a, b) * perHex,
  );
}

/**
 * The least time this unit could possibly spend on one hex, over any grade.
 *
 * Scales the pathfinder's heuristic into hours. `astar`'s default counts hexes at 1.0,
 * which is an overestimate the moment a unit moves faster than 1 km/h — a courier on a
 * highway crosses a hex in a tenth of an hour — and an overestimating heuristic makes A*
 * return whatever it finds first rather than the cheapest route. Multiplying the hex
 * count by the fastest step available keeps it a lower bound, so the path stays optimal.
 */
export const fastestHoursPerHex = (cfg: CampaignConfig, unit: Unit): number => {
  const best = Math.max(
    ...GRADES.map((g) => unitSpeedKmh(cfg, unit, g)),
  );
  return best > 0 ? 1 / best : 1;
};

/** Total hours for a path, and whether every step of it can actually be made. */
export function pathHours(
  world: World,
  cfg: CampaignConfig,
  unit: Unit,
  path: readonly Hex[],
): number {
  let total = 0;
  for (let i = 1; i < path.length; i++) {
    total += hoursToEnter(world, cfg, unit, path[i - 1]!, path[i]!);
  }
  return total;
}

export interface Reach {
  /** Hours to reach each hex within the budget. */
  readonly hours: ReadonlyMap<string, number>;
  /** The step each hex was reached from, for drawing the route to any of them. */
  readonly cameFrom: ReadonlyMap<string, string>;
}

/**
 * Everywhere a unit can get within a time budget.
 *
 * A Dijkstra rather than an A*, because there is no goal to aim at. This is what the
 * client draws as a reach overlay — and on a commander's screen it runs over the *masked*
 * world, so the ground a unit could reach through country nobody has scouted simply is
 * not offered. That is correct rather than a limitation: a plan is only as good as the map
 * it was made on.
 */
export function reachable(
  world: World,
  cfg: CampaignConfig,
  unit: Unit,
  budgetHours: number,
  from?: Hex,
): Reach {
  const start = from ?? unit.column[0];
  const hours = new Map<string, number>();
  const cameFrom = new Map<string, string>();
  if (start === undefined || budgetHours < 0) return { hours, cameFrom };

  hours.set(key(start), 0);

  // A simple sorted frontier. The reach of a day's march is a few hundred hexes at most,
  // so a binary heap would be optimising a loop that does not appear in any profile.
  const frontier: { at: Hex; cost: number }[] = [{ at: start, cost: 0 }];

  while (frontier.length > 0) {
    frontier.sort((a, b) => a.cost - b.cost || a.at.q - b.at.q || a.at.r - b.at.r);
    const { at, cost } = frontier.shift()!;
    if (cost > (hours.get(key(at)) ?? Infinity)) continue;

    for (const next of neighbors(at)) {
      if (hexAt(world, next) === undefined) continue;
      const step = hoursToEnter(world, cfg, unit, at, next);
      if (!Number.isFinite(step)) continue;

      const total = cost + step;
      if (total > budgetHours) continue;
      if (total >= (hours.get(key(next)) ?? Infinity)) continue;

      hours.set(key(next), total);
      cameFrom.set(key(next), key(at));
      frontier.push({ at: next, cost: total });
    }
  }

  return { hours, cameFrom };
}

/**
 * Hours a unit has left today before the rules' hard march cap.
 *
 * "Max march is 20 hours for any unit, patrol, or convoy." Fatigue accrues long before
 * that and is what actually limits a column; this is the wall behind it.
 */
export const marchHoursLeftToday = (cfg: CampaignConfig, unit: Unit): number =>
  Math.max(0, cfg.maxMarchHoursPerDay - unit.hoursMarchedToday);

/** Whether two hexes are adjacent — the one geometric precondition of a march step. */
export const isStep = (a: Hex, b: Hex): boolean => distance(a, b) === 1;

/**
 * How much of a unit's own column is on a hex it just left.
 *
 * Not used for movement cost, but the reason a unit cannot simply be "moved" — the tail
 * takes `catchupHours` to follow, and until it does the unit is spread across the ground
 * behind it.
 */
export const tailLengthHexes = (unit: Unit, grade: Grade): number => columnHexes(unit, grade) - 1;

/** Whether a unit may field patrols without spending effectives. */
export const canPatrolFreely = (unit: Unit): boolean => hasTrait(unit, 'scout');
