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
import { hasTrait, type RoadHour, type Trait, type Unit } from './unit.js';
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
 * The least-*time* route from a unit's head to a goal, through any waypoints on the way.
 *
 * Least time, not least distance, and the difference is the point: a road that runs the
 * long way round is often quicker than the direct line over broken ground, and a courier
 * routed by distance would ignore every road on the map.
 *
 * Uses the shared `astar`, whose heuristic counts hexes at 1.0. That is only admissible
 * if no step can cost less than an hour — which is false on a highway, where a courier
 * covers ten hexes in one. So the heuristic is scaled by the fastest step available to
 * this unit, keeping it a lower bound and the result optimal.
 *
 * `via` chains one search per leg, as `planRide` does for couriers. Null if *any* leg has
 * no route: a waypoint on the far bank of an unbridged river fails the whole march rather
 * than being quietly dropped, because a referee who named it meant it.
 */
export function planMarch(
  world: World,
  cfg: CampaignConfig,
  unit: Unit,
  goal: Hex,
  from?: Hex,
  via: readonly Hex[] = [],
): Hex[] | null {
  const start = from ?? unit.column[0];
  if (start === undefined) return null;

  const legs = [start, ...via, goal];
  const route: Hex[] = [start];

  for (let i = 1; i < legs.length; i++) {
    const leg = marchLeg(world, cfg, unit, legs[i - 1]!, legs[i]!);
    if (leg === null) return null;
    // The first hex of each leg is the last of the previous one.
    route.push(...leg.slice(1));
  }
  return route;
}

/**
 * One leg of a march: a single least-time search between two named places.
 *
 * Kept separate because the legs are searched independently. That is not the same as the
 * cheapest route through all of them — a waypoint is an instruction, so the detour it
 * costs is the point of naming it — but each leg is individually optimal, which is what
 * "and go by way of X" means.
 */
function marchLeg(
  world: World,
  cfg: CampaignConfig,
  unit: Unit,
  start: Hex,
  goal: Hex,
): Hex[] | null {
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

/** The hours a day holds, for a cap read over any twenty-four of them. */
const HOURS_PER_DAY = 24;

/** The most a column may be on the road in any twenty-four hours: the rules' cap, less rest. */
export const marchCapHours = (cfg: CampaignConfig): number =>
  Math.max(0, Math.min(cfg.maxMarchHoursPerDay, HOURS_PER_DAY - cfg.minRestHoursPerDay));

/**
 * Hours on the road in the twenty-four hours up to and including the hour at `atHours`.
 *
 * Read over elapsed time rather than since midnight, so a column that marched through the
 * night does not find its day handed back to it at twelve o'clock.
 */
export const roadHoursWithin = (unit: Unit, atHours: number): number => {
  const hour = Math.floor(atHours);
  let total = 0;
  for (const r of unit.roadHours ?? []) {
    if (r.hour > hour - HOURS_PER_DAY && r.hour <= hour) total += r.hours;
  }
  return total;
};

/**
 * Hours a unit may still march in the hour at `atHours` before the rules' hard cap.
 *
 * "Max march is 20 hours for any unit, patrol, or convoy." Fatigue accrues long before
 * that and is what actually limits a column; this is the wall behind it.
 */
export const marchHoursLeft = (cfg: CampaignConfig, unit: Unit, atHours: number): number =>
  Math.max(0, marchCapHours(cfg) - roadHoursWithin(unit, atHours));

/**
 * Whether a column has been off the road long enough to count as rested at `atHours`.
 *
 * Only asked of a column with hours to shed: one that has not marched since its last rest
 * has nothing to reset. The rest runs from the later of its last hour on the road and the
 * hour a referee set its hours since a rest by hand, so a value set by hand stands until
 * the column has rested after it. A unit with neither — logged before the hours were
 * kept — keeps what it was given until it next marches and stops.
 */
export const hasRested = (cfg: CampaignConfig, unit: Unit, atHours: number): boolean => {
  if (unit.hoursMarchedToday <= 0) return false;
  const last = unit.roadHours?.at(-1);
  const offRoad = Math.max(
    last === undefined ? -Infinity : last.hour + 1,
    unit.restFromHours ?? -Infinity,
  );
  return Math.floor(atHours) - offRoad >= cfg.minRestHoursPerDay;
};

/**
 * `hours` on the road laid down as whole hours running back from `untilHour`, the earliest
 * taking any fraction: a history for a column somebody says has already marched.
 */
export const roadHoursEnding = (hours: number, untilHour: number): RoadHour[] => {
  const out: RoadHour[] = [];
  let left = Math.min(hours, HOURS_PER_DAY);
  for (let hour = Math.floor(untilHour) - 1; left > 1e-9; hour--) {
    out.unshift({ hour, hours: Math.min(1, left) });
    left -= 1;
  }
  return out;
};

/**
 * A unit after `hours` more on the road in the hour at `atHours`.
 *
 * Kept by the hour, and only a day of it: the cap never looks further back.
 */
export const onRoad = (unit: Unit, atHours: number, hours: number): Unit => {
  const hour = Math.floor(atHours);
  const kept = (unit.roadHours ?? []).filter((r) => r.hour > hour - HOURS_PER_DAY);
  const last = kept[kept.length - 1];
  const roadHours =
    last !== undefined && last.hour === hour
      ? [...kept.slice(0, -1), { hour, hours: last.hours + hours }]
      : [...kept, { hour, hours }];
  return { ...unit, hoursMarchedToday: unit.hoursMarchedToday + hours, roadHours };
};

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

/** Whether a unit may field patrols without spending troops. */
export const canPatrolFreely = (unit: Unit): boolean => hasTrait(unit, 'scout');
