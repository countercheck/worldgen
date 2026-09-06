/**
 * River crossings.
 *
 * The rules split rivers in two and treat them very differently:
 *
 *   Major (navigable, wide, deep)   with a bridge: 1 hour per division
 *                                   without:       impossible
 *   Minor (fordable, streams)       with a bridge: free
 *                                   without:       1 hour per division, at a ford
 *
 * The line between them is already drawn in the generator. `haulage.navigable()` tests a
 * watercourse's discharge against `navigable_min_discharge` and is documented there as
 * the difference between "something you ford" and "something you ship grain down" —
 * which is exactly the rules' Major/Minor distinction, already calibrated against the
 * hydrology. Reusing it means the two models cannot disagree about which river is which.
 *
 * Discharge is computed from upstream catchment, not from `riverFlow`: the latter is a
 * normalised rank for drawing line widths, and reading it as a physical quantity would
 * make a river's class depend on how many other rivers the map happens to have.
 *
 * A note on what this costs a corps. The delay is charged per division, so ten divisions
 * at one bridge queue for ten hours. That is not a special case in the code — it falls
 * out of each unit paying — and it is the sort of thing the game is about.
 */

import type { CampaignConfig } from './config.js';
import type { Hex } from './hex.js';
import { CODES, soft, type Violation } from './ruling.js';
import { isRiver } from './terrain.js';
import { hasTrait, type Unit } from './unit.js';
import { edgeKey, hexAt, type World, type WorldHex } from './world.js';

export type RiverClass = 'none' | 'minor' | 'major';

/**
 * Discharge in the generator's units: catchment area times runoff depth, km² × mm.
 *
 * Mirrors `haulage.navigable`, which computes the same product before comparing it to
 * `navigable_min_discharge`.
 */
export const discharge = (hex: WorldHex, world: World): number =>
  hex.catchmentKm2 * world.config.meanPrecipMm;

/** Which side of the rules' Major/Minor line a hex's watercourse falls. */
export function riverClass(hex: WorldHex, world: World): RiverClass {
  if (!isRiver(hex)) return 'none';
  return discharge(hex, world) >= world.config.navigableMinDischarge ? 'major' : 'minor';
}

/**
 * Whether an existing crossing serves this step.
 *
 * Three things count, and the last is the one that matters for worlds the generator
 * built without `CrossingStage`:
 *
 * - a `bridge` tag, placed by `CrossingStage` where traffic justified the capital;
 * - a `ford` tag, placed where the water is wadeable;
 * - a road edge across the channel. If the generator ran a road over a river here, it
 *   built whatever the road needed to get across.
 */
export function crossingAt(world: World, from: Hex, to: Hex): 'bridge' | 'ford' | null {
  const hex = hexAt(world, to);
  if (hex === undefined) return null;
  if (hex.tags.has('bridge')) return 'bridge';
  if (world.roadEdges.has(edgeKey(from, to))) return 'bridge';
  if (hex.tags.has('ford')) return 'ford';
  return null;
}

export interface CrossingResult {
  readonly river: RiverClass;
  /** Hours the crossing adds to this step. `Infinity` means it cannot be made. */
  readonly hours: number;
  /** How the unit got across, for the log and for the UI. */
  readonly how: 'none' | 'bridge' | 'ford' | 'pontoon' | 'blocked';
  readonly violations: readonly Violation[];
}

const CLEAR: CrossingResult = { river: 'none', hours: 0, how: 'none', violations: [] };

/**
 * What crossing into `to` costs, and whether it can be done at all.
 *
 * Fires only when entering a river hex. Marching *along* a bank is not a crossing, and a
 * unit that is already in the channel is not crossing it again — both are handled by only
 * charging when the destination carries the river and the origin does not.
 */
export function crossingFor(
  world: World,
  cfg: CampaignConfig,
  unit: Unit,
  from: Hex,
  to: Hex,
): CrossingResult {
  const target = hexAt(world, to);
  if (target === undefined) return CLEAR;

  const river = riverClass(target, world);
  if (river === 'none') return CLEAR;

  // Already in the water: moving along a watercourse is not a fresh crossing. Without
  // this a unit following a valley would pay a crossing every hex, which is both wrong
  // and would make river valleys — the best going on the map — the worst.
  const origin = hexAt(world, from);
  if (origin !== undefined && isRiver(origin)) return CLEAR;

  const crossing = crossingAt(world, from, to);

  if (river === 'major') {
    if (crossing === 'bridge') {
      return { river, hours: cfg.majorCrossingHours, how: 'bridge', violations: [] };
    }
    if (hasTrait(unit, 'pontooneers')) {
      return { river, hours: cfg.pontoonBuildHours, how: 'pontoon', violations: [] };
    }
    // Soft, not hard: it is a rule of the game, and a referee modelling a hard frost or
    // a boat bridge the map does not know about must be able to let a unit across.
    return {
      river,
      hours: Infinity,
      how: 'blocked',
      violations: [
        soft(
          CODES.MAJOR_RIVER_UNBRIDGED,
          `a major river at ${to.q},${to.r} cannot be crossed without a bridge, and ` +
            `${unit.id} has no pontooneers`,
        ),
      ],
    };
  }

  // Minor river: a bridge is free, a ford costs an hour, and there is always a ford —
  // that is what makes it minor.
  if (crossing === 'bridge') return { river, hours: 0, how: 'bridge', violations: [] };
  return { river, hours: cfg.fordHours, how: 'ford', violations: [] };
}
