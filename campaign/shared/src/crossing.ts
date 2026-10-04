/**
 * River crossings.
 *
 * The rules split rivers in two and treat them very differently:
 *
 *   Major (navigable, wide, deep)   with a bridge: 1 hour per division
 *                                   at a ford:     1 hour per division — rare, and
 *                                                  mostly where a road already crosses
 *                                   without:       impossible
 *   Minor (fordable, streams)       with a bridge: free
 *                                   without:       1 hour per division, wading anywhere
 *                                   — except across rapids or a cataract, which is
 *                                   as closed to a column as a major river
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
import { neighbors, type Hex } from './hex.js';
import { CODES, soft, type Violation } from './ruling.js';
import { hasTrait, type Unit } from './unit.js';
import {
  edgeKey,
  riverSideBetween,
  TAG_BRIDGE,
  TAG_CATARACT,
  TAG_FORD,
  TAG_RAPIDS,
  type RiverSide,
  type World,
} from './world.js';

export type RiverClass = 'none' | 'minor' | 'major';

/**
 * Discharge in the generator's units: catchment area times runoff depth, km² × mm.
 *
 * Mirrors `haulage.catchment_carries_a_barge`, which computes the same product before
 * comparing it to `navigable_min_discharge`, with the runoff the world was generated with.
 */
export const discharge = (side: RiverSide, world: World): number =>
  side.catchmentKm2 * world.config.runoffMm;

/** Which side of the rules' Major/Minor line a river side falls. */
export function riverClass(side: RiverSide | undefined, world: World): RiverClass {
  if (side === undefined) return 'none';
  return discharge(side, world) >= world.config.navigableMinDischarge ? 'major' : 'minor';
}

/** Whether a river side runs white — rapids or a cataract — so nobody wades it. */
export const whiteWater = (side: RiverSide): boolean =>
  side.tags.has(TAG_RAPIDS) || side.tags.has(TAG_CATARACT);

/**
 * The rivers along a hex's six sides, largest first, each with the hex across it.
 *
 * What a hex panel describes and what a unit standing there would have to cross to go
 * that way.
 */
export function riversBeside(
  world: World,
  at: Hex,
): { readonly side: RiverSide; readonly across: Hex }[] {
  return neighbors(at)
    .map((n) => ({ side: riverSideBetween(world, at, n), across: n }))
    .filter((x): x is { side: RiverSide; across: Hex } => x.side !== undefined)
    .sort((a, b) => b.side.catchmentKm2 - a.side.catchmentKm2);
}

/**
 * Whether a bridge carries this step over the river between the two hexes.
 *
 * The one place the rules ask. Today it reads the world: a `bridge` on the side, placed
 * by `CrossingStage` where traffic justified the capital or by the road network on a
 * primary road; or any road edge across it — if the generator ran a road over a river
 * here, it built whatever the road needed to get across. A bridge the campaign builds or
 * blows up will be read here too.
 */
export function bridgeAt(world: World, from: Hex, to: Hex): boolean {
  const side = riverSideBetween(world, from, to);
  if (side === undefined) return false;
  return side.tags.has(TAG_BRIDGE) || world.roadEdges.has(edgeKey(from, to));
}

/** Whether an existing crossing serves this step: a bridge, a ford, or none. */
export function crossingAt(world: World, from: Hex, to: Hex): 'bridge' | 'ford' | null {
  if (bridgeAt(world, from, to)) return 'bridge';
  return riverSideBetween(world, from, to)?.tags.has(TAG_FORD) ? 'ford' : null;
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
 * What the step from `from` to `to` costs to cross, and whether it can be made at all.
 *
 * Rivers run along hexsides, so a step crosses a river exactly when the side between the
 * two hexes is one a river runs along. A march along a bank never crosses anything.
 */
export function crossingFor(
  world: World,
  cfg: CampaignConfig,
  unit: Unit,
  from: Hex,
  to: Hex,
): CrossingResult {
  const side = riverSideBetween(world, from, to);
  const river = riverClass(side, world);
  if (river === 'none') return CLEAR;

  const bridged = bridgeAt(world, from, to);

  if (river === 'major') {
    if (bridged) {
      return { river, hours: cfg.majorCrossingHours, how: 'bridge', violations: [] };
    }
    // A big river is waded only where it spreads slack and shallow, and the generator
    // keeps those rare: a ford here is a known crossing, and a column can use it.
    if (side?.tags.has(TAG_FORD) === true) {
      return { river, hours: cfg.majorCrossingHours, how: 'ford', violations: [] };
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
          `a major river between ${from.q},${from.r} and ${to.q},${to.r} cannot be ` +
            `crossed without a bridge, and ${unit.id} has no pontooneers`,
        ),
      ],
    };
  }

  // Minor river: a bridge is free, and wading costs an hour almost anywhere — that is
  // what makes it minor. Not across rapids or a cataract, though: there the water is as
  // closed to a column as a major river's.
  if (bridged) return { river, hours: 0, how: 'bridge', violations: [] };
  if (side !== undefined && whiteWater(side)) {
    if (hasTrait(unit, 'pontooneers')) {
      return { river, hours: cfg.pontoonBuildHours, how: 'pontoon', violations: [] };
    }
    return {
      river,
      hours: Infinity,
      how: 'blocked',
      violations: [
        soft(
          CODES.WHITE_WATER_UNBRIDGED,
          `the river between ${from.q},${from.r} and ${to.q},${to.r} runs white here and ` +
            `cannot be waded, and ${unit.id} has no pontooneers`,
        ),
      ],
    };
  }
  return { river, hours: cfg.fordHours, how: 'ford', violations: [] };
}
