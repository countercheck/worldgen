/**
 * Grading the ground into the rules' four movement categories.
 *
 * The rules give a speed table with four columns — Highway, Road, Off-road, Bad Going —
 * and say nothing about which ground is which. This is where the generator's output is
 * mapped onto them.
 *
 * Deliberately derived at play time rather than baked into `world.json`. The generator
 * knows nothing about this ruleset and should keep it that way: another game on the same
 * maps would grade them differently, and a tuning change here should not mean
 * regenerating worlds.
 */

import type { CampaignConfig, Grade } from './config.js';
import type { Hex } from './hex.js';
import { hexAt, isWater, roadBetween, type World, type WorldHex } from './world.js';

/**
 * What grade a step from one hex to the next is.
 *
 * The edge decides it first: a road is a road whatever it runs over. `PRIMARY` is the
 * rules' Highway — a metalled, maintained route — while `SECONDARY` and `TRACK` are Road.
 * That mapping is the whole reason `RoadTier` matters to this game.
 *
 * Off the road, the destination hex decides. Bad Going is standing water, closed canopy,
 * bare high ground, or anything steep enough that a column with guns cannot take it
 * straight on. Everything else is Off-road.
 */
export function gradeOf(world: World, cfg: CampaignConfig, from: Hex, to: Hex): Grade {
  const road = roadBetween(world, from, to);
  if (road !== undefined) return road.tier === 'primary' ? 'highway' : 'road';

  const hex = hexAt(world, to);
  if (hex === undefined) return 'bad_going';
  return gradeOfHex(hex, cfg);
}

/** The off-road grade of a hex, ignoring any road on it. */
export function gradeOfHex(hex: WorldHex, cfg: CampaignConfig): Grade {
  if (hex.landCover !== null && cfg.badGoingCovers.includes(hex.landCover)) return 'bad_going';
  if (hex.slope >= cfg.badGoingSlopeMPerKm) return 'bad_going';
  return 'off_road';
}

/** Whether a land unit can be on this hex at all. Water is not a grade, it is a wall. */
export function isPassable(world: World, c: Hex): boolean {
  const hex = hexAt(world, c);
  return hex !== undefined && !isWater(hex);
}

/**
 * Whether a hex carries a river.
 *
 * The generator tags river hexes and gives them an upstream catchment; either alone would
 * do, but a hex with a catchment and no tag is a drainage artefact rather than a
 * watercourse, so both are required.
 */
export const isRiver = (hex: WorldHex): boolean => hex.tags.has('river') && hex.catchmentKm2 > 0;
