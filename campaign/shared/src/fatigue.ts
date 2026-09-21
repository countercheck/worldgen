/**
 * What a march costs the troops.
 *
 * Fatigue is the rules' real limit on movement. The twenty-hour cap is a wall; this is the
 * gradient that makes a commander stop before they hit it. A division that force-marches
 * through a night arrives, and arrives useless — `presentUnderArms` is `paperStrength`
 * reduced by fatigue, so the cost is paid in the line of battle rather than on the road.
 *
 * ## Cumulative, not per-hour
 *
 * The table is held as a running total by hour, so the cost of marching from hour *a* to
 * hour *b* is `at(b) - at(a)`. That is what lets fatigue be charged a hex at a time
 * without the engine having to remember which band it was in — an hour that straddles the
 * eleventh hour is charged correctly by subtraction and by nothing else.
 *
 * ## Two clocks, and the tail is on the second one
 *
 * The hours that drive the table are the *head's*: how long the tip of the column was on
 * the road. Night is measured differently, over any part of the column, because a division
 * two kilometres long is not off the road when its head halts — the rear is still marching
 * in the dark for as long as it takes to close up. A march that ends at dusk costs a night
 * march for the tail even though the head camped in daylight.
 */

import { catchupHours } from './column.js';
import type { CampaignConfig } from './config.js';
import type { Unit } from './unit.js';

/**
 * Cumulative march fatigue at a number of hours on the road.
 *
 * Experience shifts the column: a veteran reads the table an hour to the left and a raw
 * formation an hour to the right, per point. That is the rules' own mechanism and it is
 * the reason experience is worth having — it does not make troops march faster, it makes
 * them arrive able to fight.
 */
export function marchFatigueAt(cfg: CampaignConfig, unit: Unit, hours: number): number {
  const curve = cfg.marchFatigue[cfg.fatigueClass[unit.kind]];
  const beyond = cfg.marchFatiguePerHourBeyond[cfg.fatigueClass[unit.kind]];
  if (curve.length === 0) return 0;

  // Whole hours: the table has no reading for half an hour, and rounding down is the
  // reading that does not charge troops for time they have not yet spent.
  // Minus, not plus: a veteran reads the table to the left, which is a lower reading for
  // the same hours on the road.
  const shifted = Math.floor(Math.max(0, hours)) - unit.experience;
  if (shifted < 0) return curve[0]!;
  if (shifted < curve.length) return curve[shifted]!;

  const last = curve.length - 1;
  return curve[last]! + beyond * (shifted - last);
}

/** What marching from `from` hours to `to` hours costs, both measured on the head. */
export const marchFatigueBetween = (
  cfg: CampaignConfig,
  unit: Unit,
  from: number,
  to: number,
): number => Math.max(0, marchFatigueAt(cfg, unit, to) - marchFatigueAt(cfg, unit, from));

/** Whether the sun is down at a campaign hour. */
export function isDark(cfg: CampaignConfig, atHours: number): boolean {
  const hour = ((atHours % 24) + 24) % 24;
  return hour < cfg.sunriseHour || hour >= cfg.sunsetHour;
}

/**
 * How many of the hours in `[from, to)` are dark.
 *
 * Walks the boundaries rather than sampling: a step of a few minutes that happens to
 * straddle sunset has to be charged for the part of it that was dark, and a sampled
 * midpoint would charge all or nothing. Handles a window spanning several nights, which a
 * twenty-hour march does.
 */
export function darkHoursBetween(cfg: CampaignConfig, from: number, to: number): number {
  if (!(to > from)) return 0;

  let dark = 0;
  let at = from;
  while (at < to) {
    // The next sunrise or sunset strictly after `at`, whichever comes first.
    const day = Math.floor(at / 24) * 24;
    const marks = [
      day + cfg.sunriseHour,
      day + cfg.sunsetHour,
      day + 24 + cfg.sunriseHour,
      day + 24 + cfg.sunsetHour,
    ];
    const next = Math.min(to, ...marks.filter((m) => m > at + 1e-12));
    if (isDark(cfg, (at + next) / 2)) dark += next - at;
    at = next;
  }
  return dark;
}

/**
 * The window over which *any part* of a column is still moving, for a head that marched
 * `[from, to)`.
 *
 * The tail is behind the head by the time it takes to close up, so the column is not clear
 * of the road until `catchupHours` after the head has stopped. This is the union of the
 * head's window and the tail's, which is simply the head's window with the catch-up added
 * to the end.
 */
export const columnMotionWindow = (
  unit: Unit,
  from: number,
  to: number,
  speedKmh: number,
): { from: number; to: number } => ({ from, to: to + catchupHours(unit, speedKmh) });

/** Fatigue for the dark hours any part of the column spent on the road in a window. */
export const nightFatigue = (
  cfg: CampaignConfig,
  from: number,
  to: number,
): number => cfg.nightFatiguePerHour * darkHoursBetween(cfg, from, to);
