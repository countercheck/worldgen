/**
 * The hours of the day: when the sun is up, and when a column is on the road.
 *
 * Two things live here because they are the same question asked by two people. The
 * referee says when it is light — the season moves, and a campaign that opens in June and
 * is still marching in October has lost four hours of daylight on the way. A commander says
 * when their column marches — "the head steps off at five, is off the road by seven in the
 * evening, and does not spend more than ten hours on it" — which is how a march day was
 * actually ordered, and why no division in the period marched until its twentieth hour
 * unless somebody had decided it should.
 *
 * ## Standing orders belong to a formation, and a commander sets their own
 *
 * A commander may set standing orders for the formation they ride with and no other. For
 * anyone further away the orders go by rider like everything else, and the referee — who
 * reads the despatch — sets them there. A corps commander writing standing orders that
 * reached every division the instant they were written would be the one piece of paper in
 * the campaign that travelled faster than a horse.
 *
 * ## Hours of the day, not hours of the campaign
 *
 * Each limit is read against the clock face. `startHour` and `latestHour` bound a window
 * within one day; `maxHoursOnRoad` is measured on the head, in the same hours the fatigue
 * table reads — `hoursMarchedToday`, which includes time spent standing formed up on a
 * blocked road, because that is time on the road whatever it felt like. A window never
 * crosses midnight. A referee who wants a night march clears the orders.
 */

import type { CampaignConfig } from './config.js';

/**
 * When the head of a column may be on the road.
 *
 * Each limit is optional, and absent means the rules' own: marching until the day's cap.
 */
export interface StandingOrders {
  /** The hour of the day the head steps off, 0–23. Not before. */
  readonly startHour: number | null;
  /** The hour of the day by which the head is off the road, 1–24. */
  readonly latestHour: number | null;
  /** Hours the head may spend on the road in a day, before the rules' cap. */
  readonly maxHoursOnRoad: number | null;
}

/** When the sun rises and sets, as hours of the day. */
export interface Daylight {
  readonly sunriseHour: number;
  readonly sunsetHour: number;
}

export const NO_STANDING_ORDERS: StandingOrders = {
  startHour: null,
  latestHour: null,
  maxHoursOnRoad: null,
};

/** Whether a set of orders says anything at all. */
export const hasStandingOrders = (o: StandingOrders | undefined): o is StandingOrders =>
  o !== undefined && (o.startHour !== null || o.latestHour !== null || o.maxHoursOnRoad !== null);

/** The hour on the clock face, for a campaign hour. */
export const hourOfDay = (atHours: number): number => ((atHours % 24) + 24) % 24;

/**
 * What is wrong with a set of standing orders, as sentences. Empty when there is nothing.
 *
 * The engine's `check` turns these into violations; they are here, beside the orders, so a
 * console could ask the same question before sending rather than finding out from a refusal.
 */
export function standingOrdersProblems(o: StandingOrders, cfg: CampaignConfig): string[] {
  const out: string[] = [];
  const whole = (n: number): boolean => Number.isFinite(n) && Number.isInteger(n);

  if (o.startHour !== null && (!whole(o.startHour) || o.startHour < 0 || o.startHour > 23)) {
    out.push('the hour to step off is a whole hour from 0 to 23');
  }
  if (o.latestHour !== null && (!whole(o.latestHour) || o.latestHour < 1 || o.latestHour > 24)) {
    out.push('the hour to be off the road is a whole hour from 1 to 24');
  }
  if (o.startHour !== null && o.latestHour !== null && o.startHour >= o.latestHour) {
    // A window across midnight would read as a night march, and a night march is a thing a
    // referee orders by clearing the limits rather than one that falls out of a typo.
    out.push('the column has to step off before the hour it is to be off the road');
  }
  if (
    o.maxHoursOnRoad !== null &&
    (!Number.isFinite(o.maxHoursOnRoad) ||
      o.maxHoursOnRoad <= 0 ||
      o.maxHoursOnRoad > cfg.maxMarchHoursPerDay)
  ) {
    out.push(`hours on the road run from above 0 to the rules' ${cfg.maxMarchHoursPerDay}`);
  }
  return out;
}

/** What is wrong with a referee's daylight, as sentences. */
export function daylightProblems(d: Daylight): string[] {
  const out: string[] = [];
  const inDay = (n: number): boolean => Number.isFinite(n) && n >= 0 && n <= 24;
  if (!inDay(d.sunriseHour) || !inDay(d.sunsetHour)) {
    out.push('sunrise and sunset are hours of the day, from 0 to 24');
  } else if (d.sunriseHour >= d.sunsetHour) {
    out.push('the sun has to rise before it sets');
  }
  return out;
}

/**
 * Hours the head may march from `atHours` under its standing orders.
 *
 * `Infinity` when there are none, so a caller can take the minimum with the rules' own cap
 * without a special case. Zero before the hour to step off, at or after the hour to be off
 * the road, and once the day's hours on the road are spent.
 */
export function standingHoursLeft(
  orders: StandingOrders | undefined,
  hoursMarchedToday: number,
  atHours: number,
): number {
  if (orders === undefined) return Infinity;
  const hod = hourOfDay(atHours);
  let left = Infinity;
  if (orders.startHour !== null && hod < orders.startHour) return 0;
  if (orders.latestHour !== null) left = Math.min(left, orders.latestHour - hod);
  if (orders.maxHoursOnRoad !== null) {
    left = Math.min(left, orders.maxHoursOnRoad - hoursMarchedToday);
  }
  return Math.max(0, left);
}

/**
 * Hours marched today, as they will stand at a later hour.
 *
 * The day rolls at midnight, so a question about tomorrow morning is asked of a column
 * that has not marched yet. Without this a column that spent its ten hours yesterday would
 * look spent at five tomorrow and never break camp.
 */
export const hoursMarchedBy = (
  hoursMarchedToday: number,
  nowHours: number,
  thenHours: number,
): number => (Math.floor(thenHours / 24) > Math.floor(nowHours / 24) ? 0 : hoursMarchedToday);

