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
 * ## Hours of the day, and hours on the road
 *
 * `startHour` and `latestHour` are read against the clock face and bound a window within
 * one day. Left blank, the step-off hour is dawn — the referee's sunrise — because that is
 * when an order that named no hour had a column on the road. A window never crosses
 * midnight, so a column under orders is off the road by it. A referee who wants a night
 * march clears the orders.
 *
 * `maxHoursOnRoad` is not read against the clock. It counts the head's hours on the road in
 * the last twenty-four, the same count the rules' cap reads, and includes time spent standing
 * formed up on a blocked road, because that is time on the road whatever it felt like.
 */

import type { CampaignConfig } from './config.js';
import { marchCapHours } from './movement.js';

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

const FIELDS = ['startHour', 'latestHour', 'maxHoursOnRoad'] as const;

/**
 * What is wrong with a set of standing orders, as sentences. Empty when there is nothing.
 *
 * Takes whatever arrived rather than trusting the type: orders come off the wire, go into
 * the log for good, and are sent back out in every view, so a missing field or a stray one
 * is refused here rather than stored.
 *
 * The engine's `check` turns these into violations; they are here, beside the orders, so a
 * console could ask the same question before sending rather than finding out from a refusal.
 * `cfg` is the campaign's with the referee's daylight in it, for the dawn a blank step-off
 * hour means.
 */
export function standingOrdersProblems(raw: unknown, cfg: CampaignConfig): string[] {
  const shape = 'standing orders are a step-off hour, an hour to be off the road, and hours on it';
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) return [shape];
  const fields = raw as Record<string, unknown>;
  if (Object.keys(fields).some((k) => !(FIELDS as readonly string[]).includes(k))) return [shape];
  if (FIELDS.some((f) => fields[f] !== null && typeof fields[f] !== 'number')) {
    return ['each of the standing orders is an hour, or blank'];
  }
  const o = raw as StandingOrders;

  const out: string[] = [];
  const whole = (n: number): boolean => Number.isFinite(n) && Number.isInteger(n);
  const cap = marchCapHours(cfg);

  const startOk =
    o.startHour === null || (whole(o.startHour) && o.startHour >= 0 && o.startHour <= 23);
  const latestOk =
    o.latestHour === null || (whole(o.latestHour) && o.latestHour >= 1 && o.latestHour <= 24);
  if (!startOk) out.push('the hour to step off is a whole hour from 0 to 23');
  if (!latestOk) out.push('the hour to be off the road is a whole hour from 1 to 24');
  if (startOk && latestOk) {
    if (o.startHour !== null && o.latestHour !== null && o.startHour >= o.latestHour) {
      // A window across midnight would read as a night march, and a night march is a thing
      // a referee orders by clearing the limits rather than one that falls out of a typo.
      out.push('the column has to step off before the hour it is to be off the road');
    } else if (o.startHour === null && o.latestHour !== null && o.latestHour <= cfg.sunriseHour) {
      out.push('the column steps off at dawn, so it has to be off the road after it');
    }
  }
  if (
    o.maxHoursOnRoad !== null &&
    (!Number.isFinite(o.maxHoursOnRoad) || o.maxHoursOnRoad <= 0 || o.maxHoursOnRoad > cap)
  ) {
    out.push(`hours on the road run from above 0 to the rules' ${cap}`);
  }
  return out;
}

/** The orders and nothing else, for the log. */
export const standingOrdersOf = (o: StandingOrders): StandingOrders => ({
  startHour: o.startHour,
  latestHour: o.latestHour,
  maxHoursOnRoad: o.maxHoursOnRoad,
});

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
 * without a special case. Zero before the hour to step off — dawn, if the orders name none —
 * at or after the hour to be off the road, and once the hours on the road are spent.
 * `roadHours` is the head's hours on the road over the last twenty-four, as the cap reads
 * them.
 */
export function standingHoursLeft(
  orders: StandingOrders | undefined,
  roadHours: number,
  atHours: number,
  sunriseHour: number,
): number {
  if (!hasStandingOrders(orders)) return Infinity;
  const hod = hourOfDay(atHours);
  let left = Infinity;
  if (hod < (orders.startHour ?? sunriseHour)) return 0;
  if (orders.latestHour !== null) left = Math.min(left, orders.latestHour - hod);
  if (orders.maxHoursOnRoad !== null) {
    left = Math.min(left, orders.maxHoursOnRoad - roadHours);
  }
  return Math.max(0, left);
}
