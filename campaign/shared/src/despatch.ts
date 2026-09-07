/**
 * Despatches: the paper, the rider, and the fate of both.
 *
 * Everything a commander knows beyond his own horizon arrives here. A despatch runs from
 * one man to another — the formation is only the address a rider must find — and it takes
 * real hours to get there, may be intercepted on the way, and is never acknowledged
 * unless a second despatch makes the return trip.
 *
 * ## An order is a message, not a command
 *
 * The body is prose. The engine executes nothing on its own: the referee reads what was
 * written, decides what the addressee makes of it, and sets a task. That separation is
 * the whole design and it buys three things at once. Order validation lives where
 * movement already lives rather than existing twice. Interception becomes intelligence
 * rather than a coordinate dump. And "a formation with no new orders continues its last
 * one" needs no special case, because the task never stopped.
 *
 * ## The route is the leak
 *
 * A rider takes the least-time path to the addressee's **actual** position, so the route
 * is computed from ground truth — which means the route betrays that position. Show a
 * commander where his rider went and you have told him exactly where his detached corps
 * is, and he need never read a report again: he would read his own outbox instead.
 *
 * So `route` is referee-only, absolutely, and so is anything derived from it. A delivery
 * estimate is a distance, and a distance is a position, so no despatch carries an ETA
 * either. `senderCopy` below is the only shape a sender is ever shown, and the leakage
 * suite asserts against the serialised bytes that nothing else escapes.
 *
 * The client may still show a commander *his own* guess — routed from his last report of
 * that formation, over the map he actually holds, and labelled as the guess it is. That
 * is the shared engine earning its keep: the same routing code, run over worse data.
 */

import { occupied } from './column.js';
import type { CampaignConfig } from './config.js';
import { crossingAt, riverClass } from './crossing.js';
import { astar, distance, key, neighbors, type Hex } from './hex.js';
import { speedKmh } from './movement.js';
import type { Contact } from './recon.js';
import { gradeOf, isPassable, isRiver } from './terrain.js';
import type { Unit, UnitReport } from './unit.js';
import { hexAt, type World, type WorldHex } from './world.js';

/**
 * What a despatch is for.
 *
 * Three kinds and no more. An order travels downward, a report travels anywhere, and an
 * acknowledgement is the only feedback channel in the game — which is why it is a
 * despatch in its own right, and can itself be lost.
 */
export type DespatchKind = 'order' | 'report' | 'acknowledgement';

/**
 * What is written on the paper.
 *
 * An order is text and nothing else. An automatic contact report is data and nothing
 * else. A commander writing up a sighting himself may send both, which is why these are
 * optional fields rather than a union.
 */
export interface DespatchBody {
  readonly text?: string;
  readonly contacts?: readonly Contact[];
  /**
   * Where the sender's own formation stood when he sealed it.
   *
   * Attached to every despatch, whether or not anybody asked for it: a rider who has come
   * from III Corps knows where III Corps was when he left, and that is most of what a
   * despatch was actually for. It is why an order arriving also refreshes the recipient's
   * picture of the man who sent it — and why silence from a corps is not merely a missing
   * order but a stale map.
   */
  readonly unitReport?: UnitReport;
}

/**
 * What became of a rider.
 *
 * Known to the referee always, to the addressee once it has arrived, to a captor when he
 * takes it — and to the sender never. That last is not an oversight to be tidied up
 * later: it is the mechanic.
 */
export type Fate =
  | { readonly kind: 'in_transit' }
  | { readonly kind: 'delivered'; readonly atHours: number }
  /** The rider was stopped and the paper lost with him. */
  | {
      readonly kind: 'lost';
      readonly by: string;
      readonly atHours: number;
      readonly dice: readonly number[];
    }
  /** Worse: the paper was read. `by` is the faction that now holds it. */
  | {
      readonly kind: 'captured';
      readonly by: string;
      readonly atHours: number;
      readonly dice: readonly number[];
    };

export interface Despatch {
  readonly id: string;
  readonly kind: DespatchKind;
  /** Commander id. Despatches run man to man; the formation is only the address. */
  readonly from: string;
  readonly to: string;
  readonly faction: string;
  /** The hour it was written, which is the hour its contents describe. */
  readonly sentAtHours: number;
  readonly body: DespatchBody;
  /** Waypoints the sender insisted on — around a wood he thinks holds pickets. */
  readonly via: readonly Hex[];
  /** A report being passed on. Two lags stack, which is very much the period. */
  readonly forwardedFrom: string | null;
  readonly inReplyTo: string | null;
  /**
   * The rider's path to the addressee's actual position.
   *
   * REFEREE ONLY. See the module comment: this field is the one place in the design
   * where a leak would be both invisible and total.
   */
  readonly route: readonly Hex[];
  /** How far along `route` the rider has got. Referee only, for the same reason. */
  readonly progress: number;
  readonly fate: Fate;
  /**
   * Handed over rather than ridden: the two formations were touching.
   *
   * No rider, no time, no interception. Concentration buys command, which is the tension
   * the period turns on — concentrate and command well, disperse and forage well.
   */
  readonly handed: boolean;
}

export const isInTransit = (d: Despatch): boolean => d.fate.kind === 'in_transit';
export const isDelivered = (d: Despatch): boolean => d.fate.kind === 'delivered';

/** The hour a despatch reached its addressee, or null if it never has. */
export const deliveredAt = (d: Despatch): number | null =>
  d.fate.kind === 'delivered' ? d.fate.atHours : null;

/**
 * A despatch as its **sender** may see it.
 *
 * Everything he wrote, and not one thing more. No route, because the route is where the
 * addressee is; no fate, because learning that his rider was taken would tell him his
 * order never arrived, which no commander in 1815 could know without being told.
 *
 * Built by construction rather than by deletion. A `delete d.route` would leak the day
 * somebody adds a field and forgets the line.
 */
export interface SentDespatch {
  readonly id: string;
  readonly kind: DespatchKind;
  readonly to: string;
  readonly sentAtHours: number;
  readonly body: DespatchBody;
  /** His own waypoints, which he chose and therefore already knows. */
  readonly via: readonly Hex[];
  readonly inReplyTo: string | null;
  /** Whether it was handed over on the spot. He watched that happen. */
  readonly handed: boolean;
  /**
   * Whether an acknowledgement has come back for it.
   *
   * The only thing a sender ever learns about a despatch's fate, and only because a
   * second rider made the return trip.
   */
  readonly acknowledged: boolean;
}

/** A despatch as its **addressee** sees it: only once it is actually in his hand. */
export interface ReceivedDespatch {
  readonly id: string;
  readonly kind: DespatchKind;
  readonly from: string;
  /** The hour it describes. Shown first, because it is what he now knows about. */
  readonly sentAtHours: number;
  /** The hour it reached him. The gap between the two is the fog. */
  readonly receivedAtHours: number;
  readonly body: DespatchBody;
  readonly forwardedFrom: string | null;
  readonly inReplyTo: string | null;
  /**
   * An order overtaken by a later one already in hand.
   *
   * Marked rather than hidden: correct staff practice is to disregard it, and seeing
   * that happen is half of understanding why the corps did what it did.
   */
  readonly superseded: boolean;
}

/** A despatch as a **captor** sees it: the body, and the fact that he took it. */
export interface CapturedDespatch {
  readonly id: string;
  readonly kind: DespatchKind;
  readonly from: string;
  readonly to: string;
  readonly faction: string;
  readonly sentAtHours: number;
  readonly capturedAtHours: number;
  readonly body: DespatchBody;
}

export const senderCopy = (d: Despatch, acknowledged: boolean): SentDespatch => ({
  id: d.id,
  kind: d.kind,
  to: d.to,
  sentAtHours: d.sentAtHours,
  body: d.body,
  via: d.via,
  inReplyTo: d.inReplyTo,
  handed: d.handed,
  acknowledged,
});

export const addresseeCopy = (d: Despatch, superseded: boolean): ReceivedDespatch => ({
  id: d.id,
  kind: d.kind,
  from: d.from,
  sentAtHours: d.sentAtHours,
  receivedAtHours: deliveredAt(d) ?? d.sentAtHours,
  body: d.body,
  forwardedFrom: d.forwardedFrom,
  inReplyTo: d.inReplyTo,
  superseded,
});

export const captorCopy = (d: Despatch): CapturedDespatch => ({
  id: d.id,
  kind: d.kind,
  from: d.from,
  to: d.to,
  faction: d.faction,
  sentAtHours: d.sentAtHours,
  capturedAtHours: d.fate.kind === 'captured' ? d.fate.atHours : d.sentAtHours,
  body: d.body,
});

/**
 * Whether an arriving order has been overtaken.
 *
 * Every despatch carries the hour it was written, and a commander already holding a later
 * order disregards an earlier one that turns up afterwards. The comparison is on the date
 * alone — which is all an engine can do with prose, and all it needs to do.
 *
 * Only orders supersede orders. A report is never stale in this sense: an old report is
 * still a fact about an old hour, and remains worth having.
 */
export function isSuperseded(d: Despatch, held: readonly Despatch[]): boolean {
  if (d.kind !== 'order') return false;
  const mine = deliveredAt(d);
  if (mine === null) return false;

  return held.some(
    (other) =>
      other.id !== d.id &&
      other.kind === 'order' &&
      other.to === d.to &&
      other.sentAtHours > d.sentAtHours &&
      (deliveredAt(other) ?? Infinity) <= mine,
  );
}

// ---- riding ------------------------------------------------------------

/**
 * What a lone rider pays to cross water, where a division would be stopped.
 *
 * A courier is one man on one horse: he fords where a column cannot, finds the boat, or
 * swims the animal. So a major river costs him an hour rather than being impassable —
 * the rules do not say so explicitly, but a courier system in which one river ends
 * communication altogether is not the period, and the whole point of the rider is that
 * he gets through or is caught trying.
 */
function courierCrossingHours(cfg: CampaignConfig, world: World, from: Hex, to: Hex): number {
  const target = hexAt(world, to);
  if (target === undefined || !isRiver(target)) return 0;

  const origin = hexAt(world, from);
  if (origin !== undefined && isRiver(origin)) return 0;

  if (crossingAt(world, from, to) === 'bridge') return 0;
  return riverClass(target, world) === 'major' ? cfg.courierMajorCrossingHours : cfg.fordHours;
}

/** Hours for a rider to enter one hex from an adjacent one. `Infinity` if he cannot. */
export function courierStepHours(world: World, cfg: CampaignConfig, from: Hex, to: Hex): number {
  if (!isPassable(world, to)) return Infinity;
  const grade = gradeOf(world, cfg, from, to);
  return 1 / speedKmh(cfg, 'courier', grade) + courierCrossingHours(cfg, world, from, to);
}

/** The least-*time* ride between two hexes, roads and all. */
export function ridePath(world: World, cfg: CampaignConfig, from: Hex, to: Hex): Hex[] | null {
  // The heuristic counts hexes, so it must be scaled by the fastest step a rider can
  // make or it stops being a lower bound and A* returns whatever it finds first. A
  // courier on a highway crosses a hex in a tenth of an hour.
  const perHex = 1 / Math.max(...Object.values(cfg.speeds.courier));

  return astar<WorldHex>(
    world.hexes,
    from,
    to,
    () => 0,
    (_a, _b, fromCoord, toCoord) => courierStepHours(world, cfg, fromCoord, toCoord),
    (a, b) => distance(a, b) * perHex,
  );
}

/**
 * The whole ride: from the sender's formation, through any insisted-on waypoints, to
 * where the addressee's formation actually stands.
 *
 * Returns null when no legal ride exists — an addressee across an ocean, or a waypoint
 * on water. The caller raises a soft violation rather than swallowing it, because a
 * referee may legitimately want the rider sent anyway.
 */
export function planRide(
  world: World,
  cfg: CampaignConfig,
  from: Hex,
  to: Hex,
  via: readonly Hex[] = [],
): Hex[] | null {
  const legs = [from, ...via, to];
  const route: Hex[] = [from];

  for (let i = 1; i < legs.length; i++) {
    const leg = ridePath(world, cfg, legs[i - 1]!, legs[i]!);
    if (leg === null) return null;
    // The first hex of each leg is the last of the previous one.
    route.push(...leg.slice(1));
  }
  return route;
}

/** Hours a route takes from a point along it to its end. */
export function rideHours(
  world: World,
  cfg: CampaignConfig,
  route: readonly Hex[],
  fromIndex = 0,
): number {
  let total = 0;
  for (let i = Math.max(1, fromIndex + 1); i < route.length; i++) {
    total += courierStepHours(world, cfg, route[i - 1]!, route[i]!);
  }
  return total;
}

/**
 * Whether two formations are close enough to pass paper by hand.
 *
 * Any hex of one column adjacent to any hex of the other — generous, and right, because
 * a rider covers intermingled baggage in minutes and nobody would call that a despatch.
 * Measured against the whole column rather than the head, for the same reason recon is:
 * a division is eighteen kilometres of road, and its tail is as much part of it as its
 * front.
 */
export function formationsTouch(a: Unit, b: Unit): boolean {
  const mine = new Set(occupied(a).map(key));
  for (const c of occupied(b)) {
    if (mine.has(key(c))) return true;
    for (const n of neighbors(c)) {
      if (mine.has(key(n))) return true;
    }
  }
  return false;
}
