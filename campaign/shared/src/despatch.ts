/**
 * Despatches: the paper, the rider, and the fate of both.
 *
 * Everything a commander knows beyond their own horizon arrives here. A despatch runs from
 * one commander to another — the formation is only the address a rider must find — and it takes
 * real hours to get there, may be intercepted on the way, and is never acknowledged
 * unless the addressee writes back — which is simply another despatch, on another rider.
 *
 * There is one kind. An order is a despatch with orders written in it, a report is a
 * despatch with news in it, and the engine does not need to tell them apart because it
 * executes neither: see below.
 *
 * ## Who may write to whom
 *
 * One link of the chain of command, up or down, or anyone on their own side they can see.
 * Coordinating with a corps out of sight means writing to the common superior and waiting
 * for them to pass it on, which is slow on purpose: the chain of command costs time, and a
 * side that disperses pays it. See `mayWriteTo`.
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
 * commander where their rider went and you have told them exactly where their detached corps
 * is, and they need never read a report again: they would read their own outbox instead.
 *
 * So `route` is referee-only, absolutely, and so is anything derived from it. A delivery
 * estimate is a distance, and a distance is a position, so no despatch carries an ETA
 * either. `senderCopy` below is the only shape a sender is ever shown, and the leakage
 * suite asserts against the serialised bytes that nothing else escapes.
 *
 * The client may still show a commander *their own* guess — routed from their last report of
 * that formation, over the map they actually hold, and labelled as the guess it is. That
 * is the shared engine earning its keep: the same routing code, run over worse data.
 */

import { FOOTPRINT, occupied, type FootprintShape } from './column.js';
import { inChain } from './commander.js';
import type { CampaignConfig } from './config.js';
import { crossingAt, riverClass } from './crossing.js';
import { astar, distance, key, neighbors, type Hex } from './hex.js';
import { speedKmh } from './movement.js';
import { publicContact, reconZone, type PublicContact, type Sighting } from './recon.js';
import type { CampaignState } from './state.js';
import { gradeOf, isPassable, isRiver } from './terrain.js';
import type { Formation, Unit, UnitReport } from './unit.js';
import { hexAt, type World, type WorldHex } from './world.js';

/**
 * What is written on the paper.
 *
 * An order is text and nothing else. An automatic contact report is data and nothing
 * else. A commander writing up a sighting themselves may send both, which is why these are
 * optional fields rather than a union.
 */
export interface DespatchBody {
  readonly text?: string;
  /**
   * Sightings attached to the paper.
   *
   * Held internally as `Sighting`, which names the formation actually seen, because the
   * recipient's staff has to be able to tell whether this is the column they are already
   * watching. That name never reaches a client: `addresseeCopy` and `captorCopy` strip it,
   * and the leakage suite asserts on the serialised bytes that they did.
   */
  readonly contacts?: readonly Sighting[];
  /**
   * Where the sender's own formation stood when they sealed it.
   *
   * Attached to every despatch, whether or not anybody asked for it: a rider who has come
   * from III Corps knows where III Corps was when they left, and that is most of what a
   * despatch was actually for. It is why an order arriving also refreshes the recipient's
   * picture of the commander who sent it — and why silence from a corps is not merely a missing
   * order but a stale map.
   */
  readonly unitReport?: UnitReport;
}

/**
 * What became of a rider.
 *
 * Known to the referee always, to the addressee once it has arrived, to a captor when they
 * take it — and to the sender never. That last is not an oversight to be tidied up
 * later: it is the mechanic.
 */
export type Fate =
  | { readonly kind: 'in_transit' }
  | { readonly kind: 'delivered'; readonly atHours: number }
  /** The rider was stopped and the paper lost with them. */
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
  /** Commander id. Despatches run commander to commander; the formation is only the address. */
  readonly from: string;
  readonly to: string;
  readonly faction: string;
  /** The hour it was written, which is the hour its contents describe. */
  readonly sentAtHours: number;
  readonly body: DespatchBody;
  /** Waypoints the sender insisted on — around a wood they think holds pickets. */
  readonly via: readonly Hex[];
  /** A report being passed on. Two lags stack, which is very much the period. */
  readonly forwardedFrom: string | null;
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
 * Everything they wrote, and not one thing more. No route, because the route is where the
 * addressee is; no fate, because learning that their rider was taken would tell them their
 * order never arrived, which no commander in 1815 could know without being told.
 *
 * Built by construction rather than by deletion. A `delete d.route` would leak the day
 * somebody adds a field and forgets the line.
 */
export interface SentDespatch {
  readonly id: string;
  readonly to: string;
  readonly sentAtHours: number;
  readonly body: PublicBody;
  /** Their own waypoints, which they chose and therefore already knows. */
  readonly via: readonly Hex[];
  /** Whether it was handed over on the spot. They watched that happen. */
  readonly handed: boolean;
}

/** A despatch as its **addressee** sees it: only once it is actually in their hand. */
export interface ReceivedDespatch {
  readonly id: string;
  readonly from: string;
  /** The hour it describes. Shown first, because it is what they now know about. */
  readonly sentAtHours: number;
  /** The hour it reached them. The gap between the two is the fog. */
  readonly receivedAtHours: number;
  readonly body: PublicBody;
  readonly forwardedFrom: string | null;
}

/** A despatch as a **captor** sees it: the body, and the fact that they took it. */
export interface CapturedDespatch {
  readonly id: string;
  readonly from: string;
  readonly to: string;
  readonly faction: string;
  readonly sentAtHours: number;
  readonly capturedAtHours: number;
  readonly body: PublicBody;
}

/**
 * A body as it goes on the wire.
 *
 * The prose unchanged — it is what somebody wrote — and the sightings stripped of the
 * formation they name. A captured despatch is intelligence about *where* the enemy
 * believes things are, not a key to their order of battle.
 */
export interface PublicBody {
  readonly text?: string;
  readonly contacts?: readonly PublicContact[];
  readonly unitReport?: UnitReport | CapturedReport;
}

/**
 * The sender's own return, as the commander who took the paper off the rider reads it.
 *
 * Where the formation stood and what hour it is speaking about — the thing a captured
 * despatch was actually worth — and not one field of its internal state. A letter signed
 * by a marshal names their corps, so the name and the echelon stay; its returns do not
 * travel with it, and neither does the engine's id for it, which is the one field that
 * would let a captor correlate every later sighting for free. See `recon.ts` on why that
 * correlation is meant to cost a patrol.
 */
export interface CapturedReport {
  readonly name: string;
  readonly faction: string;
  readonly kind: UnitReport['kind'];
  readonly echelon: UnitReport['echelon'];
  readonly atHours: number;
  readonly head: Hex;
  readonly corps: string | null;
}

export const capturedReport = (r: UnitReport): CapturedReport => ({
  name: r.name,
  faction: r.faction,
  kind: r.kind,
  echelon: r.echelon,
  atHours: r.atHours,
  head: r.head,
  corps: r.corps,
});

/**
 * `report` says whose side is reading it: their own commander's return travels intact, a
 * captured one is cut down to `capturedReport`.
 */
const publicBody = (b: DespatchBody, report: 'own' | 'captured' = 'own'): PublicBody => ({
  ...(b.text === undefined ? {} : { text: b.text }),
  ...(b.contacts === undefined
    ? {}
    : // No id has been minted for these — they are somebody else's sightings, not filed
      // knowledge — so they are labelled by position, which is all the paper carries.
      {
        contacts: b.contacts.map((c, i) =>
          publicContact({ ...c, id: `s${i + 1}`, inSight: false }),
        ),
      }),
  ...(b.unitReport === undefined
    ? {}
    : { unitReport: report === 'own' ? b.unitReport : capturedReport(b.unitReport) }),
});

export const senderCopy = (d: Despatch): SentDespatch => ({
  id: d.id,
  to: d.to,
  sentAtHours: d.sentAtHours,
  body: publicBody(d.body),
  via: d.via,
  handed: d.handed,
});

export const addresseeCopy = (d: Despatch): ReceivedDespatch => ({
  id: d.id,
  from: d.from,
  sentAtHours: d.sentAtHours,
  receivedAtHours: deliveredAt(d) ?? d.sentAtHours,
  body: publicBody(d.body),
  forwardedFrom: d.forwardedFrom,
});

export const captorCopy = (d: Despatch): CapturedDespatch => ({
  id: d.id,
  from: d.from,
  to: d.to,
  faction: d.faction,
  sentAtHours: d.sentAtHours,
  capturedAtHours: d.fate.kind === 'captured' ? d.fate.atHours : d.sentAtHours,
  body: publicBody(d.body, 'captured'),
});

/**
 * A despatch as it was stored before there was only one kind.
 *
 * Logs written by older builds carry `kind` ('order', 'report', 'acknowledgement') and
 * `inReplyTo` on every despatch. The log is the campaign and cannot be rewritten, so the
 * reducer passes each one through here on the way in: the fields are dropped and the
 * despatch is an ordinary despatch, which is what every one of them now is.
 */
export function normaliseDespatch(d: Despatch): Despatch {
  const { kind: _kind, inReplyTo: _inReplyTo, ...rest } = d as Despatch & {
    kind?: unknown;
    inReplyTo?: unknown;
  };
  return rest;
}

// ---- who may write to whom ---------------------------------------------

/**
 * Whether one commander can see another's formation from where they stand.
 *
 * Any hex of the other column inside the sender's recon zone: the same test that decides
 * whether they can see an enemy, pointed at a friend. One-way, like sight — a scouting
 * division sees further than the line division beside it, so the scouts may be able to
 * write to them before they can write back.
 */
export function inSight(
  state: CampaignState,
  world: World,
  cfg: CampaignConfig,
  id: string,
  targetId: string,
): boolean {
  const from = state.commanders.get(id);
  const to = state.commanders.get(targetId);
  if (from === undefined || to === undefined) return false;
  const fromUnit = state.units.get(from.unitId);
  const toUnit = state.units.get(to.unitId);
  if (fromUnit === undefined || toUnit === undefined) return false;

  const zone = reconZone(world, cfg, fromUnit);
  return occupied(toUnit, 'road', cfg.footprint).some((c) => zone.has(key(c)));
}

/**
 * Whether `id` may send a despatch to `targetId`.
 *
 * Their direct superior, their direct subordinates, or anyone on their own side whose
 * column they can see. Never the enemy, and never themselves. Hard, in `check`: this is
 * who a rider can be sent to, not a rule a referee bends.
 */
export function mayWriteTo(
  state: CampaignState,
  world: World,
  cfg: CampaignConfig,
  id: string,
  targetId: string,
): boolean {
  if (inChain(state, id, targetId)) return true;
  const from = state.commanders.get(id);
  const to = state.commanders.get(targetId);
  if (from === undefined || to === undefined) return false;
  if (id === targetId || from.faction !== to.faction) return false;
  return inSight(state, world, cfg, id, targetId);
}

/** Everyone `id` may write to right now, in id order. */
export const addresseesOf = (
  state: CampaignState,
  world: World,
  cfg: CampaignConfig,
  id: string,
): string[] =>
  [...state.commanders.keys()].filter((t) => mayWriteTo(state, world, cfg, id, t)).sort();

// ---- riding ------------------------------------------------------------

/**
 * What a lone rider pays to cross water, where a division would be stopped.
 *
 * A courier is one rider on one horse: they ford where a column cannot, finds the boat, or
 * swims the animal. So a major river costs them an hour rather than being impassable —
 * the rules do not say so explicitly, but a courier system in which one river ends
 * communication altogether is not the period, and the whole point of the rider is that
 * they get through or is caught trying.
 */
function courierCrossingHours(cfg: CampaignConfig, world: World, from: Hex, to: Hex): number {
  const target = hexAt(world, to);
  if (target === undefined || !isRiver(target)) return 0;

  const origin = hexAt(world, from);
  if (origin !== undefined && isRiver(origin)) return 0;

  if (crossingAt(world, from, to) === 'bridge') return 0;
  return riverClass(target, world) === 'major' ? cfg.courierMajorCrossingHours : cfg.fordHours;
}

/** Hours for a rider to enter one hex from an adjacent one. `Infinity` if they cannot. */
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
export function formationsTouch(
  a: Unit,
  b: Unit,
  shapes: Readonly<Record<Formation, FootprintShape>> = FOOTPRINT,
): boolean {
  const mine = new Set(occupied(a, 'road', shapes).map(key));
  for (const c of occupied(b, 'road', shapes)) {
    if (mine.has(key(c))) return true;
    for (const n of neighbors(c)) {
      if (mine.has(key(n))) return true;
    }
  }
  return false;
}
