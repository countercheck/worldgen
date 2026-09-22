/**
 * Tasks and decision points: what a formation is doing, and when it stops to ask.
 *
 * A task is the other half of the split that phase 2 rests on. A despatch is prose a
 * commander wrote; a task is a march the engine is running. The referee is the bridge
 * between them — they read the paper, decides what the addressee makes of it, and sets a
 * task. Nothing here is ever created by an arriving despatch on its own.
 *
 * ## A destination, not a path
 *
 * The referee names where a formation is to go, and the engine routes it. Three reasons,
 * and they point the same way. It is what you actually wrote in 1815: you told a corps
 * where to be, not which fields to cross. It protects a commander's own planning, where
 * a path drawn from a wrongly-believed starting position is nonsense while a destination
 * survives being wrong about the start. And it lets the route be recomputed when the
 * ground turns out not to be what the map said, which is what the deferred issued map
 * exists to make happen.
 *
 * Waypoints are the exception that proves it. A referee who says "by way of the bridge at
 * Genappe" is still naming places rather than fields, and the engine still routes between
 * them — so `via` shapes the march without ever becoming a path. A column marches through
 * a waypoint without stopping: it steers the route, it is not an objective.
 *
 * ## Stopping to ask
 *
 * A formation continues its task until it discovers something, and then stops. That is
 * the whole of "a unit with no new orders follows its last one" — the task never ended,
 * so nothing had to continue it. What ends it is a `PendingDecision`, raised against the
 * **commander** rather than the unit, because deciding is something a commander does and the
 * referee should decide from that commander's information rather than from the map.
 *
 * Traffic is the exception, and it is the exception because it is not about belief: two
 * columns cannot both be in the same hex whatever anyone knows. Those decisions carry no
 * commander and are simply the referee's.
 *
 * The scheduler's primary control is *advance until something needs a human*. This is
 * also where sub-commander personalities land later: a personality is a policy that
 * resolves some of these without asking, and it drops in exactly here.
 */

import { key, type Hex } from './hex.js';

/**
 * What a formation is doing.
 *
 * `nextHex` and `progressHours` are the running state of the march: the head is between
 * hexes, and this says which one it is walking into and how much of the walk is done.
 *
 * ## The clock runs in whole hours
 *
 * Every hour, a marching column is given an hour of movement and spends it. Infantry on a
 * road covers three hexes with it; a convoy off-road covers two thirds of one and banks
 * the rest. That banked remainder is `progressHours`, and it is what makes every row of
 * the movement table mean something on a grid where the smallest unit of time is an hour
 * — without it a convoy at two thirds of a kilometre an hour would never move at all.
 *
 * Hours rather than a fraction of the way there, because what a hex costs is not known
 * until the column is looking at it: a river crossing adds an hour to the far bank, and
 * the same hex costs a different amount to a cavalry division than to a convoy.
 */
export interface Task {
  readonly unitId: string;
  readonly destination: Hex;
  /** Waypoints the referee insisted on. The march is routed through them in order. */
  readonly via: readonly Hex[];
  readonly setAtHours: number;
  /** The despatch the referee was reading when they set it. The audit trail of an order. */
  readonly fromDespatchId: string | null;
  /** The hex the head is marching into, or null when the column has arrived. */
  readonly nextHex: Hex | null;
  /**
   * Hours of movement already spent walking into `nextHex`.
   *
   * Reset to nothing each time the head enters a hex, less whatever was left over. Zero
   * for a column that is not going anywhere.
   */
  readonly progressHours: number;
  /** Whether the destination has been reached. Kept, so the referee can see it was. */
  readonly complete: boolean;
  /**
   * How many of `via` the column has already passed. The waypoints still ahead are
   * `via.slice(viaIndex)`, and those are what the next leg is routed through.
   *
   * An index rather than a stored path, for the reason the header gives: a path goes stale
   * the moment the ground turns out not to be what the map said, while "two waypoints left"
   * survives being re-routed from wherever the column now stands.
   */
  readonly viaIndex: number;
}

/**
 * Why a formation stopped and asked.
 *
 * Config lists which of these actually halt the clock, so a referee running a large
 * campaign can let their columns march through a distant sighting and stop only for what
 * they care about.
 */
export type DecisionTrigger =
  /** An enemy came into view of a formation that could not see them before. */
  | 'enemy_contact'
  /** The route ahead cannot be marched: an unbridged river, or ground gone impassable. */
  | 'crossing_impassable'
  | 'gunfire_heard'
  | 'objective_reached'
  /** A despatch arrived. Somebody has to read the prose and decide what it means. */
  | 'despatch_arrived'
  | 'out_of_provisions'
  | 'attacked'
  /** A column's head ran into ground another column is standing on. */
  | 'column_blocked'
  /** Two heads were entering the same hex at once, and neither was the faster. */
  | 'column_contested'
  /** A patrol ran into something. Twenty troopers meeting anything is the referee's. */
  | 'patrol_contact'
  /** A player wrote to the referee directly, out of the game. `context.text` is the note. */
  | 'referee_note';

export const DECISION_TRIGGERS: readonly DecisionTrigger[] = [
  'enemy_contact',
  'crossing_impassable',
  'gunfire_heard',
  'objective_reached',
  'despatch_arrived',
  'out_of_provisions',
  'attacked',
  'column_blocked',
  'column_contested',
  'patrol_contact',
  'referee_note',
];

/**
 * A formation has stopped, and a commander has to decide what it does next.
 *
 * `context` is a plain record rather than a union: it exists to be shown to the referee
 * beside the decision, and every trigger wants to say something different. Typing it as a
 * union would mean the scheduler and the console agreeing on a schema for what is
 * essentially a caption.
 */
export interface PendingDecision {
  readonly id: string;
  /**
   * The commander the referee should decide as, or null when there is nobody to decide as.
   *
   * Traffic is the case: two columns meeting on a road is a fact about the ground rather
   * than about what anyone believes, and it needs settling even where neither formation
   * has a commander riding with it. Every decision is the referee's — a commander's view
   * carries none of them — so this names whose information to read it by, not who is
   * being asked.
   */
  readonly commanderId: string | null;
  /** The formation that ran into it. Usually, but not always, the one they ride with. */
  readonly unitId: string;
  readonly atHours: number;
  readonly trigger: DecisionTrigger;
  readonly context: Readonly<Record<string, unknown>>;
  /** Set when the referee has dealt with it. Kept, so the queue has a history. */
  readonly resolvedAtHours: number | null;
  readonly note: string | null;
  /**
   * The formation the referee ruled in favour of, where the decision was a contest.
   *
   * Only `column_contested` uses it: two heads entering one hex at the same cost is a tie
   * the rules hand to the referee, and this is their ruling. Null on every other trigger,
   * and on a contest they dealt with without naming anyone — which re-asks, because nothing
   * about the ground has changed.
   */
  readonly favouring: string | null;
}

export const isOpen = (d: PendingDecision): boolean => d.resolvedAtHours === null;

/**
 * The other formations named in a decision's context, where it is a contest.
 *
 * `context` is deliberately untyped — it is a caption for the referee — so this is the one
 * place that reads a field out of it, and it reads defensively. A decision from an older
 * build, or one raised by something that never set the field, simply has no contestants.
 */
export function contestants(d: PendingDecision): string[] {
  const other = d.context['withUnitId'];
  return typeof other === 'string' ? [other] : [];
}

/** The hex a contest is over, if the decision is one. */
export function contestedHex(d: PendingDecision): Hex | null {
  const at = d.context['at'];
  if (typeof at !== 'object' || at === null) return null;
  const { q, r } = at as { q?: unknown; r?: unknown };
  return typeof q === 'number' && typeof r === 'number' ? { q, r } : null;
}

/** Whether a formation still has somewhere to be. */
export const isRunning = (t: Task): boolean => !t.complete && t.nextHex !== null;

/**
 * How many waypoints are behind a column standing on `at`.
 *
 * Loops rather than testing one, so that two waypoints named on the same hex both clear —
 * otherwise the second would sit forever on ground the column is already standing on and
 * the march would never finish.
 *
 * The scheduler and the reducer both call this rather than each deciding for itself when a
 * waypoint counts as reached. They run at different moments on the same march, and a
 * disagreement between them would route the column through a waypoint it had already
 * passed.
 */
export function viaIndexAt(task: Task, at: Hex): number {
  const here = key(at);
  let i = task.viaIndex;
  while (i < task.via.length && key(task.via[i]!) === here) i++;
  return i;
}

/** The waypoints a column standing on `at` still has to make. */
export const viaAhead = (task: Task, at: Hex): readonly Hex[] =>
  task.via.slice(viaIndexAt(task, at));
