/**
 * Tasks and decision points: what a formation is doing, and when it stops to ask.
 *
 * A task is the other half of the split that phase 2 rests on. A despatch is prose a
 * commander wrote; a task is a march the engine is running. The referee is the bridge
 * between them — he reads the paper, decides what the addressee makes of it, and sets a
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
 * ## Stopping to ask
 *
 * A formation continues its task until it discovers something, and then stops. That is
 * the whole of "a unit with no new orders follows its last one" — the task never ended,
 * so nothing had to continue it. What ends it is a `PendingDecision`, raised against the
 * **commander** rather than the unit, because deciding is something a man does and the
 * referee should decide from that man's information rather than from the map.
 *
 * The scheduler's primary control is *advance until something needs a human*. This is
 * also where sub-commander personalities land later: a personality is a policy that
 * resolves some of these without asking, and it drops in exactly here.
 */

import type { Hex } from './hex.js';

/**
 * What a formation is doing.
 *
 * `nextHex` and `arrivesAtHours` are the running state of the march: the head is between
 * hexes, and this says which one it is entering and when it gets there. Storing the
 * arrival hour rather than an accumulated fraction means the clock can be advanced by any
 * amount, in one jump or in twenty, and produce the same events either way.
 */
export interface Task {
  readonly unitId: string;
  readonly destination: Hex;
  /** Waypoints the referee insisted on. The march is routed through them in order. */
  readonly via: readonly Hex[];
  readonly setAtHours: number;
  /** The despatch the referee was reading when he set it. The audit trail of an order. */
  readonly fromDespatchId: string | null;
  /** The hex the head is marching into, or null when the column has arrived. */
  readonly nextHex: Hex | null;
  /** When the head reaches `nextHex`. Null when there is nowhere left to go. */
  readonly arrivesAtHours: number | null;
  /** Whether the destination has been reached. Kept, so the referee can see it was. */
  readonly complete: boolean;
}

/**
 * Why a formation stopped and asked.
 *
 * Config lists which of these actually halt the clock, so a referee running a large
 * campaign can let his columns march through a distant sighting and stop only for what
 * he cares about.
 */
export type DecisionTrigger =
  /** An enemy came into view of a formation that could not see him before. */
  | 'enemy_contact'
  /** The route ahead cannot be marched: an unbridged river, or ground gone impassable. */
  | 'crossing_impassable'
  | 'gunfire_heard'
  | 'objective_reached'
  /** A despatch arrived. Somebody has to read the prose and decide what it means. */
  | 'despatch_arrived'
  | 'out_of_provisions'
  | 'attacked';

export const DECISION_TRIGGERS: readonly DecisionTrigger[] = [
  'enemy_contact',
  'crossing_impassable',
  'gunfire_heard',
  'objective_reached',
  'despatch_arrived',
  'out_of_provisions',
  'attacked',
];

/**
 * A formation has stopped, and a man has to decide what it does next.
 *
 * `context` is a plain record rather than a union: it exists to be shown to the referee
 * beside the decision, and every trigger wants to say something different. Typing it as a
 * union would mean the scheduler and the console agreeing on a schema for what is
 * essentially a caption.
 */
export interface PendingDecision {
  readonly id: string;
  readonly commanderId: string;
  /** The formation that ran into it. Usually, but not always, the one he rides with. */
  readonly unitId: string;
  readonly atHours: number;
  readonly trigger: DecisionTrigger;
  readonly context: Readonly<Record<string, unknown>>;
  /** Set when the referee has dealt with it. Kept, so the queue has a history. */
  readonly resolvedAtHours: number | null;
  readonly note: string | null;
}

export const isOpen = (d: PendingDecision): boolean => d.resolvedAtHours === null;

/** Whether a formation still has somewhere to be. */
export const isRunning = (t: Task): boolean => !t.complete && t.nextHex !== null;
