/**
 * The event log: what happened, as facts.
 *
 * A campaign is stored as an ordered log of events, and its state is the fold of that log.
 * Nothing else is persisted. That buys several things the referee actually needs — rewind
 * to any point, after-action review, and an audit of every rule that was bent and by whom
 * — and it makes the engine testable, because a campaign is a value rather than a
 * database.
 *
 * **Dice are recorded, not re-rolled.** An interception event carries the dice that
 * decided it. Replay therefore reproduces the campaign exactly, regardless of engine,
 * platform, or any later change to `rng.ts` — and the log reads as an account of what
 * happened rather than as a seed nobody can interpret. The generator decides; the log
 * remembers.
 *
 * Events are facts in the past tense and are never rejected on replay. Everything that
 * could refuse a change has already happened in `check`, before the event existed.
 */

import type { Hex } from './hex.js';
import type { Strictness, Violation } from './ruling.js';
import type { Experience, Formation, Trait, Unit, UnitKind } from './unit.js';

export const CAMPAIGN_SCHEMA_VERSION = '1.0';
export const SUPPORTED_CAMPAIGN_VERSIONS = new Set([CAMPAIGN_SCHEMA_VERSION]);

/** Who caused an event. Players are named by faction; the referee is unnamed. */
export type Actor = { readonly kind: 'referee' } | { readonly kind: 'faction'; readonly id: string };

export const REFEREE: Actor = { kind: 'referee' };
export const byFaction = (id: string): Actor => ({ kind: 'faction', id });

/** Identifies the world a campaign is played on, and detects one swapped underneath it. */
export interface WorldRef {
  readonly seed: number;
  readonly width: number;
  readonly height: number;
  readonly layout: string;
  readonly schemaVersion: string;
  /** sha256 of the world document. A regenerated world invalidates every coordinate. */
  readonly hash: string;
}

export interface Faction {
  readonly id: string;
  readonly name: string;
  /** `#rrggbb`, used for unit markers. */
  readonly color: string;
}

/**
 * What actually happened.
 *
 * This union grows as the rules land. Movement, couriers, recon and combat each add
 * their own kinds; the envelope and the reducer do not change when they do.
 */
export type EventPayload =
  | {
      readonly kind: 'campaign_created';
      readonly world: WorldRef;
      readonly seed: number;
      readonly name: string;
      readonly startHours: number;
    }
  | { readonly kind: 'faction_added'; readonly faction: Faction }
  | { readonly kind: 'unit_added'; readonly unit: Unit }
  | { readonly kind: 'unit_removed'; readonly unitId: string }
  | { readonly kind: 'clock_advanced'; readonly toHours: number }
  /** Referee: put a unit somewhere, no movement rule applying. */
  | { readonly kind: 'unit_teleported'; readonly unitId: string; readonly column: readonly Hex[] }
  /** Referee: hand a faction knowledge it did not earn. */
  | { readonly kind: 'hexes_revealed'; readonly faction: string; readonly coords: readonly Hex[] }
  /** Referee: take knowledge away, the one thing that shrinks `seen`. */
  | { readonly kind: 'hexes_concealed'; readonly faction: string; readonly coords: readonly Hex[] }
  | {
      readonly kind: 'unit_stat_set';
      readonly unitId: string;
      readonly changes: UnitStatChanges;
    };

/** The mutable stats a referee may set directly. Deliberately not every field. */
export interface UnitStatChanges {
  readonly effectives?: number;
  readonly fatigue?: number;
  readonly morale?: number;
  readonly provisions?: number;
  readonly equipment?: number;
  readonly experience?: Experience;
  readonly formation?: Formation;
  readonly traits?: readonly Trait[];
  readonly hoursMarchedToday?: number;
  readonly corps?: string | null;
}

export type EventKind = EventPayload['kind'];

/**
 * A logged event: what happened, when, at whose hand, and what rules it broke.
 *
 * `bypassed` is the audit trail. A non-empty list means the command was allowed through
 * in spite of those violations — by `force`, or by a permissive strictness — and that is
 * exactly what a referee needs to see when reviewing a contested campaign later.
 */
export interface LoggedEvent {
  readonly seq: number;
  /** Campaign clock at which this happened, in hours since the scenario epoch. */
  readonly clockHours: number;
  readonly actor: Actor;
  readonly payload: EventPayload;
  readonly forced: boolean;
  readonly strictness: Strictness;
  readonly bypassed: readonly Violation[];
}

/** Whether an event was allowed through in spite of a rule. */
export const wasOverridden = (e: LoggedEvent): boolean => e.bypassed.length > 0;

/** A unit's kind, for events and commands that name one without a whole unit. */
export type { UnitKind };
