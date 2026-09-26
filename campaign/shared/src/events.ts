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

import type { Commander } from './commander.js';
import type { Despatch } from './despatch.js';
import type { Grade } from './config.js';
import type { Hex } from './hex.js';
import type { Strictness, Violation } from './ruling.js';
import type { Contact } from './recon.js';
import type { StandingOrders } from './standing.js';
import type { PendingDecision, Task } from './task.js';
import type {
  Echelon,
  Experience,
  Formation,
  RoadHour,
  Trait,
  Unit,
  UnitKind,
  UnitReport,
} from './unit.js';

export const CAMPAIGN_SCHEMA_VERSION = '1.0';
export const SUPPORTED_CAMPAIGN_VERSIONS = new Set([CAMPAIGN_SCHEMA_VERSION]);

/**
 * Who caused an event.
 *
 * A commander, not a faction. A side does not decide anything; a commander does, and when a
 * campaign is reviewed afterwards the question is always which of them gave the order.
 * The referee is unnamed because they are not in the war.
 */
export type Actor =
  | { readonly kind: 'referee' }
  | { readonly kind: 'commander'; readonly id: string };

export const REFEREE: Actor = { kind: 'referee' };
export const byCommander = (id: string): Actor => ({ kind: 'commander', id });

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
  | { readonly kind: 'commander_added'; readonly commander: Commander }
  | { readonly kind: 'commander_removed'; readonly commanderId: string }
  /** A commander moves to another formation, or is given a new superior. */
  | {
      readonly kind: 'commander_reassigned';
      readonly commanderId: string;
      readonly unitId?: string;
      readonly superiorId?: string | null;
    }
  | { readonly kind: 'unit_added'; readonly unit: Unit }
  | { readonly kind: 'unit_removed'; readonly unitId: string }
  /**
   * A formation sent out a patrol.
   *
   * The patrol is a unit like any other, carrying `parentUnitId`. `costPaperStrength` is what
   * detaching it took off the parent's rolls — nothing for the first few, and permanent
   * for the ones after that, which is the rules' way of saying a division can only spare
   * so many good horsemen before it starts to feel it.
   */
  | {
      readonly kind: 'patrol_detached';
      readonly patrol: Unit;
      readonly parentUnitId: string;
      readonly costPaperStrength: number;
    }
  | { readonly kind: 'clock_advanced'; readonly toHours: number }
  /** Referee: put a unit somewhere, no movement rule applying. */
  | { readonly kind: 'unit_teleported'; readonly unitId: string; readonly column: readonly Hex[] }
  /** Referee: a patrol answers to another formation, and reports what it sees to its commander. */
  | { readonly kind: 'patrol_reassigned'; readonly unitId: string; readonly parentUnitId: string }
  /** Ground a commander's formations have surveyed, or a referee has simply given them. */
  | {
      readonly kind: 'hexes_surveyed';
      readonly commanderId: string;
      readonly coords: readonly Hex[];
    }
  /**
   * Referee: this ground is being fought over.
   *
   * Not a battle object, deliberately. A campaign map resolves to a kilometre, and a
   * division's frontage is a kilometre, so everything that makes a battle a battle
   * happens below this grid. What the campaign needs to know is only *where* it is, so
   * that the rules which assume open country stop applying there — who is fighting whom,
   * and how it goes, belongs to whatever resolves it.
   */
  | { readonly kind: 'battle_declared'; readonly coords: readonly Hex[] }
  /** Referee: the fighting here is over. The ground goes back to being ground. */
  | { readonly kind: 'battle_ended'; readonly coords: readonly Hex[] }
  /** Referee: take knowledge away, the one thing that shrinks what a commander has surveyed. */
  | {
      readonly kind: 'hexes_forgotten';
      readonly commanderId: string;
      readonly coords: readonly Hex[];
    }
  /**
   * Word of where a formation was reached a commander.
   *
   * By rider, by a column marching into sight of their own, or because the formation is the
   * one they are standing next to. Held rather than recomputed: the hour on it is the whole
   * of the fog, and a snapshot taken when a client asks would always read "now".
   */
  | {
      readonly kind: 'report_filed';
      readonly commanderId: string;
      readonly report: UnitReport;
    }
  /**
   * A commander has been told where an enemy was.
   *
   * By their own column seeing it, or by a report reaching them. The contact carries the
   * label their own staff gave it, minted when the trail was cold and reused while it is
   * warm — see `knowledge.ts`, which decides which of those happened.
   */
  | {
      readonly kind: 'contact_filed';
      readonly commanderId: string;
      readonly contact: Contact;
    }
  /**
   * A column their pickets were watching has gone out of view.
   *
   * The contact stays on their map at the hex they last saw it — losing sight of something
   * does not unsee it. What changes is that the next sighting will be a *new* contact
   * rather than a continuation of this one.
   */
  | {
      readonly kind: 'contact_lost';
      readonly commanderId: string;
      readonly contactId: string;
      readonly atHours: number;
    }
  | {
      readonly kind: 'unit_stat_set';
      readonly unitId: string;
      readonly changes: UnitStatSet;
    }
  // ---- despatches -------------------------------------------------------
  /** A commander wrote something and put a rider on the road with it. */
  | { readonly kind: 'despatch_sent'; readonly despatch: Despatch }
  /**
   * Where a rider had got to when the clock stopped.
   *
   * One of these per rider per advance rather than one per hex: a courier covers ten
   * hexes an hour, and a day's advance would otherwise write two hundred events saying
   * nothing but "still riding". The route travels with it because a rider re-routes when
   * their addressee moves, and the log has to carry the path they actually took.
   */
  | {
      readonly kind: 'despatch_progressed';
      readonly despatchId: string;
      readonly progress: number;
      readonly route: readonly Hex[];
    }
  | {
      readonly kind: 'despatch_delivered';
      readonly despatchId: string;
      readonly atHours: number;
    }
  /**
   * A rider was stopped. `outcome` says whether the paper was merely lost or was read.
   *
   * The dice are in the event, so a referee adjudicating a disputed interception can show
   * them, and so replay reproduces the campaign without re-rolling anything.
   */
  | {
      readonly kind: 'despatch_stopped';
      readonly despatchId: string;
      /** The faction whose column the rider tried to pass. */
      readonly by: string;
      readonly atHours: number;
      readonly at: Hex;
      readonly dice: readonly number[];
      readonly outcome: 'lost' | 'captured';
    }
  // ---- tasks and the clock ---------------------------------------------
  /** The referee, having read a despatch, set a formation marching. */
  | { readonly kind: 'task_set'; readonly task: Task }
  | { readonly kind: 'task_cleared'; readonly unitId: string }
  /** The head of a column entered a hex. The tail follows along `column`. */
  | {
      readonly kind: 'unit_marched';
      readonly unitId: string;
      readonly to: Hex;
      readonly atHours: number;
      readonly grade: Grade;
      readonly stepHours: number;
      /** The next hex of the march. Null on arrival. */
      readonly nextHex: Hex | null;
      /** Hours of this hour's movement left over, banked towards `nextHex`. */
      readonly progressHours: number;
    }
  /**
   * An hour spent on the road that did not finish a hex.
   *
   * A convoy at two thirds of a kilometre an hour takes an hour and a half to cross one,
   * and the half hour has to go somewhere or it never crosses at all. This is where.
   */
  | {
      readonly kind: 'march_progressed';
      readonly unitId: string;
      readonly atHours: number;
      readonly progressHours: number;
      readonly grade: Grade;
      /** Hours of the hour actually spent walking. Charged against the day like a march. */
      readonly spentHours: number;
    }
  /**
   * A column's head could not enter the hex it was marching into, because another column
   * was in it or entering it. It stands where it is and tries again.
   *
   * Its own event rather than a `unit_marched` that went nowhere: nothing moved, no hours
   * were spent, and the log should read as an account of a road that was blocked rather
   * than of a march with a gap in it.
   */
  | {
      readonly kind: 'march_blocked';
      readonly unitId: string;
      /** The column in the way. */
      readonly byUnitId: string;
      readonly at: Hex;
      readonly atHours: number;
      /** Whether both heads were entering it at once, or one was simply standing there. */
      readonly contested: boolean;
      /**
       * Hours spent standing in the road, waiting.
       *
       * Charged against the day only if the column had already broken camp — a formation
       * blocked before it took a single step has not marched, while one halted three hexes
       * into its day is standing formed up on a road in column of march, which is work.
       * The reducer decides which of those it was; this is only the duration.
       */
      readonly waitedHours: number;
    }
  /**
   * A formation started changing what it is: making camp, forming for battle, breaking
   * camp to march. It is still the old formation until `formation_changed`.
   */
  | {
      readonly kind: 'formation_change_began';
      readonly unitId: string;
      readonly from: Formation;
      readonly to: Formation;
      readonly atHours: number;
      readonly completesAtHours: number;
      /** Why, for the referee reading the log: 'day_spent' | 'ordered' | 'break_camp'. */
      readonly reason: string;
    }
  | {
      readonly kind: 'formation_changed';
      readonly unitId: string;
      readonly to: Formation;
      readonly atHours: number;
    }
  /**
   * What a stretch of marching cost the troops.
   *
   * Its own event rather than a field on `unit_marched`, because the tail goes on paying
   * after the head has stopped: a column that halts at dusk is still marching in the dark
   * at the rear, and that arrives as one of these with no march beside it.
   */
  | {
      readonly kind: 'fatigue_accrued';
      readonly unitId: string;
      readonly atHours: number;
      readonly fatigue: number;
      /** Split out so the log says what it was for rather than only how much. */
      readonly fromMarching: number;
      readonly fromNight: number;
    }
  | { readonly kind: 'task_completed'; readonly unitId: string; readonly atHours: number }
  /** Midnight. Marked for the log and for provisions; the march cap does not read it. */
  | { readonly kind: 'day_rolled'; readonly toHours: number }
  /**
   * A column has been off the road for `minRestHoursPerDay`, and the fatigue table reads
   * from its first hour again.
   */
  | { readonly kind: 'unit_rested'; readonly unitId: string; readonly atHours: number }
  /** Referee: the sun now rises and sets at these hours of the day. */
  | { readonly kind: 'daylight_set'; readonly sunriseHour: number; readonly sunsetHour: number }
  /**
   * When a formation's head may be on the road. Null lifts every limit.
   *
   * Set by the commander who rides with it, or by the referee having read a despatch.
   */
  | {
      readonly kind: 'standing_orders_set';
      readonly unitId: string;
      readonly orders: StandingOrders | null;
    }
  // ---- decisions --------------------------------------------------------
  | { readonly kind: 'decision_raised'; readonly decision: PendingDecision }
  | {
      readonly kind: 'decision_resolved';
      readonly decisionId: string;
      readonly atHours: number;
      readonly note: string | null;
      /** The formation the referee gave the hex to, on a contest. Null on everything else. */
      readonly favouring: string | null;
    };

/**
 * The values a referee may set by hand: every stat on a unit.
 *
 * Not the unit's identity or structure — its id, its side, its patrols' parentage — and not
 * where it stands, which `teleport_unit` moves. A formation set here is set outright, with
 * any change under way dropped: the referee is saying what it is, not ordering it to become so.
 */
export interface UnitStatChanges {
  readonly name?: string;
  readonly kind?: UnitKind;
  readonly paperStrength?: number;
  readonly fatigue?: number;
  readonly morale?: number;
  readonly provisions?: number;
  readonly maxProvisions?: number;
  readonly equipment?: number;
  readonly maxEquipment?: number;
  readonly guns?: number;
  readonly experience?: Experience;
  readonly marchSpeedKmh?: number;
  readonly spacingM?: number;
  readonly spacingMultiplier?: number;
  readonly formation?: Formation;
  /**
   * A change of formation under way, set outright: what it is becoming and the campaign hour
   * it finishes, or null for none. Set alongside `formation` or without it; a `formation`
   * set without it drops whatever change was under way.
   */
  readonly formationChange?: {
    readonly to: Formation;
    readonly completesAtHours: number;
  } | null;
  readonly traits?: readonly Trait[];
  /** Hours on the road since the column last rested: what the fatigue table reads. */
  readonly hoursMarchedToday?: number;
  /**
   * Hours on the road in the last twenty-four, as one number: what the cap reads.
   *
   * Laid down as whole hours running back from the present, which is as much as a referee
   * saying "they have marched twelve hours today" means.
   */
  readonly roadHoursLast24?: number;
  readonly corps?: string | null;
  readonly echelon?: Echelon;
}

/** A referee's edit as the log keeps it: the hours on the road by the hour, as the unit holds them. */
export type UnitStatSet = Omit<UnitStatChanges, 'roadHoursLast24'> & {
  readonly roadHours?: readonly RoadHour[];
};

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
