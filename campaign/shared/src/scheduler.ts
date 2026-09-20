/**
 * Running the clock.
 *
 * The referee's primary control is not "advance two hours" but **advance until something
 * needs a human**. Columns march, riders ride, and the clock stops the moment a formation
 * discovers something its commander would have to decide about. That is what makes a
 * refereed game playable at this scale: the engine does the bookkeeping between the
 * interesting moments and hands back at each of them.
 *
 * ## It produces events, and nothing else
 *
 * `advance` is pure. It folds its own payloads into a private copy of the state as it
 * goes — so a rider that moves this tick is where it should be for the next one — and
 * returns the list. The engine stamps and logs them, and `reduce` folds them again for
 * real. Nothing here writes to the state it was given, and the same call with the same
 * dice produces the same list.
 *
 * ## Hours, and why they are the clock
 *
 * Time runs in whole hours and in nothing finer. Every cost in the rules is a whole number
 * of them, a referee adjudicates by the hour, and a column that arrived at twenty past
 * would be a precision the rest of the design cannot honour. A column is handed an hour of
 * movement and spends it — several hexes on a road, part of one in a bog, with the
 * remainder banked on its task as `progressHours` so that every row of the movement table
 * still means something.
 *
 * The campaign clock only moves when something happens: a quiet hour emits nothing at all.
 * Events are stamped at the hour they occurred rather than at the hour the referee
 * clicked, and a twelve-hour advance through empty country costs one event rather than
 * twelve.
 *
 * ## What a rider does when his man has moved
 *
 * The route is planned to where the addressee stood when the rider left. When he gets
 * there and finds the corps gone, he re-plans from where he is and rides on — which is
 * what a real despatch rider did, and why "riders always find their man" is a rule rather
 * than an approximation. The cost is his time and the extra ground he has to cross, both
 * of which are exactly the things that are supposed to hurt.
 */

import { catchupHours, occupied } from './column.js';
import { directSubordinates, superiors } from './commander.js';
import type { CampaignConfig, Grade } from './config.js';
import {
  courierStepHours,
  formationsTouch,
  planRide,
  type Despatch,
  type DespatchBody,
  type DespatchKind,
} from './despatch.js';
import { REFEREE, type EventPayload, type LoggedEvent } from './events.js';
import { key, type Hex } from './hex.js';
import { fileSightings } from './knowledge.js';
import {
  hoursToEnter,
  marchHoursLeftToday,
  planMarch,
  unitSpeedKmh,
} from './movement.js';
import { marchFatigueBetween, nightFatigue } from './fatigue.js';
import { detectionDice, spottedBy, type Sighting } from './recon.js';
import { ones, type Rng } from './rng.js';
import { reduce, type CampaignState } from './state.js';
import {
  contestedHex,
  isOpen,
  viaAhead,
  type DecisionTrigger,
  type PendingDecision,
  type Task,
} from './task.js';
import { gradeOf } from './terrain.js';
import { isPatrol, reportOf, type Formation, type Unit } from './unit.js';
import type { World } from './world.js';

export interface AdvanceOptions {
  readonly hours: number;
  /** Stop at the first discovery a referee has said he wants to see. */
  readonly untilDecision?: boolean;
}

export interface AdvanceResult {
  readonly payloads: readonly EventPayload[];
  /** The hour the clock actually reached, which is earlier when something halted it. */
  readonly toHours: number;
  readonly halted: PendingDecision | null;
}

/** Wrap a payload so it can be folded during simulation. Never logged; the engine relogs. */
const envelope = (s: CampaignState, payload: EventPayload): LoggedEvent => ({
  seq: s.nextSeq,
  clockHours: s.clockHours,
  actor: REFEREE,
  payload,
  forced: false,
  strictness: 'strict',
  bypassed: [],
});

const HOURS_PER_DAY = 24;

/**
 * The grain of the clock, in hours.
 *
 * One hour, and not a dial. This game is played in hours: every cost in the rules is a
 * whole number of them, a referee adjudicates by the hour, and a formation that arrived at
 * twenty past would be a precision the rest of the design cannot honour. Movement finer
 * than a hex is banked as progress rather than modelled as a moment.
 */
const HOURS_PER_STEP = 1;

/** Progress a rider has made, held through one advance and written back once at the end. */
interface Rider {
  /** Whole part is the last hex passed; the fraction is how far into the next one. */
  progress: number;
  route: readonly Hex[];
  moved: boolean;
}

/** What a commander is asking to have carried. */
export interface SendSpec {
  readonly from: string;
  readonly to: string;
  readonly kind: DespatchKind;
  readonly body: DespatchBody;
  readonly via?: readonly Hex[];
  readonly inReplyTo?: string;
  readonly forwardedFrom?: string;
}

/**
 * One simulation, held open so that more than one command can drive it.
 *
 * Sending a despatch and advancing the clock are different commands but the same
 * machinery: a handed-over despatch is delivered, an arriving order is cascaded, and each
 * of those raises the same decisions with the same ids however it was triggered. Writing
 * that twice — once in `decide` for a send, once here for an arrival — is exactly how the
 * two paths drift, and it would drift in the direction of a commander seeing something he
 * should not.
 */
function simulate(state: CampaignState, world: World, cfg: CampaignConfig, rng: Rng) {
  const halts = new Set<DecisionTrigger>(cfg.haltTriggers);

  let s = state;
  const payloads: EventPayload[] = [];
  let halted: PendingDecision | null = null;
  let now = state.clockHours;

  /** Fold a payload in as it is produced, so later ticks see its effect. */
  const emit = (p: EventPayload): void => {
    payloads.push(p);
    s = reduce(s, envelope(s, p));
  };

  /**
   * Move the clock to the hour something is about to happen at.
   *
   * Called immediately before emitting, never on a schedule, which is what keeps a quiet
   * tick free and an event stamped at its own hour rather than at the end of the advance.
   */
  const clockTo = (atHours: number): void => {
    if (s.clockHours < atHours) emit({ kind: 'clock_advanced', toHours: atHours });
  };

  // Ids come from the sequence number the payload's event will take, which is exactly
  // `nextSeq` at the moment it is emitted. Deterministic, unique, and it makes an id a
  // pointer into the log rather than a random string nobody can trace.
  const nextId = (prefix: string): string => `${prefix}${s.nextSeq}`;

  const riding = new Map<string, Rider>();
  for (const d of s.despatches.values()) {
    if (d.fate.kind === 'in_transit') {
      riding.set(d.id, { progress: d.progress, route: d.route, moved: false });
    }
  }

  // What every formation could already see when the clock started. A contact is only a
  // discovery if it was not one of these — otherwise an advance would halt on the enemy
  // it was already looking at, every single time.
  const alreadySeen = new Set<string>();
  for (const unit of s.units.values()) {
    for (const id of spottedBy(s, world, cfg, unit).keys()) {
      alreadySeen.add(`${unit.id}->${id}`);
    }
  }

  const raise = (
    trigger: DecisionTrigger,
    commanderId: string | null,
    unitId: string,
    atHours: number,
    context: Record<string, unknown>,
  ): void => {
    clockTo(atHours);
    const decision: PendingDecision = {
      id: nextId('k'),
      commanderId,
      unitId,
      atHours,
      trigger,
      context,
      resolvedAtHours: null,
      note: null,
      favouring: null,
    };
    emit({ kind: 'decision_raised', decision });
    if (halted === null && halts.has(trigger)) halted = decision;
  };

  /**
   * Put a despatch in a commander's hand, with everything that follows from it.
   *
   * An arriving order is work for a referee — somebody has to read the prose and decide
   * what the addressee makes of it — so it raises a decision. And a commander the referee
   * is running passes it straight down as fresh despatches with fresh riders, so the
   * copies can still be intercepted individually even though the decision cost nothing.
   */
  const deliver = (d: Despatch, atHours: number): void => {
    clockTo(atHours);
    emit({ kind: 'despatch_delivered', despatchId: d.id, atHours });
    riding.delete(d.id);

    // The rider came from somewhere and knows where that was. Every despatch refreshes
    // the recipient's picture of the man who sent it — dated when it was written, not
    // when it arrived, which is exactly the lag the design is about.
    if (d.body.unitReport !== undefined) {
      emit({ kind: 'report_filed', commanderId: d.to, report: d.body.unitReport });
    }

    // Sightings attached to the paper, filed under the *recipient's* own labels. Two
    // commanders who both hear about the same column hold two contacts with two different
    // numbers, and nothing in either man's payload says they are the same thing — which
    // is the correlation these rules make you buy with a patrol.
    if (d.body.contacts !== undefined && d.body.contacts.length > 0) {
      const filed = fileSightings(s, cfg, d.to, d.body.contacts, d.sentAtHours, 'reported');
      for (const payload of filed) emit(payload);
    }

    const to = s.commanders.get(d.to);
    if (to === undefined) return;

    raise('despatch_arrived', d.to, to.unitId, atHours, {
      despatchId: d.id,
      kind: d.kind,
      from: d.from,
      sentAtHours: d.sentAtHours,
    });

    if (d.kind !== 'order' || !to.autoCascade) return;
    // The text passes down verbatim. A referee who wants his divisions doing different
    // things writes to them himself — cascading is a convenience, not a judgement.
    for (const sub of directSubordinates(s, d.to)) {
      send({ from: d.to, to: sub.id, kind: 'order', body: d.body, inReplyTo: d.id }, atHours);
    }
  };

  /**
   * Write a despatch and put a rider on the road with it.
   *
   * Two formations whose columns touch hand paper over instead: no rider, no time, no
   * interception. That is the mechanical reward for concentration, and the reason a corps
   * marching together can be commanded while one strung across a province cannot.
   */
  const send = (spec: SendSpec, atHours: number): void => {
    const sender = s.commanders.get(spec.from);
    const addressee = s.commanders.get(spec.to);
    if (sender === undefined || addressee === undefined) return;

    const fromUnit = s.units.get(sender.unitId);
    const toUnit = s.units.get(addressee.unitId);
    const origin = fromUnit?.column[0];
    const destination = toUnit?.column[0];
    if (fromUnit === undefined || toUnit === undefined) return;
    if (origin === undefined || destination === undefined) return;

    const via = spec.via ?? [];
    // Waypoints are an instruction to the rider, so they are obeyed even when they are
    // slower — that is the whole point of insisting on them. Only touching formations
    // skip the ride, and then there is no route to insist on.
    const handed = via.length === 0 && formationsTouch(fromUnit, toUnit, cfg.footprint);
    // No legal ride — an addressee across water, or a waypoint the rider cannot reach.
    // `check` has already said so as a soft violation, so arriving here means a referee
    // sent him anyway. He sets out and is still out there: `rideTick` re-plans from where
    // he stands on every tick, so he finds his man if the ground ever allows it and never
    // if it does not. Standing him on the origin hex and calling that a delivery would
    // put the paper in the addressee's hand at the hour it was written, across an ocean.
    const route = handed ? [origin] : (planRide(world, cfg, origin, destination, via) ?? [origin]);

    clockTo(atHours);
    const despatch: Despatch = {
      id: nextId('d'),
      kind: spec.kind,
      from: spec.from,
      to: spec.to,
      faction: sender.faction,
      sentAtHours: atHours,
      // Where he stood when he sealed it, attached whether or not he thought to say so —
      // and written last, so it is the ground truth about the sender rather than whatever
      // a client put in the field. A body that could override it is a forged report, and
      // a forged report is believed: it is filed as knowledge the moment it arrives. It
      // also makes a cascaded order carry the cascading commander's position rather than
      // the original sender's, which is the one a rider coming from him would know.
      body: { ...spec.body, unitReport: reportOf(fromUnit, atHours) },
      via,
      forwardedFrom: spec.forwardedFrom ?? null,
      inReplyTo: spec.inReplyTo ?? null,
      route,
      progress: 0,
      fate: { kind: 'in_transit' },
      handed,
    };
    emit({ kind: 'despatch_sent', despatch });

    if (handed) {
      deliver(despatch, atHours);
      return;
    }
    riding.set(despatch.id, { progress: 0, route, moved: false });
  };

  /**
   * A rider passing an enemy column.
   *
   * The rules' pool: one die, plus one for cavalry, one for a scouting formation, one for
   * a division. One `1` and the rider is stopped; two and the paper is read as well,
   * which is much worse — a captured order is intelligence, and it lands in the captor's
   * hands with the sender none the wiser.
   */
  const interception = (d: Despatch, at: Hex, atHours: number): boolean => {
    const k = key(at);
    const enemies = [...s.units.values()]
      .filter((u) => u.faction !== d.faction && occupied(u, 'road', cfg.footprint).some((c) => key(c) === k))
      .sort((a, b) => (a.id < b.id ? -1 : 1));
    if (enemies.length === 0) return false;

    // The formation best placed to stop him decides it. A rider slipping between two
    // columns is caught by the more watchful, not by both in turn.
    const best = enemies.reduce((a, b) => (detectionDice(cfg, b) > detectionDice(cfg, a) ? b : a));
    const dice = rng.pool(cfg.interceptDiceBase + detectionDice(cfg, best));
    const struck = ones(dice);
    if (struck < cfg.interceptLoseOnes) return false;

    clockTo(atHours);
    emit({
      kind: 'despatch_stopped',
      despatchId: d.id,
      by: best.faction,
      atHours,
      at,
      dice,
      outcome: struck >= cfg.interceptCaptureOnes ? 'captured' : 'lost',
    });
    riding.delete(d.id);
    return true;
  };

  /**
   * The commander who rides with a formation, preferring the senior one.
   *
   * Seniority is depth in the chain of command: the fewer men above him, the senior. A
   * corps commander fallen back on one of his own divisions decides for it, and the
   * divisional commander does not. Ties — two men of equal depth on one formation — break
   * on id, which is arbitrary but stable, and is the only part of this that ever was.
   */
  const commanderRiding = (unitId: string): string | null => {
    const riders = [...s.commanders.values()]
      .filter((c) => c.unitId === unitId)
      .map((c) => ({ id: c.id, depth: superiors(s, c.id).length }))
      .sort((a, b) => a.depth - b.depth || (a.id < b.id ? -1 : 1));
    return riders[0]?.id ?? null;
  };

  /**
   * The next hex of a march, or null when there is not one.
   *
   * Routed fresh from where the column now stands, because a task names a destination and
   * not a path — so a river that turns out to be unbridged, or a road nobody knew about,
   * changes the march rather than breaking it.
   *
   * Waypoints still to make are routed through, and standing on the destination with
   * waypoints left is not arrival — a route may cross its own destination on the way to a
   * place the referee insisted on, and ending the march there would halt the column on
   * ground it was only passing over.
   */
  const nextLeg = (unit: Unit, task: Task, from: Hex): Hex | null => {
    const ahead = viaAhead(task, from);
    if (ahead.length === 0 && key(from) === key(task.destination)) return null;

    const path = planMarch(world, cfg, unit, task.destination, from, ahead);
    const next = path?.[1];
    if (next === undefined) return null;
    return Number.isFinite(hoursToEnter(world, cfg, unit, from, next)) ? next : null;
  };

  /** Move every rider, rolling for interception on each hex he enters. */
  const rideTick = (tickEnd: number): void => {
    for (const [id, rider] of [...riding]) {
      const d = s.despatches.get(id);
      if (d === undefined || d.fate.kind !== 'in_transit') continue;

      let budget = tickEnd - now;
      while (budget > 1e-9) {
        const i = Math.floor(rider.progress + 1e-9);
        const at = rider.route[i];
        if (at === undefined) break;

        if (rider.route[i + 1] === undefined) {
          // End of the planned route. Either his man is here, or the corps has marched
          // on and he has to find it — which is a fresh ride from where he now stands.
          const addressee = s.commanders.get(d.to);
          const toUnit = addressee === undefined ? undefined : s.units.get(addressee.unitId);
          const head = toUnit?.column[0];
          if (head === undefined) break;

          if (key(head) === key(at)) {
            deliver(d, tickEnd);
            break;
          }
          const replanned = planRide(world, cfg, at, head);
          if (replanned === null || replanned.length <= 1) break;

          // Recorded as it happens rather than at the end, so the log carries the path
          // the rider actually took rather than only the last one he was on.
          clockTo(tickEnd);
          rider.route = replanned;
          rider.progress = 0;
          rider.moved = true;
          emit({ kind: 'despatch_progressed', despatchId: id, progress: 0, route: replanned });
        }

        const base = Math.floor(rider.progress + 1e-9);
        const from = rider.route[base];
        const next = rider.route[base + 1];
        if (from === undefined || next === undefined) break;

        const step = courierStepHours(world, cfg, from, next);
        if (!Number.isFinite(step) || step <= 0) break;

        const remaining = step * (1 - (rider.progress - base));
        rider.moved = true;

        if (budget < remaining) {
          rider.progress += budget / step;
          break;
        }

        budget -= remaining;
        rider.progress = base + 1;
        if (interception(d, next, tickEnd)) break;
      }
    }
  };

  /**
   * Move every column as far through the tick as its hours will carry it.
   *
   * A hex at a time, but as many hexes as fit: infantry on a road crosses one in twenty
   * minutes, and a tick is a referee's unit of attention rather than a speed limit. One
   * step per tick would silently cap every march at `1 / tickHours` km/h — four, on the
   * default — which is slower than the rules' slowest going and would never look like a
   * bug, only like mud. Each step is re-read from the folded state, because `scheduleNext`
   * charges the hours against the day and the next leg depends on what is left of it.
   */
  /**
   * Units whose tail has already been charged for closing up after their last march.
   *
   * A column pays for its tail once per stretch of marching, not once per tick it spends
   * standing: the rear closes up, and then it is in. Cleared the moment the head steps off
   * again, because then there is a new tail to bring in.
   */
  const closedUp = new Set<string>();

  /**
   * Charge a unit for what a stretch of road cost it. Silent when it cost nothing.
   *
   * Patrols are never charged. The fatigue table is written for a division marching in
   * column; twenty troopers riding ahead of one are not doing that, and the rules give
   * them their own consequences — a die that destroys or recoils them — rather than a
   * share of the column's exhaustion.
   */
  const charge = (unitId: string, atHours: number, fromMarching: number, fromNight: number): void => {
    const unit = s.units.get(unitId);
    if (unit !== undefined && isPatrol(unit)) return;
    const total = fromMarching + fromNight;
    if (total <= 0) return;
    clockTo(atHours);
    emit({ kind: 'fatigue_accrued', unitId, atHours, fatigue: total, fromMarching, fromNight });
  };

  /**
   * Bring the tail in, and charge for any of it spent in the dark.
   *
   * The head halting is not the column halting. A division two kilometres long has men on
   * the road for `catchupHours` after its tip has stopped, and if the sun went down while
   * they were walking that is a night march for them whatever the head was doing.
   */
  const closeUp = (unit: Unit, atHours: number, speedKmh: number): void => {
    if (closedUp.has(unit.id)) return;
    closedUp.add(unit.id);
    const tail = catchupHours(unit, speedKmh);
    if (!Number.isFinite(tail) || tail <= 0) return;
    charge(unit.id, atHours, 0, nightFatigue(cfg, atHours, atHours + tail));
  };

  /** Begin a change of formation, at the rules' cost for that change. */
  const beginChange = (unit: Unit, to: Formation, atHours: number, reason: string): void => {
    const hours = cfg.formationChangeHours[unit.formation][to];
    clockTo(atHours);
    emit({
      kind: 'formation_change_began',
      unitId: unit.id,
      from: unit.formation,
      to,
      atHours,
      completesAtHours: atHours + hours,
      reason,
    });
  };

  /**
   * Finish every change of formation whose hours have run out.
   *
   * Before the columns move, so a formation that finishes breaking camp on the stroke of
   * the hour marches in the same tick rather than losing a quarter of an hour to the order
   * the ticks happen to run in.
   */
  const formationTick = (tickEnd: number): void => {
    const units = [...s.units.values()].sort((a, b) => (a.id < b.id ? -1 : 1));
    for (const unit of units) {
      const change = unit.formationChange;
      if (change == null || change.completesAtHours > tickEnd + 1e-9) continue;
      const atHours = Math.max(change.completesAtHours, now);
      clockTo(atHours);
      emit({ kind: 'formation_changed', unitId: unit.id, to: change.to, atHours });
    }
  };

  /**
   * The column standing on a hex, if any, and whether the hex is its head.
   *
   * Read fresh from the folded state each time rather than indexed once a tick: a column
   * that has just marched has vacated ground, and a stale index would have the next
   * formation bounce off a road nobody is on any more.
   */
  const standingOn = (
    at: Hex,
    exceptUnitId: string,
    ignore: (unitId: string) => boolean = () => false,
  ): { unitId: string; isHead: boolean } | null => {
    const k = key(at);
    for (const unit of [...s.units.values()].sort((a, b) => (a.id < b.id ? -1 : 1))) {
      if (unit.id === exceptUnitId || ignore(unit.id)) continue;
      const body = occupied(unit, 'road', cfg.footprint);
      const i = body.findIndex((c) => key(c) === k);
      if (i >= 0) return { unitId: unit.id, isHead: i === 0 };
    }
    return null;
  };

  /**
   * Another column whose head is walking into the same hex.
   *
   * No comparison of arrival times, because there are none to compare: the hour is the
   * smallest thing the clock has, and two columns walking into one hex during the same
   * hour are walking into it at the same time as far as this game is concerned.
   */
  const convergingOn = (
    at: Hex,
    exceptUnitId: string,
    ignore: (unitId: string) => boolean = () => false,
  ): Task | null => {
    const k = key(at);
    for (const t of [...s.tasks.values()].sort((a, b) => (a.unitId < b.unitId ? -1 : 1))) {
      if (t.unitId === exceptUnitId || ignore(t.unitId) || t.complete || t.nextHex === null) {
        continue;
      }
      if (key(t.nextHex) === k) return t;
    }
    return null;
  };

  /** A decision of this kind already standing over this hex, so the same one is not asked twice. */
  const openOver = (trigger: DecisionTrigger, unitId: string, at: Hex): boolean => {
    const k = key(at);
    for (const d of s.decisions.values()) {
      if (!isOpen(d) || d.trigger !== trigger || d.unitId !== unitId) continue;
      const over = contestedHex(d);
      if (over !== null && key(over) === k) return true;
    }
    return false;
  };

  /** A contest over this hex nobody has settled yet. Nothing enters until somebody does. */
  const contestStanding = (at: Hex): boolean => {
    const k = key(at);
    for (const d of s.decisions.values()) {
      if (!isOpen(d) || d.trigger !== 'column_contested') continue;
      const over = contestedHex(d);
      if (over !== null && key(over) === k) return true;
    }
    return false;
  };

  /** The formation a referee has already given this hex to, if he has ruled on it. */
  const ruledFor = (at: Hex): string | null => {
    const k = key(at);
    let ruling: { atHours: number; unitId: string } | null = null;
    for (const d of s.decisions.values()) {
      if (d.trigger !== 'column_contested' || d.favouring === null) continue;
      const over = contestedHex(d);
      if (over === null || key(over) !== k) continue;
      // The latest ruling stands: a hex can be contested, settled, and contested again.
      if (ruling === null || (d.resolvedAtHours ?? 0) >= ruling.atHours) {
        ruling = { atHours: d.resolvedAtHours ?? 0, unitId: d.favouring };
      }
    }
    return ruling?.unitId ?? null;
  };

  /**
   * One column, one hex, with the traffic rules applied.
   *
   * Returns whether the head actually entered. A column that could not spends no hours and
   * keeps whatever it had already walked; the rules are the two the ruleset gives. A head
   * meeting any part of a column that is simply standing there stops; two heads walking
   * into the same ground during the same hour contest it, and the cheaper step wins,
   * because a formation coming up a highway is moving faster than one in a bog and that is
   * what "faster" has to mean on ground this varied.
   */
  const enterHex = (
    task: Task,
    to: Hex,
    atHours: number,
    budget: number,
    blocked: Set<string>,
  ): boolean => {
    const unit = s.units.get(task.unitId);
    const head = unit?.column[0];
    if (unit === undefined || head === undefined) return false;

    /**
     * Whether either party to this meeting is a patrol.
     *
     * A patrol running into anything is not traffic. The rules resolve it with a pool of
     * dice that may simply destroy the patrol, and that roll is the referee's — so the
     * column rules stand aside and he is asked instead. It reads both ways round: twenty
     * troopers walking into a division and a division walking into twenty troopers are the
     * same meeting, and only one of them should be reported.
     */
    const patrolMeeting = (otherId: string): boolean => {
      const other = s.units.get(otherId);
      return isPatrol(unit) || (other !== undefined && isPatrol(other));
    };

    /**
     * Whether these two simply ride past each other.
     *
     * A patrol is twenty troopers on horseback and its own side's traffic is not an
     * obstacle to it: it rides down the column, through the halt, and out the far end.
     * The rule is symmetric, because a division is not stopped by its own vedettes
     * either — they get out of the road.
     *
     * Only its own side. An enemy column is the whole reason the patrol is out there, and
     * meeting one is a contact rather than a traffic problem.
     */
    const ridesPast = (otherId: string): boolean => {
      const other = s.units.get(otherId);
      return (
        other !== undefined && other.faction === unit.faction && patrolMeeting(otherId)
      );
    };

    /** What the referee needs to roll the rules' contact dice, gathered for him. */
    const contactContext = (otherId: string): Record<string, unknown> => {
      const other = s.units.get(otherId);
      return {
        at: to,
        withUnitId: otherId,
        destination: task.destination,
        patrolUnitId: isPatrol(unit) ? unit.id : otherId,
        hostile: other !== undefined && other.faction !== unit.faction,
        // The rules' pool: one die to start, and the modifiers for what was met.
        dice:
          cfg.interceptDiceBase +
          (other === undefined ? 0 : detectionDice(cfg, other)),
      };
    };

    const stop = (byUnitId: string, contested: boolean, trigger: DecisionTrigger | null): void => {
      clockTo(atHours);
      emit({
        kind: 'march_blocked',
        unitId: unit.id,
        byUnitId,
        at: to,
        atHours,
        contested,
        // What is left of the hour is spent standing in the road rather than walking up
        // it, and standing formed up is charged like marching once a column has broken
        // camp. What is left, not a whole hour: a column stopped half an hour into its
        // hour has already been charged for the half it walked.
        waitedHours: Math.max(0, Math.min(budget, marchHoursLeftToday(cfg, unit))),
      });
      closeUp(unit, atHours, unitSpeedKmh(cfg, unit, gradeOf(world, cfg, head, to)));

      // A meeting involving a patrol is always reported, even where the column rules would
      // have passed over it in silence — a contested hex nobody has ruled on, say. Twenty
      // troopers standing off a division is exactly the moment the referee is playing for.
      const asked = patrolMeeting(byUnitId) ? 'patrol_contact' : trigger;
      if (asked !== null && !openOver(asked, unit.id, to)) {
        raise(
          asked,
          commanderRiding(unit.id),
          unit.id,
          atHours,
          asked === 'patrol_contact'
            ? contactContext(byUnitId)
            : { at: to, withUnitId: byUnitId, destination: task.destination },
        );
      }
      blocked.add(unit.id);
    };

    // Ground being fought over is not traffic. Formations in a battle are intermingled by
    // definition — that is what a battle is — so the rules that keep two columns off one
    // road have nothing to say about it, and applying them would have brigades bouncing
    // off each other as though the field were a crossroads. What happens in there is
    // below the resolution of this map and belongs to whoever adjudicates it.
    if (s.battle.has(key(to))) return true;

    // A contest nobody has ruled on holds the hex against everyone, including the column
    // that would otherwise have won it outright once the other was told to wait.
    //
    // Except a patrol, which is not party to it. Two divisions arguing over a crossroads
    // is not a reason twenty troopers cannot ride over it, and holding them there would
    // be the traffic rules reaching a formation they were never written for. If one of
    // the two is an enemy the patrol will meet it below, as a contact.
    if (!isPatrol(unit) && contestStanding(to)) {
      const other = convergingOn(to, unit.id) ?? standingOn(to, unit.id);
      stop(other?.unitId ?? unit.id, true, null);
      return false;
    }

    const rival = convergingOn(to, unit.id, ridesPast);
    if (rival !== null) {
      const ruled = ruledFor(to);
      if (ruled !== null) {
        if (ruled !== unit.id) {
          stop(rival.unitId, true, null);
          return false;
        }
      } else {
        const other = s.units.get(rival.unitId);
        const otherHead = other?.column[0];
        const mine = hoursToEnter(world, cfg, unit, head, to);
        const theirs =
          other === undefined || otherHead === undefined
            ? Infinity
            : hoursToEnter(world, cfg, other, otherHead, to);

        // Strictly cheaper wins. Equal is the tie the rules hand to the referee, and it is
        // the common case rather than the rare one: two infantry divisions on the same
        // open ground cost exactly the same hour.
        if (Math.abs(mine - theirs) <= 1e-9) {
          stop(rival.unitId, true, 'column_contested');
          return false;
        }

        // Strictly slower, so the rules have already settled it and there is nothing to
        // ask. It waits, and it waits the way a column waits for one that is simply
        // standing in the road — deliberately not as a contest, because an open contest
        // holds the hex against everyone, the faster column included. Raising one here
        // would deadlock the pair whenever the slower column happened to be looked at
        // first, which is a fact about identifiers rather than about the ground.
        if (mine > theirs) {
          stop(rival.unitId, false, 'column_blocked');
          return false;
        }
      }
    }

    const sitting = standingOn(to, unit.id, ridesPast);
    if (sitting !== null) {
      stop(sitting.unitId, false, 'column_blocked');
      return false;
    }
    return true;
  };

  /**
   * Give one column its hour of marching.
   *
   * The hour is the unit of time and the unit of accounting. A column is handed an hour of
   * movement and spends it: infantry on a road puts three hexes behind it, a convoy
   * off-road gets two thirds of the way into one and banks the rest. What it cannot do is
   * arrive at half past — there is no half past.
   */
  const marchHour = (started: Task, atHours: number, blocked: Set<string>): void => {
    let task = started;
    const unit0 = s.units.get(task.unitId);
    if (unit0 === undefined) return;

    // The day's ceiling bites before the hour does: a column with a quarter of an hour of
    // its twenty left walks for a quarter of an hour and then stops for good. Checked
    // before the camp, because a formation that has spent its day is not going to break
    // the camp it just built — it would undo it on the hour and rebuild it on the next.
    const budgetToday = marchHoursLeftToday(cfg, unit0);
    if (budgetToday <= 0) return;

    // A formation is not in column of march until it is. A division still building its
    // camp, or already in it, has to break camp before it steps off — which is what a
    // march order given to a resting corps actually costs, and the reason a referee thinks
    // twice before halting one.
    if (unit0.formationChange != null) {
      if (unit0.formationChange.to !== 'march') beginChange(unit0, 'march', atHours, 'break_camp');
      return;
    }
    if (unit0.formation !== 'march') {
      beginChange(unit0, 'march', atHours, 'break_camp');
      return;
    }

    let budget = Math.min(HOURS_PER_STEP, budgetToday);

    let moved = false;
    let lastGrade: Grade = 'road';
    let lastSpeed = unitSpeedKmh(cfg, unit0, lastGrade);

    while (budget > 1e-9) {
      const current = s.tasks.get(task.unitId);
      const unit = s.units.get(task.unitId);
      const head = unit?.column[0];
      if (current === undefined || unit === undefined || head === undefined) break;
      task = current;
      if (task.complete || task.nextHex === null) break;

      const to = task.nextHex;
      const cost = hoursToEnter(world, cfg, unit, head, to);
      if (!Number.isFinite(cost)) break;

      const owing = cost - task.progressHours;
      if (owing > budget + 1e-9) {
        // Not enough hour left to finish the hex. What was walked is banked, and the rest
        // of the walk happens next hour.
        const grade = gradeOf(world, cfg, head, to);
        clockTo(atHours);
        emit({
          kind: 'march_progressed',
          unitId: unit.id,
          atHours,
          progressHours: task.progressHours + budget,
          grade,
          spentHours: budget,
        });
        charge(
          unit.id,
          atHours,
          marchFatigueBetween(cfg, unit, unit.hoursMarchedToday, unit.hoursMarchedToday + budget),
          nightFatigue(cfg, atHours, atHours + budget),
        );
        closedUp.delete(unit.id);
        moved = true;
        lastGrade = grade;
        lastSpeed = unitSpeedKmh(cfg, unit, grade);
        budget = 0;
        break;
      }

      if (!enterHex(task, to, atHours, budget, blocked)) break;

      const grade = gradeOf(world, cfg, head, to);
      const spent = Math.max(0, owing);
      const before = unit.hoursMarchedToday;

      clockTo(atHours);
      const onward = nextLeg(unit, task, to);
      emit({
        kind: 'unit_marched',
        unitId: unit.id,
        to,
        atHours,
        grade,
        stepHours: spent,
        nextHex: onward,
        progressHours: 0,
      });

      closedUp.delete(unit.id);
      charge(
        unit.id,
        atHours,
        marchFatigueBetween(cfg, unit, before, before + spent),
        nightFatigue(cfg, atHours, atHours + spent),
      );

      budget -= spent;
      moved = true;
      lastGrade = grade;
      lastSpeed = unitSpeedKmh(cfg, unit, grade);

      if (onward === null) {
        // Nowhere onward, for one of two very different reasons. Arrived is done. Stopped
        // by ground he cannot cross is *not* done — `task_completed` there would report
        // "the march is finished" for a column standing on the wrong bank of a river, and
        // the referee's own queue would say so while the decision beside it said the
        // opposite. The task stays open with nowhere to go, and resolving the decision is
        // what starts it again.
        const arrived =
          viaAhead(task, to).length === 0 && key(to) === key(task.destination);
        if (arrived) emit({ kind: 'task_completed', unitId: unit.id, atHours });

        raise(
          arrived ? 'objective_reached' : 'crossing_impassable',
          commanderRiding(unit.id),
          unit.id,
          atHours,
          arrived ? { destination: task.destination } : { at: to, destination: task.destination },
        );
        break;
      }
    }

    const after = s.units.get(task.unitId);
    if (after === undefined) return;

    if (moved) {
      // A formation that has spent its twenty hours is done for the day and builds a camp
      // where it stands. The rules make this the one change that happens without an order:
      // nobody decides to stop after twenty hours on the road, they simply stop.
      if (marchHoursLeftToday(cfg, after) <= 1e-9) {
        closeUp(after, atHours + HOURS_PER_STEP, lastSpeed);
        beginChange(after, 'rest', atHours + HOURS_PER_STEP, 'day_spent');
      } else if (s.tasks.get(task.unitId)?.nextHex == null) {
        closeUp(after, atHours + HOURS_PER_STEP, lastSpeed);
      }
    }
    void lastGrade;
  };

  /**
   * Every column's hour, in one pass.
   *
   * Sorted by unit id, which is arbitrary but fixed, so a replay reproduces it. Order only
   * decides who is asked first; who actually gets a contested hex is settled by the rules
   * in `enterHex` rather than by who came first in the list.
   */
  const marchTick = (atHours: number): void => {
    const blocked = new Set<string>();
    const running = [...s.tasks.values()].sort((a, b) => (a.unitId < b.unitId ? -1 : 1));
    for (const task of running) {
      if (task.complete || task.nextHex === null) continue;
      if (blocked.has(task.unitId)) continue;
      marchHour(task, atHours, blocked);
    }
  };

  /**
   * Ask every formation what it can see that it could not before.
   *
   * A new sighting does two things at once: it raises a decision for the man riding with
   * the formation, and it sends a report up the chain — automatically, because a division
   * that sees an enemy corps does not wait to be asked. The report is a despatch like any
   * other, so it takes a rider, takes time, and can be intercepted; unless the two
   * formations are touching, in which case it is simply handed over.
   */
  const discoveryTick = (tickEnd: number): void => {
    const units = [...s.units.values()].sort((a, b) => (a.id < b.id ? -1 : 1));

    for (const unit of units) {
      const fresh: Sighting[] = [];
      for (const [enemyId, contact] of spottedBy(s, world, cfg, unit)) {
        const k = `${unit.id}->${enemyId}`;
        if (alreadySeen.has(k)) continue;
        alreadySeen.add(k);
        fresh.push(contact);
      }
      if (fresh.length === 0) continue;

      const commanderId = commanderRiding(unit.id);
      if (commanderId === null) continue;

      raise('enemy_contact', commanderId, unit.id, tickEnd, {
        contacts: fresh.map((c) => ({ coord: c.coord, intelLevel: c.intelLevel })),
      });

      // Filed for the man who saw it, here rather than only in the store's pass after the
      // whole command. That pass looks at the final state, so an enemy sighted and lost
      // again during a long advance never reached the observer's own contacts at all —
      // his superior got it by despatch and he did not, which is precisely backwards.
      for (const payload of fileSightings(s, cfg, commanderId, fresh, tickEnd)) emit(payload);

      const commander = s.commanders.get(commanderId);
      if (commander != null && commander.superiorId !== null) {
        send(
          { from: commanderId, to: commander.superiorId, kind: 'report', body: { contacts: fresh } },
          tickEnd,
        );
      }
    }
  };

  // ---- the loop ---------------------------------------------------------

  const run = (opts: AdvanceOptions): void => {
    // Whole hours, always. A referee who asks for two and a half gets two — there is no
    // half hour for the extra to happen in, and rounding up would run the clock past what
    // he asked for.
    const asked = Math.floor(Math.min(Math.max(0, opts.hours), cfg.maxAdvanceHours));
    const target = now + asked;
    const stopAtDecision = opts.untilDecision ?? false;

    while (now < target) {
      const hourStart = now;
      const hourEnd = hourStart + HOURS_PER_STEP;
      const before = payloads.length;

      // Midnight: the day's marching starts again, and provisions will tick here when
      // they are built.
      //
      // At the top of the hour that begins the new day, not at the end of the hour that
      // reaches it. Marching is stamped at `hourStart`, so rolling the day on the hour
      // from eleven to midnight would credit a column's twenty-third hour of marching to
      // tomorrow — it would step off again an hour early, every night.
      if (hourStart > 0 && hourStart % HOURS_PER_DAY === 0) {
        emit({ kind: 'day_rolled', toHours: hourStart });
      }

      rideTick(hourEnd);
      // On the hour it is due, not at the end of the hour it falls in. Completing it at
      // `hourEnd` while columns march at `hourStart` lets a formation march out of a camp
      // it has not finished breaking — an hour early, and only visible as an off-by-one in
      // the log.
      formationTick(hourStart);
      // Marching is stamped at the hour it begins: a column given the hour from six to
      // seven is on the road at six, and the referee reading the log wants the hour the
      // men stepped off rather than the hour they stopped.
      marchTick(hourStart);
      if (payloads.length > before) discoveryTick(hourEnd);

      now = hourEnd;
      if (stopAtDecision && halted !== null) break;
    }

    // Riders still on the road: one event each, carrying where they got to and the path
    // they are on. See `events.ts` for why this is not one event per hex.
    for (const [id, rider] of riding) {
      if (!rider.moved) continue;
      const d = s.despatches.get(id);
      if (d === undefined || d.fate.kind !== 'in_transit') continue;
      emit({
        kind: 'despatch_progressed',
        despatchId: id,
        progress: rider.progress,
        route: rider.route,
      });
    }

    // Only a halt the referee asked for truncates the clock. A decision raised during a
    // plain `advance 12h` goes into his queue and the clock runs on regardless — that is
    // the entire difference between the two controls, and reading `halted` here without
    // `stopAtDecision` would silently collapse them into one.
    clockTo(
      stopAtDecision && halted !== null ? Math.min(target, halted.atHours) : target,
    );
  };

  return {
    payloads,
    send,
    run,
    get halted(): PendingDecision | null {
      return halted;
    },
    get clockHours(): number {
      return s.clockHours;
    },
  };
}

/**
 * Advance the campaign.
 *
 * `decide` calls this for `advance_clock`, which is why the whole of marching, riding,
 * interception and discovery reaches the log through one command and one audit trail
 * rather than through a dozen side doors.
 */
export function advance(
  state: CampaignState,
  world: World,
  cfg: CampaignConfig,
  rng: Rng,
  opts: AdvanceOptions,
): AdvanceResult {
  const sim = simulate(state, world, cfg, rng);
  sim.run(opts);
  return { payloads: sim.payloads, toHours: sim.clockHours, halted: sim.halted };
}

/**
 * Put one despatch on the road, with everything that follows immediately from it.
 *
 * Immediately means: handed over if the two formations are touching, and if so delivered,
 * cascaded and queued for the referee before the clock moves at all. Everything else is
 * the rider's business and happens in `advance`.
 */
export function despatchNow(
  state: CampaignState,
  world: World,
  cfg: CampaignConfig,
  rng: Rng,
  spec: SendSpec,
): readonly EventPayload[] {
  const sim = simulate(state, world, cfg, rng);
  sim.send(spec, state.clockHours);
  return sim.payloads;
}
