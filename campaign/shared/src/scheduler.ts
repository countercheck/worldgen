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
 * ## Ticks, and why they are not the clock
 *
 * Time runs in `cfg.tickHours` steps, but the campaign clock only moves when something
 * happens: a quiet tick emits nothing at all. Events are therefore stamped at the hour
 * they occurred rather than at the hour the referee clicked, and a twelve-hour advance
 * through empty country costs one event rather than forty-eight.
 *
 * ## What a rider does when his man has moved
 *
 * The route is planned to where the addressee stood when the rider left. When he gets
 * there and finds the corps gone, he re-plans from where he is and rides on — which is
 * what a real despatch rider did, and why "riders always find their man" is a rule rather
 * than an approximation. The cost is his time and the extra ground he has to cross, both
 * of which are exactly the things that are supposed to hurt.
 */

import { occupied } from './column.js';
import { directSubordinates } from './commander.js';
import type { CampaignConfig } from './config.js';
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
import { hoursToEnter, marchHoursLeftToday, planMarch } from './movement.js';
import { detectionDice, spottedBy, type Contact } from './recon.js';
import { ones, type Rng } from './rng.js';
import { reduce, type CampaignState } from './state.js';
import type { DecisionTrigger, PendingDecision, Task } from './task.js';
import { gradeOf } from './terrain.js';
import { reportOf, type Unit } from './unit.js';
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
    commanderId: string,
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
    const handed = via.length === 0 && formationsTouch(fromUnit, toUnit);
    const route = handed
      ? [origin]
      : (planRide(world, cfg, origin, destination, via) ?? [origin]);

    clockTo(atHours);
    const despatch: Despatch = {
      id: nextId('d'),
      kind: spec.kind,
      from: spec.from,
      to: spec.to,
      faction: sender.faction,
      sentAtHours: atHours,
      // Where he stood when he sealed it, attached whether or not he thought to say so.
      body: { unitReport: reportOf(fromUnit, atHours), ...spec.body },
      via,
      forwardedFrom: spec.forwardedFrom ?? null,
      inReplyTo: spec.inReplyTo ?? null,
      route,
      progress: 0,
      fate: { kind: 'in_transit' },
      handed,
    };
    emit({ kind: 'despatch_sent', despatch });

    if (handed || route.length <= 1) {
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
      .filter((u) => u.faction !== d.faction && occupied(u).some((c) => key(c) === k))
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

  /** The commander who rides with a formation, preferring the senior one. */
  const commanderRiding = (unitId: string): string | null => {
    const riders = [...s.commanders.values()]
      .filter((c) => c.unitId === unitId)
      .sort((a, b) => (a.id < b.id ? -1 : 1));
    return riders.find((c) => c.superiorId === null)?.id ?? riders[0]?.id ?? null;
  };

  /**
   * The next leg of a march, or null when there is not one.
   *
   * Routed fresh from where the column now stands, because a task names a destination and
   * not a path — so a river that turns out to be unbridged, or a road nobody knew about,
   * changes the march rather than breaking it.
   *
   * A column that has used up its twenty hours stops until midnight. The rules' cap is a
   * wall rather than a gradient: fatigue is what actually limits a march, and it is
   * deferred, so this is the only limit in force.
   */
  const scheduleNext = (
    unit: Unit,
    task: Task,
    from: Hex,
    atHours: number,
    lastStepHours: number,
  ): { nextHex: Hex; arrivesAtHours: number } | null => {
    if (key(from) === key(task.destination)) return null;

    const marched = unit.hoursMarchedToday + (Number.isFinite(lastStepHours) ? lastStepHours : 0);
    const left = marchHoursLeftToday(cfg, { ...unit, hoursMarchedToday: marched });

    const path = planMarch(world, cfg, unit, task.destination, from);
    const next = path?.[1];
    if (next === undefined) return null;

    const step = hoursToEnter(world, cfg, unit, from, next);
    if (!Number.isFinite(step)) return null;

    // Out of hours: resume at midnight rather than never. The head sits where it is and
    // the tail closes up, which is what a halted column actually does.
    const startAt = left > 0 ? atHours : Math.floor(atHours / HOURS_PER_DAY + 1) * HOURS_PER_DAY;
    return { nextHex: next, arrivesAtHours: startAt + step };
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

  /** Move every column whose next hex is due, and route it onward. */
  const marchTick = (tickEnd: number): void => {
    const running = [...s.tasks.values()].sort((a, b) => (a.unitId < b.unitId ? -1 : 1));

    for (const task of running) {
      if (task.complete || task.nextHex === null || task.arrivesAtHours === null) continue;
      if (task.arrivesAtHours > tickEnd) continue;

      const unit = s.units.get(task.unitId);
      const head = unit?.column[0];
      if (unit === undefined || head === undefined) continue;

      const to = task.nextHex;
      const atHours = Math.max(task.arrivesAtHours, now);
      const grade = gradeOf(world, cfg, head, to);
      const stepHours = hoursToEnter(world, cfg, unit, head, to);

      clockTo(atHours);
      const onward = scheduleNext(unit, task, to, atHours, stepHours);
      emit({
        kind: 'unit_marched',
        unitId: unit.id,
        to,
        atHours,
        grade,
        stepHours: Number.isFinite(stepHours) ? stepHours : 0,
        nextHex: onward?.nextHex ?? null,
        arrivesAtHours: onward?.arrivesAtHours ?? null,
      });

      if (onward !== null) continue;

      const arrived = key(to) === key(task.destination);
      emit({ kind: 'task_completed', unitId: unit.id, atHours });

      const commander = commanderRiding(unit.id);
      if (commander === null) continue;
      raise(
        arrived ? 'objective_reached' : 'crossing_impassable',
        commander,
        unit.id,
        atHours,
        arrived ? { destination: task.destination } : { at: to, destination: task.destination },
      );
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
      const fresh: Contact[] = [];
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
    const target = now + Math.min(Math.max(0, opts.hours), cfg.maxAdvanceHours);
    const stopAtDecision = opts.untilDecision ?? false;

    while (now < target) {
      const tickEnd = Math.min(target, now + cfg.tickHours);
      const before = payloads.length;

      // Midnight: the day's marching starts again, and provisions will tick here when
      // they are built. Fired on the tick that crosses it rather than scheduled, which
      // keeps the loop free of a second notion of time.
      const midnight = Math.floor(now / HOURS_PER_DAY + 1) * HOURS_PER_DAY;
      if (tickEnd >= midnight) emit({ kind: 'day_rolled', toHours: midnight });

      rideTick(tickEnd);
      marchTick(tickEnd);
      if (payloads.length > before) discoveryTick(tickEnd);

      now = tickEnd;
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
