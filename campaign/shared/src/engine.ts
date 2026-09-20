/**
 * Commands in, events out.
 *
 * The engine is the only place validation and mutation meet, and it keeps them apart even
 * here:
 *
 *   check(cmd, state, world)      -> Violation[]   pure, mutates nothing
 *   decide(cmd, state, world, rng) -> EventPayload[] pure given rng, mutates nothing
 *   reduce(state, event)          -> state          pure, total, no rng
 *
 * `apply` runs them in that order and stops at the first refusal, so nothing is written
 * when a command is rejected. That is the whole of the "variable rule enforcement"
 * requirement: strictness and `force` change only whether `check`'s result stops the
 * command, and never what `decide` or `reduce` do.
 *
 * Referee commands are the second override route and a deliberately different one.
 * `force` says "do this move anyway"; a referee command says "this is not a move". Their
 * checks return only hard violations, because no movement rule applies to a teleport in
 * the first place. Both routes are logged.
 */

import { type Commander, mayOrder, mayWriteTo, wouldCycle } from './commander.js';
import { DEFAULT_CONFIG, type CampaignConfig } from './config.js';
import { planRide, type DespatchBody, type DespatchKind } from './despatch.js';
import { planMarch } from './movement.js';
import { advance, despatchNow } from './scheduler.js';
import { contestants, type Task } from './task.js';
import type { EventPayload, Faction, LoggedEvent, UnitStatChanges, WorldRef } from './events.js';
import { byCommander, REFEREE, type Actor } from './events.js';
import { key, type Hex } from './hex.js';
import { rngFor, type Rng } from './rng.js';
import {
  bypassed,
  CODES,
  hard,
  refuses,
  RuleViolation,
  soft,
  type Strictness,
  type Violation,
} from './ruling.js';
import { EMPTY_STATE, patrolsOf, reduce, type CampaignState } from './state.js';
import { hasTrait, isDivision, type Formation, type Unit } from './unit.js';
import { hexAt, type World } from './world.js';

export type Command =
  | {
      readonly kind: 'create_campaign';
      readonly name: string;
      readonly world: WorldRef;
      readonly seed: number;
      readonly startHours?: number;
    }
  | { readonly kind: 'add_faction'; readonly faction: Faction }
  | { readonly kind: 'add_commander'; readonly commander: Commander }
  | { readonly kind: 'remove_commander'; readonly commanderId: string }
  | {
      readonly kind: 'reassign_commander';
      readonly commanderId: string;
      readonly unitId?: string;
      readonly superiorId?: string | null;
    }
  | { readonly kind: 'add_unit'; readonly unit: Unit }
  | { readonly kind: 'remove_unit'; readonly unitId: string }
  | {
      readonly kind: 'advance_clock';
      readonly hours: number;
      /** Stop at the first discovery a referee wants to see, rather than at the hour. */
      readonly untilDecision?: boolean;
    }
  /** The one command a commander issues himself. Everything else is the referee's. */
  | {
      readonly kind: 'send_despatch';
      readonly from: string;
      readonly to: string;
      readonly despatchKind: DespatchKind;
      readonly body: DespatchBody;
      /** Waypoints the sender insists his rider takes — around pickets, say. */
      readonly via?: readonly Hex[];
      readonly inReplyTo?: string;
      readonly forwardedFrom?: string;
    }
  /** The referee, having read a despatch, sets a formation marching. */
  | {
      readonly kind: 'set_task';
      readonly unitId: string;
      readonly destination: Hex;
      readonly via?: readonly Hex[];
      /** The despatch he was reading. The paper trail from prose to march. */
      readonly fromDespatchId?: string;
    }
  | { readonly kind: 'clear_task'; readonly unitId: string }
  | {
      readonly kind: 'resolve_decision';
      readonly decisionId: string;
      readonly note?: string;
      /**
       * On a contested hex, the formation the referee gives it to. Ignored elsewhere.
       *
       * Optional because most decisions have nothing to rule between, and because a
       * referee may deliberately deal with a contest without settling it — the columns
       * then meet again and ask again, which is the honest outcome of not deciding.
       */
      readonly favouring?: string;
    }
  /**
   * The referee orders a change of formation: form for battle, make camp, occupy a town.
   *
   * Making camp at the end of a day happens by itself, and breaking it to march happens by
   * itself; everything else is a decision somebody makes, which means the referee.
   */
  | { readonly kind: 'set_formation'; readonly unitId: string; readonly formation: Formation }
  /**
   * Send a patrol out from a formation.
   *
   * The referee's, like every order. A patrol is twenty troopers with a parent, and the
   * parent is what makes it worth having: what the patrol sees is what that formation's
   * commander comes to know, and it is recalled to the division it came from.
   */
  | {
      readonly kind: 'detach_patrol';
      readonly unitId: string;
      /** Where it starts. Defaults to its parent's head. */
      readonly at?: Hex;
      readonly patrolId?: string;
      readonly name?: string;
    }
  | { readonly kind: 'teleport_unit'; readonly unitId: string; readonly column: readonly Hex[] }
  | { readonly kind: 'reveal'; readonly commanderId: string; readonly coords: readonly Hex[] }
  | { readonly kind: 'conceal'; readonly commanderId: string; readonly coords: readonly Hex[] }
  | { readonly kind: 'set_unit_stats'; readonly unitId: string; readonly changes: UnitStatChanges }
  /** Referee: this ground is being fought over, and open-country rules stop applying. */
  | { readonly kind: 'declare_battle'; readonly coords: readonly Hex[] }
  | { readonly kind: 'end_battle'; readonly coords: readonly Hex[] };

export type CommandKind = Command['kind'];

/**
 * Commands that are the referee's by definition, not by override.
 *
 * These bypass no rule, because no rule applies: putting a unit somewhere without
 * marching it is not an illegal march. Their checkers return only hard violations.
 */
const REFEREE_ONLY: ReadonlySet<CommandKind> = new Set([
  'create_campaign',
  'add_faction',
  'add_commander',
  'remove_commander',
  'reassign_commander',
  'add_unit',
  'remove_unit',
  'advance_clock',
  'teleport_unit',
  'reveal',
  'conceal',
  'set_unit_stats',
  'declare_battle',
  'end_battle',
  // Tasks are the referee's, always. A commander writes prose; turning prose into a
  // march is the adjudication this whole design exists to keep in human hands.
  'set_task',
  'clear_task',
  'set_formation',
  'detach_patrol',
  'resolve_decision',
]);

export const isRefereeCommand = (c: Command): boolean => REFEREE_ONLY.has(c.kind);

export interface ApplyOptions {
  readonly actor?: Actor;
  /** Override this one command's rules. Never lifts a hard violation. */
  readonly force?: boolean;
  /** Override the campaign's strictness for this one command. */
  readonly strictness?: Strictness;
  /** The campaign's tuning. Routing, riding and the clock all need it. */
  readonly cfg?: CampaignConfig;
}

export interface Outcome {
  readonly ok: boolean;
  readonly state: CampaignState;
  readonly events: readonly LoggedEvent[];
  /** Everything `check` found, whether or not it stopped the command. */
  readonly violations: readonly Violation[];
}

const onMap = (world: World, c: Hex): boolean => hexAt(world, c) !== undefined;

/**
 * Everything wrong with a command, in a fixed order.
 *
 * Pure. It is given the state and world and returns findings; it holds no references to
 * anything it could mutate, and the tests deep-freeze both arguments to prove it.
 */
export function check(
  cmd: Command,
  state: CampaignState,
  world: World,
  cfg: CampaignConfig = DEFAULT_CONFIG,
): Violation[] {
  const v: Violation[] = [];

  const requireUnit = (id: string): Unit | undefined => {
    const u = state.units.get(id);
    if (u === undefined) {
      v.push(hard(CODES.NO_SUCH_UNIT, `there is no unit ${id}`));
    }
    return u;
  };

  const requireFaction = (id: string): void => {
    if (!state.factions.has(id)) {
      v.push(hard(CODES.NO_SUCH_FACTION, `there is no faction ${id}`));
    }
  };

  const requireCommander = (id: string): Commander | undefined => {
    const c = state.commanders.get(id);
    if (c === undefined) {
      v.push(hard(CODES.NO_SUCH_COMMANDER, `there is no commander ${id}`));
    }
    return c;
  };

  const requireOnMap = (coords: readonly Hex[], what: string): void => {
    for (const c of coords) {
      if (!onMap(world, c)) {
        v.push(hard(CODES.OFF_MAP, `${what} ${key(c)} is not on the map`));
      }
    }
  };

  switch (cmd.kind) {
    case 'create_campaign': {
      if (state.nextSeq > 0) {
        v.push(hard(CODES.MALFORMED, 'this campaign has already been created'));
      }
      if (
        cmd.world.seed !== world.seed ||
        cmd.world.width !== world.width ||
        cmd.world.height !== world.height ||
        cmd.world.layout !== world.layout
      ) {
        v.push(
          hard(
            CODES.WORLD_MISMATCH,
            'the world given does not match the world this campaign names',
          ),
        );
      }
      break;
    }

    case 'add_faction':
      if (state.factions.has(cmd.faction.id)) {
        v.push(hard(CODES.DUPLICATE_ID, `faction ${cmd.faction.id} already exists`));
      }
      break;

    case 'add_commander': {
      const c = cmd.commander;
      if (state.commanders.has(c.id)) {
        v.push(hard(CODES.DUPLICATE_ID, `commander ${c.id} already exists`));
      }
      requireFaction(c.faction);
      if (!state.units.has(c.unitId)) {
        v.push(hard(CODES.NO_SUCH_UNIT, `there is no unit ${c.unitId} for ${c.id} to ride with`));
      } else if (state.units.get(c.unitId)!.faction !== c.faction) {
        // A man cannot ride with the enemy's baggage. This one is hard because the state
        // it would produce is not merely irregular, it is unreadable: every question
        // about what he can see would have two contradictory answers.
        v.push(
          hard(
            CODES.WRONG_FACTION,
            `${c.id} is ${c.faction} but ${c.unitId} is not`,
          ),
        );
      }
      if (c.superiorId !== null) {
        const superior = state.commanders.get(c.superiorId);
        if (superior === undefined) {
          v.push(hard(CODES.NO_SUCH_COMMANDER, `there is no commander ${c.superiorId}`));
        } else if (superior.faction !== c.faction) {
          v.push(hard(CODES.WRONG_FACTION, `${c.id} cannot answer to the other side`));
        }
      }
      break;
    }

    case 'remove_commander':
      requireCommander(cmd.commanderId);
      break;

    case 'reassign_commander': {
      const c = requireCommander(cmd.commanderId);
      if (c !== undefined) {
        if (cmd.unitId !== undefined) {
          const unit = state.units.get(cmd.unitId);
          if (unit === undefined) {
            v.push(hard(CODES.NO_SUCH_UNIT, `there is no unit ${cmd.unitId}`));
          } else if (unit.faction !== c.faction) {
            v.push(hard(CODES.WRONG_FACTION, `${c.id} cannot ride with ${cmd.unitId}`));
          }
        }
        if (cmd.superiorId !== undefined && cmd.superiorId !== null) {
          const superior = state.commanders.get(cmd.superiorId);
          if (superior === undefined) {
            v.push(hard(CODES.NO_SUCH_COMMANDER, `there is no commander ${cmd.superiorId}`));
          } else if (superior.faction !== c.faction) {
            v.push(hard(CODES.WRONG_FACTION, `${c.id} cannot answer to the other side`));
          } else if (wouldCycle(state, cmd.commanderId, cmd.superiorId)) {
            // Hard, and worth its own code: a loop in the chain of command makes every
            // tree walk in the engine non-terminating, and the state that produced it
            // cannot be drawn, ordered through, or reasoned about at all.
            v.push(
              hard(
                CODES.COMMAND_CYCLE,
                `${cmd.superiorId} already answers to ${cmd.commanderId}`,
              ),
            );
          }
        }
      }
      break;
    }

    case 'add_unit': {
      if (state.units.has(cmd.unit.id)) {
        v.push(hard(CODES.DUPLICATE_ID, `unit ${cmd.unit.id} already exists`));
      }
      requireFaction(cmd.unit.faction);
      if (cmd.unit.column.length === 0) {
        v.push(hard(CODES.MALFORMED, `unit ${cmd.unit.id} has no position`));
      }
      requireOnMap(cmd.unit.column, `unit ${cmd.unit.id} column hex`);
      // Soft: the rules set a floor on a division, but a referee modelling a battered
      // remnant or a scenario's oddity should be able to place one under it.
      if (isDivision(cmd.unit) && cmd.unit.paperStrength < cfg.minDivisionPaperStrength) {
        v.push(
          soft(
            CODES.UNIT_TOO_SMALL,
            `a division is at least ${cfg.minDivisionPaperStrength} paperStrength; ` +
              `${cmd.unit.id} has ${cmd.unit.paperStrength}`,
          ),
        );
      }
      break;
    }

    case 'remove_unit':
      requireUnit(cmd.unitId);
      break;

    case 'advance_clock':
      if (!Number.isFinite(cmd.hours)) {
        v.push(hard(CODES.MALFORMED, 'hours must be a finite number'));
      } else if (cmd.hours < 0) {
        // Hard, not soft: rewinding is done by replaying a prefix of the log, not by
        // running the clock backwards. A negative advance would leave events stamped
        // after the clock they happened at.
        v.push(hard(CODES.TIME_REVERSED, 'the campaign clock does not run backwards'));
      }
      break;

    case 'teleport_unit':
      requireUnit(cmd.unitId);
      if (cmd.column.length === 0) {
        v.push(hard(CODES.MALFORMED, 'a unit must be somewhere'));
      }
      requireOnMap(cmd.column, 'column hex');
      break;

    case 'reveal':
    case 'conceal':
      requireCommander(cmd.commanderId);
      requireOnMap(cmd.coords, 'hex');
      break;

    case 'declare_battle':
    case 'end_battle':
      if (cmd.coords.length === 0) {
        v.push(hard(CODES.MALFORMED, 'a battle has to be somewhere'));
      }
      requireOnMap(cmd.coords, 'hex');
      break;

    case 'set_unit_stats': {
      const u = requireUnit(cmd.unitId);
      if (u !== undefined) {
        const c = cmd.changes;
        if (c.paperStrength !== undefined && c.paperStrength < 0) {
          v.push(hard(CODES.MALFORMED, 'paperStrength cannot be negative'));
        }
        if (c.fatigue !== undefined && (c.fatigue < 0 || c.fatigue > 100)) {
          v.push(hard(CODES.MALFORMED, 'fatigue runs from 0 to 100'));
        }
        if (c.morale !== undefined && c.morale < 0) {
          v.push(hard(CODES.MALFORMED, 'morale cannot be negative'));
        }
      }
      break;
    }

    case 'send_despatch': {
      const from = requireCommander(cmd.from);
      const to = requireCommander(cmd.to);
      requireOnMap(cmd.via ?? [], 'waypoint');

      const empty =
        (cmd.body.text ?? '').trim() === '' &&
        (cmd.body.contacts ?? []).length === 0 &&
        cmd.body.unitReport === undefined;
      if (empty) {
        v.push(hard(CODES.MALFORMED, 'a despatch with nothing written on it is not a despatch'));
      }

      if (from !== undefined && to !== undefined) {
        // Hard: writing to the other side is not a despatch. If it ever becomes a
        // mechanic — a summons to surrender, a parley — it will be a different one with
        // its own rules, and letting it through here would produce state neither the
        // inbox nor the fog knows how to describe.
        if (!mayWriteTo(state, cmd.from, cmd.to)) {
          v.push(
            hard(
              CODES.WRONG_FACTION,
              `${cmd.from} cannot write to ${cmd.to}`,
            ),
          );
        } else if (cmd.despatchKind === 'order' && !mayOrder(state, cmd.from, cmd.to)) {
          // Soft: orders travel downward, but a referee reconstructing a moment where
          // one marshal did give another instructions should be able to say so.
          v.push(
            soft(
              CODES.NOT_IN_COMMAND,
              `${cmd.to} does not answer to ${cmd.from}; that is a message, not an order`,
            ),
          );
        }

        const fromUnit = state.units.get(from.unitId);
        const toUnit = state.units.get(to.unitId);
        const origin = fromUnit?.column[0];
        const destination = toUnit?.column[0];

        if (origin === undefined || destination === undefined) {
          v.push(hard(CODES.NO_SUCH_UNIT, 'a despatch needs a formation at both ends'));
        } else if (planRide(world, cfg, origin, destination, cmd.via ?? []) === null) {
          v.push(
            soft(
              CODES.NO_COURIER_ROUTE,
              `no rider can get from ${key(origin)} to ${key(destination)} that way`,
            ),
          );
        }
      }
      break;
    }

    case 'set_task': {
      const unit = requireUnit(cmd.unitId);
      requireOnMap([cmd.destination, ...(cmd.via ?? [])], 'destination');
      if (unit !== undefined && onMap(world, cmd.destination)) {
        // Soft, and checked against the real ground: a referee ordering a march to the
        // far bank of an unbridged river should be told, and should still be able to
        // order it — the column will discover the problem where it stands, which is the
        // point.
        const via = cmd.via ?? [];
        if (planMarch(world, cfg, unit, cmd.destination, undefined, via) === null) {
          // Naming the waypoint matters: "cannot reach Quatre Bras" sends a referee
          // looking at the destination when the ground he cannot cross is two legs back.
          const legs = [...via, cmd.destination];
          const blocked = legs.find(
            (leg, i) => planMarch(world, cfg, unit, leg, undefined, legs.slice(0, i)) === null,
          );
          v.push(
            soft(
              CODES.NO_MARCH_ROUTE,
              `${cmd.unitId} cannot reach ${key(blocked ?? cmd.destination)} by any route`,
            ),
          );
        }
      }
      break;
    }

    case 'set_formation': {
      const unit = requireUnit(cmd.unitId);
      if (unit !== undefined && unit.formation === 'rout' && cmd.formation !== 'rout') {
        // Soft: a broken formation is not re-formed by an order, it is rallied, and what
        // that costs is the referee's to adjudicate. He may still insist.
        v.push(
          soft(
            CODES.NOT_IN_COMMAND,
            `${cmd.unitId} is routing and cannot simply be told to form up`,
          ),
        );
      }
      break;
    }

    case 'detach_patrol': {
      const unit = requireUnit(cmd.unitId);
      if (cmd.patrolId !== undefined && state.units.has(cmd.patrolId)) {
        v.push(hard(CODES.DUPLICATE_ID, `unit ${cmd.patrolId} already exists`));
      }
      if (cmd.at !== undefined) requireOnMap([cmd.at], 'patrol start');
      if (unit === undefined) break;

      // Soft, both of them. The rules give patrols to formations with Scout and let a
      // fourth be bought with men; a referee running a scenario where a line division
      // pushes out vedettes should be able to say so, and be told what it costs.
      if (!hasTrait(unit, 'scout')) {
        v.push(
          soft(
            CODES.NOT_IN_COMMAND,
            `${cmd.unitId} has no Scout and cannot field patrols as of right`,
          ),
        );
      }
      const out = patrolsOf(state, cmd.unitId).length;
      if (out >= cfg.freePatrols) {
        v.push(
          soft(
            CODES.UNIT_TOO_SMALL,
            `${cmd.unitId} already has ${out} patrols out; a further one costs ` +
              `${cfg.extraPatrolCost} paperStrength permanently`,
          ),
        );
      }
      break;
    }

    case 'clear_task':
      requireUnit(cmd.unitId);
      break;

    case 'resolve_decision': {
      const decision = state.decisions.get(cmd.decisionId);
      if (decision === undefined) {
        v.push(hard(CODES.NO_SUCH_DECISION, `there is no decision ${cmd.decisionId}`));
        break;
      }
      if (cmd.favouring !== undefined) {
        // Hard, and checked against the decision rather than against the order of battle:
        // giving the hex to a formation that was not contesting it would produce a ruling
        // the scheduler cannot act on.
        const contesting = [decision.unitId, ...contestants(decision)];
        if (!contesting.includes(cmd.favouring)) {
          v.push(
            hard(
              CODES.MALFORMED,
              `${cmd.favouring} is not contesting the hex in decision ${cmd.decisionId}`,
            ),
          );
        }
      }
      break;
    }
  }

  return v;
}

/**
 * What a command does, as events.
 *
 * Pure given `rng`. Called only after `check` has been consulted, so it may assume the
 * command is coherent — but not that it is legal, since a forced command reaches here
 * with its soft violations intact.
 */
export function decide(
  cmd: Command,
  state: CampaignState,
  world: World,
  rng: Rng,
  cfg: CampaignConfig = DEFAULT_CONFIG,
): EventPayload[] {
  switch (cmd.kind) {
    case 'create_campaign':
      return [
        {
          kind: 'campaign_created',
          world: cmd.world,
          seed: cmd.seed,
          name: cmd.name,
          startHours: cmd.startHours ?? 0,
        },
      ];

    case 'add_faction':
      return [{ kind: 'faction_added', faction: cmd.faction }];

    case 'add_commander':
      return [{ kind: 'commander_added', commander: cmd.commander }];

    case 'remove_commander':
      return [{ kind: 'commander_removed', commanderId: cmd.commanderId }];

    case 'reassign_commander':
      return [
        {
          kind: 'commander_reassigned',
          commanderId: cmd.commanderId,
          ...(cmd.unitId !== undefined ? { unitId: cmd.unitId } : {}),
          ...(cmd.superiorId !== undefined ? { superiorId: cmd.superiorId } : {}),
        },
      ];

    case 'add_unit':
      return [{ kind: 'unit_added', unit: cmd.unit }];

    case 'remove_unit':
      return [{ kind: 'unit_removed', unitId: cmd.unitId }];

    case 'advance_clock':
      // Not a single event any more. Advancing the clock is the campaign happening:
      // columns march, riders ride, and the whole of it comes back as the facts it
      // produced, in the order it produced them.
      return [
        ...advance(state, world, cfg, rng, {
          hours: cmd.hours,
          ...(cmd.untilDecision !== undefined ? { untilDecision: cmd.untilDecision } : {}),
        }).payloads,
      ];

    case 'send_despatch':
      return [
        ...despatchNow(state, world, cfg, rng, {
          from: cmd.from,
          to: cmd.to,
          kind: cmd.despatchKind,
          body: cmd.body,
          ...(cmd.via !== undefined ? { via: cmd.via } : {}),
          ...(cmd.inReplyTo !== undefined ? { inReplyTo: cmd.inReplyTo } : {}),
          ...(cmd.forwardedFrom !== undefined ? { forwardedFrom: cmd.forwardedFrom } : {}),
        }),
      ];

    case 'set_task': {
      const unit = state.units.get(cmd.unitId);
      const via = cmd.via ?? [];
      const path =
        unit === undefined ? null : planMarch(world, cfg, unit, cmd.destination, undefined, via);
      const next = path?.[1];

      const task: Task = {
        unitId: cmd.unitId,
        destination: cmd.destination,
        via,
        setAtHours: state.clockHours,
        fromDespatchId: cmd.fromDespatchId ?? null,
        nextHex: next ?? null,
        // Nothing walked yet. The first hour of movement the scheduler hands out is what
        // starts the column moving; a task is an order, not a head start.
        progressHours: 0,
        // A march to where the column already stands is over before it starts, which is
        // a legitimate thing for a referee to order and should not leave a task running.
        // Waypoints on other ground are not: there the column has somewhere to go even
        // though it is standing on its destination, and `next` will have found it.
        complete: next === undefined,
        viaIndex: 0,
      };
      return [{ kind: 'task_set', task }];
    }

    case 'set_formation': {
      const unit = state.units.get(cmd.unitId);
      if (unit === undefined) return [];

      const hours = cfg.formationChangeHours[unit.formation][cmd.formation];
      // Already there, or already on the way there. A second click on "occupy" is the
      // referee saying the same thing twice, not a fresh order — and restarting the
      // change would charge the twenty-four hours again from the present hour, so a
      // formation could be kept changing forever by a commander who kept asking.
      const already =
        unit.formationChange == null
          ? unit.formation === cmd.formation
          : unit.formationChange.to === cmd.formation;
      if (already) return [];

      const out: EventPayload[] = [];
      // A march is a thing a formation does in column of march. Told to make camp, form
      // for battle or occupy a town, it is no longer doing that — so the order is called
      // off rather than left standing to restart the moment the change completes.
      if (cmd.formation !== 'march' && state.tasks.get(cmd.unitId)?.complete === false) {
        out.push({ kind: 'task_cleared', unitId: cmd.unitId });
      }

      out.push(
        hours <= 0
          ? { kind: 'formation_changed', unitId: cmd.unitId, to: cmd.formation, atHours: state.clockHours }
          : {
              kind: 'formation_change_began',
              unitId: cmd.unitId,
              from: unit.formation,
              to: cmd.formation,
              atHours: state.clockHours,
              completesAtHours: state.clockHours + hours,
              reason: 'ordered',
            },
      );
      return out;
    }

    case 'detach_patrol': {
      const parent = state.units.get(cmd.unitId);
      const at = cmd.at ?? parent?.column[0];
      if (parent === undefined || at === undefined) return [];

      const out = patrolsOf(state, cmd.unitId).length;
      // Free until the rules' allowance is spent, and then paid for out of the rolls —
      // permanently, because the men do not come back when the patrol does.
      const costPaperStrength = out >= cfg.freePatrols ? cfg.extraPatrolCost : 0;

      // The first free number rather than the count. A patrol that has been destroyed or
      // recalled leaves a gap, and numbering from the ones still out would hand the next
      // detachment an identifier a live patrol is already using — which `check` does not
      // catch, because it only guards ids the caller named, and which `reduce` would
      // then silently overwrite.
      let nth = out + 1;
      while (state.units.has(`${cmd.unitId}-p${nth}`)) nth += 1;

      const id = cmd.patrolId ?? `${cmd.unitId}-p${nth}`;
      const defaults = cfg.kindDefaults.cavalry;

      const patrol: Unit = {
        id,
        name: cmd.name ?? `${parent.name} patrol ${nth}`,
        faction: parent.faction,
        // Cavalry, because the rules move a patrol as cavalry — it is a handful of
        // troopers however the division it came from marches.
        kind: 'cavalry',
        paperStrength: cfg.patrolPaperStrength,
        // Zero, and meaning "not tracked" rather than "none left". A patrol is not a
        // formation in miniature: twenty troopers do not hold a line, do not run out of
        // food on a two-day ride, and are not worn down by a fatigue table written for a
        // division. `isBroken` and `isStarving` read the parentage before the number, so
        // nothing takes these zeroes for a crisis.
        fatigue: 0,
        experience: parent.experience,
        morale: 0,
        provisions: 0,
        maxProvisions: 0,
        equipment: 0,
        maxEquipment: 0,
        guns: 0,
        marchSpeedKmh: defaults.marchSpeedKmh,
        spacingM: defaults.spacingM,
        spacingMultiplier: 1,
        // Scout travels with them: a patrol is the parent's eyes, detached.
        traits: ['scout'],
        formation: 'march',
        formationChange: null,
        column: [at],
        hoursMarchedToday: 0,
        corps: parent.corps,
        parentUnitId: parent.id,
      };

      return [{ kind: 'patrol_detached', patrol, parentUnitId: parent.id, costPaperStrength }];
    }

    case 'clear_task':
      return [{ kind: 'task_cleared', unitId: cmd.unitId }];

    case 'resolve_decision':
      return [
        {
          kind: 'decision_resolved',
          decisionId: cmd.decisionId,
          atHours: state.clockHours,
          note: cmd.note ?? null,
          favouring: cmd.favouring ?? null,
        },
      ];

    case 'teleport_unit':
      return [{ kind: 'unit_teleported', unitId: cmd.unitId, column: cmd.column }];

    case 'reveal':
      return [{ kind: 'hexes_surveyed', commanderId: cmd.commanderId, coords: cmd.coords }];

    case 'conceal':
      return [{ kind: 'hexes_forgotten', commanderId: cmd.commanderId, coords: cmd.coords }];

    case 'set_unit_stats':
      return [{ kind: 'unit_stat_set', unitId: cmd.unitId, changes: cmd.changes }];

    case 'declare_battle':
      return [{ kind: 'battle_declared', coords: cmd.coords }];

    case 'end_battle':
      return [{ kind: 'battle_ended', coords: cmd.coords }];
  }
}

/**
 * Check, then act. Nothing is written when the checks refuse.
 *
 * `opts.strictness` overrides the campaign's setting for this one command, which is what
 * lets a referee be permissive once without loosening the whole session.
 */
export function apply(
  cmd: Command,
  state: CampaignState,
  world: World,
  strictness: Strictness,
  opts: ApplyOptions = {},
): Outcome {
  const effective = opts.strictness ?? strictness;
  const forced = opts.force ?? false;
  const actor = opts.actor ?? REFEREE;

  const cfg = opts.cfg ?? DEFAULT_CONFIG;

  const violations = check(cmd, state, world, cfg);
  if (refuses(violations, effective, forced)) {
    return { ok: false, state, events: [], violations };
  }

  const bent = bypassed(violations, effective, forced);
  const rng = rngFor(state.seed, state.nextSeq);
  const payloads = decide(cmd, state, world, rng, cfg);

  let next = state;
  const events: LoggedEvent[] = [];
  for (const payload of payloads) {
    const event: LoggedEvent = {
      seq: next.nextSeq,
      // Stamped from the state as it stands, so an event that advances the clock is
      // logged at the hour it happened rather than the hour it produced.
      clockHours: next.clockHours,
      actor,
      payload,
      forced,
      strictness: effective,
      // Only the first event of a command carries the audit; the rest are consequences
      // of a decision already recorded, and repeating it would overstate what was bent.
      bypassed: events.length === 0 ? bent : [],
    };
    events.push(event);
    next = reduce(next, event);
  }

  return { ok: true, state: next, events, violations };
}

/** `apply`, but a refusal throws. Convenient in scenario setup and in tests. */
export function applyOrThrow(
  cmd: Command,
  state: CampaignState,
  world: World,
  strictness: Strictness,
  opts: ApplyOptions = {},
): Outcome {
  const outcome = apply(cmd, state, world, strictness, opts);
  if (!outcome.ok) throw new RuleViolation(outcome.violations);
  return outcome;
}

/** Run a sequence of commands, stopping at the first refusal. */
export function applyAll(
  cmds: readonly Command[],
  state: CampaignState,
  world: World,
  strictness: Strictness,
  opts: ApplyOptions = {},
): Outcome {
  let s = state;
  const events: LoggedEvent[] = [];
  for (const cmd of cmds) {
    const outcome = apply(cmd, s, world, strictness, opts);
    events.push(...outcome.events);
    s = outcome.state;
    if (!outcome.ok) return { ok: false, state: s, events, violations: outcome.violations };
  }
  return { ok: true, state: s, events, violations: [] };
}

export { EMPTY_STATE, byCommander, REFEREE };
