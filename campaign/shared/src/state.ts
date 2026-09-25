/**
 * Campaign state, and the fold that derives it from the log.
 *
 * `reduce` is pure, total and free of randomness: it never rejects an event and never
 * consults a generator. Everything that could refuse a change has already happened in
 * `check`, before the event existed. That is what makes replay trustworthy — a log always
 * folds, and always to the same thing.
 *
 * The log is deliberately *not* part of the state. State is derived; the log is the
 * record. Keeping the log out means `reduce` cannot accidentally read history it should
 * not, and the store stays free to page or truncate what it keeps in memory.
 */

import { advanceColumn, FOOTPRINT, occupied, type FootprintShape } from './column.js';
import type { Commander } from './commander.js';
import { normaliseDespatch, type Despatch } from './despatch.js';
import { key, type Hex, type HexKey } from './hex.js';
import { onRoad } from './movement.js';
import type { Faction, LoggedEvent, WorldRef } from './events.js';
import { viaIndexAt, type PendingDecision, type Task } from './task.js';
import type { Contact } from './recon.js';
import type { CampaignConfig } from './config.js';
import type { Daylight, StandingOrders } from './standing.js';
import type { Formation, Unit, UnitReport } from './unit.js';

/**
 * One commander's picture of the war.
 *
 * Knowledge belongs to the commander, not to their side and not to their formation. A side does
 * not know anything — two of its corps commanders can hold flatly contradictory pictures and
 * frequently did. A formation observes, but observing is not knowing: what a division
 * sees becomes knowledge when the commander riding with it takes note of it.
 *
 * `surveyed` grows monotonically, the one exception being a referee's `hexes_forgotten`.
 * What is visible *right now* is derived from the formation's position each time it is
 * asked and is never stored, because a stored copy is a second fact that can disagree
 * with where the column actually stands.
 */
export interface CommanderKnowledge {
  readonly commanderId: string;
  /** Ground their formations have covered. Terrain memory, and nothing about the enemy. */
  readonly surveyed: ReadonlySet<HexKey>;
  /** Campaign hour each hex was last looked at. Drives how stale a memory reads. */
  readonly lastSurveyedHours: ReadonlyMap<HexKey, number>;
  /**
   * Where they last heard each formation under them was, keyed by unit id.
   *
   * Held rather than computed, and that is the whole of the fog now that the ground is
   * public. A view that snapshotted these from the units at the moment of asking would
   * hand them every position live and date them "now" — which is what the console did
   * before riders existed, and which made a design about not knowing where your own corps
   * is display a list of exactly where it was.
   *
   * A report is replaced only by a *later* one. Riders overtake each other, and a
   * despatch that arrives carrying older news than the map already holds is not news.
   */
  readonly reports: ReadonlyMap<string, UnitReport>;
  /**
   * Enemies they have been told about, keyed by their own label for them.
   *
   * Held, not recomputed — the same correction reports needed. A view that asked
   * `spottedBy` at the moment a client requested it would show them exactly what their
   * column can see this instant and nothing else: an enemy would appear the moment it
   * came into view and vanish the moment it left, when what actually happens is that it
   * stops being current and starts being a place somebody was once seen.
   *
   * Contacts are never removed. They go stale, and staleness is the reader's problem.
   */
  readonly contacts: ReadonlyMap<string, Contact>;
  /**
   * The number their staff will give the next new contact.
   *
   * A counter rather than anything derived from the enemy's identity. A label that could
   * be computed from the observed unit — a hash, say — would be brute-forceable against a
   * small and guessable id space, and would hand back exactly the correlation the label
   * exists to withhold.
   */
  readonly nextContactNo: number;
}

export interface CampaignState {
  readonly name: string;
  readonly world: WorldRef;
  /** Seeds every roll, via `rngFor(seed, seq)`. */
  readonly seed: number;
  /** Hours since the scenario epoch. */
  readonly clockHours: number;
  /** The sequence number the next event will take. */
  readonly nextSeq: number;
  readonly factions: ReadonlyMap<string, Faction>;
  readonly commanders: ReadonlyMap<string, Commander>;
  readonly units: ReadonlyMap<string, Unit>;
  /** Keyed by commander id. */
  readonly knowledge: ReadonlyMap<string, CommanderKnowledge>;
  /**
   * Every despatch ever written, in flight or not.
   *
   * Kept rather than pruned on delivery. A commander's inbox is a filter over this, an
   * intercepted despatch has to stay readable by its captor, and the log of who wrote
   * what to whom is most of what an after-action review is for.
   */
  readonly despatches: ReadonlyMap<string, Despatch>;
  /** What each formation is doing, keyed by unit id. One task at a time, by design. */
  readonly tasks: ReadonlyMap<string, Task>;
  /** The referee's queue, resolved entries included so it has a history. */
  readonly decisions: ReadonlyMap<string, PendingDecision>;
  /**
   * Ground currently being fought over.
   *
   * A set of hexes and nothing more. There is no battle object, no roster and no start
   * hour, because the campaign layer does not resolve battles — it only needs to know
   * which ground has stopped behaving like open country. Who is engaged is *derived* from
   * where the formations are standing (`engagedIn`), which cannot go stale the way a
   * stored roster would when a column marches out of the fighting.
   */
  readonly battle: ReadonlySet<HexKey>;
  /**
   * When the sun rises and sets, if the referee has said.
   *
   * Null until they do, and then the campaign's config holds. State rather than config
   * because it moves: config is fixed when a campaign is created, and the season is not.
   */
  readonly daylight: Daylight | null;
  /** When each formation's head may be on the road, keyed by unit id. Absent: no limits. */
  readonly standingOrders: ReadonlyMap<string, StandingOrders>;
}

const EMPTY_WORLD: WorldRef = {
  seed: 0,
  width: 0,
  height: 0,
  layout: 'axial',
  schemaVersion: '0',
  hash: '',
};

/**
 * The state before any event.
 *
 * Not usable as a campaign — `campaign_created` fills it in. It exists so `reduce` has a
 * total starting point and replay needs no special case for the first event.
 */
export const EMPTY_STATE: CampaignState = {
  name: '',
  world: EMPTY_WORLD,
  seed: 0,
  clockHours: 0,
  nextSeq: 0,
  factions: new Map(),
  commanders: new Map(),
  units: new Map(),
  knowledge: new Map(),
  despatches: new Map(),
  tasks: new Map(),
  decisions: new Map(),
  battle: new Set(),
  daylight: null,
  standingOrders: new Map(),
};

const withUnit = (s: CampaignState, unit: Unit): CampaignState => ({
  ...s,
  units: new Map(s.units).set(unit.id, unit),
});

const withCommander = (s: CampaignState, commander: Commander): CampaignState => ({
  ...s,
  commanders: new Map(s.commanders).set(commander.id, commander),
});

const withDespatch = (s: CampaignState, d: Despatch): CampaignState => ({
  ...s,
  despatches: new Map(s.despatches).set(d.id, d),
});

function knowledgeFor(s: CampaignState, commanderId: string): CommanderKnowledge {
  return (
    s.knowledge.get(commanderId) ?? {
      commanderId,
      surveyed: new Set<HexKey>(),
      lastSurveyedHours: new Map<HexKey, number>(),
      reports: new Map<string, UnitReport>(),
      contacts: new Map<string, Contact>(),
      nextContactNo: 1,
    }
  );
}

function withKnowledge(s: CampaignState, k: CommanderKnowledge): CampaignState {
  return { ...s, knowledge: new Map(s.knowledge).set(k.commanderId, k) };
}

function survey(
  s: CampaignState,
  commanderId: string,
  coords: readonly Hex[],
  atHours: number,
): CampaignState {
  const k = knowledgeFor(s, commanderId);
  const surveyed = new Set(k.surveyed);
  const last = new Map(k.lastSurveyedHours);
  for (const c of coords) {
    const kk = key(c);
    surveyed.add(kk);
    last.set(kk, atHours);
  }
  return withKnowledge(s, { ...k, commanderId, surveyed, lastSurveyedHours: last });
}

function forget(
  s: CampaignState,
  commanderId: string,
  coords: readonly Hex[],
): CampaignState {
  const k = knowledgeFor(s, commanderId);
  const surveyed = new Set(k.surveyed);
  const last = new Map(k.lastSurveyedHours);
  for (const c of coords) {
    const kk = key(c);
    surveyed.delete(kk);
    last.delete(kk);
  }
  return withKnowledge(s, { ...k, commanderId, surveyed, lastSurveyedHours: last });
}

/**
 * File a report, unless a later one is already on the table.
 *
 * Riders overtake each other, so a despatch carrying older news than the map already
 * holds is not news. Same rule as an order arriving out of turn, and for the same reason:
 * the date is the only thing an engine can compare, and it is enough.
 */
function file(s: CampaignState, commanderId: string, report: UnitReport): CampaignState {
  const k = knowledgeFor(s, commanderId);
  const held = k.reports.get(report.unitId);
  if (held !== undefined && held.atHours >= report.atHours) return s;
  return withKnowledge(s, {
    ...k,
    reports: new Map(k.reports).set(report.unitId, report),
  });
}

/**
 * Fold one event into the state.
 *
 * Total by construction: every branch returns a state, and an event naming something that
 * has since vanished is ignored rather than throwing. A log that cannot be folded is a
 * campaign that cannot be opened, which is a far worse failure than an event that turns
 * out to be a no-op.
 */
export function reduce(state: CampaignState, event: LoggedEvent): CampaignState {
  const s: CampaignState = { ...state, nextSeq: Math.max(state.nextSeq, event.seq + 1) };
  const p = event.payload;

  switch (p.kind) {
    case 'campaign_created':
      return {
        ...s,
        name: p.name,
        world: p.world,
        seed: p.seed,
        clockHours: p.startHours,
      };

    case 'faction_added':
      return { ...s, factions: new Map(s.factions).set(p.faction.id, p.faction) };

    case 'commander_added':
      return withKnowledge(withCommander(s, p.commander), {
        commanderId: p.commander.id,
        surveyed: new Set(),
        lastSurveyedHours: new Map(),
        reports: new Map(),
        contacts: new Map(),
        nextContactNo: 1,
      });

    case 'commander_removed': {
      const commanders = new Map(s.commanders);
      commanders.delete(p.commanderId);
      // Knowledge is deliberately kept. A commander who falls is replaced, and what their
      // headquarters knew does not evaporate with them — their successor inherits the maps
      // and the last despatches on the table. Dropping it here would make succession
      // lose information that a real one does not.
      return { ...s, commanders };
    }

    case 'commander_reassigned': {
      const commander = s.commanders.get(p.commanderId);
      if (commander === undefined) return s;
      return withCommander(s, {
        ...commander,
        ...(p.unitId !== undefined ? { unitId: p.unitId } : {}),
        ...(p.superiorId !== undefined ? { superiorId: p.superiorId } : {}),
      });
    }

    case 'unit_added':
      return withUnit(s, p.unit);

    case 'unit_removed': {
      const units = new Map(s.units);
      units.delete(p.unitId);
      const standingOrders = new Map(s.standingOrders);
      standingOrders.delete(p.unitId);
      return { ...s, units, standingOrders };
    }

    case 'clock_advanced':
      return { ...s, clockHours: p.toHours };

    case 'unit_teleported': {
      const unit = s.units.get(p.unitId);
      if (unit === undefined) return s;
      return withUnit(s, { ...unit, column: p.column });
    }

    case 'battle_declared': {
      const battle = new Set(s.battle);
      for (const c of p.coords) battle.add(key(c));
      return { ...s, battle };
    }

    case 'battle_ended': {
      const battle = new Set(s.battle);
      for (const c of p.coords) battle.delete(key(c));
      return { ...s, battle };
    }

    case 'hexes_surveyed':
      return survey(s, p.commanderId, p.coords, event.clockHours);

    case 'hexes_forgotten':
      return forget(s, p.commanderId, p.coords);

    case 'report_filed':
      return file(s, p.commanderId, p.report);

    case 'contact_filed': {
      const k = knowledgeFor(s, p.commanderId);
      const held = k.contacts.get(p.contact.id);
      // An older sighting never overwrites a newer one. Two riders can arrive out of
      // order carrying word of the same column, and the later hour is the better fact.
      if (held !== undefined && held.seenAtHours > p.contact.seenAtHours) return s;
      return withKnowledge(s, {
        ...k,
        contacts: new Map(k.contacts).set(p.contact.id, p.contact),
        // A label already in use was minted by an earlier filing; only a new one moves
        // the counter, so replaying a log twice cannot inflate it.
        nextContactNo: held === undefined ? k.nextContactNo + 1 : k.nextContactNo,
      });
    }

    case 'contact_lost': {
      const k = knowledgeFor(s, p.commanderId);
      const held = k.contacts.get(p.contactId);
      if (held === undefined || !held.inSight) return s;
      return withKnowledge(s, {
        ...k,
        contacts: new Map(k.contacts).set(p.contactId, { ...held, inSight: false }),
      });
    }

    case 'unit_stat_set': {
      const unit = s.units.get(p.unitId);
      if (unit === undefined) return s;
      // Spread only the keys actually present, so an absent field means "leave alone"
      // rather than "set to undefined" — the difference matters for `corps`, which is
      // legitimately null.
      const changes = Object.fromEntries(
        Object.entries(p.changes).filter(([, v]) => v !== undefined),
      );
      return withUnit(s, { ...unit, ...changes });
    }

    case 'despatch_sent':
      return withDespatch(s, normaliseDespatch(p.despatch));

    case 'despatch_progressed': {
      const d = s.despatches.get(p.despatchId);
      if (d === undefined) return s;
      return withDespatch(s, { ...d, progress: p.progress, route: p.route });
    }

    case 'despatch_delivered': {
      const d = s.despatches.get(p.despatchId);
      if (d === undefined) return s;
      return withDespatch(s, {
        ...d,
        progress: Math.max(0, d.route.length - 1),
        fate: { kind: 'delivered', atHours: p.atHours },
      });
    }

    case 'despatch_stopped': {
      const d = s.despatches.get(p.despatchId);
      if (d === undefined) return s;
      return withDespatch(s, {
        ...d,
        fate:
          p.outcome === 'captured'
            ? { kind: 'captured', by: p.by, atHours: p.atHours, dice: p.dice }
            : { kind: 'lost', by: p.by, atHours: p.atHours, dice: p.dice },
      });
    }

    case 'task_set':
      return { ...s, tasks: new Map(s.tasks).set(p.task.unitId, p.task) };

    case 'task_cleared': {
      const tasks = new Map(s.tasks);
      tasks.delete(p.unitId);
      return { ...s, tasks };
    }

    case 'unit_marched': {
      const unit = s.units.get(p.unitId);
      if (unit === undefined) return s;

      const moved = withUnit(s, {
        ...onRoad(unit, p.atHours, p.stepHours),
        column: advanceColumn(unit, p.to, p.grade),
      });

      const task = moved.tasks.get(p.unitId);
      if (task === undefined) return moved;
      return {
        ...moved,
        tasks: new Map(moved.tasks).set(p.unitId, {
          ...task,
          nextHex: p.nextHex,
          progressHours: p.progressHours,
          // Derived from the hex the head entered rather than carried on the event, so a
          // replay of the same log lands on the same waypoint count without the log
          // having to record it.
          viaIndex: viaIndexAt(task, p.to),
        }),
      };
    }

    case 'patrol_detached': {
      const parent = s.units.get(p.parentUnitId);
      const withParent =
        parent === undefined
          ? s
          : withUnit(s, {
              ...parent,
              paperStrength: Math.max(0, parent.paperStrength - p.costPaperStrength),
            });
      return withUnit(withParent, p.patrol);
    }

    case 'formation_change_began': {
      const unit = s.units.get(p.unitId);
      if (unit === undefined) return s;
      const changing = withUnit(s, {
        ...unit,
        formationChange: { to: p.to, completesAtHours: p.completesAtHours },
      });

      // Breaking camp gates the march behind it, but nothing has to be written down for
      // that: the scheduler will not give an hour of movement to a formation that is not
      // in column of march, and the banked progress waits where it is.
      return changing;
    }

    case 'formation_changed': {
      const unit = s.units.get(p.unitId);
      if (unit === undefined) return s;
      return withUnit(s, { ...unit, formation: p.to, formationChange: null });
    }

    case 'fatigue_accrued': {
      const unit = s.units.get(p.unitId);
      if (unit === undefined) return s;
      // Capped at a hundred, which is the scale the rules give it: fatigue is read as a
      // percentage of the unit that is no longer fit to stand in the line.
      return withUnit(s, {
        ...unit,
        fatigue: Math.min(100, Math.max(0, unit.fatigue + p.fatigue)),
      });
    }

    case 'march_progressed': {
      // An hour of walking that did not finish a hex. The ground is closer than it was,
      // and the day is that much more spent.
      const unit = s.units.get(p.unitId);
      const walked =
        unit === undefined
          ? s
          : withUnit(s, onRoad(unit, p.atHours, p.spentHours));

      const task = walked.tasks.get(p.unitId);
      if (task === undefined) return walked;
      return {
        ...walked,
        tasks: new Map(walked.tasks).set(p.unitId, { ...task, progressHours: p.progressHours }),
      };
    }

    case 'march_blocked': {
      // Nothing moved, so the column keeps its ground and its head stays pointed at the
      // same hex. What it does not keep is the time: a formation that has broken camp is
      // standing formed up in column of march, and waiting in that state costs it the same
      // hours as marching would. A column blocked before it took a step has not marched at
      // all, and is simply still in camp.
      const unit = s.units.get(p.unitId);
      const waited =
        unit === undefined || unit.hoursMarchedToday <= 0
          ? s
          : withUnit(s, onRoad(unit, p.atHours, p.waitedHours));

      // The head keeps its ground and its banked progress: it did not walk, so it is no
      // closer, and it has not lost what it had already walked either.
      return waited;
    }

    case 'task_completed': {
      const task = s.tasks.get(p.unitId);
      if (task === undefined) return s;
      return {
        ...s,
        tasks: new Map(s.tasks).set(p.unitId, {
          ...task,
          complete: true,
          nextHex: null,
          progressHours: 0,
        }),
      };
    }

    case 'day_rolled':
      // Midnight no longer hands a column its day back: the cap reads the last twenty-four
      // hours, and fatigue starts again at a rest, whenever that falls.
      return { ...s, clockHours: Math.max(s.clockHours, p.toHours) };

    case 'unit_rested': {
      const unit = s.units.get(p.unitId);
      return unit === undefined ? s : withUnit(s, { ...unit, hoursMarchedToday: 0 });
    }

    case 'daylight_set':
      return { ...s, daylight: { sunriseHour: p.sunriseHour, sunsetHour: p.sunsetHour } };

    case 'standing_orders_set': {
      const standingOrders = new Map(s.standingOrders);
      if (p.orders === null) standingOrders.delete(p.unitId);
      else standingOrders.set(p.unitId, p.orders);
      return { ...s, standingOrders };
    }

    case 'decision_raised':
      return { ...s, decisions: new Map(s.decisions).set(p.decision.id, p.decision) };

    case 'decision_resolved': {
      const decision = s.decisions.get(p.decisionId);
      if (decision === undefined) return s;
      return {
        ...s,
        decisions: new Map(s.decisions).set(p.decisionId, {
          ...decision,
          resolvedAtHours: p.atHours,
          note: p.note,
          favouring: p.favouring,
        }),
      };
    }
  }
}

/**
 * The numbers in force right now: the campaign's config, with the referee's daylight over it.
 *
 * Everything that reads the hour of sunset reads it off config, so the referee's setting
 * reaches all of them — night fatigue, the console's sun and moon — by being laid over the
 * one place they already look, rather than by each of them learning to ask the state.
 */
export const configAt = (cfg: CampaignConfig, s: CampaignState): CampaignConfig =>
  s.daylight === null ? cfg : { ...cfg, ...s.daylight };

/** Fold a whole log. The only way a campaign is ever loaded. */
export const replay = (events: Iterable<LoggedEvent>, from = EMPTY_STATE): CampaignState => {
  let s = from;
  for (const e of events) s = reduce(s, e);
  return s;
};

/**
 * The formation a patrol was detached from, if it is a patrol and the parent still exists.
 *
 * The one place to ask. A patrol has no morale, supply or fatigue of its own and is immune
 * to all three; where something needs to know the state of the troops it came from — what it
 * rejoins as, what its parent can still field — it reads them off the parent through this
 * rather than off a copy stored on the patrol. A copy would be right at the hour it was
 * detached and wrong by the afternoon.
 */
export const parentOf = (s: CampaignState, u: Unit): Unit | undefined =>
  u.parentUnitId == null ? undefined : s.units.get(u.parentUnitId);

/** The patrols a formation has in the field, in a stable order. */
export const patrolsOf = (s: CampaignState, unitId: string): Unit[] =>
  [...s.units.values()]
    .filter((u) => u.parentUnitId === unitId)
    .sort((a, b) => (a.id < b.id ? -1 : 1));

/**
 * Whether any part of a formation is standing on ground being fought over.
 *
 * Derived rather than flagged, and derived from the whole footprint rather than the head:
 * a division whose leading brigade is in the fighting is in the fighting, and a battle
 * that a column's tail is still trailing through has not let go of it. The corollary is
 * that a unit disengages by marching out, which is the right way round — nobody has to
 * remember to clear a flag.
 */
export const isEngaged = (
  s: CampaignState,
  u: Unit,
  shapes: Readonly<Record<Formation, FootprintShape>> = FOOTPRINT,
): boolean => s.battle.size > 0 && occupied(u, 'road', shapes).some((c) => s.battle.has(key(c)));

/** Every formation standing in the fighting, in a stable order. */
export const engaged = (
  s: CampaignState,
  shapes: Readonly<Record<Formation, FootprintShape>> = FOOTPRINT,
): Unit[] =>
  [...s.units.values()]
    .filter((u) => isEngaged(s, u, shapes))
    .sort((a, b) => (a.id < b.id ? -1 : 1));

/** Units belonging to one faction, in a stable order. */
export const unitsOf = (s: CampaignState, faction: string): Unit[] =>
  [...s.units.values()].filter((u) => u.faction === faction).sort((a, b) => (a.id < b.id ? -1 : 1));

/**
 * Factions in a fixed order.
 *
 * Sorted, not insertion-ordered: turn order and resolution order must not depend on the
 * order a referee happened to add factions in, for the same reason `CrossingStage` sorts
 * before it picks bridge sites.
 */
export const factionIds = (s: CampaignState): string[] => [...s.factions.keys()].sort();

/** Whether a commander's formations have ever covered a hex. */
export const hasSurveyed = (s: CampaignState, commanderId: string, c: Hex): boolean =>
  s.knowledge.get(commanderId)?.surveyed.has(key(c)) ?? false;

/**
 * Despatches in a stable order.
 *
 * By the hour written, then by id. Sorting on the hour rather than on insertion is what
 * makes an inbox read as a sequence of events in the war rather than as a queue of
 * arrivals — two despatches written an hour apart belong in that order however their
 * riders fared.
 */
const byWritten = (a: Despatch, b: Despatch): number =>
  a.sentAtHours - b.sentAtHours || (a.id < b.id ? -1 : 1);

/** Everything a commander has written, whatever became of it. */
export const despatchesFrom = (s: CampaignState, commanderId: string): Despatch[] =>
  [...s.despatches.values()].filter((d) => d.from === commanderId).sort(byWritten);

/** Everything addressed to a commander — including what never reached them. */
export const despatchesTo = (s: CampaignState, commanderId: string): Despatch[] =>
  [...s.despatches.values()].filter((d) => d.to === commanderId).sort(byWritten);

/**
 * A commander's inbox: what is actually in their hand.
 *
 * Delivered only. Nothing in transit toward them is visible, because a rider still on the
 * road has told them nothing — and showing them a despatch before it arrives would let them
 * read their subordinate's mind at the speed of light.
 */
export const inboxOf = (s: CampaignState, commanderId: string): Despatch[] =>
  despatchesTo(s, commanderId).filter((d) => d.fate.kind === 'delivered');

/** Despatches riders are still carrying. The referee's problem, and nobody else's. */
export const inFlight = (s: CampaignState): Despatch[] =>
  [...s.despatches.values()].filter((d) => d.fate.kind === 'in_transit').sort(byWritten);

/** Enemy paper a faction has taken off a rider. */
export const capturedBy = (s: CampaignState, faction: string): Despatch[] =>
  [...s.despatches.values()]
    .filter((d) => d.fate.kind === 'captured' && d.fate.by === faction)
    .sort(byWritten);

/** Decisions nobody has dealt with yet, oldest first. */
export const openDecisions = (s: CampaignState): PendingDecision[] =>
  [...s.decisions.values()]
    .filter((d) => d.resolvedAtHours === null)
    .sort((a, b) => a.atHours - b.atHours || (a.id < b.id ? -1 : 1));

/** What a formation is doing, if anything. */
export const taskFor = (s: CampaignState, unitId: string): Task | undefined =>
  s.tasks.get(unitId);

/**
 * What a commander knows about the enemy, oldest sighting last.
 *
 * Newest first because a contact's whole meaning is its hour: the top of this list is the
 * least wrong thing they hold.
 */
export const contactsOf = (s: CampaignState, commanderId: string): Contact[] =>
  [...(s.knowledge.get(commanderId)?.contacts.values() ?? [])].sort(
    (a, b) => b.seenAtHours - a.seenAtHours || (a.id < b.id ? -1 : 1),
  );
