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

import { advanceColumn } from './column.js';
import type { Commander } from './commander.js';
import type { Despatch } from './despatch.js';
import { key, type Hex, type HexKey } from './hex.js';
import type { Faction, LoggedEvent, WorldRef } from './events.js';
import type { PendingDecision, Task } from './task.js';
import type { Unit, UnitReport } from './unit.js';

/**
 * One commander's picture of the war.
 *
 * Knowledge belongs to the man, not to his side and not to his formation. A side does not
 * know anything — two of its corps commanders can hold flatly contradictory pictures and
 * frequently did. A formation observes, but observing is not knowing: what a division
 * sees becomes knowledge when the man riding with it takes note of it.
 *
 * `surveyed` grows monotonically, the one exception being a referee's `hexes_forgotten`.
 * What is visible *right now* is derived from the formation's position each time it is
 * asked and is never stored, because a stored copy is a second fact that can disagree
 * with where the column actually stands.
 */
export interface CommanderKnowledge {
  readonly commanderId: string;
  /** Ground his formations have covered. Terrain memory, and nothing about the enemy. */
  readonly surveyed: ReadonlySet<HexKey>;
  /** Campaign hour each hex was last looked at. Drives how stale a memory reads. */
  readonly lastSurveyedHours: ReadonlyMap<HexKey, number>;
  /**
   * Where he last heard each formation under him was, keyed by unit id.
   *
   * Held rather than computed, and that is the whole of the fog now that the ground is
   * public. A view that snapshotted these from the units at the moment of asking would
   * hand him every position live and date them "now" — which is what the console did
   * before riders existed, and which made a design about not knowing where your own corps
   * is display a list of exactly where it was.
   *
   * A report is replaced only by a *later* one. Riders overtake each other, and a
   * despatch that arrives carrying older news than the map already holds is not news.
   */
  readonly reports: ReadonlyMap<string, UnitReport>;
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
      });

    case 'commander_removed': {
      const commanders = new Map(s.commanders);
      commanders.delete(p.commanderId);
      // Knowledge is deliberately kept. A commander who falls is replaced, and what his
      // headquarters knew does not evaporate with him — his successor inherits the maps
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
      return { ...s, units };
    }

    case 'clock_advanced':
      return { ...s, clockHours: p.toHours };

    case 'unit_teleported': {
      const unit = s.units.get(p.unitId);
      if (unit === undefined) return s;
      return withUnit(s, { ...unit, column: p.column });
    }

    case 'hexes_surveyed':
      return survey(s, p.commanderId, p.coords, event.clockHours);

    case 'hexes_forgotten':
      return forget(s, p.commanderId, p.coords);

    case 'report_filed':
      return file(s, p.commanderId, p.report);

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
      return withDespatch(s, p.despatch);

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
        ...unit,
        column: advanceColumn(unit, p.to, p.grade),
        hoursMarchedToday: unit.hoursMarchedToday + p.stepHours,
      });

      const task = moved.tasks.get(p.unitId);
      if (task === undefined) return moved;
      return {
        ...moved,
        tasks: new Map(moved.tasks).set(p.unitId, {
          ...task,
          nextHex: p.nextHex,
          arrivesAtHours: p.arrivesAtHours,
        }),
      };
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
          arrivesAtHours: null,
        }),
      };
    }

    case 'day_rolled': {
      const units = new Map(s.units);
      for (const [id, u] of units) units.set(id, { ...u, hoursMarchedToday: 0 });
      return { ...s, units, clockHours: Math.max(s.clockHours, p.toHours) };
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
        }),
      };
    }
  }
}

/** Fold a whole log. The only way a campaign is ever loaded. */
export const replay = (events: Iterable<LoggedEvent>, from = EMPTY_STATE): CampaignState => {
  let s = from;
  for (const e of events) s = reduce(s, e);
  return s;
};

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

/** Everything addressed to a commander — including what never reached him. */
export const despatchesTo = (s: CampaignState, commanderId: string): Despatch[] =>
  [...s.despatches.values()].filter((d) => d.to === commanderId).sort(byWritten);

/**
 * A commander's inbox: what is actually in his hand.
 *
 * Delivered only. Nothing in transit toward him is visible, because a rider still on the
 * road has told him nothing — and showing him a despatch before it arrives would let him
 * read his subordinate's mind at the speed of light.
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
