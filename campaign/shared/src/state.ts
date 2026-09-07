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

import type { Commander } from './commander.js';
import { key, type Hex, type HexKey } from './hex.js';
import type { Faction, LoggedEvent, WorldRef } from './events.js';
import type { Unit } from './unit.js';

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
};

const withUnit = (s: CampaignState, unit: Unit): CampaignState => ({
  ...s,
  units: new Map(s.units).set(unit.id, unit),
});

const withCommander = (s: CampaignState, commander: Commander): CampaignState => ({
  ...s,
  commanders: new Map(s.commanders).set(commander.id, commander),
});

function knowledgeFor(s: CampaignState, commanderId: string): CommanderKnowledge {
  return (
    s.knowledge.get(commanderId) ?? {
      commanderId,
      surveyed: new Set<HexKey>(),
      lastSurveyedHours: new Map<HexKey, number>(),
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
  return withKnowledge(s, { commanderId, surveyed, lastSurveyedHours: last });
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
  return withKnowledge(s, { commanderId, surveyed, lastSurveyedHours: last });
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
